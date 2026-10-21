"""Arbitrary-position oracle API: exhaustive agreement, bounds under cutoffs, thresholds."""
from dataclasses import replace
import unittest

from examples.endgame_oracle import generate_endgame
from texas42.models import Bid
from texas42.oracle import (PlayPosition, SearchLimits, _child, evaluate_position,
                            exhaustive_minimax, position_from_state, position_state)


def positions(sizes=((1, 4), (2, 8), (3, 3))):
    for size, count in sizes:
        for seed in range(count):
            for partial in range(4):
                yield (size, seed, partial), generate_endgame(seed, size, policy='HRR', partial_plays=partial)


class PositionOracleTests(unittest.TestCase):
    def test_exact_values_match_exhaustive_enumeration(self):
        for label, state in positions():
            position = position_from_state(state)
            for team in (0, 1):
                with self.subTest(label=label, team=team):
                    expected = {a: exhaustive_minimax(_child(state, a), objective='final_points',
                                                      reference_team=team)
                                for a in state.legal_action_ids()}
                    result = evaluate_position(position, reference_team=team)
                    self.assertEqual({r.action_id: (r.lower, r.upper) for r in result.actions},
                                     {a: (v, v) for a, v in expected.items()})
                    self.assertTrue(all(r.exact and r.termination in ('exact', 'decided') for r in result.actions))
                    root = exhaustive_minimax(state, objective='final_points', reference_team=team)
                    self.assertEqual((result.lower, result.upper), (root, root))
                    self.assertTrue(result.all_legal_actions)
                    self.assertEqual(result.units, 'final_team_points')
                    self.assertEqual(result.reference_team, team)
                    maximizing = state.seat % 2 == team
                    best = (max if maximizing else min)(expected.values())
                    self.assertEqual(result.proven_optimal_actions,
                                     tuple(a for a, v in expected.items() if v == best))

    def test_bounds_contain_exact_values_under_cutoffs(self):
        for label, state in positions(((3, 3), (4, 2))):
            position = position_from_state(state)
            team = state.seat % 2
            exact = {a: exhaustive_minimax(_child(state, a), objective='final_points', reference_team=team)
                     for a in state.legal_action_ids()} if len(state.hands[0]) <= 3 else None
            reference = evaluate_position(position)
            for nodes in (0, 1, 3, 20, 200):
                with self.subTest(label=label, nodes=nodes):
                    result = evaluate_position(position, limits=SearchLimits(max_nodes=nodes))
                    for row, truth in zip(result.actions, reference.actions):
                        self.assertLessEqual(row.lower, truth.lower)
                        self.assertGreaterEqual(row.upper, truth.upper)
                        if exact is not None:
                            self.assertTrue(row.lower <= exact[row.action_id] <= row.upper)
                        self.assertLessEqual(row.nodes, nodes)
                        if not row.exact:
                            self.assertIn(row.termination, ('node_limit', 'time_limit'))
                        # Captured points are always included in the final-total bounds.
                        self.assertGreaterEqual(row.lower, position.captured_points[team])
                        self.assertLessEqual(row.upper, 42 - position.captured_points[1 - team])
                    self.assertLessEqual(result.lower, reference.lower)
                    self.assertGreaterEqual(result.upper, reference.upper)

    def test_global_deadline_marks_unstarted_actions_with_valid_bounds(self):
        state = generate_endgame(1, 4, policy='HRR')
        result = evaluate_position(position_from_state(state), total_time_seconds=0.0)
        self.assertTrue(all(r.termination == 'not_started' and r.nodes == 0 for r in result.actions))
        truth = evaluate_position(position_from_state(state))
        for row, exact in zip(result.actions, truth.actions):
            self.assertTrue(row.lower <= exact.lower and exact.upper <= row.upper)

    def test_threshold_proofs_agree_with_exact_outcomes(self):
        for label, state in positions(((2, 6), (3, 3))):
            position = position_from_state(state)
            bidders = position.bidder % 2
            values = {r.action_id: r.lower for r in evaluate_position(position, reference_team=bidders).actions}
            for threshold in (30, 34, 38, 42):
                for nodes in (None, 5):
                    with self.subTest(label=label, threshold=threshold, nodes=nodes):
                        result = evaluate_position(position, objective='contract', threshold=threshold,
                                                   limits=SearchLimits(max_nodes=nodes))
                        self.assertEqual(result.reference_team, bidders)
                        for row in result.actions:
                            truth = values[row.action_id]
                            self.assertTrue(row.lower <= truth <= row.upper)
                            status = row.contract_status
                            if status == 'proven_makeable':
                                self.assertGreaterEqual(truth, threshold)
                            elif status == 'proven_unmakeable':
                                self.assertLess(truth, threshold)
                            elif nodes is None:
                                self.fail('An unlimited threshold search must resolve')

    def test_candidate_subsets_and_rejections(self):
        state = generate_endgame(3, 3, policy='HRR', partial_plays=1)
        position = position_from_state(state)
        legal = state.legal_action_ids()
        subset = evaluate_position(position, actions=legal[:1])
        self.assertEqual([r.action_id for r in subset.actions], [legal[0]])
        self.assertEqual(subset.all_legal_actions, len(legal) == 1)
        if len(legal) > 1:
            self.assertIsNone(subset.proven_optimal_actions)
        illegal = next(i for i in range(28) if i not in legal)
        for kwargs in (dict(actions=[illegal]), dict(actions=[]), dict(objective='nope'),
                       dict(threshold=35), dict(objective='contract', threshold=29),
                       dict(objective='contract', reference_team=1 - position.bidder % 2,
                            threshold=30)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_position(position, **kwargs)
        with self.assertRaises(ValueError):
            evaluate_position(replace(position, bid=None), objective='contract')
        result = evaluate_position(position, objective='contract')
        self.assertEqual(result.threshold, position.bid.target_points)

    def test_position_validation(self):
        state = generate_endgame(5, 3, policy='HRR', partial_plays=2)
        position = position_from_state(state)
        rebuilt = position_state(position)
        self.assertEqual(rebuilt.legal_action_ids(), state.legal_action_ids())
        self.assertEqual(rebuilt.seat, state.seat)
        bad = [replace(position, captured_points=(position.captured_points[0] + 1, position.captured_points[1])),
               replace(position, leader=(position.leader + 1) % 4),
               replace(position, next_player=(state.seat + 1) % 4),
               replace(position, hands=(position.hands[0], (*position.hands[1][:-1], position.hands[0][0]), *position.hands[2:])),
               replace(position, trump='spades')]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                position_state(value)
        self.assertEqual(Bid(marks=1).target_points, 42)


if __name__ == '__main__':
    unittest.main()
