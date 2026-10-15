"""Point search, proof aggregation and bounded complete-deal analysis."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
from time import monotonic
import unittest
from unittest.mock import patch

from examples.endgame_oracle import generate_endgame, main
from texas42.actions import PLAY_ACTIONS, TRUMP_ACTIONS
from texas42.arena import generate_schedule
from texas42.episodes import EpisodeState
from texas42.models import Bid
from texas42.oracle import (ActionValue, BidderAnalysis, DealAnalysis, SearchLimits, TrumpAnalysis,
                            TrumpLead, ValueBounds, _Search, _bidder_analysis, _child,
                            _choice_proof, _play_start, _prepare_state, _solve_state,
                            _state_key, analyze_deal, exhaustive_minimax, solve, solve_points)
from texas42.phases import Phase


def full_hands(seed=42):
    return [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, seed)[0].hands]


def synthetic_trump(trump, intervals):
    rows = tuple(ActionValue(lower, upper, index) for index, (lower, upper) in enumerate(intervals))
    bounds, choice, choices = _choice_proof(rows)
    return TrumpAnalysis(bounds.lower, bounds.upper, trump, choice, choices,
                          rows, all(row.exact for row in rows))


class PointSearchTests(unittest.TestCase):
    def test_small_reachable_positions_match_exhaustive_points(self):
        actors = set()
        trumps = set()
        for size, count in ((1, 8), (2, 20), (3, 4)):
            for seed in range(count):
                for partial in range(4):
                    state = generate_endgame(seed, size, policy='HRR', partial_plays=partial)
                    before = deepcopy(state.__dict__)
                    trumps.add(state.trump)
                    for team in (0, 1):
                        with self.subTest(size=size, seed=seed, partial=partial, team=team):
                            actors.add((team, state.seat % 2))
                            expected = exhaustive_minimax(state, objective='final_points', reference_team=team)
                            action_values = {a: exhaustive_minimax(_child(state, a), objective='final_points',
                                                                  reference_team=team)
                                             for a in state.legal_action_ids()}
                            for capacity in (0, 1, 100_000):
                                result = solve_points(state, reference_team=team,
                                                      limits=SearchLimits(max_cache_entries=capacity))
                                self.assertEqual(result.status, 'exact')
                                self.assertEqual(result.value, expected)
                                self.assertEqual({a.action_id: a.value for a in result.action_values}, action_values)
                                self.assertTrue(result.all_action_values_complete)
                                optimum = tuple(a for a, value in action_values.items() if value == expected)
                                self.assertEqual(result.optimal_actions, optimum)
                                self.assertIn(result.optimal_action, optimum)
                                if capacity == 0:
                                    self.assertEqual(result.cache_hits, 0)
                    self.assertEqual(state.__dict__, before)
        self.assertEqual(actors, {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertIn('doubles', trumps)
        self.assertIn(None, trumps)

    def test_tied_and_distinguishing_actions_and_contract_cutoffs(self):
        tied = solve_points(generate_endgame(0, 2))
        self.assertEqual(len(tied.optimal_actions), 2)
        state = generate_endgame(1, 2, partial_plays=1)
        result = solve_points(state)
        self.assertEqual({a.value for a in result.action_values}, {30, 35})
        self.assertEqual(result.value, 35)
        self.assertEqual(result.optimal_actions, (0,))
        # The contract is already made, but its final total is still undecided.
        state = generate_endgame(9, 1)
        binary = solve(state)
        points = solve_points(state)
        self.assertEqual(binary.value, 1)
        self.assertEqual(binary.nodes, 1)
        self.assertGreater(points.nodes, 1)
        self.assertGreater(points.value, state.points[state.bidder % 2])
        self.assertEqual(points.value, exhaustive_minimax(state, objective='final_points'))

    def test_terminal_points_and_same_team_across_trick_boundary(self):
        state = generate_endgame(1, 1, partial_plays=3)
        team = state.seat % 2
        child = _child(state, state.legal_action_ids()[0])
        self.assertEqual(child.seat % 2, team)
        self.assertFalse(child.trick)
        self.assertIsNotNone(child.result)
        for reference in (0, 1):
            result = solve_points(child, reference_team=reference)
            self.assertEqual(result.value, child.result.team_points[reference])
            self.assertEqual(result.optimal_actions, ())

    def test_bounds_and_proofs_at_node_budget_interruptions(self):
        for seed, partial in ((0, 0), (1, 1), (5, 1), (7, 3)):
            state = generate_endgame(seed, 2, partial_plays=partial)
            for team in (0, 1):
                truth = solve_points(state, reference_team=team)
                action_truth = {a.action_id: a.value for a in truth.action_values}
                for budget in range(truth.nodes + 1):
                    with self.subTest(seed=seed, team=team, budget=budget):
                        result = solve_points(state, reference_team=team, limits=SearchLimits(max_nodes=budget))
                        self.assertLessEqual(result.nodes, budget)
                        self.assertLessEqual(result.lower, truth.value)
                        self.assertGreaterEqual(result.upper, truth.value)
                        for row in result.action_values:
                            self.assertLessEqual(row.lower, action_truth[row.action_id])
                            self.assertGreaterEqual(row.upper, action_truth[row.action_id])
                            self.assertEqual(row.status == 'exact', row.exact)
                        if result.optimal_action is not None:
                            self.assertIn(result.optimal_action, truth.optimal_actions)
                        if result.optimal_actions is not None:
                            self.assertEqual(result.optimal_actions, truth.optimal_actions)
                        if result.status == 'exact':
                            self.assertTrue(result.exact and result.all_action_values_complete)
                zero = solve_points(state, reference_team=team, limits=SearchLimits(max_nodes=0))
                self.assertEqual((zero.lower, zero.upper), (state.points[team], 42 - state.points[1-team]))

    def test_point_binary_and_reference_team_cache_namespaces(self):
        state = generate_endgame(42, 3)
        binary = _Search(SearchLimits(), monotonic())
        expected_binary = binary.search(state)
        cache = binary.cache
        for team in (0, 1):
            points = _Search(SearchLimits(), monotonic(), objective='final_points', reference_team=team)
            points.cache = cache  # Deliberately share entries across objectives.
            expected = exhaustive_minimax(state, objective='final_points', reference_team=team)
            self.assertEqual(points.search(state).value, expected)
            self.assertIn(_state_key(state, 'final_points', team), cache)
            self.assertNotEqual(_state_key(state), _state_key(state, 'final_points', team))
        self.assertNotEqual(_state_key(state, 'final_points', 0), _state_key(state, 'final_points', 1))
        self.assertEqual(binary.search(state), expected_binary)

    def test_point_table_windows_and_research(self):
        state = generate_endgame(42, 3)
        expected = exhaustive_minimax(state, objective='final_points')
        for alpha, beta in ((-1, expected), (expected, expected+1), (expected+1, 43)):
            search = _Search(SearchLimits(), monotonic(), objective='final_points')
            bounds = search.search(state, alpha, beta)
            self.assertLessEqual(bounds.lower, expected)
            self.assertGreaterEqual(bounds.upper, expected)
            self.assertEqual(search.search(state).value, expected)
            self.assertGreater(search.cache_hits, 0)

    def test_incumbent_search_preserves_ties_and_bounds_suboptimal_actions(self):
        # Deal driver uses this very same path, but small positions keep it cheap.
        state = generate_endgame(42, 3)
        truth = solve_points(state)
        search = _Search(SearchLimits(), monotonic(), objective='final_points')
        result = _solve_state(state, search, output='all_action_values', start=monotonic(),
                              minimum=truth.value)
        expected = {a.action_id: a.value for a in truth.action_values}
        self.assertEqual(result.value, truth.value)
        self.assertEqual(result.optimal_actions, truth.optimal_actions)
        for row in result.action_values:
            self.assertLessEqual(row.lower, expected[row.action_id])
            self.assertGreaterEqual(row.upper, expected[row.action_id])
            if row.upper >= truth.value:
                self.assertTrue(row.exact)

    def test_invalid_point_options(self):
        state = generate_endgame(7, 1)
        for team in (-1, 2, True, 0.0):
            with self.assertRaises(ValueError):
                solve_points(state, reference_team=team)
        with self.assertRaises(ValueError):
            solve_points(state, output='estimate')


class DealProofTests(unittest.TestCase):
    def test_aggregation_ties_and_strictly_dominated_unresolved_alternatives(self):
        trumps = (synthetic_trump(0, [(35, 35), (35, 35), (0, 20)]),
                  synthetic_trump('doubles', [(35, 35)]),
                  synthetic_trump(None, [(0, 34)]))
        bidder = _bidder_analysis(2, trumps)
        self.assertEqual(bidder.value, 35)
        self.assertEqual(bidder.optimal_choices, (TrumpLead(0, 0), TrumpLead(0, 1), TrumpLead('doubles', 0)))
        self.assertEqual(bidder.optimal_choice, TrumpLead(0, 0))
        self.assertEqual(trumps[0].optimal_actions, (0, 1))
        self.assertEqual(trumps[0].status, 'exact')
        self.assertEqual(bidder.status, 'exact')
        self.assertFalse(bidder.all_trump_values_complete)
        self.assertFalse(bidder.all_action_values_complete)
        report = DealAnalysis(tuple(replace(bidder, seat=i) for i in range(4)), 0, 0, 0)
        self.assertEqual(report.status, 'exact')
        self.assertFalse(report.all_alternative_values_complete)

    def test_proven_choices_and_unresolved_ties_are_distinct(self):
        for values, choice, choices in (
                ([(35, 40), (0, 34)], 0, (0,)),
                ([(35, 40), (0, 35)], 0, None),
                ([(35, 35), (0, 42)], None, None),
                ([(35, 35), (0, 35)], 0, None),
                ([(0, 42), (0, 42)], None, None)):
            _, actual, complete = _choice_proof([ValueBounds(*v) for v in values])
            self.assertEqual((actual, complete), (choice, choices))
        bidder = _bidder_analysis(0, (synthetic_trump(0, [(35, 40)]), synthetic_trump(1, [(0, 34)])))
        self.assertFalse(bidder.exact)
        self.assertIsNotNone(bidder.optimal_choice)
        self.assertEqual(bidder.optimal_choices, (TrumpLead(0, 0),))

    def test_bid_mapping_and_exact_classification_without_exact_points(self):
        for points, bid in ((29, None), (30, Bid(points=30)), (41, Bid(points=41)), (42, Bid(marks=1))):
            bidder = BidderAnalysis(points, points)
            self.assertTrue(bidder.contract_exact)
            self.assertEqual(bidder.maximum_contract, bid)
            self.assertEqual(bidder.proven_contract, bid)
            self.assertEqual(bidder.possible_contract, bid)
        below = BidderAnalysis(20, 29)
        self.assertFalse(below.exact)
        self.assertTrue(below.contract_exact)
        self.assertIsNone(below.maximum_contract)
        uncertain = BidderAnalysis(30, 42)
        self.assertFalse(uncertain.contract_exact)
        self.assertIsNone(uncertain.maximum_contract)
        self.assertEqual(uncertain.proven_contract, Bid(points=30))
        self.assertEqual(uncertain.possible_contract, Bid(marks=1))


class DealAnalysisTests(unittest.TestCase):
    def test_validate_once_and_all_candidate_leaders_and_canonical_trumps(self):
        hands = full_hands()
        before = deepcopy(hands)
        with patch('texas42.oracle._prepare_state', wraps=_prepare_state) as reconstruct:
            with patch.object(EpisodeState, 'deal', autospec=True, side_effect=EpisodeState.deal) as validate:
                with patch('texas42.oracle._play_start', wraps=_play_start) as search:
                    result = analyze_deal(hands, limits=SearchLimits(max_nodes=0))
        validate.assert_called_once()
        reconstruct.assert_not_called()
        self.assertEqual(search.call_count, 36 + 252)
        for call, (bidder, trump) in zip(search.call_args_list[:36],
                                        ((seat, trump) for seat in range(4) for trump in TRUMP_ACTIONS)):
            state = _play_start(call.args[0], call.args[1], call.args[2])
            self.assertEqual(state.seat, bidder)
            self.assertEqual(state.bidder, bidder)
            self.assertEqual(state.contract.bidder, bidder)
            self.assertEqual(state.contract.bid, Bid(points=30))
            self.assertEqual(state.phase, Phase.PLAY)
            self.assertEqual(state.trump, trump)
            self.assertEqual(state.hands, hands)
            self.assertEqual(state.points, [0, 0])
            self.assertEqual((state.trick, state.completed, state.bid_history), ([], [], []))
            self.assertEqual(state.legal_action_ids(), tuple(sorted(d.id for d in hands[bidder])))
        for bidder in result.bidders:
            self.assertEqual(tuple(t.trump for t in bidder.trumps), TRUMP_ACTIONS)
            self.assertEqual(bidder.team, bidder.seat % 2)
            self.assertIsNone(bidder.value)
            self.assertIsNone(bidder.optimal_choice)
            self.assertIsNone(bidder.optimal_choices)
            for trump in bidder.trumps:
                self.assertEqual(len(trump.action_values), 7)
                self.assertTrue(all((a.lower, a.upper, a.searched, a.value) == (0, 42, False, None)
                                    for a in trump.action_values))
        self.assertEqual(hands, before)

    def test_placeholder_contract_cannot_change_play_legality_or_point_scores(self):
        session = EpisodeState('hand', 7, 0)
        session.deal(full_hands())
        for trump in TRUMP_ACTIONS:
            first = _play_start(session.hand, 1, trump)
            second = deepcopy(first)
            second.contract = replace(second.contract, bid=Bid(marks=5))
            while first.result is None:
                self.assertEqual(first.legal_action_ids(), second.legal_action_ids())
                action = first.legal_action_ids()[0]
                first.apply(action)
                second.apply(action)
                self.assertEqual((first.seat, first.trick, first.points),
                                 (second.seat, second.trick, second.points))

    def test_per_tree_budgets_and_deterministic_results(self):
        hands = full_hands()
        before = deepcopy(hands)
        for nodes in (0, 1, 50, 200):
            limits = SearchLimits(max_nodes=nodes)
            first = analyze_deal(hands, limits=limits)
            second = analyze_deal(hands, limits=limits)
            self.assertEqual(first.nodes, 252 * nodes)
            self.assertEqual(first.status, 'budget_exceeded')
            self.assertEqual(replace(first, elapsed_seconds=0), replace(second, elapsed_seconds=0))
            if nodes > 1:
                self.assertTrue(all(any(a.searched for a in t.action_values)
                                    for b in first.bidders for t in b.trumps))
        timed = analyze_deal(hands, limits=SearchLimits(time_seconds=0))
        self.assertEqual(timed.nodes, 0)
        self.assertEqual(hands, before)

    def test_invalid_deals_and_rules(self):
        hands = full_hands()
        duplicate = deepcopy(hands)
        duplicate[0][0] = duplicate[1][0]
        for invalid in (None, [], hands[:3], [hands[0][:-1], *hands[1:]], duplicate,
                        [[d.id for d in hand] for hand in hands]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'deal|hands'):
                analyze_deal(invalid, limits=SearchLimits(max_nodes=0))
        with self.assertRaisesRegex(ValueError, 'Unsupported ruleset'):
            analyze_deal(hands, ruleset_id='nello', limits=SearchLimits(max_nodes=0))

    def test_bounded_cli_and_point_endgame_smoke(self):
        with redirect_stdout(io.StringIO()) as out:
            main(['--mode', 'deal', '--seed', '42', '--max-nodes', '1', '--time-seconds', '10'])
        for text in ('Maximum nodes per tree: 1', 'all hands visible', 'mark stake is not optimized',
                     'Bidder seat 3', 'Follow Me (no trump)', 'maximum makeable contract: unresolved',
                     'Status: budget_exceeded', 'Nodes: 252;', '[252/252]'):
            self.assertIn(text, out.getvalue())
        unresolved = analyze_deal(full_hands(), limits=SearchLimits(max_nodes=0))
        with patch('examples.endgame_oracle.analyze_deal', return_value=unresolved) as analyze:
            with redirect_stdout(io.StringIO()):
                main(['--mode', 'deal', '--seed', '42'])
        limits = analyze.call_args.kwargs['limits']
        self.assertEqual((limits.max_nodes, limits.time_seconds), (1_048_576, 1800.0))
        with redirect_stdout(io.StringIO()) as out:
            main(['--objective', 'final_points', '--seed', '1', '--remaining', '2', '--max-nodes', '1000'])
        self.assertIn('guaranteed final points', out.getvalue())


if __name__ == '__main__':
    unittest.main()
