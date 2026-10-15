"""Exact solver checks against terminal-only exhaustive engine play."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
from time import monotonic
import unittest

from examples.endgame_oracle import generate_endgame, main
from texas42.models import Bid, Contract, ContractKind
from texas42.oracle import (SearchLimits, _Flag, _Search, _child,
                            _state_key, exhaustive_minimax, solve)
from texas42.phases import Phase


class OracleTests(unittest.TestCase):
    def assert_matches_reference(self, state):
        expected = exhaustive_minimax(state)
        expected_actions = {action: exhaustive_minimax(_child(state, action))
                            for action in state.legal_action_ids()}
        for capacity in (0, 1, 100_000):
            result = solve(state, limits=SearchLimits(max_cache_entries=capacity))
            self.assertEqual(result.status, 'exact')
            self.assertEqual((result.lower, result.upper, result.value), (expected,) * 3)
            self.assertTrue(result.exact)
            self.assertTrue(result.all_action_values_complete)
            self.assertEqual({a.action_id: a.value for a in result.action_values}, expected_actions)
            self.assertTrue(all(a.exact for a in result.action_values))
            optimum = tuple(a for a, value in expected_actions.items() if value == expected)
            self.assertEqual(result.optimal_actions, optimum)
            if optimum:
                self.assertIn(result.optimal_action, optimum)
            else:
                self.assertIsNone(result.optimal_action)
            if capacity == 0:
                self.assertEqual(result.cache_hits, 0)

    def test_terminal_success_and_failure(self):
        for seed, value in ((0, -1), (1, 1)):
            state = generate_endgame(seed, 1)
            while state.result is None:
                state.apply(state.legal_action_ids()[0])
            self.assertEqual(solve(state).value, value)
            self.assert_matches_reference(state)

    def test_forced_moves(self):
        state = generate_endgame(7, 1, partial_plays=3)
        self.assertEqual(len(state.legal_action_ids()), 1)
        self.assert_matches_reference(state)

    def test_equal_optima_and_action_changing_contract_outcome(self):
        equal = solve(generate_endgame(42, 3))
        self.assertEqual(len(equal.optimal_actions), 3)
        for seed, partial, maximizing in ((7, 0, True), (42, 0, False),
                                          (7, 3, False), (92, 2, True)):
            state = generate_endgame(seed, 2, partial_plays=partial)
            self.assertEqual(state.seat % 2 == state.bidder % 2, maximizing)
            result = solve(state)
            self.assertEqual({a.value for a in result.action_values}, {-1, 1})
            self.assertEqual(result.value, 1 if maximizing else -1)
            self.assert_matches_reference(state)

    def test_diverse_reachable_small_positions(self):
        trumps, teams = set(), set()
        for size, seeds in ((1, range(8)), (2, range(24)), (3, range(12))):
            for seed in seeds:
                for partial in range(4):
                    with self.subTest(size=size, seed=seed, partial=partial):
                        state = generate_endgame(seed, size, policy='HRR', partial_plays=partial)
                        trumps.add(state.trump)
                        teams.add((state.bidder % 2, state.seat % 2))
                        self.assert_matches_reference(state)
        self.assertIn(None, trumps)
        self.assertIn('doubles', trumps)
        self.assertEqual(teams, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_ordinary_mark_contracts_use_success_not_mark_magnitude(self):
        for seed in range(8):
            state = generate_endgame(seed, 2, policy='RRR', partial_plays=seed % 4)
            self.assertIsNotNone(state.contract.bid.marks)
            self.assertEqual(state.contract.bid.target_points, 42)
            self.assert_matches_reference(state)

    def test_trick_winner_can_keep_the_same_team_on_move(self):
        # This distinguishes partnership minimax from alternating/negamax search.
        witnessed = False
        for seed in range(12):
            state = generate_endgame(seed, 2, partial_plays=3)
            for action in state.legal_action_ids():
                child = _child(state, action)
                self.assertFalse(child.trick)
                self.assertEqual(len(child.completed), len(state.completed) + 1)
                if state.seat % 2 == child.seat % 2:
                    witnessed = True
                    self.assert_matches_reference(state)
        self.assertTrue(witnessed)

    def test_input_unchanged_after_exact_and_interrupted_search(self):
        state = generate_endgame(42, 3, partial_plays=2)
        before = deepcopy(state.__dict__)
        solve(state)
        solve(state, limits=SearchLimits(max_nodes=3))
        exhaustive_minimax(state)
        self.assertEqual(state.__dict__, before)

    def test_keys_cover_future_relevant_context(self):
        state = generate_endgame(7, 3, partial_plays=2)
        key = _state_key(state)
        changes = {
            'points': [state.points[0] + 1, state.points[1]],
            'trump': None if state.trump is not None else 'doubles',
            'contract': replace(state.contract, bid=Bid(points=31)),
            'trick': list(reversed(state.trick)),
            'seat': (state.seat + 1) % 4,
            'completed': state.completed[:-1],
            'ruleset_id': 'unsupported',
            'hands': list(reversed(state.hands)),
        }
        for field, value in changes.items():
            changed = deepcopy(state)
            setattr(changed, field, value)
            with self.subTest(field=field):
                self.assertNotEqual(key, _state_key(changed))
        reordered = deepcopy(state)
        for hand in reordered.hands:
            hand.reverse()
        self.assertEqual(key, _state_key(reordered))

    def test_unsupported_modes_and_inconsistent_states_fail(self):
        for kind in (ContractKind.NIL, ContractKind.SPLASH, ContractKind.PLUNGE):
            state = generate_endgame(42, 2)
            state.contract = Contract(kind, state.bidder)
            with self.assertRaisesRegex(ValueError, 'ordinary contracts'):
                solve(state)
        for field, value, message in (
                ('phase', Phase.BID, 'ordinary contracts'),
                ('phase', Phase.TRUMP, 'ordinary contracts'),
                ('ruleset_id', 'nello', 'Unsupported ruleset'),
                ('points', [42, 42], 'disagree with engine replay')):
            state = generate_endgame(42, 2)
            setattr(state, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                solve(state)
        with self.assertRaises(TypeError):
            solve(generate_endgame(42, 2).view())

    def test_budget_bounds_are_valid_at_every_interruption(self):
        for seed, partial in ((7, 0), (42, 0), (1, 1), (4, 1)):
            state = generate_endgame(seed, 2, partial_plays=partial)
            truth = solve(state)
            action_truth = {a.action_id: a.value for a in truth.action_values}
            for nodes in range(truth.nodes + 1):
                with self.subTest(seed=seed, nodes=nodes):
                    result = solve(state, limits=SearchLimits(max_nodes=nodes))
                    self.assertLessEqual(result.nodes, nodes)
                    self.assertLessEqual(result.lower, truth.value)
                    self.assertGreaterEqual(result.upper, truth.value)
                    for row in result.action_values:
                        self.assertLessEqual(row.lower, action_truth[row.action_id])
                        self.assertGreaterEqual(row.upper, action_truth[row.action_id])
                        if row.exact:
                            self.assertEqual(row.value, action_truth[row.action_id])
                    if result.optimal_action is not None:
                        self.assertIn(result.optimal_action, truth.optimal_actions)
                    if result.optimal_actions is not None:
                        self.assertEqual(result.optimal_actions, truth.optimal_actions)
                    if result.status == 'exact':
                        self.assertTrue(result.exact and result.all_action_values_complete)

    def test_proven_root_with_unresolved_actions_for_both_teams(self):
        for seed, budget, value in ((1, 4, 1), (4, 14, -1)):
            state = generate_endgame(seed, 2, partial_plays=1)
            result = solve(state, limits=SearchLimits(max_nodes=budget))
            self.assertEqual(result.status, 'budget_exceeded')
            self.assertTrue(result.exact)
            self.assertEqual(result.value, value)
            self.assertEqual(result.optimal_action, 0)
            self.assertIsNone(result.optimal_actions)
            self.assertFalse(result.all_action_values_complete)
            self.assertIsNone(result.action_values[1].value)
            self.assertFalse(result.action_values[1].searched)
            position = solve(state, output='position_value', limits=SearchLimits(max_nodes=budget))
            self.assertEqual(position.status, 'exact')
            self.assertTrue(position.exact)
            self.assertFalse(position.all_action_values_complete)

    def test_zero_budgets_and_no_tentative_action(self):
        state = generate_endgame(7, 2)
        for limits in (SearchLimits(max_nodes=0), SearchLimits(max_nodes=1),
                       SearchLimits(time_seconds=0)):
            result = solve(state, limits=limits)
            self.assertEqual(result.status, 'budget_exceeded')
            self.assertEqual((result.lower, result.upper), (-1, 1))
            self.assertIsNone(result.value)
            self.assertIsNone(result.optimal_action)
            self.assertIsNone(result.optimal_actions)
            self.assertTrue(all(not a.exact and not a.searched for a in result.action_values))

    def test_table_bound_flags_window_tightening_and_full_window_research(self):
        for seed, window, flag in ((7, (-2, -1), _Flag.LOWER_BOUND),
                                   (42, (1, 2), _Flag.UPPER_BOUND)):
            state = generate_endgame(seed, 2)
            search = _Search(SearchLimits(), monotonic())
            bounds = search.search(state, *window)
            self.assertFalse(bounds.exact)
            self.assertEqual(search.cache[_state_key(state)].flag, flag)
            expected = exhaustive_minimax(state)
            exact = search.search(state)
            self.assertEqual(exact.value, expected)
            self.assertGreater(search.cache_hits, 0)
            self.assertEqual(search.cache[_state_key(state)].flag, _Flag.EXACT)
            before = search.cache_hits
            self.assertEqual(search.search(state).value, expected)
            self.assertEqual(search.cache_hits, before + 1)

    def test_table_bounded_eviction_and_no_interrupted_exact_entry(self):
        state = generate_endgame(42, 3)
        expected = exhaustive_minimax(state)
        for capacity in (0, 1, 5, 100_000):
            search = _Search(SearchLimits(max_cache_entries=capacity), monotonic())
            self.assertEqual(search.search(state).value, expected)
            self.assertLessEqual(len(search.cache), capacity)
            if capacity == 100_000:
                self.assertGreater(search.cache_hits, 0)
        search = _Search(SearchLimits(max_nodes=2), monotonic())
        result = search.search(state)
        self.assertFalse(result.exact)
        self.assertTrue(search.exhausted)
        self.assertNotIn(_state_key(state), search.cache)

    def test_already_decided_scores_prove_all_actions_without_children(self):
        for seed, value in ((0, -1), (1, 1)):
            state = generate_endgame(seed, 1)
            result = solve(state)
            self.assertEqual(result.value, value)
            self.assertEqual(result.nodes, 1)
            self.assertTrue(result.all_action_values_complete)
            self.assertTrue(all(row.exact and not row.searched for row in result.action_values))
            self.assertEqual(result.value, exhaustive_minimax(state))

    def test_larger_positions_allowed_with_budget(self):
        for size in (4, 7):
            result = solve(generate_endgame(42, size), limits=SearchLimits(max_nodes=10))
            self.assertLessEqual(result.nodes, 10)
            self.assertIn(result.status, ('exact', 'budget_exceeded'))

    def test_invalid_options_and_limits(self):
        for kwargs in ({'max_nodes': -1}, {'max_nodes': True}, {'max_cache_entries': None},
                       {'max_cache_entries': -1}, {'time_seconds': -1},
                       {'time_seconds': float('nan')}, {'time_seconds': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SearchLimits(**kwargs)
        with self.assertRaises(ValueError):
            solve(generate_endgame(42, 2), output='heuristic')

    def test_example_and_benchmark_smoke(self):
        with redirect_stdout(io.StringIO()) as out:
            main(['--seed', '7', '--remaining', '2', '--partial-plays', '3', '--max-nodes', '1000'])
        for text in ('Contract:', 'Trump:', 'To act:', 'Seat 0:', 'Partial trick:',
                     'exact', 'Proven optimal action:', 'Status:', 'Nodes:', 'cache hits:', 'runtime:'):
            self.assertIn(text, out.getvalue())
        with redirect_stdout(io.StringIO()) as out:
            main(['--remaining', '4', '--samples', '2', '--max-nodes', '10', '--max-cache-entries', '0'])
        self.assertIn('Exact completion:', out.getvalue())
        self.assertIn('median=', out.getvalue())


if __name__ == '__main__':
    unittest.main()
