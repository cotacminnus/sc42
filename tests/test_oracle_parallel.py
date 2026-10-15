"""Spawned oracle task boundaries, global resources and unchanged proof rules."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import multiprocessing as mp
import os
from time import monotonic
import unittest
from unittest.mock import patch

from examples.endgame_oracle import generate_endgame, main
from texas42 import oracle
from texas42.actions import PLAY_ACTIONS, TRUMP_ACTIONS
from texas42.arena import generate_schedule
from texas42.oracle import (OracleResult, SearchLimits, _Search, _WorkerConfig,
                            _initialize_oracle, _oracle_worker, _parallel_tasks,
                            analyze_deal, solve, solve_points)
from texas42.parallel import OrderedPool


def spawned_probe(task):
    result = _oracle_worker(task)
    # Repeat on the same process to verify reuse without depending on which
    # worker happens to receive a later pool task.
    repeated = _oracle_worker(task)
    return result, os.getpid(), mp.get_start_method(), len(oracle._WORKER_CACHE), repeated[2]


def synthetic_combination(task):
    # Aggregation-only fixture. Real search correctness is covered by root
    # searches and all nine trumps below; full-deal integration remains bounded.
    (seat, trump), allocation = task
    config = oracle._WORKER_CONFIG
    ids = sorted(d.id for d in config.state.hands[seat])
    values = [35 if index < 2 and trump in (0, None) else 29 for index in range(7)]
    rows = tuple(oracle.ActionValue(v, v, action, True) for action, v in zip(ids, values))
    best = max(values)
    optima = tuple(r.action_id for r in rows if r.value == best)
    result = OracleResult(best, best, 'exact', optima[0], optima, rows, True)
    return result, 0, 0


class ParallelOracleTests(unittest.TestCase):
    def exact_equal(self, expected, actual):
        # Cache locality and pruning can change work, never exact proofs.
        for name in ('nodes', 'cache_hits', 'elapsed_seconds'):
            expected = replace(expected, **{name: 0})
            actual = replace(actual, **{name: 0})
        self.assertEqual(expected, actual)

    def assert_sound(self, actual, truth):
        self.assertLessEqual(actual.lower, truth.value)
        self.assertGreaterEqual(actual.upper, truth.value)
        values = {row.action_id: row.value for row in truth.action_values}
        for row in actual.action_values:
            self.assertLessEqual(row.lower, values[row.action_id])
            self.assertGreaterEqual(row.upper, values[row.action_id])
            if row.exact:
                self.assertEqual(row.value, values[row.action_id])
        if actual.optimal_action is not None:
            self.assertIn(actual.optimal_action, truth.optimal_actions)
        if actual.optimal_actions is not None:
            self.assertEqual(actual.optimal_actions, truth.optimal_actions)
        if actual.status == 'exact':
            self.assertTrue(actual.exact)

    def test_exact_roots_both_teams_ties_partial_tricks_and_immutability(self):
        with patch('texas42.parallel.available_cpus', return_value=2):
            for seed, size, partial in ((0, 2, 0), (7, 2, 3), (1, 3, 0), (42, 3, 0)):
                state = generate_endgame(seed, size, partial_plays=partial)
                before = deepcopy(state.__dict__)
                for solver in (solve, solve_points):
                    with self.subTest(seed=seed, solver=solver.__name__):
                        truth = solver(state)
                        self.exact_equal(truth, solver(state, workers=2))
                for team in (0, 1):
                    truth = solve_points(state, reference_team=team)
                    self.exact_equal(truth, solve_points(state, reference_team=team, workers=2))
                self.assertEqual(state.__dict__, before)

    def test_all_nine_trumps_and_canonical_action_order(self):
        states = {}
        for seed in range(200):
            state = generate_endgame(seed, 3, policy='HRR')
            states.setdefault(state.trump, state)
            if len(states) == 9:
                break
        self.assertEqual(set(states), set(TRUMP_ACTIONS))
        with patch('texas42.parallel.available_cpus', return_value=2):
            for trump in TRUMP_ACTIONS:
                with self.subTest(trump=trump):
                    state = states[trump]
                    result = solve_points(state, workers=2)
                    self.exact_equal(solve_points(state), result)
                    self.assertEqual(tuple(r.action_id for r in result.action_values), state.legal_action_ids())

    def test_root_global_node_budget_and_unfinished_proofs(self):
        state = generate_endgame(1, 3)
        with patch('texas42.parallel.available_cpus', return_value=2):
            for solver in (solve, solve_points):
                truth = solver(state)
                for nodes in (0, 1, 2, 7, 30, 100):
                    with self.subTest(solver=solver.__name__, nodes=nodes):
                        result = solver(state, workers=2, limits=SearchLimits(max_nodes=nodes))
                        self.assertLessEqual(result.nodes, nodes)
                        self.assert_sound(result, truth)
                        if result.status == 'exact':
                            self.assertTrue(result.all_action_values_complete)
                position = solver(state, workers=2, output='position_value', limits=SearchLimits(max_nodes=100))
                self.assert_sound(position, truth)

    def test_deal_global_budget_trump_order_and_input_unchanged(self):
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        before = deepcopy(hands)
        with patch('texas42.parallel.available_cpus', return_value=2):
            for nodes in (1, 35, 80, 1000):
                result = analyze_deal(hands, workers=2, limits=SearchLimits(max_nodes=nodes))
                self.assertLessEqual(result.nodes, nodes)
                self.assertEqual(tuple(b.seat for b in result.bidders), tuple(range(4)))
                self.assertEqual(result.status, 'budget_exceeded')
                for bidder in result.bidders:
                    self.assertEqual(tuple(t.trump for t in bidder.trumps), TRUMP_ACTIONS)
                    self.assertIsNone(bidder.optimal_choices)
                    for trump in bidder.trumps:
                        self.assertEqual(tuple(r.action_id for r in trump.action_values),
                                         tuple(sorted(d.id for d in hands[bidder.seat])))
                        self.assertLessEqual(trump.lower, trump.upper)
        self.assertEqual(hands, before)

    def test_complete_deal_aggregation_of_exact_worker_results(self):
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        with patch('texas42.parallel.available_cpus', return_value=2):
            with patch('texas42.oracle._oracle_worker', synthetic_combination):
                result = analyze_deal(hands, workers=2)
        self.assertEqual(result.status, 'exact')
        self.assertTrue(result.all_alternative_values_complete)
        for bidder in result.bidders:
            self.assertEqual(bidder.value, 35)
            self.assertEqual(str(bidder.maximum_contract), '35')
            self.assertEqual(len(bidder.optimal_choices), 4)
            self.assertEqual(tuple(c.trump for c in bidder.optimal_choices), (0, 0, None, None))

    def test_zero_budgets_do_not_start_workers(self):
        state = generate_endgame(42, 3)
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        with patch('texas42.oracle.OrderedPool', side_effect=AssertionError('unnecessary spawn')):
            for limits in (SearchLimits(max_nodes=0), SearchLimits(time_seconds=0)):
                for solver in (solve, solve_points):
                    result = solver(state, workers=2, limits=limits)
                    self.assertEqual(result.nodes, 0)
                    self.assertEqual(result.status, 'budget_exceeded')
                result = analyze_deal(hands, workers=2, limits=limits)
                self.assertEqual(result.nodes, 0)
                self.assertEqual(result.status, 'budget_exceeded')

    def test_one_worker_is_current_path_and_forced_moves_do_not_spawn(self):
        state = generate_endgame(7, 3)
        forced = generate_endgame(7, 1, partial_plays=3)
        with patch('texas42.oracle.OrderedPool', side_effect=AssertionError('unnecessary spawn')):
            for solver in (solve, solve_points):
                for nodes in (None, 0, 5):
                    first = solver(state, limits=SearchLimits(max_nodes=nodes))
                    second = solver(state, workers=1, limits=SearchLimits(max_nodes=nodes))
                    self.assertEqual(replace(first, elapsed_seconds=0), replace(second, elapsed_seconds=0))
                self.assertEqual(solver(forced, workers=8).status, 'exact')

    def test_spawn_cache_reuse_deadline_includes_startup_and_worker_failure(self):
        state = generate_endgame(42, 3)
        config = _WorkerConfig(state, 'root', 'final_points', None, monotonic(), None, 3)
        tasks = [(a, None) for a in state.legal_action_ids()]
        with OrderedPool(2, initializer=_initialize_oracle, initargs=(config,)) as pool:
            first = pool.map(spawned_probe, tasks)
            second = pool.map(spawned_probe, tasks)
        self.assertIsNone(pool.executor)
        self.assertTrue(all(pid != os.getpid() and method == 'spawn'
                            for _, pid, method, _, _ in first + second))
        self.assertTrue(all(size <= 3 for _, _, _, size, _ in first + second))
        self.assertTrue(all(hits > 0 for _, _, _, _, hits in first + second))
        # A positive duration that elapsed before spawn must not become a fresh
        # per-worker second. No wall-clock timing assertion is needed.
        expired = replace(config, start=monotonic() - 10, time_seconds=1)
        with OrderedPool(2, initializer=_initialize_oracle, initargs=(expired,)) as pool:
            results = pool.map(_oracle_worker, tasks)
        self.assertTrue(all(nodes == 0 for _, nodes, _ in results))
        with OrderedPool(2, initializer=_initialize_oracle, initargs=(config,)) as pool:
            with self.assertRaisesRegex(RuntimeError, 'Oracle root task 99 failed'):
                pool.map(_oracle_worker, [(99, 50), *tasks])
        self.assertIsNone(pool.executor)

    def test_allocation_reclaims_unused_nodes_and_divides_cache_capacity(self):
        state = generate_endgame(7, 2)
        search = _Search(SearchLimits(max_nodes=101, max_cache_entries=5), monotonic(), objective='final_points')
        seen = []

        class InlinePool:
            def __init__(self, workers, *, initializer, initargs):
                self.config = initargs[0]
                self.cache = {}
                self.assertions = (workers, self.config.cache_entries)
                seen.append(self.assertions)
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, function, tasks):
                # Exercise coordinator reservations with cheap completed tasks,
                # leaving most of every allocation available for the next wave.
                seen.append([allocation for _, allocation in tasks])
                return [(oracle.ActionValue(1, 1, key, True), 1, 0) for key, _ in tasks]

        with patch('texas42.oracle.OrderedPool', InlinePool):
            results = _parallel_tasks(state, search, list(range(6)), mode='root', start=monotonic(), workers=2)
        self.assertEqual(seen, [(2, 2), [17, 17], [25, 25], [49, 48]])
        self.assertEqual(sum(nodes for _, nodes, _ in results), 6)

    def test_invalid_worker_counts_and_cli_reporting(self):
        state = generate_endgame(42, 2)
        for value in (-1, True, 1.5, None):
            for solver in (solve, solve_points):
                with self.assertRaises(ValueError):
                    solver(state, workers=value)
        with patch('texas42.parallel.available_cpus', return_value=32):
            with redirect_stdout(io.StringIO()) as out:
                main(['--mode', 'deal', '--seed', '42', '--workers', '99', '--max-nodes', '0'])
        self.assertIn('requested oracle workers: 99', out.getvalue().lower())
        self.assertIn('resolved oracle workers: 8', out.getvalue())
        self.assertIn('independent search tasks: 36', out.getvalue())
        with redirect_stdout(io.StringIO()) as out:
            main(['--seed', '1', '--remaining', '3', '--workers', '2', '--max-nodes', '30'])
        self.assertIn('Nodes:', out.getvalue())


if __name__ == '__main__':
    unittest.main()
