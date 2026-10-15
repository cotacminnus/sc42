"""Spawned oracle trees, per-tree limits and unchanged proof rules."""
from contextlib import redirect_stdout
from concurrent.futures import Future
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
    # Aggregation-only fixture; actual search is verified on small endgames.
    return synthetic_lead(oracle._WORKER_CONFIG, task)


def synthetic_lead(config, task):
    (seat, trump, action), allocation = task
    ids = sorted(d.id for d in config.state.hands[seat])
    value = 35 if action in ids[:2] and trump in (0, None) else 29
    return oracle._LeadResult(oracle.ActionValue(value, value, action, True), status='exact')


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

    def test_root_tree_node_budget_and_unfinished_proofs(self):
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

    def test_deal_tree_budgets_trump_order_and_input_unchanged(self):
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        before = deepcopy(hands)
        with patch('texas42.parallel.available_cpus', return_value=2):
            for nodes in (1, 35, 80, 1000):
                events = []
                result = analyze_deal(hands, workers=2, limits=SearchLimits(max_nodes=nodes),
                                      progress=events.append)
                self.assertLessEqual(result.nodes, 252 * nodes)
                self.assertGreater(result.nodes, nodes)
                self.assertEqual(len(events), 252)
                self.assertEqual(len({(e.bidder, e.trump, e.action_id) for e in events}), 252)
                self.assertTrue(all(e.nodes <= nodes for e in events))
                self.assertEqual(sum(e.nodes for e in events), result.nodes)
                self.assertEqual(sum(e.cache_hits for e in events), result.cache_hits)
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
        with patch('texas42.oracle._search_task', side_effect=lambda config, task, cache: synthetic_lead(config, task)):
            sequential = analyze_deal(hands)
        self.assertEqual(replace(result, elapsed_seconds=0), replace(sequential, elapsed_seconds=0))
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

        deal = generate_schedule(1, 42)[0]
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in deal.hands]
        deal_config = replace(config, state=oracle.HandState(hands, 0, (0, 0), 7), mode='deal')
        with OrderedPool(2, initializer=_initialize_oracle, initargs=(deal_config,)) as pool:
            with self.assertRaisesRegex(RuntimeError, r'Oracle deal task \(0, 0, 99\) failed'):
                list(pool.iter_completed(_oracle_worker, [((0, 0, 99), 5)]))
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
        self.assertIn('independent search tasks: 252', out.getvalue())
        with redirect_stdout(io.StringIO()) as out:
            main(['--seed', '1', '--remaining', '3', '--workers', '2', '--max-nodes', '30'])
        self.assertIn('Nodes:', out.getvalue())

    def test_defaults_distinct_tasks_fresh_node_allowances_and_global_deadline(self):
        defaults = SearchLimits()
        self.assertEqual((defaults.time_seconds, defaults.max_nodes), (1800.0, 1_048_576))
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        seen, events = [], []
        original = oracle._search_task

        def record(config, task, cache):
            seen.append((config, task))
            # Verify default allocations without executing million-node trees.
            return oracle._LeadResult(oracle.ActionValue(0, 42, task[0][2]))

        with patch('texas42.oracle._search_task', side_effect=record):
            analyze_deal(hands, progress=events.append)
        self.assertEqual(len(seen), 252)
        self.assertEqual(len(set(task[0] for _, task in seen)), 252)
        self.assertTrue(all(task[1] == 1_048_576 for _, task in seen))
        self.assertEqual(len(set(c.start for c, _ in seen)), 1)
        for config, ((seat, trump, action), _) in seen:
            state = oracle._play_start(config.state, seat, trump)
            self.assertEqual(state.seat, seat)
            self.assertIn(action, state.legal_action_ids())
        self.assertEqual([e.completed for e in events], list(range(1, 253)))

        clock = [100.0]
        events.clear()
        def finish_tree(config, task, cache):
            result = original(config, task, cache)
            clock[0] += 10
            return result
        with patch('texas42.oracle.monotonic', side_effect=lambda: clock[0]):
            with patch('texas42.oracle._search_task', side_effect=finish_tree) as worker:
                result = analyze_deal(hands, limits=SearchLimits(time_seconds=1, max_nodes=5),
                                      progress=events.append)
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(result.nodes, 5)
        self.assertEqual([e.status for e in events], ['node_limit'] + ['not_started'] * 251)
        self.assertTrue(all(not a.searched and (a.lower, a.upper) == (0, 42)
                            for b in result.bidders[1:] for t in b.trumps for a in t.action_values))

    def test_deal_worker_global_deadline_and_continuation_matches_reference(self):
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        state = oracle.HandState(hands, 0, (0, 0), 7)
        config = _WorkerConfig(state, 'deal', 'final_points', None, 0.0, 1.0, 5)
        for seat, trump in ((0, 0), (1, None), (3, 'doubles')):
            action = min(d.id for d in hands[seat])
            with patch('texas42.oracle.monotonic', return_value=100.0):
                result = oracle._search_task(config, ((seat, trump, action), 5), {})
            self.assertEqual((result.nodes, result.status), (0, 'not_started'))
            self.assertEqual((result.action.lower, result.action.upper), (0, 42))

        # Exercise the exact same continuation call used by deal tasks against
        # terminal-only exhaustive minimax, both teams and partial tricks.
        for seed, partial in ((0, 0), (1, 3), (7, 0)):
            state = generate_endgame(seed, 2, partial_plays=partial)
            for team in (0, 1):
                config = _WorkerConfig(state, 'root', 'final_points', team, monotonic(), None, 100)
                for action in state.legal_action_ids():
                    actual, _, _ = oracle._search_task(config, (action, None), {})
                    truth = oracle.exhaustive_minimax(oracle._child(state, action),
                                                     objective='final_points', reference_team=team)
                    self.assertEqual(actual.value, truth)

    def test_out_of_order_aggregation_and_partial_proofs(self):
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        events, configs = [], []
        class ReversePool:
            def __init__(self, workers, *, initializer, initargs):
                configs.append(initargs[0])
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_completed(self, function, tasks, *, deadline):
                self_tasks = list(tasks)
                for i in reversed(range(len(self_tasks))):
                    key, allocation = self_tasks[i]
                    seat, trump, action = key
                    # One exact maximum proves bidder value, unresolved ties
                    # must still prevent claiming a complete optimal set.
                    row = (oracle.ActionValue(42, 42, action, True) if i % 63 == 0
                           else oracle.ActionValue(0, 42, action, True))
                    yield i, oracle._LeadResult(row, 1, 0, status='exact' if row.exact else 'node_limit'), 'completed'
        with patch('texas42.parallel.available_cpus', return_value=2):
            with patch('texas42.oracle.OrderedPool', ReversePool):
                result = analyze_deal(hands, workers=2, progress=events.append)
        self.assertEqual(configs[0].cache_entries, 50_000)
        self.assertEqual(events[0].bidder, 3)
        self.assertEqual(events[-1].completed, 252)
        self.assertEqual(result.nodes, 252)
        for bidder in result.bidders:
            self.assertEqual(bidder.value, 42)
            self.assertTrue(bidder.contract_exact)
            self.assertIsNotNone(bidder.optimal_choice)
            self.assertIsNone(bidder.optimal_choices)
            self.assertFalse(bidder.all_action_values_complete)
            self.assertEqual(tuple(t.trump for t in bidder.trumps), TRUMP_ACTIONS)
            for t in bidder.trumps:
                self.assertEqual(tuple(a.action_id for a in t.action_values),
                                 tuple(sorted(d.id for d in hands[bidder.seat])))

    def test_validation_time_and_zero_deadline_settle_all_without_starting(self):
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]
        clock, events = [100.0], []
        original = oracle.EpisodeState.deal
        def validate(session, hands):
            original(session, hands)
            clock[0] += 2
        with patch('texas42.oracle.EpisodeState.deal', validate):
            with patch('texas42.oracle.monotonic', side_effect=lambda: clock[0]):
                with patch('texas42.oracle._search_task', side_effect=AssertionError('late task')):
                    result = analyze_deal(hands, workers=2, limits=SearchLimits(time_seconds=1),
                                          progress=events.append)
        self.assertEqual(result.nodes, 0)
        self.assertEqual(len(events), 252)
        self.assertTrue(all(e.status == 'not_started' for e in events))
        self.assertEqual(result.elapsed_seconds, 2)

    def test_completion_scheduler_refills_before_slow_first_task_and_propagates_errors(self):
        class Executor:
            def __init__(self):
                self.futures = []
                self.closed = False
            def submit(self, function, task):
                future = Future()
                if task != 0:
                    future.set_result(task)
                self.futures.append(future)
                return future
            def shutdown(self, **kwargs): self.closed = True

        pool = OrderedPool(1)
        pool.workers = 2
        executor = pool.executor = Executor()
        iterator = pool.iter_completed(abs, range(4))
        self.assertEqual(next(iterator), (1, 1, 'completed'))
        # The second worker receives new work while index zero is unfinished.
        self.assertEqual(next(iterator), (2, 2, 'completed'))
        self.assertFalse(executor.futures[0].done())
        executor.futures[0].set_result(0)
        remaining = list(iterator)
        self.assertEqual({i for i, _, _ in remaining}, {0, 3})
        pool.close()
        self.assertTrue(executor.closed)

        pool = OrderedPool(1)
        executor = pool.executor = Executor()
        iterator = pool.iter_completed(abs, [0])
        # Fail a submitted future without starting an actual process.
        def fail_wait(pending, **kwargs):
            future = next(iter(pending))
            future.set_exception(ValueError('broken opening tree'))
            return {future}, set()
        with patch('texas42.parallel.wait', side_effect=fail_wait):
            with self.assertRaisesRegex(RuntimeError, 'task 0 failed: broken opening tree'):
                list(iterator)
        self.assertTrue(executor.closed)
        self.assertIsNone(pool.executor)

    def test_completion_scheduler_deadline_cancels_queue_and_stops_submission(self):
        clock = [100.0]
        class Executor:
            def __init__(self): self.calls = 0
            def submit(self, function, task):
                self.calls += 1
                future = Future()
                if task == 0:
                    future.set_result(0)
                else:
                    clock[0] += 10
                return future
            def shutdown(self, **kwargs): pass
        pool = OrderedPool(1)
        pool.workers = 2
        executor = pool.executor = Executor()
        with patch('texas42.parallel.monotonic', side_effect=lambda: clock[0]):
            results = list(pool.iter_completed(abs, range(5), deadline=101.0))
        pool.close()
        self.assertEqual(executor.calls, 2)
        self.assertEqual(dict((i, status) for i, _, status in results),
                         {0: 'completed', 1: 'cancelled', 2: 'not_started',
                          3: 'not_started', 4: 'not_started'})

    def test_opening_worker_bounds_and_exact_values_against_small_real_positions(self):
        for seed in (0, 1, 7):
            state = generate_endgame(seed, 2)
            # Reuse the deal worker's entry path on a small legal state; this
            # fixture substitutes only setup, never engine moves or minimax.
            config = _WorkerConfig(state, 'deal', 'final_points', None, monotonic(), None, 100)
            bidder = state.contract.bidder
            for action in state.legal_action_ids():
                child = oracle._child(state, action)
                truth = oracle.exhaustive_minimax(child, objective='final_points', reference_team=bidder % 2)
                for cap in (0, 1, 3, None):
                    with patch('texas42.oracle._play_start', return_value=state):
                        result = oracle._search_task(config, ((bidder, state.trump, action), cap), {})
                    self.assertLessEqual(result.action.lower, truth)
                    self.assertGreaterEqual(result.action.upper, truth)
                    if cap is None:
                        self.assertEqual(result.action.value, truth)
                        self.assertEqual(result.status, 'exact')
                    else:
                        self.assertLessEqual(result.nodes, cap)

        clock = [100.0]
        config = replace(config, start=100.0, time_seconds=1)
        enter = oracle._Search.enter
        def limited_enter(search):
            if search.nodes == 2:
                clock[0] += 10
            return enter(search)
        with patch('texas42.oracle.monotonic', side_effect=lambda: clock[0]):
            with patch('texas42.oracle._Search.enter', limited_enter):
                with patch('texas42.oracle._play_start', return_value=state):
                    result = oracle._search_task(config, ((bidder, state.trump, action), None), {})
        self.assertEqual(result.status, 'time_limit')
        self.assertTrue(result.exhausted)
        self.assertLessEqual(result.action.lower, truth)
        self.assertGreaterEqual(result.action.upper, truth)


if __name__ == '__main__':
    unittest.main()
