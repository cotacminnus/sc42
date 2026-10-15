"""Exact ordering, threshold proofs and coordinator-owned proof cancellation."""
from dataclasses import replace
from time import monotonic
import unittest

from examples.endgame_oracle import generate_endgame
from texas42 import oracle
from texas42.actions import TRUMP_ACTIONS


class OrderingTests(unittest.TestCase):
    def test_all_modes_against_exhaustive_both_actors_and_partial_tricks(self):
        actors = set()
        for seed in (0, 1, 7):
            for partial in (0, 3):
                state = generate_endgame(seed, 2, partial_plays=partial)
                actors.add(state.seat % 2)
                for team in (0, 1):
                    truth = oracle.exhaustive_minimax(state, objective='final_points', reference_team=team)
                    for ordering in ('none', 'heuristic', 'tt', 'combined'):
                        with self.subTest(seed=seed, partial=partial, team=team, ordering=ordering):
                            result = oracle.solve_points(state, reference_team=team, move_ordering=ordering)
                            self.assertEqual(result.value, truth)
                            self.assertTrue(result.all_action_values_complete)
                            self.assertEqual(tuple(a.action_id for a in result.action_values), state.legal_action_ids())
                            for row in result.action_values:
                                expected = oracle.exhaustive_minimax(oracle._child(state, row.action_id),
                                                   objective='final_points', reference_team=team)
                                self.assertEqual(row.value, expected)
                            binary = oracle.solve(state, move_ordering=ordering)
                            self.assertEqual(binary.value, oracle.exhaustive_minimax(state))
        self.assertEqual(actors, {0, 1})

    def test_legal_hint_and_no_moves_removed_all_nine_trumps(self):
        states = {}
        for seed in range(200):
            state = generate_endgame(seed, 2, policy='HRR', partial_plays=2)
            states.setdefault(state.trump, state)
            if len(states) == 9:
                break
        self.assertEqual(set(states), set(TRUMP_ACTIONS))
        for state in states.values():
            legal = state.legal_action_ids()
            for mode in ('none', 'heuristic', 'tt', 'combined'):
                search = oracle._Search(oracle.SearchLimits(), monotonic(), objective='final_points', move_ordering=mode)
                for hint in (None, 99, legal[-1]):
                    ordered = search.ordered_actions(state, hint)
                    self.assertEqual(sorted(ordered), sorted(legal))
                    self.assertEqual(len(set(ordered)), len(legal))
                    if mode in ('tt', 'combined') and hint in legal:
                        self.assertEqual(ordered[0], hint)
                truth = oracle.exhaustive_minimax(state, objective='final_points')
                self.assertEqual(search.search(state).value, truth)

    def test_cached_hints_are_not_proofs_and_interrupted_results_not_exact(self):
        state = generate_endgame(1, 3)
        truth = oracle.solve_points(state).value
        for hint in (*state.legal_action_ids(), 99):
            search = oracle._Search(oracle.SearchLimits(), monotonic(), objective='final_points', move_ordering='combined')
            key = oracle._state_key(state, 'final_points')
            search.cache[key] = oracle._Entry(0, oracle._Flag.LOWER_BOUND, oracle.ValueBounds(0, 42), hint)
            self.assertEqual(search.search(state).value, truth)
            self.assertGreater(search.statistics.tt_cutoffs, 0)
        for cap in (0, 1, 7):
            search = oracle._Search(oracle.SearchLimits(max_nodes=cap), monotonic(),
                                    objective='final_points', move_ordering='combined')
            bounds = search.search(state)
            self.assertLessEqual(bounds.lower, truth)
            self.assertGreaterEqual(bounds.upper, truth)
            self.assertNotIn(oracle._state_key(state, 'final_points'), search.cache)

    def test_ordering_validation_and_statistics(self):
        state = generate_endgame(1, 3)
        for function in (oracle.solve, oracle.solve_points):
            with self.assertRaisesRegex(ValueError, 'move_ordering'):
                function(state, move_ordering='random')
        result = oracle.solve_points(state, move_ordering='combined')
        self.assertGreater(result.statistics.ordering_calls, 0)
        self.assertGreater(result.statistics.score_bound_cutoffs, 0)
        self.assertEqual(result.statistics + oracle.SearchStatistics(), result.statistics)


class ContractProofTests(unittest.TestCase):
    def test_threshold_search_against_exhaustive_and_shared_node_budget(self):
        for seed in range(8):
            for partial in (0, 3):
                state = generate_endgame(seed, 2, partial_plays=partial)
                for team in (0, 1):
                    truth = oracle.exhaustive_minimax(state, objective='final_points', reference_team=team)
                    for ordering in ('none', 'combined'):
                        search = oracle._Search(oracle.SearchLimits(), monotonic(), objective='final_points',
                                                reference_team=team, move_ordering=ordering)
                        result = oracle._contract_search(state, search)
                        self.assertLessEqual(result.lower, truth)
                        self.assertGreaterEqual(result.upper, truth)
                        self.assertEqual(oracle._makeable_bid(result.lower), oracle._makeable_bid(truth))
                        self.assertEqual(oracle._makeable_bid(result.upper), oracle._makeable_bid(truth))
                        if truth >= 30:
                            self.assertEqual(result.value, truth)
                    for cap in (0, 1, 7, 20):
                        search = oracle._Search(oracle.SearchLimits(max_nodes=cap), monotonic(),
                                                objective='final_points', reference_team=team)
                        result = oracle._contract_search(state, search)
                        self.assertLessEqual(search.nodes, cap)
                        self.assertLessEqual(result.lower, truth)
                        self.assertGreaterEqual(result.upper, truth)

    def test_aggregate_proof_requires_sound_upper_bounds(self):
        def bidder(intervals):
            rows = tuple(oracle.ActionValue(lo, hi, i, True) for i, (lo, hi) in enumerate(intervals))
            return oracle._bidder_analysis(0, (oracle.TrumpAnalysis(action_values=rows),))
        winning = bidder([(42, 42), (0, 42)])
        self.assertTrue(winning.contract_exact)
        self.assertEqual(winning.maximum_contract.marks, 1)
        self.assertIsNotNone(winning.optimal_choice)
        self.assertIsNone(winning.optimal_choices)
        self.assertFalse(winning.all_action_values_complete)
        self.assertFalse(bidder([(0, 29), (0, 30)]).contract_exact)
        below = bidder([(0, 29), (10, 28)])
        self.assertTrue(below.contract_exact)
        self.assertIsNone(below.maximum_contract)
        self.assertIsNone(below.value)
        self.assertFalse(bidder([(35, 35), (0, 36)]).contract_exact)
        self.assertTrue(bidder([(35, 35), (0, 35)]).contract_exact)
        self.assertFalse(bidder([(41, 41), (0, 42)]).contract_exact)
        self.assertTrue(bidder([(41, 41), (0, 41)]).contract_exact)
        deal = oracle.DealAnalysis((below,), 0, 0, 0, output='maximum_contract')
        self.assertTrue(deal.requested_output_complete)
        self.assertEqual(deal.status, 'budget_exceeded')
        self.assertFalse(replace(deal, output='all_action_values').requested_output_complete)


def synthetic_proof_worker(task):
    # Every choice is 42 in this fixture; only actual returned choices may be
    # exact, and unvisited alternatives remain [0, 42] despite known fixture truth.
    (seat, trump, action), _ = task
    return oracle._LeadResult(oracle.ActionValue(42, 42, action, True), 1, status='exact')


def synthetic_failure(task):
    raise ValueError('synthetic proof worker failed')


class ProofSchedulingTests(unittest.TestCase):
    @staticmethod
    def hands():
        from texas42.arena import generate_schedule
        from texas42.actions import PLAY_ACTIONS
        return [[PLAY_ACTIONS[i] for i in hand] for hand in generate_schedule(1, 42)[0].hands]

    def test_sequential_proof_stops_after_42_but_exhaustive_retains_all_ties(self):
        from unittest.mock import patch
        from copy import deepcopy
        hands = self.hands()
        before = deepcopy(hands)
        with patch('texas42.oracle._search_task', side_effect=lambda config, task, cache: synthetic_proof_worker(task)):
            events = []
            result = oracle.analyze_deal(hands, output='maximum_contract', progress=events.append)
            all_values = oracle.analyze_deal(hands)
            no_cancel = oracle.analyze_deal(hands, output='maximum_contract', proof_cancellation=False)
        self.assertEqual(hands, before)
        self.assertEqual((result.nodes, all_values.nodes, no_cancel.nodes), (4, 252, 252))
        self.assertEqual(dict(result.task_status_counts), {'exact': 4, 'skipped_proven': 248})
        self.assertEqual([e.completed for e in events], list(range(1, 253)))
        self.assertEqual(sum(e.bidder_contract_proven for e in events), 4)
        self.assertTrue(result.requested_output_complete)
        self.assertEqual(result.status, 'budget_exceeded')
        self.assertFalse(result.all_alternative_values_complete)
        for bidder in result.bidders:
            self.assertEqual(bidder.value, 42)
            self.assertEqual(bidder.maximum_contract.marks, 1)
            self.assertIsNone(bidder.optimal_choices)
            self.assertIsNotNone(bidder.optimal_choice)
            for trump in bidder.trumps:
                for row in trump.action_values:
                    if not row.searched:
                        self.assertEqual((row.lower, row.upper, row.value), (0, 42, None))
        self.assertEqual(all_values.status, 'exact')
        self.assertTrue(all_values.requested_output_complete)
        self.assertTrue(all(len(b.optimal_choices) == 63 for b in all_values.bidders))

    def test_below_30_and_intermediate_classifications_wait_for_all_upper_bounds(self):
        from unittest.mock import patch
        def classify(config, task, cache):
            seat, trump, action = task[0]
            bounds = ((0, 29), (35, 35), (41, 41), (42, 42))[seat]
            return oracle._LeadResult(oracle.ActionValue(*bounds, action, True), 1,
                                     status='exact' if bounds[0] == bounds[1] else 'contract_proven')
        events = []
        with patch('texas42.oracle._search_task', side_effect=classify):
            result = oracle.analyze_deal(self.hands(), output='maximum_contract', progress=events.append)
        self.assertTrue(result.requested_output_complete)
        self.assertEqual(result.nodes, 63 * 3 + 1)
        for seat in (0, 1, 2):
            proof = next(e for e in events if e.bidder == seat and e.bidder_contract_proven)
            # No earlier task for this bidder can be skipped or prove a class.
            earlier = [e for e in events[:proof.completed] if e.bidder == seat]
            self.assertEqual(len(earlier), 63)
        self.assertIsNone(result.bidders[0].value)
        self.assertTrue(result.bidders[0].contract_exact)
        self.assertIsNone(result.bidders[0].maximum_contract)

    def test_spawn_coordinator_proofs_late_results_and_canonical_order(self):
        from unittest.mock import patch
        events = []
        with patch('texas42.parallel.available_cpus', return_value=2):
            with patch('texas42.oracle._oracle_worker', synthetic_proof_worker):
                result = oracle.analyze_deal(self.hands(), workers=2, output='maximum_contract', progress=events.append)
        self.assertTrue(result.requested_output_complete)
        self.assertLess(result.nodes, 252)
        self.assertEqual(len(events), 252)
        self.assertEqual(len({(e.bidder, e.trump, e.action_id) for e in events}), 252)
        self.assertEqual(sum(e.nodes for e in events), result.nodes)
        self.assertEqual(sum(e.bidder_contract_proven for e in events), 4)
        self.assertEqual(tuple(b.seat for b in result.bidders), (0, 1, 2, 3))
        for bidder in result.bidders:
            self.assertEqual(bidder.maximum_contract.marks, 1)
            self.assertEqual(tuple(t.trump for t in bidder.trumps), TRUMP_ACTIONS)
            self.assertIsNone(bidder.optimal_choices)
            for trump in bidder.trumps:
                ids = tuple(a.action_id for a in trump.action_values)
                self.assertEqual(ids, tuple(sorted(ids)))

    def test_real_spawn_shared_stop_flags_and_worker_shutdown(self):
        from multiprocessing.shared_memory import SharedMemory
        from texas42.parallel import OrderedPool
        memory = SharedMemory(create=True, size=4)
        try:
            memory.buf[:] = bytes((1, 0, 0, 0))
            state = oracle.HandState(self.hands(), 0, (0, 0), 7)
            config = oracle._WorkerConfig(state, 'deal', 'final_points', None, monotonic(), 5, 100,
                                          'combined', 'maximum_contract', memory.name)
            actions = [min(d.id for d in hand) for hand in state.hands]
            with OrderedPool(2, initializer=oracle._initialize_oracle, initargs=(config,)) as pool:
                results = pool.map(oracle._oracle_worker, [((seat, 0, actions[seat]), 5) for seat in (0, 1)])
            self.assertIsNone(pool.executor)
            self.assertEqual((results[0].status, results[0].nodes, results[0].action.searched),
                             ('skipped_proven', 0, False))
            self.assertLessEqual(results[1].nodes, 5)
            self.assertGreater(results[1].nodes, 0)
        finally:
            memory.close()
            memory.unlink()

    def test_running_task_cooperatively_returns_partial_bounds_without_caching_root(self):
        from unittest.mock import patch
        class Flags:
            calls = 0
            def __getitem__(self, seat):
                self.calls += 1
                return self.calls >= 3
        state = generate_endgame(42, 4)
        bidder = state.contract.bidder
        config = oracle._WorkerConfig(state, 'deal', 'final_points', None, monotonic(), None, 100)
        cache, flags = {}, Flags()
        action = state.legal_action_ids()[0]
        with patch('texas42.oracle._play_start', return_value=state):
            result = oracle._search_task(config, ((bidder, state.trump, action), 1000), cache, stop_flags=flags)
        self.assertEqual(result.status, 'cancelled_proven')
        self.assertEqual(result.nodes, 128)
        self.assertTrue(result.action.searched)
        self.assertFalse(result.action.exact)
        child = oracle._child(state, action)
        self.assertNotIn(oracle._state_key(child, 'final_points', bidder % 2), cache)
        truth = oracle.exhaustive_minimax(child, objective='final_points', reference_team=bidder % 2)
        self.assertLessEqual(result.action.lower, truth)
        self.assertGreaterEqual(result.action.upper, truth)

    def test_pending_skip_preserves_completed_futures_and_cancels_queued(self):
        from concurrent.futures import Future
        from texas42.parallel import OrderedPool
        pool = OrderedPool(1)
        pool.workers = 2
        class Executor:
            def submit(self, function, task):
                future = Future()
                if task == 0:
                    future.set_result(42)
                return future
            def shutdown(self, **kwargs): pass
        pool.executor = Executor()
        proven = [False]
        iterator = pool.iter_completed(abs, range(4), skip=lambda i: 'skipped_proven' if proven[0] else None)
        self.assertEqual(next(iterator), (0, 42, 'completed'))
        proven[0] = True
        rest = list(iterator)
        self.assertEqual({(i, status) for i, _, status in rest},
                         {(1, 'skipped_proven'), (2, 'skipped_proven'), (3, 'skipped_proven')})
        pool.close()

    def test_ordered_parallel_endgames_and_proof_search_without_cache(self):
        from unittest.mock import patch
        state = generate_endgame(1, 3, partial_plays=1)
        with patch('texas42.parallel.available_cpus', return_value=2):
            for ordering in ('heuristic', 'tt', 'combined'):
                for team in (0, 1):
                    first = oracle.solve_points(state, reference_team=team, move_ordering=ordering)
                    second = oracle.solve_points(state, reference_team=team, move_ordering=ordering, workers=2)
                    self.assertEqual(first.value, second.value)
                    self.assertEqual(first.optimal_actions, second.optimal_actions)
                    self.assertEqual(first.action_values, second.action_values)
        for team in (0, 1):
            truth = oracle.exhaustive_minimax(state, objective='final_points', reference_team=team)
            for cache in (0, 1):
                search = oracle._Search(oracle.SearchLimits(max_cache_entries=cache), monotonic(),
                                        objective='final_points', reference_team=team, move_ordering='combined')
                bounds = oracle._contract_search(state, search)
                self.assertLessEqual(bounds.lower, truth)
                self.assertGreaterEqual(bounds.upper, truth)
                self.assertEqual(oracle._makeable_bid(bounds.lower), oracle._makeable_bid(truth))
                self.assertEqual(oracle._makeable_bid(bounds.upper), oracle._makeable_bid(truth))

    def test_invalid_options_and_deadline_in_proof_mode(self):
        events = []
        result = oracle.analyze_deal(self.hands(), output='maximum_contract', workers=2,
                                    limits=oracle.SearchLimits(time_seconds=0), progress=events.append)
        self.assertEqual(result.nodes, 0)
        self.assertFalse(result.requested_output_complete)
        self.assertEqual([e.completed for e in events], list(range(1, 253)))
        self.assertTrue(all(e.status == 'cancelled_deadline' for e in events))
        for kwargs in ({'output': 'guess'}, {'move_ordering': 'random'}, {'proof_cancellation': 1}):
            with self.assertRaises(ValueError):
                oracle.analyze_deal(self.hands(), **kwargs)

    def test_worker_error_propagates_and_shared_proof_memory_is_unlinked(self):
        from multiprocessing.shared_memory import SharedMemory
        from unittest.mock import patch
        names = []
        def record(*args, **kwargs):
            memory = SharedMemory(*args, **kwargs)
            names.append(memory.name)
            return memory
        with patch('texas42.parallel.available_cpus', return_value=2):
            with patch('texas42.oracle.SharedMemory', side_effect=record):
                with patch('texas42.oracle._oracle_worker', synthetic_failure):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic proof worker failed'):
                        oracle.analyze_deal(self.hands(), workers=2, output='maximum_contract',
                                            limits=oracle.SearchLimits(5, 10, 100))
        self.assertEqual(len(names), 1)
        for name in names:
            with self.assertRaises(FileNotFoundError):
                SharedMemory(name=name)
