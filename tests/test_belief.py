"""Belief modelling: exact counts, uniform sampling, replay, rollouts, evaluation.

Correctness strategy: the counting DP is checked against an independent brute
force that tries every assignment of unseen dominoes and keeps those the ENGINE
can replay. Agreement shows the hard constraints are neither too loose nor too
strict. All randomness is seeded so failures reproduce.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
import math
import random
import unittest

from texas42 import SelfPlayEnv
from texas42.agents import (ActionProbabilities, ConservativeRandomAgent, HeuristicAgent,
                            HybridAgent, PrivilegedState, RandomAgent, variant_factory)
from texas42.belief import (
    _copy_state, _replay, build_belief, count_worlds, enumerate_worlds, evaluate_actions,
    hidden_constraints, policy_likelihood, sample_world, simulate, tile_marginals, world_state,
)
from texas42.selfplay import observation_from_view

HHH = [variant_factory('HHH')] * 4
RANDOM = [lambda seat, seed: RandomAgent(seed)] * 4


def heuristic_action(obs):
    agent = HeuristicAgent()
    agent.reset(obs.seat)
    return agent.act(obs)


def play_until(seed, predicate, *, chooser=None):
    """Advance a seeded hand until predicate(obs) holds; return (env, obs)."""
    env = SelfPlayEnv(seed=seed)
    rng = random.Random(seed)
    obs = env.reset()
    while obs is not None and not predicate(obs):
        action = chooser(obs) if chooser else rng.choice(obs.legal_action_ids)
        obs = env.step(action).observation
    if obs is None:
        raise AssertionError('Hand ended before the requested position')
    return env, obs


def at_trick(n, *, lead=True):
    return lambda o: (o.phase == 'play' and len(o.view.completed_tricks) == n
                      and (not o.view.trick if lead else bool(o.view.trick)))


def true_world(env):
    """The real remaining hands. TEST-ONLY privileged access for checking answers."""
    return PrivilegedState(tuple(tuple(sorted(d.id for d in hand))
                                 for hand in env._session.hand.hands))


def brute_force_count(obs, constraints):
    """Count assignments of unseen dominoes that the engine accepts on replay."""
    seats = [s for s in range(4) if constraints.slots[s]]
    total = 0

    def assign(index, rest, hands):
        nonlocal total
        if index == len(seats):
            full = [list(h) for h in hands]
            full[obs.seat] = list(constraints.observer_hand)
            try:
                _replay(obs, PrivilegedState(tuple(tuple(sorted(h)) for h in full)))
                total += 1
            except ValueError:
                pass
            return
        seat = seats[index]
        for chosen in combinations(rest, constraints.slots[seat]):
            hands[seat] = chosen
            assign(index + 1, [d for d in rest if d not in chosen], hands)
        hands[seat] = ()

    assign(0, list(constraints.unknown), [(), (), (), ()])
    return total


class CountingTests(unittest.TestCase):
    def test_counts_match_engine_brute_force_and_include_truth(self):
        checked = 0
        for seed in range(6):
            env = SelfPlayEnv(seed=seed)
            rng = random.Random(seed)
            obs = env.reset()
            while obs is not None:
                constraints = hidden_constraints(obs)
                count = count_worlds(constraints)
                if count <= 400:
                    self.assertEqual(count, brute_force_count(obs, constraints))
                    worlds = list(enumerate_worlds(constraints))
                    self.assertEqual(len(worlds), count)
                    self.assertEqual(len(set(worlds)), count)
                    self.assertIn(true_world(env), worlds)
                    checked += 1
                obs = env.step(rng.choice(obs.legal_action_ids)).observation
        self.assertGreater(checked, 30)

    def test_bidding_count_and_vacant_places(self):
        _, obs = play_until(1, lambda o: True)
        constraints = hidden_constraints(obs)
        # 21 unseen dominoes dealt 7/7/7: 21! / (7!)^3.
        self.assertEqual(count_worlds(constraints), 399_072_960)
        for row in tile_marginals(constraints).values():
            self.assertEqual(row[obs.seat], 0.0)
            for seat in range(4):
                if seat != obs.seat:
                    self.assertAlmostEqual(row[seat], 1 / 3)

    def test_marginals_match_enumeration_and_respect_voids(self):
        for seed in range(30):
            env, obs = play_until(seed, at_trick(4))
            constraints = hidden_constraints(obs)
            if constraints.voids:
                break
        else:
            self.fail('No position with a revealed void found')
        exact = tile_marginals(constraints)
        belief = build_belief(obs, rng=random.Random(0), enumerate_below=10_000)
        self.assertEqual(belief.mode, 'enumerated')
        for domino, row in belief.marginals().items():
            self.assertAlmostEqual(sum(row), 1.0)
            for a, b in zip(row, exact[domino]):
                self.assertAlmostEqual(a, b)
        for domino, seats in zip(constraints.unknown, constraints.allowed):
            for seat in range(4):
                if seat not in seats:
                    self.assertEqual(exact[domino][seat], 0.0)

    def test_inconsistent_observation_is_rejected(self):
        _, obs = play_until(2, at_trick(1))
        broken = replace(obs, view=replace(obs.view, hand=obs.view.hand[:-1]))
        with self.assertRaises(ValueError):
            hidden_constraints(broken)


class SamplingTests(unittest.TestCase):
    def test_sampling_is_uniform_and_reproducible(self):
        _, obs = play_until(4, at_trick(5))
        constraints = hidden_constraints(obs)
        worlds = list(enumerate_worlds(constraints))
        rng = random.Random(123)
        draws = 300 * len(worlds)
        counts = Counter(sample_world(constraints, rng) for _ in range(draws))
        self.assertEqual(set(counts), set(worlds))
        # Chi-square goodness of fit against uniform. The bound is the 0.999
        # quantile approximated as df + 4.5*sqrt(2*df), a loose, fixed-seed check.
        expected = draws / len(worlds)
        chi2 = sum((counts[w] - expected) ** 2 / expected for w in worlds)
        df = len(worlds) - 1
        self.assertLess(chi2, df + 4.5 * math.sqrt(2 * df))
        self.assertEqual(sample_world(constraints, random.Random(9)),
                         sample_world(constraints, random.Random(9)))

    def test_large_beliefs_sample_and_small_ones_enumerate(self):
        _, opening = play_until(5, at_trick(0))
        belief = build_belief(opening, rng=random.Random(0), samples=10)
        self.assertEqual((belief.mode, len(belief.worlds)), ('sampled', 10))
        self.assertAlmostEqual(belief.effective_sample_size, 10)
        _, late = play_until(5, at_trick(6))
        belief = build_belief(late, rng=random.Random(0))
        self.assertEqual(belief.mode, 'enumerated')
        self.assertEqual(len(belief.worlds), belief.world_count)


class ReplayTests(unittest.TestCase):
    def test_every_world_reproduces_the_same_observation(self):
        _, obs = play_until(7, at_trick(3, lead=False))
        def normalized(o):
            # Dealt order within a hand carries no information; compare as sets.
            view = replace(o.view, hand=tuple(sorted(o.view.hand)),
                           legal_plays=tuple(sorted(o.view.legal_plays)))
            return replace(o, view=view)

        for world in list(enumerate_worlds(hidden_constraints(obs), limit=50_000))[:200]:
            state = world_state(obs, world)
            self.assertEqual(normalized(observation_from_view(state.view())), normalized(obs))

    def test_worlds_contradicting_history_are_rejected(self):
        env, obs = play_until(7, at_trick(3))
        truth = true_world(env)
        wrong_own = list(truth.hands)
        other = (obs.seat + 1) % 4
        if not wrong_own[other]:
            self.skipTest('Neighbour has no dominoes left')
        # Swap one domino between the observer and a neighbour.
        mine, theirs = wrong_own[obs.seat][0], wrong_own[other][0]
        wrong_own[obs.seat] = tuple(sorted((*wrong_own[obs.seat][1:], theirs)))
        wrong_own[other] = tuple(sorted((*wrong_own[other][1:], mine)))
        with self.assertRaises(ValueError):
            world_state(obs, PrivilegedState(tuple(wrong_own)))

    def test_state_copy_is_independent(self):
        env, obs = play_until(8, at_trick(2))
        original = world_state(obs, true_world(env))
        before = (original.hands[obs.seat][:], len(original.trick), list(original.points))
        clone = _copy_state(original)
        while clone.result is None:
            clone.apply(clone.legal_action_ids()[0])
        self.assertEqual((original.hands[obs.seat], len(original.trick), list(original.points)),
                         before)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.env, self.obs = play_until(10, at_trick(2))
        self.state = world_state(self.obs, true_world(self.env))

    def test_choices_are_indices_and_input_is_not_mutated(self):
        legal = self.state.legal_action_ids()
        before = list(self.state.hands[self.obs.seat])
        rollout = simulate(self.state, rng=random.Random(0), choices=(len(legal) - 1,), until='trick')
        self.assertEqual(rollout.action_ids[0], legal[-1])
        self.assertEqual(rollout.explicit_steps, 1)
        self.assertEqual(rollout.stop_reason, 'trick')
        self.assertEqual(len(rollout.action_ids), 4)
        self.assertEqual(sum(rollout.points_delta), sum(rollout.end_points) - sum(rollout.start_points))
        self.assertEqual(list(self.state.hands[self.obs.seat]), before)

    def test_random_rollout_probability_is_product_of_uniform_choices(self):
        rollout = simulate(self.state, rng=random.Random(1), until='hand')
        # Recompute legal counts along the recorded path with the engine.
        replay, expected = _copy_state(self.state), 1.0
        for action_id in rollout.action_ids:
            expected /= len(replay.legal_action_ids())
            replay.apply(action_id)
        self.assertAlmostEqual(rollout.rollout_probability, expected)
        self.assertEqual(rollout.stop_reason, 'hand')
        self.assertIsNotNone(rollout.hand_result)
        self.assertEqual(sum(rollout.end_points), 42)

    def test_policy_rollouts_and_stop_conditions(self):
        heuristic = simulate(self.state, rng=random.Random(2), policies=HHH, until='hand')
        self.assertEqual(heuristic.rollout_probability, 1.0)  # Deterministic policy.
        untracked = simulate(self.state, rng=random.Random(2), policies=HHH, until='hand',
                             track_probability=False)
        self.assertEqual(untracked.action_ids, heuristic.action_ids)
        self.assertIsNone(untracked.rollout_probability)
        _, bid_obs = play_until(10, lambda o: True)
        bid_state = world_state(bid_obs, sample_world(hidden_constraints(bid_obs), random.Random(0)))
        phase = simulate(bid_state, rng=random.Random(3), until='phase')
        self.assertEqual(phase.stop_reason, 'phase')
        self.assertEqual(phase.final_state.phase, 'trump')
        self.assertEqual(simulate(self.state, rng=random.Random(4), policies=HHH, until='trick').action_ids,
                         simulate(self.state, rng=random.Random(4), policies=HHH, until='trick').action_ids)

    def test_turn_stops_at_the_starting_seats_next_decision(self):
        start = self.state.seat
        stops = set()
        for seed in range(12):
            rollout = simulate(self.state, rng=random.Random(seed), choices=(seed % len(self.obs.legal_action_ids),),
                               policies=HHH, until='turn')
            stops.add(rollout.stop_reason)
            # Replay the path: the starting seat acts first, then never again until the stop.
            replay = _copy_state(self.state)
            for step, action_id in enumerate(rollout.action_ids):
                self.assertEqual(replay.seat == start, step == 0)
                replay.apply(action_id)
            if rollout.stop_reason == 'turn':
                self.assertEqual(rollout.final_state.seat, start)
                self.assertIsNone(rollout.hand_result)
            again = simulate(self.state, rng=random.Random(seed), choices=(seed % len(self.obs.legal_action_ids),),
                             policies=HHH, until='turn')
            self.assertEqual(again.action_ids, rollout.action_ids)
        self.assertIn('turn', stops)
        self.assertLessEqual(stops, {'turn', 'hand'})
        # In the last trick the starting seat never acts again: the hand ends first.
        env, last = play_until(10, at_trick(6))
        final = simulate(world_state(last, true_world(env)), rng=random.Random(0), policies=HHH, until='turn')
        self.assertEqual(final.stop_reason, 'hand')
        self.assertIsNotNone(final.hand_result)
        with self.assertRaises(ValueError):  # At most 7 plays reach the next own decision.
            simulate(self.state, rng=random.Random(0), choices=(0,) * 8, until='turn')

    def test_invalid_choices_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate(self.state, rng=random.Random(0), choices=(99,))
        with self.assertRaises(ValueError):
            simulate(self.state, rng=random.Random(0), choices=(0,) * 5, until='trick')
        with self.assertRaises(ValueError):
            simulate(self.state, rng=random.Random(0), until='round')


class LikelihoodTests(unittest.TestCase):
    def test_true_world_is_most_likely_when_the_model_is_correct(self):
        # Everyone actually plays Heuristic; the model assumes Heuristic too.
        env, obs = play_until(12, at_trick(4), chooser=heuristic_action)
        belief = build_belief(obs, rng=random.Random(0), enumerate_below=10_000,
                              likelihood_policies=HHH, epsilon=0.1)
        weight = dict(zip(belief.worlds, belief.weights))
        self.assertAlmostEqual(weight[true_world(env)], max(belief.weights))
        self.assertLess(belief.effective_sample_size, len(belief.worlds))

    def test_epsilon_one_means_a_uniformly_random_player_model(self):
        # epsilon=1 replaces the model with "each player picks uniformly among
        # its legal moves". That is NOT a flat belief: the number of legal
        # moves a player had depends on its hidden hand, so worlds differ.
        _, obs = play_until(13, at_trick(5), chooser=heuristic_action)
        mixed = build_belief(obs, rng=random.Random(0), likelihood_policies=HHH, epsilon=1.0)
        random_model = build_belief(obs, rng=random.Random(0), likelihood_policies=RANDOM, epsilon=0.0)
        for a, b in zip(mixed.weights, random_model.weights):
            self.assertAlmostEqual(a, b)
        flat = build_belief(obs, rng=random.Random(0))
        self.assertTrue(all(math.isclose(w, flat.weights[0]) for w in flat.weights))
        world = flat.worlds[0]
        with self.assertRaises(ValueError):
            policy_likelihood(obs, world, HHH, epsilon=1.5)

        class Opaque:
            def __init__(self, seat, seed):
                self.inner = RandomAgent(seed)

            def reset(self, seat):
                self.inner.reset(seat)

            def act(self, obs):
                return self.inner.act(obs)

        with self.assertRaises(TypeError):
            policy_likelihood(obs, world, [Opaque] * 4)


class EvaluationTests(unittest.TestCase):
    def test_paired_estimates_for_every_legal_action(self):
        _, obs = play_until(14, at_trick(1))
        evaluation = evaluate_actions(obs, rng=random.Random(5), rollout_policies=HHH, samples=16)
        self.assertEqual([e.action_id for e in evaluation.estimates], list(obs.legal_action_ids))
        best = [e for e in evaluation.estimates if e.action_id == evaluation.best_action_id][0]
        self.assertEqual(best.advantage, 0.0)
        self.assertTrue(all(e.advantage <= 1e-12 for e in evaluation.estimates))
        self.assertTrue(all(e.expected_signed_marks is not None for e in evaluation.estimates))
        self.assertEqual(evaluation.mode, 'sampled')
        again = evaluate_actions(obs, rng=random.Random(5), rollout_policies=HHH, samples=16)
        self.assertEqual(again, evaluation)

    def test_exact_values_in_an_enumerated_endgame(self):
        _, obs = play_until(15, at_trick(5))
        evaluation = evaluate_actions(obs, rng=random.Random(0), rollout_policies=HHH)
        self.assertEqual(evaluation.mode, 'enumerated')
        belief = build_belief(obs, rng=random.Random(0))
        team = obs.seat % 2
        for index, estimate in enumerate(evaluation.estimates):
            # Independent recomputation: uniform average of engine outcomes.
            values = [simulate(world_state(obs, w), rng=random.Random(0), choices=(index,),
                               policies=HHH, until='hand').points_delta[team]
                      for w in belief.worlds]
            self.assertAlmostEqual(estimate.expected_points, sum(values) / len(values))

    def test_leaf_evaluator_hook_receives_hypothetical_states(self):
        _, obs = play_until(16, at_trick(3))
        seen = []

        def leaf(state, observer):
            seen.append((len(state.completed), observer))
            return 0.0

        evaluation = evaluate_actions(obs, rng=random.Random(0), rollout_policies=HHH, until='trick',
                                      leaf_evaluator=leaf, samples=4)
        self.assertTrue(seen and all(entry == (4, obs.seat) for entry in seen))
        self.assertTrue(all(e.expected_signed_marks is None for e in evaluation.estimates))


class ActionProbabilityTests(unittest.TestCase):
    def test_builtin_agents_report_pure_distributions(self):
        env = SelfPlayEnv(seed=20)
        obs = env.reset()
        rng = random.Random(20)
        observations = []
        while obs is not None:
            observations.append(obs)
            obs = env.step(rng.choice(obs.legal_action_ids)).observation
        for make in (lambda: RandomAgent(3), lambda: ConservativeRandomAgent(3),
                     HeuristicAgent, lambda: HybridAgent(seed=3)):
            probe, reference = make(), make()
            self.assertIsInstance(probe, ActionProbabilities)
            for o in observations:
                probe.reset(o.seat)
                reference.reset(o.seat)
                distribution = probe.action_probabilities(o)
                self.assertTrue(set(distribution) <= set(o.legal_action_ids))
                self.assertAlmostEqual(sum(distribution.values()), 1.0)
                # Querying must not consume RNG: acts stay in lockstep.
                chosen = probe.act(o)
                self.assertEqual(chosen, reference.act(o))
                self.assertGreater(distribution.get(chosen, 0.0), 0.0)


if __name__ == '__main__':
    unittest.main()
