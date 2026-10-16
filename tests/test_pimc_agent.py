"""PIMC agent: protocol, determinism, phase routing, composition and CLI smoke.

Sample counts are tiny (2-3 worlds per decision) to keep the suite fast; these
tests check mechanics and reproducibility, not playing strength.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from examples.evaluate_agent_vs_agent import main as evaluate_main
from research.ppo.composition import (CompositeAgent, CompositeAgentFactory, CompositePolicySpec,
                                      PhasePolicySpec, evaluate_composed)
from research.ppo.opponents import PolicySpec
from texas42.agents import HeuristicAgent, variant_factory
from texas42.agents.pimc import PIMCAgent
from texas42.arena import Deal, generate_schedule, run_game
from texas42.selfplay import SelfPlayEnv


def pimc_factory(samples=2):
    return lambda seat, seed: PIMCAgent(seed, samples=samples)


def observations(seed, predicate):
    """Observations from a seeded random-legal hand that satisfy predicate."""
    env, rng, found = SelfPlayEnv(seed=seed), random.Random(seed), []
    obs = env.reset()
    while obs is not None:
        if predicate(obs):
            found.append(obs)
        obs = env.step(rng.choice(obs.legal_action_ids)).observation
    return found


class PIMCAgentTests(unittest.TestCase):
    def test_full_hand_is_legal_and_seed_deterministic(self):
        # run_game validates every action against the legal IDs.
        schedule = generate_schedule(1, seed=3)
        factories = (pimc_factory(), variant_factory('HHH')) * 2
        first = run_game(factories, schedule, seed=11)
        again = run_game(factories, schedule, seed=11)
        self.assertEqual(first, again)
        self.assertEqual(sum(first.hands[0].result.team_points), 42)

    def test_bidding_is_delegated_to_the_heuristic(self):
        for obs in observations(1, lambda o: o.phase == 'bid'):
            agent, heuristic = PIMCAgent(0), HeuristicAgent()
            agent.reset(obs.seat)
            heuristic.reset(obs.seat)
            self.assertEqual(agent.act(obs), heuristic.act(obs))
            self.assertEqual(agent.would_pass_if_allowed(obs), heuristic.would_pass_if_allowed(obs))
            self.assertIsNone(agent.last_evaluation)

    def test_trump_and_play_use_belief_evaluation(self):
        searched = forced = 0
        for obs in observations(2, lambda o: o.phase in ('trump', 'play')):
            agent = PIMCAgent(5, samples=2)
            agent.reset(obs.seat)
            action = agent.act(obs)
            self.assertIn(action, obs.legal_action_ids)
            if len(obs.legal_action_ids) == 1:
                self.assertIsNone(agent.last_evaluation)  # Forced move: no search.
                forced += 1
            else:
                self.assertEqual(agent.last_evaluation.best_action_id, action)
                self.assertEqual([e.action_id for e in agent.last_evaluation.estimates],
                                 list(obs.legal_action_ids))
                searched += 1
        self.assertGreater(searched, 0)
        self.assertGreater(forced, 0)

    def test_validation_and_seat_assignment(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                PIMCAgent(samples=bad)
        with self.assertRaises(ValueError):
            PIMCAgent(enumerate_below=-1)
        agent = PIMCAgent(0)
        with self.assertRaises(ValueError):
            agent.reset(4)
        obs = observations(3, lambda o: True)[0]
        agent.reset((obs.seat + 1) % 4)
        with self.assertRaises(RuntimeError):
            agent.act(obs)
        self.assertFalse(hasattr(agent, 'action_probabilities'))  # Consumes RNG; not a likelihood model.


class PIMCCompositionTests(unittest.TestCase):
    def spec(self, name='P'):
        return CompositePolicySpec(name, PhasePolicySpec('heuristic'), PhasePolicySpec('pimc'),
                                   PhasePolicySpec('pimc'))

    def test_pimc_allowed_for_trump_and_play_only(self):
        PhasePolicySpec('pimc').validate('trump')
        PhasePolicySpec('pimc').validate('play')
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            PhasePolicySpec('pimc').validate('bid')
        self.assertEqual(self.spec().label(), '[Heuristic Bid | PIMC Trump | PIMC Play]')

    def test_factory_routes_phases_and_sample_setting(self):
        agent = CompositeAgentFactory(self.spec(), pimc_samples=3)(0, 9)
        self.assertIsInstance(agent, CompositeAgent)
        self.assertIsInstance(agent.delegates['play'], PIMCAgent)
        self.assertIsInstance(agent.delegates['trump'], PIMCAgent)
        self.assertEqual(agent.delegates['play'].samples, 3)
        self.assertIsInstance(agent.delegates['bid'], HeuristicAgent)

    def test_existing_composite_identities_are_unchanged(self):
        # Digest computed on main before 'pimc' existed: a factory-level sample
        # setting must not alter identities stored in training checkpoints.
        spec = CompositePolicySpec('PH', PhasePolicySpec('conservative'), PhasePolicySpec('random'),
                                   PhasePolicySpec('heuristic'))
        self.assertEqual(PolicySpec('PH', 'composite', composition=spec).sha256,
                         '16e18b42ebb2f2668cd255be92238fed7e216092ef3148923e1885d7429b4350')

    def test_paired_arena_comparison(self):
        heuristic = CompositePolicySpec('H')
        report = evaluate_composed(self.spec('A'), heuristic, pimc_samples=2, num_deals=1, seed=4,
                                   bootstrap_samples=5)
        self.assertEqual(len(report.games), 2)  # One duplicate group: ABAB and BABA.

    def test_cli_records_pimc_and_rejects_bad_settings(self):
        quiet = (redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()))
        with tempfile.TemporaryDirectory() as directory, quiet[0], quiet[1]:
            output = Path(directory) / 'result.json'
            evaluate_main(['--agent-a-play', 'pimc', '--pimc-samples', '2', '--deals', '1',
                           '--bootstrap-samples', '5', '--output', str(output)])
            data = json.loads(output.read_text())
            self.assertEqual(data['composition']['pimc_samples'], 2)
            evaluate_main(['--deals', '1', '--bootstrap-samples', '5', '--output', str(output)])
            self.assertNotIn('pimc_samples', json.loads(output.read_text())['composition'])
            for argv in (['--agent-a-bid', 'pimc'], ['--agent-a-play', 'pimc', '--pimc-samples', '0']):
                with self.assertRaises(SystemExit):
                    evaluate_main(argv + ['--deals', '1'])


if __name__ == '__main__':
    unittest.main()
