"""Policy delegation and reproducible experiments; no strength assumptions."""
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from examples.policy_ablation_benchmark import main
from texas42.actions import ACTION_REGISTRY, PLAY_ACTIONS
from texas42.agents import (Agent, ConservativeRandomAgent, HeuristicAgent,
                            HybridAgent, VARIANTS, variant_factory)
from texas42.arena import compare_agents, generate_schedule, run_game
from texas42.episodes import EpisodeState
from texas42.models import Bid
from texas42.phases import Phase
from texas42.players import BidView, TrumpView, PlayView
from texas42.selfplay import observation_from_view


def observations():
    hand = tuple(PLAY_ACTIONS[i] for i in range(21, 28))
    return [observation_from_view(v) for v in (
        BidView(0, 0, hand, None, None, False),
        TrumpView(0, hand, Bid(points=30)),
        PlayView(0, hand, hand, (), (), 0, Bid(points=30), 6, (0, 0), (0, 0)))]


class AblationTests(unittest.TestCase):
    def test_conservative_bidding_and_reset(self):
        agent = ConservativeRandomAgent(12)
        self.assertIsInstance(agent, Agent)
        bid_obs = observations()[0]
        with self.assertRaises(RuntimeError): agent.act(bid_obs)
        for bad in (-1, 4, True, 1.0):
            with self.assertRaises(ValueError): agent.reset(bad)
        agent.reset(0)
        hand = tuple(PLAY_ACTIONS[i] for i in range(7))
        for current in (None, Bid(points=30), Bid(points=41), Bid(marks=5)):
            obs = observation_from_view(BidView(0, 0, hand, current,
                                               1 if current else None, False))
            self.assertEqual(agent.act(obs), ACTION_REGISTRY.encode(Phase.BID, None))
        forced = observation_from_view(BidView(0, 0, hand, None, None, True,
                                     bid_history=((1, None), (2, None), (3, None))))
        chosen = agent.act(forced)
        self.assertIn(chosen, forced.legal_action_ids)
        self.assertEqual(ACTION_REGISTRY[chosen].value, Bid(points=30))
        agent.reset(1)
        with self.assertRaises(RuntimeError): agent.act(bid_obs)

    def test_conservative_random_stream_and_reset_continuation(self):
        agent = ConservativeRandomAgent(13)
        reference = random.Random(13)
        for _ in range(10):
            agent.reset(0)  # Does not rewind the generator.
            for obs in observations()[1:]:
                action = agent.act(obs)
                self.assertIs(type(action), int)
                self.assertEqual(action, reference.choice(obs.legal_action_ids))

    def test_all_variants_delegate_exactly_with_documented_seeds(self):
        obs_by_phase = observations()
        for name, strategies in VARIANTS.items():
            agent = variant_factory(name)(0, 42)
            self.assertIsInstance(agent, Agent)
            agent.reset(0)
            root = random.Random(42)
            streams = [random.Random(root.getrandbits(128)) for _ in range(3)]
            heuristic = HeuristicAgent()
            heuristic.reset(0)
            for _ in range(12):
                for obs, strategy, stream in zip(obs_by_phase, strategies, streams):
                    expected = (heuristic.act(obs) if strategy == 'heuristic' else
                                ACTION_REGISTRY.encode(Phase.BID, None) if strategy == 'conservative'
                                else stream.choice(obs.legal_action_ids))
                    action = agent.act(obs)
                    self.assertIs(type(action), int)
                    self.assertEqual(action, expected, (name, obs.phase))
                    self.assertIn(action, obs.legal_action_ids)
                agent.reset(0)
        for kwargs in ({'bidding': 'unknown'}, {'trump': 'conservative'}, {'playing': 'conservative'}):
            with self.assertRaises(ValueError): HybridAgent(**kwargs)
        with self.assertRaises(ValueError): variant_factory('XYZ')

    def test_phase_streams_and_instances_are_independent(self):
        hhr = variant_factory('HHR')(0, 7)
        hrr = variant_factory('HRR')(0, 7)
        untouched = variant_factory('HRR')(0, 7)
        for agent in (hhr, hrr, untouched): agent.reset(0)
        bid, trump, play = observations()
        for _ in range(20):
            hrr.act(trump)
            hrr.act(bid)
        self.assertEqual([hhr.act(play) for _ in range(20)],
                         [hrr.act(play) for _ in range(20)])
        # A distinct instance is still at the start of its play stream.
        fresh = variant_factory('HRR')(0, 7)
        fresh.reset(0)
        self.assertEqual([untouched.act(play) for _ in range(20)],
                         [fresh.act(play) for _ in range(20)])

    def test_full_games_privacy_lifecycle_and_direct_engine_replay(self):
        test = self
        for variant in VARIANTS:
            for mode in ('hand', 'match'):
                made, actions = [], []
                class Checked:
                    def __init__(self, seat, seed):
                        self.agent = variant_factory(variant)(seat, seed)
                        self.resets = []
                        self.original = HeuristicAgent()
                    def reset(self, seat):
                        self.resets.append(seat)
                        self.agent.reset(seat)
                        self.original.reset(seat)
                    def act(self, obs):
                        def check(value):
                            if is_dataclass(value):
                                test.assertTrue(type(value).__dataclass_params__.frozen)
                                for field in fields(value):
                                    test.assertNotIn(field.name, ('hands', 'state', 'initial_deal', 'privileged'))
                                    check(getattr(value, field.name))
                            elif isinstance(value, tuple):
                                for item in value: check(item)
                        check(obs)
                        action = self.agent.act(obs)
                        if variant == "HHH":
                            test.assertEqual(action, self.original.act(obs))
                        test.assertIs(type(action), int)
                        test.assertIn(action, obs.legal_action_ids)
                        actions.append((obs, action))
                        return action
                def factory(seat, seed):
                    agent = Checked(seat, seed)
                    made.append(agent)
                    return agent
                schedule = generate_schedule(5, 21)
                snapshot = repr(schedule)
                game = run_game((factory,) * 4, schedule, episode=mode, target_marks=3)
                self.assertEqual([a.resets for a in made], [[s] for s in range(4)])
                self.assertEqual(len({id(a.agent) for a in made}), 4)
                direct = EpisodeState(mode, 3, schedule[0].shaker)
                recorded = iter(actions)
                for deal, hand in zip(schedule, game.hands):
                    direct.deal([[PLAY_ACTIONS[i] for i in h] for h in deal.hands])
                    for _ in range(33):
                        obs, action = next(recorded)
                        self.assertEqual(obs, observation_from_view(direct.hand.view()))
                        transition = direct.apply(action)
                    self.assertEqual(hand.decisions, 33)
                    self.assertEqual(sum(hand.result.team_points), 42)
                    self.assertEqual(hand.result, direct.hand.result)
                self.assertTrue(direct.terminated)
                self.assertEqual(tuple(direct.marks), game.final_marks)
                self.assertEqual(transition.rewards, game.rewards)
                self.assertEqual(repr(schedule), snapshot)
                if mode == 'match': self.assertEqual(game.winner, direct.match_winner)

    def test_experiments_reproducible_swaps_and_symmetry(self):
        for a, b in (('HHH', 'CRR'), ('HHH', 'HHR'), ('HHR', 'HRR'),
                     ('HHH', 'CHH'), ('HHH', 'RRR'), ('HHH', 'HHH')):
            for episode in ('hand', 'match'):
                kwargs = dict(num_deals=2, seed=42, bootstrap_samples=10,
                              bootstrap_seed=9, episode=episode, target_marks=2)
                report = compare_agents(variant_factory(a), variant_factory(b), **kwargs)
                self.assertEqual(report, compare_agents(variant_factory(a), variant_factory(b), **kwargs))
                json.dumps(report.to_dict())
                for left, right in zip(report.games[::2], report.games[1::2]):
                    self.assertEqual(left.schedule_ids, right.schedule_ids)
                    self.assertEqual(left.agent_seeds, right.agent_seeds)
                    self.assertEqual(left.hands[0].shaker, right.hands[0].shaker)
                    self.assertEqual(left.seating, ('A', 'B', 'A', 'B'))
                    self.assertEqual(right.seating, ('B', 'A', 'B', 'A'))
                if a == b:
                    self.assertEqual([m.win_rate for m in report.metrics], [.5, .5])
                    self.assertEqual([m.mean_signed_marks for m in report.metrics], [0, 0])

    def test_cli_export_format_and_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            args = ['--deals', '2', '--bootstrap-samples', '10', '--output', str(output)]
            with redirect_stdout(io.StringIO()) as terminal:
                first = main(args)
                data = output.read_text()
                second = main(args)
            self.assertEqual(first, second)
            self.assertEqual(data, output.read_text())
            saved = json.loads(data)
            self.assertEqual(saved['benchmark']['policy_b'], 'CRR')
            self.assertIn('phase_configurations', saved['benchmark'])
            for key in ('metrics', 'intervals', 'breakdown'): self.assertIn(key, saved)
            for text in ('Runtime', 'decisions/s', 'Resolved contracts', '95% CI', 'auctions won'):
                self.assertIn(text, terminal.getvalue())
        subprocess.run([sys.executable, '-c', '''
import sys
from examples.policy_ablation_benchmark import main
main(['--deals', '1', '--bootstrap-samples', '2'])
assert not {'numpy', 'torch', 'pyspiel', 'open_spiel'} & sys.modules.keys()
'''], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
