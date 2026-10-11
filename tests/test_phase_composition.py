"""Small phase-routing, checkpoint provenance and paired-arena regression fixtures."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from research.ppo.composition import (PhasePolicySpec, CompositePolicySpec, CompositeAgentFactory,
    CheckpointIdentity, comparison_metadata, evaluate_composed, phase_network)
from texas42.agents import VARIANTS, variant_factory, Agent
from texas42.arena import compare_agents
from texas42.selfplay import SelfPlayEnv

TORCH = importlib.util.find_spec('torch') is not None


def variant(name, label='A'):
    return CompositePolicySpec(label, *(PhasePolicySpec(s) for s in VARIANTS[name]))


class CompositionTests(unittest.TestCase):
    def test_validation_and_independent_seats(self):
        for phase in ('bid', 'trump', 'play'):
            with self.assertRaisesRegex(ValueError, 'requires'):
                PhasePolicySpec('neural').validate(phase)
        for phase in ('trump', 'play'):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                PhasePolicySpec('conservative').validate(phase)
        factory = CompositeAgentFactory(variant('RRR'))
        agents = [factory(s, 42 + s) for s in range(4)]
        for s, agent in enumerate(agents):
            agent.reset(s)
            self.assertIsInstance(agent, Agent)
        self.assertEqual(len({id(d) for a in agents for d in a.delegates.values()}), 12)
        env = SelfPlayEnv(seed=8); obs = env.reset()
        with self.assertRaisesRegex(RuntimeError, 'Reset'):
            agents[(obs.seat + 1) % 4].act(obs)

    def test_non_neural_matches_existing_variants(self):
        for a, b in (('HHH', 'HHH'), ('HHH', 'CHH'), ('HHH', 'HHR'), ('RRR', 'HHH')):
            with self.subTest(a=a, b=b):
                expected = compare_agents(variant_factory(a), variant_factory(b), num_deals=3, bootstrap_samples=13)
                actual = evaluate_composed(variant(a), variant(b, 'B'), num_deals=3, bootstrap_samples=13)
                self.assertEqual(actual.to_dict(), expected.to_dict())
                for game in actual.games:
                    for hand in game.hands:
                        self.assertEqual(hand.decisions, 33)
                        self.assertEqual(sum(hand.result.team_points), 42)
                        self.assertIn(hand.contract_decided_at_trick, range(1, 8))

    def test_random_phase_streams_and_privacy(self):
        left = CompositeAgentFactory(variant('RRR'))(0, 123)
        right = CompositeAgentFactory(variant('HRR'))(0, 123)
        for phase in ('trump', 'play'):
            self.assertEqual(left.delegates[phase]._rng.getstate(), right.delegates[phase]._rng.getstate())
        env = SelfPlayEnv(seed=99); obs = env.reset()
        agents = [CompositeAgentFactory(variant('RRR'))(s, s) for s in range(4)]
        for s, agent in enumerate(agents): agent.reset(s)
        while True:
            self.assertFalse(hasattr(obs, 'state'))
            self.assertFalse(hasattr(obs.view, 'hands'))
            action = agents[obs.seat].act(obs)
            self.assertIs(type(action), int)
            self.assertIn(action, obs.legal_action_ids)
            self.assertLess(action, 55)
            step = env.step(action)
            if step.terminated: break
            obs = step.observation

    def test_dependency_free_cli_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, logs = Path(tmp) / 'results.json', Path(tmp) / 'logs'
            # Refuse optional imports, even if extras are present in this interpreter.
            code = '''import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','numpy','pyspiel'): raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from examples.evaluate_agent_vs_agent import main
main(sys.argv[1:])
'''
            result = subprocess.run([sys.executable, '-c', code, '--deals', '2', '--bootstrap-samples', '5',
                '--agent-b-bid', 'conservative', '--output', str(output), '--log-dir', str(logs)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output.read_text())
            self.assertIn('composition', data)
            self.assertIn('breakdown', data)
            self.assertEqual(data['games_played'], 4)
            self.assertTrue((logs / 'results.log').exists())
            self.assertIn('agent_b_bid_strategy', (logs / 'results.csv').read_text())

    def test_non_neural_spawn_and_matches(self):
        for episode in ('hand', 'match'):
            sequential = evaluate_composed(variant('RRR'), variant('HHH', 'B'), num_deals=4,
                bootstrap_samples=17, episode=episode, target_marks=2, threads=1)
            parallel = evaluate_composed(variant('RRR'), variant('HHH', 'B'), num_deals=4,
                bootstrap_samples=17, episode=episode, target_marks=2, threads=4)
            self.assertEqual(sequential.to_dict(), parallel.to_dict())
            for first, second in zip(parallel.games[::2], parallel.games[1::2]):
                self.assertEqual(first.group_id, second.group_id)
                self.assertEqual(first.hands[0].deal_id, second.hands[0].deal_id)
                self.assertEqual(first.hands[0].shaker, second.hands[0].shaker)
                self.assertEqual(first.seating, ('A', 'B', 'A', 'B'))
                self.assertEqual(second.seating, ('B', 'A', 'B', 'A'))


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class NeuralCompositionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from research.ppo.trainer import Config, Trainer
        from research.ppo.checkpoint import save, load_model
        from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
        from research.ppo.selfplay_checkpoint import save as save_phase
        from research.ppo.opponents import PolicySpec
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu'))
        self.addCleanup(legacy.close)
        self.legacy = root / 'legacy.pt'; save(self.legacy, legacy)
        anchor = PolicySpec.checkpoint('anchor', self.legacy)
        model, _ = load_model(self.legacy)
        config = SelfPlayConfig(total_hands=2, hidden_size=8, device='cpu', anchor=anchor,
            learn_phases=('bid', 'trump', 'play'), rollout_decisions=32,
            opponent_policies=(anchor, PolicySpec('HHH')), evaluation_names=('anchor', 'HHH'), eval_deals=1, bootstrap_samples=5)
        trainer = SelfPlayTrainer(config, model); self.addCleanup(trainer.close)
        # Explicit metadata fixture representing a trained three-phase checkpoint.
        trainer.model.trained_phases = ('bid', 'trump', 'play')
        self.joint = root / 'joint.pt'; save_phase(self.joint, trainer)
        self.clone = root / 'clone.pt'
        data = torch.load(self.joint, weights_only=True)
        data['git_dirty'] = not data['git_dirty']  # Different bytes/identity, same network.
        torch.save(data, self.clone)
        self.untrained = root / 'untrained.pt'
        data['metadata']['trained_phases'] = ('play',)
        torch.save(data, self.untrained)
        self.changed = root / 'changed.pt'
        data['metadata']['trained_phases'] = ('bid', 'trump', 'play')
        data['model']['networks.play.policy.bias'] += 1
        torch.save(data, self.changed)

    def neural(self, path):
        return PhasePolicySpec('neural', CheckpointIdentity.from_path(path))

    def test_phase_availability_rejection(self):
        for phase in ('bid', 'trump'):
            with self.assertRaisesRegex(ValueError, f'Neural {phase} unavailable.*legacy.pt.*play'):
                phase_network(self.neural(self.legacy), phase)
            with self.assertRaisesRegex(ValueError, 'trained neural phases: play'):
                phase_network(self.neural(self.untrained), phase)
        self.assertIsNotNone(phase_network(self.neural(self.joint), 'trump'))

    def test_actual_parameters_and_override_isolation(self):
        n, clone, changed = (self.neural(p) for p in (self.joint, self.clone, self.changed))
        a = CompositePolicySpec('A', n, PhasePolicySpec(), n)
        b = CompositePolicySpec('B', playing=clone)
        meta = comparison_metadata(a, b)
        self.assertTrue(meta['neural_parameters_identical']['play'])
        self.assertTrue(meta['only_one_phase_differs'])
        self.assertFalse(meta['shared_complete_checkpoint'])
        different = comparison_metadata(a, replace(b, playing=changed))
        self.assertFalse(different['neural_parameters_identical']['play'])
        self.assertEqual(different['differing_phases'], ['bid', 'play'])
        agent = CompositeAgentFactory(replace(a, playing=changed))(0, 42)
        self.assertIs(agent.delegates['bid'].network, phase_network(n, 'bid'))
        self.assertIs(agent.delegates['play'].network, phase_network(changed, 'play'))
        self.assertIsNot(agent.delegates['bid'].network, agent.delegates['play'].network)

    def test_neural_combinations_full_game_and_forced_bidding(self):
        n = self.neural(self.joint)
        configs = (
            CompositePolicySpec('A', n, PhasePolicySpec(), n),
            CompositePolicySpec('A', playing=self.neural(self.legacy)),
            CompositePolicySpec('A', n, PhasePolicySpec(), self.neural(self.clone)),
            CompositePolicySpec('A', PhasePolicySpec('conservative'), PhasePolicySpec('random'), self.neural(self.legacy)),
            CompositePolicySpec('A', n, n, n))
        for spec in configs:
            factory = CompositeAgentFactory(spec); factory.prepare()
            env = SelfPlayEnv(seed=2); obs = env.reset()
            agents = [factory(s, s + 123) for s in range(4)]
            for s, agent in enumerate(agents): agent.reset(s)
            decisions = 0
            while True:
                action = agents[obs.seat].act(obs)
                self.assertIn(action, obs.legal_action_ids)
                self.assertIs(type(action), int)
                self.assertLess(action, 55)
                self.assertFalse(hasattr(obs.view, 'hands'))
                step = env.step(action); decisions += 1
                if step.terminated: break
                obs = step.observation
            self.assertEqual(decisions, 33)
            self.assertEqual(sum(step.hand_result.team_points), 42)
        env = SelfPlayEnv(seed=8); obs = env.reset()
        for _ in range(3): obs = env.step(28).observation
        agent = CompositeAgentFactory(configs[0])(obs.seat, 42); agent.reset(obs.seat)
        self.assertTrue(obs.view.forced)
        self.assertNotEqual(agent.act(obs), 28)
        self.assertIsNone(agent.would_pass_if_allowed(obs))

    def test_spawn_reproducibility_weights_and_global_rng(self):
        import random
        import numpy as np
        import torch
        n = self.neural(self.joint)
        a = CompositePolicySpec('A', n, PhasePolicySpec('random'), n)
        b = CompositePolicySpec('B', PhasePolicySpec('conservative'), PhasePolicySpec('random'), self.neural(self.legacy))
        before = {phase: {k: v.clone() for k, v in phase_network(n, phase).state_dict().items()}
                  for phase in ('bid', 'play')}
        for stochastic in (False, True):
            python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            threads = torch.get_num_threads()
            sequential = evaluate_composed(a, b, num_deals=4, bootstrap_samples=19, threads=1, stochastic=stochastic)
            parallel = evaluate_composed(a, b, num_deals=4, bootstrap_samples=19, threads=4, stochastic=stochastic)
            self.assertEqual(sequential.to_dict(), parallel.to_dict())
            self.assertEqual(random.getstate(), python)
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu))
            self.assertEqual(torch.get_num_threads(), threads)
        for phase, values in before.items():
            network = phase_network(n, phase)
            self.assertTrue(all(torch.equal(v, network.state_dict()[k]) for k, v in values.items()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in network.parameters()))

    def test_cached_loading_and_phase_random_streams(self):
        from unittest.mock import patch
        from research.ppo.checkpoint import load_model
        from research.ppo.composition import _CHECKPOINTS
        n = self.neural(self.joint)
        _CHECKPOINTS.pop(n.checkpoint, None)
        spec = CompositePolicySpec('A', n, n, n)
        with patch('research.ppo.checkpoint.load_model', wraps=load_model) as loader:
            factory = CompositeAgentFactory(spec, greedy=False)
            factory.prepare()
            agents = [factory(seat, 42) for seat in range(4)]
            for seat, agent in enumerate(agents): agent.reset(seat)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(len({id(a.delegates['play'].network) for a in agents}), 1)
            self.assertEqual(len({id(a.delegates['play'].generator) for a in agents}), 4)
        left = factory(0, 123)
        right = CompositeAgentFactory(replace(spec, trump=PhasePolicySpec('random')), greedy=False)(0, 123)
        import torch
        self.assertTrue(torch.equal(left.delegates['play'].generator.get_state(), right.delegates['play'].generator.get_state()))

    def test_checkpoint_resolution_and_missing_sources(self):
        import argparse
        from examples.evaluate_agent_vs_agent import policy_spec
        values = dict(agent_a_checkpoint=self.joint, agent_a_bid='neural',
            agent_a_trump='heuristic', agent_a_play='neural', agent_a_bid_checkpoint=None,
            agent_a_trump_checkpoint=None, agent_a_play_checkpoint=self.legacy)
        spec = policy_spec(argparse.Namespace(**values), 'a')
        self.assertEqual(spec.bidding.checkpoint.path, str(self.joint))
        self.assertEqual(spec.playing.checkpoint.path, str(self.legacy))
        values['agent_a_bid_checkpoint'] = self.clone
        spec = policy_spec(argparse.Namespace(**values), 'a')
        self.assertEqual(spec.bidding.checkpoint.path, str(self.clone))
        self.assertEqual(spec.playing.checkpoint.path, str(self.legacy))
        values['agent_a_checkpoint'] = values['agent_a_bid_checkpoint'] = None
        with self.assertRaisesRegex(ValueError, 'Neural bid requires'):
            policy_spec(argparse.Namespace(**values), 'a')
        values['agent_a_bid'] = 'heuristic'
        values['agent_a_bid_checkpoint'] = self.joint
        with self.assertRaisesRegex(ValueError, 'override requires neural'):
            policy_spec(argparse.Namespace(**values), 'a')

    def test_cli_same_joint_play_and_checkpoint_identity_exports(self):
        from examples.evaluate_agent_vs_agent import main
        from contextlib import redirect_stdout
        from io import StringIO
        import csv
        root = Path(self.tmp.name); output, logs = root / 'out.json', root / 'logs'
        stream = StringIO()
        with redirect_stdout(stream):
            main(['--agent-a-checkpoint', str(self.joint), '--agent-a-bid', 'neural',
                  '--agent-a-play', 'neural', '--agent-b-checkpoint', str(self.joint),
                  '--agent-b-play', 'neural', '--deals', '2', '--bootstrap-samples', '5',
                  '--output', str(output), '--log-dir', str(logs)])
        self.assertIn('Shared playing parameters: YES', stream.getvalue())
        data = json.loads(output.read_text())
        self.assertTrue(data['composition']['comparisons']['only_one_phase_differs'])
        self.assertTrue(data['composition']['comparisons']['neural_parameters_identical']['play'])
        with (logs / 'out.csv').open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]['agent_a_play_sha256'], rows[0]['agent_b_play_sha256'])
        self.assertEqual(rows[0]['agent_b_play_strategy'], 'neural')


if __name__ == '__main__':
    unittest.main()
