"""Model-guided PIMC: fast neural inference, value leaves, marks objective, wiring.

Tiny hidden_size=8 checkpoints (as in test_phase_composition): these tests
check mechanics, determinism and invariants, not search strength.
"""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import contextlib
    import io
    import json
    import random
    import tempfile
    from pathlib import Path
    import torch
    from texas42 import SelfPlayEnv
    from texas42.agents import variant_factory
    from texas42.agents.pimc import PIMCAgent
    from texas42.belief import evaluate_actions, hidden_constraints, sample_world, world_state
    from texas42.encoding import encode
    from research.ppo.checkpoint import save as save_legacy, load_model
    from research.ppo.composition import (CompositeAgentFactory, CompositePolicySpec, PhasePolicySpec,
                                          _pimc_delegate, phase_network, resolve_policy_spec)
    from research.ppo.opponents import CheckpointIdentity, PolicySpec
    from research.ppo.search import ValueLeaf, model_factories, model_policy_spec, trained_phases
    from research.ppo.selfplay_checkpoint import save as save_phase
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
    from research.ppo.trainer import Config, Trainer


def observations(seed, predicate=lambda o: True):
    env, rng, found = SelfPlayEnv(seed=seed), random.Random(seed), []
    obs = env.reset()
    while obs is not None:
        if predicate(obs):
            found.append(obs)
        obs = env.step(rng.choice(obs.legal_action_ids)).observation
    return found


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class ModelGuidedPIMCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu'))
        cls.legacy = root / 'legacy.pt'  # Version 1: trained play only.
        save_legacy(cls.legacy, legacy)
        legacy.close()
        anchor = PolicySpec.checkpoint('anchor', cls.legacy)
        model, _ = load_model(cls.legacy)
        config = SelfPlayConfig(total_hands=2, hidden_size=8, device='cpu', anchor=anchor,
            learn_phases=('bid', 'trump', 'play'), rollout_decisions=32,
            opponent_policies=(anchor, PolicySpec('HHH')), evaluation_names=('anchor', 'HHH'),
            eval_deals=1, bootstrap_samples=5)
        trainer = SelfPlayTrainer(config, model)
        # Explicit metadata fixture representing a trained three-phase checkpoint.
        trainer.model.trained_phases = ('bid', 'trump', 'play')
        cls.joint = root / 'joint.pt'
        save_phase(cls.joint, trainer)
        trainer.close()
        cls.identity = CheckpointIdentity.from_path(cls.joint)
        cls.legacy_identity = CheckpointIdentity.from_path(cls.legacy)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def neural_spec(self):
        return CompositePolicySpec('M', *(PhasePolicySpec('neural', self.identity) for _ in range(3)))

    def test_fast_greedy_matches_select_and_probabilities_are_pure(self):
        greedy = CompositeAgentFactory(self.neural_spec())
        checked = 0
        for obs in observations(1) + observations(2):
            agent = greedy(obs.seat, 0)
            agent.reset(obs.seat)
            network = phase_network(PhasePolicySpec('neural', self.identity), obs.phase)
            with torch.inference_mode():
                reference = int(network.select(torch.tensor([encode(obs)]), torch.tensor([obs.action_mask]),
                                               greedy=True)[0].item())
            action = agent.act(obs)
            self.assertEqual(action, reference)
            probabilities = agent.action_probabilities(obs)
            self.assertEqual(set(probabilities), set(obs.legal_action_ids))
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=9)
            self.assertEqual(max(probabilities, key=probabilities.get), action)
            checked += 1
        self.assertGreater(checked, 50)
        # Querying probabilities never advances a stochastic agent's sampling stream.
        stochastic = CompositeAgentFactory(self.neural_spec(), greedy=False)
        plays = observations(3, lambda o: o.phase == 'play' and len(o.legal_action_ids) > 1)
        probe, reference = stochastic(0, 7), stochastic(0, 7)
        for obs in plays:
            probe.reset(obs.seat)
            reference.reset(obs.seat)
            probe.action_probabilities(obs)
            self.assertEqual(probe.act(obs), reference.act(obs))

    def test_value_leaf_sign_and_phase_rules(self):
        obs = observations(4, lambda o: o.phase == 'play' and len(o.view.completed_tricks) == 2)[0]
        state = world_state(obs, sample_world(hidden_constraints(obs), random.Random(0)))
        leaf = ValueLeaf(self.identity)
        network = phase_network(PhasePolicySpec('neural', self.identity), 'play')
        with torch.inference_mode():
            _, value = network(torch.tensor([encode(obs)], dtype=torch.float32))
        self.assertAlmostEqual(leaf(state, obs.seat), float(value.item()), places=6)
        self.assertAlmostEqual(leaf(state, (obs.seat + 1) % 4), -float(value.item()), places=6)
        self.assertAlmostEqual(leaf(state, (obs.seat + 2) % 4), float(value.item()), places=6)
        bid = observations(4, lambda o: o.phase == 'bid')[0]
        bid_state = world_state(bid, sample_world(hidden_constraints(bid), random.Random(0)))
        with self.assertRaisesRegex(ValueError, 'trained bid network'):
            ValueLeaf(self.legacy_identity)(bid_state, bid.seat)  # Version 1 trains play only.
        self.assertEqual(trained_phases(self.legacy_identity), ('play',))
        spec = model_policy_spec(self.legacy_identity)
        self.assertEqual([s.strategy for _, s in spec.items()], ['heuristic', 'heuristic', 'neural'])

    def test_marks_objective(self):
        obs = observations(5, lambda o: o.phase == 'play' and len(o.view.completed_tricks) == 4)[0]
        hhh = (variant_factory('HHH'),) * 4
        marks = evaluate_actions(obs, rng=random.Random(1), rollout_policies=hhh, objective='marks')
        self.assertEqual(marks.objective, 'marks')
        for estimate in marks.estimates:
            # Full-hand rollouts: the marks objective equals the signed-marks diagnostic.
            self.assertAlmostEqual(estimate.expected_points, estimate.expected_signed_marks)
        points = evaluate_actions(obs, rng=random.Random(1), rollout_policies=hhh)
        self.assertEqual(points.objective, 'points')
        with self.assertRaisesRegex(ValueError, 'leaf_evaluator'):
            evaluate_actions(obs, rng=random.Random(1), rollout_policies=hhh, until='trick', objective='marks')

    def test_model_guided_agent_is_deterministic_and_uses_the_leaf(self):
        calls = []
        leaf = ValueLeaf(self.identity)

        def counting_leaf(state, observer):
            calls.append(observer)
            return leaf(state, observer)

        plays = observations(6, lambda o: o.phase == 'play' and len(o.legal_action_ids) > 1)[:4]
        models = model_factories(self.identity)
        actions = []
        for _ in range(2):
            agent = PIMCAgent(11, samples=3, rollout_policies=models, likelihood_policies=models,
                              until='trick', leaf_evaluator=counting_leaf, objective='marks')
            run = []
            for obs in plays:
                agent.reset(obs.seat)
                action = agent.act(obs)
                self.assertIn(action, obs.legal_action_ids)
                self.assertEqual(agent.last_evaluation.objective, 'marks')
                run.append(action)
            actions.append(run)
        self.assertEqual(actions[0], actions[1])
        self.assertTrue(calls)
        with self.assertRaisesRegex(ValueError, 'leaf_evaluator'):
            PIMCAgent(0, until='trick', objective='marks')

    def test_composition_specs_and_delegates(self):
        PhasePolicySpec('pimc', self.identity).validate('play')
        with self.assertRaisesRegex(ValueError, 'model-guided pimc'):
            PhasePolicySpec('heuristic', self.identity).validate('play')
        with self.assertRaises(ValueError):
            CompositeAgentFactory(CompositePolicySpec('X'), pimc_leaf='oracle')
        plain = _pimc_delegate(PhasePolicySpec('pimc'), 1, True, 2, 'value', 'model', 0.1)
        self.assertEqual((plain.until, plain.objective, plain._likelihood_policies), ('hand', 'points', None))
        guided = _pimc_delegate(PhasePolicySpec('pimc', self.identity), 1, True, 2, 'value', 'model', 0.1)
        self.assertEqual((guided.until, guided.objective), ('trick', 'marks'))
        self.assertIsInstance(guided._leaf_evaluator, ValueLeaf)
        self.assertIsNotNone(guided._likelihood_policies)
        # A default neural checkpoint never turns plain PIMC into model-guided PIMC.
        spec = resolve_policy_spec('A', ('neural', 'pimc', 'pimc'), self.joint)
        self.assertIsNone(spec.trump.checkpoint)
        spec = resolve_policy_spec('A', ('neural', 'pimc', 'pimc'), self.joint, pimc_checkpoint=self.legacy)
        self.assertEqual(spec.playing.checkpoint, self.legacy_identity)
        CompositeAgentFactory(spec, pimc_leaf='value').prepare()

    def test_cli_model_guided_and_validation(self):
        from examples.evaluate_agent_vs_agent import main
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            output = Path(directory) / 'result.json'
            main(['--agent-a-play', 'pimc', '--agent-a-pimc-model', str(self.joint), '--pimc-leaf', 'value',
                  '--pimc-likelihood', 'model', '--pimc-samples', '2', '--deals', '1',
                  '--bootstrap-samples', '5', '--output', str(output), '--log-dir', str(Path(directory) / 'log')])
            composition = json.loads(output.read_text())['composition']
            self.assertEqual((composition['pimc_leaf'], composition['pimc_likelihood']), ('value', 'model'))
            self.assertEqual(composition['agents']['A']['playing']['checkpoint']['sha256'], self.identity.sha256)
            for argv in (['--agent-a-play', 'pimc', '--pimc-leaf', 'value'],
                         ['--agent-a-pimc-model', str(self.joint)]):
                with self.assertRaises(SystemExit):
                    main(argv + ['--deals', '1', '--log-dir', str(Path(directory) / 'bad')])

    def test_search_accuracy_smoke(self):
        from examples.search_accuracy import main
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / 'accuracy.json'
            rows = main(['--model', str(self.joint), '--games', '2', '--positions-per-game', '1',
                         '--samples', '2', '--output', str(output), '--log-dir', str(Path(directory) / 'log')])
            data = json.loads(output.read_text())
            self.assertTrue(any(Path(directory, 'log').glob('*.csv')))  # Per-position CSV export.
        summary = data['summary']
        self.assertEqual(len(rows), summary['positions'])
        self.assertEqual(set(summary['beliefs']), {'uniform', 'heuristic-likelihood', 'model-likelihood'})
        self.assertEqual(set(summary['values']),
                         {'heuristic-rollouts', 'model-rollouts', 'model-rollouts+value-leaf'})
        self.assertEqual(len(data['rows']), summary['positions'])
        for row in data['rows']:
            for scores in row['beliefs'].values():
                self.assertTrue(0.0 <= scores['p_true'] <= 1.0)


if __name__ == '__main__':
    unittest.main()
