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
    from dataclasses import replace
    from research.ppo.search import ValueLeaf, model_factories, model_policy_spec, trained_phases, value_heads_trained
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
        # Centralized critic: the privileged critic replaces the local value heads
        # in the value loss, so those heads are never fitted.
        central = SelfPlayTrainer(replace(config, critic_mode='centralized'), load_model(cls.legacy)[0])
        central.model.trained_phases = ('bid', 'trump', 'play')
        cls.central = root / 'central.pt'
        save_phase(cls.central, central)
        central.close()
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

    def test_value_leaf_requires_fitted_local_value_heads(self):
        central = CheckpointIdentity.from_path(self.central)
        self.assertTrue(value_heads_trained(self.identity))
        self.assertTrue(value_heads_trained(self.legacy_identity))  # Predates critic modes: local.
        self.assertFalse(value_heads_trained(central))
        obs = observations(4, lambda o: o.phase == 'play' and len(o.view.completed_tricks) == 2)[0]
        state = world_state(obs, sample_world(hidden_constraints(obs), random.Random(0)))
        with self.assertRaisesRegex(ValueError, 'local value heads'):
            ValueLeaf(central)(state, obs.seat)
        spec = resolve_policy_spec('A', ('heuristic', 'pimc', 'pimc'), pimc_checkpoint=self.central)
        for leaf in ('value', 'value-turn'):
            with self.assertRaisesRegex(ValueError, 'local value heads'):
                CompositeAgentFactory(spec, pimc_leaf=leaf).prepare()
        CompositeAgentFactory(spec).prepare()  # Full rollouts never read the value heads.

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
        # The own-turn leaf: same value head, rollouts stop at the searcher's next decision.
        turn = _pimc_delegate(PhasePolicySpec('pimc', self.identity), 1, True, 2, 'value-turn', 'uniform', 0.1)
        self.assertEqual((turn.until, turn.objective), ('turn', 'marks'))
        self.assertIsInstance(turn._leaf_evaluator, ValueLeaf)
        self.assertIsNone(turn._likelihood_policies)
        full = _pimc_delegate(PhasePolicySpec('pimc', self.identity), 1, True, 2, 'none', 'uniform', 0.1)
        self.assertEqual((full.until, full.objective, full._leaf_evaluator), ('hand', 'points', None))
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
        # Without --rank-check the output is unchanged: no per-move estimates, no ranking block.
        self.assertNotIn('ranking', summary)
        self.assertFalse(any('q' in mode for row in data['rows'] for mode in row['values'].values()))

    def test_ranking_metrics(self):
        from examples.search_accuracy import best_action, ranking
        reference = {3: 1.0, 5: 3.0, 9: 2.0}
        same = ranking({3: 10.0, 5: 30.0, 9: 20.0}, reference)  # Same order, different scale.
        self.assertEqual((same['agree'], same['ordered'], same['top1'], same['regret']), (3, 3, 1.0, 0.0))
        self.assertAlmostEqual(same['spearman'], 1.0)
        reverse = ranking({3: 3.0, 5: 1.0, 9: 2.0}, reference)
        self.assertEqual((reverse['agree'], reverse['ordered'], reverse['top1']), (0, 3, 0.0))
        self.assertAlmostEqual(reverse['spearman'], -1.0)
        self.assertAlmostEqual(reverse['regret'], 2.0)  # Reference loses 3.0 - 1.0 by trusting it.
        # Pairs tied by either search carry no order information; ties go to the lowest action ID.
        tied = ranking({3: 1.0, 5: 1.0, 9: 0.0}, reference)
        self.assertEqual((tied['agree'], tied['ordered']), (1, 2))
        self.assertEqual(best_action({3: 1.0, 5: 1.0, 9: 0.0}), 3)
        self.assertAlmostEqual(tied['regret'], 2.0)
        two = ranking({1: 0.0, 2: 1.0}, {1: 0.5, 2: 0.0})  # Spearman is undefined below 3 moves.
        self.assertIsNone(two['spearman'])
        self.assertAlmostEqual(two['regret'], 0.5)
        constant = ranking({3: 0.0, 5: 0.0, 9: 0.0}, reference)  # A constant ranking has no order.
        self.assertIsNone(constant['spearman'])
        self.assertEqual(constant['ordered'], 0)

    def test_search_accuracy_rank_check_with_pair_guide(self):
        from examples.search_accuracy import RANK_MODES, RANK_PAIRS, main
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / 'rank.json'
            main(['--model', str(self.joint), '--partner-model', str(self.legacy), '--table', 'heuristic',
                  '--games', '2', '--positions-per-game', '1', '--samples', '2', '--rank-check',
                  '--output', str(output), '--log-dir', str(Path(directory) / 'log')])
            data = json.loads(output.read_text())
        self.assertTrue(data['rank_check'])
        self.assertEqual(data['partner_model']['sha256'], self.legacy_identity.sha256)
        summary = data['summary']
        self.assertEqual(set(summary['values']),  # Points modes never enter the marks accuracy table.
                         {'heuristic-rollouts', 'model-rollouts', 'model-rollouts+value-leaf'})
        self.assertEqual(list(summary['ranking']), [name for name, _, _ in RANK_PAIRS])
        for s in summary['ranking'].values():
            self.assertTrue(s['concordance'] is None or 0.0 <= s['concordance'] <= 1.0)
            self.assertTrue(0.0 <= s['top1'] <= 1.0)
            self.assertGreaterEqual(s['regret'], 0.0)
        for row in data['rows']:
            self.assertTrue(set(RANK_MODES) <= set(row['values']))
            moves = {action for action, _ in row['values']['model-rollouts']['q']}
            self.assertIn(row['action_id'], moves)
            for mode in row['values'].values():  # Every mode estimates the same legal moves.
                self.assertEqual({action for action, _ in mode['q']}, moves)

    def test_pair_aware_guide_routes_by_partner_seat(self):
        plays = observations(8, lambda o: o.phase == 'play' and len(o.legal_action_ids) > 1)
        pair = model_factories(self.identity, partner=self.legacy_identity)
        first = CompositeAgentFactory(model_policy_spec(self.identity))
        second = CompositeAgentFactory(model_policy_spec(self.legacy_identity))
        pair_leaf, leaves = ValueLeaf(self.identity, self.legacy_identity), {}
        seats = set()

        def probabilities(factory, obs):
            agent = factory(obs.seat, 0)
            agent.reset(obs.seat)
            return agent.action_probabilities(obs)

        for obs in plays:
            expected = first if obs.seat in (0, 1) else second
            self.assertEqual(probabilities(pair[obs.seat], obs), probabilities(expected, obs))
            state = world_state(obs, sample_world(hidden_constraints(obs), random.Random(0)))
            identity = self.identity if obs.seat in (0, 1) else self.legacy_identity
            single = leaves.setdefault(identity, ValueLeaf(identity))
            self.assertAlmostEqual(pair_leaf(state, obs.seat), single(state, obs.seat), places=6)
            seats.add(obs.seat)
        self.assertEqual(seats, {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()
