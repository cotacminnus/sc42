"""Optional critic-free (policy-only) PPO: critic_mode none beside local and centralized."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import contextlib
    import io
    import json
    import tempfile
    from dataclasses import replace
    from pathlib import Path
    import torch
    from examples.train_ppo import main as train_legacy
    from examples.train_selfplay_ppo import main as train
    from research.ppo.buffer import Experience, RolloutBuffer
    from research.ppo.checkpoint import load_model, read, resume as resume_legacy, save as save_legacy
    from research.ppo.multiphase import LearnerPhaseSpec, PHASES
    from research.ppo.opponents import PolicySpec
    from research.ppo.selfplay_checkpoint import optimizer_parameters, resume, save
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
    from research.ppo.trainer import Config, Trainer, clipped_objective
    from texas42.agents import VARIANTS


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def record(seat, value, reward=0.0, done=False, episode=0, action=0):
    mask = tuple(i == action for i in range(58))
    return Experience((1.0,), mask, action, -.5, value, reward, done, seat, episode)


def value_parameters(model):
    return [p for name, p in model.named_parameters() if name.split('.')[-2] == 'value']


def equal_states(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class ReturnTests(unittest.TestCase):
    def test_monte_carlo_returns_with_zero_baseline(self):
        buffer = RolloutBuffer()
        # Stored values (e.g. an untrained value head's output) must never matter.
        buffer.add_trajectory([record(0, 5.0), record(0, -3.0), record(0, 9.0, 2.0, True)],
                              estimator='monte-carlo', baseline=False)
        buffer.add_trajectory([record(1, 7.0), record(1, 1.0, -2.0, True)], estimator='monte-carlo', baseline=False)
        # gamma = 1: every decision of a seat trajectory has that seat's terminal signed marks.
        self.assertEqual(buffer.returns, [2.0, 2.0, 2.0, -2.0, -2.0])
        self.assertEqual(buffer.advantages, buffer.returns)
        self.assertTrue(all(r.old_state_value == 0.0 for r in buffer.records))
        discounted = RolloutBuffer()
        discounted.add_trajectory([record(0, 4.0), record(0, 4.0, 1.0), record(0, 4.0, 2.0, True)], gamma=.5,
                                  estimator='monte-carlo', baseline=False)
        self.assertEqual(discounted.returns, [1.0, 2.0, 2.0])  # G_t = sum gamma^(k-t) r_k
        self.assertEqual(discounted.advantages, discounted.returns)
        for kwargs in (dict(estimator='gae'), dict(estimator='monte-carlo', bootstrap_value=1.0)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                RolloutBuffer().add_trajectory([record(0, 1.0, 1.0, 'bootstrap_value' not in kwargs)],
                                               baseline=False, **kwargs)
        with self.assertRaises(ValueError):  # Seats never mix.
            RolloutBuffer().add_trajectory([record(0, 0.0), record(1, 0.0, 1.0, True)],
                                           estimator='monte-carlo', baseline=False)


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class ConfigurationTests(unittest.TestCase):
    def test_modes_estimators_and_oracle(self):
        self.assertEqual((Config().critic_mode, Config().return_estimator), ('local', 'gae'))
        self.assertEqual(Config(critic_mode='none').return_estimator, 'monte-carlo')
        self.assertEqual(Config(critic_mode='none', return_estimator='monte-carlo').return_estimator, 'monte-carlo')
        self.assertEqual(Config(return_estimator='monte-carlo').critic_mode, 'local')
        for changes in (dict(critic_mode='none', return_estimator='gae'), dict(critic_mode='centralized'),
                        dict(critic_mode='value'), dict(return_estimator='td')):
            with self.subTest(**changes), self.assertRaises(ValueError):
                Config(**changes)
        base = dict(anchor=None, opponent_policies=(PolicySpec('HHH'), PolicySpec('CHH')),
                    opponent_probabilities=(('HHH', '1'),), evaluation_names=('HHH', 'CHH'), primary_opponent='HHH')
        for mode, estimator in (('none', 'monte-carlo'), ('local', 'gae'), ('centralized', 'gae')):
            self.assertEqual(SelfPlayConfig(critic_mode=mode, **base).return_estimator, estimator)
        self.assertEqual(SelfPlayConfig(**base).critic_mode, 'local')
        with self.assertRaisesRegex(ValueError, 'monte-carlo'):
            SelfPlayConfig(critic_mode='none', return_estimator='gae', **base)
        with self.assertRaisesRegex(ValueError, 'unavailable with critic_mode none'):
            SelfPlayConfig(critic_mode='none', oracle_aux_coef=.5, oracle_dataset='x', oracle_dataset_sha256='y', **base)


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class LegacyTrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def trainer(self, **changes):
        config = Config(**dict(dict(total_hands=12, rollout_play_steps=56, hidden_size=8, minibatch_size=16,
                                    ppo_epochs=2, device='cpu', seed=7), **changes))
        trainer = Trainer(config)
        self.addCleanup(trainer.close)
        return trainer

    def test_value_head_excluded_and_loss_policy_only(self):
        t = self.trainer(critic_mode='none', ppo_epochs=1, minibatch_size=1000)
        in_optimizer = {id(p) for group in t.optimizer.param_groups for p in group['params']}
        self.assertFalse(any(p.requires_grad or id(p) in in_optimizer for p in value_parameters(t.model)))
        self.assertTrue(all(id(p) in in_optimizer for n, p in t.model.named_parameters() if not n.startswith('value')))
        buffer, _ = t.collect()
        self.assertEqual(buffer.advantages, buffer.returns)
        self.assertTrue(all(r.old_state_value == 0.0 for r in buffer.records))
        # Reference: one full-batch clipped-PPO step with entropy and no value term.
        reference = Trainer(t.config)
        self.addCleanup(reference.close)
        reference.model.load_state_dict(t.model.state_dict())
        data = buffer.tensors()
        adv = data['advantages']
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        logp, entropy, _ = reference.model.evaluate(data['features'], data['masks'], data['actions'])
        loss = clipped_objective(logp, data['old_log_probs'], adv, .2) - .01 * entropy.mean()
        reference.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.model.parameters(), .5)
        reference.optimizer.step()
        head = [p.detach().clone() for p in value_parameters(t.model)]
        metrics = t.update(buffer)
        for a, b in zip(t.model.parameters(), reference.model.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(head, value_parameters(t.model))))
        self.assertTrue(all(p.grad is None for p in value_parameters(t.model)))
        self.assertIsNone(metrics['value_loss'])
        self.assertNotIn('critic_gradient_norm', metrics)
        for key in ('policy_loss', 'entropy', 'approx_kl', 'clip_fraction', 'gradient_norm'):
            self.assertIsInstance(metrics[key], float)

    def test_default_local_update_is_unchanged(self):
        # The omitted mode is the historical actor-critic: GAE on stored values and
        # one Adam step on policy + value_coef * value - entropy over all parameters.
        t = self.trainer(ppo_epochs=1, minibatch_size=1000)
        self.assertEqual((t.config.critic_mode, t.config.return_estimator), ('local', 'gae'))
        self.assertEqual(sum(len(g['params']) for g in t.optimizer.param_groups), len(list(t.model.parameters())))
        buffer, _ = t.collect()
        self.assertTrue(any(r.old_state_value != 0.0 for r in buffer.records))
        self.assertNotEqual(buffer.advantages, buffer.returns)
        reference = Trainer(t.config)
        self.addCleanup(reference.close)
        reference.model.load_state_dict(t.model.state_dict())
        data = buffer.tensors()
        adv = data['advantages']
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        logp, entropy, values = reference.model.evaluate(data['features'], data['masks'], data['actions'])
        value = (values - data['returns']).square().mean()
        loss = clipped_objective(logp, data['old_log_probs'], adv, .2) + .5 * value - .01 * entropy.mean()
        reference.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.model.parameters(), .5)
        reference.optimizer.step()
        metrics = t.update(buffer)
        for a, b in zip(t.model.parameters(), reference.model.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertAlmostEqual(metrics['value_loss'], value.item(), places=5)

    def test_values_and_value_coef_do_not_matter(self):
        runs = []
        for coefficient, perturb in ((.5, False), (7.0, True)):
            t = self.trainer(critic_mode='none', value_coef=coefficient)
            if perturb:
                with torch.no_grad():
                    t.model.value.bias.fill_(100.0)
            buffer, _ = t.collect()
            records = [(r.action_id, r.old_log_probability) for r in buffer.records]
            advantages = list(buffer.advantages)
            t.update(buffer)
            runs.append((records, advantages, [p.detach().clone() for n, p in t.model.named_parameters()
                                               if not n.startswith('value')]))
        self.assertEqual(runs[0][:2], runs[1][:2])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(runs[0][2], runs[1][2])))

    def test_same_actor_same_policy_across_modes(self):
        # Critic mode changes no actor initialization or action sampling.
        none, local = self.trainer(critic_mode='none'), self.trainer()
        self.assertTrue(equal_states(none.model, local.model))
        a, b = none.collect()[0], local.collect()[0]
        self.assertEqual([(r.action_id, r.old_log_probability, r.legal_action_mask) for r in a.records],
                         [(r.action_id, r.old_log_probability, r.legal_action_mask) for r in b.records])

    def test_checkpoint_resume_and_mode_changes(self):
        t = self.trainer(critic_mode='none')
        t.update(t.collect()[0])
        save_legacy(self.path / 'split.pt', t)
        t.update(t.collect()[0])
        data = read(self.path / 'split.pt')
        self.assertEqual((data['metadata']['critic_mode'], data['metadata']['return_estimator'],
                          data['metadata']['critic_optimized']), ('none', 'monte-carlo', False))
        resumed = self.trainer(critic_mode='none')
        resume_legacy(self.path / 'split.pt', resumed)
        resumed.update(resumed.collect()[0])
        self.assertTrue(equal_states(t.model, resumed.model))
        with self.assertRaisesRegex(ValueError, 'critic_mode'):
            resume_legacy(self.path / 'split.pt', self.trainer())
        # Checkpoints from before critic modes resume only as the local actor-critic.
        local = self.trainer()
        save_legacy(self.path / 'local.pt', local)
        old = torch.load(self.path / 'local.pt', weights_only=True)
        for key in ('critic_mode', 'return_estimator'):
            del old['config'][key]
        torch.save(old, self.path / 'old.pt')
        resume_legacy(self.path / 'old.pt', self.trainer())
        with self.assertRaisesRegex(ValueError, 'critic_mode'):
            resume_legacy(self.path / 'old.pt', self.trainer(critic_mode='none'))
        model, _ = load_model(self.path / 'split.pt')  # Inactive value head kept for compatibility.
        self.assertIn('value.weight', model.state_dict())

    def test_cli(self):
        def run(name, *flags):
            return quiet(train_legacy, ['--total-hands', '4', '--rollout-play-steps', '28', '--hidden-size', '8',
                                        '--minibatch-size', '28', '--eval-deals', '2', '--bootstrap-samples', '5',
                                        '--eval-every-hands', '4', '--device', 'cpu', *flags,
                                        '--output-dir', str(self.path / name), '--log-dir', str(self.path / (name + '-l'))])
        t = run('none', '--critic-mode', 'none')
        self.assertEqual((t.config.critic_mode, t.config.return_estimator), ('none', 'monte-carlo'))
        rows = [json.loads(l) for l in (self.path / 'none/progress.jsonl').read_text().splitlines()]
        self.assertIsNone(rows[-1]['value_loss'])
        self.assertEqual(run('local').config.critic_mode, 'local')
        for flags in (['--critic-mode', 'none', '--return-estimator', 'gae'], ['--critic-mode', 'centralized']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                run('bad-' + flags[-1], *flags)
        latest = str(self.path / 'none/checkpoints/latest.pt')
        resumed = quiet(train_legacy, ['--resume', latest, '--critic-mode', 'none', '--total-hands', '8',
                                       '--rollout-play-steps', '28', '--hidden-size', '8', '--minibatch-size', '28',
                                       '--eval-deals', '2', '--bootstrap-samples', '5', '--eval-every-hands', '4',
                                       '--device', 'cpu', '--output-dir', str(self.path / 'none'),
                                       '--log-dir', str(self.path / 'r-l')])
        self.assertEqual(resumed.hands, 8)


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class SelfPlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.variants = tuple(PolicySpec(n) for n in VARIANTS)

    def config(self, states=('neural-train',) * 3, pool=(('self-current', '1'),), **changes):
        specs = tuple(LearnerPhaseSpec('neural-train', None, 'random') if s == 'neural-train'
                      else LearnerPhaseSpec('heuristic', initialization='none') for s in states)
        c = SelfPlayConfig(total_hands=8, hidden_size=8, device='cpu', seed=42, from_scratch=True,
                           learner_phases=specs, learn_phases=tuple(p for p, s in zip(PHASES, states) if s == 'neural-train'),
                           rollout_decisions=32, ppo_epochs=2, minibatch_size=8, anchor=None,
                           opponent_policies=self.variants, opponent_probabilities=pool,
                           evaluation_names=('HHH', 'CHH'), primary_opponent='HHH', eval_deals=2,
                           bootstrap_samples=5, forgetting_hands=1, snapshot_every_hands=4, snapshot_pool_size=2,
                           critic_mode='none')
        return replace(c, **changes)

    def trainer(self, config, initial=None):
        trainer = SelfPlayTrainer(config, initial)
        self.addCleanup(trainer.close)
        return trainer

    def cli(self, name, *flags, total='8'):
        return quiet(train, ['--total-hands', total, '--rollout-decisions', '32', '--minibatch-size', '16',
                             '--ppo-epochs', '1', '--eval-deals', '2', '--bootstrap-samples', '5',
                             '--forgetting-hands', '1', '--device', 'cpu', '--seed', '42', *flags,
                             '--output-dir', str(self.path / name), '--log-dir', str(self.path / (name + '-logs'))])

    def test_phase_configurations_and_opponents(self):
        play = ('heuristic', 'heuristic', 'neural-train')
        cases = {'play-only': dict(states=play),
                 'bid-only': dict(states=('neural-train', 'heuristic', 'heuristic')),
                 'trump-only': dict(states=('heuristic', 'neural-train', 'heuristic'), max_rollout_hands=16),
                 'joint': dict(), 'mixed': dict(pool=(('HHH', '1'), ('CHH', '1'), ('self-current', '2'))),
                 'snapshots': dict(states=play, pool=(('self-snapshots', '1'), ('HHH', '1')))}
        for name, changes in cases.items():
            with self.subTest(name):
                t = self.trainer(self.config(total_hands=64, **changes))
                self.assertIsNone(t.critic)
                heads = [p.detach().clone() for p in value_parameters(t.model)]
                members = optimizer_parameters(t)
                self.assertTrue(members)
                self.assertFalse([n for n in members if '.value.' in n])
                self.assertEqual(set(members), {n for n, p in t.model.named_parameters()
                                                if n.split('.')[1] in t.config.learn_phases and '.value.' not in n})
                for _ in range(2):
                    buffer, _ = t.collect()
                    if not len(buffer):
                        continue
                    self.assertEqual(buffer.advantages, buffer.returns)
                    metrics = t.update(buffer)
                    self.assertIsNone(metrics['value_loss'])
                    self.assertIsNone(t.value_diagnostics['explained_variance'])
                    if name == 'snapshots':
                        t.publish_snapshot(self.path / 'snapshots')
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(heads, value_parameters(t.model))))
                self.assertTrue(set(t.model.trained_phases) & set(t.config.learn_phases))
        self.assertTrue(list((self.path / 'snapshots').glob('policy-*.pt')))

    def test_actor_initialization_and_rollout_match_other_modes(self):
        t = {mode: self.trainer(self.config(critic_mode=mode)) for mode in ('none', 'local', 'centralized')}
        self.assertTrue(equal_states(t['none'].model, t['local'].model))
        self.assertTrue(equal_states(t['none'].model, t['centralized'].model))
        actions = {mode: [(r.action_id, r.old_log_probability) for r in trainer.collect()[0].records]
                   for mode, trainer in t.items()}
        self.assertEqual(actions['none'], actions['local'])
        self.assertEqual(actions['none'], actions['centralized'])

    def test_frozen_phases_and_checkpoint_initialization(self):
        local = self.trainer(self.config(('neural-train', 'heuristic', 'neural-train'), critic_mode='local'))
        local.update(local.collect()[0])
        save(self.path / 'local.pt', local)
        initial, _ = load_model(self.path / 'local.pt')
        from research.ppo.opponents import CheckpointIdentity
        source = CheckpointIdentity.from_path(self.path / 'local.pt')
        specs = (LearnerPhaseSpec('neural-train', source), LearnerPhaseSpec('heuristic', initialization='none'),
                 LearnerPhaseSpec('neural-frozen', source))
        config = replace(self.config(), from_scratch=False, learner_phases=specs, learn_phases=('bid',),
                         opponent_probabilities=(('HHH', '1'), ('self-current', '1')))
        t = self.trainer(config, initial)
        frozen = {'play': [v.clone() for v in t.model.networks['play'].state_dict().values()]}
        self.assertEqual(optimizer_parameters(t), ['networks.bid.trunk.0.weight', 'networks.bid.trunk.0.bias',
                                                   'networks.bid.trunk.2.weight', 'networks.bid.trunk.2.bias',
                                                   'networks.bid.policy.weight', 'networks.bid.policy.bias'])
        t.update(t.collect()[0])
        for phase, values in frozen.items():
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(values, t.model.networks[phase].state_dict().values())))

    def test_exact_resume(self):
        config = self.config(('neural-train', 'heuristic', 'neural-train'), pool=(('self-snapshots', '1'), ('HHH', '1')))
        continuous = self.trainer(config)
        continuous.update(continuous.collect()[0])
        continuous.publish_snapshot(self.path / 'snapshots')
        save(self.path / 'split.pt', continuous)
        continuous.update(continuous.collect()[0])
        resumed = self.trainer(config)
        resume(self.path / 'split.pt', resumed)
        self.assertEqual(resumed.snapshots, continuous.snapshots)
        resumed.update(resumed.collect()[0])
        self.assertTrue(equal_states(continuous.model, resumed.model))
        self.assertEqual(continuous.optimizer.state_dict()['param_groups'], resumed.optimizer.state_dict()['param_groups'])
        self.assertEqual(continuous.opponent_assignments, resumed.opponent_assignments)
        self.assertEqual((continuous.hands, continuous.optimizer_steps), (resumed.hands, resumed.optimizer_steps))
        m = read(self.path / 'split.pt')['metadata']
        self.assertEqual((m['critic_mode'], m['return_estimator'], m['critic_optimized']), ('none', 'monte-carlo', False))
        for mode in ('local', 'centralized'):
            with self.subTest(mode), self.assertRaisesRegex(ValueError, 'critic_mode'):
                resume(self.path / 'split.pt', self.trainer(replace(config, critic_mode=mode)))

    def test_cli_stages_resume_and_evaluation(self):
        from examples.evaluate_agent_vs_agent import main as versus
        from examples.evaluate_selfplay_ppo import main as evaluate
        joint = ['--learner-bid', 'neural-train', '--learner-trump', 'neural-train', '--learner-play', 'neural-train']
        pool = ['--opponent-pool', 'HHH:0.3,self-current:0.7', '--eval-opponents', 'HHH,CHH', '--selection-opponent', 'HHH']
        t = self.cli('none', '--from-scratch', '--critic-mode', 'none', *pool, '--hidden-size', '8')
        self.assertEqual((t.config.critic_mode, t.config.return_estimator), ('none', 'monte-carlo'))
        self.assertEqual(t.config.learn_phases, PHASES)
        self.assertEqual(self.cli('default', '--from-scratch', *pool, '--hidden-size', '8').config.critic_mode, 'local')
        latest = self.path / 'none/checkpoints/latest.pt'
        for flags in (['--critic-mode', 'local'], ['--return-estimator', 'gae']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.cli('none', '--resume', str(latest), *flags, total='16')
        with self.assertRaises(SystemExit):
            self.cli('oracle', '--from-scratch', '--critic-mode', 'none', '--oracle-aux-coef', '0.5', '--hidden-size', '8')
        resumed = self.cli('none', '--resume', str(latest), total='16')
        self.assertEqual((resumed.hands, resumed.config.critic_mode), (16, 'none'))
        # New stages may change the mode; the optimizer and critic start fresh.
        for name, source, mode in (('to-centralized', latest, 'centralized'),
                                   ('to-none', self.path / 'default/checkpoints/latest.pt', 'none')):
            with self.subTest(name):
                stage = self.cli(name, '--init-checkpoint', str(source), *joint, *pool, '--critic-mode', mode)
                m = read(self.path / name / 'checkpoints/latest.pt')['metadata']
                self.assertEqual((m['critic_mode'], m['initial_critic_mode']),
                                 (mode, 'none' if name == 'to-centralized' else 'local'))
                self.assertEqual(stage.critic_initialization, 'fresh' if mode == 'centralized' else None)
                self.assertEqual(stage.config.return_estimator, 'gae' if mode == 'centralized' else 'monte-carlo')
        best = str(self.path / 'none/checkpoints/best.pt')
        reports = quiet(evaluate, ['--checkpoint', best, '--deals', '2', '--log-dir', str(self.path / 'e-logs')])
        self.assertEqual(set(reports), {'HHH', 'CHH', 'HHR'})
        quiet(versus, ['--agent-a-checkpoint', best, '--agent-a-bid', 'neural', '--agent-a-trump', 'neural',
                       '--agent-a-play', 'neural', '--agent-b-bid', 'heuristic', '--agent-b-trump', 'heuristic',
                       '--agent-b-play', 'heuristic', '--deals', '2', '--log-dir', str(self.path / 'v-logs')])


if __name__ == '__main__':
    unittest.main()
