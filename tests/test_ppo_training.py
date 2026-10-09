import importlib.util
import unittest
TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    from dataclasses import replace
    from pathlib import Path
    import tempfile
    from unittest.mock import patch
    import torch
    from research.ppo.trainer import Config, Trainer
    from research.ppo.checkpoint import save, resume
    from texas42.selfplay import SelfPlayEnv
    from texas42.agents import HeuristicAgent


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class TrainingTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(Config(total_hands=4, rollout_play_steps=14, hidden_size=16,
                              ppo_epochs=2, minibatch_size=8, device='cpu'), **kwargs)

    def test_collection_rewards_privacy_and_frozen_heuristic(self):
        trainer = Trainer(self.config(rollout_play_steps=28))
        transitions, decisions = [], []
        original = SelfPlayEnv.step
        def step(env, action):
            obs = env.observe()
            decisions.append(obs)
            self.assertFalse(hasattr(obs.view, 'hands'))
            if obs.phase != 'play':
                h = HeuristicAgent(); h.reset(obs.seat)
                self.assertEqual(action, h.act(obs))
            result = original(env, action)
            if result.terminated: transitions.append(result)
            return result
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        with patch.object(SelfPlayEnv, 'step', step):
            buffer, _ = trainer.collect()
        self.assertEqual(len(buffer), 28)
        self.assertEqual(len(decisions), 66)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items()))
        for episode, result in enumerate(transitions):
            self.assertEqual(sum(result.hand_result.team_points), 42)
            self.assertEqual(result.rewards[0], result.rewards[2])
            self.assertEqual(result.rewards[1], -result.rewards[0])
            team = (episode + trainer.config.seed) % 2
            for seat in range(4):
                rows = [r for r in buffer.records if (r.episode_id, r.seat) == (episode, seat)]
                if seat % 2 != team:
                    self.assertEqual(rows, [])
                    continue
                self.assertEqual(len(rows), 7)
                self.assertEqual([r.reward for r in rows], [0] * 6 + [result.rewards[seat]])
                self.assertEqual([r.done for r in rows], [False] * 6 + [True])
                self.assertTrue(all(r.action_id < 28 for r in rows))
                self.assertTrue(all(r.observation_features[:3] == (0., 0., 1.) for r in rows))

    def test_update_changes_weights_finite_and_clips_gradients(self):
        trainer = Trainer(self.config())
        buffer, _ = trainer.collect()
        before = [p.clone() for p in trainer.model.parameters()]
        metrics = trainer.update(buffer)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters())))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        norm = torch.sqrt(sum(p.grad.square().sum() for p in trainer.model.parameters()))
        self.assertLessEqual(norm.item(), trainer.config.max_grad_norm + 1e-5)
        self.assertEqual(len(buffer), 0)
        self.assertGreater(trainer.optimizer_steps, 0)

    def test_reproducibility_and_exact_resume(self):
        c = self.config()
        first = Trainer(c)
        b, _ = first.collect(); m = first.update(b)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'; save(path, first)
            b, _ = first.collect(); expected_records = b.records.copy()
            expected_metrics = first.update(b)
            resumed = Trainer(c); resume(path, resumed)
            b, _ = resumed.collect()
            self.assertEqual(expected_records, b.records)
            self.assertEqual(expected_metrics, resumed.update(b))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.model.parameters(), resumed.model.parameters())))
        repeat = Trainer(c); b, _ = repeat.collect()
        self.assertEqual(m, repeat.update(b))

    def test_invalid_config(self):
        for kw in ({'gamma': 1.1}, {'gae_lambda': -1}, {'learning_rate': float('nan')},
                   {'minibatch_size': 0}, {'device': 'tpu'}, {'entropy_coef': -1}):
            with self.assertRaises(ValueError): self.config(**kw)

    def test_cli_checkpoints_logs_and_constant_advantages(self):
        import io
        import json
        from contextlib import redirect_stdout
        from examples.train_ppo import main
        from research.ppo.checkpoint import load_model
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(io.StringIO()) as output:
                trainer = main(['--total-hands', '2', '--rollout-play-steps', '14',
                                '--ppo-epochs', '1', '--minibatch-size', '14',
                                '--hidden-size', '16', '--device', 'cpu',
                                '--eval-deals', '2', '--bootstrap-samples', '2',
                                '--output-dir', directory])
            self.assertIn('Evaluation HHH', output.getvalue())
            rows = [json.loads(line) for line in (Path(directory) / 'progress.jsonl').read_text().splitlines()]
            self.assertEqual([r['hands'] for r in rows], [1, 2])
            self.assertIn('evaluation', rows[-1])
            for name in ('best', 'latest'):
                load_model(Path(directory) / 'checkpoints' / f'{name}.pt')
        # Population std safeguard must also work when all advantages coincide.
        fresh = Trainer(self.config())
        buffer, _ = fresh.collect()
        buffer.advantages = [0.] * len(buffer)
        metrics = fresh.update(buffer)
        self.assertEqual(metrics['policy_loss'], 0.)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
