"""Episode-seeded samples, synchronized updates, and explicit checkpoint migration."""
import importlib.util
import unittest
TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    from dataclasses import replace
    from pathlib import Path
    import tempfile
    import torch
    from research.ppo.trainer import Config, Trainer
    from research.ppo.checkpoint import save, resume, load_model
    from research.ppo.evaluation import evaluate
    from research.ppo.rollout import stream_seed


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class ParallelPPOTests(unittest.TestCase):
    def config(self, **changes):
        return replace(Config(total_hands=16, rollout_play_steps=112, hidden_size=16,
                              ppo_epochs=1, minibatch_size=32, device='cpu', sampling_scheme='episode-v1'), **changes)

    def test_samples_and_updates_equal_across_worker_counts(self):
        records = parameters = advantages = metrics = None
        for workers in (1, 2, 4, 0):
            with Trainer(self.config(threads=workers)) as trainer:
                before = [p.clone() for p in trainer.model.parameters()]
                buffer, reward = trainer.collect()
                self.assertEqual(len(buffer), 112)
                self.assertEqual(sorted({r.episode_id for r in buffer.records}), list(range(8)))
                self.assertEqual(len({stream_seed(42, 'parallel-episode-v1', i) for i in range(8)}), 8)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, trainer.model.parameters())))
                for episode in range(8):
                    rows = [r for r in buffer.records if r.episode_id == episode]
                    seats = sorted({r.seat for r in rows})
                    self.assertEqual(len(seats), 2)
                    self.assertEqual(seats[0] % 2, seats[1] % 2)
                    paths = [[r for r in rows if r.seat == seat] for seat in seats]
                    for path in paths:
                        self.assertEqual(len(path), 7)
                        self.assertEqual([r.reward for r in path[:-1]], [0.] * 6)
                        self.assertEqual([r.done for r in path], [False] * 6 + [True])
                        self.assertTrue(all(r.action_id < 28 and r.legal_action_mask[r.action_id] for r in path))
                    self.assertEqual(paths[0][-1].reward, paths[1][-1].reward)
                data = buffer.tensors()
                with torch.no_grad():
                    logp, _, _ = trainer.model.evaluate(data['features'], data['masks'], data['actions'])
                self.assertTrue(torch.allclose(logp, data['old_log_probs'], atol=1e-6))
                if records is None:
                    records, advantages = buffer.records.copy(), buffer.advantages.copy()
                else:
                    self.assertEqual(records, buffer.records)
                    self.assertEqual(advantages, buffer.advantages)
                actual_metrics = trainer.update(buffer)
                actual_parameters = [p.clone() for p in trainer.model.parameters()]
                if metrics is None:
                    metrics, parameters = actual_metrics, actual_parameters
                else:
                    self.assertEqual(metrics, actual_metrics)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters, actual_parameters)))
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actual_parameters)))
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in actual_metrics.values()))
                # Partial final rollout must not exceed the remaining hand budget.
                trainer.hands = 15
                last, _ = trainer.collect()
                self.assertEqual(len(last), 14)
                self.assertEqual({r.episode_id for r in last.records}, {15})
            self.assertIsNone(trainer.pool)

    def test_parallel_resume_with_changed_workers_and_legacy_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'save.pt'
            with Trainer(self.config(threads=2)) as original:
                b, _ = original.collect(); original.update(b); save(path, original)
                expected, _ = original.collect()
            with Trainer(self.config(threads=4)) as continued:
                resume(path, continued)
                actual, _ = continued.collect()
                self.assertEqual(expected.records, actual.records)
            with Trainer(self.config(threads=1, sampling_scheme='legacy-v1')) as legacy:
                b, _ = legacy.collect(); legacy.update(b); save(path, legacy)
                data = torch.load(path, weights_only=True)
                data.pop('execution')
                data['config'].pop('torch_threads'); data['config'].pop('sampling_scheme')
                torch.save(data, path)
                expected, _ = legacy.collect()
            load_model(path)
            with Trainer(self.config(threads=1, sampling_scheme='legacy-v1')) as continued:
                resume(path, continued)
                actual, _ = continued.collect()
                self.assertEqual(expected.records, actual.records)
            with Trainer(self.config(threads=2)) as incompatible:
                with self.assertRaisesRegex(ValueError, 'Sampling scheme mismatch'):
                    resume(path, incompatible)

    def test_frozen_ppo_arena_factory(self):
        with Trainer(self.config()) as trainer:
            torch.set_num_threads(2)
            seq = evaluate(trainer.model, deals=4, bootstrap_samples=10)
            self.assertEqual(torch.get_num_threads(), 2)
            parallel = evaluate(trainer.model, deals=4, bootstrap_samples=10, threads=2)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertEqual(seq.to_dict(), parallel.to_dict())
            torch.set_num_threads(1)
