"""Validation execution settings leave training and experimental data intact."""
import csv
import io
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch
from examples.policy_ablation_benchmark import main


class ExportTests(unittest.TestCase):
    def test_readable_log_and_csv(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / 'result.v1.json'
            log_directory = Path(directory) / 'dedicated-logs'
            report = main(['--deals', '2', '--bootstrap-samples', '10', '--output', str(output),
                           '--log-dir', str(log_directory)])
            log_base = log_directory / output.name
            self.assertFalse(output.with_suffix('.log').exists())
            self.assertFalse(output.with_suffix('.csv').exists())
            self.assertIn('Contract decision timing', log_base.with_suffix('.log').read_text())
            with log_base.with_suffix('.csv').open() as sink:
                rows = list(csv.DictReader(sink))
            self.assertEqual(len(rows), len(report.games))
            self.assertTrue(all(int(r['decisions']) == 33 for r in rows))
            self.assertTrue((log_directory / 'result.v1-metrics.csv').exists())

    def test_default_directory(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / 'logs' / 'experiments'
            with patch('examples.experiment_output.DEFAULT_LOG_DIRECTORY', root):
                main(['--deals', '1', '--bootstrap-samples', '1'])
            folders = list(root.iterdir())
            self.assertEqual(len(folders), 1)
            self.assertTrue(folders[0].name.startswith('policy_ablation_benchmark-'))
            self.assertTrue((folders[0] / 'policy_ablation_benchmark.log').exists())
            self.assertTrue((folders[0] / 'policy_ablation_benchmark.csv').exists())


try:
    import torch
    import numpy as np
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional RL dependencies unavailable')
class ValidationTests(unittest.TestCase):
    def test_rng_and_rollout_unchanged(self):
        import random
        from research.ppo.trainer import Config, Trainer
        from research.ppo.evaluation import evaluate
        trainer = Trainer(Config(total_hands=4, hidden_size=16, rollout_play_steps=28))
        try:
            state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            sampling = trainer.sampling.get_state().clone()
            expected = evaluate(trainer.model, deals=4, threads=1, bootstrap_samples=10).to_dict()
            self.assertEqual(expected, evaluate(trainer.model, deals=4, threads=2, bootstrap_samples=10).to_dict())
            self.assertEqual(state[0], random.getstate())
            self.assertEqual(state[1][0], np.random.get_state()[0])
            np.testing.assert_array_equal(state[1][1], np.random.get_state()[1])
            self.assertEqual(state[1][2:], np.random.get_state()[2:])
            self.assertTrue(torch.equal(state[2], torch.get_rng_state()))
            self.assertTrue(torch.equal(sampling, trainer.sampling.get_state()))
            first, _ = trainer.collect()
            other = Trainer(trainer.config)
            try:
                second, _ = other.collect()
                self.assertEqual(first.records, second.records)
            finally:
                other.close()
        finally:
            trainer.close()
