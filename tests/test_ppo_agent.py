import importlib.util
import unittest
TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    from pathlib import Path
    import tempfile
    import torch
    from texas42.agents import Agent
    from research.ppo.agent import PPOAgent
    from research.ppo.checkpoint import save, load_model
    from research.ppo.trainer import Config, Trainer
    from research.ppo.evaluation import evaluate


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class AgentTests(unittest.TestCase):
    def test_checkpoint_cpu_compatibility_and_arena(self):
        trainer = Trainer(Config(total_hands=1, hidden_size=16, device='cpu'))
        agent = PPOAgent(trainer.model)
        self.assertIsInstance(agent, Agent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'; save(path, trainer)
            model, data = load_model(path)
            self.assertEqual(next(model.parameters()).device.type, 'cpu')
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), trainer.model.parameters())))
            for opponent in ('HHH', 'CHH', 'HHR'):
                report = evaluate(model, opponent=opponent, deals=2, bootstrap_samples=2)
                self.assertEqual(report.games_played, 4)
                self.assertEqual(report, evaluate(model, opponent=opponent, deals=2, bootstrap_samples=2))
                self.assertTrue(all(h.decisions == 33 for g in report.games for h in g.hands))
            for key in ('action_space_version', 'feature_schema_version', 'encoder_layout_version', 'ruleset_id'):
                bad = dict(data, metadata=dict(data['metadata'], **{key: 'incompatible'}))
                torch.save(bad, path)
                with self.assertRaisesRegex(ValueError, key): load_model(path)
