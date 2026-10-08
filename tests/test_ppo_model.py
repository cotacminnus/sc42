import importlib.util
import math
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from research.ppo.model import ActorCritic
    from research.ppo.trainer import clipped_objective
    from texas42.actions import NUM_ACTIONS
    from texas42.encoding import FEATURE_SIZE


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class ModelTests(unittest.TestCase):
    def test_shapes_gradients_and_mask(self):
        torch.set_num_threads(1)
        model = ActorCritic(16)
        x = torch.zeros(8, FEATURE_SIZE)
        mask = torch.zeros(8, NUM_ACTIONS, dtype=torch.bool)
        mask[:, [0, 4]] = True
        logits, value = model(x)
        self.assertEqual(logits.shape, (8, NUM_ACTIONS))
        self.assertEqual(value.shape, (8,))
        for _ in range(20):
            actions, logp, _ = model.select(x, mask)
            self.assertTrue(((actions == 0) | (actions == 4)).all())
        lp, entropy, values = model.evaluate(x, mask, actions)
        self.assertTrue(torch.allclose((lp - logp).exp(), torch.ones(8)))
        (lp.mean() + values.square().mean() - entropy.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        greedy = model.select(x, mask, greedy=True)[0]
        self.assertTrue(torch.equal(greedy, model.select(x, mask, greedy=True)[0]))
        with self.assertRaisesRegex(ValueError, 'no legal'):
            model.select(x, torch.zeros_like(mask))
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
        d, _ = model.distribution(x, mask)
        self.assertTrue(torch.allclose(d.entropy(), torch.full((8,), math.log(2))))
        self.assertTrue((d.probs[:, 55:] == 0).all())

    def test_numerical_clipped_objective(self):
        # Positive advantage clips ratio 1.5 to 1.2; negative advantage clips .5 to .8.
        lp = torch.log(torch.tensor([1.5, .5], requires_grad=True))
        loss = clipped_objective(lp, torch.zeros(2), torch.tensor([2., -2.]), .2)
        self.assertAlmostEqual(loss.item(), -.4, places=6)
        loss.backward()
