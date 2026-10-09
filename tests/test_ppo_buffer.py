import importlib.util
import unittest
TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    from dataclasses import replace
    from research.ppo.buffer import Experience, RolloutBuffer, gae


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class BufferTests(unittest.TestCase):
    def test_exact_gae_and_terminal_bootstrap(self):
        adv, ret = gae([0., 2.], [.5, 1.], 1., .95)
        self.assertEqual(adv, [1.45, 1.])
        self.assertEqual(ret, [1.95, 2.])
        adv, ret = gae([0., 2.], [.5, 1.], .5, 1.)
        self.assertEqual(ret, [1., 2.])
        self.assertEqual(gae([0, 0, -2], [0, 0, 0], 1, 1)[1], [-2, -2, -2])

    def test_trajectory_boundaries_and_clear(self):
        b = RolloutBuffer()
        r = Experience((0.,), (True,), 0, 0., 0., 0., False, 0, 0)
        b.add_trajectory([r, replace(r, reward=2, done=True)], 1, 1)
        b.add_trajectory([replace(r, seat=2, reward=-1, done=True)], 1, 1)
        b.add_trajectory([replace(r, episode_id=1, reward=3, done=True)], 1, 1)
        self.assertEqual(b.returns, [2, 2, -1, 3])
        with self.assertRaises(ValueError): b.add_trajectory([r])
        with self.assertRaises(ValueError): b.add_trajectory([r, replace(r, seat=1, done=True)])
        with self.assertRaises(ValueError): b.add_trajectory([replace(r, done=True)])
        b.clear()
        self.assertEqual(len(b), 0)
