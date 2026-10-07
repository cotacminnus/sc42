from __future__ import annotations

import importlib.util
import random
import unittest

from texas42 import Bid, Contract, ContractKind, SelfPlayEnv, training_metadata
from texas42.encoding import (
    ENCODER_LAYOUT_VERSION, FEATURE_LAYOUT, FEATURE_SIZE, FEATURE_SLICES, action_one_hot,
    encode, encode_view, relative_seat, revealed_voids, to_numpy,
)
from texas42.models import Domino
from texas42.training import TrainingPlayView


def rollout(seed: int):
    """Yield every observation of one random-legal hand."""
    env = SelfPlayEnv(seed=seed)
    rng = random.Random(seed)
    obs = env.reset()
    while obs is not None:
        yield obs
        obs = env.step(rng.choice(obs.legal_action_ids)).observation


def field(vector, name):
    return vector[FEATURE_SLICES[name]]


class EncodingTests(unittest.TestCase):
    def test_layout_is_contiguous_and_versioned(self):
        offset = 0
        for name, _ in FEATURE_LAYOUT:
            self.assertEqual(FEATURE_SLICES[name].start, offset)
            offset = FEATURE_SLICES[name].stop
        self.assertEqual(offset, FEATURE_SIZE)
        self.assertEqual(FEATURE_SIZE, 387)
        self.assertEqual(training_metadata()["encoder_layout_version"], ENCODER_LAYOUT_VERSION)

    def test_every_observation_encodes_consistently(self):
        for seed in range(20):
            for obs in rollout(seed):
                vector = encode(obs)
                self.assertEqual(len(vector), FEATURE_SIZE)
                self.assertTrue(all(0.0 <= value <= 1.0 for value in vector))
                self.assertEqual(sum(field(vector, "phase")), 1)
                self.assertEqual(sum(field(vector, "shaker")), 1)
                hand = field(vector, "hand")
                self.assertEqual({i for i, bit in enumerate(hand) if bit}, set(obs.view.hand))
                self.assertEqual(sum(field(vector, "calls")), len(obs.view.bid_history))
                if obs.phase == "play":
                    played = sum(len(t) for t in obs.view.completed_tricks) + len(obs.view.trick)
                    self.assertEqual(sum(field(vector, "played")), played)
                    self.assertEqual(sum(field(vector, "trump")), 1)
                    self.assertEqual(sum(field(vector, "led_suit")), int(bool(obs.view.trick)))
                    # The actor's own row of "played" never overlaps its hand.
                    own = field(vector, "played")[:28]
                    self.assertFalse(any(a and b for a, b in zip(own, hand)))

    def test_shaker_is_stable_across_phases(self):
        env = SelfPlayEnv(seed=7)
        rng = random.Random(7)
        obs = env.reset(shaker=2)
        while obs is not None:
            expected = relative_seat(2, obs.seat)
            self.assertEqual(field(encode(obs), "shaker")[expected], 1)
            obs = env.step(rng.choice(obs.legal_action_ids)).observation

    def test_seats_and_teams_are_relative_to_the_actor(self):
        self.assertEqual([relative_seat(seat, 3) for seat in range(4)], [1, 2, 3, 0])
        view = self._play_view(seat=1, team_points=(10, 20), match_marks=(3, 5))
        vector = encode_view(view)
        self.assertEqual(field(vector, "points"), [20 / 42, 10 / 42])
        self.assertEqual(field(vector, "marks"), [5 / 7, 3 / 7])
        # Seat 0 is to the actor's right (relative 3) and won the contract.
        self.assertEqual(field(vector, "bidder"), [0, 0, 0, 1])

    def test_voids_come_from_failing_to_follow(self):
        lead = Domino(5, 4).id
        off_suit = Domino(2, 1).id
        view = self._play_view(seat=3, trick=((1, lead), (2, off_suit)))
        self.assertEqual(revealed_voids(view), {(2, 5)})
        voids = field(encode_view(view), "voids")
        # Seat 2 is relative seat 3 to actor 3 -> voids row 2, suit 5.
        self.assertEqual([i for i, bit in enumerate(voids) if bit], [2 * 8 + 5])

    def test_action_one_hot(self):
        self.assertEqual(action_one_hot(28).index(1.0), 28)
        with self.assertRaises(ValueError):
            action_one_hot(999)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy not installed")
    def test_numpy_batch_shapes(self):
        observations = list(rollout(3))
        features, masks = to_numpy(observations)
        self.assertEqual(features.shape, (len(observations), FEATURE_SIZE))
        self.assertEqual(masks.shape, (len(observations), 58))
        self.assertEqual(str(features.dtype), "float32")

    @staticmethod
    def _play_view(*, seat, trick=(), team_points=(0, 0), match_marks=(0, 0)):
        bid = Bid(points=30)
        history = ((1, None), (2, None), (3, None), (0, bid))
        return TrainingPlayView(
            seat=seat, hand=(), match_marks=match_marks, bid_history=history, target_marks=7,
            contract=Contract(ContractKind.NORMAL, 0, bid), legal_plays=(), trick=trick,
            completed_tricks=(), bidder=0, bid=bid, trump=6, team_points=team_points,
        )



if __name__ == "__main__":
    unittest.main()
