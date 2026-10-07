"""Validate external-deal boundaries used by the native and OpenSpiel APIs."""

import unittest

from texas42.episodes import EpisodeState
from texas42.models import double_six_set


class EpisodeTests(unittest.TestCase):
    def test_invalid_deals_leave_episode_unchanged(self):
        deck = list(double_six_set())
        hands = [deck[seat * 7:(seat + 1) * 7] for seat in range(4)]
        invalid = [hands[:3], [hand[:6] for hand in hands], [hands[0]] * 4,
                   [[domino.id for domino in hand] for hand in hands],
                   [[{"id": domino.id} for domino in hand] for hand in hands]]
        session = EpisodeState("match", 7, 0)
        for deal in invalid:
            with self.subTest(deal=deal), self.assertRaises(ValueError):
                session.deal(deal)
            self.assertIsNone(session.hand)
            self.assertEqual(session.marks, [0, 0])
            self.assertTrue(session.needs_deal)
        session.deal(hands)
        old_hand = session.hand
        with self.assertRaises(RuntimeError):
            session.deal(hands)
        self.assertIs(session.hand, old_hand)
        # HandState owns its lists; changing the caller's deal cannot alter play.
        hands[0].clear()
        self.assertEqual(len(session.hand.hands[0]), 7)
