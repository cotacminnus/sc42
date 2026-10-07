from __future__ import annotations

import random
import unittest

from texas42.engine import Game, GameConfig
from texas42.models import Bid, Domino, double_six_set
from texas42.players import HumanPlayer, PlayView, RandomPlayer
from texas42.rules import legal_plays, trick_points, trick_winner


class RulesTests(unittest.TestCase):
    def test_models_reject_noninteger_values(self) -> None:
        # Numeric equality is insufficient when values become permanent IDs or
        # index entries in the fixed action vocabulary.
        for high, low in ((1.0, 0), (1, 0.0), (True, 0), (1, False), ("1", 0)):
            with self.subTest(high=high, low=low), self.assertRaises(ValueError):
                Domino(high, low)
        for kwargs in ({"points": 30.0}, {"points": "30"}, {"marks": 1.0},
                       {"marks": True}, {"marks": "1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Bid(**kwargs)

    def test_double_six_set_has_28_unique_dominoes_and_35_counter_points(self) -> None:
        dominoes = double_six_set()
        self.assertEqual(len(dominoes), 28)
        self.assertEqual(len(set(dominoes)), 28)
        self.assertEqual(sum(domino.count_value for domino in dominoes), 35)

    def test_bid_parsing_and_strength(self) -> None:
        self.assertEqual(Bid.parse("42"), Bid(marks=1))
        self.assertEqual(Bid.parse("2m"), Bid(marks=2))
        self.assertGreater(Bid(marks=1).strength, Bid(points=41).strength)

    def test_doubles_are_a_callable_trump_suit(self) -> None:
        plays = [
            (0, Domino.make(5, 4)),
            (1, Domino.make(0, 0)),
            (2, Domino.make(6, 5)),
            (3, Domino.make(4, 4)),
        ]
        self.assertEqual(trick_winner(plays, "doubles"), 3)

    def test_player_must_follow_numbered_suit(self) -> None:
        hand = [Domino.make(6, 4), Domino.make(2, 1), Domino.make(4, 4)]
        legal = legal_plays(hand, Domino.make(4, 1), trump=2)
        self.assertEqual(set(legal), {Domino.make(6, 4), Domino.make(4, 4)})

    def test_trick_points_include_counters_and_one_for_the_trick(self) -> None:
        plays = [
            (0, Domino.make(5, 5)),
            (1, Domino.make(6, 4)),
            (2, Domino.make(3, 2)),
            (3, Domino.make(1, 0)),
        ]
        self.assertEqual(trick_points(plays), 26)


class EngineTests(unittest.TestCase):
    def test_target_marks_requires_a_positive_integer(self) -> None:
        for target in (0, -1, True, 1.5, "7"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                GameConfig(target_marks=target)

    def test_all_ai_match_completes_and_preserves_42_points_per_hand(self) -> None:
        players = [RandomPlayer(f"Bot {seat}", random.Random(seat)) for seat in range(4)]
        output: list[str] = []
        game = Game(players, GameConfig(target_marks=1, seed=42), output.append)
        winner = game.play_match()
        self.assertIn(winner, (0, 1))
        self.assertEqual(max(game.marks), 1)
        point_lines = [line for line in output if line.startswith("The bidding team")]
        self.assertEqual(len(point_lines), 1)

    def test_debug_mode_reveals_all_four_hands(self) -> None:
        players = [RandomPlayer(f"Bot {seat}", random.Random(seat)) for seat in range(4)]
        output: list[str] = []
        game = Game(players, GameConfig(target_marks=1, debug=True, seed=7), output.append)
        game.play_hand()
        self.assertIn("DEBUG - all hands:", output)
        self.assertEqual(sum(line.startswith("  P") for line in output), 4)


class HumanPlayerTests(unittest.TestCase):
    def test_play_numbers_reject_zero_negative_and_out_of_range(self) -> None:
        hand = (Domino.make(0, 0), Domino.make(1, 0))
        view = PlayView(seat=0, hand=hand, legal_plays=hand, trick=(), completed_tricks=(),
                        bidder=0, bid=Bid(points=30), trump=None, team_points=(0, 0),
                        match_marks=(0, 0))
        answers = iter(("0", "-1", "3", "2"))
        output = []
        player = HumanPlayer(read=lambda _: next(answers), write=output.append)
        self.assertEqual(player.choose_domino(view), hand[1])
        self.assertEqual(output.count("Choose one of the displayed domino numbers."), 3)


if __name__ == "__main__":
    unittest.main()
