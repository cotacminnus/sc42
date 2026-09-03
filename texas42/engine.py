from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .models import Bid, Domino, Trump, double_six_set, trump_name
from .players import BidView, PlayView, Player, TrumpView, format_hand
from .rules import is_legal_bid, legal_plays, trick_points, trick_winner


@dataclass(frozen=True)
class GameConfig:
    target_marks: int = 7
    debug: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.target_marks < 1:
            raise ValueError("target_marks must be positive")


@dataclass(frozen=True)
class HandResult:
    bidder: int
    bid: Bid
    trump: Trump
    team_points: tuple[int, int]
    scoring_team: int
    marks_awarded: int


class Game:
    def __init__(
        self,
        players: Sequence[Player],
        config: GameConfig | None = None,
        write: Callable[[str], None] = print,
    ) -> None:
        if len(players) != 4:
            raise ValueError("Texas 42 requires exactly four players")
        self.players = tuple(players)
        self.config = config or GameConfig()
        self.write = write
        self.rng = random.Random(self.config.seed)
        self.shaker = self.rng.randrange(4)
        self.marks = [0, 0]

    @staticmethod
    def team_for(seat: int) -> int:
        return seat % 2

    def play_match(self) -> int:
        self.write("Texas 42 - you and Player 3 versus Players 2 and 4")
        self.write(f"{self.players[self.shaker].name} shakes first.")
        while max(self.marks) < self.config.target_marks:
            self.play_hand()
            self.shaker = (self.shaker + 1) % 4
        winner = 0 if self.marks[0] >= self.config.target_marks else 1
        names = "Players 1 & 3" if winner == 0 else "Players 2 & 4"
        self.write(f"{names} win the match, {self.marks[0]}-{self.marks[1]}!")
        return winner

    def play_hand(self) -> HandResult:
        hands = self._deal()
        self.write(f"\n--- New hand: {self.players[self.shaker].name} shakes ---")
        if self.config.debug:
            self.write("DEBUG - all hands:")
            for seat, hand in enumerate(hands):
                self.write(f"  P{seat + 1} {self.players[seat].name}: {format_hand(hand)}")

        bidder, bid = self._run_bidding(hands)
        trump = self._choose_trump(bidder, bid, hands)
        self.write(
            f"{self.players[bidder].name} wins the bid at {bid} "
            f"and calls {trump_name(trump)}."
        )

        team_points = [0, 0]
        completed: list[tuple[tuple[int, Domino], ...]] = []
        leader = bidder
        for trick_number in range(1, 8):
            trick: list[tuple[int, Domino]] = []
            for offset in range(4):
                seat = (leader + offset) % 4
                legal = legal_plays(hands[seat], trick[0][1] if trick else None, trump)
                view = PlayView(
                    seat=seat,
                    hand=tuple(hands[seat]),
                    legal_plays=legal,
                    trick=tuple(trick),
                    completed_tricks=tuple(completed),
                    bidder=bidder,
                    bid=bid,
                    trump=trump,
                    team_points=tuple(team_points),
                    match_marks=tuple(self.marks),
                )
                domino = self.players[seat].choose_domino(view)
                if domino not in legal:
                    raise ValueError(f"{self.players[seat].name} returned an illegal play: {domino}")
                hands[seat].remove(domino)
                trick.append((seat, domino))
                self.write(f"P{seat + 1} {self.players[seat].name} plays {domino}")

            winner = trick_winner(trick, trump)
            points = trick_points(trick)
            team_points[self.team_for(winner)] += points
            completed.append(tuple(trick))
            leader = winner
            self.write(
                f"Trick {trick_number}: {self.players[winner].name} wins "
                f"{points} point{'s' if points != 1 else ''}."
            )

        bidding_team = self.team_for(bidder)
        made = team_points[bidding_team] >= bid.target_points
        scoring_team = bidding_team if made else 1 - bidding_team
        self.marks[scoring_team] += bid.value_in_marks
        outcome = "made" if made else "was set on"
        self.write(
            f"The bidding team {outcome} {bid}; hand points "
            f"{team_points[0]}-{team_points[1]}."
        )
        self.write(f"Match marks: {self.marks[0]}-{self.marks[1]}")
        return HandResult(
            bidder=bidder,
            bid=bid,
            trump=trump,
            team_points=tuple(team_points),
            scoring_team=scoring_team,
            marks_awarded=bid.value_in_marks,
        )

    def _deal(self) -> list[list[Domino]]:
        deck = list(double_six_set())
        self.rng.shuffle(deck)
        return [deck[seat * 7 : (seat + 1) * 7] for seat in range(4)]

    def _run_bidding(self, hands: list[list[Domino]]) -> tuple[int, Bid]:
        current_bid: Bid | None = None
        current_bidder: int | None = None
        first = (self.shaker + 1) % 4
        for offset in range(4):
            seat = (first + offset) % 4
            forced = offset == 3 and current_bid is None
            view = BidView(
                seat=seat,
                shaker=self.shaker,
                hand=tuple(hands[seat]),
                current_bid=current_bid,
                current_bidder=current_bidder,
                forced=forced,
            )
            candidate = self.players[seat].choose_bid(view)
            if candidate is None and forced:
                candidate = Bid(points=30)
                self.write(f"{self.players[seat].name} is forced to bid 30.")
            elif candidate is None:
                self.write(f"{self.players[seat].name} passes.")
                continue

            if not is_legal_bid(candidate, current_bid):
                raise ValueError(f"{self.players[seat].name} returned an illegal bid: {candidate}")
            current_bid = candidate
            current_bidder = seat
            self.write(f"{self.players[seat].name} bids {candidate}.")

        assert current_bid is not None and current_bidder is not None
        return current_bidder, current_bid

    def _choose_trump(self, bidder: int, bid: Bid, hands: list[list[Domino]]) -> Trump:
        trump = self.players[bidder].choose_trump(
            TrumpView(seat=bidder, hand=tuple(hands[bidder]), bid=bid)
        )
        if trump not in {*range(7), "doubles", None}:
            raise ValueError(f"{self.players[bidder].name} returned invalid trump: {trump}")
        return trump
