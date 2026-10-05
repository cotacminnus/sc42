from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from .models import DOUBLE_TRUMP, Bid, Domino, Trump, trump_name
from .rules import is_legal_bid


@dataclass(frozen=True)
class BidView:
    """Bidding snapshot; forced is true when the shaker is last and all have passed."""

    seat: int
    shaker: int
    hand: tuple[Domino, ...]
    current_bid: Bid | None
    current_bidder: int | None
    forced: bool


@dataclass(frozen=True)
class TrumpView:
    """The winning bidder's hand and contract, before any dominoes are played."""

    seat: int
    hand: tuple[Domino, ...]
    bid: Bid


@dataclass(frozen=True)
class PlayView:
    """Decision snapshot with zero-based seats and team totals in seat-parity order.

    Each trick stores (seat, domino) pairs in play order; completed_tricks is
    chronological. legal_plays retains the order of the player's remaining hand.
    """

    seat: int
    hand: tuple[Domino, ...]
    legal_plays: tuple[Domino, ...]
    trick: tuple[tuple[int, Domino], ...]
    completed_tricks: tuple[tuple[tuple[int, Domino], ...], ...]
    bidder: int
    bid: Bid
    trump: Trump
    team_points: tuple[int, int]
    match_marks: tuple[int, int]


class Player(ABC):
    """Extension point for human, heuristic, or model-backed players.

    Views expose the player's own hand and public game state, never an opponent's
    hand. A future AI only needs to implement these three decision methods.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def choose_bid(self, view: BidView) -> Bid | None:
        """Return a legal raise or None; the engine turns a forced pass into 30."""

    @abstractmethod
    def choose_trump(self, view: TrumpView) -> Trump:
        """Choose numbered trump, doubles, or None for Follow Me."""

    @abstractmethod
    def choose_domino(self, view: PlayView) -> Domino:
        """Choose one domino from view.legal_plays."""


class RandomPlayer(Player):
    """Placeholder bot: passes unless forced and otherwise plays legally at random."""

    def __init__(self, name: str, rng: random.Random | None = None) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()

    def choose_bid(self, view: BidView) -> Bid | None:
        return None

    def choose_trump(self, view: TrumpView) -> Trump:
        return self.rng.choice([*range(7), DOUBLE_TRUMP, None])

    def choose_domino(self, view: PlayView) -> Domino:
        return self.rng.choice(view.legal_plays)


class HumanPlayer(Player):
    def __init__(
        self,
        name: str = "You",
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        super().__init__(name)
        self.read = read
        self.write = write

    def choose_bid(self, view: BidView) -> Bid | None:
        while True:
            current = str(view.current_bid) if view.current_bid else "none"
            forced = "; you cannot pass" if view.forced else ""
            raw = self.read(
                f"Your hand: {format_hand(view.hand)}\n"
                f"Current bid: {current}{forced}. Bid 30-41, 42/1m-5m, or pass: "
            ).strip().lower()
            if raw in {"p", "pass"}:
                if view.forced:
                    self.write("The shaker must bid when everyone else passes.")
                    continue
                return None
            try:
                bid = Bid.parse(raw)
            except (ValueError, TypeError) as error:
                self.write(f"Invalid bid: {error}")
                continue
            if not is_legal_bid(bid, view.current_bid):
                self.write("That bid is not a legal increase.")
                continue
            return bid

    def choose_trump(self, view: TrumpView) -> Trump:
        choices: dict[str, Trump] = {
            **{str(number): number for number in range(7)},
            "b": DOUBLE_TRUMP,
            "double": DOUBLE_TRUMP,
            "doubles": DOUBLE_TRUMP,
            "n": None,
            "no": None,
            "none": None,
            "follow": None,
            "followme": None,
        }
        while True:
            raw = self.read(
                f"You won the bid at {view.bid}. Call trump "
                "(0-6, doubles, or no trump): "
            ).strip().lower().replace(" ", "")
            if raw in choices:
                return choices[raw]
            self.write("Enter 0-6, doubles, or no trump.")

    def choose_domino(self, view: PlayView) -> Domino:
        while True:
            self.write(f"Your hand: {format_hand(view.hand, numbered=True)}")
            if view.trick:
                trick = " ".join(f"P{seat + 1}:{domino}" for seat, domino in view.trick)
                self.write(f"Current trick: {trick}")
            legal_indices = [
                index + 1 for index, domino in enumerate(view.hand) if domino in view.legal_plays
            ]
            raw = self.read(f"Play a domino {legal_indices}: ").strip()
            try:
                index = int(raw) - 1
                domino = view.hand[index]
            except (ValueError, IndexError):
                self.write("Choose one of the displayed domino numbers.")
                continue
            if domino not in view.legal_plays:
                self.write("You must follow suit when possible.")
                continue
            return domino


def format_hand(hand: tuple[Domino, ...] | list[Domino], numbered: bool = False) -> str:
    ordered = sorted(hand, reverse=True)
    if numbered:
        # Preserve the passed order so displayed indices match the engine's hand.
        return " ".join(f"{index}:{domino}" for index, domino in enumerate(hand, start=1))
    return " ".join(str(domino) for domino in ordered)


def describe_trump(trump: Trump) -> str:
    return trump_name(trump)

