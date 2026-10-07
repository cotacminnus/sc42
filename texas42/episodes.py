"""Dependency-free episode progression shared by self-play front ends."""

from dataclasses import dataclass
from typing import Literal

from .models import Domino
from .state import HandResult, HandState


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    """Per-action seat rewards; a hand boundary need not end a match episode."""

    rewards: tuple[int, int, int, int]
    terminated: bool
    hand_result: HandResult | None = None
    match_winner: int | None = None


class EpisodeState:
    """Accept externally dealt hands; apply shared global engine action IDs."""

    def __init__(self, episode: Literal["hand", "match"], target_marks: int,
                 shaker: int) -> None:
        if episode not in ("hand", "match"):
            raise ValueError("episode must be 'hand' or 'match'")
        if type(target_marks) is not int or target_marks < 1:
            raise ValueError("target_marks must be a positive integer")
        if type(shaker) is not int or shaker not in range(4):
            raise ValueError("shaker must be a seat from 0 to 3")
        self.episode = episode
        self.target_marks = target_marks
        self.shaker = shaker
        self.marks = [0, 0]
        self.hand: HandState | None = None
        self.terminated = False
        self.match_winner: int | None = None

    @property
    def needs_deal(self) -> bool:
        """True only before a hand or between unfinished match hands."""
        return not self.terminated and (self.hand is None or self.hand.result is not None)

    def deal(self, hands: list[list[Domino]]) -> None:
        """Validate a complete double-six deal without consuming an RNG stream."""
        if not self.needs_deal:
            raise RuntimeError("Cannot deal during a hand or after episode termination")
        deck = [domino for hand in hands for domino in hand]
        if (len(hands) != 4 or any(len(hand) != 7 for hand in hands)
                or any(not isinstance(domino, Domino) for domino in deck)
                or len(set(deck)) != 28):
            raise ValueError("A deal must contain four hands of seven unique dominoes")
        self.hand = HandState(hands, self.shaker, tuple(self.marks), self.target_marks)

    def apply(self, action_id: int) -> EpisodeTransition:
        """Advance one decision, committing marks only on the final play."""
        if self.needs_deal or self.terminated:
            raise RuntimeError("Deal a hand before applying an action")
        self.hand.apply(action_id)
        result = self.hand.result
        rewards = (0, 0, 0, 0)
        if result is not None:
            self.marks[result.scoring_team] += result.marks_awarded
            if max(self.marks) >= self.target_marks:
                self.match_winner = result.scoring_team
            self.terminated = self.episode == "hand" or self.match_winner is not None
            # Hand returns preserve bid stakes. Match returns measure the final
            # winner, so intermediate hand awards must not pay policy rewards.
            if self.episode == "hand":
                rewards = tuple(result.marks_awarded if seat % 2 == result.scoring_team
                                else -result.marks_awarded for seat in range(4))
            elif self.match_winner is not None:
                rewards = tuple(1 if seat % 2 == self.match_winner else -1 for seat in range(4))
            if not self.terminated:
                # Keep the completed hand available until the caller supplies
                # the next deal (OpenSpiel must expose its chance nodes first).
                self.shaker = (self.shaker + 1) % 4
        return EpisodeTransition(rewards, self.terminated, result, self.match_winner)
