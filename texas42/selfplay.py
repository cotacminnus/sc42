"""Dependency-free, turn-based self-play API (not a Gymnasium adapter)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .models import double_six_set
from .state import DecisionView, HandResult, HandState, Phase


@dataclass(frozen=True)
class Observation:
    """An immutable snapshot for the acting seat, with no other players' hands."""

    seat: int
    phase: Phase
    view: DecisionView
    legal_action_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]


@dataclass(frozen=True)
class StepResult:
    """Rewards are for seats 0, 1, 2, 3, and apply only to this transition.

    observation belongs to the NEXT acting player. A hand_result marks a hand
    boundary; in match mode it need not mean the episode has terminated.
    """

    observation: Observation | None
    rewards: tuple[int, int, int, int]
    terminated: bool
    hand_result: HandResult | None = None
    match_winner: int | None = None


class SelfPlayEnv:
    """Explicit reset/step lifecycle, with a single hand or match per episode.

    Hand episodes pay signed marks at hand end. Match episodes pay signed one
    only at match end. Each partnership receives identical rewards. reset(seed=N)
    restarts the random stream; reset() continues it. No players or I/O required.
    """

    def __init__(self, *, episode: Literal["hand", "match"] = "hand",
                 target_marks: int = 7, seed: int | None = None) -> None:
        if episode not in ("hand", "match"):
            raise ValueError("episode must be 'hand' or 'match'")
        if type(target_marks) is not int or target_marks < 1:
            raise ValueError("target_marks must be a positive integer")
        self.episode = episode
        self.target_marks = target_marks
        self._rng = random.Random(seed)
        self._state: HandState | None = None
        self._marks = [0, 0]
        self._shaker = 0
        self._terminated = False

    @property
    def match_marks(self) -> tuple[int, int]:
        return tuple(self._marks)

    @property
    def shaker(self) -> int:
        return self._shaker

    def reset(self, *, seed: int | None = None, shaker: int | None = None) -> Observation:
        """Discard the previous episode and deal; optional shaker is a seat 0-3."""
        if shaker is not None and (type(shaker) is not int or not 0 <= shaker < 4):
            raise ValueError("shaker must be a seat from 0 to 3")
        if seed is not None:
            self._rng.seed(seed)
        self._shaker = self._rng.randrange(4) if shaker is None else shaker
        self._marks = [0, 0]
        self._terminated = False
        self._deal()
        return self.observe()

    def _deal(self) -> None:
        deck = list(double_six_set())
        self._rng.shuffle(deck)
        hands = [deck[seat * 7:(seat + 1) * 7] for seat in range(4)]
        self._state = HandState(hands, self._shaker, tuple(self._marks), self.target_marks)

    def observe(self) -> Observation:
        """Return the current actor's snapshot; unavailable outside an episode."""
        if self._state is None or self._terminated:
            raise RuntimeError("Call reset() before observing or stepping an episode")
        legal = self._state.legal_action_ids()
        return Observation(self._state.seat, self._state.phase, self._state.view(), legal,
                           tuple(i in legal for i in range(len(self._state.actions()))))

    def step(self, action_id: int) -> StepResult:
        """Apply one legal, phase-local ID. Invalid actions leave state unchanged."""
        if self._state is None or self._terminated:
            raise RuntimeError("Call reset() before observing or stepping an episode")
        self._state.apply(action_id)
        result = self._state.result
        rewards = (0, 0, 0, 0)
        winner = None
        if result is not None:
            self._marks[result.scoring_team] += result.marks_awarded
            if max(self._marks) >= self.target_marks:
                winner = result.scoring_team
            self._terminated = self.episode == "hand" or winner is not None
            if self.episode == "hand":
                rewards = tuple(result.marks_awarded if seat % 2 == result.scoring_team
                                else -result.marks_awarded for seat in range(4))
            elif winner is not None:
                rewards = tuple(1 if seat % 2 == winner else -1 for seat in range(4))
            if not self._terminated:
                self._shaker = (self._shaker + 1) % 4
                self._deal()
        return StepResult(None if self._terminated else self.observe(), rewards,
                          self._terminated, result, winner)
