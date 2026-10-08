"""Dependency-free, turn-based self-play API (not a Gymnasium adapter).

Typical loop: obs = env.reset(); encode obs with texas42.encoding; pick an ID
from obs.legal_action_ids; env.step(id) -> StepResult with the next obs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal
from .collection import TrainingCollector
from .features import FEATURE_SCHEMA_VERSION
from .actions import ACTION_REGISTRY, ACTION_SPACE_VERSION, NUM_ACTIONS

from .models import double_six_set
from .episodes import EpisodeState
from .state import HandResult, Phase
from .training import TrainingView, training_view
from .rulesets import RULESET_ID, validate_ruleset


@dataclass(frozen=True, slots=True)
class Observation:
    """An immutable snapshot for the acting seat, with no other players' hands."""

    seat: int
    phase: Phase
    view: TrainingView
    legal_action_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    action_space_version: int = ACTION_SPACE_VERSION
    num_actions: int = NUM_ACTIONS
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    ruleset_id: str = RULESET_ID


@dataclass(frozen=True, slots=True)
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

    An optional runner-owned collector records transitions. Full initial deals
    require its explicit opt-in and never enter Observation or StepResult.
    """

    def __init__(self, *, episode: Literal["hand", "match"] = "hand",
                 target_marks: int = 7, seed: int | None = None,
                 ruleset_id: str = RULESET_ID,
                 collector: TrainingCollector | None = None) -> None:
        self._collector = collector
        self.ruleset_id = validate_ruleset(ruleset_id)
        if episode not in ("hand", "match"):
            raise ValueError("episode must be 'hand' or 'match'")
        if type(target_marks) is not int or target_marks < 1:
            raise ValueError("target_marks must be a positive integer")
        self.episode = episode
        self.target_marks = target_marks
        self._rng = random.Random(seed)
        self._session: EpisodeState | None = None

    @property
    def match_marks(self) -> tuple[int, int]:
        return tuple(self._session.marks) if self._session else (0, 0)

    @property
    def shaker(self) -> int:
        return self._session.shaker if self._session else 0

    def reset(self, *, seed: int | None = None, shaker: int | None = None) -> Observation:
        """Discard the previous episode and deal; optional shaker is a seat 0-3."""
        if shaker is not None and (type(shaker) is not int or not 0 <= shaker < 4):
            raise ValueError("shaker must be a seat from 0 to 3")
        if seed is not None:
            self._rng.seed(seed)
        self._session = EpisodeState(self.episode, self.target_marks,
                                     self._rng.randrange(4) if shaker is None else shaker,
                                     self.ruleset_id)
        if self._collector is not None:
            self._collector.begin_episode()
        self._deal()
        return self.observe()

    def _deal(self) -> None:
        deck = list(double_six_set())
        self._rng.shuffle(deck)
        hands = [deck[seat * 7:(seat + 1) * 7] for seat in range(4)]
        self._session.deal(hands)
        if self._collector is not None:
            self._collector.begin_hand(self.shaker,
                                       hands if self._collector.include_initial_deal else None)

    def observe(self) -> Observation:
        """Return the current actor's snapshot; unavailable outside an episode."""
        if self._session is None or self._session.terminated:
            raise RuntimeError("Call reset() before observing or stepping an episode")
        state = self._session.hand
        legal = state.legal_action_ids()
        return Observation(state.seat, state.phase, training_view(state.view()), legal,
                           ACTION_REGISTRY.mask(legal), ruleset_id=self.ruleset_id)

    def step(self, action_id: int) -> StepResult:
        """Apply one legal global ID. Invalid actions leave state unchanged."""
        if self._session is None or self._session.terminated:
            raise RuntimeError("Call reset() before observing or stepping an episode")
        observation = self.observe() if self._collector is not None else None
        transition = self._session.apply(action_id)
        if self._collector is not None:
            # Record before auto-dealing: a terminal play belongs to the OLD
            # hand, even when step returns the next hand's first observation.
            self._collector.record(observation, action_id, transition, self.match_marks)
        if self._session.needs_deal:
            # Native self-play samples chance automatically. The returned
            # observation is from the next hand, but hand_result is the old one.
            self._deal()
        return StepResult(None if transition.terminated else self.observe(), transition.rewards,
                          transition.terminated, transition.hand_result, transition.match_winner)
