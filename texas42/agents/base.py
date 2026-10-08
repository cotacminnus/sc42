"""Dependency-free decision protocols; policy state belongs to one seat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..selfplay import Observation


@runtime_checkable
class Agent(Protocol):
    """One instance per seat, reset once per episode (not per match hand).

    Instances may share immutable model weights, but never private recurrent,
    search or belief memory. Return an exact Python int in legal_action_ids.
    PIMC/belief policies also use this ordinary imperfect-information interface.
    """

    def reset(self, seat: int) -> None: ...

    def act(self, obs: Observation) -> int: ...


class BidPreference(Protocol):
    """Optional pure diagnostic, queried before act when passing is illegal.

    True means the policy would pass if allowed; False means it would bid.
    None means unknown. Must not change memory, consume RNG, or call act.
    This is separate from Agent and from the engine's forced-auction flag.
    """

    def would_pass_if_allowed(self, obs: Observation) -> bool | None: ...


@dataclass(frozen=True, slots=True)
class PrivilegedState:
    """Remaining hands in absolute-seat order, using permanent domino IDs.

    Public context comes from Observation. This is a snapshot, never a mutable
    engine reference. Only a future explicitly privileged runner supplies it;
    AgentPlayer never constructs or forwards it.
    """

    hands: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        hands = tuple(tuple(hand) for hand in self.hands)
        ids = [i for hand in hands for i in hand]
        if (len(hands) != 4 or any(len(hand) > 7 for hand in hands)
                or any(type(i) is not int or not 0 <= i < 28 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Expected four disjoint remaining hands of domino IDs')
        object.__setattr__(self, 'hands', hands)


@runtime_checkable
class PrivilegedAgent(Protocol):
    """Separate oracle/teacher API; does not extend ordinary Agent.act."""

    def reset(self, seat: int) -> None: ...

    def act_privileged(self, obs: Observation, state: PrivilegedState) -> int: ...
