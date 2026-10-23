"""Privileged critic features: the actor's public vector plus every remaining hand.

Training-only. Actors never receive these features; a centralized critic does.
Seats are relative to the acting seat (0 = actor, 1 = left opponent, 2 = partner,
3 = right opponent), and the value perspective is always the actor's team.

Together with the public "played" field, the four remaining hands determine the
complete initial deal, so the critic sees every input that the (non-recurrent)
seat policies have seen so far. Opponent identity is not encoded: values average
over the training opponent mixture.
"""
from __future__ import annotations

from collections.abc import Sequence

from .encoding import FEATURE_SIZE, encode, relative_seat
from .phases import Phase

# Bump on any change to fields, order or scaling; critic weights depend on it.
PRIVILEGED_LAYOUT_VERSION = 1

PRIVILEGED_LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("public", (FEATURE_SIZE,)),   # texas42.encoding vector of the acting seat
    ("hands", (4, 28)),            # remaining hands by relative seat
    ("leader", (4,)),              # relative leader of the current trick (play only)
    ("next_player", (4,)),         # relative seat to act (the actor in decision states)
    ("perspective", (2,)),         # (actor's team is reference, other team); always [1, 0]
)


def _slices():
    slices, offset = {}, 0
    for name, shape in PRIVILEGED_LAYOUT:
        length = 1
        for dim in shape:
            length *= dim
        slices[name] = slice(offset, offset + length)
        offset += length
    return slices, offset


PRIVILEGED_SLICES, PRIVILEGED_SIZE = _slices()


def encode_privileged(observation, hands: Sequence[Sequence[int]]) -> list[float]:
    """Critic input for `observation`, given all four remaining hands as domino IDs.

    The actor's own hand in `hands` must match the observation, which guards
    against pairing a decision with another deal's privileged state.
    """
    view = observation.view
    actor = view.seat
    if len(hands) != 4 or sorted(hands[actor]) != sorted(view.hand):
        raise ValueError("Privileged hands do not match the acting observation")
    out = encode(observation) + [0.0] * (PRIVILEGED_SIZE - FEATURE_SIZE)

    def put(name: str, index: int) -> None:
        out[PRIVILEGED_SLICES[name].start + index] = 1.0

    for seat, hand in enumerate(hands):
        for domino_id in hand:
            put("hands", relative_seat(seat, actor) * 28 + domino_id)
    if observation.phase == Phase.PLAY:
        leader = view.trick[0][0] if view.trick else actor
        put("leader", relative_seat(leader, actor))
    put("next_player", 0)
    put("perspective", 0)
    return out


__all__ = ["PRIVILEGED_LAYOUT", "PRIVILEGED_LAYOUT_VERSION", "PRIVILEGED_SIZE",
           "PRIVILEGED_SLICES", "encode_privileged"]
