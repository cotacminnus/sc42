"""Fixed-length feature vectors for neural policies, from the actor's perspective.

Input: an Observation from SelfPlayEnv (or its TrainingView). Output: a list of
FEATURE_SIZE floats. Seats are rotated so 0 = actor, 1 = left opponent,
2 = partner, 3 = right opponent; team values are ordered (actor's team, other).
Only public information and the actor's own hand are encoded.

Dependency-free; to_numpy() imports NumPy lazily for batching. Layout and
scaling are versioned by ENCODER_LAYOUT_VERSION (see docs/encoding.md).
"""
from __future__ import annotations

from collections.abc import Iterable

from .actions import ACTION_REGISTRY, NUM_ACTIONS, PLAY_ACTIONS, TRUMP_ACTIONS
from .features import ENCODER_LAYOUT_VERSION
from .models import DOUBLE_TRUMP, Suit, Trump
from .phases import Phase
from .rules import follows_suit, led_suit
from .training import TrainingBidView, TrainingPlayView, TrainingTrumpView, TrainingView

_BID_SLOTS = ACTION_REGISTRY.phase_slots(Phase.BID)  # 21 calls, pass first.
# Key on type as well as value so None (pass) and future declarations stay distinct.
_BID_SLOT_BY_CALL = {(type(entry.value), entry.value): slot for slot, entry in enumerate(_BID_SLOTS)}
_PHASES = (Phase.BID, Phase.TRUMP, Phase.PLAY)
_SUITS: tuple[Suit, ...] = (*range(7), DOUBLE_TRUMP)  # Possible led suits.

# (name, shape) in vector order. Multi-row fields are flattened row-major.
FEATURE_LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("phase", (3,)),              # bid / trump / play one-hot
    ("hand", (28,)),              # actor's remaining dominoes
    ("played", (4, 28)),          # who played each domino this hand (relative seat)
    ("trick", (3, 28)),           # current trick, by play position 0-2
    ("led_suit", (8,)),           # suit of the current trick: 0-6, doubles
    ("voids", (3, 8)),            # suits each other seat has failed to follow
    ("trump", (9,)),              # TRUMP_ACTIONS order: 0-6, doubles, no trump
    ("bidder", (4,)),             # high/winning bidder (relative seat)
    ("bid", (21,)),               # high/winning call, bid phase-slot order
    ("calls", (4, 21)),           # each seat's auction call, bid phase-slot order
    ("shaker", (4,)),             # relative seat of the shaker
    ("forced", (1,)),             # actor must bid
    ("points", (2,)),             # hand points / 42
    ("marks", (2,)),              # match marks / target marks
    ("tricks_done", (1,)),        # completed tricks / 7
)


def _build_slices() -> tuple[dict[str, slice], int]:
    slices, offset = {}, 0
    for name, shape in FEATURE_LAYOUT:
        length = 1
        for dim in shape:
            length *= dim
        slices[name] = slice(offset, offset + length)
        offset += length
    return slices, offset


FEATURE_SLICES, FEATURE_SIZE = _build_slices()


def relative_seat(seat: int, actor: int) -> int:
    """0 = actor, 1 = next seat clockwise, 2 = partner, 3 = previous seat."""
    return (seat - actor) % 4


def encode(observation) -> list[float]:
    """Encode a SelfPlayEnv Observation; same as encode_view(observation.view)."""
    return encode_view(observation.view)


def encode_view(view: TrainingView) -> list[float]:
    """Return FEATURE_SIZE floats; use FEATURE_SLICES to address named fields."""
    out = [0.0] * FEATURE_SIZE
    actor = view.seat

    def put(name: str, index: int, value: float = 1.0) -> None:
        out[FEATURE_SLICES[name].start + index] = value

    if isinstance(view, TrainingBidView):
        phase = Phase.BID
    elif isinstance(view, TrainingTrumpView):
        phase = Phase.TRUMP
    elif isinstance(view, TrainingPlayView):
        phase = Phase.PLAY
    else:
        raise TypeError(f"Unsupported view type: {type(view).__name__}")
    put("phase", _PHASES.index(phase))

    for domino_id in view.hand:
        put("hand", domino_id)

    # Auction: every seat's call, the shaker (who calls last), and the high bid.
    for seat, call in view.bid_history:
        put("calls", relative_seat(seat, actor) * 21 + _bid_slot(call))
    if phase == Phase.BID:
        shaker, bidder, call = view.shaker, view.current_bidder, view.current_bid
        if view.forced:
            put("forced", 0)
    else:
        # Bidding starts left of the shaker, so the first caller identifies it.
        shaker = (view.bid_history[0][0] - 1) % 4
        bidder = view.contract.bidder if view.contract else None
        call = view.contract.bid if view.contract else None
    put("shaker", relative_seat(shaker, actor))
    if bidder is not None:
        put("bidder", relative_seat(bidder, actor))
    if call is not None:
        put("bid", _bid_slot(call))

    team = actor % 2
    put("marks", 0, view.match_marks[team] / view.target_marks)
    put("marks", 1, view.match_marks[1 - team] / view.target_marks)

    if phase == Phase.PLAY:
        _encode_play(view, put, team)
    return out


def _encode_play(view: TrainingPlayView, put, team: int) -> None:
    actor = view.seat
    put("trump", _trump_slot(view.trump))
    put("points", 0, view.team_points[team] / 42)
    put("points", 1, view.team_points[1 - team] / 42)
    put("tricks_done", 0, len(view.completed_tricks) / 7)
    for position, (seat, domino_id) in enumerate(view.trick):
        put("trick", position * 28 + domino_id)
    if view.trick:
        put("led_suit", _SUITS.index(led_suit(PLAY_ACTIONS[view.trick[0][1]], view.trump)))
    for seat, suit in revealed_voids(view):
        rel = relative_seat(seat, actor)
        if rel:  # The actor's own hand is already encoded exactly.
            put("voids", (rel - 1) * 8 + _SUITS.index(suit))
    for trick in (*view.completed_tricks, view.trick):
        for seat, domino_id in trick:
            put("played", relative_seat(seat, actor) * 28 + domino_id)


def revealed_voids(view: TrainingPlayView) -> set[tuple[int, Suit]]:
    """(absolute seat, suit) pairs proven empty: the seat failed to follow that suit."""
    voids = set()
    for trick in (*view.completed_tricks, view.trick):
        if not trick:
            continue
        suit = led_suit(PLAY_ACTIONS[trick[0][1]], view.trump)
        for seat, domino_id in trick[1:]:
            if not follows_suit(PLAY_ACTIONS[domino_id], suit, view.trump):
                voids.add((seat, suit))
    return voids


def _bid_slot(call) -> int:
    try:
        return _BID_SLOT_BY_CALL[(type(call), call)]
    except KeyError:
        raise ValueError(f"No bid slot for auction call {call!r}") from None


def _trump_slot(trump: Trump) -> int:
    for slot, candidate in enumerate(TRUMP_ACTIONS):
        if type(candidate) is type(trump) and candidate == trump:
            return slot
    raise ValueError(f"Unknown trump {trump!r}")


def action_one_hot(action_id: int) -> list[float]:
    """NUM_ACTIONS-wide one-hot for Q(state, action) networks such as Deep Monte Carlo."""
    ACTION_REGISTRY[action_id]  # Validates the ID.
    out = [0.0] * NUM_ACTIONS
    out[action_id] = 1.0
    return out


def to_numpy(observations: Iterable):
    """Batch observations into (features float32 [B, FEATURE_SIZE], masks bool [B, NUM_ACTIONS])."""
    import numpy as np  # Optional dependency; keep the core package import-free.

    observations = list(observations)
    features = np.array([encode(obs) for obs in observations], dtype=np.float32)
    masks = np.array([obs.action_mask for obs in observations], dtype=bool)
    return features.reshape(len(observations), FEATURE_SIZE), masks.reshape(len(observations), NUM_ACTIONS)


__all__ = [
    "ENCODER_LAYOUT_VERSION", "FEATURE_LAYOUT", "FEATURE_SIZE", "FEATURE_SLICES",
    "action_one_hot", "encode", "encode_view", "relative_seat", "revealed_voids", "to_numpy",
]
