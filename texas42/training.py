"""Immutable policy inputs with stable integer domino IDs."""

from copy import deepcopy
from dataclasses import dataclass

from .actions import (
    ACTION_REGISTRY, ACTION_SPACE_VERSION, NUM_ACTIONS, BID_ACTIONS,
    TRUMP_ACTIONS, PLAY_ACTIONS, legacy_action_id,
)
from .models import Bid, Trump
from .players import BidView, PlayView, TrumpView


@dataclass(frozen=True, slots=True)
class TrainingView:
    """Private actor snapshot; hand entries are permanent IDs, not positions."""

    seat: int
    hand: tuple[int, ...]
    match_marks: tuple[int, int]
    bid_history: tuple[tuple[int, Bid | None], ...]
    target_marks: int


@dataclass(frozen=True, slots=True)
class TrainingBidView(TrainingView):
    shaker: int
    current_bid: Bid | None
    current_bidder: int | None
    forced: bool


@dataclass(frozen=True, slots=True)
class TrainingTrumpView(TrainingView):
    bid: Bid


@dataclass(frozen=True, slots=True)
class TrainingPlayView(TrainingView):
    """Tricks retain chronological (seat, domino_id) pairs for policy recall."""

    legal_plays: tuple[int, ...]
    trick: tuple[tuple[int, int], ...]
    completed_tricks: tuple[tuple[tuple[int, int], ...], ...]
    bidder: int
    bid: Bid
    trump: Trump
    team_points: tuple[int, int]


def training_view(view: BidView | TrumpView | PlayView) -> TrainingView:
    """Encode an existing private view without exposing simulator hands."""
    common = dict(seat=view.seat, hand=tuple(domino.id for domino in view.hand),
                  match_marks=view.match_marks, bid_history=view.bid_history,
                  target_marks=view.target_marks)
    if isinstance(view, BidView):
        return TrainingBidView(**common, shaker=view.shaker, current_bid=view.current_bid,
                               current_bidder=view.current_bidder, forced=view.forced)
    if isinstance(view, TrumpView):
        return TrainingTrumpView(**common, bid=view.bid)
    return TrainingPlayView(
        **common, legal_plays=tuple(domino.id for domino in view.legal_plays),
        trick=tuple((seat, domino.id) for seat, domino in view.trick),
        completed_tricks=tuple(tuple((seat, domino.id) for seat, domino in trick)
                               for trick in view.completed_tricks),
        bidder=view.bidder, bid=view.bid, trump=view.trump, team_points=view.team_points,
    )


def upgrade_selfplay_record(record: dict) -> dict:
    """Copy a schema-2 JSONL record into schema 3 without changing its meaning.

    Old data used phase-local IDs and masks. Convert both together; converting
    only the chosen action would silently mislabel the policy's legal choices.
    This migrates datasets, not network weights or optimizer checkpoints.
    """
    if record.get("schema_version") != 2:
        raise ValueError("Expected a schema-2 self-play record")
    upgraded = deepcopy(record)
    observation = upgraded["observation"]
    phase = observation["phase"]
    tables = {"bid": BID_ACTIONS, "trump": TRUMP_ACTIONS, "play": PLAY_ACTIONS}
    if phase not in tables:
        raise ValueError("Unknown observation phase")
    old_mask = observation["action_mask"]
    old_legal = observation["legal_action_ids"]
    if (len(old_mask) != len(tables[phase]) or any(type(bit) is not bool for bit in old_mask)
            or list(old_legal) != [i for i, bit in enumerate(old_mask) if bit]):
        raise ValueError("Legacy mask and legal action IDs disagree")
    legal = tuple(legacy_action_id(phase, local_id) for local_id in old_legal)
    chosen = legacy_action_id(phase, upgraded["action_id"])
    if chosen not in legal:
        raise ValueError("Chosen action is not legal in the legacy observation")
    observation["legal_action_ids"] = list(legal)
    observation["action_mask"] = list(ACTION_REGISTRY.mask(legal))
    observation["action_space_version"] = ACTION_SPACE_VERSION
    observation["num_actions"] = NUM_ACTIONS
    upgraded["action_id"] = chosen
    upgraded["schema_version"] = 3
    return upgraded
