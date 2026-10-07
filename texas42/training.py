"""Immutable policy inputs with stable integer domino IDs.

These views are the structured input to texas42.encoding, which flattens
them into fixed-length feature vectors for neural networks.
"""

from copy import deepcopy
from dataclasses import dataclass, field

from .features import ENCODER_LAYOUT_VERSION, FEATURE_SCHEMA_VERSION
from .actions import (
    ACTION_REGISTRY, ACTION_SPACE_VERSION, NUM_ACTIONS, BID_ACTIONS,
    TRUMP_ACTIONS, PLAY_ACTIONS, legacy_action_id,
)
from .models import AuctionCall, Bid, Contract, Trump
from .phases import Phase
from .rulesets import RULESET_ID, validate_ruleset
from .players import BidView, PlayView, TrumpView


@dataclass(frozen=True, slots=True)
class TrainingView:
    """Private actor snapshot; hand entries are permanent IDs, not positions."""

    seat: int
    hand: tuple[int, ...]
    match_marks: tuple[int, int]
    bid_history: tuple[tuple[int, AuctionCall], ...]
    target_marks: int
    contract: Contract | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class TrainingBidView(TrainingView):
    shaker: int
    current_bid: AuctionCall
    current_bidder: int | None
    forced: bool


@dataclass(frozen=True, slots=True)
class TrainingTrumpView(TrainingView):
    bid: Bid | None


@dataclass(frozen=True, slots=True)
class TrainingPlayView(TrainingView):
    """Tricks retain chronological (seat, domino_id) pairs for policy recall."""

    legal_plays: tuple[int, ...]
    trick: tuple[tuple[int, int], ...]
    completed_tricks: tuple[tuple[tuple[int, int], ...], ...]
    bidder: int
    bid: Bid | None
    trump: Trump
    team_points: tuple[int, int]


def training_view(view: BidView | TrumpView | PlayView) -> TrainingView:
    """Encode an existing private view without exposing simulator hands."""
    common = dict(seat=view.seat, hand=tuple(domino.id for domino in view.hand),
                  match_marks=view.match_marks, bid_history=view.bid_history,
                  target_marks=view.target_marks, contract=view.contract)
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


def training_metadata(ruleset_id: str = RULESET_ID) -> dict:
    """Persist rules identity and neural-head layout with experiments/checkpoints."""
    return {
        "ruleset_id": validate_ruleset(ruleset_id),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "encoder_layout_version": ENCODER_LAYOUT_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "num_actions": NUM_ACTIONS,
        "action_manifest": ACTION_REGISTRY.manifest(),
        "phase_slots": {phase.value: [entry.action_id for entry in ACTION_REGISTRY.phase_slots(phase)]
                        for phase in Phase},
    }


def upgrade_selfplay_record(record: dict, *, source_ruleset_id: str = RULESET_ID) -> dict:
    """Copy schema-2/3/4 base-engine records into schema 5.

    Old data used phase-local IDs and masks. Convert both together; converting
    only the chosen action would silently mislabel the policy's legal choices.
    This migrates datasets, not network weights or optimizer checkpoints.
    """
    schema = record.get("schema_version")
    if schema not in (2, 3, 4):
        raise ValueError("Expected a schema-2, schema-3, or schema-4 self-play record")
    # Old logs did not record a ruleset. Their provenance must be the known base
    # engine; do not guess the semantics of data from other house-rule variants.
    validate_ruleset(source_ruleset_id)
    if record.get("ruleset_id", source_ruleset_id) != source_ruleset_id:
        raise ValueError("Conflicting ruleset identity in source record")
    upgraded = deepcopy(record)
    observation = upgraded["observation"]
    if observation.get("ruleset_id", source_ruleset_id) != source_ruleset_id:
        raise ValueError("Conflicting ruleset identity in source observation")
    for source in (record, observation):
        if source.get("feature_schema_version", FEATURE_SCHEMA_VERSION) != FEATURE_SCHEMA_VERSION:
            raise ValueError("Unknown source feature schema")
    phase = observation["phase"]
    tables = {"bid": BID_ACTIONS, "trump": TRUMP_ACTIONS, "play": PLAY_ACTIONS}
    if phase not in tables:
        raise ValueError("Unknown observation phase")
    old_mask = observation["action_mask"]
    old_legal = observation["legal_action_ids"]
    expected_size = len(tables[phase]) if schema == 2 else NUM_ACTIONS
    if (len(old_mask) != expected_size or any(type(bit) is not bool for bit in old_mask)
            or list(old_legal) != [i for i, bit in enumerate(old_mask) if bit]):
        raise ValueError("Legacy mask and legal action IDs disagree")
    legal = (tuple(legacy_action_id(phase, local_id) for local_id in old_legal)
             if schema == 2 else tuple(old_legal))
    chosen = (legacy_action_id(phase, upgraded["action_id"]) if schema == 2
              else upgraded["action_id"])
    ACTION_REGISTRY[chosen]  # Validate types even when bool compares equal to an ID.
    if upgraded.get("action_key", ACTION_REGISTRY[chosen].key) != ACTION_REGISTRY[chosen].key:
        raise ValueError("Source action key disagrees with its ID")
    if schema in (3, 4):
        if observation.get("action_space_version") != ACTION_SPACE_VERSION:
            raise ValueError("Unknown source action-space version")
        if observation.get("num_actions") != NUM_ACTIONS:
            raise ValueError("Source vocabulary size disagrees with its mask")
    if any(ACTION_REGISTRY[action_id].phase != phase for action_id in legal):
        raise ValueError("Source legal IDs include another phase")
    if chosen not in legal:
        raise ValueError("Chosen action is not legal in the legacy observation")
    observation["legal_action_ids"] = list(legal)
    observation["action_mask"] = list(ACTION_REGISTRY.mask(legal))
    observation["action_space_version"] = ACTION_SPACE_VERSION
    observation["num_actions"] = NUM_ACTIONS
    observation["ruleset_id"] = source_ruleset_id
    view = observation.get("view")
    if view is not None and "contract" not in view:
        view["contract"] = (None if phase == "bid" else {
            "kind": "normal", "bidder": view["seat"] if phase == "trump" else view["bidder"],
            "bid": deepcopy(view["bid"]),
        })
    result = upgraded.get("hand_result")
    if result is not None and "contract" not in result:
        result["contract"] = {"kind": "normal", "bidder": result["bidder"],
                              "bid": deepcopy(result["bid"])}
    upgraded["action_id"] = chosen
    upgraded["action_key"] = ACTION_REGISTRY[chosen].key
    upgraded["ruleset_id"] = source_ruleset_id
    upgraded["schema_version"] = 5
    upgraded["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    observation["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    return upgraded
