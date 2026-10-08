"""Strict schema-5 observation loading for offline datasets.

These functions restore policy inputs only. Transition labels and optional
privileged fields remain outside Observation. Older records must first pass
through upgrade_selfplay_record; unknown versions are never guessed.
"""
from dataclasses import fields
import json

from .actions import ACTION_REGISTRY, ACTION_SPACE_VERSION, NUM_ACTIONS
from .features import ENCODER_LAYOUT_VERSION, FEATURE_SCHEMA_VERSION, PUBLIC_HISTORY_VERSION
from .collection import PRIVILEGED_DATA_VERSION
from .history import PublicHistory, public_history
from .models import Bid, Contract, ContractDeclaration
from .phases import Phase
from .rulesets import validate_ruleset
from .selfplay import Observation
from .training import TrainingBidView, TrainingTrumpView, TrainingPlayView, TrainingView


def _version(data, key, expected, *, required=True):
    if not required and key not in data:
        return
    value = data.get(key)
    if type(value) is not int or value != expected:
        raise ValueError(f"Unsupported {key}: {value!r}")


def _seat(value):
    if type(value) is not int or value not in range(4):
        raise ValueError("Seat must be 0-3")
    return value


def _ids(values):
    result = tuple(values)
    if any(type(i) is not int or i not in range(28) for i in result):
        raise ValueError("Domino IDs must be integers 0-27")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate domino IDs")
    return result


def _pair(values):
    result = tuple(values)
    if len(result) != 2 or any(type(i) is not int or i < 0 for i in result):
        raise ValueError("Expected two nonnegative team values")
    return result


def _call(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Expected a serialized auction call")
    if set(value) == {"name"}:
        return ContractDeclaration(**value)
    if set(value) != {"points", "marks"}:
        raise ValueError("Malformed bid")
    return Bid(**value)


def training_view_from_dict(phase: Phase | str, data: dict) -> TrainingView:
    """Restore enums, calls, contracts and all nested tuples without mutation."""
    try:
        phase = Phase(phase)
        cls = {Phase.BID: TrainingBidView, Phase.TRUMP: TrainingTrumpView,
               Phase.PLAY: TrainingPlayView}[phase]
        if set(data) != {f.name for f in fields(cls)}:
            raise ValueError("View fields do not match its phase")
        view = dict(data)
        view["seat"] = _seat(view["seat"])
        view["hand"] = _ids(view["hand"])
        view["match_marks"] = _pair(view["match_marks"])
        view["bid_history"] = tuple((_seat(seat), _call(call))
                                    for seat, call in view["bid_history"])
        if type(view["target_marks"]) is not int or view["target_marks"] < 1:
            raise ValueError("Invalid target marks")
        contract = view["contract"]
        if contract is not None:
            view["contract"] = Contract(**{**contract, "bid": _call(contract["bid"])})
        if phase == Phase.BID:
            view["shaker"] = _seat(view["shaker"])
            view["current_bid"] = _call(view["current_bid"])
            if view["current_bidder"] is not None:
                _seat(view["current_bidder"])
            if type(view["forced"]) is not bool:
                raise ValueError("forced must be boolean")
        else:
            view["bid"] = _call(view["bid"])
            if view["bid"] is not None and not isinstance(view["bid"], Bid):
                raise ValueError("bid must be a normal Bid or None")
        if phase == Phase.PLAY:
            view["legal_plays"] = _ids(view["legal_plays"])
            view["bidder"] = _seat(view["bidder"])
            ACTION_REGISTRY.encode(Phase.TRUMP, view["trump"])
            view["team_points"] = _pair(view["team_points"])
            def trick(items):
                pairs = tuple((_seat(seat), domino) for seat, domino in items)
                _ids(domino for _, domino in pairs)
                return pairs
            view["trick"] = trick(view["trick"])
            view["completed_tricks"] = tuple(trick(t) for t in view["completed_tricks"])
        return cls(**view)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed training view") from error


def observation_from_dict(data: dict) -> Observation:
    """Restore the observation member of a schema-5 record (no hidden labels)."""
    try:
        for key, version in (("action_space_version", ACTION_SPACE_VERSION),
                             ("feature_schema_version", FEATURE_SCHEMA_VERSION),
                             ("num_actions", NUM_ACTIONS)):
            _version(data, key, version)
        _version(data, "encoder_layout_version", ENCODER_LAYOUT_VERSION, required=False)
        ruleset = validate_ruleset(data["ruleset_id"])
        phase = Phase(data["phase"])
        view = training_view_from_dict(phase, data["view"])
        if _seat(data["seat"]) != view.seat:
            raise ValueError("Observation and view seats disagree")
        legal = tuple(data["legal_action_ids"])
        mask = tuple(data["action_mask"])
        if (any(type(bit) is not bool for bit in mask)
                or mask != ACTION_REGISTRY.mask(legal)
                or legal != tuple(i for i, bit in enumerate(mask) if bit)
                or any(ACTION_REGISTRY[i].phase != phase for i in legal)):
            raise ValueError("Mask and legal actions disagree")
        return Observation(view.seat, phase, view, legal, mask, ruleset_id=ruleset)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed observation") from error


def observation_from_record(record: dict) -> Observation:
    """Load a schema-5 transition's pre-action observation and check provenance."""
    try:
        _version(record, "schema_version", 5)
        _version(record, "feature_schema_version", FEATURE_SCHEMA_VERSION)
        _version(record, "encoder_layout_version", ENCODER_LAYOUT_VERSION, required=False)
        _version(record, "action_space_version", ACTION_SPACE_VERSION, required=False)
        _version(record, "public_history_version", PUBLIC_HISTORY_VERSION, required=False)
        if "privileged" in record:
            _version(record["privileged"], "version", PRIVILEGED_DATA_VERSION)
        observation = observation_from_dict(record["observation"])
        if validate_ruleset(record["ruleset_id"]) != observation.ruleset_id:
            raise ValueError("Conflicting ruleset identity")
        if "public_history" in record:
            if PublicHistory.from_dict(record["public_history"]) != public_history(observation.view):
                raise ValueError("Public history disagrees with observation")
        return observation
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed self-play record") from error


def loads_observation(line: str) -> Observation:
    """Decode one schema-5 JSONL line; does not retain the enclosing record."""
    return observation_from_record(json.loads(line))
