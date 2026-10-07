"""Append-only player action vocabulary shared by engine, training, and OpenSpiel.

IDs are part of the persisted data format. Never insert, reorder, or reuse an
entry: add future declarations at the end, even when they belong to bidding.
Chance outcomes are a separate OpenSpiel vocabulary, not player actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from .models import ContractDeclaration, Bid, Domino, Trump, double_six_set
from .phases import Phase

PLAY_OFFSET = 0
BID_OFFSET = 28
TRUMP_OFFSET = 46
ACTION_SPACE_VERSION = 1

# Retain decoded phase tables for callers and old-log migration. Their indices
# are local lookup positions, not the IDs accepted by the engine anymore.
PLAY_ACTIONS = double_six_set()
BID_ACTIONS: tuple[Bid | None, ...] = (
    None, *(Bid(points=points) for points in range(30, 42)),
    *(Bid(marks=marks) for marks in range(1, 6)),
)
TRUMP_ACTIONS: tuple[Trump, ...] = (*range(7), "doubles", None)


# Preserve the public token name and serialized {"name": ...} representation.
ContractAction = ContractDeclaration


@dataclass(frozen=True, slots=True)
class ActionSpec:
    action_id: int
    key: str
    phase: Phase
    value: Domino | Bid | Trump | ContractAction
    implemented: bool = True


@dataclass(frozen=True, slots=True)
class ActionRegistry:
    entries: tuple[ActionSpec, ...]

    def __post_init__(self):
        if type(self.entries) is not tuple:
            raise ValueError("action entries must be an immutable tuple")
        keys = set()
        for index, entry in enumerate(self.entries):
            if type(entry.action_id) is not int or entry.action_id != index:
                raise ValueError("action IDs must be contiguous and match their permanent positions")
            if type(entry.key) is not str or not entry.key or entry.key in keys:
                raise ValueError("action keys must be nonempty and unique")
            if entry.phase not in tuple(Phase):
                raise ValueError("unknown action phase")
            if type(entry.implemented) is not bool:
                raise ValueError("implemented must be a boolean")
            if any(previous.phase == entry.phase and type(previous.value) is type(entry.value)
                   and previous.value == entry.value for previous in self.entries[:index]):
                raise ValueError("decoded actions must be unique within each phase")
            keys.add(entry.key)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, action_id: int) -> ActionSpec:
        if type(action_id) is not int or not 0 <= action_id < len(self):
            raise ValueError(f"Unknown action ID: {action_id!r}")
        return self.entries[action_id]

    def phase_slots(self, phase: Phase) -> tuple[ActionSpec, ...]:
        """Stable neural-head order, including disabled declarations."""
        phase = Phase(phase)
        return tuple(entry for entry in self.entries if entry.phase == phase)

    def implemented_for_phase(self, phase: Phase) -> tuple[ActionSpec, ...]:
        """Exclude declarations whose rules have not been implemented."""
        return tuple(entry for entry in self.phase_slots(phase) if entry.implemented)

    def for_phase(self, phase: Phase) -> tuple[ActionSpec, ...]:
        """Compatibility alias; neural-head sizing must use phase_slots."""
        return self.implemented_for_phase(phase)

    def global_id(self, phase: Phase, slot: int) -> int:
        slots = self.phase_slots(phase)
        if type(slot) is not int or not 0 <= slot < len(slots):
            raise ValueError("Invalid phase-head slot")
        return slots[slot].action_id

    def phase_slot(self, phase: Phase, action_id: int) -> int:
        entry = self[action_id]
        for slot, candidate in enumerate(self.phase_slots(phase)):
            if candidate == entry:
                return slot
        raise ValueError("Action belongs to a different phase")

    def phase_mask(self, phase: Phase, legal_action_ids: tuple[int, ...]) -> tuple[bool, ...]:
        # Capacity and legality differ: reserved logits exist before enabling
        # rules, so enabling a contract cannot move its neural-head slot.
        self.mask(legal_action_ids)
        legal = set(legal_action_ids)
        slots = self.phase_slots(phase)
        if not legal <= {entry.action_id for entry in slots}:
            raise ValueError("Legal actions include another phase")
        return tuple(entry.action_id in legal for entry in slots)

    def encode(self, phase: Phase, value) -> int:
        # Pass and no-trump both decode to None; phase disambiguates them.
        # Exact types also prevent bool/float from becoming numbered trump.
        for entry in self.for_phase(phase):
            if type(value) is type(entry.value) and value == entry.value:
                return entry.action_id
        raise ValueError(f"No implemented {phase} action for {value!r}")

    def mask(self, legal_action_ids: tuple[int, ...]) -> tuple[bool, ...]:
        legal = set(legal_action_ids)
        for action_id in legal:
            if not self[action_id].implemented:
                raise ValueError("An unimplemented action cannot be legal")
        return tuple(action_id in legal for action_id in range(len(self)))

    def appended(self, *entries: ActionSpec) -> ActionRegistry:
        """Create a larger vocabulary without mutating any existing ID.

        Future rule implementations should append to the module's registry and
        update engine eligibility/transitions explicitly. This helper alone does
        not install new rules or change a running environment's tensor shape.
        """
        return ActionRegistry(self.entries + entries)

    def manifest(self) -> list[dict]:
        """Stable names and IDs for dataset/checkpoint compatibility checks."""
        return [{"id": entry.action_id, "key": entry.key, "phase": entry.phase,
                 "implemented": entry.implemented} for entry in self.entries]


_BASE_ACTIONS = (
    *(ActionSpec(domino.id, f"play.{domino.id}", "play", domino) for domino in PLAY_ACTIONS),
    *(ActionSpec(BID_OFFSET + index,
                 "bid.pass" if bid is None else f"bid.points.{bid.points}" if bid.points is not None
                 else f"bid.marks.{bid.marks}", "bid", bid)
      for index, bid in enumerate(BID_ACTIONS)),
    *(ActionSpec(TRUMP_OFFSET + index, f"trump.{index}", "trump", trump)
      for index, trump in enumerate(TRUMP_ACTIONS)),
)
# These IDs are deliberately visible in masks but always false. Reserving a
# declaration does not define the house rules needed to make it playable.
NIL_ACTION_ID = 55
SPLASH_ACTION_ID = 56
PLUNGE_ACTION_ID = 57
ACTION_REGISTRY = ActionRegistry(_BASE_ACTIONS).appended(
    ActionSpec(NIL_ACTION_ID, "contract.nil", "bid", ContractAction("nil"), False),
    ActionSpec(SPLASH_ACTION_ID, "contract.splash", "bid", ContractAction("splash"), False),
    ActionSpec(PLUNGE_ACTION_ID, "contract.plunge", "bid", ContractAction("plunge"), False),
)
ACTIONS = ACTION_REGISTRY.entries
NUM_ACTIONS = len(ACTION_REGISTRY)


def legacy_action_id(phase: Phase, local_id: int) -> int:
    """Convert schema-2 phase-local IDs to the shared global vocabulary."""
    tables = {"bid": BID_ACTIONS, "trump": TRUMP_ACTIONS, "play": PLAY_ACTIONS}
    if phase not in tables or type(local_id) is not int or not 0 <= local_id < len(tables[phase]):
        raise ValueError("Invalid legacy phase/action ID")
    return ACTION_REGISTRY.encode(phase, tables[phase][local_id])
