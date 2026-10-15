"""Exact open-hand analysis of ordinary contracts; never a policy input.

All values use the bidding partnership's perspective, including at defending
nodes. Only the engine generates moves, advances tricks, and scores the hand.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from enum import StrEnum
import math
from time import monotonic
from typing import Literal

from .actions import ACTION_REGISTRY
from .arena import contract_decided
from .models import ContractKind, Domino
from .phases import Phase
from .rulesets import validate_ruleset
from .state import HandState


@dataclass(frozen=True, slots=True)
class SearchLimits:
    time_seconds: float | None = None
    max_nodes: int | None = None
    max_cache_entries: int = 100_000

    def __post_init__(self):
        if (self.time_seconds is not None and
                (type(self.time_seconds) not in (int, float)
                 or not math.isfinite(self.time_seconds) or self.time_seconds < 0)):
            raise ValueError('time_seconds must be finite and nonnegative, or None')
        for name in ('max_nodes', 'max_cache_entries'):
            value = getattr(self, name)
            if value is None and name == 'max_nodes':
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')


@dataclass(frozen=True, slots=True)
class ValueBounds:
    lower: int = -1
    upper: int = 1

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    @property
    def value(self) -> int | None:
        return self.lower if self.exact else None


@dataclass(frozen=True, slots=True)
class ActionValue(ValueBounds):
    action_id: int = 0
    # False with [-1, +1] means unknown, not a heuristic evaluation. A score
    # proof at the root can establish an action without visiting its child.
    searched: bool = False


@dataclass(frozen=True, slots=True)
class OracleResult(ValueBounds):
    status: Literal['exact', 'budget_exceeded'] = 'budget_exceeded'
    optimal_action: int | None = None
    optimal_actions: tuple[int, ...] | None = None
    action_values: tuple[ActionValue, ...] = ()
    all_action_values_complete: bool = False
    # No continuation is assembled in this initial implementation. In
    # particular, a transposition-table move would not itself prove a PV.
    principal_variation: tuple[int, ...] | None = None
    nodes: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0


class _Flag(StrEnum):
    EXACT = 'exact'
    LOWER_BOUND = 'lower_bound'
    UPPER_BOUND = 'upper_bound'


@dataclass(frozen=True, slots=True)
class _Entry:
    value: int
    flag: _Flag

    @property
    def bounds(self) -> ValueBounds:
        if self.flag == _Flag.EXACT:
            return ValueBounds(self.value, self.value)
        if self.flag == _Flag.LOWER_BOUND:
            return ValueBounds(self.value, 1)
        return ValueBounds(-1, self.value)


def _state_key(state: HandState) -> tuple:
    """Readable, immutable, absolute-seat key for this hand-only objective.

    Completed trick *count* affects engine termination. Their contents/order
    cannot affect future legality or scoring once points have been awarded.
    Auction history, shaker and match marks likewise cannot affect this objective.
    """
    return ('ordinary_contract_success_v1', state.ruleset_id,
            state.contract.bidder % 2, state.phase, state.contract, state.bidder,
            state.trump, state.seat,
            tuple(tuple(sorted(d.id for d in hand)) for hand in state.hands),
            tuple((seat, d.id) for seat, d in state.trick),
            len(state.completed), tuple(state.points), state.result)


def _prepare_state(state: HandState) -> HandState:
    if type(state) is not HandState:
        raise TypeError('The oracle requires a complete HandState, not an observation')
    validate_ruleset(state.ruleset_id)
    if (state.contract is None or state.contract.kind != ContractKind.NORMAL
            or state.phase != Phase.PLAY or state.bidder != state.contract.bidder):
        raise ValueError('Only ordinary contracts with bidding and trump fixed are supported')
    ACTION_REGISTRY.encode(Phase.TRUMP, state.trump)
    # Verify the supplied play history through the engine. Besides detecting
    # malformed snapshots, a full legal deal justifies the 42-point score bound.
    snapshot = deepcopy(state)
    hands = [list(hand) for hand in snapshot.hands]
    plays = [play for trick in snapshot.completed for play in trick] + snapshot.trick
    for seat, domino in plays:
        if type(seat) is not int or seat not in range(4) or not isinstance(domino, Domino):
            raise ValueError('Invalid play history')
        hands[seat].append(domino)
    deck = [d for hand in hands for d in hand]
    if (len(hands) != 4 or any(len(hand) != 7 for hand in hands)
            or any(not isinstance(d, Domino) for d in deck) or len(set(deck)) != 28):
        raise ValueError('A complete legal hand history and four remaining hands are required')
    replay = HandState(hands, snapshot.shaker, snapshot.marks, snapshot.target_marks,
                       snapshot.ruleset_id)
    replay.phase, replay.contract = Phase.PLAY, snapshot.contract
    replay.bidder = replay.seat = snapshot.bidder
    replay.trump = snapshot.trump
    for seat, domino in plays:
        if seat != replay.seat:
            raise ValueError('Play history has an inconsistent turn order')
        replay.apply(domino.id)
    if (_state_key(replay) != _state_key(snapshot)
            or replay.completed != snapshot.completed):
        raise ValueError('State scores, trick history, actor or result disagree with engine replay')
    return snapshot


def _child(state: HandState, action: int) -> HandState:
    # Play transitions mutate only these containers. Dominoes, contracts and
    # completed trick tuples are immutable; no deepcopy is needed per search node.
    child = copy(state)
    child.hands = [list(hand) for hand in state.hands]
    child.trick = list(state.trick)
    child.completed = list(state.completed)
    child.points = list(state.points)
    child.apply(action)
    return child


def exhaustive_minimax(full_state: HandState) -> int:
    """Unpruned, uncached terminal-only reference, intended for tiny positions.

    This intentionally does not use contract threshold cutoffs or alpha-beta.
    There is no search budget; use solve() for larger positions.
    """
    state = _prepare_state(full_state)
    team = state.contract.bidder % 2

    def visit(node):
        if node.result is not None:
            return 1 if node.result.scoring_team == team else -1
        values = [visit(_child(node, action)) for action in node.legal_action_ids()]
        return (max if node.seat % 2 == team else min)(values)

    return visit(state)


def _decided_value(state: HandState) -> int | None:
    team = state.contract.bidder % 2
    if state.result is not None:
        return 1 if state.result.scoring_team == team else -1
    target = state.contract.bid.target_points
    if contract_decided(state.points[team], state.points[1 - team], target):
        return 1 if state.points[team] >= target else -1
    return None


def _combine(values: list[ValueBounds], *, maximizing: bool, complete: bool) -> ValueBounds:
    # An unsearched child has the full outcome range, never an estimated value.
    if not complete:
        values = [*values, ValueBounds()]
    select = max if maximizing else min
    return ValueBounds(select(v.lower for v in values), select(v.upper for v in values))


class _Search:
    def __init__(self, limits: SearchLimits, start: float):
        self.limits = limits
        self.deadline = None if limits.time_seconds is None else start + limits.time_seconds
        self.nodes = 0
        self.cache_hits = 0
        self.exhausted = False
        # Insertion-ordered dict, FIFO eviction. Survival is never required for correctness.
        self.cache: dict[tuple, _Entry] = {}

    def enter(self) -> bool:
        if ((self.limits.max_nodes is not None and self.nodes >= self.limits.max_nodes)
                or (self.deadline is not None and monotonic() >= self.deadline)):
            self.exhausted = True
            return False
        self.nodes += 1
        return True

    def store(self, key: tuple, bounds: ValueBounds, alpha: int, beta: int) -> None:
        if self.exhausted or not self.limits.max_cache_entries:
            return
        # Classify against the caller's original window, before TT tightening.
        if bounds.exact:
            entry = _Entry(bounds.lower, _Flag.EXACT)
        elif bounds.lower >= beta:
            entry = _Entry(bounds.lower, _Flag.LOWER_BOUND)
        elif bounds.upper <= alpha:
            entry = _Entry(bounds.upper, _Flag.UPPER_BOUND)
        else:
            return
        if key not in self.cache and len(self.cache) >= self.limits.max_cache_entries:
            del self.cache[next(iter(self.cache))]
        self.cache[key] = entry

    def search(self, state: HandState, alpha: int = -2, beta: int = 2) -> ValueBounds:
        if not self.enter():
            return ValueBounds()
        decided = _decided_value(state)
        if decided is not None:
            return ValueBounds(decided, decided)
        original_alpha, original_beta = alpha, beta
        key = _state_key(state)
        prior = ValueBounds()
        entry = self.cache.get(key)
        if entry is not None:
            self.cache_hits += 1
            prior = entry.bounds
            alpha, beta = max(alpha, prior.lower), min(beta, prior.upper)
            if prior.exact or alpha >= beta:
                return prior
        maximizing = state.seat % 2 == state.contract.bidder % 2
        actions = state.legal_action_ids()  # Stable global-ID ordering only.
        values = []
        for action in actions:
            values.append(self.search(_child(state, action), alpha, beta))
            bounds = _combine(values, maximizing=maximizing, complete=len(values) == len(actions))
            bounds = ValueBounds(max(bounds.lower, prior.lower), min(bounds.upper, prior.upper))
            if maximizing:
                alpha = max(alpha, bounds.lower)
            else:
                beta = min(beta, bounds.upper)
            # The utility domain itself can prove an extreme before all moves.
            if bounds.exact or alpha >= beta or self.exhausted:
                break
        self.store(key, bounds, original_alpha, original_beta)
        return bounds


def solve(full_state: HandState, *, output: Literal['position_value', 'all_action_values'] =
          'all_action_values', limits: SearchLimits | None = None) -> OracleResult:
    """Solve a legal full state without mutating it or changing policy inputs.

    status refers to the requested output: position_value can finish with some
    action values unresolved. value/exact refer only to the root bounds. Unknown
    actions retain [-1, +1]; optimal_actions is supplied only when complete.
    Budgets count entered nodes (including root, terminal and cache-hit nodes).
    Time includes input copying/validation and is checked between search nodes.
    """
    start = monotonic()
    if output not in ('position_value', 'all_action_values'):
        raise ValueError('output must be position_value or all_action_values')
    state = _prepare_state(full_state)
    search = _Search(limits if limits is not None else SearchLimits(), start)
    actions = state.legal_action_ids()
    rows = [ActionValue(action_id=action) for action in actions]
    bounds = ValueBounds()
    if search.enter():
        decided = _decided_value(state)
        if decided is not None:
            bounds = ValueBounds(decided, decided)
            rows = [ActionValue(decided, decided, action) for action in actions]
        else:
            maximizing = state.seat % 2 == state.contract.bidder % 2
            for index, action in enumerate(actions):
                before = search.nodes
                # Full window for EACH root child, sharing only the table.
                child_bounds = search.search(_child(state, action))
                rows[index] = ActionValue(child_bounds.lower, child_bounds.upper,
                                          action, search.nodes > before)
                bounds = _combine(list(rows), maximizing=maximizing, complete=True)
                if search.exhausted or (output == 'position_value' and bounds.exact):
                    break
    complete = all(row.exact for row in rows)
    optimal = next((row.action_id for row in rows if row.exact and row.value == bounds.value), None)
    optimal_actions = (tuple(row.action_id for row in rows if row.value == bounds.value)
                       if bounds.exact and complete else None)
    fulfilled = bounds.exact and (not actions or optimal is not None)
    if output == 'all_action_values':
        fulfilled &= complete
    return OracleResult(bounds.lower, bounds.upper,
                        status='exact' if fulfilled else 'budget_exceeded',
                        optimal_action=optimal, optimal_actions=optimal_actions,
                        action_values=tuple(rows), all_action_values_complete=complete,
                        nodes=search.nodes, cache_hits=search.cache_hits,
                        elapsed_seconds=monotonic() - start)
