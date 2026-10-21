"""Exact open-hand analysis of ordinary contracts; never a policy input.

Contract-success values use the bidding partnership's perspective. The point
objective has an explicit reference team, defaulting to the bidders. Only the
engine generates moves, advances tricks, and scores the hand.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
import atexit
import math
from multiprocessing.shared_memory import SharedMemory
from time import monotonic
from typing import Callable, Literal, Sequence

from .actions import ACTION_REGISTRY, PLAY_ACTIONS, TRUMP_ACTIONS
from .arena import contract_decided
from .episodes import EpisodeState
from .models import Bid, Contract, ContractKind, Domino, Trump
from .parallel import AUTO_WORKER_CAP, OrderedPool, resolve_workers
from .phases import Phase
from .rulesets import RULESET_ID, validate_ruleset
from .rules import is_trump, led_suit, suit_rank, trick_winner
from .state import HandState


@dataclass(frozen=True, slots=True)
class SearchLimits:
    """Position budgets; deal mode uses nodes per lead and time for the whole deal."""

    time_seconds: float | None = 1800.0
    max_nodes: int | None = 1_048_576
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
    # False with the full outcome range means unknown, not an estimate. A score
    # proof at the root can establish an action without visiting its child.
    searched: bool = False

    @property
    def status(self) -> Literal['exact', 'unresolved']:
        return 'exact' if self.exact else 'unresolved'


MoveOrdering = Literal['none', 'heuristic', 'tt', 'combined']


def _validate_ordering(ordering: str) -> None:
    if ordering not in ('none', 'heuristic', 'tt', 'combined'):
        raise ValueError('move_ordering must be none, heuristic, tt, or combined')


@dataclass(frozen=True, slots=True)
class SearchStatistics:
    alpha_beta_cutoffs: int = 0
    tt_cutoffs: int = 0
    score_bound_cutoffs: int = 0
    ordering_calls: int = 0

    def __add__(self, other: SearchStatistics) -> SearchStatistics:
        return SearchStatistics(self.alpha_beta_cutoffs + other.alpha_beta_cutoffs,
                                self.tt_cutoffs + other.tt_cutoffs,
                                self.score_bound_cutoffs + other.score_bound_cutoffs,
                                self.ordering_calls + other.ordering_calls)


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
    statistics: SearchStatistics = field(default=SearchStatistics(), compare=False)


@dataclass(frozen=True, slots=True)
class TrumpAnalysis(ValueBounds):
    lower: int = 0
    upper: int = 42
    trump: Trump = None
    optimal_action: int | None = None
    optimal_actions: tuple[int, ...] | None = None
    action_values: tuple[ActionValue, ...] = ()
    all_action_values_complete: bool = False

    @property
    def status(self) -> Literal['exact', 'unresolved']:
        return 'exact' if self.exact and self.optimal_actions is not None else 'unresolved'


@dataclass(frozen=True, slots=True)
class TrumpLead:
    trump: Trump
    action_id: int


def _makeable_bid(points: int) -> Bid | None:
    if points < 30:
        return None
    return Bid(marks=1) if points == 42 else Bid(points=points)


@dataclass(frozen=True, slots=True)
class BidderAnalysis(ValueBounds):
    lower: int = 0
    upper: int = 42
    seat: int = 0
    optimal_choice: TrumpLead | None = None
    optimal_choices: tuple[TrumpLead, ...] | None = None
    trumps: tuple[TrumpAnalysis, ...] = ()

    @property
    def team(self) -> int:
        return self.seat % 2

    @property
    def contract_exact(self) -> bool:
        return _makeable_bid(self.lower) == _makeable_bid(self.upper)

    @property
    def maximum_contract(self) -> Bid | None:
        # contract_exact disambiguates unknown from proven no makeable bid.
        return _makeable_bid(self.lower) if self.contract_exact else None

    @property
    def proven_contract(self) -> Bid | None:
        return _makeable_bid(self.lower)

    @property
    def possible_contract(self) -> Bid | None:
        return _makeable_bid(self.upper)

    @property
    def all_trump_values_complete(self) -> bool:
        return all(t.exact for t in self.trumps)

    @property
    def all_action_values_complete(self) -> bool:
        return all(t.all_action_values_complete for t in self.trumps)

    @property
    def status(self) -> Literal['exact', 'unresolved']:
        return 'exact' if self.exact and self.optimal_choices is not None else 'unresolved'


@dataclass(frozen=True, slots=True)
class DealAnalysis:
    bidders: tuple[BidderAnalysis, ...]
    nodes: int
    cache_hits: int
    elapsed_seconds: float
    output: Literal['all_action_values', 'maximum_contract'] = 'all_action_values'
    statistics: SearchStatistics = field(default=SearchStatistics(), compare=False)
    task_status_counts: tuple[tuple[str, int], ...] = ()

    @property
    def requested_output_complete(self) -> bool:
        if self.output == 'maximum_contract':
            return all(b.contract_exact for b in self.bidders)
        return self.all_alternative_values_complete

    @property
    def status(self) -> Literal['exact', 'budget_exceeded']:
        return 'exact' if all(b.status == 'exact' for b in self.bidders) else 'budget_exceeded'

    @property
    def all_alternative_values_complete(self) -> bool:
        return all(b.all_action_values_complete for b in self.bidders)


class _Flag(StrEnum):
    EXACT = 'exact'
    LOWER_BOUND = 'lower_bound'
    UPPER_BOUND = 'upper_bound'


@dataclass(frozen=True, slots=True)
class _Entry:
    value: int
    flag: _Flag
    domain: ValueBounds = ValueBounds()
    preferred_action: int | None = None

    @property
    def bounds(self) -> ValueBounds:
        if self.flag == _Flag.EXACT:
            return ValueBounds(self.value, self.value)
        if self.flag == _Flag.LOWER_BOUND:
            return ValueBounds(self.value, self.domain.upper)
        return ValueBounds(self.domain.lower, self.value)


def _state_key(state: HandState, objective: str = 'contract_success',
               reference_team: int | None = None) -> tuple:
    """Readable, immutable, absolute-seat key for this hand-only objective.

    Completed trick *count* affects engine termination. Their contents/order
    cannot affect future legality or scoring once points have been awarded.
    Auction history, shaker and match marks likewise cannot affect this objective.
    """
    return (f'ordinary_{objective}_v1', state.ruleset_id,
            state.contract.bidder % 2 if reference_team is None else reference_team,
            state.phase, state.contract, state.bidder,
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


def _copy_state(state: HandState) -> HandState:
    # Play transitions mutate only these containers. Dominoes, contracts and
    # completed trick tuples are immutable; no deepcopy is needed per search node.
    child = copy(state)
    child.hands = [list(hand) for hand in state.hands]
    child.trick = list(state.trick)
    child.completed = list(state.completed)
    child.points = list(state.points)
    return child


def _child(state: HandState, action: int) -> HandState:
    child = _copy_state(state)
    child.apply(action)
    return child


def exhaustive_minimax(full_state: HandState, *, objective: Literal['contract_success',
                       'final_points'] = 'contract_success', reference_team: int | None = None) -> int:
    """Unpruned, uncached terminal-only reference, intended for tiny positions.

    This intentionally does not use contract threshold cutoffs or alpha-beta.
    There is no search budget; use solve() for larger positions.
    """
    state = _prepare_state(full_state)
    search = _Search(SearchLimits(), monotonic(), objective=objective, reference_team=reference_team)
    team = search.team(state)

    def visit(node):
        if node.result is not None:
            if objective == 'final_points':
                return node.result.team_points[team]
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


def _combine(values: list[ValueBounds], *, maximizing: bool, complete: bool,
             unknown: ValueBounds = ValueBounds()) -> ValueBounds:
    # An unsearched child has the full outcome range, never an estimated value.
    if not complete:
        values = [*values, unknown]
    select = max if maximizing else min
    return ValueBounds(select(v.lower for v in values), select(v.upper for v in values))


def _choice_proof(values: Sequence[ValueBounds], *, maximizing: bool = True
                  ) -> tuple[ValueBounds, int | None, tuple[int, ...] | None]:
    """Prove choices from bounds, including strict dominance of unresolved ones.

    A single surviving interval can be a proven optimum without an exact value.
    Touching intervals do not establish a tie unless each choice is proven.
    """
    bounds = _combine(list(values), maximizing=maximizing, complete=True)
    proven = []
    excluded = []
    for index, value in enumerate(values):
        others = [v for i, v in enumerate(values) if i != index]
        if maximizing:
            optimal = all(value.lower >= v.upper for v in others)
            suboptimal = value.upper < bounds.lower
        else:
            optimal = all(value.upper <= v.lower for v in others)
            suboptimal = value.lower > bounds.upper
        if optimal:
            proven.append(index)
        if suboptimal:
            excluded.append(index)
    complete = len(proven) + len(excluded) == len(values)
    return bounds, next(iter(proven), None), tuple(proven) if complete else None


def _move_priority(state: HandState, action: int) -> tuple:
    """Ordering only: favor this actor's team, never estimate a search leaf.

    Partial winners are provisional. Feeding counters to a winning partnership
    and shedding cheap losers are hints, not claims that a trick is secured.
    All suit, trump and winner comparisons are delegated to the rule functions.
    """
    domino = PLAY_ACTIONS[action]
    rank = suit_rank(domino, led_suit(domino, state.trump))
    strength = rank[0] * 8 + rank[1]
    trump = is_trump(domino, state.trump)
    if not state.trick:
        return (strength, int(trump), -domino.count_value)
    winner = trick_winner([*state.trick, (state.seat, domino)], state.trump)
    stake = 1 + sum(d.count_value for _, d in state.trick) + domino.count_value
    own = winner % 2 == state.seat % 2
    return (stake if own else -stake, -int(trump), -strength)


class _Search:
    def __init__(self, limits: SearchLimits, start: float, *,
                 objective: Literal['contract_success', 'final_points'] = 'contract_success',
                 reference_team: int | None = None, move_ordering: MoveOrdering = 'none',
                 stop_check: Callable[[], bool] | None = None):
        if objective not in ('contract_success', 'final_points'):
            raise ValueError('Unknown oracle objective')
        if reference_team is not None and (type(reference_team) is not int or reference_team not in (0, 1)):
            raise ValueError('reference_team must be 0 or 1')
        if objective == 'contract_success' and reference_team is not None:
            raise ValueError('Contract success always uses the bidding team')
        self.objective = objective
        _validate_ordering(move_ordering)
        self.move_ordering = move_ordering
        self.reference_team = reference_team
        self.domain = ValueBounds(0, 42) if objective == 'final_points' else ValueBounds()
        self.limits = limits
        self.deadline = None if limits.time_seconds is None else start + limits.time_seconds
        self.nodes = 0
        self.cache_hits = 0
        self.exhausted = False
        self.cancelled = False
        self.stop_check = stop_check
        self.alpha_beta_cutoffs = 0
        self.tt_cutoffs = 0
        self.score_bound_cutoffs = 0
        self.ordering_calls = 0
        # Insertion-ordered dict, FIFO eviction. Survival is never required for correctness.
        self.cache: dict[tuple, _Entry] = {}

    @property
    def statistics(self) -> SearchStatistics:
        return SearchStatistics(self.alpha_beta_cutoffs, self.tt_cutoffs,
                                self.score_bound_cutoffs, self.ordering_calls)

    def ordered_actions(self, state: HandState, preferred: int | None = None) -> tuple[int, ...]:
        actions = state.legal_action_ids()
        if len(actions) < 2 or self.move_ordering == 'none':
            return actions
        self.ordering_calls += 1
        if self.move_ordering in ('heuristic', 'combined'):
            actions = tuple(sorted(actions, key=lambda a: (*(-v for v in _move_priority(state, a)), a)))
        if self.move_ordering in ('tt', 'combined') and preferred in actions:
            actions = (preferred, *(a for a in actions if a != preferred))
        return actions

    def team(self, state: HandState) -> int:
        return state.contract.bidder % 2 if self.reference_team is None else self.reference_team

    def initial_bounds(self, state: HandState) -> ValueBounds:
        if self.objective == 'final_points':
            team = self.team(state)
            return ValueBounds(state.points[team], 42 - state.points[1 - team])
        return self.domain

    def decided_bounds(self, state: HandState) -> ValueBounds | None:
        if self.objective == 'final_points':
            bounds = self.initial_bounds(state)
            return bounds if bounds.exact else None
        value = _decided_value(state)
        return ValueBounds(value, value) if value is not None else None

    def enter(self) -> bool:
        if ((self.limits.max_nodes is not None and self.nodes >= self.limits.max_nodes)
                or (self.deadline is not None and monotonic() >= self.deadline)):
            self.exhausted = True
            return False
        # Shared proof flags are monotone, coordinator-owned ordering hints for
        # cancellation only. Stale reads can waste work, never establish proofs.
        if self.stop_check is not None and self.nodes % 128 == 0 and self.stop_check():
            self.exhausted = self.cancelled = True
            return False
        self.nodes += 1
        return True

    def store(self, key: tuple, bounds: ValueBounds, alpha: int, beta: int,
              preferred_action: int | None = None) -> None:
        if self.exhausted or not self.limits.max_cache_entries:
            return
        # Classify against the caller's original window, before TT tightening.
        if bounds.exact:
            entry = _Entry(bounds.lower, _Flag.EXACT, self.domain, preferred_action)
        elif bounds.lower >= beta:
            entry = _Entry(bounds.lower, _Flag.LOWER_BOUND, self.domain, preferred_action)
        elif bounds.upper <= alpha:
            entry = _Entry(bounds.upper, _Flag.UPPER_BOUND, self.domain, preferred_action)
        else:
            return
        if key not in self.cache and len(self.cache) >= self.limits.max_cache_entries:
            del self.cache[next(iter(self.cache))]
        self.cache[key] = entry

    def search(self, state: HandState, alpha: int | None = None, beta: int | None = None) -> ValueBounds:
        prior = self.initial_bounds(state)
        if not self.enter():
            return prior
        alpha = self.domain.lower - 1 if alpha is None else alpha
        beta = self.domain.upper + 1 if beta is None else beta
        decided = self.decided_bounds(state)
        if decided is not None:
            self.score_bound_cutoffs += 1
            return decided
        original_alpha, original_beta = alpha, beta
        key = _state_key(state, self.objective, self.reference_team)
        entry = self.cache.get(key)
        if entry is not None:
            self.cache_hits += 1
            cached = entry.bounds
            prior = ValueBounds(max(prior.lower, cached.lower), min(prior.upper, cached.upper))
            alpha, beta = max(alpha, prior.lower), min(beta, prior.upper)
            if prior.exact or alpha >= beta:
                self.tt_cutoffs += 1
                return prior
        maximizing = state.seat % 2 == self.team(state)
        actions = self.ordered_actions(state, entry.preferred_action if entry is not None else None)
        values = []
        preferred = None
        best = None
        for action in actions:
            value = self.search(_child(state, action), alpha, beta)
            values.append(value)
            candidate = value.lower if maximizing else value.upper
            if best is None or (candidate > best if maximizing else candidate < best):
                best, preferred = candidate, action
            bounds = _combine(values, maximizing=maximizing, complete=len(values) == len(actions),
                              unknown=prior)
            bounds = ValueBounds(max(bounds.lower, prior.lower), min(bounds.upper, prior.upper))
            if maximizing:
                alpha = max(alpha, bounds.lower)
            else:
                beta = min(beta, bounds.upper)
            # The utility domain itself can prove an extreme before all moves.
            if bounds.exact or alpha >= beta or self.exhausted:
                if len(values) < len(actions):
                    if bounds.exact:
                        self.score_bound_cutoffs += 1
                    elif alpha >= beta:
                        self.alpha_beta_cutoffs += 1
                break
        self.store(key, bounds, original_alpha, original_beta, preferred)
        return bounds


def _contract_search(state: HandState, search: _Search) -> ValueBounds:
    """Refine integer point bounds only until their ordinary bid classes agree.

    First ask whether 42 is forceable; subsequent unit-width integer windows
    bisect the remaining bid range. Every query uses the same node counter,
    deadline and TT. Below 30, exact point utility is deliberately unnecessary.
    A fail-high/low is retained as a bound, never as an exact score.
    """
    bounds = search.initial_bounds(state)
    while _makeable_bid(bounds.lower) != _makeable_bid(bounds.upper) and not search.exhausted:
        threshold = 42 if bounds.upper == 42 else max(30, (bounds.lower + bounds.upper + 1) // 2)
        proof = search.search(state, alpha=threshold - 1, beta=threshold)
        bounds = ValueBounds(max(bounds.lower, proof.lower), min(bounds.upper, proof.upper))
    return bounds


def solve(full_state: HandState, *, output: Literal['position_value', 'all_action_values'] =
          'all_action_values', limits: SearchLimits | None = None, workers: int = 1,
          move_ordering: MoveOrdering = 'none') -> OracleResult:
    """Solve a legal full state without mutating it or changing policy inputs.

    status refers to the requested output: position_value can finish with some
    action values unresolved. value/exact refer only to the root bounds. Unknown
    actions retain [-1, +1]; optimal_actions is supplied only when complete.
    Budgets count entered nodes (including root, terminal and cache-hit nodes).
    Time includes input copying/validation and is checked between search nodes.
    workers=1 preserves sequential execution; more workers split root children
    into independent spawned searches under the same position limits.
    move_ordering changes traversal only; reported root rows remain canonical.
    """
    start = monotonic()
    if output not in ('position_value', 'all_action_values'):
        raise ValueError('output must be position_value or all_action_values')
    state = _prepare_state(full_state)
    search = _Search(limits if limits is not None else SearchLimits(), start, move_ordering=move_ordering)
    count = min(AUTO_WORKER_CAP, resolve_workers(workers, len(state.legal_action_ids())))
    if count > 1 and _budget_available(search):
        return _parallel_root(state, search, output=output, start=start, workers=count)
    return _solve_state(state, search, output=output, start=start)


def solve_points(full_state: HandState, *, output: Literal['position_value', 'all_action_values'] =
                 'all_action_values', limits: SearchLimits | None = None,
                 reference_team: int | None = None, workers: int = 1,
                 move_ordering: MoveOrdering = 'none') -> OracleResult:
    """Maximum guaranteed final points for a fixed trump, using the shared search.

    The reference defaults to the bidding partnership. Unlike solve(), making
    the contract does not end this objective. Values/bounds range from 0 to 42.
    workers selects the same optional root-action parallelism as solve().
    """
    start = monotonic()
    if output not in ('position_value', 'all_action_values'):
        raise ValueError('output must be position_value or all_action_values')
    state = _prepare_state(full_state)
    search = _Search(limits if limits is not None else SearchLimits(), start,
                     objective='final_points', reference_team=reference_team, move_ordering=move_ordering)
    count = min(AUTO_WORKER_CAP, resolve_workers(workers, len(state.legal_action_ids())))
    if count > 1 and _budget_available(search):
        return _parallel_root(state, search, output=output, start=start, workers=count)
    return _solve_state(state, search, output=output, start=start)


def _solve_state(state: HandState, search: _Search, *, output: str, start: float,
                  minimum: int | None = None) -> OracleResult:
    """Analyze root actions of a trusted state with caller-owned search resources.

    minimum is a caller-provided incumbent for a MAX choice. Searching
    at alpha=minimum-1 preserves ties while permitting strictly inferior bounds.
    Ordinary all-action APIs retain full windows for every root child.
    """
    actions = state.legal_action_ids()
    initial = search.initial_bounds(state)
    rows = [ActionValue(initial.lower, initial.upper, action) for action in actions]
    bounds = initial
    if search.enter():
        decided = search.decided_bounds(state)
        if decided is not None:
            search.score_bound_cutoffs += 1
            bounds = decided
            rows = [ActionValue(decided.lower, decided.upper, action) for action in actions]
        else:
            maximizing = state.seat % 2 == search.team(state)
            for index, action in enumerate(actions):
                before = search.nodes
                alpha = None if minimum is None else max(minimum, bounds.lower) - 1
                child_bounds = search.search(_child(state, action), alpha=alpha)
                rows[index] = ActionValue(child_bounds.lower, child_bounds.upper,
                                          action, search.nodes > before)
                bounds = _combine(list(rows), maximizing=maximizing, complete=True)
                if search.exhausted or (output == 'position_value' and bounds.exact):
                    break
    return _root_result(state, search, rows, bounds, output=output, start=start)


def _root_result(state: HandState, search: _Search, rows: list[ActionValue], bounds: ValueBounds,
                 *, output: str, start: float) -> OracleResult:
    """Existing proof/aggregation rules shared by sequential and parallel roots."""
    actions = state.legal_action_ids()
    complete = all(row.exact for row in rows)
    optimal = next((row.action_id for row in rows if row.exact and row.value == bounds.value), None)
    optimal_actions = (tuple(row.action_id for row in rows if row.value == bounds.value)
                       if bounds.exact and complete else None)
    if search.objective == 'final_points' and rows:
        _, choice, choices = _choice_proof(rows, maximizing=state.seat % 2 == search.team(state))
        optimal = rows[choice].action_id if choice is not None else None
        optimal_actions = tuple(rows[i].action_id for i in choices) if choices is not None else None
    fulfilled = bounds.exact and (not actions or optimal is not None)
    if output == 'all_action_values':
        fulfilled &= complete
    return OracleResult(bounds.lower, bounds.upper,
                        status='exact' if fulfilled else 'budget_exceeded',
                        optimal_action=optimal, optimal_actions=optimal_actions,
                        action_values=tuple(rows), all_action_values_complete=complete,
                        nodes=search.nodes, cache_hits=search.cache_hits,
                        elapsed_seconds=monotonic() - start, statistics=search.statistics)


def _play_start(deal: HandState, bidder: int, trump: Trump) -> HandState:
    """Fresh play start from a boundary-validated deal; no history reconstruction.

    The normal 30-point placeholder drives engine transitions, never point
    utility. Ordinary play legality depends on hand/lead/trump, not bid/stake.
    """
    state = _copy_state(deal)
    state.contract = Contract(ContractKind.NORMAL, bidder, Bid(points=30))
    state.bidder = state.seat = bidder
    state.phase = Phase.TRUMP
    state.apply(ACTION_REGISTRY.encode(Phase.TRUMP, trump))
    return state


def _bidder_analysis(seat: int, trumps: Sequence[TrumpAnalysis]) -> BidderAnalysis:
    # Flattening both MAX levels preserves every trump/lead tie. No requirement
    # to solve a strictly dominated alternative merely to prove the optimum.
    values = [action for trump in trumps for action in trump.action_values]
    pairs = [TrumpLead(trump.trump, action.action_id)
             for trump in trumps for action in trump.action_values]
    bounds, choice, choices = _choice_proof(values)
    return BidderAnalysis(bounds.lower, bounds.upper, seat,
                          pairs[choice] if choice is not None else None,
                          tuple(pairs[i] for i in choices) if choices is not None else None,
                          tuple(trumps))


@dataclass(frozen=True, slots=True)
class DealProgress:
    """One settled opening-lead task, reported only by the coordinator.

    exact refers to its bounds, not completion of the other 251 alternatives.
    elapsed_seconds excludes queueing; global_elapsed_seconds includes it.
    """
    completed: int
    total: int
    bidder: int
    trump: Trump
    action_id: int
    status: Literal['exact', 'contract_proven', 'node_limit', 'time_limit', 'not_started',
                    'cancelled', 'skipped_proven', 'cancelled_proven', 'cancelled_deadline']
    nodes: int
    cache_hits: int
    elapsed_seconds: float
    global_elapsed_seconds: float
    bidder_contract_proven: bool = False
    maximum_contract: Bid | None = None


@dataclass(frozen=True, slots=True)
class _LeadResult:
    action: ActionValue
    nodes: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0
    status: str = 'not_started'
    exhausted: bool = False
    statistics: SearchStatistics = SearchStatistics()


def analyze_deal(hands: Sequence[Sequence[Domino]], *, ruleset_id: str = RULESET_ID,
                 limits: SearchLimits | None = None, workers: int = 1,
                 progress: Callable[[DealProgress], None] | None = None,
                 move_ordering: MoveOrdering = 'none',
                 output: Literal['all_action_values', 'maximum_contract'] = 'all_action_values',
                 proof_cancellation: bool = True) -> DealAnalysis:
    """Analyze 252 independent opening-lead continuations, silently by default.

    Each lead receives max_nodes; time_seconds is one global cooperative limit,
    including validation, startup and shutdown. The continuation root counts as
    one node. Worker-local caches share the aggregate max_cache_entries cap.
    Progress runs in the coordinator as tasks settle; final ordering is canonical.
    maximum_contract uses threshold proofs and optional bidder-level stopping;
    requested_output_complete distinguishes that goal from legacy exact status.
    """
    start = monotonic()
    if output not in ('all_action_values', 'maximum_contract'):
        raise ValueError('output must be all_action_values or maximum_contract')
    if type(proof_cancellation) is not bool:
        raise ValueError('proof_cancellation must be a boolean')
    session = EpisodeState('hand', 7, 0, ruleset_id)
    try:
        detached = [list(hand) for hand in hands]
    except TypeError as error:
        raise ValueError('A deal requires four hands of seven unique dominoes') from error
    session.deal(detached)
    deal = session.hand
    search = _Search(limits if limits is not None else SearchLimits(), start, objective='final_points',
                     move_ordering=move_ordering)
    keys = [(seat, trump, action) for seat in range(4) for trump in TRUMP_ACTIONS
            for action in _play_start(deal, seat, trump).legal_action_ids()]
    count = min(AUTO_WORKER_CAP, resolve_workers(workers, len(keys)))
    config = _WorkerConfig(deal, 'deal', 'final_points', None, start,
                           search.limits.time_seconds, search.limits.max_cache_entries // count,
                           move_ordering, output)
    tasks = [(key, search.limits.max_nodes) for key in keys]
    results = {}
    statistics = SearchStatistics()
    status_counts = {}
    rows = [ActionValue(0, 42, action) for _, _, action in keys]
    proven = [False] * 4
    stop_memory = None

    order = list(range(len(keys)))
    if output == 'maximum_contract':
        # Deterministic, bidder-interleaved priority. Concentrated trumps and
        # strong leads are scheduling hints only; every legal choice remains.
        priorities = []
        for seat in range(4):
            ranked = []
            for trump in TRUMP_ACTIONS:
                state = _play_start(deal, seat, trump)
                concentration = sum(is_trump(d, trump) for d in deal.hands[seat])
                for action in state.legal_action_ids():
                    index = keys.index((seat, trump, action))
                    ranked.append(((-concentration, *(-v for v in _move_priority(state, action)), index), index))
            priorities.append([index for _, index in sorted(ranked)])
        order = [ranked[i] for i in range(63) for ranked in priorities]
    tasks = [tasks[i] for i in order]

    def skip(index):
        return ('skipped_proven' if output == 'maximum_contract' and proof_cancellation
                and proven[keys[order[index]][0]] else None)

    def settle(index, result, status):
        nonlocal statistics
        seat, trump, action = keys[index]
        if result is None:
            if status in ('not_started', 'cancelled'):
                status = 'cancelled_deadline'
            result = _LeadResult(ActionValue(0, 42, action), status=status)
        elif result.status == 'not_started':
            result = replace(result, status='cancelled_deadline')
        results[index] = result.action
        rows[index] = result.action
        search.nodes += result.nodes
        search.cache_hits += result.cache_hits
        statistics += result.statistics
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        bidder_rows = rows[seat * 63:(seat + 1) * 63]
        lower, upper = max(a.lower for a in bidder_rows), max(a.upper for a in bidder_rows)
        newly_proven = not proven[seat] and _makeable_bid(lower) == _makeable_bid(upper)
        if newly_proven:
            proven[seat] = True
            if stop_memory is not None:
                stop_memory.buf[seat] = 1
        if progress is not None:
            progress(DealProgress(len(results), len(keys), seat, trump, action, result.status,
                                  result.nodes, result.cache_hits, result.elapsed_seconds,
                                  monotonic() - start, newly_proven,
                                  _makeable_bid(lower) if proven[seat] else None))

    if count > 1 and _budget_available(search):
        try:
            if output == 'maximum_contract' and proof_cancellation:
                stop_memory = SharedMemory(create=True, size=4)
                stop_memory.buf[:] = bytes(4)
                config = replace(config, stop_name=stop_memory.name)
            with OrderedPool(count, initializer=_initialize_oracle, initargs=(config,)) as pool:
                scheduling = {'skip': skip} if output == 'maximum_contract' and proof_cancellation else {}
                for index, result, status in pool.iter_completed(_oracle_worker, tasks,
                                                                deadline=search.deadline, **scheduling):
                    settle(order[index], result, status)
        finally:
            if stop_memory is not None:
                stop_memory.close()
                stop_memory.unlink()
    else:
        for index, task in enumerate(tasks):
            if search.deadline is not None and monotonic() >= search.deadline:
                settle(order[index], None, 'not_started')
            elif skip(index) is not None:
                settle(order[index], None, 'skipped_proven')
            else:
                settle(order[index], _search_task(config, task, search.cache), 'completed')
    bidders = []
    offset = 0
    for seat in range(4):
        trumps = []
        for trump in TRUMP_ACTIONS:
            rows = tuple(results[i] for i in range(offset, offset + 7))
            offset += 7
            bounds, choice, choices = _choice_proof(rows)
            trumps.append(TrumpAnalysis(bounds.lower, bounds.upper, trump,
                                        rows[choice].action_id if choice is not None else None,
                                        tuple(rows[i].action_id for i in choices) if choices is not None else None,
                                        rows, all(row.exact for row in rows)))
        bidders.append(_bidder_analysis(seat, trumps))
    return DealAnalysis(tuple(bidders), search.nodes, search.cache_hits, monotonic() - start,
                        output=output, statistics=statistics, task_status_counts=tuple(sorted(status_counts.items())))


def _budget_available(search: _Search) -> bool:
    return ((search.limits.max_nodes is None or search.nodes < search.limits.max_nodes)
            and (search.deadline is None or monotonic() < search.deadline))


@dataclass(frozen=True, slots=True)
class _WorkerConfig:
    # Sent once per process, never once per node or action. Each spawned process
    # owns its deserialized state and cache; ordinary policies receive neither.
    state: HandState
    mode: Literal['root', 'deal']
    objective: Literal['contract_success', 'final_points']
    reference_team: int | None
    start: float
    time_seconds: float | None
    cache_entries: int
    move_ordering: MoveOrdering = 'none'
    output: Literal['all_action_values', 'maximum_contract'] = 'all_action_values'
    stop_name: str | None = None


_WORKER_CONFIG: _WorkerConfig | None = None
_WORKER_CACHE: dict[tuple, _Entry] = {}
_WORKER_STOP: SharedMemory | None = None


def _initialize_oracle(config: _WorkerConfig) -> None:
    global _WORKER_CONFIG, _WORKER_CACHE, _WORKER_STOP
    _WORKER_CONFIG = config
    _WORKER_CACHE = {}
    _WORKER_STOP = SharedMemory(name=config.stop_name) if config.stop_name is not None else None
    if _WORKER_STOP is not None:
        atexit.register(_WORKER_STOP.close)


def _search_task(config: _WorkerConfig, task: tuple, cache: dict, *, stop_flags=None) -> tuple | _LeadResult:
    """Workers reuse the shared alpha-beta implementation and engine transitions."""
    key, allocation = task
    task_start = monotonic()
    limits = SearchLimits(config.time_seconds, allocation, config.cache_entries)
    reference_team = key[0] % 2 if config.mode == 'deal' else config.reference_team
    search = _Search(limits, config.start, objective=config.objective,
                     reference_team=reference_team, move_ordering=config.move_ordering,
                     stop_check=(lambda: bool(stop_flags[key[0]])) if stop_flags is not None else None)
    search.cache = cache
    if config.mode == 'deal':
        seat, trump, action = key
        # A queued task may start after the coordinator's deadline check. Do
        # not construct or enter its game tree after that absolute deadline.
        if search.deadline is not None and task_start >= search.deadline:
            return _LeadResult(ActionValue(0, 42, action), status='not_started')
        if stop_flags is not None and stop_flags[seat]:
            return _LeadResult(ActionValue(0, 42, action), status='skipped_proven')
        state = _child(_play_start(config.state, seat, trump), action)
        bounds = (_contract_search(state, search) if config.output == 'maximum_contract'
                  else search.search(state))
        row = ActionValue(bounds.lower, bounds.upper, action, search.nodes > 0)
        if bounds.exact:
            status = 'exact'
        elif config.output == 'maximum_contract' and _makeable_bid(bounds.lower) == _makeable_bid(bounds.upper):
            status = 'contract_proven'
        elif search.cancelled:
            status = 'cancelled_proven' if search.nodes else 'skipped_proven'
        elif allocation is not None and search.nodes >= allocation:
            status = 'node_limit'
        else:
            status = 'time_limit'
        return _LeadResult(row, search.nodes, search.cache_hits, monotonic() - task_start,
                           status, search.exhausted, search.statistics)
    bounds = search.search(_child(config.state, key))
    return (ActionValue(bounds.lower, bounds.upper, key, search.nodes > 0), search.nodes,
            search.cache_hits, search.statistics)


def _oracle_worker(task: tuple) -> tuple | _LeadResult:
    try:
        if _WORKER_STOP is not None:
            return _search_task(_WORKER_CONFIG, task, _WORKER_CACHE, stop_flags=_WORKER_STOP.buf)
        return _search_task(_WORKER_CONFIG, task, _WORKER_CACHE)
    except Exception as error:
        raise RuntimeError(f'Oracle {_WORKER_CONFIG.mode} task {task[0]!r} failed: {error}') from error


def _parallel_tasks(state: HandState, search: _Search, keys: Sequence, *, mode: str,
                    start: float, workers: int) -> list[tuple]:
    """Share one endgame position's budget among independent root branches.

    Disjoint node allocations are reserved in waves and unused nodes reclaimed.
    Worker-local caches survive tasks within the aggregate cache cap.
    """
    config = _WorkerConfig(state, mode, search.objective, search.reference_team,
                           start, search.limits.time_seconds,
                           search.limits.max_cache_entries // workers, search.move_ordering)
    remaining = (None if search.limits.max_nodes is None else
                 search.limits.max_nodes - search.nodes)
    results = []
    with OrderedPool(workers, initializer=_initialize_oracle, initargs=(config,)) as pool:
        for offset in range(0, len(keys), workers):
            wave = keys[offset:offset + workers]
            expired = search.deadline is not None and monotonic() >= search.deadline
            share, extra = divmod(remaining, len(keys) - offset) if remaining is not None else (None, 0)
            allocations = [0 if expired else (None if share is None else share + (i < extra))
                           for i in range(len(wave))]
            jobs = [(key, allocation) for key, allocation in zip(wave, allocations) if allocation != 0]
            completed = iter(pool.map(_oracle_worker, jobs)) if jobs else iter(())
            for key, allocation in zip(wave, allocations):
                # No process is needed to represent an unvisited task. Use the
                # same zero-budget logic, including any awarded-point bounds.
                result = (next(completed) if allocation != 0 else
                          _search_task(config, (key, 0), {}))
                search.alpha_beta_cutoffs += result[3].alpha_beta_cutoffs
                search.tt_cutoffs += result[3].tt_cutoffs
                search.score_bound_cutoffs += result[3].score_bound_cutoffs
                search.ordering_calls += result[3].ordering_calls
                results.append(result[:3])
                if remaining is not None:
                    remaining -= result[1]
    return results


def _parallel_root(state: HandState, search: _Search, *, output: str, start: float,
                   workers: int) -> OracleResult:
    actions = state.legal_action_ids()
    initial = search.initial_bounds(state)
    bounds = initial
    rows = [ActionValue(initial.lower, initial.upper, action) for action in actions]
    if search.enter():
        decided = search.decided_bounds(state)
        if decided is not None:
            search.score_bound_cutoffs += 1
            bounds = decided
            rows = [ActionValue(decided.lower, decided.upper, action) for action in actions]
        elif _budget_available(search):
            results = _parallel_tasks(state, search, actions, mode='root', start=start, workers=workers)
            rows = [row for row, _, _ in results]
            search.nodes += sum(nodes for _, nodes, _ in results)
            search.cache_hits += sum(hits for _, _, hits in results)
            bounds = _combine(rows, maximizing=state.seat % 2 == search.team(state), complete=True)
    return _root_result(state, search, rows, bounds, output=output, start=start)


# ---------------------------------------------------------------------------
# Arbitrary play positions (training labels, belief leaves, diagnostics).
# ---------------------------------------------------------------------------

PositionObjective = Literal['final_points', 'contract']


@dataclass(frozen=True, slots=True)
class PlayPosition:
    """A play-phase position without its earlier trick contents.

    hands: remaining permanent domino IDs per absolute seat. trick: current
    partial trick as (seat, domino ID) in play order, started by leader.
    captured_points: points already awarded to teams (0, 1) by completed tricks.
    trump: a decoded trump value (0-6, DOUBLE_TRUMP or None for Follow Me).
    bid is the contract stake; None uses the 30-point placeholder that the
    final-point objective never reads. next_player, when given, is verified.
    """

    hands: tuple[tuple[int, ...], ...]
    trick: tuple[tuple[int, int], ...]
    leader: int
    trump: Trump
    bidder: int
    captured_points: tuple[int, int]
    bid: Bid | None = None
    next_player: int | None = None
    contract_kind: ContractKind = ContractKind.NORMAL

    @property
    def completed_tricks(self) -> int:
        played = {seat for seat, _ in self.trick}
        return 7 - len(self.hands[self.leader]) - int(self.leader in played)


@dataclass(frozen=True, slots=True)
class PositionActionResult:
    """Certified final-total bounds for one candidate action.

    lower/upper are final team points for the reference team, including points
    already captured. contract_status is None for the final-point objective.
    termination: exact, threshold_proven, decided, node_limit, time_limit or
    not_started (the global deadline expired first). Never an estimate.
    """

    action_id: int
    lower: int
    upper: int
    contract_status: Literal['proven_makeable', 'proven_unmakeable', 'unresolved'] | None
    nodes: int
    termination: str

    @property
    def exact(self) -> bool:
        return self.lower == self.upper


@dataclass(frozen=True, slots=True)
class PositionEvaluation:
    """Per-action and root bounds; units are always final team points (0-42)."""

    objective: PositionObjective
    reference_team: int
    acting_seat: int
    threshold: int | None
    actions: tuple[PositionActionResult, ...]
    lower: int
    upper: int
    all_legal_actions: bool
    proven_optimal_actions: tuple[int, ...] | None
    nodes: int
    cache_hits: int
    elapsed_seconds: float
    units: str = 'final_team_points'

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    @property
    def contract_status(self) -> str | None:
        return None if self.threshold is None else _threshold_status(self.lower, self.upper, self.threshold)

    @property
    def perspective(self) -> str:
        return f'team {self.reference_team}'


def _threshold_status(lower: int, upper: int, threshold: int) -> str:
    if lower >= threshold:
        return 'proven_makeable'
    if upper < threshold:
        return 'proven_unmakeable'
    return 'unresolved'


def position_from_state(state: HandState) -> PlayPosition:
    """Privileged snapshot of a live play-phase HandState (never a policy input)."""
    if state.phase != Phase.PLAY or state.result is not None or state.contract is None:
        raise ValueError('Expected an unfinished play-phase hand')
    leader = state.trick[0][0] if state.trick else state.seat
    return PlayPosition(tuple(tuple(sorted(d.id for d in hand)) for hand in state.hands),
                        tuple((seat, d.id) for seat, d in state.trick), leader, state.trump,
                        state.contract.bidder, tuple(state.points), state.contract.bid,
                        state.seat, state.contract.kind)


def position_state(position: PlayPosition) -> HandState:
    """Validate a PlayPosition and build a search-only HandState.

    Earlier trick contents are unknown, so `completed` holds empty placeholders:
    only their count affects termination, legality and scoring. Consistency
    checks: disjoint hands, turn order from the leader, hand sizes matching the
    trick count, captured points equal to completed tricks plus counters of
    every domino no longer in a hand or the partial trick, and partial-trick
    followers either following suit or still void in it. Do not pass the
    result to solve(), which requires a fully replayable history.
    """
    if position.contract_kind != ContractKind.NORMAL:
        raise ValueError('Only ordinary contracts are supported by the position oracle')
    seats = (position.leader, position.bidder)
    if any(type(s) is not int or s not in range(4) for s in seats):
        raise ValueError('leader and bidder must be seats 0-3')
    hands = tuple(tuple(hand) for hand in position.hands)
    trick = tuple(tuple(play) for play in position.trick)
    ids = [i for hand in hands for i in hand] + [d for _, d in trick]
    if (len(hands) != 4 or any(type(i) is not int or i not in range(28) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Expected four hands and a trick of distinct domino IDs 0-27')
    if len(trick) > 3:
        raise ValueError('A partial trick has at most three plays')
    if [seat for seat, _ in trick] != [(position.leader + i) % 4 for i in range(len(trick))]:
        raise ValueError('Partial trick seats must follow the leader in turn order')
    actor = (position.leader + len(trick)) % 4
    if position.next_player is not None and position.next_player != actor:
        raise ValueError('next_player disagrees with the leader and partial trick')
    completed = position.completed_tricks
    played = {seat for seat, _ in trick}
    if (not 0 <= completed < 7 or
            any(len(hands[s]) != 7 - completed - int(s in played) for s in range(4))):
        raise ValueError('Hand sizes are inconsistent with one shared trick count')
    ACTION_REGISTRY.encode(Phase.TRUMP, position.trump)
    points = tuple(position.captured_points)
    gone = set(range(28)) - set(ids)
    expected = completed + sum(PLAY_ACTIONS[i].count_value for i in gone)
    if (len(points) != 2 or any(type(p) is not int or p < 0 for p in points)
            or sum(points) != expected):
        raise ValueError(f'Captured points must total {expected} for this position')
    if trick:
        suit = led_suit(PLAY_ACTIONS[trick[0][1]], position.trump)
        from .rules import follows_suit
        for seat, domino in trick[1:]:
            if (not follows_suit(PLAY_ACTIONS[domino], suit, position.trump)
                    and any(follows_suit(PLAY_ACTIONS[i], suit, position.trump) for i in hands[seat])):
                raise ValueError(f'Seat {seat} failed to follow suit while holding it')
    bid = position.bid if position.bid is not None else Bid(points=30)
    state = HandState([[PLAY_ACTIONS[i] for i in hand] for hand in hands],
                      (position.bidder - 1) % 4, (0, 0), 7)
    state.phase, state.contract = Phase.PLAY, Contract(ContractKind.NORMAL, position.bidder, bid)
    state.bidder, state.trump, state.seat = position.bidder, position.trump, actor
    state.trick = [(seat, PLAY_ACTIONS[d]) for seat, d in trick]
    state.completed = [()] * completed
    state.points = list(points)
    return state


def evaluate_position(position: PlayPosition, *, objective: PositionObjective = 'final_points',
                      reference_team: int | None = None, threshold: int | None = None,
                      actions: Sequence[int] | None = None, limits: SearchLimits | None = None,
                      total_time_seconds: float | None = None, move_ordering: MoveOrdering = 'none',
                      cache: dict | None = None) -> PositionEvaluation:
    """Certified bounds for each candidate action of an arbitrary play position.

    Budgets: `limits.max_nodes` and `limits.time_seconds` apply to EACH action
    search separately (a fresh node counter and clock per action);
    `total_time_seconds` is one optional global deadline for the whole call.
    Exhausted searches keep their valid bounds and are never labelled exact.

    objective='final_points' bounds the reference team's final points (default
    team: the acting seat's). objective='contract' asks whether the bidding
    team can force `threshold` (default: the contract's target points): a
    lower bound >= threshold proves makeable, an upper bound < threshold proves
    unmakeable. Its reference team is always the bidders. Both objectives use
    the same alpha-beta search, transposition entries and final-total units;
    `cache` may be shared across calls because keys include the ruleset,
    objective and reference team.
    """
    start = monotonic()
    if objective not in ('final_points', 'contract'):
        raise ValueError("objective must be 'final_points' or 'contract'")
    state = position_state(position)
    bidders = position.bidder % 2
    if objective == 'contract':
        if reference_team not in (None, bidders):
            raise ValueError('The contract objective uses the bidding team as reference')
        reference_team = bidders
        if threshold is None:
            if position.bid is None:
                raise ValueError('The contract objective needs a bid or an explicit threshold')
            threshold = position.bid.target_points
        if type(threshold) is not int or not 30 <= threshold <= 42:
            raise ValueError('An ordinary contract threshold must be 30-42 points')
    else:
        if threshold is not None:
            raise ValueError('threshold applies only to the contract objective')
        reference_team = state.seat % 2 if reference_team is None else reference_team
    if reference_team not in (0, 1):
        raise ValueError('reference_team must be 0 or 1')
    if total_time_seconds is not None and (not math.isfinite(total_time_seconds) or total_time_seconds < 0):
        raise ValueError('total_time_seconds must be finite and nonnegative')
    limits = limits if limits is not None else SearchLimits()
    legal = state.legal_action_ids()
    requested = legal if actions is None else tuple(actions)
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(legal):
        raise ValueError('Candidate actions must be distinct legal play IDs')
    deadline = None if total_time_seconds is None else start + total_time_seconds
    cache = {} if cache is None else cache
    rows, nodes, hits = [], 0, 0
    for action in requested:
        now = monotonic()
        child = _child(state, action)
        if deadline is not None and now >= deadline:
            search = _Search(limits, now, objective='final_points', reference_team=reference_team)
            bounds, termination, used = search.initial_bounds(child), 'not_started', 0
        else:
            budget = limits
            if deadline is not None:
                remaining = deadline - now
                if limits.time_seconds is None or limits.time_seconds > remaining:
                    budget = replace(limits, time_seconds=remaining)
            search = _Search(budget, now, objective='final_points', reference_team=reference_team,
                             move_ordering=move_ordering)
            search.cache = cache
            if threshold is None:
                bounds = search.search(child)
            else:
                bounds = search.search(child, alpha=threshold - 1, beta=threshold)
            used = search.nodes
            hits += search.cache_hits
            if bounds.exact:
                termination = 'exact' if used else 'decided'
            elif threshold is not None and _threshold_status(bounds.lower, bounds.upper, threshold) != 'unresolved':
                termination = 'threshold_proven'
            elif budget.max_nodes is not None and used >= budget.max_nodes:
                termination = 'node_limit'
            else:
                termination = 'time_limit'
        nodes += used
        rows.append(PositionActionResult(
            action, bounds.lower, bounds.upper,
            None if threshold is None else _threshold_status(bounds.lower, bounds.upper, threshold),
            used, termination))
    complete = set(requested) == set(legal)
    maximizing = state.seat % 2 == reference_team
    prior = _Search(limits, start, objective='final_points', reference_team=reference_team).initial_bounds(state)
    root = _combine([ValueBounds(r.lower, r.upper) for r in rows], maximizing=maximizing,
                    complete=complete, unknown=prior)
    proven = None
    if complete and threshold is None:
        _, _, choices = _choice_proof([ValueBounds(r.lower, r.upper) for r in rows], maximizing=maximizing)
        proven = tuple(rows[i].action_id for i in choices) if choices is not None else None
    return PositionEvaluation(objective, reference_team, state.seat, threshold, tuple(rows),
                              root.lower, root.upper, complete, proven, nodes, hits,
                              monotonic() - start)
