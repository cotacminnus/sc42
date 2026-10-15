"""Exact open-hand analysis of ordinary contracts; never a policy input.

Contract-success values use the bidding partnership's perspective. The point
objective has an explicit reference team, defaulting to the bidders. Only the
engine generates moves, advances tricks, and scores the hand.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from enum import StrEnum
import math
from time import monotonic
from typing import Callable, Literal, Sequence

from .actions import ACTION_REGISTRY, TRUMP_ACTIONS
from .arena import contract_decided
from .episodes import EpisodeState
from .models import Bid, Contract, ContractKind, Domino, Trump
from .parallel import AUTO_WORKER_CAP, OrderedPool, resolve_workers
from .phases import Phase
from .rulesets import RULESET_ID, validate_ruleset
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


class _Search:
    def __init__(self, limits: SearchLimits, start: float, *,
                 objective: Literal['contract_success', 'final_points'] = 'contract_success',
                 reference_team: int | None = None):
        if objective not in ('contract_success', 'final_points'):
            raise ValueError('Unknown oracle objective')
        if reference_team is not None and (type(reference_team) is not int or reference_team not in (0, 1)):
            raise ValueError('reference_team must be 0 or 1')
        if objective == 'contract_success' and reference_team is not None:
            raise ValueError('Contract success always uses the bidding team')
        self.objective = objective
        self.reference_team = reference_team
        self.domain = ValueBounds(0, 42) if objective == 'final_points' else ValueBounds()
        self.limits = limits
        self.deadline = None if limits.time_seconds is None else start + limits.time_seconds
        self.nodes = 0
        self.cache_hits = 0
        self.exhausted = False
        # Insertion-ordered dict, FIFO eviction. Survival is never required for correctness.
        self.cache: dict[tuple, _Entry] = {}

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
        self.nodes += 1
        return True

    def store(self, key: tuple, bounds: ValueBounds, alpha: int, beta: int) -> None:
        if self.exhausted or not self.limits.max_cache_entries:
            return
        # Classify against the caller's original window, before TT tightening.
        if bounds.exact:
            entry = _Entry(bounds.lower, _Flag.EXACT, self.domain)
        elif bounds.lower >= beta:
            entry = _Entry(bounds.lower, _Flag.LOWER_BOUND, self.domain)
        elif bounds.upper <= alpha:
            entry = _Entry(bounds.upper, _Flag.UPPER_BOUND, self.domain)
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
                return prior
        maximizing = state.seat % 2 == self.team(state)
        actions = state.legal_action_ids()  # Stable global-ID ordering only.
        values = []
        for action in actions:
            values.append(self.search(_child(state, action), alpha, beta))
            bounds = _combine(values, maximizing=maximizing, complete=len(values) == len(actions),
                              unknown=prior)
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
          'all_action_values', limits: SearchLimits | None = None, workers: int = 1) -> OracleResult:
    """Solve a legal full state without mutating it or changing policy inputs.

    status refers to the requested output: position_value can finish with some
    action values unresolved. value/exact refer only to the root bounds. Unknown
    actions retain [-1, +1]; optimal_actions is supplied only when complete.
    Budgets count entered nodes (including root, terminal and cache-hit nodes).
    Time includes input copying/validation and is checked between search nodes.
    workers=1 preserves sequential execution; more workers split root children
    into independent spawned searches under the same position limits.
    """
    start = monotonic()
    if output not in ('position_value', 'all_action_values'):
        raise ValueError('output must be position_value or all_action_values')
    state = _prepare_state(full_state)
    search = _Search(limits if limits is not None else SearchLimits(), start)
    count = min(AUTO_WORKER_CAP, resolve_workers(workers, len(state.legal_action_ids())))
    if count > 1 and _budget_available(search):
        return _parallel_root(state, search, output=output, start=start, workers=count)
    return _solve_state(state, search, output=output, start=start)


def solve_points(full_state: HandState, *, output: Literal['position_value', 'all_action_values'] =
                 'all_action_values', limits: SearchLimits | None = None,
                 reference_team: int | None = None, workers: int = 1) -> OracleResult:
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
                     objective='final_points', reference_team=reference_team)
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
                        elapsed_seconds=monotonic() - start)


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
    status: Literal['exact', 'node_limit', 'time_limit', 'not_started', 'cancelled']
    nodes: int
    cache_hits: int
    elapsed_seconds: float
    global_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _LeadResult:
    action: ActionValue
    nodes: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0
    status: str = 'not_started'
    exhausted: bool = False


def analyze_deal(hands: Sequence[Sequence[Domino]], *, ruleset_id: str = RULESET_ID,
                 limits: SearchLimits | None = None, workers: int = 1,
                 progress: Callable[[DealProgress], None] | None = None) -> DealAnalysis:
    """Analyze 252 independent opening-lead continuations, silently by default.

    Each lead receives max_nodes; time_seconds is one global cooperative limit,
    including validation, startup and shutdown. The continuation root counts as
    one node. Worker-local caches share the aggregate max_cache_entries cap.
    Progress runs in the coordinator as tasks settle; final ordering is canonical.
    """
    start = monotonic()
    session = EpisodeState('hand', 7, 0, ruleset_id)
    try:
        detached = [list(hand) for hand in hands]
    except TypeError as error:
        raise ValueError('A deal requires four hands of seven unique dominoes') from error
    session.deal(detached)
    deal = session.hand
    search = _Search(limits if limits is not None else SearchLimits(), start, objective='final_points')
    keys = [(seat, trump, action) for seat in range(4) for trump in TRUMP_ACTIONS
            for action in _play_start(deal, seat, trump).legal_action_ids()]
    count = min(AUTO_WORKER_CAP, resolve_workers(workers, len(keys)))
    config = _WorkerConfig(deal, 'deal', 'final_points', None, start,
                           search.limits.time_seconds, search.limits.max_cache_entries // count)
    tasks = [(key, search.limits.max_nodes) for key in keys]
    results = {}

    def settle(index, result, status):
        seat, trump, action = keys[index]
        if result is None:
            result = _LeadResult(ActionValue(0, 42, action), status=status)
        results[index] = result.action
        search.nodes += result.nodes
        search.cache_hits += result.cache_hits
        if progress is not None:
            progress(DealProgress(len(results), len(keys), seat, trump, action, result.status,
                                  result.nodes, result.cache_hits, result.elapsed_seconds,
                                  monotonic() - start))

    if count > 1 and _budget_available(search):
        with OrderedPool(count, initializer=_initialize_oracle, initargs=(config,)) as pool:
            for index, result, status in pool.iter_completed(_oracle_worker, tasks, deadline=search.deadline):
                settle(index, result, status)
    else:
        for index, task in enumerate(tasks):
            if search.deadline is not None and monotonic() >= search.deadline:
                settle(index, None, 'not_started')
            else:
                settle(index, _search_task(config, task, search.cache), 'completed')
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
    return DealAnalysis(tuple(bidders), search.nodes, search.cache_hits, monotonic() - start)


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


_WORKER_CONFIG: _WorkerConfig | None = None
_WORKER_CACHE: dict[tuple, _Entry] = {}


def _initialize_oracle(config: _WorkerConfig) -> None:
    global _WORKER_CONFIG, _WORKER_CACHE
    _WORKER_CONFIG = config
    _WORKER_CACHE = {}


def _search_task(config: _WorkerConfig, task: tuple, cache: dict) -> tuple | _LeadResult:
    """Only the task boundary is parallel; minimax itself is unchanged."""
    key, allocation = task
    task_start = monotonic()
    limits = SearchLimits(config.time_seconds, allocation, config.cache_entries)
    reference_team = key[0] % 2 if config.mode == 'deal' else config.reference_team
    search = _Search(limits, config.start, objective=config.objective,
                     reference_team=reference_team)
    search.cache = cache
    if config.mode == 'deal':
        seat, trump, action = key
        # A queued task may start after the coordinator's deadline check. Do
        # not construct or enter its game tree after that absolute deadline.
        if search.deadline is not None and task_start >= search.deadline:
            return _LeadResult(ActionValue(0, 42, action), status='not_started')
        state = _child(_play_start(config.state, seat, trump), action)
        bounds = search.search(state)
        row = ActionValue(bounds.lower, bounds.upper, action, search.nodes > 0)
        status = ('exact' if bounds.exact else 'node_limit'
                  if allocation is not None and search.nodes >= allocation else 'time_limit')
        return _LeadResult(row, search.nodes, search.cache_hits, monotonic() - task_start,
                           status, search.exhausted)
    bounds = search.search(_child(config.state, key))
    return ActionValue(bounds.lower, bounds.upper, key, search.nodes > 0), search.nodes, search.cache_hits


def _oracle_worker(task: tuple) -> tuple | _LeadResult:
    try:
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
                           search.limits.max_cache_entries // workers)
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
                results.append(result)
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
            bounds = decided
            rows = [ActionValue(decided.lower, decided.upper, action) for action in actions]
        elif _budget_available(search):
            results = _parallel_tasks(state, search, actions, mode='root', start=start, workers=workers)
            rows = [row for row, _, _ in results]
            search.nodes += sum(nodes for _, nodes, _ in results)
            search.cache_hits += sum(hits for _, _, hits in results)
            bounds = _combine(rows, maximizing=state.seat % 2 == search.team(state), complete=True)
    return _root_result(state, search, rows, bounds, output=output, start=start)
