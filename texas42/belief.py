"""Belief modelling over the hidden deal: count, weight, sample, simulate, evaluate.

WHAT THIS MODULE ANSWERS
    From one seat's point of view, which deals of the unseen dominoes are still
    possible, how likely is each, and what is each legal move worth on average
    over them? Everything here is derived from that seat's ordinary
    Observation. No true hidden hand is ever read. See docs/belief.md for the
    long-form explanation, worked numbers, and the research references.

WHY THIS IS TRACTABLE IN TEXAS 42
    The only hidden information in a hand is the initial deal: after the shuffle
    there are no draws, dice, or other chance events. Bridge and Skat share this
    structure, which is where Perfect Information Monte Carlo (PIMC: sample
    deals, then reason as if each were fully known) has worked best (Ginsberg's
    GIB, JAIR 2001; Long et al., AAAI 2010). Also, every hidden-information
    constraint the rules create can be stated per domino: "this domino cannot be
    in seat s, because s once failed to follow a suit that this domino belongs
    to". That makes exact counting cheap (section 2).

PIPELINE AND DATA FORMATS (absolute seats 0-3, permanent domino IDs 0-27)

    Observation (selfplay.Observation: actor's own hand + public history)
      | hidden_constraints()                                     [section 1]
      v
    HiddenConstraints: unknown tile IDs, open slots per seat,
                       allowed seats per unknown tile (from revealed voids)
      | count_worlds() / tile_marginals()   exact integer DP      [section 2]
      | enumerate_worlds() / sample_world()                       [section 3]
      v
    PrivilegedState worlds (remaining hands of all four seats, as IDs)
      | world_state()  rebuild deal + replay public history       [section 4]
      v                through HandState.apply (engine stays authoritative)
    HandState (a full simulator state for ONE hypothetical world)
      | policy_likelihood()  optional soft evidence from actions  [section 5]
      | build_belief()       -> Belief: worlds + normalized weights
      | simulate()           -> Rollout: action IDs, probability, points [6]
      | evaluate_actions()   -> ActionEvaluation: expected points per legal
                                action, paired against the best one    [7]

PRIVACY CONTRACT
    Inputs are Observation objects only, plus explicit random.Random streams.
    The PrivilegedState and HandState objects created here are HYPOTHESES built
    from public information, not the true deal. Passing them to a perfect-
    information evaluator (for example a future oracle as `leaf_evaluator`)
    therefore leaks nothing. Never pass the environment's real state in.

REPRODUCIBILITY
    Every random choice draws from a caller-supplied random.Random. The same
    Observation, arguments and seeded stream give identical results.
"""
from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

from .actions import PLAY_ACTIONS
from .agents.base import PrivilegedState
from .encoding import revealed_voids
from .history import public_history
from .models import Suit, Trump
from .rules import follows_suit
from .selfplay import Observation, observation_from_view
from .state import HandResult, HandState
from .training import TrainingBidView, TrainingPlayView

if TYPE_CHECKING:
    # Annotation-only: belief code never needs the arena (process pools,
    # statistics) at runtime, and agents.pimc imports this module.
    from .arena import AgentFactory

SEATS = (0, 1, 2, 3)
STOP_CONDITIONS = ('trick', 'phase', 'hand', 'turn')

# At or below this many consistent worlds, build_belief lists every world
# (exact expectations) instead of sampling. Cost grows linearly with worlds:
# evaluate_actions runs one rollout per (world, legal action). 256 makes exact
# evaluation the default from about trick 5 (90 worlds without voids) and
# often trick 4 once voids appear; raise it when time allows (docs/belief.md).
DEFAULT_ENUMERATE_BELOW = 256
DEFAULT_SAMPLES = 64
# Probability mass mixed into every observed action's likelihood so that one
# deterministic policy decision can never rule a world out completely. See
# policy_likelihood for the reasoning.
DEFAULT_EPSILON = 0.1


# ---------------------------------------------------------------------------
# 1. Hard evidence: which seats could still hold each unseen domino
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HiddenConstraints:
    """Everything the observer knows for certain about the unseen dominoes.

    Fields (all absolute seats, permanent domino IDs):
      observer       seat whose Observation produced these constraints.
      observer_hand  the observer's remaining hand (sorted). Known exactly.
      unknown        sorted IDs of dominoes neither in observer_hand nor played.
      slots          per-seat count of unseen dominoes still held: 7 minus
                     dominoes that seat has played, and 0 for the observer.
                     sum(slots) == len(unknown) always.
      allowed        aligned with `unknown`: the seats that could hold that
                     domino without contradicting a revealed void.
      voids          (seat, suit) pairs proven empty; suit is 0-6 or "doubles".
      trump          the hand's trump (None for Follow Me, or before trump).

    "Hard" means these are logical certainties. A world violating them is
    impossible, not merely unlikely. Soft evidence (what bids and play choices
    suggest) is handled separately as likelihood weights (section 5).
    """

    observer: int
    observer_hand: tuple[int, ...]
    unknown: tuple[int, ...]
    slots: tuple[int, int, int, int]
    allowed: tuple[tuple[int, ...], ...]
    voids: frozenset[tuple[int, Suit]]
    trump: Trump

    def groups(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """Unknown dominoes grouped by identical allowed-seat sets.

        Returns ((allowed seats, domino IDs), ...) in a deterministic order.
        Dominoes with the same allowed seats are interchangeable for counting.
        With three other seats there are at most 7 distinct non-empty seat
        sets, so the counting DP below has at most 7 stages.
        """
        by_seats: dict[tuple[int, ...], list[int]] = {}
        for domino_id, seats in zip(self.unknown, self.allowed):
            by_seats.setdefault(seats, []).append(domino_id)
        return tuple((seats, tuple(ids)) for seats, ids in sorted(by_seats.items()))


def hidden_constraints(obs: Observation) -> HiddenConstraints:
    """Derive the hard constraints on the hidden deal from one Observation.

    Theory: under Texas 42 rules a hand's only hidden quantity is who holds the
    unplayed dominoes. Two kinds of certain knowledge exist:
      1. Counting: every seat started with 7 dominoes, so seat s holds exactly
         7 - (dominoes s has played) of the unseen ones.
      2. Revealed voids: a player must follow the led suit when able
         (rules.legal_plays). Playing off-suit therefore proves the player
         held no domino of that suit at that moment, and hands only shrink,
         so the void lasts for the rest of the hand. encoding.revealed_voids
         already computes these; it is reused rather than reimplemented.
    Bidding and trump calls carry no hard constraints under the base rules:
    any hand may make any legal call.

    Data flow:
      obs.view.hand                              -> observer_hand
      completed_tricks + current trick           -> played IDs, per-seat counts
      28 IDs - observer_hand - played            -> unknown
      revealed_voids(view) + rules.follows_suit  -> allowed seats per unknown ID

    Raises ValueError when the observation is internally inconsistent (counts
    that do not add up, or a domino no seat could hold).
    """
    view = obs.view
    observer = obs.seat
    observer_hand = tuple(sorted(view.hand))
    played_count = [0, 0, 0, 0]
    played: set[int] = set()
    voids: frozenset[tuple[int, Suit]] = frozenset()
    trump: Trump = None
    if isinstance(view, TrainingPlayView):
        # Bid and trump views have no plays yet: no voids, everyone holds 7.
        trump = view.trump
        for trick in (*view.completed_tricks, view.trick):
            for seat, domino_id in trick:
                played_count[seat] += 1
                played.add(domino_id)
        voids = frozenset(revealed_voids(view))
    if len(observer_hand) != 7 - played_count[observer]:
        raise ValueError('Observer hand size disagrees with its played dominoes')
    unknown = tuple(sorted(set(range(28)) - set(observer_hand) - played))
    slots = tuple(0 if seat == observer else 7 - played_count[seat] for seat in SEATS)
    if sum(slots) != len(unknown):
        raise ValueError('Unseen domino count disagrees with remaining hand sizes')

    def can_hold(seat: int, domino_id: int) -> bool:
        # A seat may hold the domino unless the domino belongs (under this
        # trump) to some suit the seat has been proven void in.
        domino = PLAY_ACTIONS[domino_id]
        return not any(void_seat == seat and follows_suit(domino, suit, trump)
                       for void_seat, suit in voids)

    allowed = tuple(tuple(seat for seat in SEATS if slots[seat] > 0 and can_hold(seat, d))
                    for d in unknown)
    if any(not seats for seats in allowed):
        raise ValueError('An unseen domino cannot belong to any seat')
    return HiddenConstraints(observer, observer_hand, unknown, slots, allowed, voids, trump)


# ---------------------------------------------------------------------------
# 2. Exact counting: how many deals are consistent, and tile marginals
# ---------------------------------------------------------------------------
#
# Theory: a "world" is an assignment of every unknown domino to a seat such
# that each domino goes to an allowed seat and each seat receives exactly
# slots[s] dominoes. Dominoes with the same allowed-seat set are
# interchangeable, so we count per group: decide how many of the group's c
# dominoes go to each allowed seat (a "split" k_s, with sum k_s = c), then
# multiply by the multinomial c! / prod(k_s!) ways to pick WHICH dominoes those
# are. Summing over all splits, group by group, with the remaining seat
# capacities as DP state, gives the exact count:
#
#   ways(g, capacity) = sum over splits k of group g that fit capacity of
#                       multinomial(c_g; k) * ways(g + 1, capacity - k)
#   ways(G, capacity) = 1 if capacity is all zeros else 0
#
# This is the same "split constraints into interchangeable groups and count
# combinatorially" idea used by exact Minesweeper and Battleship probability
# solvers. With no voids it reduces to the bridge "vacant places" rule: a given
# unseen domino is in seat s with probability slots[s] / sum(slots).
# Python integers are exact, so counts never overflow or round.


def _multinomial(total: int, parts: Sequence[int]) -> int:
    """total! / prod(part!) as an exact integer."""
    result = math.factorial(total)
    for part in parts:
        result //= math.factorial(part)
    return result


def _splits(count: int, seats: tuple[int, ...], capacity: tuple[int, ...]
            ) -> Iterator[tuple[int, int, int, int]]:
    """Yield every 4-seat split of `count` dominoes over `seats` within capacity.

    Output: tuples (k0, k1, k2, k3) with k_s = 0 for seats outside `seats`,
    k_s <= capacity[s], and sum = count. Order is deterministic.
    """
    def place(index: int, left: int, split: list[int]):
        if index == len(seats):
            if left == 0:
                yield tuple(split)
            return
        seat = seats[index]
        for k in range(min(left, capacity[seat]) + 1):
            split[seat] = k
            yield from place(index + 1, left - k, split)
        split[seat] = 0

    yield from place(0, count, [0, 0, 0, 0])


class _WorldCounter:
    """Memoized DP over allowed-seat groups (see the section comment above).

    Built from (groups, slots). Holds no randomness, so it can be reused by
    counting, marginals, sampling and enumeration.
    """

    def __init__(self, groups, slots: tuple[int, ...]):
        self.groups = groups
        self.slots = tuple(slots)
        self._memo: dict[tuple[int, tuple[int, ...]], int] = {}

    def ways(self, index: int = 0, capacity: tuple[int, ...] | None = None) -> int:
        """Number of valid assignments of groups[index:] into `capacity`."""
        capacity = self.slots if capacity is None else capacity
        if index == len(self.groups):
            return 1 if not any(capacity) else 0
        key = (index, capacity)
        if key not in self._memo:
            seats, ids = self.groups[index]
            self._memo[key] = sum(
                _multinomial(len(ids), split) * self.ways(index + 1, _minus(capacity, split))
                for split in _splits(len(ids), seats, capacity))
        return self._memo[key]


def _minus(capacity: tuple[int, ...], split: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(c - k for c, k in zip(capacity, split))


def count_worlds(constraints: HiddenConstraints) -> int:
    """Exact number of hidden deals consistent with the hard constraints.

    Every such world is equally likely before soft evidence is considered:
    the shuffle was uniform, and conditioning a uniform distribution on a
    logical event keeps it uniform over the surviving outcomes. Typical values
    with no voids: 399,072,960 during bidding, 1,680 after four tricks, 6
    before the last trick (docs/belief.md has the full table).
    """
    return _WorldCounter(constraints.groups(), constraints.slots).ways()


def tile_marginals(constraints: HiddenConstraints) -> dict[int, tuple[float, float, float, float]]:
    """Exact P(seat holds domino) for every unknown domino, uniform over worlds.

    Output: {domino ID: (p_seat0, p_seat1, p_seat2, p_seat3)}, each row
    summing to 1, with 0 for the observer and for disallowed seats.

    Method: P(d in s) = (#worlds with d fixed in s) / (#worlds). Fixing d in s
    is the same counting problem with d removed and slots[s] reduced by one,
    so it reuses the DP. This is the uniform-prior "ground truth" belief: a
    baseline for learned belief models (roadmap step 7) and a direct check of
    the vacant-places rule.
    """
    groups = constraints.groups()
    total = _WorldCounter(groups, constraints.slots).ways()
    result: dict[int, tuple[float, float, float, float]] = {}
    for index, (seats, ids) in enumerate(groups):
        # All dominoes in one group share the same marginal (they are
        # interchangeable), so compute it once per group.
        reduced_groups = groups[:index] + ((seats, ids[1:]),) + groups[index + 1:]
        row = [0.0, 0.0, 0.0, 0.0]
        for seat in seats:
            slots = list(constraints.slots)
            slots[seat] -= 1
            row[seat] = _WorldCounter(reduced_groups, tuple(slots)).ways() / total
        for domino_id in ids:
            result[domino_id] = tuple(row)
    return result


# ---------------------------------------------------------------------------
# 3. Producing worlds: exact enumeration and exact uniform sampling
# ---------------------------------------------------------------------------

def _world(constraints: HiddenConstraints, assigned: Sequence[Sequence[int]]) -> PrivilegedState:
    """Combine assigned unknown dominoes with the observer's known hand."""
    hands = [sorted(hand) for hand in assigned]
    hands[constraints.observer] = list(constraints.observer_hand)
    return PrivilegedState(tuple(tuple(hand) for hand in hands))


def sample_world(constraints: HiddenConstraints, rng: random.Random) -> PrivilegedState:
    """Draw ONE consistent world, exactly uniformly over all consistent worlds.

    Method (exact sampling from a counted DP): walk the groups in order. For
    each group, choose a split with probability
        multinomial(split) * ways(next group, capacity - split) / ways(this group),
    i.e. in proportion to how many complete worlds each choice leads to. Then
    shuffle the group's dominoes and deal them out according to the split,
    which picks each of the multinomial(split) arrangements equally often. The
    probabilities telescope to 1 / count_worlds for every world: no rejection
    sampling and no bias. rng.randrange on exact integers avoids float rounding.

    Uniform is correct for hard evidence alone. Soft evidence is applied
    afterwards as weights (build_belief) or by resampling.
    """
    counter = _WorldCounter(constraints.groups(), constraints.slots)
    capacity = counter.slots
    assigned: list[list[int]] = [[], [], [], []]
    for index, (seats, ids) in enumerate(counter.groups):
        options = [(split, _multinomial(len(ids), split) * counter.ways(index + 1, _minus(capacity, split)))
                   for split in _splits(len(ids), seats, capacity)]
        pick = rng.randrange(sum(weight for _, weight in options))
        for split, weight in options:
            if pick < weight:
                break
            pick -= weight
        shuffled = list(ids)
        rng.shuffle(shuffled)
        start = 0
        for seat in SEATS:
            assigned[seat].extend(shuffled[start:start + split[seat]])
            start += split[seat]
        capacity = _minus(capacity, split)
    return _world(constraints, assigned)


def enumerate_worlds(constraints: HiddenConstraints, *, limit: int | None = None
                     ) -> Iterator[PrivilegedState]:
    """Yield EVERY consistent world exactly once, in a deterministic order.

    Use when count_worlds is small (late in a hand), so that expectations are
    exact rather than Monte Carlo estimates. This "maintain the full set of
    possible states" approach is what StrangeFish (Reconnaissance Blind Chess)
    does with board sets. Pass `limit` to fail fast (ValueError) instead of
    accidentally listing millions of worlds.
    """
    counter = _WorldCounter(constraints.groups(), constraints.slots)
    if limit is not None and counter.ways() > limit:
        raise ValueError(f'{counter.ways()} worlds exceed enumeration limit {limit}')

    def assign_group(index: int, capacity: tuple[int, ...], assigned: list[list[int]]):
        if index == len(counter.groups):
            if not any(capacity):
                yield _world(constraints, assigned)
            return
        seats, ids = counter.groups[index]
        for split in _splits(len(ids), seats, capacity):
            if counter.ways(index + 1, _minus(capacity, split)) == 0:
                continue  # Prune splits that cannot be completed.
            yield from deal_split(index, capacity, assigned, seats, list(ids), split, 0)

    def deal_split(index, capacity, assigned, seats, remaining, split, seat_index):
        # Choose which specific dominoes of this group go to each allowed seat.
        if seat_index == len(seats):
            yield from assign_group(index + 1, _minus(capacity, split), assigned)
            return
        seat = seats[seat_index]
        for chosen in combinations(remaining, split[seat]):
            rest = [d for d in remaining if d not in chosen]
            assigned[seat].extend(chosen)
            yield from deal_split(index, capacity, assigned, seats, rest, split, seat_index + 1)
            del assigned[seat][len(assigned[seat]) - len(chosen):]

    yield from assign_group(0, counter.slots, [[], [], [], []])


# ---------------------------------------------------------------------------
# 4. From a world to a full simulator state (engine stays authoritative)
# ---------------------------------------------------------------------------

def _played_by_seat(obs: Observation) -> list[list[int]]:
    played: list[list[int]] = [[], [], [], []]
    if isinstance(obs.view, TrainingPlayView):
        for trick in (*obs.view.completed_tricks, obs.view.trick):
            for seat, domino_id in trick:
                played[seat].append(domino_id)
    return played


def _replay(obs: Observation, world: PrivilegedState,
            on_decision: Callable[[Observation, int], None] | None = None) -> HandState:
    """Rebuild the hypothetical initial deal and replay every public decision.

    Steps and formats:
      1. initial hand of seat s = world.hands[s] (remaining) + dominoes s played.
         These are permanent IDs, converted to Domino objects for HandState.
      2. Recover the shaker. Bid views carry it; afterwards the first caller
         sits left of the shaker (same rule as encoding.encode_view).
      3. For each PublicEvent from history.public_history(obs.view), check that
         the expected seat is acting, optionally report (actor's observation in
         this world, chosen action ID) to `on_decision`, then HandState.apply().
         apply() validates legality, so a world that contradicts the public
         record raises ValueError instead of silently diverging.
      4. Check that the final state is the observer's turn with the same legal
         IDs as the Observation, which confirms the reconstruction is faithful.
    """
    view = obs.view
    if tuple(sorted(world.hands[obs.seat])) != tuple(sorted(view.hand)):
        raise ValueError("World disagrees with the observer's own hand")
    played = _played_by_seat(obs)
    initial = [[PLAY_ACTIONS[i] for i in (*sorted(world.hands[seat]), *played[seat])]
               for seat in SEATS]
    if isinstance(view, TrainingBidView):
        shaker = view.shaker
    else:
        shaker = (view.bid_history[0][0] - 1) % 4
    state = HandState(initial, shaker, tuple(view.match_marks), view.target_marks, obs.ruleset_id)
    for event in public_history(view).events:
        if state.seat != event.seat:
            raise ValueError('Public history does not match the reconstructed turn order')
        if on_decision is not None:
            on_decision(observation_from_view(state.view(), ruleset_id=obs.ruleset_id),
                        event.action_id)
        state.apply(event.action_id)
    if state.seat != obs.seat or state.legal_action_ids() != tuple(obs.legal_action_ids):
        raise ValueError('Reconstructed world does not reproduce the observation')
    return state


def _copy_state(state: HandState) -> HandState:
    """Independent HandState copy, much cheaper than copy.deepcopy.

    HandState keeps immutable values (Domino, Bid, Contract, HandResult,
    tuples) inside five mutable lists: hands (and each hand), bid_history,
    trick, completed and points. Copying exactly those lists makes the copy
    safe to step without touching the original. If HandState gains another
    mutable field, add it here (tests/test_belief.py checks independence).
    The oracle branch has a similar private helper; a shared HandState.copy()
    could replace both once that branch merges.
    """
    clone = copy.copy(state)
    clone.hands = [list(hand) for hand in state.hands]
    clone.bid_history = list(state.bid_history)
    clone.trick = list(state.trick)
    clone.completed = list(state.completed)
    clone.points = list(state.points)
    return clone


def world_state(obs: Observation, world: PrivilegedState) -> HandState:
    """Return a fresh HandState for `world`, positioned at the observer's decision.

    The result is a private, mutable simulator state for one HYPOTHETICAL deal.
    It can be stepped with HandState.apply, copied, rolled out (simulate) or
    handed to a perfect-information solver. Rules are never re-implemented
    here: every transition goes through the engine.
    """
    return _replay(obs, world)


# ---------------------------------------------------------------------------
# 5. Soft evidence: weighting worlds by how well they explain observed actions
# ---------------------------------------------------------------------------
#
# Theory (Bayes' rule over deals): for a world W and public history h,
#   P(W | h)  is proportional to  P(W) * prod over other players' decisions t of
#             P(a_t | that player's view at t in W)
# P(W) is uniform over consistent worlds (section 2). The product is the
# likelihood: how probable the observed moves would be if W were true, under an
# assumed model of how the other players act. This is Policy-Based Inference
# (Rebstock et al. 2019) and the Kermit Skat program's bid inference (Buro et
# al. 2009); poker's Loki kept the same per-holding "weight table".
#
# The observer's own decisions are skipped: they depend only on the observer's
# view, which is identical in every consistent world, so their factor is a
# constant that cancels on normalization.
#
# Deterministic policies (HeuristicAgent) give probability 1 or 0, so one
# unexpected move would zero out the true world. We therefore mix in
# epsilon: p' = (1 - epsilon) * p + epsilon / (number of legal actions).
# Epsilon expresses "the real player is not exactly my model", and matters
# most when modelling humans or unknown agents.


def policy_likelihood(obs: Observation, world: PrivilegedState, policies: Sequence[AgentFactory],
                      *, epsilon: float = DEFAULT_EPSILON) -> float:
    """Relative probability of the public history if `world` were the true deal.

    Inputs:
      policies  four arena-style factories, factory(seat, seed) -> agent, whose
                agents implement agents.base.ActionProbabilities. Each models
                how that seat chooses moves. Fresh agents are built for every
                world, so no memory leaks between hypotheses.
      epsilon   in [0, 1]; see the section comment above. epsilon = 1 does
                NOT give a flat belief: it models every player as choosing
                uniformly among its legal moves, and how many legal moves a
                player had depends on its hidden hand (follow-suit). For a flat
                belief, omit likelihood_policies in build_belief instead.
    Output: an unnormalized positive float. Compare it only with values for
    other worlds of the same Observation (build_belief normalizes them).
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError('epsilon must be between 0 and 1')
    if len(policies) != 4:
        raise ValueError('Provide one policy factory per seat')
    # Seeds are irrelevant here: ActionProbabilities must not consume RNG.
    agents = [factory(seat, seat) for seat, factory in enumerate(policies)]
    for seat, agent in enumerate(agents):
        agent.reset(seat)
    weight = 1.0

    def on_decision(decision: Observation, action_id: int) -> None:
        nonlocal weight
        if decision.seat == obs.seat:
            return  # Constant across worlds; see the section comment.
        query = getattr(agents[decision.seat], 'action_probabilities', None)
        if query is None:
            raise TypeError('Likelihood policies must implement action_probabilities')
        probabilities = query(decision)
        _check_distribution(probabilities, decision)
        uniform = 1.0 / len(decision.legal_action_ids)
        weight *= (1.0 - epsilon) * probabilities.get(action_id, 0.0) + epsilon * uniform

    _replay(obs, world, on_decision)
    return weight


def _check_distribution(probabilities: dict[int, float], obs: Observation) -> None:
    # Tolerance 1e-6: learned policies compute softmaxes in floating point, so
    # exact sums of 1 are not guaranteed; real mistakes are far larger.
    if (not set(probabilities) <= set(obs.legal_action_ids)
            or any(p < 0 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError('action_probabilities must be a distribution over legal IDs')


@dataclass(frozen=True, slots=True)
class Belief:
    """A weighted set of hypothetical worlds for one observer.

    worlds        PrivilegedState hypotheses (absolute seats, domino IDs).
    weights       normalized to sum to 1, aligned with worlds.
    world_count   exact number of hard-consistent worlds (count_worlds).
    mode          "enumerated": worlds are every consistent world, so weighted
                  averages are exact expectations under the model.
                  "sampled": worlds are i.i.d. uniform samples (duplicates
                  possible), so averages are Monte Carlo estimates. With policy
                  weights, this is self-normalized importance sampling with
                  the uniform distribution as the proposal.
    likelihood    "uniform" (hard evidence only) or "policy" (section 5).
    """

    observer: int
    constraints: HiddenConstraints
    world_count: int
    mode: str
    worlds: tuple[PrivilegedState, ...]
    weights: tuple[float, ...]
    likelihood: str
    epsilon: float | None = None

    @property
    def effective_sample_size(self) -> float:
        """1 / sum(w^2): how many equally weighted worlds the weights are worth.

        Standard sequential Monte Carlo diagnostic (Doucet et al.). It equals
        the number of worlds when weights are uniform, and approaches 1 when
        one world dominates. A small value means estimates rest on few worlds:
        sample more, raise epsilon, or resample.
        """
        return 1.0 / sum(w * w for w in self.weights)

    def marginals(self) -> dict[int, tuple[float, float, float, float]]:
        """Weighted P(seat holds domino) for unknown dominoes, from these worlds.

        Same format as tile_marginals. With uniform weights in enumerated mode
        the two agree exactly. With policy weights, this is the posterior
        belief that a learned belief model would try to predict.
        """
        result = {d: [0.0, 0.0, 0.0, 0.0] for d in self.constraints.unknown}
        for world, weight in zip(self.worlds, self.weights):
            for seat, hand in enumerate(world.hands):
                if seat == self.observer:
                    continue
                for domino_id in hand:
                    result[domino_id][seat] += weight
        return {d: tuple(row) for d, row in result.items()}


def build_belief(obs: Observation, *, rng: random.Random, samples: int = DEFAULT_SAMPLES,
                 enumerate_below: int = DEFAULT_ENUMERATE_BELOW,
                 likelihood_policies: Sequence[AgentFactory] | None = None,
                 epsilon: float = DEFAULT_EPSILON) -> Belief:
    """Construct the observer's belief over hidden deals.

    Decision rule: if count_worlds <= enumerate_below, list every world (exact).
    Otherwise draw `samples` uniform worlds (Monte Carlo). Weights are uniform,
    or proportional to policy_likelihood when likelihood_policies is given.

    When to use which:
      - Uniform: "all consistent deals equally likely". Correct if other
        players' choices are uninformative, and the right baseline for RQ2.
      - Policy: when you trust a model of the other players (for example the
        same Heuristic or PPO policies used in evaluation). Bids and partner
        signals then shift the belief, at the cost of one replay per world.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError('samples must be a positive integer')
    constraints = hidden_constraints(obs)
    count = count_worlds(constraints)
    if count <= enumerate_below:
        mode, worlds = 'enumerated', tuple(enumerate_worlds(constraints))
    else:
        mode, worlds = 'sampled', tuple(sample_world(constraints, rng) for _ in range(samples))
    if likelihood_policies is None:
        raw = [1.0] * len(worlds)
        likelihood, used_epsilon = 'uniform', None
    else:
        raw = [policy_likelihood(obs, w, likelihood_policies, epsilon=epsilon) for w in worlds]
        likelihood, used_epsilon = 'policy', epsilon
    total = sum(raw)
    if total <= 0:
        raise ValueError('Every world has zero likelihood; increase epsilon')
    return Belief(obs.seat, constraints, count, mode, worlds,
                  tuple(r / total for r in raw), likelihood, used_epsilon)


# ---------------------------------------------------------------------------
# 6. Simulation: play forward from a world until a stopping point
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Rollout:
    """The outcome of one simulated continuation in one world.

    action_ids           every global action ID applied, in order (permanent
                         IDs, never list positions, so they stay meaningful
                         across worlds).
    explicit_steps       how many leading actions came from `choices`.
    rollout_probability  probability that the modelled players would have
                         chosen the NON-explicit steps: the product of
                         per-step probabilities (uniform 1/|legal| for random
                         rollouts, ActionProbabilities for policies). None if a
                         policy cannot report probabilities. Explicit choices
                         are interventions ("what if I play X"), not chance
                         events, so they contribute no factor.
    start_points/end_points  team points (team 0, team 1) before and after.
    stop_reason          "trick", "phase", "turn" or "hand": the condition that ended it.
    hand_result          engine HandResult if the hand finished, else None.
    final_state          the private HandState where simulation stopped (for
                         a leaf evaluator). Mutating it affects nothing else.
    """

    action_ids: tuple[int, ...]
    explicit_steps: int
    rollout_probability: float | None
    start_points: tuple[int, int]
    end_points: tuple[int, int]
    stop_reason: str
    hand_result: HandResult | None
    final_state: HandState = field(repr=False, compare=False)

    @property
    def points_delta(self) -> tuple[int, int]:
        """Points each team gained during the rollout (team 0, team 1)."""
        return (self.end_points[0] - self.start_points[0], self.end_points[1] - self.start_points[1])


def simulate(state: HandState, *, rng: random.Random, choices: Sequence[int] = (),
             policies: Sequence[AgentFactory] | None = None, until: str = 'phase',
             track_probability: bool = True) -> Rollout:
    """Play forward from a COPY of `state` until `until`, returning what happened.

    This is the "rollout" of backgammon programs (Tesauro & Galperin 1996) and
    the inner simulation of PIMC and Recursive Monte Carlo search (Furtak &
    Buro 2013). The caller's state is never modified.

    Inputs:
      state     a HandState, typically from world_state (one hypothetical deal).
      rng       explicit stream for random decisions and agent seeds. Four
                64-bit seeds are drawn first (one per seat), so explicit and
                random steps consume the stream identically across calls.
      choices   leading decisions given as INDICES into the sorted legal action
                IDs at each step (0 = lowest ID). This lets a caller walk a
                "decision tree branch" without knowing IDs in advance. The IDs
                actually applied are reported in Rollout.action_ids.
      policies  four factories (arena convention) used after `choices` run
                out. None means uniformly random legal moves. Agents are built
                fresh and see observations only from this point on, so agents
                that remember earlier observations lack that history.
      until     "trick": stop when the next trick completes (or the hand ends).
                "phase": stop when the bid/trump/play phase changes, or the
                         hand ends (for play this is the end of the hand).
                "hand":  play to the end of the hand.
                "turn":  stop when the seat that acted first is to act again
                         (or the hand ends). In evaluate_actions that seat is
                         the searcher, so a leaf evaluator then scores the
                         searcher's OWN later information state. By perfect
                         recall that refines the root information, so an
                         information-set value leaf is unbiased there (tower
                         property; docs/search_training.md, section 4.6). After
                         "trick", whoever won the trick acts, and its value
                         ignores the searcher's private hand.
      track_probability  compute rollout_probability. Policy steps then call
                ActionProbabilities in addition to act (about twice the policy
                cost). Pass False when only the outcome matters, as
                evaluate_actions does; rollout_probability is then None.
    """
    if until not in STOP_CONDITIONS:
        raise ValueError(f'until must be one of {STOP_CONDITIONS}')
    if policies is not None and len(policies) != 4:
        raise ValueError('Provide one policy factory per seat')
    state = _copy_state(state)
    seeds = [rng.getrandbits(64) for _ in SEATS]
    agents = None
    if policies is not None:
        agents = [factory(seat, seeds[seat]) for seat, factory in enumerate(policies)]
        for seat, agent in enumerate(agents):
            agent.reset(seat)
    start_points = tuple(state.points)
    start_phase, start_tricks, start_seat = state.phase, len(state.completed), state.seat
    applied: list[int] = []
    probability: float | None = 1.0 if track_probability else None
    stop_reason = 'hand'
    while state.result is None:
        # legal: sorted global action IDs for whoever acts now, from the engine.
        legal = state.legal_action_ids()
        step = len(applied)
        if step < len(choices):
            # Branch 1, explicit intervention: the caller's index picks the move.
            # It adds no probability factor (see Rollout.rollout_probability).
            index = choices[step]
            if type(index) is not int or not 0 <= index < len(legal):
                raise ValueError(f'Choice {index!r} at step {step} is not an index into {len(legal)} legal actions')
            action_id = legal[index]
        elif agents is None:
            # Branch 2, uniform random continuation: each legal move has 1/|legal|.
            action_id = rng.choice(legal)
            probability = None if probability is None else probability / len(legal)
        else:
            # Branch 3, modelled player: the acting seat's agent sees only its
            # own Observation in this hypothetical world, exactly as in a real
            # game. Its pure probability query (when tracked) prices the move
            # that act() then chooses.
            obs = observation_from_view(state.view(), ruleset_id=state.ruleset_id)
            agent = agents[obs.seat]
            query = getattr(agent, 'action_probabilities', None) if track_probability else None
            if query is not None and probability is not None:
                probabilities = query(obs)
                _check_distribution(probabilities, obs)
            action_id = agent.act(obs)
            if type(action_id) is not int or action_id not in legal:
                raise ValueError(f'Policy returned illegal action {action_id!r}')
            if probability is not None:
                probability = None if query is None else probability * probabilities.get(action_id, 0.0)
        # The engine applies (and re-validates) the move; trick resolution and
        # scoring happen inside HandState.apply.
        state.apply(action_id)
        applied.append(action_id)
        # Stop checks, in priority order: the hand ending always stops.
        if state.result is not None:
            stop_reason = 'hand'
            break
        if until == 'trick' and len(state.completed) > start_tricks:
            stop_reason = 'trick'
            break
        if until == 'phase' and state.phase != start_phase:
            stop_reason = 'phase'
            break
        if until == 'turn' and state.seat == start_seat:
            stop_reason = 'turn'
            break
    if len(choices) > len(applied):
        raise ValueError('choices extend past the stopping point')
    return Rollout(tuple(applied), min(len(choices), len(applied)), probability, start_points,
                   tuple(state.points), stop_reason, state.result, state)


# ---------------------------------------------------------------------------
# 7. Decision support: expected points of each legal action over the belief
# ---------------------------------------------------------------------------
#
# Method (PIMC with rollouts, paired across actions):
#   for each world i (weight w_i):
#       rebuild its state; draw one rollout seed s_i
#       for each legal action a: value_i(a) = points the observer's team gains
#           after playing a, rolled out with seed s_i (+ leaf estimate if any)
#   estimate(a) = sum_i w_i * value_i(a)
#
# Using the same worlds AND the same rollout seed for every action is "common
# random numbers", the variance-reduction step that made backgammon rollouts
# practical. Differences between actions then reflect the actions, not luck
# in the deal or the rollout. The arena's duplicate deals use the same idea.
#
# Known limitation: PIMC assumes that after this move everyone acts as if the
# sampled world were known ("strategy fusion" / "non-locality"; Frank & Basin
# 1998, Long et al. 2010). Rollouts with ordinary policies, which see only
# their own observations, avoid the strongest form of this. Perfect-information
# leaf evaluators reintroduce it after the stopping point. Postponing that
# point (until="trick" instead of an immediate solve) is the EPIMC remedy
# (Arjonilla, Saffidine & Cazenave 2024).


@dataclass(frozen=True, slots=True)
class ActionEstimate:
    """Belief-weighted value of one legal action for the observer's team.

    expected_points        weighted mean of the objective value for the
                           observer's team: points gained from now until the
                           stopping point (+ leaf) by default, or signed marks
                           for objective='marks' (the name follows the default
                           objective; see ActionEvaluation.objective).
    std                    weighted standard deviation of that value across
                           worlds: how much the answer depends on the deal.
    standard_error         approximate error of expected_points as an estimate
                           (delta-method SE for self-normalized weights; equals
                           std / sqrt(n) for n uniform samples). In enumerated
                           mode with deterministic policies the mean is exact,
                           and this mostly reflects spread across worlds.
    advantage              weighted mean of value(this) - value(best) on the
                           SAME worlds and seeds (paired), so <= 0, and 0 for
                           the best action.
    advantage_standard_error  SE of that paired difference; usually much
                           smaller than standard_error thanks to pairing.
    expected_signed_marks  if every rollout finished the hand: weighted mean
                           signed marks for the observer's team, else None.
    contract_made_rate     if every rollout finished: weighted share in which
                           the bidding team made its contract, else None.
    """

    action_id: int
    expected_points: float
    std: float
    standard_error: float
    advantage: float
    advantage_standard_error: float
    expected_signed_marks: float | None
    contract_made_rate: float | None


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    """Result of evaluate_actions: one estimate per legal action, plus context.

    objective is 'points' or 'marks': the unit of every expected_points value.
    """

    observer: int
    estimates: tuple[ActionEstimate, ...]
    best_action_id: int
    world_count: int
    mode: str
    worlds_used: int
    effective_sample_size: float
    objective: str = 'points'


def _weighted_stats(values: Sequence[float], weights: Sequence[float]) -> tuple[float, float, float]:
    """(mean, std, standard error) for normalized weights.

    mean = sum w v; variance = sum w (v - mean)^2;
    SE = sqrt(sum w^2 (v - mean)^2), the delta-method standard error of a
    self-normalized weighted mean (Doucet et al.). With w = 1/n this is the
    familiar std / sqrt(n).
    """
    mean = sum(w * v for w, v in zip(weights, values))
    variance = sum(w * (v - mean) ** 2 for w, v in zip(weights, values))
    se = math.sqrt(sum(w * w * (v - mean) ** 2 for w, v in zip(weights, values)))
    return mean, math.sqrt(variance), se


def evaluate_actions(obs: Observation, *, rng: random.Random, belief: Belief | None = None,
                     rollout_policies: Sequence[AgentFactory] | None = None, until: str = 'hand',
                     leaf_evaluator: Callable[[HandState, int], float] | None = None,
                     samples: int = DEFAULT_SAMPLES, enumerate_below: int = DEFAULT_ENUMERATE_BELOW,
                     likelihood_policies: Sequence[AgentFactory] | None = None,
                     epsilon: float = DEFAULT_EPSILON, objective: str = 'points') -> ActionEvaluation:
    """Estimate the expected value of every legal action for the observer.

    Inputs:
      obs               the observer's ordinary Observation (any phase).
      rng               explicit stream: belief sampling, then one rollout
                        seed per world.
      belief            reuse a Belief from build_belief; otherwise one is
                        built with samples/enumerate_below/likelihood_policies/
                        epsilon.
      rollout_policies  four factories for the continuation (for example
                        hybrid.variant_factory("HHH") for each seat). None
                        means uniformly random continuations, which are
                        unbiased but weak, so expect noisy, pessimistic values.
      until             where each rollout stops ("trick", "phase", "hand", or
                        "turn": the observer's own next decision; see simulate).
      leaf_evaluator    optional f(state, observer_seat) -> the observer team's
                        remaining value from the stopping state, in objective
                        units: ADDITIONAL points (objective='points') or the
                        expected FINAL signed marks (objective='marks'; marks
                        are only awarded at hand end, so nothing has accrued
                        yet). Hooks: a perfect-information oracle, or a trained
                        value head (research.ppo.search.ValueLeaf). It receives
                        only hypothetical states. Ignored for rollouts that
                        finished the hand.
      objective         'points' (default): points gained by the observer's
                        team, its share of the remaining 42. Marks and
                        contract-made rates are then reported as diagnostics
                        when rollouts reach the end of the hand.
                        'marks': the observer team's signed marks for the hand
                        (+stake if it scores, -stake otherwise), the quantity
                        PPO value heads predict. Use it with a value-head leaf,
                        and for bid/trump decisions, where points ignore the
                        contract. Stopping before the hand ends then requires
                        a leaf_evaluator.
    Output: ActionEvaluation. ActionEstimate.expected_points holds the expected
    objective value (points, or signed marks for objective='marks');
    ActionEvaluation.objective records which. best_action_id maximizes it
    (ties go to the lowest action ID).
    """
    if objective not in ('points', 'marks'):
        raise ValueError("objective must be 'points' or 'marks'")
    if objective == 'marks' and until != 'hand' and leaf_evaluator is None:
        raise ValueError("objective='marks' with rollouts that stop early needs a leaf_evaluator")
    # Step 1: the belief (worlds + weights). Built here unless the caller
    # passes one in, for example to reuse it across several evaluations.
    if belief is None:
        belief = build_belief(obs, rng=rng, samples=samples, enumerate_below=enumerate_below,
                              likelihood_policies=likelihood_policies, epsilon=epsilon)
    elif belief.observer != obs.seat:
        raise ValueError('Belief belongs to a different observer')
    team = obs.seat % 2
    legal = tuple(obs.legal_action_ids)
    # Step 2: fill value tables, indexed [action index a][world index i].
    # values: observer-team points; marks/made: hand diagnostics (dropped to
    # None as soon as any rollout stops before the hand ends).
    values = [[0.0] * len(belief.worlds) for _ in legal]
    marks: list[list[float]] | None = [[0.0] * len(belief.worlds) for _ in legal]
    made: list[list[float]] | None = [[0.0] * len(belief.worlds) for _ in legal]
    for i, world in enumerate(belief.worlds):
        base = world_state(obs, world)
        rollout_seed = rng.getrandbits(64)  # Shared by every action: common random numbers.
        for a, _ in enumerate(legal):
            # choices=(a,) plays legal[a] first: world_state guarantees each
            # world has exactly the observation's legal IDs, so index a means
            # the same move in every world.
            rollout = simulate(base, rng=random.Random(rollout_seed), choices=(a,),
                               policies=rollout_policies, until=until, track_probability=False)
            if objective == 'points':
                value = float(rollout.points_delta[team])
            elif rollout.hand_result is not None:
                result = rollout.hand_result
                value = float(result.marks_awarded if result.scoring_team == team else -result.marks_awarded)
            else:
                value = 0.0  # Marks accrue only at hand end; the leaf supplies the estimate.
            if rollout.hand_result is None:
                if leaf_evaluator is not None:
                    value += float(leaf_evaluator(rollout.final_state, obs.seat))
                marks = made = None  # Some rollout stopped early: no hand diagnostics.
            elif marks is not None and made is not None:
                result = rollout.hand_result
                sign = 1 if result.scoring_team == team else -1
                marks[a][i] = sign * result.marks_awarded
                made[a][i] = 1.0 if result.scoring_team == result.bidder % 2 else 0.0
            values[a][i] = value
    # Step 3: aggregate with the belief weights. The best action maximizes the
    # weighted mean (ties: lowest ID). Advantages compare every action with the
    # best one world by world (paired), which cancels shared deal luck.
    weights = belief.weights
    means = [_weighted_stats(v, weights)[0] for v in values]
    best = max(range(len(legal)), key=lambda a: (means[a], -legal[a]))
    estimates = []
    for a, action_id in enumerate(legal):
        mean, std, se = _weighted_stats(values[a], weights)
        diffs = [x - y for x, y in zip(values[a], values[best])]
        advantage, _, advantage_se = _weighted_stats(diffs, weights)
        estimates.append(ActionEstimate(
            action_id, mean, std, se, advantage, advantage_se,
            None if marks is None else _weighted_stats(marks[a], weights)[0],
            None if made is None else _weighted_stats(made[a], weights)[0]))
    return ActionEvaluation(obs.seat, tuple(estimates), legal[best], belief.world_count,
                            belief.mode, len(belief.worlds), belief.effective_sample_size, objective)


__all__ = [
    'ActionEstimate', 'ActionEvaluation', 'Belief', 'HiddenConstraints', 'Rollout',
    'build_belief', 'count_worlds', 'enumerate_worlds', 'evaluate_actions', 'hidden_constraints',
    'policy_likelihood', 'sample_world', 'simulate', 'tile_marginals', 'world_state',
]
