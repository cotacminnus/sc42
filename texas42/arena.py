"""Reproducible ordinary-policy evaluation; all transitions belong to EpisodeState."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Callable, Sequence

from .actions import ACTION_REGISTRY, PLAY_ACTIONS
from .agents import Agent
from .episodes import EpisodeState
from .models import ContractKind
from .selfplay import observation_from_view
from .state import HandResult

AgentFactory = Callable[[int, int], Agent]
Seating = tuple[str, str, str, str]


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _seed(seed: int, *parts: object) -> int:
    """Domain-separated streams, independent of Python's randomized hash()."""
    if type(seed) is not int:
        raise ValueError("seed must be a Python integer")
    return int.from_bytes(hashlib.sha256(repr((seed, *parts)).encode()).digest(), 'big')


@dataclass(frozen=True, slots=True)
class Deal:
    """Immutable absolute-seat hands, containing permanent domino IDs."""
    hands: tuple[tuple[int, ...], ...]
    shaker: int

    def __post_init__(self):
        hands = tuple(tuple(hand) for hand in self.hands)
        ids = [value for hand in hands for value in hand]
        if (len(hands) != 4 or any(len(hand) != 7 for hand in hands)
                or any(type(value) is not int for value in ids)
                or sorted(ids) != list(range(28))):
            raise ValueError("A deal requires exactly the 28 unique domino IDs in four hands")
        if type(self.shaker) is not int or self.shaker not in range(4):
            raise ValueError("shaker must be an integer seat")
        object.__setattr__(self, 'hands', tuple(tuple(sorted(hand)) for hand in hands))

    @property
    def deal_id(self) -> str:
        return hashlib.sha256(repr((self.hands, self.shaker)).encode()).hexdigest()


def generate_schedule(num_hands: int, seed: int, *, shaker: int = 0) -> tuple[Deal, ...]:
    """Precompute deals without consulting agents; rotate shaker each hand."""
    _positive(num_hands, 'num_hands')
    if type(shaker) is not int or shaker not in range(4):
        raise ValueError("shaker must be an integer seat")
    rng = random.Random(_seed(seed, 'deals'))
    deals = []
    for index in range(num_hands):
        deck = list(range(28))
        rng.shuffle(deck)
        deals.append(Deal(tuple(tuple(deck[seat * 7:(seat + 1) * 7])
                                for seat in range(4)), (shaker + index) % 4))
    return tuple(deals)


@dataclass(frozen=True, slots=True)
class HandEvaluation:
    deal_id: str
    seating: Seating
    shaker: int
    result: HandResult
    decisions: int
    bid_counts: tuple[int, int, int, int]
    rewards: tuple[int, int, int, int]
    # Origin of the resolved contract, not the shaker identity or final value.
    forced_bid: bool = False
    # None supports older records and contracts with different winning conditions.
    contract_decided_at_trick: int | None = None
    # Only queried for a forced auction call; None means unavailable/inapplicable.
    would_pass_if_allowed: bool | None = None

    @property
    def contract_success(self) -> bool:
        return self.result.scoring_team == self.result.bidder % 2


@dataclass(frozen=True, slots=True)
class GameEvaluation:
    game_id: str
    group_id: str
    episode: str
    target_marks: int
    seating: Seating
    agent_seeds: tuple[int, ...]
    schedule_ids: tuple[str, ...]
    hands: tuple[HandEvaluation, ...]
    winner: int
    final_marks: tuple[int, int]
    rewards: tuple[int, int, int, int]

    @property
    def num_hands(self) -> int:
        return len(self.hands)


class AgentExecutionError(RuntimeError):
    """Factory/reset/decision failure with context; the failed game is abandoned."""


def contract_decided(declaring_points: int, defending_points: int, target: int) -> bool:
    """Normal-contract certainty from cumulative awarded points, after a trick."""
    return declaring_points >= target or defending_points > 42 - target


def run_game(factories: Sequence[AgentFactory], schedule: Sequence[Deal], *,
             seating: Seating = ('seat0', 'seat1', 'seat2', 'seat3'),
             seed: int = 0, episode: str = 'hand', target_marks: int = 7,
             game_id: str = 'game', group_id: str | None = None) -> GameEvaluation:
    """Run one episode. Factories must return fresh agents; never retry actions.

    Match schedules must cover the maximum possible length, 2*target_marks-1.
    Only immutable actor observations cross the ordinary-policy boundary.
    """
    schedule, factories, seating = tuple(schedule), tuple(factories), tuple(seating)
    if len(factories) != 4 or len(seating) != 4 or any(type(p) is not str or not p for p in seating):
        raise ValueError("Provide four factories and four nonempty policy labels")
    _positive(target_marks, 'target_marks')
    if episode not in ('hand', 'match'):
        raise ValueError("episode must be hand or match")
    needed = 1 if episode == 'hand' else 2 * target_marks - 1
    if len(schedule) < needed or any(not isinstance(d, Deal) for d in schedule):
        raise ValueError(f"Supply at least {needed} validated deals")
    if episode == 'match' and any(d.shaker != (schedule[0].shaker + i) % 4
                                 for i, d in enumerate(schedule)):
        raise ValueError("Match schedule must follow official shaker rotation")
    state = EpisodeState(episode, target_marks, schedule[0].shaker)
    seeds = tuple(_seed(seed, 'agent', seat) for seat in range(4))
    agents = []
    for seat, factory in enumerate(factories):
        try:
            agent = factory(seat, seeds[seat])
            if any(agent is other for other in agents):
                raise ValueError("Factories must return independent agent instances")
            agents.append(agent)
        except Exception as exc:
            raise AgentExecutionError(f"game={game_id} hand=0 seat={seat} phase=factory") from exc
    for seat, agent in enumerate(agents):
        try:
            agent.reset(seat)
        except Exception as exc:
            raise AgentExecutionError(f"game={game_id} hand=0 seat={seat} phase=reset") from exc
    hands = []
    for hand_index, deal in enumerate(schedule):
        state.deal([[PLAY_ACTIONS[value] for value in hand] for hand in deal.hands])
        decisions, bids = 0, [0] * 4
        bid_origins = {}
        bid_preferences = {}
        decided_at = None
        while state.hand.result is None:
            obs = observation_from_view(state.hand.view())
            try:
                preference = None
                if obs.phase == 'bid' and obs.view.forced:
                    diagnostic = getattr(agents[obs.seat], 'would_pass_if_allowed', None)
                    if diagnostic is not None:
                        preference = diagnostic(obs)
                        if preference is not None and type(preference) is not bool:
                            raise ValueError('Bid preference must be bool or None')
                action_id = agents[obs.seat].act(obs)
                if type(action_id) is not int or action_id not in obs.legal_action_ids:
                    raise ValueError(f"Illegal {obs.phase} action ID: {action_id!r}")
            except Exception as exc:
                raise AgentExecutionError(
                    f"game={game_id} hand={hand_index} seat={obs.seat} phase={obs.phase}") from exc
            if obs.phase == 'bid' and ACTION_REGISTRY[action_id].value is not None:
                bids[obs.seat] += 1
                # Snapshot the engine flag before applying this call. Resolve by
                # bidder AND bid so an overbid cannot inherit an earlier origin.
                bid_origins[(obs.seat, ACTION_REGISTRY[action_id].value)] = obs.view.forced
                bid_preferences[(obs.seat, ACTION_REGISTRY[action_id].value)] = preference
            completed_before = len(state.hand.completed)
            transition = state.apply(action_id)
            # Only completed tricks award points. Never inspect a next observation:
            # the last trick may also have terminated the episode.
            hand = state.hand
            if decided_at is None and len(hand.completed) > completed_before:
                contract = hand.contract
                if contract.kind == ContractKind.NORMAL:
                    team = contract.bidder % 2
                    if contract_decided(hand.points[team], hand.points[1 - team],
                                        contract.bid.target_points):
                        decided_at = len(hand.completed)
            decisions += 1
        hands.append(HandEvaluation(deal.deal_id, seating, deal.shaker, transition.hand_result,
                                    decisions, tuple(bids), transition.rewards,
                                    forced_bid=bid_origins[(transition.hand_result.bidder,
                                                            transition.hand_result.bid)],
                                    contract_decided_at_trick=decided_at,
                                    would_pass_if_allowed=bid_preferences[(transition.hand_result.bidder,
                                                                          transition.hand_result.bid)]))
        if transition.terminated:
            winner = (transition.hand_result.scoring_team if episode == 'hand'
                      else transition.match_winner)
            return GameEvaluation(game_id, group_id or schedule[0].deal_id, episode, target_marks, seating,
                                  seeds, tuple(d.deal_id for d in schedule), tuple(hands),
                                  winner, tuple(state.marks), transition.rewards)
    raise RuntimeError("Schedule exhausted before termination")  # defensive: base rules bound length


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    policy: str
    games_played: int
    win_rate: float | None
    mean_signed_marks: float | None
    contracts_declared: int
    contract_success_rate: float | None
    mean_contract_value: float | None
    bidding_frequency: float | None


def _mean(values):
    return sum(values) / len(values) if values else None


def policy_metrics(games: Sequence[GameEvaluation], policy: str) -> PolicyMetrics:
    """Seat-averaged outcomes; marks weighted by played hands, bids by seat turns."""
    wins, marks, contracts, bids, turns = [], [], [], 0, 0
    for game in games:
        seats = [s for s, label in enumerate(game.seating) if label == policy]
        if not seats:
            continue
        wins.append(sum(s % 2 == game.winner for s in seats) / len(seats))
        for hand in game.hands:
            result = hand.result
            marks.append(sum(result.marks_awarded * (1 if s % 2 == result.scoring_team else -1)
                             for s in seats) / len(seats))
            bids += sum(hand.bid_counts[s] for s in seats)
            turns += len(seats)
            if result.bidder in seats:
                contracts.append(hand)
    return PolicyMetrics(policy, len(wins), _mean(wins), _mean(marks), len(contracts),
                         _mean([h.contract_success for h in contracts]),
                         _mean([h.result.bid.value_in_marks for h in contracts]),
                         bids / turns if turns else None)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class PolicyIntervals:
    policy: str
    win_rate: ConfidenceInterval | None
    mean_signed_marks: ConfidenceInterval | None


def bootstrap(games: Sequence[GameEvaluation], policy: str, *, seed: int = 0,
              samples: int = 2000) -> PolicyIntervals:
    """Percentile 95% intervals, resampling complete schedule/deal groups."""
    _positive(samples, 'samples')
    groups = {}
    for game in games:
        groups.setdefault(game.group_id, []).append(game)
    if len(groups) < 2:
        return PolicyIntervals(policy, None, None)
    groups = tuple(groups.values())
    rng = random.Random(_seed(seed, 'bootstrap'))
    wins, marks = [], []
    for _ in range(samples):
        selected = [game for _ in groups for game in rng.choice(groups)]
        metrics = policy_metrics(selected, policy)
        if metrics.win_rate is None:  # an undefined replicate cannot support an interval
            return PolicyIntervals(policy, None, None)
        wins.append(metrics.win_rate)
        marks.append(metrics.mean_signed_marks)

    def interval(values):
        values.sort()
        def quantile(p):
            index = (len(values) - 1) * p
            lower = int(index)
            upper = min(lower + 1, len(values) - 1)
            return values[lower] + (values[upper] - values[lower]) * (index - lower)
        return ConfidenceInterval(quantile(.025), quantile(.975))
    return PolicyIntervals(policy, interval(wins), interval(marks))


@dataclass(frozen=True, slots=True)
class ArenaReport:
    seed: int
    bootstrap_seed: int
    bootstrap_samples: int
    games: tuple[GameEvaluation, ...]
    metrics: tuple[PolicyMetrics, ...]
    intervals: tuple[PolicyIntervals, ...]

    @property
    def games_played(self) -> int:
        return len(self.games)

    @property
    def average_match_length(self) -> float | None:
        return _mean([g.num_hands for g in self.games if g.episode == 'match'])

    @property
    def contract_success_rate(self) -> float | None:
        return _mean([h.contract_success for g in self.games for h in g.hands])

    def to_dict(self) -> dict:
        """JSON-ready nested values; hand rows can also be flattened for CSV."""
        data = asdict(self)
        for game, row in zip(self.games, data['games']):
            row['num_hands'] = game.num_hands
            for hand, hand_row in zip(game.hands, row['hands']):
                hand_row['contract_success'] = hand.contract_success
        return {**data, 'games_played': self.games_played,
                'average_match_length': self.average_match_length,
                'contract_success_rate': self.contract_success_rate}

    def summary(self) -> str:
        lines = [f'{len({g.group_id for g in self.games})} duplicate groups; {self.games_played} games']
        for metric, ci in zip(self.metrics, self.intervals):
            def display(value, bounds):
                if value is None:
                    return 'unavailable'
                suffix = (f' (95% CI {bounds.low:.3f}, {bounds.high:.3f})' if bounds else ' (CI unavailable)')
                return f'{value:.3f}' + suffix
            lines.append(f'{metric.policy}: win rate {display(metric.win_rate, ci.win_rate)}; '
                         f'marks/hand {display(metric.mean_signed_marks, ci.mean_signed_marks)}')
        return '\n'.join(lines)


def compare_agents(policy_a: AgentFactory, policy_b: AgentFactory, *, num_deals: int = 100,
                   seed: int = 42, episode: str = 'hand', target_marks: int = 7,
                   bootstrap_seed: int = 0, bootstrap_samples: int = 2000) -> ArenaReport:
    """A/B homogeneous partnerships, swapped on every fixed absolute-seat schedule.

    In match mode num_deals counts paired match schedules, not individual hands.
    Agent streams are matched by absolute seat within a pair, never by policy.
    """
    _positive(num_deals, 'num_deals')
    _positive(target_marks, 'target_marks')
    _positive(bootstrap_samples, 'bootstrap_samples')
    if episode not in ('hand', 'match'):
        raise ValueError('episode must be hand or match')
    _seed(bootstrap_seed)
    schedules = tuple(generate_schedule(1 if episode == 'hand' else 2 * target_marks - 1,
                                       _seed(seed, 'schedule', i), shaker=i % 4)
                      for i in range(num_deals))
    games = []
    for index, schedule in enumerate(schedules):
        group = f'{index}:{schedule[0].deal_id}'
        for labels, factories in ((('A', 'B', 'A', 'B'), (policy_a, policy_b, policy_a, policy_b)),
                                  (('B', 'A', 'B', 'A'), (policy_b, policy_a, policy_b, policy_a))):
            games.append(run_game(factories, schedule, seating=labels,
                                  seed=_seed(seed, 'episode', index), episode=episode,
                                  target_marks=target_marks, game_id=f'{index}:{labels[0]}', group_id=group))
    games = tuple(games)
    return ArenaReport(seed, bootstrap_seed, bootstrap_samples, games,
                       tuple(policy_metrics(games, p) for p in ('A', 'B')),
                       tuple(bootstrap(games, p, seed=bootstrap_seed, samples=bootstrap_samples)
                             for p in ('A', 'B')))
