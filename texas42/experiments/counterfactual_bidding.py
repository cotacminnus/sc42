"""One voluntary bid/pass intervention per hand, replayed by the existing arena.

No engine hooks: wrappers call ordinary agents, verify replay, then substitute
one legal pass. Observations stay private; exports include only the selected
actor's hand, never the full deal or other actors' observations.
"""
from dataclasses import asdict, dataclass
import random
from statistics import median
from typing import Sequence

from ..actions import ACTION_REGISTRY, PLAY_ACTIONS
from ..agents import variant_factory
from ..arena import (AgentFactory, Deal, HandEvaluation, _positive, _seed,
                     generate_schedule, run_game)
from ..models import Bid, Domino
from ..rules import is_trump, suit_rank
from ..selfplay import Observation

PASS = ACTION_REGISTRY.encode('bid', None)


@dataclass(frozen=True)
class DecisionFeatures:
    hand: tuple[int, ...]
    candidate_trump: int | str
    suit_length: int
    doubles: int
    hand_counters: int
    suit_counters: int
    owns_highest_trump: bool
    current_bid: Bid | None
    current_bidder_team: int | None
    partner_leads: bool
    match_marks: tuple[int, int]
    target_marks: int


def decision_features(obs: Observation) -> DecisionFeatures:
    """Actor-only descriptors; suit ties use V1's count/rank/counter ordering.

    These are analysis labels, not a second policy or a new observation encoder.
    Candidate suits are numbered suits and doubles (Follow Me is not a suit).
    """
    hand = [PLAY_ACTIONS[i] for i in obs.view.hand]
    def quality(suit):
        tiles = [d for d in hand if is_trump(d, suit)]
        return (len(tiles), max((suit_rank(d, suit) for d in tiles), default=(0, 0)),
                sum(d.count_value for d in tiles))
    suit = max((*range(7), 'doubles'), key=quality)
    tiles = [d for d in hand if is_trump(d, suit)]
    highest = Domino(6, 6) if suit == 'doubles' else Domino(suit, suit)
    bidder = obs.view.current_bidder
    return DecisionFeatures(obs.view.hand, suit, len(tiles), sum(d.is_double for d in hand),
                            sum(d.count_value for d in hand), sum(d.count_value for d in tiles),
                            highest in hand, obs.view.current_bid,
                            None if bidder is None else bidder % 2,
                            bidder == (obs.seat + 2) % 4, obs.view.match_marks, obs.view.target_marks)


def signed_marks(hand: HandEvaluation, seat: int) -> int:
    result = hand.result
    return result.marks_awarded * (1 if result.scoring_team == seat % 2 else -1)


@dataclass(frozen=True)
class BranchOutcome:
    hand: HandEvaluation
    signed_marks: int
    acting_team_won: bool
    contract_success: bool
    # Public action sequence allows inspection without exporting other hands.
    action_ids: tuple[int, ...]


@dataclass(frozen=True)
class Intervention:
    deal_id: str
    group_id: str
    game_id: str
    acting_seat: int
    acting_policy: str
    shaker: int
    auction_position: int  # one-based
    original_bid: Bid
    original_action_id: int
    alternative_action_id: int
    forced: bool
    agent_seeds: tuple[int, ...]
    features: DecisionFeatures
    bid: BranchOutcome
    passed: BranchOutcome
    delta_marks: int
    same_winning_partnership: bool
    bid_replay_verified: bool


@dataclass(frozen=True)
class HandExperiment:
    deal_id: str
    group_id: str
    game_id: str
    seating: tuple[str, ...]
    eligible_decisions: int
    skipped_eligible_decisions: int
    intervention: Intervention | None


def evaluate_hand(factories: Sequence[AgentFactory], deal: Deal, *, seating: tuple[str, ...],
                  target_policy: str, seed: int = 0, target_marks: int = 7,
                  game_id: str = 'game', group_id: str | None = None) -> HandExperiment:
    """Original plus two fresh replays; selection is based only on original play.

    Call the policy even at the overridden decision, preserving RNG consumption
    and policy-side memory updates. A mismatched prefix or BID trajectory aborts
    the experiment. Factories must return fresh objects across seats AND branches.
    """
    if target_policy not in seating:
        raise ValueError('Target policy must occupy at least one seat')
    group_id = group_id or deal.deal_id
    retained_agents = []  # keep identities alive to reject cross-branch reuse

    def execute(expected=None, override=None):
        trace = []
        def wrap(factory):
            def create(seat, agent_seed):
                delegate = factory(seat, agent_seed)
                if any(delegate is old for old in retained_agents):
                    raise ValueError('Replay factories must create fresh agents')
                retained_agents.append(delegate)
                class RecordingAgent:
                    def reset(self, assigned):
                        delegate.reset(assigned)
                    def act(self, obs):
                        action = delegate.act(obs)
                        if type(action) is not int or action not in obs.legal_action_ids:
                            raise ValueError('Policy returned an illegal action')
                        index = len(trace)
                        if expected is not None and (override is None or index <= override):
                            if (obs, action) != expected[index]:
                                raise ValueError(f'Replay diverged at decision {index + 1}')
                        if index == override:
                            if obs.phase != 'bid' or obs.view.forced or PASS not in obs.legal_action_ids:
                                raise ValueError('Intervention requires a voluntary bidding opportunity')
                            if not isinstance(ACTION_REGISTRY[action].value, Bid):
                                raise ValueError('Intervention requires an original bid')
                            action = PASS
                        trace.append((obs, action))
                        return action
                return RecordingAgent()
            return create
        game = run_game(tuple(wrap(f) for f in factories), (deal,), seating=seating,
                        seed=seed, target_marks=target_marks, game_id=game_id, group_id=group_id)
        return game, trace

    original, baseline = execute()
    eligible = [i for i, (obs, action) in enumerate(baseline)
                if obs.phase == 'bid' and seating[obs.seat] == target_policy
                and PASS in obs.legal_action_ids and not obs.view.forced
                and isinstance(ACTION_REGISTRY[action].value, Bid)]
    if not eligible:
        return HandExperiment(deal.deal_id, group_id, game_id, tuple(seating), 0, 0, None)
    selected = eligible[0]
    obs, action = baseline[selected]
    bid_game, bid_trace = execute(expected=baseline)
    if bid_game != original or bid_trace != baseline:
        raise ValueError('BID branch did not reproduce the original hand')
    pass_game, pass_trace = execute(expected=baseline, override=selected)
    def outcome(game, trace):
        hand = game.hands[0]
        if hand.result.contract.kind != 'normal':
            raise ValueError('Only normal contracts are supported')
        return BranchOutcome(hand, signed_marks(hand, obs.seat),
                             hand.result.scoring_team == obs.seat % 2, hand.contract_success,
                             tuple(a for _, a in trace))
    bid, passed = outcome(bid_game, bid_trace), outcome(pass_game, pass_trace)
    intervention = Intervention(deal.deal_id, group_id, game_id, obs.seat, target_policy,
        deal.shaker, selected + 1, ACTION_REGISTRY[action].value, action, PASS, obs.view.forced,
        original.agent_seeds, decision_features(obs), bid, passed, bid.signed_marks - passed.signed_marks,
        bid.hand.result.scoring_team == passed.hand.result.scoring_team, True)
    return HandExperiment(deal.deal_id, group_id, game_id, tuple(seating), len(eligible),
                          len(eligible) - 1, intervention)


def bootstrap_delta(records, *, seed=0, samples=2000):
    """Percentile CI, resampling whole contributing duplicate groups.

    For conditional categories only groups containing that category contribute.
    Fewer than two contributing groups gives an unavailable interval.
    """
    _positive(samples, 'bootstrap_samples')
    groups = {}
    for row in records:
        groups.setdefault(row.group_id, []).append(row.delta_marks)
    if len(groups) < 2:
        return None
    groups = tuple(groups.values())
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        chosen = [v for _ in groups for v in rng.choice(groups)]
        values.append(sum(chosen) / len(chosen))
    values.sort()
    def quantile(p):
        x = (len(values) - 1) * p
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return {'low': quantile(.025), 'high': quantile(.975)}


def summarize(records, *, seed=0, samples=2000):
    records = tuple(records)
    n = len(records)
    deltas = [r.delta_marks for r in records]
    better = sum(v > 0 for v in deltas)
    worse = sum(v < 0 for v in deltas)
    equal = sum(v == 0 for v in deltas)
    return dict(samples=n, mean_delta_marks=sum(deltas)/n if n else None,
                median_delta_marks=median(deltas) if n else None,
                bid_better=better, pass_better=worse, equal=equal,
                fraction_bid_better=better/n if n else None,
                fraction_pass_better=worse/n if n else None, fraction_equal=equal/n if n else None,
                mean_bid_marks=sum(r.bid.signed_marks for r in records)/n if n else None,
                mean_pass_marks=sum(r.passed.signed_marks for r in records)/n if n else None,
                mean_delta_ci95=bootstrap_delta(records, seed=seed, samples=samples))


@dataclass(frozen=True)
class CounterfactualReport:
    policy_a: str
    policy_b: str
    target_policy: str
    seed: int
    bootstrap_seed: int
    bootstrap_samples: int
    target_marks: int
    hands: tuple[HandExperiment, ...]

    def to_dict(self):
        rows = [h.intervention for h in self.hands if h.intervention is not None]
        def stats(items):
            return summarize(items, seed=self.bootstrap_seed, samples=self.bootstrap_samples)
        def grouped(key):
            groups = {}
            for row in rows:
                groups.setdefault(str(key(row)), []).append(row)
            return {k: stats(v) for k, v in sorted(groups.items())}
        return {**asdict(self), 'episode': 'hand',
                'total_hands': len(self.hands), 'eligible_hands': len(rows),
                'ineligible_hands': len(self.hands) - len(rows), 'selected_interventions': len(rows),
                'skipped_eligible_decisions': sum(h.skipped_eligible_decisions for h in self.hands),
                'overall': stats(rows),
                'by_bid': grouped(lambda r: ACTION_REGISTRY[r.original_action_id].key),
                'by_hand_strength': {name: grouped(lambda r, n=name: getattr(r.features, n))
                    for name in ('suit_length', 'doubles', 'suit_counters', 'owns_highest_trump', 'partner_leads')}}


def compare_bidding(policy_a='HHH', policy_b='CHH', *, target_policy='HHH', num_deals=128,
                    seed=42, episode='hand', target_marks=7, bootstrap_seed=0,
                    bootstrap_samples=2000):
    """Use exactly the arena's duplicate schedule and absolute-seat seed scheme."""
    if episode != 'hand':
        raise ValueError('Counterfactual bidding currently supports hand episodes only')
    _positive(num_deals, 'num_deals')
    _positive(target_marks, 'target_marks')
    _positive(bootstrap_samples, 'bootstrap_samples')
    _seed(bootstrap_seed)  # exact Python integer validation, consistent with arena
    if target_policy not in (policy_a, policy_b):
        raise ValueError('Target policy must be policy_a or policy_b')
    a, b = variant_factory(policy_a), variant_factory(policy_b)
    hands = []
    for i in range(num_deals):
        deal = generate_schedule(1, _seed(seed, 'schedule', i), shaker=i % 4)[0]
        for suffix, names, factories in (
                ('A', (policy_a, policy_b) * 2, (a, b) * 2),
                ('B', (policy_b, policy_a) * 2, (b, a) * 2)):
            hands.append(evaluate_hand(factories, deal, seating=names, target_policy=target_policy,
                seed=_seed(seed, 'episode', i), target_marks=target_marks,
                game_id=f'{i}:{suffix}', group_id=f'{i}:{deal.deal_id}'))
    return CounterfactualReport(policy_a, policy_b, target_policy, seed, bootstrap_seed,
                                bootstrap_samples, target_marks, tuple(hands))
