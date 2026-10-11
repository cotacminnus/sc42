"""Spawn-safe phase-aware factories with frozen checkpoint caching per process."""
from dataclasses import dataclass, field
from io import BytesIO
import hashlib
from time import perf_counter
import torch
from texas42.arena import compare_agents
from .agent import PPOAgent
from .multiphase import PhaseActorCritic
from .rollout import snapshot

_MODELS = {}


def frozen_model(spec):
    key = (spec.path, spec.sha256)
    if key not in _MODELS:
        with open(spec.path, 'rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != spec.sha256:
                raise ValueError(f'Frozen checkpoint changed: {spec.path}')
        from .checkpoint import load_model
        model, _ = load_model(spec.path, 'cpu')
        model.eval().requires_grad_(False)
        _MODELS[key] = model
    return _MODELS[key]


@dataclass
class FrozenFactory:
    spec: object

    def prepare(self):
        frozen_model(self.spec)

    def __call__(self, seat, seed):
        return PPOAgent(frozen_model(self.spec), seed=seed, greedy=True)


@dataclass
class PhaseFactory:
    weights: bytes
    hidden_size: int
    neural_phases: tuple
    model: object = field(default=None, init=False, repr=False)

    def __getstate__(self):
        return dict(weights=self.weights, hidden_size=self.hidden_size,
                    neural_phases=self.neural_phases, model=None)

    def prepare(self):
        if self.model is None:
            with torch.random.fork_rng(devices=[]):
                self.model = PhaseActorCritic(self.hidden_size, self.neural_phases)
            self.model.load_state_dict(torch.load(BytesIO(self.weights), weights_only=True, map_location='cpu'))
            self.model.eval().requires_grad_(False)

    def __call__(self, seat, seed):
        self.prepare()
        return PPOAgent(self.model, seed=seed, greedy=True)


def evaluate_phases(model, policies, *, deals=128, seed=12345, bootstrap_seed=0,
                    bootstrap_samples=200, threads=1, performance=None):
    """All opponents use the same paired deals; training RNG is untouched."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    factory = PhaseFactory(snapshot(model), model.hidden_size, model.neural_phases)
    try:
        reports = {}
        for spec in policies:
            telemetry, started = {}, perf_counter()
            reports[spec.name] = compare_agents(factory, spec.factory(), num_deals=deals,
                seed=seed, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
                threads=threads, performance=telemetry)
            telemetry['evaluation_total_seconds'] = perf_counter() - started
            if performance is not None:
                performance[spec.name] = telemetry
        return reports
    finally:
        torch.set_num_threads(previous)


def evaluation_diagnostics(report):
    """Resolved-contract diagnostics and all four public auction-call frequencies."""
    from examples.heuristic_benchmark import breakdown
    details = breakdown(report)
    for policy in ('A', 'B'):
        roles = {role: dict(hands=0, wins=0, signed_marks_sum=0) for role in ('declarer', 'defender')}
        calls = bids = 0
        for game in report.games:
            seats = tuple(s for s, name in enumerate(game.seating) if name == policy)
            for hand in game.hands:
                calls += len(seats)
                bids += sum(hand.bid_counts[s] for s in seats)
                role = 'declarer' if hand.seating[hand.result.bidder] == policy else 'defender'
                won = any(s % 2 == hand.result.scoring_team for s in seats)
                roles[role]['hands'] += 1
                roles[role]['wins'] += int(won)
                roles[role]['signed_marks_sum'] += hand.result.marks_awarded * (1 if won else -1)
        for stats in roles.values():
            n = stats['hands']
            stats['win_rate'] = stats['wins'] / n if n else None
            stats['mean_signed_marks'] = stats['signed_marks_sum'] / n if n else None
        details[policy].update(roles=roles, bid_calls=bids, pass_calls=calls - bids,
                              bid_frequency=bids / calls if calls else None,
                              hands_before_resolution=1 if calls else None)
    return details
