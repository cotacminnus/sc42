"""Frozen CPU policy factories, prepared once per arena worker."""
from dataclasses import dataclass, field
import multiprocessing
from time import perf_counter
import torch
from .agent import PPOAgent
from .rollout import snapshot, from_snapshot
from .opponents import evaluation_opponents, PolicySpec
from .composition import PairedAgentFactory, evaluate_matchups


@dataclass
class PPOFactory:
    weights: bytes
    hidden_size: int
    neural_phases: tuple | None = None
    model: object = field(default=None, init=False, repr=False)

    def __getstate__(self):
        # Even a previously prepared factory sends only the CPU weight payload.
        return dict(weights=self.weights, hidden_size=self.hidden_size,
                    neural_phases=self.neural_phases, model=None)

    def prepare(self):
        if self.model is None:
            if multiprocessing.current_process().name != 'MainProcess' and torch.get_num_threads() != 1:
                torch.set_num_threads(1)
            self.model = from_snapshot(self.weights, self.hidden_size, self.neural_phases)
            self.model.requires_grad_(False)

    def __call__(self, seat, seed):
        self.prepare()
        return PPOAgent(self.model, seed=seed, greedy=True)


def evaluate_many(model, *, opponents=('HHH',), deals=128, seed=12345,
                  bootstrap_seed=0, bootstrap_samples=200, threads=1, performance=None):
    """Separate labelled reports on identical schedules, sharing one CPU snapshot."""
    names = evaluation_opponents(opponents)
    return evaluate_policies(model, tuple(PolicySpec(n) for n in names), deals=deals, seed=seed,
        bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
        threads=threads, performance=performance)


def evaluate_policies(model, policies, *, deals=128, seed=12345, bootstrap_seed=0,
                      bootstrap_samples=200, threads=1, performance=None, partner_model=None):
    """Validate either in-memory architecture against variants or frozen archives.

    Preserve PPOAgent inference for the legacy APIs; per-phase composition has
    its own explicit routing and RNG scheme. Both use the same arena execution.
    partner_model (paired-partner training) plays the team's second seat;
    without it, `model` plays both seats as before.
    """
    previous_threads = torch.get_num_threads()
    if previous_threads != 1:
        torch.set_num_threads(1)
    started = perf_counter()
    try:
        neural_phases = getattr(model, 'neural_phases', None)
        factory = PPOFactory(snapshot(model), model.hidden_size, neural_phases)
        if partner_model is not None:
            factory = PairedAgentFactory(factory, PPOFactory(
                snapshot(partner_model), partner_model.hidden_size,
                getattr(partner_model, 'neural_phases', None)))
        snapshot_seconds = perf_counter() - started
        telemetry = {}
        reports = evaluate_matchups(factory, ((s.name, s.factory()) for s in policies),
            num_deals=deals, seed=seed, bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples, threads=threads, performance=telemetry)
        if neural_phases is None:
            for values in telemetry.values():
                values['snapshot_seconds'] = snapshot_seconds
        if performance is not None:
            performance.update(telemetry)
        return reports
    finally:
        if previous_threads != 1:
            torch.set_num_threads(previous_threads)


def evaluate(model, *, opponent='HHH', deals=128, seed=12345, bootstrap_seed=0,
             bootstrap_samples=200, threads=1, performance=None):
    """Compatible single-opponent API; ordinary agents still receive only observations."""
    telemetry = {}
    report = evaluate_many(model, opponents=(opponent,), deals=deals, seed=seed,
                           bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
                           threads=threads, performance=telemetry)[opponent]
    if performance is not None:
        performance.update(telemetry[opponent])
    return report
