"""Frozen CPU policy factories, prepared once per arena worker."""
from dataclasses import dataclass, field
import multiprocessing
from time import perf_counter
import torch
from texas42.agents.hybrid import variant_factory
from texas42.arena import compare_agents
from .agent import PPOAgent
from .rollout import snapshot, from_snapshot
from .opponents import evaluation_opponents


@dataclass
class PPOFactory:
    weights: bytes
    hidden_size: int
    model: object = field(default=None, init=False, repr=False)

    def __getstate__(self):
        # Even a previously prepared factory sends only the CPU weight payload.
        return dict(weights=self.weights, hidden_size=self.hidden_size, model=None)

    def prepare(self):
        if self.model is None:
            if multiprocessing.current_process().name != 'MainProcess' and torch.get_num_threads() != 1:
                torch.set_num_threads(1)
            self.model = from_snapshot(self.weights, self.hidden_size)

    def __call__(self, seat, seed):
        self.prepare()
        return PPOAgent(self.model, seed=seed, greedy=True)


def evaluate_many(model, *, opponents=('HHH',), deals=128, seed=12345,
                  bootstrap_seed=0, bootstrap_samples=200, threads=1, performance=None):
    """Separate labelled reports on identical schedules, sharing one CPU snapshot."""
    names = evaluation_opponents(opponents)
    previous_threads = torch.get_num_threads()
    if previous_threads != 1:
        torch.set_num_threads(1)
    started = perf_counter()
    try:
        factory = PPOFactory(snapshot(model), model.hidden_size)
        snapshot_seconds = perf_counter() - started
        reports = {}
        for opponent in names:
            opponent_started = perf_counter()
            telemetry = {}
            reports[opponent] = compare_agents(factory, variant_factory(opponent),
                num_deals=deals, seed=seed, bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples, threads=threads, performance=telemetry)
            telemetry.update(snapshot_seconds=snapshot_seconds,
                             evaluation_total_seconds=perf_counter() - opponent_started)
            if performance is not None:
                performance[opponent] = telemetry
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
