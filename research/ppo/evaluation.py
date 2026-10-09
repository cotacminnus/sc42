"""Frozen CPU policy factories, prepared once per arena worker."""
from dataclasses import dataclass, field
import multiprocessing
import torch
from texas42.agents.hybrid import variant_factory
from texas42.arena import compare_agents
from .agent import PPOAgent
from .rollout import snapshot, from_snapshot


@dataclass
class PPOFactory:
    weights: bytes
    hidden_size: int
    model: object = field(default=None, init=False, repr=False)

    def __getstate__(self):
        # Even a previously prepared factory sends only the CPU weight payload.
        return dict(weights=self.weights, hidden_size=self.hidden_size, model=None)

    def prepare(self):
        if multiprocessing.current_process().name != 'MainProcess':
            torch.set_num_threads(1)
        if self.model is None:
            self.model = from_snapshot(self.weights, self.hidden_size)

    def __call__(self, seat, seed):
        self.prepare()
        return PPOAgent(self.model, seed=seed, greedy=True)


def evaluate(model, *, opponent='HHH', deals=128, seed=12345, bootstrap_seed=0,
             bootstrap_samples=200, threads=1, performance=None):
    # Evaluation uses identical CPU numerical threading in sequential and worker
    # paths. Restore the optimizer's main-process thread setting afterward.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return compare_agents(PPOFactory(snapshot(model), model.hidden_size), variant_factory(opponent),
                              num_deals=deals, seed=seed, bootstrap_seed=bootstrap_seed,
                              bootstrap_samples=bootstrap_samples, threads=threads, performance=performance)
    finally:
        torch.set_num_threads(previous_threads)
