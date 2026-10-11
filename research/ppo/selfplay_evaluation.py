"""Compatibility APIs for phase-aware validation and frozen historical policies."""
from dataclasses import dataclass
from .agent import PPOAgent
from .diagnostics import evaluation_diagnostics
from .checkpoint import load_frozen
from .evaluation import PPOFactory as PhaseFactory, evaluate_policies as evaluate_phases


def frozen_model(spec):
    return load_frozen(spec)[0]


@dataclass
class FrozenFactory:
    spec: object

    def prepare(self):
        frozen_model(self.spec)

    def __call__(self, seat, seed):
        return PPOAgent(frozen_model(self.spec), seed=seed, greedy=True)
