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
        model = load_frozen(self.spec)[0]
        neural = getattr(model, 'neural_phases', ('play',))
        if not set(neural) <= set(getattr(model, 'trained_phases', ('play',))):
            raise ValueError(f'Frozen opponent {self.spec.path} contains untrained neural phases')

    def __call__(self, seat, seed):
        return PPOAgent(frozen_model(self.spec), seed=seed, greedy=True)
