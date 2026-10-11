"""Independent phase modules: frozen play cannot share a trainable bid trunk."""
import torch
from torch import nn
from .model import ActorCritic

PHASES = ('bid', 'trump', 'play')


def phases(value):
    values = tuple(value.split(',')) if isinstance(value, str) else tuple(value)
    if not values or len(set(values)) != len(values) or any(p not in PHASES for p in values):
        raise ValueError('Expected distinct phases from bid,trump,play')
    return tuple(p for p in PHASES if p in values)


class PhaseActorCritic(nn.Module):
    def __init__(self, hidden_size=256, neural_phases=('bid', 'play')):
        super().__init__()
        self.hidden_size = hidden_size
        self.neural_phases = phases(neural_phases)
        self.networks = nn.ModuleDict({p: ActorCritic(hidden_size) for p in PHASES})

    def configure_learning(self, learn_phases):
        enabled = phases(learn_phases)
        if not set(enabled) <= set(self.neural_phases):
            raise ValueError('Cannot train a phase configured as heuristic')
        for phase, module in self.networks.items():
            module.requires_grad_(phase in enabled)
            module.zero_grad(set_to_none=True)

    def forward(self, features):
        # The existing encoder's first three features are the phase one-hot.
        if features.ndim != 2 or not torch.all(features[:, :3].sum(-1) == 1):
            raise ValueError('Expected a batch of encoded phase observations')
        indices = features[:, :3].argmax(-1)
        logits = features.new_zeros((len(features), self.networks['play'].policy.out_features))
        values = features.new_zeros(len(features))
        for index, phase in enumerate(PHASES):
            selected = indices == index
            if selected.any():
                if phase not in self.neural_phases:
                    raise ValueError(f'{phase} uses a heuristic, not a neural policy')
                policy, value = self.networks[phase](features[selected])
                logits[selected], values[selected] = policy, value
        return logits, values

    # Reuse exactly the Stage-1 masking, sampling, and likelihood implementation.
    distribution = ActorCritic.distribution
    select = ActorCritic.select
    evaluate = ActorCritic.evaluate

    @classmethod
    def from_play(cls, model, learn_phases=('bid',)):
        enabled = phases(learn_phases)
        result = cls(model.hidden_size, tuple(p for p in PHASES if p == 'play' or p in enabled))
        result.networks['play'].load_state_dict(model.state_dict(), strict=True)
        # Transfer representation only; bidding/trump output rows were never trained.
        for phase in ('bid', 'trump'):
            result.networks[phase].trunk.load_state_dict(model.trunk.state_dict())
        result.trained_phases = ('play',)
        result.configure_learning(enabled)
        return result
