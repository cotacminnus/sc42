"""Centralized training critic over privileged features; never an actor input.

The main head predicts expected final signed marks (the PPO return target)
from the acting seat's team perspective under the rollout policies. The
separate auxiliary head predicts double-dummy final team points / 42 for the
same team; it is supervised only by certified oracle bounds.
"""
import torch
from torch import nn
from texas42.privileged import PRIVILEGED_SIZE

# none: policy-only PPO (Monte Carlo returns, zero baseline, no value loss);
# local: per-network value heads; centralized: the privileged critic below.
CRITIC_MODES = ('none', 'local', 'centralized')
RETURN_ESTIMATORS = ('gae', 'monte-carlo')
CRITIC_ARCHITECTURE = 'central-critic-v1'
ORACLE_POINT_SCALE = 42.0


class CentralCritic(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(nn.Linear(PRIVILEGED_SIZE, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.value = nn.Linear(hidden_size, 1)
        self.oracle = nn.Linear(hidden_size, 1)

    def forward(self, features):
        """Main signed-mark value only; the oracle head is evaluated separately."""
        return self.value(self.trunk(features)).squeeze(-1)

    def oracle_points(self, features):
        """Normalized double-dummy final points (points / 42) for the actor's team."""
        return self.oracle(self.trunk(features)).squeeze(-1)


def interval_loss(prediction, lower, upper):
    """relu(L - v)^2 + relu(v - U)^2, per sample; squared error when L == U."""
    if not torch.all(lower <= upper):
        raise ValueError('Interval lower bounds must not exceed upper bounds')
    return torch.relu(lower - prediction).square() + torch.relu(prediction - upper).square()


def new_critic(hidden_size, seed):
    """Fresh critic weights from their own seed, without perturbing global RNG."""
    from .seeds import stream_seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stream_seed(seed, 'central-critic-init-v1'))
        return CentralCritic(hidden_size)
