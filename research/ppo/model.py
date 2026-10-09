"""Shared actor-relative actor/critic with the engine's global action vocabulary."""
import torch
from torch import nn
from torch.distributions import Categorical
from texas42.encoding import FEATURE_SIZE
from texas42.actions import NUM_ACTIONS


class ActorCritic(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(nn.Linear(FEATURE_SIZE, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.policy = nn.Linear(hidden_size, NUM_ACTIONS)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, features):
        latent = self.trunk(features)
        return self.policy(latent), self.value(latent).squeeze(-1)

    def distribution(self, features, legal_mask):
        logits, values = self(features)
        if legal_mask.dtype != torch.bool or legal_mask.shape != logits.shape:
            raise ValueError('Expected a boolean legal mask matching logits')
        if not legal_mask.any(dim=-1).all():
            raise ValueError('Observation has no legal actions')
        return Categorical(logits=logits.masked_fill(~legal_mask, -torch.inf)), values

    def select(self, features, legal_mask, *, greedy=False, generator=None):
        distribution, values = self.distribution(features, legal_mask)
        actions = (distribution.logits.argmax(-1) if greedy else
                   torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1))
        return actions, distribution.log_prob(actions), values

    def evaluate(self, features, legal_mask, actions):
        distribution, values = self.distribution(features, legal_mask)
        return distribution.log_prob(actions), distribution.entropy(), values
