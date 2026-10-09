"""Ordinary policy adapter: heuristic auction and trump, neural trick play."""
import torch
from texas42.agents import HeuristicAgent
from texas42.encoding import encode


class PPOAgent:
    def __init__(self, model, *, seed=0, greedy=True):
        self.model, self.greedy = model, greedy
        self.heuristic = HeuristicAgent()
        self.seat = None
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed % (2**63))

    def reset(self, seat):
        self.heuristic.reset(seat)
        self.seat = seat

    def would_pass_if_allowed(self, obs):
        return self.heuristic.would_pass_if_allowed(obs)

    @torch.no_grad()
    def act(self, obs):
        if obs.seat != self.seat:
            raise RuntimeError('Reset for the acting seat before use')
        if obs.phase != 'play':
            return self.heuristic.act(obs)
        features = torch.tensor([encode(obs)], dtype=torch.float32, device=self.device)
        masks = torch.tensor([obs.action_mask], dtype=torch.bool, device=self.device)
        actions, _, _ = self.model.select(features, masks, greedy=self.greedy, generator=self.generator)
        action = int(actions.item())
        if action not in obs.legal_action_ids:
            raise ValueError('Neural agent selected an illegal action')
        return action
