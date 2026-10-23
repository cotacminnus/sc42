"""Completed per-seat trajectories; discount time advances only on that seat's plays.

Perspective: every reward, value, return and advantage in one trajectory belongs
to that seat's partnership (signed marks). Trajectories never mix seats, so no
bootstrap ever crosses from one team's value to the other's. Other players'
turns are skipped rather than discounted, and phase changes (bid -> trump ->
play) are ordinary steps of the same seat trajectory.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Experience:
    observation_features: tuple[float, ...]
    legal_action_mask: tuple[bool, ...]
    action_id: int
    old_log_probability: float
    old_state_value: float
    reward: float
    done: bool
    seat: int
    episode_id: int
    # Privileged centralized-critic input; None for the local actor-critic.
    critic_features: tuple[float, ...] | None = None


def gae(rewards, values, gamma=1.0, gae_lambda=0.95, bootstrap_value=0.0):
    """One seat trajectory; no trick resets.

    A terminal trajectory bootstraps from zero. A truncated trajectory passes
    the same team's value estimate of the state after its last decision.
    """
    if len(rewards) != len(values) or not rewards:
        raise ValueError('Expected matching nonempty terminal trajectory')
    advantages = [0.0] * len(values)
    carry, next_value = 0.0, float(bootstrap_value)
    for i in reversed(range(len(values))):
        delta = rewards[i] + gamma * next_value - values[i]
        carry = delta + gamma * gae_lambda * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, [a + v for a, v in zip(advantages, values)]


def monte_carlo(rewards, values, gamma=1.0, bootstrap_value=0.0):
    """Diagnostic baseline: complete discounted returns, advantage = return - value.

    Equivalent to GAE with lambda = 1. With gamma = 1 every decision of a
    terminal hand trajectory has the final signed marks as its return.
    """
    if len(rewards) != len(values) or not rewards:
        raise ValueError('Expected matching nonempty terminal trajectory')
    returns, total = [0.0] * len(rewards), float(bootstrap_value)
    for i in reversed(range(len(rewards))):
        total = rewards[i] + gamma * total
        returns[i] = total
    return [g - v for g, v in zip(returns, values)], returns



class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.records, self.advantages, self.returns = [], [], []
        self._keys = set()

    def __len__(self):
        return len(self.records)

    def add_trajectory(self, trajectory, gamma=1.0, gae_lambda=0.95, *, estimator='gae',
                       bootstrap_value=None):
        """bootstrap_value=None requires a terminal last decision; a float marks truncation."""
        trajectory = tuple(trajectory)
        if not trajectory:
            raise ValueError('Empty trajectory')
        key = (trajectory[0].episode_id, trajectory[0].seat)
        if key in self._keys or any((r.episode_id, r.seat) != key for r in trajectory):
            raise ValueError('Duplicate or mixed seat/episode trajectory')
        if any(r.done for r in trajectory[:-1]) or trajectory[-1].done != (bootstrap_value is None):
            raise ValueError('Only the last seat decision may be terminal; truncation needs a bootstrap value')
        if len({r.critic_features is None for r in trajectory}) != 1:
            raise ValueError('Mixed local and centralized critic records')
        if any(type(r.action_id) is not int or not 0 <= r.action_id < len(r.legal_action_mask)
               or not r.legal_action_mask[r.action_id] for r in trajectory):
            raise ValueError('Illegal stored action')
        rewards, values = [r.reward for r in trajectory], [r.old_state_value for r in trajectory]
        bootstrap = 0.0 if bootstrap_value is None else bootstrap_value
        if estimator == 'gae':
            advantages, returns = gae(rewards, values, gamma, gae_lambda, bootstrap)
        elif estimator == 'monte-carlo':
            advantages, returns = monte_carlo(rewards, values, gamma, bootstrap)
        else:
            raise ValueError('estimator must be gae or monte-carlo')
        self._keys.add(key)
        self.records.extend(trajectory)
        self.advantages.extend(advantages)
        self.returns.extend(returns)

    def tensors(self):
        if not self.records:
            raise ValueError('Empty rollout')
        r = self.records
        central = r[0].critic_features is not None
        if any((x.critic_features is not None) != central for x in r):
            raise ValueError('Mixed local and centralized critic records')
        data = dict(features=torch.tensor([x.observation_features for x in r], dtype=torch.float32),
                    masks=torch.tensor([x.legal_action_mask for x in r], dtype=torch.bool),
                    actions=torch.tensor([x.action_id for x in r], dtype=torch.long),
                    old_log_probs=torch.tensor([x.old_log_probability for x in r]),
                    advantages=torch.tensor(self.advantages), returns=torch.tensor(self.returns))
        if central:
            data['critic_features'] = torch.tensor([x.critic_features for x in r], dtype=torch.float32)
        return data

    @staticmethod
    def minibatches(data, size, generator):
        order = torch.randperm(len(data['actions']), generator=generator)
        for start in range(0, len(order), size):
            yield {key: value[order[start:start + size]] for key, value in data.items()}
