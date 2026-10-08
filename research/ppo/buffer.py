"""Completed per-seat trajectories; discount time advances only on that seat's plays."""
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


def gae(rewards, values, gamma=1.0, gae_lambda=0.95):
    """One complete seat trajectory, terminal bootstrap zero; no trick resets."""
    if len(rewards) != len(values) or not rewards:
        raise ValueError('Expected matching nonempty terminal trajectory')
    advantages = [0.0] * len(values)
    carry = next_value = 0.0
    for i in reversed(range(len(values))):
        delta = rewards[i] + gamma * next_value - values[i]
        carry = delta + gamma * gae_lambda * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, [a + v for a, v in zip(advantages, values)]


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.records, self.advantages, self.returns = [], [], []
        self._keys = set()

    def __len__(self):
        return len(self.records)

    def add_trajectory(self, trajectory, gamma=1.0, gae_lambda=0.95):
        trajectory = tuple(trajectory)
        if not trajectory:
            raise ValueError('Empty trajectory')
        key = (trajectory[0].episode_id, trajectory[0].seat)
        if key in self._keys or any((r.episode_id, r.seat) != key for r in trajectory):
            raise ValueError('Duplicate or mixed seat/episode trajectory')
        if not trajectory[-1].done or any(r.done for r in trajectory[:-1]):
            raise ValueError('Only the last seat decision must be terminal')
        if any(type(r.action_id) is not int or not 0 <= r.action_id < len(r.legal_action_mask)
               or not r.legal_action_mask[r.action_id] for r in trajectory):
            raise ValueError('Illegal stored action')
        advantages, returns = gae([r.reward for r in trajectory],
                                  [r.old_state_value for r in trajectory], gamma, gae_lambda)
        self._keys.add(key)
        self.records.extend(trajectory)
        self.advantages.extend(advantages)
        self.returns.extend(returns)

    def tensors(self):
        if not self.records:
            raise ValueError('Empty rollout')
        r = self.records
        return dict(features=torch.tensor([x.observation_features for x in r], dtype=torch.float32),
                    masks=torch.tensor([x.legal_action_mask for x in r], dtype=torch.bool),
                    actions=torch.tensor([x.action_id for x in r], dtype=torch.long),
                    old_log_probs=torch.tensor([x.old_log_probability for x in r]),
                    advantages=torch.tensor(self.advantages), returns=torch.tensor(self.returns))

    @staticmethod
    def minibatches(data, size, generator):
        order = torch.randperm(len(data['actions']), generator=generator)
        for start in range(0, len(order), size):
            yield {key: value[order[start:start + size]] for key, value in data.items()}
