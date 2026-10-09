"""Single-process hand-level PPO; the policy is fixed throughout each rollout."""
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import random
import numpy as np
import torch
from texas42.agents import HeuristicAgent
from texas42.encoding import encode
from texas42.selfplay import SelfPlayEnv
from .buffer import Experience, RolloutBuffer
from .model import ActorCritic


def stream_seed(seed, name, index=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}:{index}'.encode()).digest()[:8], 'big') % (2**63)


@dataclass(frozen=True)
class Config:
    total_hands: int = 100000
    rollout_play_steps: int = 4096
    ppo_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = .95
    clip_epsilon: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    hidden_size: int = 256
    seed: int = 42
    device: str = 'auto'
    eval_every_hands: int = 5000
    eval_deals: int = 128
    checkpoint_every_hands: int = 5000
    bootstrap_samples: int = 200
    target_kl: float = 0.0
    threads: int = 1

    def __post_init__(self):
        for name in ('total_hands', 'rollout_play_steps', 'ppo_epochs', 'minibatch_size',
                     'hidden_size', 'eval_every_hands', 'eval_deals', 'checkpoint_every_hands',
                     'bootstrap_samples', 'threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('learning_rate', 'max_grad_norm', 'clip_epsilon'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.clip_epsilon >= 1:
            raise ValueError('clip_epsilon must be less than one')
        for name in ('gamma', 'gae_lambda'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        for name in ('entropy_coef', 'value_coef', 'target_kl'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')


def clipped_objective(log_probs, old_log_probs, advantages, epsilon):
    ratio = (log_probs - old_log_probs.detach()).exp()
    advantages = advantages.detach()
    return -torch.minimum(ratio * advantages,
                          ratio.clamp(1 - epsilon, 1 + epsilon) * advantages).mean()


class Trainer:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(config.threads)
        random.seed(config.seed)
        np.random.seed(config.seed % 2**32)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        self.device = torch.device(device)
        # Strict CPU tests; GPU reproducibility remains build/hardware dependent.
        torch.use_deterministic_algorithms(device == 'cpu')
        self.model = ActorCritic(config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.sampling = torch.Generator(device=self.device).manual_seed(stream_seed(config.seed, 'actions'))
        self.shuffle = torch.Generator().manual_seed(stream_seed(config.seed, 'minibatches'))
        self.hands = self.samples = self.updates = self.optimizer_steps = 0
        self.best_score = -float('inf')
        self.last_eval = self.last_checkpoint = 0

    @torch.no_grad()
    def collect(self):
        c = self.config
        buffer, episode_rewards = RolloutBuffer(), []
        self.model.eval()
        while len(buffer) < c.rollout_play_steps and self.hands < c.total_hands:
            episode = self.hands
            # Alternating partnerships balances both orientations without consuming RNG.
            team = (episode + c.seed) % 2
            env = SelfPlayEnv(episode='hand', seed=stream_seed(c.seed, 'deal', episode))
            obs = env.reset()
            agents = [HeuristicAgent() for _ in range(4)]
            for seat, agent in enumerate(agents):
                agent.reset(seat)
            paths = {seat: [] for seat in range(4) if seat % 2 == team}
            decisions = 0
            while True:
                if obs.phase == 'play' and obs.seat in paths:
                    features = tuple(encode(obs))
                    x = torch.tensor([features], dtype=torch.float32, device=self.device)
                    mask = torch.tensor([obs.action_mask], dtype=torch.bool, device=self.device)
                    action, log_prob, value = self.model.select(x, mask, generator=self.sampling)
                    action = int(action.item())
                    paths[obs.seat].append(Experience(features, obs.action_mask, action,
                        float(log_prob.item()), float(value.item()), 0.0, False, obs.seat, episode))
                else:
                    action = agents[obs.seat].act(obs)
                transition = env.step(action)
                decisions += 1
                if transition.terminated:
                    break
                obs = transition.observation
            if decisions != 33 or sum(transition.hand_result.team_points) != 42:
                raise RuntimeError('Incomplete normal hand')
            for seat, path in paths.items():
                if len(path) != 7:
                    raise RuntimeError('Expected seven decisions per learning seat')
                path[-1] = replace(path[-1], reward=float(transition.rewards[seat]), done=True)
                buffer.add_trajectory(path, c.gamma, c.gae_lambda)
            episode_rewards.append(transition.rewards[team])
            self.hands += 1
        self.samples += len(buffer)
        return buffer, sum(episode_rewards) / len(episode_rewards)

    def update(self, buffer):
        c = self.config
        data = buffer.tensors()
        adv = data['advantages']
        data['advantages'] = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        self.model.train()
        totals, weight = {}, 0
        stop = False
        for _ in range(c.ppo_epochs):
            for batch in buffer.minibatches(data, c.minibatch_size, self.shuffle):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logp, entropy, values = self.model.evaluate(batch['features'], batch['masks'], batch['actions'])
                delta = logp - batch['old_log_probs']
                kl = ((delta.exp() - 1) - delta).mean()
                if c.target_kl and kl.item() > c.target_kl:
                    stop = True
                    break
                policy = clipped_objective(logp, batch['old_log_probs'], batch['advantages'], c.clip_epsilon)
                value = (values - batch['returns']).square().mean()
                loss = policy + c.value_coef * value - c.entropy_coef * entropy.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss')
                self.optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer_steps += 1
                metrics = dict(policy_loss=policy.item(), value_loss=value.item(), entropy=entropy.mean().item(),
                               approx_kl=kl.item(), clip_fraction=((delta.exp() - 1).abs() > c.clip_epsilon).float().mean().item(),
                               gradient_norm=norm.item())
                n = len(batch['actions'])
                for key, v in metrics.items():
                    if not math.isfinite(v):
                        raise FloatingPointError(f'Non-finite {key}')
                    totals[key] = totals.get(key, 0) + v * n
                weight += n
            if stop:
                break
        if not weight:
            raise RuntimeError('No PPO optimizer steps; old policy or KL configuration inconsistent')
        self.updates += 1
        buffer.clear()
        return {k: v / weight for k, v in totals.items()}
