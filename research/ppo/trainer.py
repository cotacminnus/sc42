"""Synchronized hand-level PPO; the policy is fixed throughout each rollout."""
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from .buffer import RolloutBuffer
from .model import ActorCritic
from .opponents import DEFAULT_OPPONENTS, OpponentPool, evaluation_opponents, select_primary
from .rollout import stream_seed, collect_episode, collect_chunk, initialize_worker, snapshot, from_snapshot
from texas42.parallel import OrderedPool, resolve_workers
from time import perf_counter



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
    torch_threads: int = 1
    sampling_scheme: str = 'auto'
    train_opponents: tuple = DEFAULT_OPPONENTS
    eval_opponents: tuple = ('HHH',)
    selection_opponent: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'train_opponents', OpponentPool(self.train_opponents).probabilities)
        names = evaluation_opponents(self.eval_opponents)
        object.__setattr__(self, 'eval_opponents', names)
        object.__setattr__(self, 'selection_opponent', select_primary(names, self.selection_opponent))
        for name in ('total_hands', 'rollout_play_steps', 'ppo_epochs', 'minibatch_size',
                     'hidden_size', 'eval_every_hands', 'eval_deals', 'checkpoint_every_hands',
                     'bootstrap_samples', 'torch_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        resolve_workers(self.threads)
        if self.sampling_scheme not in ('auto', 'legacy-v1', 'episode-v1'):
            raise ValueError('sampling_scheme must be auto, legacy-v1 or episode-v1')
        if self.sampling_scheme == 'legacy-v1' and self.threads != 1:
            raise ValueError('legacy-v1 sampling requires threads=1')
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
        torch.set_num_threads(config.torch_threads)
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
        self.workers = resolve_workers(config.threads,
                                       min(config.total_hands, math.ceil(config.rollout_play_steps / 14)))
        self.sampling_scheme = (('legacy-v1' if config.threads == 1 else 'episode-v1')
                                if config.sampling_scheme == 'auto' else config.sampling_scheme)
        self.pool = None
        self.collection_seconds = self.optimization_seconds = 0.0
        self.worker_pids = []
        self.opponents = OpponentPool(config.train_opponents)
        self.opponent_metrics = {}
        self.opponent_assignments = []
        self.rollout_performance = {}

    def execution_metadata(self):
        return dict(requested_workers=self.config.threads, resolved_workers=self.workers,
                    backend='spawn' if self.workers > 1 else 'sequential',
                    sampling_scheme=self.sampling_scheme, torch_threads=self.config.torch_threads)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def collect(self):
        started = perf_counter()
        c = self.config
        count = min(math.ceil(c.rollout_play_steps / 14), c.total_hands - self.hands)
        if count <= 0:
            raise ValueError('Training hand budget exhausted')
        indices = tuple(range(self.hands, self.hands + count))
        self.model.eval()
        if self.sampling_scheme == 'legacy-v1':
            results = [collect_episode(self.model, self.sampling, c.seed, i, self.opponents) for i in indices]
            self.worker_pids = []
        else:
            self.synchronize()
            weights = snapshot(self.model)
            if self.workers == 1:
                # Same CPU inference/sampling as workers; restore the caller's thread count.
                previous_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                try:
                    model = from_snapshot(weights, c.hidden_size)
                    results = [collect_episode(model, torch.Generator().manual_seed(
                        stream_seed(c.seed, 'parallel-episode-v1', i)), c.seed, i, self.opponents) for i in indices]
                finally:
                    torch.set_num_threads(previous_threads)
                self.worker_pids = []
            else:
                if self.pool is None:
                    self.pool = OrderedPool(self.workers, initializer=initialize_worker)
                size = math.ceil(count / self.workers)
                tasks = [(weights, c.hidden_size, c.seed, indices[i:i + size], self.opponents)
                         for i in range(0, count, size)]
                try:
                    chunks = self.pool.map(collect_chunk, tasks)
                except BaseException:
                    self.close()
                    raise
                results = [row for rows, _ in chunks for row in rows]
                self.worker_pids = sorted({pid for _, pid in chunks})
        results.sort(key=lambda row: row[0])
        if [row[0] for row in results] != list(indices):
            raise RuntimeError('Rollout duplicated or omitted an episode')
        merging_started = perf_counter()
        buffer, rewards = RolloutBuffer(), []
        self.opponent_assignments = [(episode, opponent) for episode, _, _, opponent, _ in results]
        self.opponent_metrics = {name: dict(hands=0, reward_sum=0.0, mean_signed_reward=None)
                                 for name, _ in c.train_opponents}
        self.rollout_performance = {key: sum(row[4][key] for row in results)
                                    for key in ('assignment_seconds', 'agent_construction_seconds', 'simulation_seconds')}
        for _, paths, reward, opponent, _ in results:
            for path in paths:
                buffer.add_trajectory(path, c.gamma, c.gae_lambda)
            rewards.append(reward)
            stats = self.opponent_metrics[opponent]
            stats['hands'] += 1
            stats['reward_sum'] += reward
        for stats in self.opponent_metrics.values():
            if stats['hands']:
                stats['mean_signed_reward'] = stats['reward_sum'] / stats['hands']
        self.rollout_performance['merging_seconds'] = perf_counter() - merging_started
        if len(buffer) != count * 14:
            raise RuntimeError('Incorrect Stage-1 rollout size')
        self.hands += count
        self.samples += len(buffer)
        self.collection_seconds = perf_counter() - started
        return buffer, sum(rewards) / count

    def update(self, buffer):
        self.synchronize()
        started = perf_counter()
        c = self.config
        data = buffer.tensors()
        adv = data['advantages']
        data['advantages'] = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        self.model.train()
        totals, weight = {}, 0
        phase_totals, phase_weights = {}, {}
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
                if hasattr(self.model, 'neural_phases'):
                    for index, phase in enumerate(('bid', 'trump', 'play')):
                        selected = batch['features'][:, index] == 1
                        count = int(selected.sum())
                        if not count:
                            continue
                        d = delta[selected]
                        phase_values = dict(
                            policy_loss=clipped_objective(logp[selected], batch['old_log_probs'][selected],
                                batch['advantages'][selected], c.clip_epsilon).item(),
                            value_loss=(values[selected] - batch['returns'][selected]).square().mean().item(),
                            entropy=entropy[selected].mean().item(),
                            approx_kl=((d.exp() - 1) - d).mean().item(),
                            clip_fraction=((d.exp() - 1).abs() > c.clip_epsilon).float().mean().item())
                        phase_weights[phase] = phase_weights.get(phase, 0) + count
                        for key, v in phase_values.items():
                            name = phase + '_' + key
                            phase_totals[name] = phase_totals.get(name, 0.0) + count * v
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
        self.synchronize()
        self.optimization_seconds = perf_counter() - started
        result = {k: v / weight for k, v in totals.items()}
        result.update({k: v / phase_weights[k.split('_')[0]] for k, v in phase_totals.items()})
        return result
