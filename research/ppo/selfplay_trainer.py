"""Multi-phase collection atop the unchanged clipped PPO optimizer and seat GAE."""
from dataclasses import dataclass
import math
from time import perf_counter
import torch
from texas42.parallel import OrderedPool, resolve_workers
from .trainer import Config, Trainer
from .multiphase import PhaseActorCritic, phases
from .opponents import HistoricalPool, PolicySpec
from .buffer import RolloutBuffer
from .rollout import initialize_worker, snapshot
from .selfplay_rollout import phase_chunk


@dataclass(frozen=True)
class SelfPlayConfig(Config):
    learn_phases: tuple = ('bid',)
    rollout_decisions: int = 1024
    opponent_probabilities: tuple = (('HHH', '1/2'), ('anchor', '1/2'))
    opponent_policies: tuple = ()
    anchor: PolicySpec | None = None
    evaluation_names: tuple = ('HHH', 'CHH', 'anchor')
    primary_opponent: str = 'anchor'
    snapshot_every_hands: int = 32768
    snapshot_pool_size: int = 8
    bid_bc_hands: int = 0
    forgetting_hands: int = 8

    def __post_init__(self):
        super().__post_init__()
        enabled = phases(self.learn_phases)
        if not {'bid', 'play'} & set(enabled):
            raise ValueError('Trump-only training is not supported; include bid or play')
        object.__setattr__(self, 'learn_phases', enabled)
        if self.anchor is None or self.anchor.kind != 'checkpoint':
            raise ValueError('An immutable original play checkpoint anchor is required')
        pool = HistoricalPool(self.opponent_probabilities, self.opponent_policies, self.anchor)
        object.__setattr__(self, 'opponent_probabilities', pool.probabilities)
        names = {p.name for p in self.opponent_policies}
        if (not self.evaluation_names or len(set(self.evaluation_names)) != len(self.evaluation_names)
                or not set(self.evaluation_names) <= names or self.primary_opponent not in self.evaluation_names):
            raise ValueError('Evaluation names and primary opponent must exist in frozen policy registry')
        for key in ('rollout_decisions', 'snapshot_every_hands', 'snapshot_pool_size', 'forgetting_hands'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if type(self.bid_bc_hands) is not int or self.bid_bc_hands < 0:
            raise ValueError('bid_bc_hands must be nonnegative')
        if self.sampling_scheme not in ('auto', 'episode-v1'):
            raise ValueError('Multi-phase training uses episode-seeded streams only')


class SelfPlayTrainer(Trainer):
    def __init__(self, config, initial):
        super().__init__(config)
        if initial.hidden_size != config.hidden_size:
            raise ValueError('Initial checkpoint architecture does not match hidden_size')
        if isinstance(initial, PhaseActorCritic):
            neural = tuple(p for p in ('bid', 'trump', 'play')
                           if p in initial.neural_phases or p in config.learn_phases)
            model = PhaseActorCritic(config.hidden_size, neural)
            model.load_state_dict(initial.state_dict())
        else:
            model = PhaseActorCritic.from_play(initial, config.learn_phases)
        model.trained_phases = tuple(getattr(initial, 'trained_phases', ('play',)))
        model.configure_learning(config.learn_phases)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                          lr=config.learning_rate)
        self.sampling_scheme = 'multiphase-episode-v1'
        minimum = self.minimum_decisions
        self.workers = resolve_workers(config.threads, min(config.total_hands,
                                math.ceil(config.rollout_decisions / minimum)))
        self.snapshots = []
        self.last_snapshot = 0
        self.anchor_sha256 = config.anchor.sha256
        self.initial_checkpoint_sha256 = getattr(initial, 'checkpoint_sha256', None)
        self.initial_checkpoint_path = getattr(initial, 'checkpoint_path', None)
        self.phase_counts = {}
        self.bc_metrics = None
        self.forgetting_data = None

    @property
    def minimum_decisions(self):
        return 2 * int('bid' in self.config.learn_phases) + 14 * int('play' in self.config.learn_phases)

    def opponent_pool(self):
        c = self.config
        return HistoricalPool(c.opponent_probabilities, c.opponent_policies, c.anchor, tuple(self.snapshots))

    @torch.no_grad()
    def collect(self):
        c, started = self.config, perf_counter()
        if self.hands >= c.total_hands:
            raise ValueError('Training budget exhausted')
        weights, pool = snapshot(self.model), self.opponent_pool()
        rows, pids, indices, actual = [], set(), [], 0
        previous = torch.get_num_threads()
        try:
            while actual < c.rollout_decisions and self.hands + len(indices) < c.total_hands:
                remaining = c.total_hands - self.hands - len(indices)
                count = min(math.ceil((c.rollout_decisions - actual) / self.minimum_decisions), remaining)
                # Neural trump produces a variable extra sample. Fixed-size waves
                # keep boundaries independent of worker scheduling and use actual
                # accumulated decisions. At most one eight-hand wave overshoots.
                if 'trump' in c.learn_phases:
                    count = min(count, 8)
                first = self.hands + len(indices)
                wave = tuple(range(first, first + count))
                indices.extend(wave)
                size = math.ceil(count / self.workers)
                tasks = [(weights, c.hidden_size, self.model.neural_phases, c.learn_phases, c.seed,
                          wave[i:i + size], pool) for i in range(0, count, size)]
                if self.workers == 1:
                    torch.set_num_threads(1)
                    chunks = [phase_chunk(task) for task in tasks]
                else:
                    if self.pool is None:
                        self.pool = OrderedPool(self.workers, initializer=initialize_worker)
                    chunks = self.pool.map(phase_chunk, tasks)
                completed = [r for chunk, pid in chunks for r in chunk]
                pids.update(pid for _, pid in chunks)
                rows.extend(completed)
                actual += sum(len(path) for r in completed for path in r['paths'])
        except BaseException:
            self.close()
            raise
        finally:
            torch.set_num_threads(previous)
        rows.sort(key=lambda r: r['episode'])
        if [r['episode'] for r in rows] != indices:
            raise RuntimeError('Duplicated or missing episode')
        count = len(indices)
        self.worker_pids = sorted(pids)
        merging_started = perf_counter()
        buffer = RolloutBuffer()
        self.opponent_assignments, self.opponent_metrics, self.phase_counts = [], {}, {}
        for row in rows:
            self.opponent_assignments.append((row['episode'], row['opponent'], row['opponent_sha256']))
            stats = self.opponent_metrics.setdefault(row['opponent'], dict(hands=0, reward_sum=0.0))
            stats['hands'] += 1
            stats['reward_sum'] += row['reward']
            for path in row['paths']:
                buffer.add_trajectory(path, c.gamma, c.gae_lambda)
                for record in path:
                    phase = ('bid', 'trump', 'play')[record.observation_features[:3].index(1.0)]
                    self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        for stats in self.opponent_metrics.values():
            stats['mean_signed_reward'] = stats['reward_sum'] / stats['hands']
        expected = count * self.minimum_decisions
        if not expected <= len(buffer) <= expected + count * int('trump' in c.learn_phases):
            raise RuntimeError('Incorrect enabled-phase rollout count')
        self.rollout_performance = {k: sum(r[k] for r in rows)
                                   for k in ('inference_seconds', 'neural_inference_seconds', 'opponent_loading_seconds')}
        self.rollout_performance['worker_peak_rss_bytes'] = max((r['worker_rss_bytes'] or 0 for r in rows), default=0) or None
        self.rollout_performance['merging_seconds'] = perf_counter() - merging_started
        self.hands += count
        self.samples += len(buffer)
        self.collection_seconds = perf_counter() - started
        return buffer, sum(r['reward'] for r in rows) / count

    def update(self, buffer):
        trained = set(getattr(self.model, 'trained_phases', ('play',))) | set(self.phase_counts)
        metrics = super().update(buffer)
        self.model.trained_phases = tuple(p for p in ('bid', 'trump', 'play') if p in trained)
        return metrics

    def publish_snapshot(self, directory):
        from pathlib import Path
        from .selfplay_checkpoint import save
        path = Path(directory) / f'policy-{self.hands:09d}.pt'
        if path.exists():
            raise ValueError(f'Immutable snapshot already exists: {path}')
        self.last_snapshot = self.hands
        save(path, self)
        spec = PolicySpec.checkpoint(f'snapshot-{self.hands}', path)
        self.snapshots.append(spec)
        self.snapshots = self.snapshots[-self.config.snapshot_pool_size:]
        # Files remain immutable archive artifacts; only active membership is bounded.
        return spec
