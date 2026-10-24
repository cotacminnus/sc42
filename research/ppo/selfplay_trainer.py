"""Multi-phase collection atop the unchanged clipped PPO optimizer and seat GAE."""
from dataclasses import dataclass
import math
from time import perf_counter
import torch
from texas42.parallel import OrderedPool, resolve_workers
from .trainer import Config, Trainer
from .multiphase import PhaseActorCritic, LearnerPhaseSpec, phases, PHASES
from .opponents import DYNAMIC_OPPONENTS, NO_ANCHOR, HistoricalPool, PolicySpec
from .buffer import RolloutBuffer
from .rollout import initialize_worker, snapshot
from .selfplay_rollout import phase_chunk
from .critic import CRITIC_MODES, ORACLE_POINT_SCALE, interval_loss, new_critic
from .model import freeze_value_heads
from .seeds import stream_seed


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
    learner_phases: tuple = ()  # Empty migrates the legacy initialization path.
    max_rollout_hands: int = 4096  # Bounds zero-sample trump-only collection.
    # critic_mode and return_estimator are inherited; defaults reproduce the
    # existing local actor-critic with GAE exactly.
    critic_modes = CRITIC_MODES
    oracle_aux_coef: float = 0.0
    oracle_dataset: str | None = None
    oracle_dataset_sha256: str | None = None
    # Random initialization of every neural learner phase; no checkpoint or anchor needed.
    from_scratch: bool = False

    @property
    def snapshot_fallback(self):
        """Legacy anchor fallback for self-snapshots; the current policy when anchor-free or from scratch."""
        return 'current' if self.from_scratch or self.anchor is None else 'anchor'

    def __post_init__(self):
        super().__post_init__()
        enabled = phases(self.learn_phases)
        object.__setattr__(self, 'learn_phases', enabled)
        specs = tuple(LearnerPhaseSpec(**s) if isinstance(s, dict) else s for s in self.learner_phases)
        object.__setattr__(self, 'learner_phases', specs)
        if specs:
            if len(specs) != 3 or not all(isinstance(s, LearnerPhaseSpec) for s in specs):
                raise ValueError('Expected three learner phase specifications')
            if not any(s.state == 'neural-train' for s in specs):
                raise ValueError('Training requires at least one neural-train phase')
            if tuple(p for p, s in zip(PHASES, specs) if s.state == 'neural-train') != enabled:
                raise ValueError('learn_phases must match neural-train learner phases')
        if type(self.from_scratch) is not bool:
            raise ValueError('from_scratch must be a boolean')
        if self.from_scratch and (not specs or any(s.initialization not in ('none', 'random') for s in specs)):
            raise ValueError('From-scratch training requires random or heuristic learner phases only')
        if not self.from_scratch and any(s.initialization == 'random' for s in specs):
            raise ValueError('Random phase initialization is only valid for from-scratch training')
        # The anchor is optional (from-scratch and anchor-free lineages) but, when
        # present, it is always an immutable checkpoint identity.
        if self.anchor is not None and self.anchor.kind != 'checkpoint':
            raise ValueError('The original play anchor must be an immutable checkpoint')
        if any(p.name == 'anchor' for p in self.opponent_policies) != (self.anchor is not None):
            raise ValueError('The policy registry must contain the anchor exactly when one is configured')
        pool = HistoricalPool(self.opponent_probabilities, self.opponent_policies, self.anchor,
                              snapshot_fallback=self.snapshot_fallback)
        object.__setattr__(self, 'opponent_probabilities', pool.probabilities)
        names = {p.name for p in self.opponent_policies}
        for name in (*self.evaluation_names, self.primary_opponent):
            if name in DYNAMIC_OPPONENTS:
                raise ValueError(f'{name} changes during training; it cannot be a fixed validation '
                                 'or checkpoint-selection opponent')
            if name == 'anchor' and self.anchor is None:
                raise ValueError(NO_ANCHOR)
        if (not self.evaluation_names or len(set(self.evaluation_names)) != len(self.evaluation_names)
                or not set(self.evaluation_names) <= names or self.primary_opponent not in self.evaluation_names):
            raise ValueError('Evaluation names and primary opponent must exist in frozen policy registry')
        for key in ('rollout_decisions', 'snapshot_every_hands', 'snapshot_pool_size', 'forgetting_hands', 'max_rollout_hands'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if type(self.bid_bc_hands) is not int or self.bid_bc_hands < 0:
            raise ValueError('bid_bc_hands must be nonnegative')
        if self.sampling_scheme not in ('auto', 'episode-v1'):
            raise ValueError('Multi-phase training uses episode-seeded streams only')
        if not math.isfinite(self.oracle_aux_coef) or self.oracle_aux_coef < 0:
            raise ValueError('oracle_aux_coef must be finite and nonnegative')
        if self.oracle_aux_coef > 0:
            if self.critic_mode != 'centralized':
                raise ValueError('Oracle auxiliary supervision trains the centralized critic; '
                                 f'it is unavailable with critic_mode {self.critic_mode}')
            if not self.oracle_dataset or not self.oracle_dataset_sha256:
                raise ValueError('Oracle auxiliary supervision requires an oracle dataset identity')
        elif self.oracle_dataset is not None or self.oracle_dataset_sha256 is not None:
            raise ValueError('An oracle dataset requires a positive oracle_aux_coef')


class SelfPlayTrainer(Trainer):
    def __init__(self, config, initial=None, *, initial_critic=None):
        """initial: the initialization model; None (required) for from-scratch training.

        initial_critic: a CentralCritic state_dict to continue, else a fresh critic.
        critic_initialization records which happened ('fresh' or 'checkpoint');
        it is None for the local actor-critic. A from-scratch resume restores
        weights afterwards through selfplay_checkpoint.resume().
        """
        super().__init__(config)
        if config.from_scratch:
            if initial is not None:
                raise ValueError('From-scratch training does not load an initialization model')
            neural = tuple(p for p, s in zip(PHASES, config.learner_phases) if s.state != 'heuristic')
            # Normal random initialization from its own --seed stream, independent of
            # earlier global-RNG use; nothing is copied from any checkpoint.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(stream_seed(config.seed, 'scratch-actor-init-v1'))
                model = PhaseActorCritic(config.hidden_size, neural)
            model.trained_phases = ()  # Untrained until an optimizer step updates a phase.
        elif initial is None:
            raise ValueError('Checkpoint-initialized training requires an initialization model')
        elif initial.hidden_size != config.hidden_size:
            raise ValueError('Initial checkpoint architecture does not match hidden_size')
        elif config.learner_phases:
            from .checkpoint import load_frozen
            from .composition import PhasePolicySpec, phase_network
            neural = tuple(p for p, s in zip(PHASES, config.learner_phases) if s.state != 'heuristic')
            model = PhaseActorCritic(config.hidden_size, neural)
            trained = []
            for phase, spec in zip(PHASES, config.learner_phases):
                if spec.state == 'heuristic':
                    continue
                source, metadata = load_frozen(spec.source, verify=True)
                if source.hidden_size != config.hidden_size:
                    raise ValueError(f'{phase} source architecture does not match hidden_size')
                if spec.initialization == 'trained':
                    network = phase_network(PhasePolicySpec('neural', spec.source), phase)
                    model.networks[phase].load_state_dict(network.state_dict(), strict=True)
                    trained.append(phase)
                else:
                    # Prefer the play representation; otherwise use the first trained
                    # phase in canonical order. Never transfer an untrained trunk.
                    available = ('play',) if metadata['checkpoint_version'] == 1 else tuple(
                        p for p in PHASES if p in metadata['trained_phases'] and p in metadata['neural_phases'])
                    if not available:
                        raise ValueError(f'{phase} initialization source has no trained representation')
                    reference = 'play' if 'play' in available else available[0]
                    network = source.networks[reference] if isinstance(source, PhaseActorCritic) else source
                    model.networks[phase].trunk.load_state_dict(network.trunk.state_dict(), strict=True)
            model.trained_phases = tuple(trained)
        elif isinstance(initial, PhaseActorCritic):
            neural = tuple(p for p in ('bid', 'trump', 'play')
                           if p in initial.neural_phases or p in config.learn_phases)
            model = PhaseActorCritic(config.hidden_size, neural)
            model.load_state_dict(initial.state_dict())
        else:
            model = PhaseActorCritic.from_play(initial, config.learn_phases)
        if not config.learner_phases and not config.from_scratch:
            model.trained_phases = tuple(getattr(initial, 'trained_phases', ('play',)))
        model.configure_learning(config.learn_phases)
        if config.critic_mode == 'none':
            freeze_value_heads(model)  # After configure_learning, which enables whole phases.
        self.model = model.to(self.device)
        self.critic, self.critic_initialization = None, None
        # Critic mode of the initialization checkpoint (provenance of a mode change
        # between stages); None for random initialization. Set by the caller.
        self.initial_critic_mode = None
        if config.critic_mode == 'centralized':
            # Separate weights and seed stream: actor initialization stays matched across modes.
            critic = new_critic(config.hidden_size, config.seed)
            self.critic_initialization = 'fresh'
            if initial_critic is not None:
                critic.load_state_dict(initial_critic, strict=True)
                self.critic_initialization = 'checkpoint'
            self.critic = critic.to(self.device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if self.critic is not None:
            parameters += list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        self.oracle_data = self.oracle_validation = self.oracle_summary = None
        self.oracle_shuffle = self.value_diagnostics = None
        if config.oracle_aux_coef > 0:
            from .oracle_dataset import load_dataset, training_tensors
            dataset = load_dataset(config.oracle_dataset, sha256=config.oracle_dataset_sha256)
            self.oracle_data = training_tensors(dataset, 'train')
            self.oracle_validation = training_tensors(dataset, 'validation', required=False)
            self.oracle_summary = dataset.summary()
            self.oracle_shuffle = torch.Generator().manual_seed(stream_seed(config.seed, 'oracle-auxiliary-v1'))
        self.sampling_scheme = 'multiphase-episode-v1'
        minimum = self.minimum_decisions
        self.workers = resolve_workers(config.threads, min(config.total_hands,
                                math.ceil(config.rollout_decisions / max(1, minimum))))
        self.snapshots = []
        self.last_snapshot = 0
        self.anchor_sha256 = config.anchor.sha256 if config.anchor is not None else None
        self.initial_checkpoint_sha256 = getattr(initial, 'checkpoint_sha256', None)
        self.initial_checkpoint_path = getattr(initial, 'checkpoint_path', None)
        # The same resolved specification describes legacy and explicit phase syntax.
        # Retain the old weight-construction path above to preserve its RNG sequence.
        from .opponents import CheckpointIdentity
        identity = (CheckpointIdentity(self.initial_checkpoint_path, self.initial_checkpoint_sha256)
                    if self.initial_checkpoint_path else None)
        self.learner_phases = config.learner_phases or tuple(
            LearnerPhaseSpec('heuristic', initialization='none') if p not in model.neural_phases else
            LearnerPhaseSpec('neural-train' if p in config.learn_phases else 'neural-frozen', identity,
                             'trained' if p in model.trained_phases else 'representation') for p in PHASES)
        self.phase_counts = {}
        self.bc_metrics = None
        self.forgetting_data = None

    @property
    def minimum_decisions(self):
        return 2 * int('bid' in self.config.learn_phases) + 14 * int('play' in self.config.learn_phases)

    def opponent_pool(self):
        c = self.config
        return HistoricalPool(c.opponent_probabilities, c.opponent_policies, c.anchor, tuple(self.snapshots),
                              c.snapshot_fallback)

    @torch.no_grad()
    def collect(self):
        c, started = self.config, perf_counter()
        if self.hands >= c.total_hands:
            raise ValueError('Training budget exhausted')
        weights, pool = snapshot(self.model), self.opponent_pool()
        critic = (snapshot(self.critic),) if self.critic is not None else ()
        rows, pids, indices, actual = [], set(), [], 0
        previous = torch.get_num_threads()
        try:
            while actual < c.rollout_decisions and self.hands + len(indices) < c.total_hands:
                remaining = c.total_hands - self.hands - len(indices)
                if not self.minimum_decisions:
                    remaining = min(remaining, c.max_rollout_hands - len(indices))
                    if remaining <= 0:
                        break
                count = min(math.ceil((c.rollout_decisions - actual) / max(1, self.minimum_decisions)), remaining)
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
                          wave[i:i + size], pool, *critic) for i in range(0, count, size)]
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
                if not path:
                    continue  # A defending learner has no trump action/reward trajectory.
                buffer.add_trajectory(path, c.gamma, c.gae_lambda, estimator=c.return_estimator,
                                      baseline=c.critic_mode != 'none')
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

    def auxiliary_loss(self):
        """Interval loss of the critic's oracle head on certified point bounds."""
        if self.oracle_data is None:
            return None
        data = self.oracle_data
        count = len(data['lower'])
        index = torch.randint(count, (min(self.config.minibatch_size, count),), generator=self.oracle_shuffle)
        features, lower, upper = (data[k][index].to(self.device) for k in ('features', 'lower', 'upper'))
        prediction = self.critic.oracle_points(features)
        loss = interval_loss(prediction, lower, upper).mean()
        error = (torch.relu(lower - prediction) + torch.relu(prediction - upper)).mean() * ORACLE_POINT_SCALE
        return self.config.oracle_aux_coef * loss, dict(oracle_aux_loss=loss.item(),
                                                        oracle_aux_point_error=error.item())

    @torch.no_grad()
    def oracle_validation_metrics(self):
        """Held-out (deal-disjoint) interval loss; None without a validation split."""
        if self.oracle_validation is None:
            return None
        data = self.oracle_validation
        self.critic.eval()
        prediction = self.critic.oracle_points(data['features'].to(self.device))
        lower, upper = data['lower'].to(self.device), data['upper'].to(self.device)
        return dict(loss=interval_loss(prediction, lower, upper).mean().item(),
                    point_error=((torch.relu(lower - prediction) + torch.relu(prediction - upper)).mean()
                                 * ORACLE_POINT_SCALE).item(), positions=len(lower))

    def return_diagnostics(self, buffer):
        """Value-target statistics by phase, before advantage normalization.

        Policy-only PPO has no value predictions: explained variance and value
        means are reported as unavailable rather than computed from the zero baseline.
        """
        from statistics import fmean, pvariance
        records = buffer.records
        learned = self.config.critic_mode != 'none'
        returns, values = buffer.returns, [r.old_state_value for r in records]
        variance = pvariance(returns) if len(returns) > 1 else 0.0
        residual = pvariance([g - v for g, v in zip(returns, values)]) if len(returns) > 1 else 0.0
        result = dict(explained_variance=1 - residual / variance if variance > 0 and learned else None)
        for index, phase in enumerate(PHASES):
            rows = [i for i, r in enumerate(records) if r.observation_features[index] == 1.0]
            if not rows:
                continue
            g = [buffer.returns[i] for i in rows]
            a = [buffer.advantages[i] for i in rows]
            result[phase] = dict(samples=len(rows), return_mean=fmean(g), return_std=pvariance(g) ** .5,
                                 advantage_mean=fmean(a), advantage_std=pvariance(a) ** .5,
                                 value_mean=fmean(values[i] for i in rows) if learned else None)
        return result

    def update(self, buffer):
        if not len(buffer):
            raise ValueError('No trainable decisions collected; skip the PPO update')
        self.value_diagnostics = self.return_diagnostics(buffer)
        metrics = super().update(buffer)
        # A KL early stop can leave a collected phase without an optimizer step.
        trained = set(getattr(self.model, 'trained_phases', ('play',))) | {
            p for p in PHASES if p + '_policy_loss' in metrics}
        self.model.trained_phases = tuple(p for p in ('bid', 'trump', 'play') if p in trained)
        return metrics

    def publish_snapshot(self, directory):
        from pathlib import Path
        from .selfplay_checkpoint import save
        if not set(self.model.neural_phases) <= set(self.model.trained_phases):
            raise ValueError('Cannot publish a frozen opponent with untrained neural phases')
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
