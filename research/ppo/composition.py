"""Evaluation-only phase composition; PyTorch is loaded only for neural policies."""
from dataclasses import asdict, dataclass
import random
from time import perf_counter

from texas42.agents import HeuristicAgent, RandomAgent, ConservativeRandomAgent
from .seeds import stream_seed
from .opponents import CheckpointIdentity

PHASES = ('bid', 'trump', 'play')


@dataclass(frozen=True)
class PhasePolicySpec:
    strategy: str = 'heuristic'
    checkpoint: CheckpointIdentity | None = None

    def validate(self, phase):
        allowed = ('neural', 'heuristic', 'random', 'conservative') if phase == 'bid' else ('neural', 'heuristic', 'random')
        if self.strategy not in allowed:
            raise ValueError(f'{phase}: unsupported strategy {self.strategy!r}')
        if self.strategy == 'neural' and self.checkpoint is None:
            raise ValueError(f'Neural {phase} requires a phase checkpoint or agent default checkpoint')
        if self.strategy != 'neural' and self.checkpoint is not None:
            raise ValueError(f'{phase}: checkpoints apply only to neural strategies')


@dataclass(frozen=True)
class CompositePolicySpec:
    name: str
    bidding: PhasePolicySpec = PhasePolicySpec()
    trump: PhasePolicySpec = PhasePolicySpec()
    playing: PhasePolicySpec = PhasePolicySpec()

    def __post_init__(self):
        for phase, spec in self.items():
            spec.validate(phase)

    def items(self):
        return tuple(zip(PHASES, (self.bidding, self.trump, self.playing)))

    def to_dict(self):
        return asdict(self)

    def label(self):
        return '[' + ' | '.join(f'{s.strategy.title()} {p.title()}' for p, s in self.items()) + ']'


def phase_network(spec, phase):
    from .checkpoint import load_frozen
    model, metadata = load_frozen(spec.checkpoint)
    if metadata['checkpoint_version'] == 1:
        available = ('play',)
    else:
        available = tuple(p for p in metadata['neural_phases'] if p in metadata['trained_phases'])
    if phase not in available:
        raise ValueError(f'Neural {phase} unavailable in {spec.checkpoint.path}; trained neural phases: {", ".join(available) or "none"}')
    return model if metadata['checkpoint_version'] == 1 else model.networks[phase]


class _NeuralPolicy:
    """A single frozen phase network with a seat-local sampling stream."""
    def __init__(self, network, seed, greedy):
        import torch
        self.network, self.greedy = network, greedy
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def reset(self, seat):
        self.seat = seat

    def act(self, obs):
        import torch
        from texas42.encoding import encode
        if obs.seat != self.seat:
            raise RuntimeError('Reset for the acting seat before use')
        with torch.inference_mode():
            features = torch.tensor([encode(obs)], dtype=torch.float32)
            mask = torch.tensor([obs.action_mask], dtype=torch.bool)
            actions, _, _ = self.network.select(features, mask, greedy=self.greedy, generator=self.generator)
        return int(actions.item())


class CompositeAgent:
    """Independent delegate memory per seat and phase; reset does not rewind RNGs."""
    def __init__(self, spec, seed, *, greedy=True):
        self.spec, self.seat = spec, None
        self.delegates = {}
        # Match HybridAgent's documented streams for existing non-neural variants.
        rng = random.Random(seed)
        for phase, policy in spec.items():
            phase_seed = rng.getrandbits(128)
            if policy.strategy == 'neural':
                delegate = _NeuralPolicy(phase_network(policy, phase),
                    stream_seed(seed, f'composite-neural-{phase}-v1'), greedy)
            elif policy.strategy == 'heuristic':
                delegate = HeuristicAgent()
            elif policy.strategy == 'conservative':
                delegate = ConservativeRandomAgent(phase_seed)
            else:
                delegate = RandomAgent(phase_seed)
            self.delegates[phase] = delegate

    def reset(self, seat):
        self.seat = seat
        for delegate in self.delegates.values():
            delegate.reset(seat)

    def act(self, obs):
        if obs.seat != self.seat:
            raise RuntimeError('Reset for the acting seat before use')
        action = self.delegates[obs.phase].act(obs)
        if type(action) is not int or action not in obs.legal_action_ids:
            raise ValueError(f'{self.spec.name} selected an illegal {obs.phase} action: {action!r}')
        return action

    def would_pass_if_allowed(self, obs):
        if obs.phase != 'bid':
            raise ValueError('Expected a bidding observation')
        diagnostic = getattr(self.delegates['bid'], 'would_pass_if_allowed', None)
        return diagnostic(obs) if diagnostic is not None else None


@dataclass(frozen=True)
class CompositeAgentFactory:
    spec: CompositePolicySpec
    greedy: bool = True

    def prepare(self):
        identities = {s.checkpoint for _, s in self.spec.items() if s.strategy == 'neural'}
        if identities:
            import torch
            # Numerical inference threads are independent of arena process workers.
            import multiprocessing
            if multiprocessing.current_process().name != 'MainProcess':
                torch.set_num_threads(1)
        for identity in identities:
            from .checkpoint import load_frozen
            load_frozen(identity, verify=True)
        for phase, policy in self.spec.items():
            if policy.strategy == 'neural':
                phase_network(policy, phase)

    def __call__(self, seat, seed):
        return CompositeAgent(self.spec, seed, greedy=self.greedy)


def comparison_metadata(a, b):
    """Compare actual tensor values, independently of source file identity."""
    equal, neural_equal = {}, {}
    for (phase, left), (_, right) in zip(a.items(), b.items()):
        same = left.strategy == right.strategy
        neural_equal[phase] = None
        if left.strategy == right.strategy == 'neural':
            import torch
            x, y = phase_network(left, phase).state_dict(), phase_network(right, phase).state_dict()
            same = x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
            neural_equal[phase] = same
        equal[phase] = same
    identities = [{s.checkpoint.sha256 for _, s in spec.items() if s.strategy == 'neural'} for spec in (a, b)]
    differing = [p for p in PHASES if not equal[p]]
    return dict(neural_parameters_identical=neural_equal, phase_policies_identical=equal,
        shared_complete_checkpoint=bool(len(identities[0]) == 1 and identities[0] == identities[1]),
        only_one_phase_differs=len(differing) == 1, differing_phases=differing)


def evaluate_matchups(factory, opponents, *, neural=False, performance=None, **kwargs):
    """Shared arena execution for composed, checkpoint and in-memory policies.

    Named opponent factories retain their own inference and seed semantics. This
    consolidates only execution/thread management, not policy implementations.
    """
    from texas42.arena import compare_agents
    previous = None
    if neural:
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
    try:
        reports = {}
        for name, opponent in opponents:
            started, telemetry = perf_counter(), {}
            reports[name] = compare_agents(factory, opponent, performance=telemetry, **kwargs)
            telemetry['evaluation_total_seconds'] = perf_counter() - started
            if performance is not None:
                performance[name] = telemetry
        return reports
    finally:
        if previous is not None:
            torch.set_num_threads(previous)


def evaluate_composed(a, b, *, stochastic=False, performance=None, **kwargs):
    """Canonical two-agent composition API, backed by the paired arena."""
    neural = any(s.strategy == 'neural' for spec in (a, b) for _, s in spec.items())
    telemetry = {}
    report = evaluate_matchups(CompositeAgentFactory(a, not stochastic),
        ((b.name, CompositeAgentFactory(b, not stochastic)),), neural=neural,
        performance=telemetry, **kwargs)[b.name]
    if performance is not None:
        # Keep the public performance shape of this single-comparison API.
        values = telemetry[b.name]
        values.pop('evaluation_total_seconds')
        performance.update(values)
    return report
