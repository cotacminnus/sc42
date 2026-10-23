"""Evaluation-only phase composition; PyTorch is loaded only for neural policies."""
from dataclasses import asdict, dataclass
import random
from time import perf_counter

from texas42.agents import HeuristicAgent, RandomAgent, ConservativeRandomAgent
from texas42.agents.pimc import DEFAULT_PIMC_SAMPLES, PIMCAgent
from texas42.belief import DEFAULT_EPSILON
from .seeds import stream_seed
from .opponents import CheckpointIdentity

PHASES = ('bid', 'trump', 'play')
# 'pimc' (belief-based Monte Carlo, texas42.agents.pimc) maximizes expected
# points, which suits trump and play but not bidding, so bid does not offer it.
STRATEGIES = {'bid': ('neural', 'heuristic', 'random', 'conservative'),
              'trump': ('neural', 'heuristic', 'random', 'pimc'),
              'play': ('neural', 'heuristic', 'random', 'pimc')}


def strategy_label(strategy):
    return 'PIMC' if strategy == 'pimc' else strategy.title()


def partner_index(seat):
    """Paired-partner rule: 0 for a team's first seat (0 or 1), 1 for its second (2 or 3).

    Shared by paired training collection, in-training validation and paired
    evaluation. The arena seats policy A at 0/2 (ABAB) and then 1/3 (BABA), so
    "partner 1" is always the lower seat of whichever team it plays for.
    """
    return seat // 2


@dataclass(frozen=True)
class PairedAgentFactory:
    """One partnership from two factories: first seat -> `first`, partner seat -> `second`.

    Any arena-style factories work (composite, frozen PPO, variants). With no
    pairing, the arena simply calls one factory for both partner seats.
    """
    first: object
    second: object

    def prepare(self):
        for factory in (self.first, self.second):
            if hasattr(factory, 'prepare'):
                factory.prepare()

    def __call__(self, seat, seed):
        return (self.first if partner_index(seat) == 0 else self.second)(seat, seed)


@dataclass(frozen=True)
class PhasePolicySpec:
    strategy: str = 'heuristic'
    checkpoint: CheckpointIdentity | None = None

    def validate(self, phase):
        if self.strategy not in STRATEGIES[phase]:
            raise ValueError(f'{phase}: unsupported strategy {self.strategy!r}')
        if self.strategy == 'neural' and self.checkpoint is None:
            raise ValueError(f'Neural {phase} requires a phase checkpoint or agent default checkpoint')
        # A 'pimc' checkpoint means model-guided search (research.ppo.search);
        # without one, PIMC keeps its Heuristic V1 rollouts.
        if self.strategy not in ('neural', 'pimc') and self.checkpoint is not None:
            raise ValueError(f'{phase}: checkpoints apply only to neural strategies and model-guided pimc')


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
        return '[' + ' | '.join(f'{strategy_label(s.strategy)} {p.title()}' for p, s in self.items()) + ']'


def resolve_policy_spec(name, strategies, checkpoint=None, phase_checkpoints=None, pimc_checkpoint=None):
    """Resolve CLI phase overrides once for both console and arena composition.

    pimc_checkpoint (optional) is the model guiding every 'pimc' phase of this
    agent. It is separate from the neural sources so that adding a default
    --agent-X-checkpoint never silently changes how PIMC phases search.
    """
    overrides = (None,) * 3 if phase_checkpoints is None else phase_checkpoints
    if len(strategies) != 3 or len(overrides) != 3:
        raise ValueError('Specify bid, trump and play policies in that order')
    policies = []
    identities = {}
    for phase, strategy, override in zip(PHASES, strategies, overrides):
        if override is not None and strategy != 'neural':
            raise ValueError(f'{name} {phase}: checkpoint override requires neural strategy')
        path = override or checkpoint
        identity = None
        if strategy == 'pimc':
            path = pimc_checkpoint
        if strategy in ('neural', 'pimc') and path is not None:
            if path not in identities:
                identities[path] = CheckpointIdentity.from_path(path)
            identity = identities[path]
        policies.append(PhasePolicySpec(strategy, identity))
    return CompositePolicySpec(name, *policies)


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
    """A single frozen phase network with a seat-local sampling stream.

    Greedy decisions take the masked argmax of the policy logits directly.
    This is the same choice as ActorCritic.select(greedy=True), whose
    Categorical logits are those logits minus a constant (log-normalization),
    so the argmax is unchanged. Skipping the distribution object makes a
    decision several times faster, which matters when the policy runs inside
    Monte Carlo search (thousands of calls per decision). Stochastic decisions
    still use select() and this seat's generator, exactly as before.
    """
    def __init__(self, network, seed, greedy):
        import torch
        self.network, self.greedy = network, greedy
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def reset(self, seat):
        self.seat = seat

    def _masked_logits(self, obs):
        """(features, legal mask, legal-masked policy logits) for one observation."""
        import torch
        from texas42.encoding import encode
        features = torch.tensor([encode(obs)], dtype=torch.float32)
        mask = torch.tensor([obs.action_mask], dtype=torch.bool)
        logits, _ = self.network(features)
        return features, mask, logits.masked_fill(~mask, -torch.inf)

    def act(self, obs):
        import torch
        if obs.seat != self.seat:
            raise RuntimeError('Reset for the acting seat before use')
        with torch.inference_mode():
            features, mask, logits = self._masked_logits(obs)
            if self.greedy:
                return int(logits.argmax(-1).item())
            actions, _, _ = self.network.select(features, mask, greedy=False, generator=self.generator)
        return int(actions.item())

    def action_probabilities(self, obs):
        """Pure ActionProbabilities query: the legal softmax of the policy logits.

        This is the distribution stochastic act() samples from, regardless of
        the greedy setting, so belief weighting sees the model's full
        preference rather than only its top choice. It consumes no RNG and
        changes no state (agents.base.ActionProbabilities contract).
        """
        import torch
        if obs.seat != self.seat:
            raise RuntimeError('Reset for the acting seat before use')
        with torch.inference_mode():
            _, _, logits = self._masked_logits(obs)
            # Double precision so the legal probabilities sum to 1 within 1e-12.
            probabilities = torch.softmax(logits.double(), -1)[0]
        return {action_id: float(probabilities[action_id]) for action_id in obs.legal_action_ids}


class CompositeAgent:
    """Independent delegate memory per seat and phase; reset does not rewind RNGs.

    The pimc_* settings configure PIMC delegates only (see _pimc_delegate). They
    are factory/agent settings, not part of CompositePolicySpec, so stored
    composite identities are unchanged.
    """
    def __init__(self, spec, seed, *, greedy=True, pimc_samples=DEFAULT_PIMC_SAMPLES,
                 pimc_leaf='none', pimc_likelihood='uniform', pimc_epsilon=DEFAULT_EPSILON):
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
            elif policy.strategy == 'pimc':
                delegate = _pimc_delegate(policy, phase_seed, greedy, pimc_samples, pimc_leaf,
                                          pimc_likelihood, pimc_epsilon)
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

    def action_probabilities(self, obs):
        """Pure ActionProbabilities query, answered by the acting phase's delegate.

        Neural, heuristic, random and conservative delegates implement it; a
        PIMC delegate does not (its decisions consume RNG), so a composition
        with PIMC phases cannot serve as a belief likelihood model.
        """
        if obs.seat != self.seat:
            raise RuntimeError('Reset for the acting seat before use')
        query = getattr(self.delegates[obs.phase], 'action_probabilities', None)
        if query is None:
            raise TypeError(f'{self.spec.name} {obs.phase} policy does not provide action probabilities')
        return query(obs)

    def would_pass_if_allowed(self, obs):
        if obs.phase != 'bid':
            raise ValueError('Expected a bidding observation')
        diagnostic = getattr(self.delegates['bid'], 'would_pass_if_allowed', None)
        return diagnostic(obs) if diagnostic is not None else None


PIMC_LEAVES = ('none', 'value')
PIMC_LIKELIHOODS = ('uniform', 'model')


def _pimc_delegate(policy, seed, greedy, samples, leaf, likelihood, epsilon):
    """Build one PIMC phase delegate (texas42.agents.pimc.PIMCAgent).

    Without a checkpoint: the original heuristic-guided PIMC (Heuristic V1
    rollouts, uniform belief, expected points). The leaf/likelihood settings
    apply only to model-guided PIMC and are ignored here, so one run can compare
    model-guided and plain PIMC agents.

    With a checkpoint (research.ppo.search):
      rollouts    the model plays every seat's continuation (greedy unless the
                  evaluation runs with --stochastic-neural);
      likelihood  'model': hidden deals weighted by the model's move
                  probabilities; 'uniform': all consistent deals equal;
      leaf        'value': rollouts stop after the next trick and the model's
                  value head scores the rest, so the objective becomes signed
                  marks; 'none': full-hand rollouts, expected points.
    """
    if policy.checkpoint is None:
        return PIMCAgent(seed, samples=samples)
    from .search import ValueLeaf, model_factories
    models = model_factories(policy.checkpoint, greedy=greedy)
    value = leaf == 'value'
    return PIMCAgent(seed, samples=samples, rollout_policies=models,
                     likelihood_policies=models if likelihood == 'model' else None, epsilon=epsilon,
                     until='trick' if value else 'hand',
                     leaf_evaluator=ValueLeaf(policy.checkpoint) if value else None,
                     objective='marks' if value else 'points')


@dataclass(frozen=True)
class CompositeAgentFactory:
    spec: CompositePolicySpec
    greedy: bool = True
    pimc_samples: int = DEFAULT_PIMC_SAMPLES
    pimc_leaf: str = 'none'
    pimc_likelihood: str = 'uniform'
    pimc_epsilon: float = DEFAULT_EPSILON

    def __post_init__(self):
        if self.pimc_leaf not in PIMC_LEAVES:
            raise ValueError(f'pimc_leaf must be one of {PIMC_LEAVES}')
        if self.pimc_likelihood not in PIMC_LIKELIHOODS:
            raise ValueError(f'pimc_likelihood must be one of {PIMC_LIKELIHOODS}')
        if not 0.0 <= self.pimc_epsilon <= 1.0:
            raise ValueError('pimc_epsilon must be between 0 and 1')

    def prepare(self):
        # Every checkpoint this agent loads: neural phases and PIMC guide models.
        identities = {s.checkpoint for _, s in self.spec.items()
                      if s.strategy in ('neural', 'pimc') and s.checkpoint is not None}
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
            elif policy.strategy == 'pimc' and policy.checkpoint is not None:
                from .search import model_policy_spec, trained_phases
                model_policy_spec(policy.checkpoint)  # The guide must have a trained phase.
                if self.pimc_leaf == 'value' and 'play' not in trained_phases(policy.checkpoint):
                    raise ValueError(f'Value leaf needs a trained play network in {policy.checkpoint.path}')

    def __call__(self, seat, seed):
        return CompositeAgent(self.spec, seed, greedy=self.greedy, pimc_samples=self.pimc_samples,
                              pimc_leaf=self.pimc_leaf, pimc_likelihood=self.pimc_likelihood,
                              pimc_epsilon=self.pimc_epsilon)


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


def evaluate_composed(a, b, *, a_partner=None, b_partner=None, stochastic=False,
                      pimc_samples=DEFAULT_PIMC_SAMPLES, pimc_leaf='none', pimc_likelihood='uniform',
                      pimc_epsilon=DEFAULT_EPSILON, performance=None, **kwargs):
    """Canonical two-agent composition API, backed by the paired arena.

    a_partner/b_partner optionally give that team's second seat (2 or 3) its own
    composition, e.g. the second network of a paired-partner training run.
    None keeps the original mirrored team: one specification for both seats.
    """
    specs = tuple(s for s in (a, b, a_partner, b_partner) if s is not None)
    # Neural inference (including model-guided PIMC) pins PyTorch threads.
    neural = any(s.strategy == 'neural' or (s.strategy == 'pimc' and s.checkpoint is not None)
                 for spec in specs for _, s in spec.items())
    settings = dict(pimc_samples=pimc_samples, pimc_leaf=pimc_leaf, pimc_likelihood=pimc_likelihood,
                    pimc_epsilon=pimc_epsilon)

    def team(spec, partner):
        first = CompositeAgentFactory(spec, not stochastic, **settings)
        if partner is None:
            return first
        return PairedAgentFactory(first, CompositeAgentFactory(partner, not stochastic, **settings))

    telemetry = {}
    report = evaluate_matchups(team(a, a_partner), ((b.name, team(b, b_partner)),), neural=neural,
        performance=telemetry, **kwargs)[b.name]
    if performance is not None:
        # Keep the public performance shape of this single-comparison API.
        values = telemetry[b.name]
        values.pop('evaluation_total_seconds')
        performance.update(values)
    return report
