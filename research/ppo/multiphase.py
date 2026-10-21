"""Independent phase modules: frozen play cannot share a trainable bid trunk."""
import torch
from dataclasses import dataclass
from torch import nn
from .model import ActorCritic
from .opponents import CheckpointIdentity, checkpoint_source

PHASES = ('bid', 'trump', 'play')


@dataclass(frozen=True)
class LearnerPhaseSpec:
    """Resolved initialization provenance and optimizer membership for one phase.

    initialization: 'none' (heuristic), 'trained' (copied trained phase),
    'representation' (trained trunk, fresh heads) or 'random' (no source: the
    normal random initialization of trunk, policy and value heads, from-scratch only).
    """
    state: str
    source: CheckpointIdentity | None = None
    initialization: str = 'trained'

    def __post_init__(self):
        if isinstance(self.source, dict):
            object.__setattr__(self, 'source', CheckpointIdentity(**self.source))
        if self.state not in ('heuristic', 'neural-frozen', 'neural-train'):
            raise ValueError('Learner state must be heuristic, neural-frozen or neural-train')
        if self.state == 'heuristic':
            if self.source is not None or self.initialization != 'none':
                raise ValueError('Heuristic phases have no neural source')
        elif self.initialization not in ('trained', 'representation', 'random'):
            raise ValueError('Unknown phase initialization')
        elif self.initialization == 'random' and (self.source is not None or self.state != 'neural-train'):
            raise ValueError('Random initialization has no source and must be neural-train')
        elif self.initialization == 'trained' and self.source is None:
            raise ValueError('A trained neural phase requires a checkpoint source')
        elif self.state == 'neural-frozen' and self.initialization != 'trained':
            raise ValueError('Frozen neural phases require a trained source')


def resolve_learner_phases(states, sources, registry, initial):
    """Resolve new CLI states; missing trainable init phases transfer only a trunk."""
    from .checkpoint import load_frozen
    from .composition import PhasePolicySpec, phase_network
    if len(states) != 3 or len(sources) != 3:
        raise ValueError('Specify three learner phases and sources in bid,trump,play order')
    result = []
    for phase, state, source in zip(PHASES, states, sources):
        if state == 'heuristic':
            if source is not None:
                raise ValueError(f'{phase}: heuristic policy cannot have a source')
            result.append(LearnerPhaseSpec(state, initialization='none'))
            continue
        requested = source or 'init'
        identity = checkpoint_source('init' if requested == 'new' else requested, registry, initial)
        _, metadata = load_frozen(identity, verify=True)
        trained = ('play',) if metadata['checkpoint_version'] == 1 else tuple(
            p for p in metadata['neural_phases'] if p in metadata['trained_phases'])
        initialization = 'trained'
        if requested == 'new' or phase not in trained:
            if state != 'neural-train' or requested not in ('init', 'new'):
                raise ValueError(f'{phase}: source {requested} has no trained neural {phase}; '
                                 'use a trained source, or neural-train with source new')
            initialization = 'representation'
        if initialization == 'trained':
            phase_network(PhasePolicySpec('neural', identity), phase)
        result.append(LearnerPhaseSpec(state, identity, initialization))
    if len(result) != 3 or not any(s.state == 'neural-train' for s in result):
        raise ValueError('Training requires at least one neural-train phase')
    return tuple(result)


def phases(value):
    values = tuple(value.split(',')) if isinstance(value, str) else tuple(value)
    if not values or len(set(values)) != len(values) or any(p not in PHASES for p in values):
        raise ValueError('Expected distinct phases from bid,trump,play')
    return tuple(p for p in PHASES if p in values)


class PhaseActorCritic(nn.Module):
    def __init__(self, hidden_size=256, neural_phases=('bid', 'play')):
        super().__init__()
        self.hidden_size = hidden_size
        self.neural_phases = phases(neural_phases)
        self.networks = nn.ModuleDict({p: ActorCritic(hidden_size) for p in PHASES})

    def configure_learning(self, learn_phases):
        enabled = phases(learn_phases)
        if not set(enabled) <= set(self.neural_phases):
            raise ValueError('Cannot train a phase configured as heuristic')
        for phase, module in self.networks.items():
            module.requires_grad_(phase in enabled)
            module.zero_grad(set_to_none=True)

    def forward(self, features):
        # The existing encoder's first three features are the phase one-hot.
        if features.ndim != 2 or not torch.all(features[:, :3].sum(-1) == 1):
            raise ValueError('Expected a batch of encoded phase observations')
        indices = features[:, :3].argmax(-1)
        logits = features.new_zeros((len(features), self.networks['play'].policy.out_features))
        values = features.new_zeros(len(features))
        for index, phase in enumerate(PHASES):
            selected = indices == index
            if selected.any():
                if phase not in self.neural_phases:
                    raise ValueError(f'{phase} uses a heuristic, not a neural policy')
                policy, value = self.networks[phase](features[selected])
                logits[selected], values[selected] = policy, value
        return logits, values

    # Reuse exactly the Stage-1 masking, sampling, and likelihood implementation.
    distribution = ActorCritic.distribution
    select = ActorCritic.select
    evaluate = ActorCritic.evaluate

    @classmethod
    def from_play(cls, model, learn_phases=('bid',)):
        enabled = phases(learn_phases)
        result = cls(model.hidden_size, tuple(p for p in PHASES if p == 'play' or p in enabled))
        result.networks['play'].load_state_dict(model.state_dict(), strict=True)
        # Transfer representation only; bidding/trump output rows were never trained.
        for phase in ('bid', 'trump'):
            result.networks[phase].trunk.load_state_dict(model.trunk.state_dict())
        result.trained_phases = ('play',)
        result.configure_learning(enabled)
        return result
