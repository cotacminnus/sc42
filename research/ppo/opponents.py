"""Small spawn-safe variant pools, independent of neural model definitions."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import random
from typing import TYPE_CHECKING
from texas42.agents import VARIANTS, variant_factory, HeuristicAgent
from .seeds import stream_seed

if TYPE_CHECKING:
    from .composition import CompositePolicySpec


DEFAULT_OPPONENTS = (('HHH', '1'),)


@dataclass(frozen=True)
class CheckpointIdentity:
    """File identity shared by frozen opponents and phase-composable evaluation."""
    path: str
    sha256: str

    @classmethod
    def from_path(cls, path):
        import hashlib
        from pathlib import Path
        path = Path(path).resolve()
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        return cls(str(path), digest)


def canonical_opponents(entries):
    """Exact normalized rational probabilities avoid ratio/resume discrepancies."""
    pairs, seen = [], set()
    try:
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError('Expected opponent name/weight pairs')
            name, weight = entry
            if name not in VARIANTS:
                raise ValueError(f'Unknown training opponent: {name}')
            if name in seen:
                raise ValueError(f'Duplicate training opponent: {name}')
            seen.add(name)
            if isinstance(weight, bool):
                raise ValueError('Opponent weights must be finite and strictly positive')
            try:
                value = Fraction(str(weight))
            except (ValueError, ZeroDivisionError, OverflowError) as exc:
                raise ValueError('Opponent weights must be finite and strictly positive') from exc
            if value <= 0:
                raise ValueError('Opponent weights must be finite and strictly positive')
            pairs.append((name, value))
    except (TypeError, KeyError) as exc:
        raise ValueError('Expected opponent name/weight pairs') from exc
    if not pairs:
        raise ValueError('Training opponent mixture cannot be empty')
    total = sum(weight for _, weight in pairs)
    return tuple((name, str(weight / total)) for name, weight in sorted(pairs))


def parse_opponents(spec):
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('Training opponent mixture cannot be empty')
    entries = []
    for entry in spec.split(','):
        pieces = entry.split(':')
        if len(pieces) != 2 or not all(piece.strip() for piece in pieces):
            raise ValueError('Expected NAME:WEIGHT,NAME:WEIGHT,...')
        entries.append(tuple(piece.strip() for piece in pieces))
    return canonical_opponents(entries)


def evaluation_opponents(names):
    try:
        names = tuple(names)
    except TypeError as exc:
        raise ValueError('Evaluation opponents must be registered variant names') from exc
    if not names or any(name not in VARIANTS for name in names):
        raise ValueError('Evaluation opponents must be nonempty registered variant names')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate evaluation opponents')
    return names


def select_primary(names, requested=None):
    if requested is None:
        if len(names) == 1:
            return names[0]
        if 'HHH' in names:
            return 'HHH'
        raise ValueError('Multiple evaluation opponents without HHH require --selection-opponent')
    if requested not in names:
        raise ValueError('selection_opponent must be present in evaluation opponents')
    return requested


@dataclass(frozen=True, slots=True)
class OpponentPool:
    probabilities: tuple = DEFAULT_OPPONENTS

    def __post_init__(self):
        object.__setattr__(self, 'probabilities', canonical_opponents(self.probabilities))

    def choose(self, seed, episode):
        if len(self.probabilities) == 1:
            return self.probabilities[0][0]  # Fixed opponents consume no RNG.
        draw = random.Random(stream_seed(seed, 'training-opponent-v1', episode)).random()
        cumulative = 0.0
        for name, probability in self.probabilities:
            cumulative += float(Fraction(probability))
            if draw < cumulative:
                return name
        return self.probabilities[-1][0]  # Floating-point cumulative roundoff.

    def describe(self):
        return ', '.join(f'{name}:{float(Fraction(p)):.6g}' for name, p in self.probabilities)


def hand_agents(opponent, seed, episode, learning_team):
    """Fresh per-seat agents; keep the historical HHH construction path intact."""
    factory = variant_factory(opponent)
    agents = []
    for seat in range(4):
        agent = (HeuristicAgent() if seat % 2 == learning_team or opponent == 'HHH' else
                 factory(seat, stream_seed(seed, 'training-opponent-agent-v1', episode * 4 + seat)))
        agent.reset(seat)
        agents.append(agent)
    return agents


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Portable frozen-policy identity; neural imports remain optional and lazy."""
    name: str
    kind: str = 'variant'
    path: str | None = None
    sha256: str | None = None
    composition: CompositePolicySpec | None = None

    def __post_init__(self):
        if not self.name or self.kind not in ('variant', 'checkpoint', 'composite'):
            raise ValueError('Invalid frozen policy specification')
        if self.kind == 'variant' and self.name not in VARIANTS:
            raise ValueError(f'Unknown variant: {self.name}')
        if self.kind == 'checkpoint' and (not self.path or not self.sha256 or len(self.sha256) != 64):
            raise ValueError('Checkpoint policy requires a path and SHA256 identity')
        if self.kind == 'composite':
            from .composition import CompositePolicySpec, PhasePolicySpec
            import hashlib
            import json
            spec = self.composition
            if isinstance(spec, dict):
                spec = CompositePolicySpec(spec['name'], *(PhasePolicySpec(p['strategy'],
                    CheckpointIdentity(**p['checkpoint']) if p.get('checkpoint') else None)
                    for p in (spec['bidding'], spec['trump'], spec['playing'])))
                object.__setattr__(self, 'composition', spec)
            if not isinstance(spec, CompositePolicySpec) or spec.name != self.name or self.path is not None:
                raise ValueError('Composite policy requires a matching named phase specification')
            digest = hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest()
            if self.sha256 is not None and self.sha256 != digest:
                raise ValueError('Composite policy identity mismatch')
            object.__setattr__(self, 'sha256', digest)
        elif self.composition is not None:
            raise ValueError('Phase composition applies only to composite policies')

    @classmethod
    def checkpoint(cls, name, path):
        identity = CheckpointIdentity.from_path(path)
        return cls(name, 'checkpoint', identity.path, identity.sha256)

    def factory(self):
        if self.kind == 'variant':
            return variant_factory(self.name)
        if self.kind == 'composite':
            from .composition import CompositeAgentFactory
            return CompositeAgentFactory(self.composition)
        from .selfplay_evaluation import FrozenFactory
        return FrozenFactory(self)

    def identities(self):
        """All immutable checkpoint dependencies, including independent phases."""
        if self.kind == 'checkpoint':
            return (CheckpointIdentity(self.path, self.sha256),)
        if self.kind == 'composite':
            return tuple(dict.fromkeys(s.checkpoint for _, s in self.composition.items() if s.checkpoint))
        return ()


def checkpoint_source(source, registry, initial=None):
    """Resolve an initial, named archive/anchor, or file to one immutable identity."""
    from os import PathLike
    if not isinstance(source, (str, PathLike)) or not str(source):
        raise ValueError('Checkpoint source must be a nonempty name or path')
    if source == 'init':
        if initial is None:
            raise ValueError('init requires an initialization checkpoint')
        return initial
    if source in registry:
        spec = registry[source]
        if spec.kind != 'checkpoint':
            raise ValueError(f'{source} is not a single checkpoint source')
        return CheckpointIdentity(spec.path, spec.sha256)
    return CheckpointIdentity.from_path(source)


def policy_registry(anchor, checkpoints=(), *, reserved=(), compositions=(), initial=None):
    """The built-in variants, original anchor and explicitly named frozen archives."""
    registry = {name: PolicySpec(name) for name in VARIANTS}
    registry['anchor'] = anchor
    for entry in checkpoints:
        name, path = entry.split('=', 1)
        if not name or name in registry or name in reserved:
            raise ValueError('Duplicate/empty or reserved checkpoint opponent name')
        registry[name] = PolicySpec.checkpoint(name, path)
    for entry in compositions:
        import json
        from pathlib import Path
        from .composition import CompositePolicySpec, PhasePolicySpec
        name, path = entry.split('=', 1)
        if not name or name in registry or name in reserved:
            raise ValueError('Duplicate/empty or reserved composed opponent name')
        data = json.loads(Path(path).read_text())
        if set(data) != {'bid', 'trump', 'play'}:
            raise ValueError('Opponent composition must specify bid, trump and play')
        parts = []
        for phase in ('bid', 'trump', 'play'):
            value = data[phase]
            if not isinstance(value, dict) or 'strategy' not in value or set(value) - {'strategy', 'source'}:
                raise ValueError(f'Invalid {phase} opponent specification')
            identity = checkpoint_source(value['source'], registry, initial) if 'source' in value else None
            parts.append(PhasePolicySpec(value['strategy'], identity))
        registry[name] = PolicySpec(name, 'composite', composition=CompositePolicySpec(name, *parts))
    return registry


@dataclass(frozen=True, slots=True)
class HistoricalPool:
    """Canonical mixture of named policies; snapshots have a separate RNG stream."""
    probabilities: tuple
    policies: tuple[PolicySpec, ...]
    anchor: PolicySpec
    snapshots: tuple[PolicySpec, ...] = ()

    def __post_init__(self):
        entries, seen = [], set()
        known = {p.name for p in self.policies} | {'self-snapshots'}
        if len({p.name for p in self.policies}) != len(self.policies):
            raise ValueError('Duplicate policy names')
        for name, weight in self.probabilities:
            if name not in known or name in seen or isinstance(weight, bool):
                raise ValueError(f'Unknown or duplicate opponent: {name}')
            seen.add(name)
            try:
                weight = Fraction(str(weight))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError('Weights must be finite and positive') from exc
            if weight <= 0:
                raise ValueError('Weights must be finite and positive')
            entries.append((name, weight))
        if not entries:
            raise ValueError('Opponent pool must be nonempty')
        total = sum(w for _, w in entries)
        object.__setattr__(self, 'probabilities', tuple((n, str(w / total)) for n, w in sorted(entries)))

    def choose(self, seed, episode):
        name = self.probabilities[0][0]
        if len(self.probabilities) > 1:
            draw = random.Random(stream_seed(seed, 'multiphase-opponent-v1', episode)).random()
            cumulative = 0.0
            for name, probability in self.probabilities:
                cumulative += float(Fraction(probability))
                if draw < cumulative:
                    break
        if name == 'self-snapshots':
            roster = self.snapshots or (self.anchor,)
            index = random.Random(stream_seed(seed, 'historical-snapshot-v1', episode)).randrange(len(roster))
            return name, roster[index]
        return name, next(p for p in self.policies if p.name == name)


def historical_pool(spec, policies, anchor, snapshots=()):
    """Parse named ratios without introducing checkpoint paths into weight syntax."""
    try:
        entries = [tuple(part.strip() for part in entry.split(':')) for entry in spec.split(',')]
        if any(len(entry) != 2 for entry in entries):
            raise ValueError('Expected NAME:WEIGHT pairs')
        return HistoricalPool(tuple(entries), tuple(policies), anchor, tuple(snapshots))
    except (TypeError, AttributeError) as exc:
        raise ValueError('Invalid opponent pool') from exc
