"""Small spawn-safe variant pools, independent of neural model definitions."""
from dataclasses import dataclass
from fractions import Fraction
import random
from texas42.agents import VARIANTS, variant_factory, HeuristicAgent
from .seeds import stream_seed


DEFAULT_OPPONENTS = (('HHH', '1'),)


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
