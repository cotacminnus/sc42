"""Phase composition of frozen policies, with isolated per-phase RNG streams."""
import random
from types import MappingProxyType

from ..phases import Phase
from ..selfplay import Observation
from .conservative_random import ConservativeRandomAgent
from .random import RandomAgent
from .simpleheuristic import HeuristicAgent


# Ordered bidding, trump, playing. Immutable to keep named experiments stable.
VARIANTS = MappingProxyType({
    'HHH': ('heuristic', 'heuristic', 'heuristic'),
    'HHR': ('heuristic', 'heuristic', 'random'),
    'HRR': ('heuristic', 'random', 'random'),
    'CHH': ('conservative', 'heuristic', 'heuristic'),
    'CRR': ('conservative', 'random', 'random'),
    'RRR': ('random', 'random', 'random'),
})


class HybridAgent:
    """Compose ordinary agents without sharing seat or phase memory.

    random.Random(seed) generates three 128-bit seeds, always in bid/trump/play
    order, including for deterministic phases. Each delegate owns its stream.
    reset assigns all delegates a seat without rewinding any stream. RRR is
    uniform legal random play, but has different streams from RandomAgent(seed).
    Different histories/legal sets can still produce different trajectories.
    """

    def __init__(self, bidding='heuristic', trump='heuristic', playing='heuristic',
                 seed: int | None = None):
        if bidding not in ('heuristic', 'random', 'conservative'):
            raise ValueError('Unknown bidding strategy')
        if trump not in ('heuristic', 'random') or playing not in ('heuristic', 'random'):
            raise ValueError('Trump and playing strategies must be heuristic or random')
        rng = random.Random(seed)
        self._delegates = {}
        for phase, strategy in zip((Phase.BID, Phase.TRUMP, Phase.PLAY),
                                   (bidding, trump, playing)):
            phase_seed = rng.getrandbits(128)
            self._delegates[phase] = (HeuristicAgent() if strategy == 'heuristic' else
                ConservativeRandomAgent(phase_seed) if strategy == 'conservative' else
                RandomAgent(phase_seed))

    def reset(self, seat: int) -> None:
        for delegate in self._delegates.values():
            delegate.reset(seat)

    def act(self, obs: Observation) -> int:
        if obs.phase not in self._delegates:
            raise ValueError('Unknown decision phase')
        return self._delegates[obs.phase].act(obs)


def variant_factory(variant: str):
    """Return an arena factory(seat, seed); the runner calls reset(seat)."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown policy variant: {variant}')
    bidding, trump, playing = VARIANTS[variant]

    def create(seat: int, seed: int) -> HybridAgent:
        return HybridAgent(bidding, trump, playing, seed=seed)

    return create
