"""Uniform legal-action baseline, including auction raises."""
import random

from ..selfplay import Observation


class RandomAgent:
    """Seed once at construction. reset clears seat assignment, not the RNG.

    Equal seeds and equal reset/observation sequences produce equal actions.
    Construct a fresh instance with the same seed to replay the random stream.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._seat: int | None = None

    def reset(self, seat: int) -> None:
        if type(seat) is not int or seat not in range(4):
            raise ValueError('seat must be an integer from 0 to 3')
        self._seat = seat

    def act(self, obs: Observation) -> int:
        if self._seat != obs.seat:
            raise RuntimeError('Reset this agent for the acting seat before use')
        if not obs.legal_action_ids:
            raise ValueError('Observation has no legal actions')
        return self._rng.choice(obs.legal_action_ids)
