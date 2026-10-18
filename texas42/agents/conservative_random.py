"""Pass-first random baseline using only actor-visible legal actions."""
import random

from ..actions import ACTION_REGISTRY
from ..models import Bid
from ..phases import Phase
from ..selfplay import Observation


class ConservativeRandomAgent:
    """Pass when possible, otherwise bid minimally; random trump and play.

    reset assigns a seat without reseeding. Construct a fresh instance with the
    same seed to replay the stream. Bidding consumes no random draws.
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
        if obs.phase == Phase.BID:
            return self._bid(obs)
        if obs.phase not in (Phase.TRUMP, Phase.PLAY):
            raise ValueError('Unknown decision phase')
        return self._rng.choice(obs.legal_action_ids)

    @staticmethod
    def _bid(obs: Observation) -> int:
        """Deterministic auction rule shared by act and action_probabilities."""
        pass_id = ACTION_REGISTRY.encode(Phase.BID, None)
        if pass_id in obs.legal_action_ids:
            return pass_id
        bids = [(i, ACTION_REGISTRY[i].value) for i in obs.legal_action_ids
                if isinstance(ACTION_REGISTRY[i].value, Bid)]
        return min(bids, key=lambda item: item[1].strength)[0]

    def action_probabilities(self, obs: Observation) -> dict[int, float]:
        """Pure ActionProbabilities query: one-hot bid, uniform trump/play, no RNG draw."""
        if self._seat != obs.seat:
            raise RuntimeError('Reset this agent for the acting seat before use')
        if not obs.legal_action_ids:
            raise ValueError('Observation has no legal actions')
        if obs.phase == Phase.BID:
            return {self._bid(obs): 1.0}
        if obs.phase not in (Phase.TRUMP, Phase.PLAY):
            raise ValueError('Unknown decision phase')
        share = 1.0 / len(obs.legal_action_ids)
        return {action_id: share for action_id in obs.legal_action_ids}

    def would_pass_if_allowed(self, obs: Observation) -> bool:
        """Pure diagnostic: this policy always prefers passing in an auction."""
        if obs.phase != Phase.BID:
            raise ValueError('Expected a bidding observation')
        return True
