"""Bridge private object-based Player views to the common Agent protocol."""
from ..actions import ACTION_REGISTRY
from ..models import AuctionCall, Domino, Trump
from ..players import BidView, Player, PlayView, TrumpView
from ..selfplay import observation_from_view
from .base import Agent


class AgentPlayer(Player):
    """Use one adapter and one agent instance per seat; Game handles reset.

    Direct callers must call reset(seat) before requesting decisions. The agent
    only receives an immutable Observation; no state or environment is supplied.
    """

    def __init__(self, name: str, agent: Agent):
        super().__init__(name)
        self.agent = agent
        self._seat: int | None = None

    def reset(self, seat: int) -> None:
        if type(seat) is not int or seat not in range(4):
            raise ValueError('seat must be an integer from 0 to 3')
        self.agent.reset(seat)
        self._seat = seat

    def _choose(self, view):
        if self._seat != view.seat:
            raise RuntimeError('Reset the adapter for this seat before use')
        obs = observation_from_view(view)
        action_id = self.agent.act(obs)
        # Membership alone accepts True == 1 and 46.0 == 46. Require exact int,
        # then the authoritative mask excludes reserved/wrong-phase actions.
        if type(action_id) is not int or action_id not in obs.legal_action_ids:
            raise ValueError(f'Agent returned an illegal {obs.phase} action ID: {action_id!r}')
        return ACTION_REGISTRY[action_id].value

    def choose_bid(self, view: BidView) -> AuctionCall:
        return self._choose(view)

    def choose_trump(self, view: TrumpView) -> Trump:
        return self._choose(view)

    def choose_domino(self, view: PlayView) -> Domino:
        return self._choose(view)
