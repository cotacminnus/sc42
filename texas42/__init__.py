"""Texas 42 command-line game."""

from .engine import Game, GameConfig
from .models import Bid, Domino
from .selfplay import Observation, SelfPlayEnv, StepResult
from .state import BID_ACTIONS, PLAY_ACTIONS, TRUMP_ACTIONS, HandResult
from .training import (
    TrainingBidView, TrainingPlayView, TrainingTrumpView, TrainingView, upgrade_selfplay_record,
)
from .actions import (
    ACTIONS, ACTION_REGISTRY, ACTION_SPACE_VERSION, NUM_ACTIONS,
    BID_OFFSET, PLAY_OFFSET, TRUMP_OFFSET, NIL_ACTION_ID, SPLASH_ACTION_ID,
    PLUNGE_ACTION_ID, ActionSpec, ActionRegistry, ContractAction, legacy_action_id,
)

__all__ = [
    "Bid", "Domino", "Game", "GameConfig", "HandResult", "SelfPlayEnv",
    "Observation", "StepResult", "BID_ACTIONS", "TRUMP_ACTIONS", "PLAY_ACTIONS",
    "TrainingView", "TrainingBidView", "TrainingTrumpView", "TrainingPlayView",
    "ACTIONS", "ACTION_REGISTRY", "ACTION_SPACE_VERSION", "NUM_ACTIONS",
    "PLAY_OFFSET", "BID_OFFSET", "TRUMP_OFFSET", "NIL_ACTION_ID", "SPLASH_ACTION_ID",
    "PLUNGE_ACTION_ID", "ActionSpec", "ActionRegistry", "ContractAction", "legacy_action_id",
    "upgrade_selfplay_record",
]
