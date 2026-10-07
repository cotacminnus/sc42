"""Texas 42 engine, command-line game, and self-play research interfaces.

Start with README.md (project map) and docs/roadmap.md (research plan).
"""

from .encoding import ENCODER_LAYOUT_VERSION, FEATURE_SIZE, encode
from .engine import Game, GameConfig
from .features import FEATURE_SCHEMA_VERSION
from .models import AuctionCall, ContractDeclaration, Bid, Contract, ContractKind, Domino
from .phases import Phase
from .rulesets import RULESET_ID, enabled_contracts
from .selfplay import Observation, SelfPlayEnv, StepResult
from .state import BID_ACTIONS, PLAY_ACTIONS, TRUMP_ACTIONS, HandResult
from .training import (
    TrainingBidView, TrainingPlayView, TrainingTrumpView, TrainingView, upgrade_selfplay_record,
    training_metadata,
)
from .actions import (
    ACTIONS, ACTION_REGISTRY, ACTION_SPACE_VERSION, NUM_ACTIONS,
    BID_OFFSET, PLAY_OFFSET, TRUMP_OFFSET, NIL_ACTION_ID, SPLASH_ACTION_ID,
    PLUNGE_ACTION_ID, ActionSpec, ActionRegistry, ContractAction, legacy_action_id,
)

__all__ = [
    "AuctionCall", "ContractDeclaration", "FEATURE_SCHEMA_VERSION",
    "ENCODER_LAYOUT_VERSION", "FEATURE_SIZE", "encode",
    "Bid", "Domino", "Game", "GameConfig", "HandResult", "SelfPlayEnv",
    "Observation", "StepResult", "BID_ACTIONS", "TRUMP_ACTIONS", "PLAY_ACTIONS",
    "TrainingView", "TrainingBidView", "TrainingTrumpView", "TrainingPlayView",
    "ACTIONS", "ACTION_REGISTRY", "ACTION_SPACE_VERSION", "NUM_ACTIONS",
    "PLAY_OFFSET", "BID_OFFSET", "TRUMP_OFFSET", "NIL_ACTION_ID", "SPLASH_ACTION_ID",
    "PLUNGE_ACTION_ID", "ActionSpec", "ActionRegistry", "ContractAction", "legacy_action_id",
    "upgrade_selfplay_record",
    "Contract", "ContractKind", "Phase", "RULESET_ID", "enabled_contracts", "training_metadata",
]
