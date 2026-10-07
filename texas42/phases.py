"""Engine phases; extend this enum when a contract needs a new decision step."""
from enum import StrEnum


class Phase(StrEnum):
    BID = "bid"
    TRUMP = "trump"
    PLAY = "play"
    # No setup phase is active yet. Adding one requires explicit transitions,
    # views, and observer encoding; it must never fall through as a domino play.
