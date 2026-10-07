"""Policy-input compatibility, independent of action IDs and house rules."""

# First explicitly versioned layout. Unversioned OpenSpiel tensors included
# ordered partial deals and must be regenerated, not relabeled as version 1.
FEATURE_SCHEMA_VERSION = 1
PUBLIC_DECISIONS_PER_HAND = 33  # Four calls, one trump choice, 28 plays.
OBSERVER_PHASE_SLOTS = 5  # Chance, bid, trump, play, terminal.
