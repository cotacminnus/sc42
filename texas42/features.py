"""Policy-input compatibility, independent of action IDs and house rules."""

# First explicitly versioned layout. Unversioned OpenSpiel tensors included
# ordered partial deals and must be regenerated, not relabeled as version 1.
FEATURE_SCHEMA_VERSION = 1
PUBLIC_DECISIONS_PER_HAND = 33  # Four calls, one trump choice, 28 plays.
OBSERVER_PHASE_SLOTS = 5  # Chance, bid, trump, play, terminal.

# Layout of texas42.encoding's native feature vector. Bump on any change to its
# fields, order, or scaling; saved models are tied to one layout.
ENCODER_LAYOUT_VERSION = 1

# Chronological public action sequences; independent of the summary vector.
PUBLIC_HISTORY_VERSION = 1
