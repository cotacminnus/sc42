# Stable action IDs and future contracts

`texas42.actions` is the single player-action vocabulary for `HandState`, native
training observations, CLI logs, and the optional OpenSpiel adapter. IDs are
permanent: never insert, reorder, delete, or reuse entries. Append new actions
at the end even when they belong to an existing phase.

| IDs | Action |
| --- | --- |
| 0–27 | Domino plays, equal to `Domino.id` |
| 28 | Pass |
| 29–40 | Point bids 30–41 |
| 41–45 | Mark bids 1–5 |
| 46–52 | Numbered trump 0–6 |
| 53 | Doubles trump |
| 54 | No trump / Follow Me |
| 55 | Reserved Nil declaration |
| 56 | Reserved Splash declaration |
| 57 | Reserved Plunge declaration |

Nil, Splash, and Plunge are **not implemented**.
Their `ContractAction` tokens only reserve identities. Eligibility, stakes,
partner participation, trick rules, and scoring still need explicit decisions
and engine implementation. These actions are absent from legal actions, have
false mask entries in every phase, and are rejected without changing state.

## Training and OpenSpiel

`ACTIONS[id]` returns an immutable `ActionSpec` with its stable key, phase,
decoded value, and implementation status. `ACTION_REGISTRY.encode(phase, value)`
returns the unique ID of an implemented decision. Phase distinguishes pass from
no trump, which both decode to `None`.

Both native training and OpenSpiel use a single `NUM_ACTIONS`-entry mask, currently
58. Use masked sampling over this vocabulary. Reserved logits never become legal.
Native observations also expose `num_actions` and `action_space_version`;
OpenSpiel games expose `action_space_version`. Store the registry's manifest
with each dataset or model checkpoint:

```python
from texas42 import ACTION_REGISTRY, ACTION_SPACE_VERSION, NUM_ACTIONS

metadata = {
    "action_space_version": ACTION_SPACE_VERSION,
    "num_actions": NUM_ACTIONS,
    "action_manifest": ACTION_REGISTRY.manifest(),
}
```

OpenSpiel derives its action count and public-history one-hot width from the
same registry. Its chance outcome IDs are separate: a chance node uses shaker
IDs 0–3 or domino IDs 0–27, as before.

## Migrating earlier data

Schema-3 JSONL logs use global IDs and full masks. Schema-2 logs used local IDs
for bidding and trump. Convert a record without mutating the original:

```python
import json
from texas42 import upgrade_selfplay_record

with open("logs/old.jsonl") as source, open("logs/upgraded.jsonl", "w") as output:
    for line in source:
        output.write(json.dumps(upgrade_selfplay_record(json.loads(line))) + "\n")
```

The migration validates the old legal IDs and mask, translates the selected
action and legal choices, expands the mask, and adds vocabulary metadata.
Domino IDs, views, rewards, and hand results retain their meanings. It accepts
schema 2 only; logs with older domino-object encodings need a separate conversion.
`legacy_action_id(phase, local_id)` is available for individual decisions.

Old phase-specific policy heads need their output rows mapped into the new global
head (play offset 0, bid offset 28, trump offset 46). Old OpenSpiel heads already
used 0–54; extend their output to 58 with the same first 55 rows. Dataset migration
does not migrate model weights or optimizer state. OpenSpiel public-history
inputs also grew: pad each history slot's action columns, using the named
observer fields to reconstruct tensors. Padding only the end of an old flattened
observation does not preserve the layout.

## Adding contract rules later

1. Use the reserved token/ID for Nil, Splash, or Plunge. For additional
   declarations, append a new `ActionSpec` to the registry. Keep stable keys and
   IDs; the registry checks contiguous IDs and unique keys.
2. Implement contract eligibility and transitions in the engine. Extend contract
   models and player/training views where needed. Enabling a registry entry
   alone does not add a playable contract.
3. Include implemented declarations in `HandState` legality and apply logic.
   Revisit episode rewards and OpenSpiel utility/length bounds if stakes or the
   number of participants/decisions differ from ordinary hands.
4. Enable the entry only after those rules exist, bump the action-space version,
   and add engine/native/OpenSpiel parity tests plus privacy and replay checks.

`ActionRegistry.appended(...)` creates an immutable larger registry and preserves
all existing entries. For example, a future reservation can be prototyped with:

```python
from texas42 import ACTION_REGISTRY, ActionSpec, ContractAction, NUM_ACTIONS

extended = ACTION_REGISTRY.appended(
    ActionSpec(NUM_ACTIONS, "contract.future", "bid", ContractAction("future"), False)
)
assert extended.entries[:NUM_ACTIONS] == ACTION_REGISTRY.entries
```

This helper does not change a running game. Wire the new registry into the module
when adding the actual feature so all front ends get the same capacity. A wider
vocabulary preserves old action identities, but model output/input shapes and
metadata must still be migrated deliberately.
