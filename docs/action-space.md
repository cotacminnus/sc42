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
from texas42 import training_metadata

metadata = training_metadata()  # Ruleset, global IDs, and per-head slot ordering.
```

OpenSpiel derives its action count and public-history one-hot width from the
same registry. Its chance outcome IDs are separate: a chance node uses shaker
IDs 0–3 or domino IDs 0–27, as before.

## Migrating earlier data

Schema-4 JSONL logs add ruleset identity, resolved-contract metadata, and stable
`action_key` alongside the global ID and decoded value. Schema-2 logs used local
IDs; schema-3 logs used global IDs but omitted these semantic fields. Convert
these base-engine formats without mutating the original:

```python
import json
from texas42 import upgrade_selfplay_record

with open("logs/old.jsonl") as source, open("logs/upgraded.jsonl", "w") as output:
    for line in source:
        output.write(json.dumps(upgrade_selfplay_record(json.loads(line))) + "\n")
```

The migration validates old IDs/masks, translates local IDs when needed, and
adds ruleset and resolved-contract metadata. It assumes known base-engine
provenance (`source_ruleset_id="cadets_base_v1"`) and rejects unsupported rulesets.
Domino IDs, rewards, and recorded public/private information retain their meanings.
It accepts schemas 2, 3, and 4; older domino-object encodings need separate conversion.
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

## Phase heads, contracts, and rules identity

Global persistence IDs do not require a 58-output network. `phase_slots(phase)`
includes reserved declarations, so heads have fixed capacities of **bid 21,
trump 9, play 28**. `implemented_for_phase(phase)` is the separate eligibility
view; `for_phase()` remains its compatibility alias.

```python
from texas42 import ACTION_REGISTRY

mask = ACTION_REGISTRY.phase_mask(observation.phase, observation.legal_action_ids)
# Sample a local head slot with this mask, then persist/apply its global ID:
action_id = ACTION_REGISTRY.global_id(observation.phase, sampled_slot)
slot = ACTION_REGISTRY.phase_slot(observation.phase, action_id)  # inverse codec
```

Bid slots 18, 19, 20 identify Nil, Splash, Plunge and remain false today.
Enabling those rules does not resize the head. Additional declarations still
require a larger head and checkpoint migration. Slot order follows global-ID
order, including noncontiguous IDs (bid includes 28–45 followed by 55–57).

`RULESET_ID = "cadets_base_v1"` identifies game semantics independently of action
IDs. It is recorded in `GameConfig`, native environments/observations, OpenSpiel
parameters, logs, and `training_metadata()` for datasets, checkpoints, and
experiments. Unknown identifiers are rejected. Changed house rules need a distinct
ruleset identity even if actions and head sizes remain unchanged.

`ContractKind` models normal/Nil/Splash/Plunge identity separately from `Bid`.
`Contract` resolves after the auction and appears in player/training views and
hand results. Special kinds are metadata, not implemented eligibility or scoring.
`Phase` lives in `phases.py` as a string enum. Future setup phases require explicit
transitions, views, and observer encoding; unhandled phases fail rather than
being treated as ordinary plays. Phase-head stability does not guarantee identical
observations if new decisions or rules change the game flow.

## Auction and feature APIs

`AuctionCall = Bid | ContractDeclaration | None` is shared by player bidding,
auction history, and training views. `ContractAction` is a compatibility alias
for `ContractDeclaration`; reserved IDs and decoded `{"name": ...}` values stay
stable. `BidView.current_bid` keeps its existing name but now accepts any auction
call. Only normal bids are legal in the base ruleset. New declarations still need
explicit eligibility, ranking, resolution, and scoring rules.

Construct results with `HandResult(contract=..., trump=..., team_points=...,
scoring_team=..., marks_awarded=...)`. A resolved `Contract` is required;
`result.bidder` and `result.bid` are derived from it and cannot be supplied
independently. Special-contract metadata may have `bid=None`. Trump/play views
also allow that value, while normal hands retain their normal bid.

Schema-5 logs add `feature_schema_version` to both the record and observation.
`FEATURE_SCHEMA_VERSION = 1` also appears in `training_metadata()` and on
OpenSpiel games. Save it with datasets, checkpoints, and experiments alongside
ruleset and action-space versions. Migration accepts old base-engine JSONL
records from schemas 2–4; it does not migrate unversioned OpenSpiel tensors.
There is still no training/checkpoint implementation in this repository.
