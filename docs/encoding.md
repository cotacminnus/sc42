# Feature encoder

`texas42/encoding.py` converts what one seat can see into a fixed-length vector
for a neural network. It is roadmap step 2 ([roadmap.md](roadmap.md)) and the
input format for steps 6–8.

## Data flow

```text
SelfPlayEnv.reset()/step()  ->  Observation  ->  encode(obs)  ->  list[float] (FEATURE_SIZE = 387)
  texas42/selfplay.py           .view: TrainingBidView |          texas42/encoding.py
                                       TrainingTrumpView |
                                       TrainingPlayView (texas42/training.py)
                                .action_mask: 58 bools   ->  legal-action mask for the policy head

many Observations  ->  to_numpy(observations)  ->  features float32 [B, 387], masks bool [B, 58]
```

| Function | Input | Output |
| --- | --- | --- |
| `encode(obs)` | `Observation` from `SelfPlayEnv` | `list[float]` of length `FEATURE_SIZE` |
| `encode_view(view)` | any `Training*View` | same as above |
| `to_numpy(observations)` | iterable of `Observation` | `(features, masks)` NumPy arrays; imports NumPy lazily |
| `action_one_hot(action_id)` | global action ID 0–57 | `list[float]` of length 58, for Q(state, action) networks |
| `revealed_voids(play_view)` | `TrainingPlayView` | set of `(absolute seat, suit)` proven empty |
| `relative_seat(seat, actor)` | two absolute seats | 0 actor, 1 left opponent, 2 partner, 3 right opponent |

The core package stays dependency-free. Only `to_numpy` needs NumPy, which is
installed with the `openspiel` extra or separately.

## Perspective

All seats are rotated to the acting seat, and team values are ordered (actor's
team, other team). One network can therefore play every seat, and seat 0 and
seat 1 produce the same kind of input. Only the actor's own hand and public
information are encoded. Hidden hands are never inputs.

## Layout (version 1)

Fields appear in this order; multi-row fields are flattened row by row. Address
them with `vector[FEATURE_SLICES["name"]]`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `phase` | 3 | bid, trump, play one-hot |
| `hand` | 28 | actor's remaining dominoes, by domino ID |
| `played` | 4×28 | which relative seat played each domino this hand |
| `trick` | 3×28 | dominoes in the current trick, by play position |
| `led_suit` | 8 | suit led in the current trick: 0–6, doubles |
| `voids` | 3×8 | suits relative seats 1–3 have failed to follow |
| `trump` | 9 | 0–6, doubles, no trump (play phase only) |
| `bidder` | 4 | current high bidder or contract winner, relative seat |
| `bid` | 21 | current high call or contract, in bid phase-slot order |
| `calls` | 4×21 | each relative seat's auction call (pass is slot 0) |
| `shaker` | 4 | shaker, relative seat |
| `forced` | 1 | actor must bid |
| `points` | 2 | hand points ÷ 42 |
| `marks` | 2 | match marks ÷ target marks |
| `tricks_done` | 1 | completed tricks ÷ 7 |

Every value is in [0, 1]. Fields that do not apply to the current phase are zero.

`voids` is derived information, not raw observation: it encodes a simple
inference from `rules.follows_suit`. For the belief-model ablations in roadmap
step 7, zero `FEATURE_SLICES["voids"]` to give a network only raw observations.

## Using it in a training loop

```python
from texas42 import SelfPlayEnv
from texas42.encoding import FEATURE_SIZE, encode

env = SelfPlayEnv(seed=0)
obs = env.reset()
while obs is not None:
    x = encode(obs)                    # policy input, length FEATURE_SIZE
    mask = obs.action_mask             # 58 bools; set illegal logits to -inf
    action_id = policy(x, mask)        # must return an int from obs.legal_action_ids
    obs = env.step(action_id).observation
```

With PyTorch, convert a batch from `to_numpy` with `torch.from_numpy(...)`. Return
Python `int` action IDs to the environment.

## Versioning

`ENCODER_LAYOUT_VERSION` (in `features.py`) is saved by `training_metadata()`.
Bump it whenever a field, its order, its size, or its scaling changes, and do
not load a model trained on another version. It is separate from
`FEATURE_SCHEMA_VERSION`, which versions OpenSpiel observer tensors and log records.

## Not covered yet

- **Sequence models:** chronological public history is available separately,
  but recurrent models and neural training are not implemented.
- **Previous hands:** match context is only the mark totals.

## Offline JSONL observations

`loads_observation(line)` loads the pre-action observation from a schema-5
JSONL transition. `observation_from_record(record)` accepts an already parsed
record; `observation_from_dict(record["observation"])` accepts its observation
member. All return `Observation` with the appropriate `TrainingBidView`,
`TrainingTrumpView`, or `TrainingPlayView`. `Phase`, `ContractKind`, `Contract`,
`Bid`, `ContractDeclaration`, and nested tuples are restored without modifying
the input. Trump values remain integers, `"doubles"`, or `None` (Follow Me).

```python
from texas42 import loads_observation
from texas42.encoding import encode

with open("logs/selfplay.jsonl") as source:
    for line in source:
        obs = loads_observation(line)
        features = encode(obs)  # exactly the same 387 values as during rollout
```

The loader rejects incompatible record, feature, action-space, ruleset, and
vocabulary versions, mismatched view phases, and inconsistent masks. Optional
encoder, public-history, and privileged-label versions are also checked when
present. Existing schema-5 CLI logs need no encoder or history field. Upgrade
schema-2/3/4 records with `upgrade_selfplay_record()` before loading. This loader
validates policy input structure and provenance, not the legality of an entire
recorded game or its reward labels; replay the transitions for that purpose.

## Lossless public history (version 1)

`public_history(obs.view)` returns an immutable `PublicHistory`. Its `events`
are chronological `PublicEvent(phase, seat, action_id)` entries: auction calls
(including passes), then the chosen trump, then every played domino in trick
order. Only decisions preceding the observation are included. At the trump
view the trump action has not happened yet. Each four play events forms a
completed trick; the last zero to three are the current trick. The sequence
resets at hand boundaries. No draws, hidden hands, or prior hands are included.

`PUBLIC_HISTORY_VERSION` versions this representation independently and is
included in `training_metadata()`. It does not change `FEATURE_SCHEMA_VERSION`,
`ENCODER_LAYOUT_VERSION`, action IDs, or `encode()` output. The 387-value vector
summarizes played dominoes; this sequence preserves their full order for later
research. No recurrent model is implemented.

Seats are absolute for replay. `history.relative_events(obs.seat)` yields the
same sequence with actor-relative seats, invariant under rotating all seats.
Serialize with `dataclasses.asdict(history)` and restore with
`PublicHistory.from_dict(data)`. CLI and collector records include this as a
separate `public_history` field. Older schema-5 records can reconstruct it from
the observation. To reconstruct the final hand, append the terminal record's
`PublicEvent(obs.phase, obs.seat, record["action_id"])`; there is no terminal
policy observation. Context such as shaker, marks and target remains in the
record/observation, rather than being duplicated in each event.

## Privileged critic features

`texas42.privileged.encode_privileged(obs, hands)` is a separate, training-only
vector for a centralized critic: this 387-feature actor vector followed by all four
remaining hands (relative seats), the trick leader, the next player and a
perspective field, 509 floats in total (`PRIVILEGED_LAYOUT_VERSION = 1`). The actor
encoder above is unchanged and never receives these fields. See
[critic modes](multiphase_selfplay.md#critic-modes-returns-and-oracle-auxiliary-supervision).

## Privileged collection

`TrainingCollector` streams schema-5 transition dictionaries to a caller-owned
sink. Its default records contain only policy observations, public history and
transition labels. Full-deal labels require explicit opt-in:

```python
import json
from texas42 import SelfPlayEnv, TrainingCollector

with open("logs/labeled.jsonl", "w") as output:
    collector = TrainingCollector(
        lambda record: output.write(json.dumps(record) + "\n"),
        include_initial_deal=True,
    )
    env = SelfPlayEnv(seed=0, collector=collector)
    obs = env.reset()
    while obs is not None:
        # Replace with a policy accepting ONLY obs; this example picks the first legal ID.
        action = obs.legal_action_ids[0]
        obs = env.step(action).observation
```

Opted-in records add `privileged = {"version": 1, "initial_deal": ...}` outside
`observation`. The deal contains four absolute-seat hands, each seven sorted
permanent domino IDs. It captures original ownership, not remaining hands or
meaningless draw order. It is repeated per transition so shuffled samples keep
their labels. Episode and hand numbers are local to one collector; use a
separate run identifier when combining workers' output.

The terminal transition retains the old hand's deal even when match mode deals
a new hand before returning. Reset starts a new numbered episode. Invalid
actions emit no records. Sinks run synchronously after successful actions and
before the next deal; sink errors propagate, so treat them as rollout failures
rather than retrying the action. Use one collector per environment.

`Observation`, its views, and `encode()` have no privileged field or collector
reference. Offline observation loading also discards enclosing labels. Keep
collectors and labeled records with the rollout/training runner and pass only
observations to ordinary policies. This is an API separation, not a sandbox
against Python code that can inspect an environment. Labels are intended for
future belief/oracle/critic training; those models are not implemented here.
Generated `.jsonl` files and `logs/` remain gitignored.

## Episode validation

```python
import json
from texas42.validation import validate_episode, ValidationError

with open("logs/labeled.jsonl") as source:
    records = [json.loads(line) for line in source]
result = validate_episode(records, episode="hand")  # or "match"
print(result.status, result.records, result.hands, result.limitations)
```

Pass one complete, ordered episode, beginning with hand 1 and zero marks and
ending at termination. Split multi-episode collector output by episode ID first.
CLI logs have no episode ID and use `episode="match"`. The mode is required
because schema 5 does not store the reward convention; the format is unchanged.
Empty, truncated, reordered, or inconsistent episodes raise `ValidationError`
with a record index. Input records are not mutated.

`verified_replay` means every record has a privileged initial deal and agrees
with `EpisodeState`/`HandState` replay: actor observations, legal IDs and masks,
selected actions and decoded labels, chronological public history, contracts,
hand results, marks, rewards, winner and termination. Repeated deal copies must
agree. Private hand order is ignored because collection records ownership, not
draw order; public decision and trick ordering remain exact.

`partial` means some records have no privileged deal. For hands without any
recorded deal, the validator reconstructs candidate ownership from the complete
public play sequence and checks its consistency with observations and engine
transitions. Supplied deals are still checked. This can catch legality, history,
scoring and metadata errors but does not verify original ownership against an
independently captured initial deal. A partial result must not be presented as
verified replay. Neither status proves RNG provenance or dataset authenticity.
The validator requires complete episodes; it does not certify partial rollouts.

A collector sink exception aborts the current rollout: the original exception
propagates, and subsequent `step()` and `observe()` calls fail until `reset()`.
The action already advanced the engine and a sink may already have written
part of its output. Do not retry that action or treat the unfinished episode
as valid training data. There is no transactional rollback or deduplication.
