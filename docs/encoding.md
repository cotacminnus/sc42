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

- **JSONL logs:** `encode` expects in-memory `Observation` objects. Logged
  observations are JSON dicts (bids as `{"points": ..., "marks": ...}`). A
  decoder from schema-5 records back to `Training*View` is the next step for
  offline training.
- **Sequence models:** the vector summarizes the current hand. An LSTM or
  transformer over the ordered public history (as in DouZero) can be added as
  a separate input without changing this layout.
- **Previous hands:** match context is only the mark totals.
