# OpenSpiel integration

The optional adapter registers `python_texas42` with OpenSpiel. It uses the
existing engine rather than implementing a second set of Texas 42 rules.

## Installation and loading

From the checkout, using Python 3.11 or newer:

```console
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[openspiel]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m examples.openspiel_selfplay --episode match --seed 42
```

```python
import pyspiel
import texas42.openspiel  # Registers the game; importing texas42 alone does not.

hand_game = pyspiel.load_game("python_texas42")
match_game = pyspiel.load_game(
    "python_texas42(episode=match,target_marks=7,shaker=-1)"
)
state = hand_game.new_initial_state()
```

Parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `episode` | `hand` | `hand` or `match` |
| `target_marks` | `7` | Positive match target, also included in game parameters for hand episodes |
| `ruleset_id` | `cadets_base_v1` | Versioned engine rules; other variants are rejected until implemented |
| `shaker` | `-1` | Random starting shaker through chance; `0`–`3` fixes the starting seat |

## Engine ownership

```text
OpenSpiel algorithms / RL environment
                |
       texas42.openspiel
                |
       texas42.episodes.EpisodeState  <-- texas42.selfplay.SelfPlayEnv
                |
       texas42.state.HandState        <-- texas42.engine.Game / CLI
                |
       texas42.rules / texas42.models
```

`HandState` owns bidding, legal moves, trump selection, trick resolution, and hand
scoring. `EpisodeState` owns match marks, terminal rewards, and shaker rotation.
The adapter supplies explicit chance outcomes, translates action IDs, and encodes
observations. OpenSpiel and NumPy are optional dependencies of the integration;
no engine module imports either of them. Training remains a caller responsibility.

## OpenSpiel contract

The game has four sequential players, imperfect information, explicit stochastic
chance, zero-sum team returns, and terminal rewards. Seats 0/2 partner against
1/3. A hand returns signed marks staked, bounded by ±5; a match returns ±1 only at
match end. Nonterminal returns/rewards are zero, including intermediate match
hand boundaries.

The initial chance node selects a shaker uniformly unless fixed. Each hand then
has 28 chance nodes that deal domino IDs without replacement, seven contiguous
draws per seat. The chance probabilities are uniform over the remaining IDs.
There is no hidden RNG inside the adapter; an external sampler controls chance.

Player action IDs use one fixed vocabulary:

| IDs | Meaning | Engine mapping |
| --- | --- | --- |
| 0–27 | Play domino | Same permanent domino ID as `SelfPlayEnv` |
| 28–45 | Bid or pass | `BID_ACTIONS[id - 28]` |
| 46–54 | Call trump | `TRUMP_ACTIONS[id - 46]` |
| 55–57 | Reserved Nil/Splash/Plunge declarations | Disabled until rules exist |

Chance actions reuse 0–3 for a shaker or 0–27 for a domino, distinguished by the
chance state. Legal actions are sorted, phase-specific, and empty for inactive
players and terminal states. `state.legal_actions_mask()` has `NUM_ACTIONS`
entries (currently 58). Native `SelfPlayEnv`, schema-4 JSONL logs, and OpenSpiel
now share these IDs directly; no phase offset is added at the adapter boundary.
The shared registry is append-only. See [action compatibility](action-space.md)
for dataset migration and the extension procedure.

A hand has exactly 33 player decisions. A match needs at most
`2 * target_marks - 1` hands because every hand awards at least one mark.
`max_game_length()` bounds player decisions; `max_chance_nodes_in_history()`
separately bounds dealing and the optional starting-shaker choice.

## Policy inputs and recall

Use `state.observation_tensor(player)` or
`state.information_state_tensor(player)` for policies. Both are fixed-size and
available for any seat, including inactive seats and terminal states. The
observer exposes a flat float32 tensor and named NumPy views through `dict`:

| Field | Encoding |
| --- | --- |
| `player` | Four-seat one-hot |
| `hand` | Current private hand as a 28-bit domino membership mask |
| `private_deals` | Own original deal; ordered seven-by-28 one-hot draws per hand for perfect recall, a 28-bit set for ordinary observations |
| `actor` | Four seats plus chance/terminal one-hot |
| `phase` | Deal/bid/trump/play/terminal one-hot |
| `hand_index` | Zero-based hand number |
| `marks`, `points` | Two team totals each |
| `deal_counts` | Number of dominoes dealt to each seat, with no opponent IDs |
| `shakers` | Four-seat one-hot per retained hand |
| `history` | 33 ordered public decision slots per retained hand; each slot has a four-seat one-hot and `NUM_ACTIONS`-action one-hot |

Ordinary observations retain the current hand. Information states preserve
perfect recall: all previous own deals (including their observed order), shakers,
and public actions throughout a match. Tensor capacity is fixed at the maximum
number of hands for the configured target; unused slots are zero. The strings
encode the same fields sparsely and omit hidden chance outcomes.

The default observer exposes only the requested player's hand. Custom
`IIGObservationType` options support no private information or explicitly all
players' private information, and can omit public information. The latter all
players option is intended for callers that explicitly request it.

`str(state)`, `state.history()`, and the simulator's `session`/`deals` contain
hidden information for debugging, cloning, and replay. They are not policy inputs.

## Validation and limits

Tested against OpenSpiel 1.6.15 with its C++ `random_sim_test`, native
serialization, pickle, independent clones/children, and the Python
`rl_environment.Environment`. Integration tests replay native self-play hand and
match episodes, compare legal actions and returns, and verify privacy and recall.
Without the optional dependency, engine tests run and integration tests skip.

This is a registered Python game, not an upstream OpenSpiel built-in. Import the
adapter in each process before loading/deserializing it. Python transition
performance is not the same as a compiled C++ game. The four-seat partnership
model also does not make every OpenSpiel algorithm applicable: algorithms limited
to two players or perfect information need a different model or method.

Explicit dealing has a very large chance tree. Prefer sampled rollouts for
training; full-tree tabular enumeration of every deal is impractical. No neural
network, optimizer, or training loop is implemented here.

## Research references

The interface follows OpenSpiel's own
[Python Kuhn Poker implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/games/kuhn_poker.py),
which demonstrates Python game registration, explicit chance, and observers.
The [core API reference](https://github.com/google-deepmind/open_spiel/blob/master/docs/api_reference.md)
and [spiel.h](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h)
define state methods and metadata. Observer integration follows
[observation.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/observation.py).
Validation follows the upstream
[game simulation tests](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/tests/games_sim_test.py)
and [RL environment](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_environment.py).

Ruleset identity is a game parameter and is preserved by serialization.
Use `training_metadata(game.ruleset_id)` to record matching dataset/checkpoint
provenance. See [action compatibility](action-space.md) for stable 21/9/28 phase
heads, resolved-contract types, and schema-2/3 log migration.
