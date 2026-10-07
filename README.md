# SC42 / TX42

A command-line Texas 42 dominoes game for the Texas A&M Singing Cadets. The current
version pairs one human with a placeholder AI against two more placeholder AIs.
Partners are Players 1 and 3 versus Players 2 and 4; you are Player 1.

The repository is `sc42`, the Python package is `texas42`, and the optional installed
command is `tx42`. Running from source requires Python 3.11 or newer and no
third-party runtime dependencies.

The engine also serves as a research testbed for self-play learning under imperfect
information. See the [research roadmap](docs/roadmap.md) for goals, status, and
planned steps.

## Project map

```text
 Front ends        cli.py -> engine.Game -> players.Player (human, random bot)
                   selfplay.SelfPlayEnv  -> episodes.EpisodeState
                   openspiel.py (optional) -> episodes.EpisodeState
                                   |
 One hand          state.HandState: bid -> trump -> seven tricks (global action IDs)
                                   |
 Pure rules        rules.py (legality, trick winner, points), models.py (value types)

 Shared formats    actions.py (58 permanent action IDs), training.py (ID-based views,
                   checkpoint metadata, log migration), encoding.py (feature vectors),
                   features.py / rulesets.py / phases.py (version and rules identity)
```

| File | Responsibility |
| --- | --- |
| `texas42/models.py` | Dominoes, bids, trump, and contract value types |
| `texas42/rules.py` | Legal bids and plays, suit ranking, and trick scoring |
| `texas42/state.py` | One hand's transitions, legal action IDs, and hand results |
| `texas42/episodes.py` | Hand or match episodes: marks, shaker rotation, and rewards |
| `texas42/engine.py` | Console match runner for `Player` objects, with optional JSONL logging |
| `texas42/players.py` | Player interface, immutable decision views, human input, and the random bot |
| `texas42/cli.py` | Command-line options and player setup |
| `texas42/selfplay.py` | Silent reset/step API for training episodes |
| `texas42/training.py` | Integer-ID training views, `training_metadata()`, and log migration |
| `texas42/encoding.py` | Fixed-length feature vectors for neural policies |
| `texas42/actions.py` | Append-only action vocabulary and neural-head slots |
| `texas42/features.py`, `rulesets.py`, `phases.py` | Feature/encoder versions, ruleset identity, and phase names |
| `texas42/openspiel.py` | Optional OpenSpiel game, state, and observer backed by the engine |
| `tests/` | One test module per area (`test_game`, `test_selfplay`, `test_encoding`, `test_openspiel`, ...) |
| `docs/` | [Roadmap](docs/roadmap.md), [encoder](docs/encoding.md), [action IDs](docs/action-space.md), [OpenSpiel](docs/openspiel.md) |

Seats are zero-based everywhere: team 0 is seats 0 and 2, and team 1 is seats 1 and 3.

## Play

Run commands from the repository root (the directory containing `pyproject.toml`):

```console
python -m texas42
```

To install the optional command into your active Python environment:

```console
python -m pip install -e .
tx42
```

Installation uses setuptools as declared in `pyproject.toml`; installation is not
needed to play or run tests from the checkout.

### Controls

- **Bid:** enter `30` through `41`, `42` for one mark, or `1m` through `5m`.
  Enter `p` or `pass` to pass unless you are the forced bidder. Bids must satisfy
  the increase limits below.
- **Call trump:** enter `0` through `6` (`0` means blanks), `doubles`, or `n` for
  Follow Me/no trump. `follow` and `follow me` also select no trump.
- **Play:** enter one of the displayed legal domino numbers. These are positions
  in your current hand, starting at 1, and change as dominoes are removed.
- **Quit:** press Ctrl+C.

### Run options

```console
python -m texas42 --help
python -m texas42 --debug --seed 42
python -m texas42 --target-marks 3
```

`--debug` reveals all four hands at the start of each hand. `--seed` makes the
shuffle and bot choices reproducible when the same player decisions are made.
`--target-marks` must be positive and defaults to 7.

For a noninteractive self-play run, use `--all-ai`:

```console
python -m texas42 --all-ai --seed 42 --target-marks 1
```

Add `--self-play-log` to write `logs/selfplay.jsonl`, or supply a custom path:

```console
python -m texas42 --all-ai --seed 42 --self-play-log
python -m texas42 --all-ai --seed 42 --self-play-log logs/match.jsonl
```

Each JSON line records the acting seat's observation before its action (including
its hand, public state, legal action IDs, and mask), the chosen global action
ID and decoded action, and the resulting marks, hand result, match termination,
winner, and per-seat rewards. Match rewards are zero until the final action, then
+1 for winners and -1 for opponents. Seats are zero-based. The next line contains
the next acting seat's observation. Logs are overwritten on each run; `logs/` and
`*.jsonl` are ignored by Git. Logging requires `--all-ai` and does not train a model.
Logs use `schema_version: 5` and share action IDs with training and OpenSpiel.
Every domino is still an integer ID from 0 through 27. Schema-2 logs used
phase-local action IDs; `upgrade_selfplay_record(record)` migrates schema-2/3/4 records,
including their legal-action masks. See [action compatibility](docs/action-space.md).

## Implemented rules

- Four players, fixed cross-table partnerships, and seven dominoes per player.
- One round of bidding, starting after the shaker and ending with the shaker.
  The shaker must bid if everyone else passes.
- Point bids range from 30 through 41; mark bids range from one through five.
  Every bid must exceed the previous bid. The first mark bid can be at most two
  marks; subsequent mark bids increase by one. `42` is an alias for one mark.
- Numbered trumps, Follow Me/no trump, and doubles as a callable eighth trump suit.
  A non-trump lead calls its higher-numbered end. Trump dominoes belong to trump
  exclusively, even if their other end matches the led suit.
- Players must follow suit when able; otherwise any domino is legal. Trump beats
  the led suit. Within a numbered suit the double is highest, then the other end
  determines rank. In doubles trump, the higher double wins.
- Each trick is worth one point, plus counters for dominoes whose ends total 5
  or 10. A full hand has 42 points: seven tricks and 35 counter points.
- Point bids require at least the bid amount and are worth one mark. Every mark
  bid requires all 42 points and stakes the number of marks bid. The bidding team
  earns those marks if it makes the contract; otherwise the defending team does.
- All seven tricks are played. The bidder leads first; each trick's winner leads
  next. The shaker rotates after each hand during a match.
- The first team to reach the target number of marks wins.

The three dummy AIs pass every optional bid and play a random legal domino. If a
dummy AI is the forced bidder, the engine assigns it a bid of 30; it then chooses
a random trump call. As a result, `--all-ai` logs contain no real bidding; use
`SelfPlayEnv` for training data that covers the full auction.

Nello, Sevens, Splash, and Plunge are not implemented. The rules above describe
the current implementation; additional Singing Cadets house rules may need to be
added.

## Development

Run the standard-library test suite from the repository root:

```console
python -m unittest discover -s tests -v
```

Tests also run in GitHub Actions on pushes and pull requests: a dependency-free
core job and an OpenSpiel integration job. OpenSpiel tests skip when `pyspiel`
is not installed.

### Adding a real AI

Subclass `texas42.players.Player` and implement these decision methods, using the
types from `texas42.players` and `texas42.models`:

```python
def choose_bid(self, view: BidView) -> Bid | None: ...
def choose_trump(self, view: TrumpView) -> Trump: ...
def choose_domino(self, view: PlayView) -> Domino: ...
```

Each immutable view contains the player's own hand and the public information
needed for that decision. Opponents' hands are not exposed. `PlayView` includes
the current trick, completed tricks, hand points, and match marks. Choose a domino
from `view.legal_plays`; the engine validates returned bids, trump calls, and plays.
Returning `None` from `choose_bid` passes, or becomes a bid of 30 if forced.
For trump, use an integer from 0 through 6, `"doubles"`, or `None` for Follow Me.

Seats in the API are zero-based: team 0 has seats 0 and 2, and team 1 has seats 1
and 3. Point and mark tuples use that team order. Supply exactly four players in
seat order to `Game`; replace any of the bots below with your implementation:

```python
import random

from texas42 import Game, GameConfig
from texas42.players import RandomPlayer

players = [RandomPlayer(f"Bot {seat + 1}", random.Random(seat)) for seat in range(4)]
game = Game(players, GameConfig(seed=42, target_marks=1))
winning_team = game.play_match()  # Returns team 0 or 1.
```

`GameConfig.seed` controls the engine's shuffle and initial shaker; custom players
manage their own randomness. `play_hand()` returns a `HandResult` and updates
match marks, but only `play_match()` automatically rotates the shaker.

### Self-play training interface

`SelfPlayEnv` runs without player objects, console output, or third-party
dependencies. Both it and `Game` use the same hand transition logic. This is a
native turn-based API, not a Gymnasium or PettingZoo adapter.

```python
import random
from texas42 import SelfPlayEnv

env = SelfPlayEnv(episode="hand", seed=42)
policy_rng = random.Random(123)

for episode in range(100):
    observation = env.reset()
    trajectories = [[] for _ in range(4)]
    while True:
        # Replace this with masked sampling from your policy network.
        action_id = policy_rng.choice(observation.legal_action_ids)
        trajectories[observation.seat].append((observation, action_id))
        transition = env.step(action_id)
        if transition.terminated:
            # Hand mode: assign each seat's terminal team reward to its own
            # trajectory. A PPO collector should also store log-probs/values.
            for seat, trajectory in enumerate(trajectories):
                reward = transition.rewards[seat]
                # train/store (trajectory, reward) here
            break
        observation = transition.observation
```

`reset(seed=42)` restarts the deal random stream; `reset()` continues it.
`reset(shaker=0)` optionally selects the starting shaker. A reset discards any
previous episode, clears marks, and deals a fresh hand. Policy sampling needs
its own seeded RNG for complete reproducibility.

Each immutable `Observation` contains the acting `seat`, `phase`, player `view`,
`legal_action_ids`, and a boolean `action_mask`. Views expose only that seat's
hand and public information, including chronological bids/passes, match marks,
and the target mark count. Play views also contain the current trick and all
completed tricks. Do not pass the environment's private simulator state to a
policy. An observation remains unchanged after later steps.

Self-play views are slotted `TrainingBidView`, `TrainingTrumpView`, and
`TrainingPlayView` snapshots. They store every domino as an integer ID, including
`hand`, `legal_plays`, and the domino component of `(seat, domino_id)` entries in
`trick` and `completed_tricks`. Decode an ID with `PLAY_ACTIONS[id]`; convert a
`Domino` to its stable ID with `domino.id`. For example, `[0|0]` is 0, `[3|2]` is
8, and `[6|6]` is 27. Existing `Player` decision views still use `Domino` objects.
Code consuming earlier self-play observations must now decode IDs before calling
rules helpers that expect domino objects. Bids and trump calls retain their
existing representations.

Action IDs are **global and append-only**, shared by the engine and OpenSpiel:

| Phase | IDs | Exported lookup tuple |
| --- | --- | --- |
| `bid` | 28 = pass; 29–40 = 30–41 points; 41–45 = 1–5 marks | `BID_ACTIONS[id - 28]` |
| `trump` | 46–52 = numbered trump; 53 = doubles; 54 = no trump | `TRUMP_ACTIONS[id - 46]` |
| `play` | 0–27 = permanent domino IDs, independent of hand ordering | `PLAY_ACTIONS` |
| Future contracts | 55 = Nil, 56 = Splash, 57 = Plunge; all reserved and masked out | `ACTIONS[id]` |

All masks have `NUM_ACTIONS` entries (currently 58), regardless of phase. Decode
any player action with `ACTIONS[id].value`, or encode a decision with
`ACTION_REGISTRY.encode(phase, value)`. Observations include `num_actions` and
`action_space_version`, `feature_schema_version`, and `ruleset_id`. Save `training_metadata()` with checkpoints
and experiments. `phase_slots()` defines fixed neural heads: bid 21, trump 9,
play 28. `phase_mask()`, `global_id()`, and `phase_slot()` translate their choices
to persistent IDs. Logs also include stable `action_key` and resolved contracts.
Future actions append after existing entries; they must never shift old IDs.
Reserved contract tokens do not implement bidding, eligibility, or scoring rules.

For normalized domino `[high|low]`, the play ID is `high*(high+1)//2 + low`.
Use `PLAY_ACTIONS[action_id]` to decode it. Mask illegal logits before sampling.
The environment rejects illegal IDs with `ValueError` without changing state.
Use Python `int` IDs (convert tensor/NumPy scalars first). Pass is masked out for
a forced bidder; the legacy `Player` API still converts a forced `None` into 30.

`step(action_id)` returns a `StepResult`:

- `observation` belongs to the **next acting seat**, or is `None` at episode end.
- `rewards` is ordered by seats 0, 1, 2, 3. Partners receive equal rewards and
  opponents receive opposite rewards. These are transition rewards, not totals.
- `hand_result` is present only on a hand-ending transition and includes points,
  contract, scoring team, and marks awarded.
- `terminated` marks episode completion. Call `reset()` before stepping again.
- `match_winner` is a team index when the target marks have been reached.

With `episode="hand"` (the default), an episode is exactly one hand: four bids,
one trump choice, and 28 plays. Only the final step pays rewards: positive marks
awarded for the scoring team, negative marks for its opponents.

With `episode="match", target_marks=7`, hands continue automatically, marks
persist, and the shaker rotates between hands. Intermediate hand boundaries
return both `hand_result` and the next hand's bidding observation; they do not
terminate the episode. All rewards are zero until the match ends, then winners
receive +1 and losers -1. Reset hand-specific policy memory at hand boundaries
if appropriate, but keep match returns connected through those boundaries.

Keep separate trajectories for each seat. Terminal rewards apply to all four
seats, including seats that did not take the final action. This API provides
simulation and observations; the neural network and optimizer remain
training-code responsibilities.

### Feature encoder

`texas42.encoding.encode(observation)` turns an `Observation` into a list of
`FEATURE_SIZE` (387) floats from the acting seat's perspective: its own hand,
who played what, the current trick, revealed voids, trump, the auction, points,
and marks. `to_numpy(observations)` batches features and legal-action masks when
NumPy is installed. Save `training_metadata()` (which includes
`encoder_layout_version`) with every model. See [feature encoder](docs/encoding.md)
for the layout.

### OpenSpiel integration

The optional `python_texas42` OpenSpiel game supports hand and match episodes.
It delegates rules and transitions to this engine; the base CLI and native
self-play API still have no third-party runtime dependencies.

```console
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[openspiel]'
.venv/bin/python -m examples.openspiel_selfplay --episode match --seed 42
```

```python
import pyspiel
import texas42.openspiel  # Register the adapter before loading a game.

game = pyspiel.load_game("python_texas42(episode=match,target_marks=7)")
state = game.new_initial_state()
```

The adapter provides explicit chance dealing, legal-action masks, private
observations, perfect-recall information states, cloning, and serialization.
Domino play IDs remain 0–27; bid IDs are 28–45 and trump IDs are 46–54 in its
58-action vocabulary, including three disabled contract reservations. Native
self-play, logs, and OpenSpiel use identical player action IDs and masks.

See [OpenSpiel integration](docs/openspiel.md) for the research references,
architecture, parameter and tensor layouts, and validation details. Run
`.venv/bin/python -m unittest discover -s tests -v` to include integration tests;
they skip when OpenSpiel is not installed. OpenSpiel publishes wheels for Linux
and macOS; on Windows, use WSL2. Training is not implemented yet; see the
[research roadmap](docs/roadmap.md).

## License

[MIT](LICENSE).
