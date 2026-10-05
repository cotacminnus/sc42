# SC42 / TX42

A command-line Texas 42 dominoes game for the Texas A&M Singing Cadets. The current
version pairs one human with a placeholder AI against two more placeholder AIs.
Partners are Players 1 and 3 versus Players 2 and 4; you are Player 1.

The repository is `sc42`, the Python package is `texas42`, and the optional installed
command is `tx42`. Running from source requires Python 3.11 or newer and no
third-party runtime dependencies.

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

For a quick noninteractive smoke run, use the internal `--all-ai` option (hidden
from `--help`):

```console
python -m texas42 --all-ai --seed 42 --target-marks 1
```

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
a random trump call.

Nello, Sevens, Splash, and Plunge are not implemented. The rules above describe
the current implementation; additional Singing Cadets house rules may need to be
added.

## Development

Run the standard-library test suite from the repository root:

```console
python -m unittest discover -s tests -v
```

| File | Responsibility |
| --- | --- |
| `texas42/models.py` | Dominoes, bids, and trump representations |
| `texas42/rules.py` | Legal bids and plays, suit ranking, and trick scoring |
| `texas42/engine.py` | Dealing, bidding, hand progression, and match scoring |
| `texas42/players.py` | Player interface, immutable decision views, human input, and bots |
| `texas42/cli.py` | Command-line options and player setup |
| `tests/test_game.py` | Rule checks and automated game smoke tests |

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

## License

[MIT](LICENSE).
