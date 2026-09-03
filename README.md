# TX42

A dependency-free command-line Texas 42 game for one human and three placeholder AI
players. Partners are Players 1 and 3 versus Players 2 and 4.

## Play

Python 3.11 or newer is required.

```console
python -m texas42
```

Reveal all four hands for debugging and use a repeatable shuffle:

```console
python -m texas42 --debug --seed 42
```

Run the tests:

```console
python -m unittest discover -s tests -v
```

## Implemented rules

- Four players, fixed cross-table partnerships, and seven dominoes per player.
- Bids from 30 through 41, followed by one through five marks.
- Forced minimum bid by the shaker if the other players pass.
- Numbered trumps, Follow Me/no trump, and doubles as a callable eighth trump suit.
- Normal follow-suit, trump, counter, trick, set, and mark scoring rules.
- First team to seven marks wins by default.
- The three dummy AIs pass every optional bid and play a random legal domino. If a
  dummy AI is the forced bidder, the engine assigns it a bid of 30 and it chooses a
  random trump call.

Nello, Sevens, Splash, and Plunge are intentionally not implemented yet.

## Adding a real AI

Subclass `texas42.players.Player` and implement:

```python
def choose_bid(self, view: BidView) -> Bid | None: ...
def choose_trump(self, view: TrumpView) -> Trump: ...
def choose_domino(self, view: PlayView) -> Domino: ...
```

Each immutable view contains the AI's own hand and public match state. Opponents'
hands are not exposed. The engine validates bids, trump calls, and plays returned by
the player implementation.
