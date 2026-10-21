# SC42 / TX42

SC42 is a command-line Texas 42 game and a reproducible research testbed for
imperfect-information partnership learning. The Python package is `texas42`;
the installed console command is `tx42`. Seats are zero-based: team 0 occupies
0/2 and team 1 occupies 1/3. The console pairs one human with a random bot against
two random bots; research agents are available through the common agent API.

## Installation and quick start

Python 3.11+ is required. The engine runs from source without third-party runtime
dependencies. From the repository root:

```bash
python -m pip install -e .
python -m texas42
python -m unittest discover -s tests -v
```

Optional extras are `.[rl]` (PyTorch/NumPy training and neural evaluation) and
`.[openspiel]` (OpenSpiel integration). Install only the extras you need.
See the [command reference](docs/command_reference.md) for all experiment commands,
arguments, resume recipes and output locations.

### Console controls

- Bid `30`–`41`, `42` for one mark, or `1m`–`5m`; `p` passes when legal.
- Trump: `0`–`6`, `doubles`, or `n`/`follow me` for no trump.
- Play the displayed legal domino number; these are hand positions, not global IDs.
- Ctrl+C quits. `--debug` reveals hands; `--seed` reproduces shuffles and bot choices.
- `--all-ai` runs bots; `--self-play-log [PATH]` records schema-5 JSONL and requires
  `--all-ai`. Logs overwrite their destination, default `logs/selfplay.jsonl`.

## Current capabilities and guides

For a new PPO study, start with the [training workflow finder](docs/command_reference.md#workflow-finder):
pretrain play, select trainable/frozen phases, train against HHH or frozen historical
opponents, then run a controlled held-out comparison. The [mechanics guide](docs/multiphase_selfplay.md)
explains sample counts, rewards, source identities and resume. Use the
[legacy play-only guide](docs/ppo_training.md) when reproducing version-1 experiments.

| Topic | Authoritative guide |
| --- | --- |
| Commands, arguments, JSON/log/CSV paths | [Command reference](docs/command_reference.md) |
| Original play-only PPO and version-1 reproduction | [Legacy PPO](docs/ppo_training.md) |
| Independently train/freeze bidding, trump and play; composed/historical opponents | [Multi-phase PPO](docs/multiphase_selfplay.md) |
| Independently composed agents and phase ablations | [Phase-composable evaluation](docs/phase_composable_evaluation.md) |
| Duplicate deals, metrics, bootstrap uncertainty | [Arena methodology](docs/arena.md) |
| Frozen heuristic and random/conservative variants | [Heuristic V1](docs/simpleheuristic.md), [policy ablations](docs/policy_ablation.md) |
| Bid-versus-pass replay experiments | [Counterfactual bidding](docs/counterfactual_bidding.md) |
| Exact open-hand contract/point endgames and complete-deal bid benchmarks | [Endgame oracle](docs/endgame_oracle.md) |
=======
| Hidden-deal beliefs, world counting/sampling, rollouts, PIMC action values | [Belief modelling](docs/belief.md) |
| Observations, public history and training-data validation | [Encoding](docs/encoding.md) |
| Permanent global actions and reserved contracts | [Action space](docs/action-space.md) |
| Process workers and evaluation profiling | [Parallel execution](docs/parallel_execution.md), [evaluation performance](docs/evaluation_performance.md) |
| Optional OpenSpiel adapter | [OpenSpiel](docs/openspiel.md) |
| Research status and next experiments | [Roadmap](docs/roadmap.md) |

## Project map

| Directory / module | Responsibility |
| --- | --- |
| `texas42/models.py`, `rules.py`, `state.py` | Value types, pure rules and authoritative hand transitions |
| `texas42/episodes.py`, `selfplay.py` | Hand/match episodes, immutable observations, reset/step API |
| `texas42/engine.py`, `players.py`, `cli.py` | Console runner and object-based player decisions |
| `texas42/agents/` | Common protocol, built-in agents and Player adapter |
| `texas42/arena.py`, `parallel.py` | Paired evaluation, metrics and ordered process workers |
| `texas42/oracle.py` | Shared perfect-information contract/point search, arbitrary-position bounds, complete-deal analysis and exhaustive reference |
| `texas42/privileged.py` | Training-only centralized-critic features (all remaining hands); never an actor input |
| `texas42/belief.py` | Hidden-deal constraints, exact counting/sampling, rollouts and expected action values |
| `texas42/training.py`, `encoding.py`, `actions.py` | Versioned ID-based views, 387 features and legal global actions |
| `texas42/openspiel.py` | Optional adapter depending on the native engine |
| `research/ppo/` | Optional training (local or centralized critic), oracle label datasets, checkpoints and frozen policy composition |
| `examples/` | Runnable game, training and evaluation commands |
| `tests/`, `docs/` | Deterministic unittest coverage and technical documentation |
| `runs/`, `logs/` | Ignored generated checkpoints, results and readable exports |

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

## Common agents

`Agent` is the algorithm-facing protocol: `reset(seat)` and
`act(Observation) -> int`. Return an exact Python integer from
`obs.legal_action_ids`; masks include all 58 global actions. Reserved IDs 55–57
remain illegal. `Player` retains object-based `choose_bid`, `choose_trump` and
`choose_domino` methods. `AgentPlayer` adapts ordinary agents to that console API,
using the same legality and observation construction as native self-play.

Use four independent agent instances even when weights are shared. RandomAgent
samples every legal action, including bids; the console RandomPlayer normally
passes. RandomAgent resets its seat without rewinding its seeded RNG. Policies
receive only their own immutable observation, never mutable engine state or
opponents' hands. Custom agents need no base class: implement the two methods,
inspect `obs.view`, and optionally call `encoding.encode(obs)` with its legal mask.

```python
from texas42 import SelfPlayEnv
from texas42.agents import RandomAgent

agents = [RandomAgent(seed=seat) for seat in range(4)]
env = SelfPlayEnv(seed=42)
obs = env.reset()
for seat, agent in enumerate(agents):
    agent.reset(seat)
while obs is not None:
    obs = env.step(agents[obs.seat].act(obs)).observation
```

For `engine.Game`, wrap each agent with `AgentPlayer(name, agent)` and supply four
players in seat order. `Game.play_match()` resets agents once per match; standalone
`play_hand()` resets once per hand. These hooks do not reset the Game's accumulated
marks or deal RNG. Direct adapter callers must reset before requesting decisions.
Native runners own their agents and reset them once after `env.reset()`, preserving
memory between hands of a match. Construct fresh agents to restart RNG streams.

`PrivilegedAgent.act_privileged(obs, state)` remains a separate protocol for future
privileged runners. The standalone [endgame oracle](docs/endgame_oracle.md) accepts
a full `HandState` explicitly; ordinary adapters never supply `PrivilegedState`
or hidden-hand labels.

Agents may also implement the optional pure query
`action_probabilities(obs) -> {action_id: probability}`, the distribution `act`
samples from (the `ActionProbabilities` protocol). It must not consume RNG or
change memory. All built-in agents implement it. [Belief modelling](docs/belief.md)
uses it to weight hypothetical deals and to compute rollout probabilities.

## Self-play and data contracts

`SelfPlayEnv.reset(seed=...)` restarts the deal stream; `reset()` continues it.
`reset(shaker=...)` can choose the starting shaker. Observations contain the acting
seat, phase, immutable ID-based view, legal IDs and mask. `step(action_id)` returns
the next observation, seat-ordered rewards, an optional completed `hand_result`,
termination flag and match winner. Illegal actions fail without changing state.

Hand episodes finish after 33 decisions and pay terminal signed marks to each
seat's partnership. Match episodes retain marks and rotate the shaker; rewards
are zero until the match ends, then +1/-1. A hand-ending match transition can
contain both a result and the next hand's observation. Keep learning trajectories
separate by seat, and apply terminal rewards to seats that acted earlier too.
After a collector sink failure, abort and reset; do not retry the action.

Schema-5 JSONL preserves actor-visible states, global actions and public history.
Privileged initial deals are opt-in collector labels stored outside Observation.
See [encoding and collection](docs/encoding.md) for deserialization, validation,
feature layouts and privacy; [action compatibility](docs/action-space.md) owns
ID mappings and old-log migration.

## Development

Read [AGENTS.md](AGENTS.md) before contributing. Tests use standard-library
`unittest`; optional RL/OpenSpiel tests skip if their dependencies are absent.
GitHub Actions checks core and optional integrations. Run `git diff --check`
before review, and preserve seeded behavior when refactoring research code.

## License

[MIT](LICENSE).
