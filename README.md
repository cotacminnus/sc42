# SC42 / TX42

SC42 is a Texas 42 domino game for the command line and a reproducible research
testbed for imperfect-information partnership learning. It provides:

- **A rules engine** (package `texas42`, console command `tx42`) with immutable
  per-seat observations, 58 permanent global action IDs and legal-action masks.
  It has no third-party dependencies.
- **Built-in agents**: heuristic, conservative and random variants for each phase,
  and a belief-based Monte Carlo (PIMC) search agent.
- **Duplicate-deal evaluation**: seat-swapped paired deals, signed marks per hand
  and bootstrap confidence intervals.
- **Optional PPO training** (`research/ppo/`, PyTorch). Bidding, trump and play can
  each be trained, frozen or heuristic. Training can start from scratch, use
  self-play and historical opponents, and resume exactly.
- **JSON experiment pipelines**: a multi-stage experiment declared in one file and
  run with one command, with checkpoints passed between stages, recorded provenance
  and resumable progress.

## Contents

- [Installation](#installation)
- [Quickstart: play a game](#quickstart-play-a-game)
- [Quickstart: run an experiment pipeline](#quickstart-run-an-experiment-pipeline)
- [Documentation map](#documentation-map)
- [Project map](#project-map)
- [Implemented rules](#implemented-rules)
- [Agent API](#agent-api)
- [Self-play and data contracts](#self-play-and-data-contracts)
- [Testing](#testing)
- [Development](#development)

## Installation

Python 3.11 or newer is required. Run all commands from the repository root.

```bash
python -m pip install -e .               # engine, console, heuristic agents, arena
python -m pip install -e '.[rl]'         # + PyTorch/NumPy: training, neural evaluation, pipelines
python -m pip install -e '.[openspiel]'  # + optional OpenSpiel adapter
```

| Install | Needed for |
| --- | --- |
| `.` | Console game, rules engine, heuristic/random/PIMC agents, pure-heuristic evaluation |
| `.[rl]` | Every training command, neural evaluation and any pipeline with training or neural steps |
| `.[openspiel]` | `texas42.openspiel` only |

On Windows PowerShell, use double quotes: `python -m pip install -e ".[rl]"`.

## Quickstart: play a game

```bash
python -m texas42                                       # you in seat 1, three placeholder bots
python -m texas42 --players human HHH CHH HHR --seed 42 # choose a bot for each other seat
python -m texas42 --all-ai --players HHH CHH HHH CHH --seed 42 --target-marks 1
```

`tx42` is the same program as `python -m texas42`. On the console, players are
numbered 1–4: seats 1 and 3 partner against seats 2 and 4. (The Python API numbers
seats 0–3, with team 0 in seats 0/2 and team 1 in seats 1/3.) The three letters of a
bot name give its bidding, trump and play strategy, for example `HHH` (all
heuristic) or `CHH` (conservative bidding); see
[built-in variants](docs/command_reference.md#built-in-policy-variants).
`placeholder`, the default bot, passes every optional bid and plays randomly.

Console input:

- Bid `30`–`41`, `42` for one mark, or `1m`–`5m`; `p` passes when legal.
- Trump: `0`–`6`, `doubles`, or `n`/`follow me` for no trump.
- Play the displayed legal domino number. These are positions in your hand, not
  global IDs.
- Ctrl+C quits. `--debug` reveals all hands; `--seed` reproduces shuffles and bot
  choices.
- `--self-play-log [PATH]` (requires `--all-ai`) writes schema-5 JSONL and
  overwrites the file, by default `logs/selfplay.jsonl`.

Neural players and every console flag are in the
[console reference](docs/command_reference.md#console-game).

## Quickstart: run an experiment pipeline

A pipeline is a JSON file listing an ordered set of ordinary SC42 commands, for
example *train play → train bidding on top of it → evaluate*. `examples.run_pipeline`
runs them one after another and fills in each stage's checkpoint path for later
stages. It records commands, git commit and file hashes in a manifest, and with
`--resume` it continues an interrupted run without repeating finished work. The runner
adds no training or evaluation logic of its own: each step is exactly the command you
could type by hand.

### 1. Install the `rl` extra

```bash
python -m pip install -e '.[rl]'
```

Without it, the runner reports the missing dependency before running anything.

### 2. Dry-run the bundled quickstart pipeline

[`configs/pipelines/quickstart.json`](configs/pipelines/quickstart.json) is a smoke-scale
example with four steps:

| Step | Command | What it does |
| --- | --- | --- |
| `play` | `train_selfplay_ppo` | Trains trick play from random weights (`from-scratch`) against `HHH`, with heuristic bidding and trump. |
| `bidding` | `train_selfplay_ppo` | Starts a new stage from `${play.best}`: fresh bidding heads are trained while play stays frozen, against a 50/50 mix of `HHH` and the `play` stage itself. |
| `fixed-opponents` | `evaluate_selfplay_ppo` | Evaluates `${bidding.best}` against `HHH` and `CHH` on held-out deals. |
| `bidding-vs-play` | `evaluate_agent_vs_agent` | Plays the two stages head to head on duplicate deals. |

```bash
python -m examples.run_pipeline --config configs/pipelines/quickstart.json --dry-run
```

The dry run validates the file, the references and every step's arguments with that
command's own parser. It then prints each exact command line and its output
locations. It creates no files.

### 3. Run it

```bash
python -m examples.run_pipeline --config configs/pipelines/quickstart.json
```

Each command prints its normal output between progress lines:

```text
Pipeline: quickstart
[1/4] play            TRAINING
...
[4/4] bidding-vs-play COMPLETED

Pipeline completed successfully.
```

The whole pipeline took about 40 seconds on a 16-core Linux machine (CPU only).
Its budgets (2,048 training hands per stage and 128 evaluation deals) are only for
checking that everything works; the results support no conclusions.

### 4. Read the results

```text
runs/pipelines/quickstart/
├── pipeline_state.json      manifest: config + SHA256, git commit, exact commands, output hashes
├── summary.md               one row per evaluate_agent_vs_agent step: marks/hand and win rate, 95% CIs
├── play/                    checkpoints/best.pt, latest.pt · config.json · progress.jsonl · logs/
├── bidding/                 same layout as play/
├── fixed-opponents/         result.json · logs/result.log, result.csv, result-metrics.csv
└── bidding-vs-play/         same layout as fixed-opponents/
```

Start with `summary.md`, which is also printed at the end of the run, then the
readable `logs/result.log` of each evaluation step. Signed marks per hand is the
primary measure; [arena methodology](docs/arena.md) explains the metrics and
intervals.

### 5. Resume, or rerun

```bash
python -m examples.run_pipeline --config configs/pipelines/quickstart.json --resume
```

`--resume` skips completed steps after verifying their output hashes, and continues
an interrupted training step from its `latest.pt`. It requires the identical
configuration file. A plain rerun into an existing `output_root` is refused
(`pipeline_state.json exists; use --resume, or choose a new output_root`), so results
are never overwritten. To repeat the run, delete `runs/pipelines/quickstart/` or
change `output_root`.

### 6. Write your own

```bash
cp configs/pipelines/template.json configs/pipelines/my-experiment.json
```

To turn a command line into a step, drop `python -m examples.`, put the command
name in `command`, and move each flag into `args` without its `--`:

```bash
python -m examples.train_selfplay_ppo --from-scratch --opponent-pool HHH:1 --total-hands 65536 --seed 42
```

```json
{
  "schema_version": 1,
  "name": "my-experiment",
  "output_root": "runs/pipelines/my-experiment",
  "steps": [
    {
      "id": "stage1",
      "command": "train_selfplay_ppo",
      "args": {"from-scratch": true, "opponent-pool": "HHH:1", "total-hands": 65536, "seed": 42}
    },
    {
      "id": "eval",
      "command": "evaluate_selfplay_ppo",
      "args": {"checkpoint": "${stage1.best}", "opponents": "HHH,CHH", "deals": 2048, "seed": 12345}
    }
  ]
}
```

| Rule | Detail |
| --- | --- |
| Allowed `command` values | `train_selfplay_ppo`, `train_ppo`, `evaluate_agent_vs_agent`, `evaluate_selfplay_ppo`, `evaluate_ppo`, `oracle_labels`. No shell and no other modules. |
| `args` values | A string or number is passed exactly as written; `true` gives a bare flag; `false` omits the flag; a list repeats the flag. `null`, objects and empty lists are rejected. |
| References | `${STEP.best}` / `${STEP.latest}` are a training step's checkpoints; `${STEP.output}` is an evaluation report or label dataset. They may only name earlier steps and may appear inside a value, for example `"opponent-checkpoint": ["stage1=${stage1.best}"]`. |
| Outputs | Leave out `output-dir`, `output` and `log-dir`; each step gets `OUTPUT_ROOT/<id>/`. |
| `output_root` | Use a new one whenever the configuration changes. |

The full format, manifest, resume rules, troubleshooting table and more examples
(progressive curriculum, self-play, critic comparison, evaluation-only round robins)
are in [experiment pipelines](docs/experiment_pipelines.md). The flags each step can
take are in the [command reference](docs/command_reference.md).

## Documentation map

| Topic | Authoritative guide |
| --- | --- |
| Every command, argument, output path and recipe | [Command reference](docs/command_reference.md) |
| Where to start a PPO study | [Training workflow finder](docs/command_reference.md#workflow-finder) |
| JSON pipelines: format, references, outputs, manifest, resume | [Experiment pipelines](docs/experiment_pipelines.md) |
| Training from random weights and self-play | [From-scratch training](docs/command_reference.md#recipe-14-from-scratch-training-and-self-play) |
| Original play-only PPO and version-1 reproduction | [Legacy PPO](docs/ppo_training.md) |
| Independently trained/frozen bidding, trump and play; historical opponents; paired partners | [Multi-phase PPO](docs/multiphase_selfplay.md) |
| Composing agents from phases and phase ablations | [Phase-composable evaluation](docs/phase_composable_evaluation.md) |
| Comparing models against baselines; round robins with a summary table | [Model comparisons](docs/command_reference.md#model-comparisons-and-round-robins) |
| Duplicate deals, metrics, bootstrap uncertainty | [Arena methodology](docs/arena.md) |
| Heuristic and random/conservative variants | [Heuristic V1](docs/simpleheuristic.md), [policy ablations](docs/policy_ablation.md) |
| Bid-versus-pass replay experiments | [Counterfactual bidding](docs/counterfactual_bidding.md) |
| Exact open-hand endgames and complete-deal bid benchmarks | [Endgame oracle](docs/endgame_oracle.md) |
| Hidden-deal beliefs, PIMC and model-guided PIMC | [Belief modelling](docs/belief.md) |
| Monte Carlo search inside training (`--search-fraction`) | [Search in training](docs/search_training.md) |
| Observations, public history, training-data validation | [Encoding](docs/encoding.md) |
| Permanent global action IDs and reserved contracts | [Action space](docs/action-space.md) |
| Worker processes and evaluation profiling | [Parallel execution](docs/parallel_execution.md), [evaluation performance](docs/evaluation_performance.md) |
| OpenSpiel adapter | [OpenSpiel](docs/openspiel.md) |
| Research status and next experiments | [Roadmap](docs/roadmap.md) |

## Project map

| Directory / module | Responsibility |
| --- | --- |
| `texas42/models.py`, `rules.py`, `state.py` | Value types, pure rules and authoritative hand transitions |
| `texas42/episodes.py`, `selfplay.py` | Hand/match episodes, immutable observations, reset/step API |
| `texas42/engine.py`, `players.py`, `cli.py` | Console runner and object-based player decisions |
| `texas42/agents/` | Common protocol, built-in agents and the `Player` adapter |
| `texas42/arena.py`, `parallel.py` | Paired evaluation, metrics and ordered process workers |
| `texas42/oracle.py` | Perfect-information contract/point search, position bounds, complete-deal analysis |
| `texas42/privileged.py` | Training-only centralized-critic features; never an actor input |
| `texas42/belief.py` | Hidden-deal constraints, exact counting/sampling, rollouts, expected action values |
| `texas42/training.py`, `encoding.py`, `actions.py` | Versioned ID-based views, 387 features, legal global actions |
| `texas42/openspiel.py` | Optional OpenSpiel adapter over the native engine |
| `research/ppo/` | Optional PPO training (local, centralized or no critic), oracle label datasets, checkpoints, frozen-policy composition, model-guided search |
| `examples/` | Runnable training, evaluation and benchmark commands, including `run_pipeline.py` |
| `configs/pipelines/` | Example pipeline JSON files (other files in this directory are git-ignored) |
| `tests/`, `docs/` | Deterministic `unittest` coverage and technical documentation |
| `runs/`, `logs/` | Generated checkpoints, results and logs (git-ignored) |

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

Nello, Sevens, Splash and Plunge are not implemented; other Singing Cadets house
rules may need to be added. The `placeholder` console bot passes every optional bid;
as forced bidder the engine bids 30 for it, and it then calls a random trump. Logs of
four placeholders therefore contain no real bidding: use `SelfPlayEnv` or variant bots
for data covering the full auction.

## Agent API

`Agent` is the algorithm-facing protocol: `reset(seat)` and
`act(Observation) -> int`. Return an exact Python integer from
`obs.legal_action_ids`; masks cover all 58 global actions, and reserved IDs 55–57
are always illegal. Custom agents need no base class: implement the two methods,
inspect `obs.view`, and optionally call `encoding.encode(obs)` for the 387-feature
vector and legal mask.

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

- Use four independent agent instances, even when weights are shared. Policies
  receive only their own immutable observation, never mutable engine state or
  opponents' hands.
- `RandomAgent` samples every legal action, including bids; `reset` sets its seat
  without rewinding its seeded RNG. Construct fresh agents to restart RNG streams.
- `Player` keeps the object-based `choose_bid`, `choose_trump` and `choose_domino`
  methods. `AgentPlayer(name, agent)` adapts any agent to it, using the same legality
  and observation construction as self-play. Give `engine.Game` four players in seat
  order. `Game.play_match()` resets agents once per match, and a standalone
  `play_hand()` resets them once per hand. Neither resets the game's marks or deal RNG.
- Agents may implement the pure query `action_probabilities(obs) -> {action_id:
  probability}` (the `ActionProbabilities` protocol), which must not consume RNG or
  change memory. All built-in agents except PIMC implement it;
  [belief modelling](docs/belief.md) uses it to weight hypothetical deals.
- `texas42.agents.pimc.PIMCAgent` is a search baseline built on those beliefs. Use it
  through the `pimc` trump/play strategy of `examples.evaluate_agent_vs_agent`;
  `--agent-X-pimc-model PATH` lets a trained model play the rollouts (model-guided
  PIMC, the strongest agent measured; see [results](docs/belief.md#model-guided-results)).
- `PrivilegedAgent.act_privileged(obs, state)` is a separate protocol for future
  privileged runners. The [endgame oracle](docs/endgame_oracle.md) takes a full
  `HandState` explicitly; ordinary adapters never receive hidden hands.

## Self-play and data contracts

`SelfPlayEnv.reset(seed=...)` restarts the deal stream; `reset()` continues it, and
`reset(shaker=...)` chooses the starting shaker. Observations contain the acting
seat, phase, immutable ID-based view, legal IDs and mask. `step(action_id)` returns
the next observation, seat-ordered rewards, an optional completed `hand_result`,
termination flag and match winner. Illegal actions fail without changing state.

Hand episodes finish after 33 decisions and pay terminal signed marks to each
seat's partnership. Match episodes keep marks and rotate the shaker; rewards are
zero until the match ends, then +1/−1. A hand-ending match transition can contain
both a result and the next hand's observation. Keep learning trajectories separate
by seat, and apply terminal rewards to seats that acted earlier too. After a
collector sink failure, abort and reset; do not retry the action.

Schema-5 JSONL keeps actor-visible states, global actions and public history.
Privileged initial deals are opt-in collector labels stored outside `Observation`.
[Encoding](docs/encoding.md) covers deserialization, validation, feature layouts and
privacy; [action space](docs/action-space.md) owns ID mappings and old-log migration.

## Testing

```bash
python -m unittest discover -s tests -v     # full suite (~390 tests, ~8 min with .[rl])
python -m unittest tests.test_experiment_pipeline -v   # one module
git diff --check                            # whitespace check before review
```

Tests that need PyTorch or OpenSpiel skip themselves when the extra is missing, so
check the skip count. The [testing reference](docs/command_reference.md#testing) is
canonical: it has a 12-second core check, a map of which module covers which area,
the CI jobs and the known Windows-only failures.

## Development

Read [AGENTS.md](AGENTS.md) before contributing. GitHub Actions
(`.github/workflows/tests.yml`) runs the suite on every push and pull request for
the core, OpenSpiel and PyTorch configurations. Preserve seeded behavior when
refactoring research code.

## License

[MIT](LICENSE).
