# SC42 command reference

Canonical, copy-ready commands and argument tables for every CLI in this checkout.
Method, rationale and compatibility details live in the linked topic guides; this
page owns *how to run things*.

> **Conventions.** Run every command from the repository root. Examples use bash/WSL2
> line continuation (`\`). Replace checkpoint and output paths with your own. Every
> CLI accepts `-h`/`--help`, e.g. `python -m examples.train_ppo --help`.

## Contents

- [Setup](#setup)
- [Command index](#command-index)
- [Shared conventions](#shared-conventions) — duplicate deals, common flags, policy variants, outputs
- [Console game](#console-game) — `texas42`, self-play examples
- [Fixed-policy experiments](#fixed-policy-experiments) — policy ablations, counterfactual bidding
- [Endgame oracle](#endgame-oracle) — open-hand endgames, complete-deal bidding, [training labels](#oracle-training-labels)
- [PPO training overview](#ppo-training-overview) — choosing a trainer, [workflow finder](#workflow-finder)
- [Legacy trainer: train_ppo](#legacy-trainer-train_ppo) — Recipe 1, resume, mixed opponents
- [Multi-phase trainer: train_selfplay_ppo](#multi-phase-trainer-train_selfplay_ppo) — Recipes 2–11, 13 and 14 (from scratch), arguments, artifacts, troubleshooting
- [Evaluation](#evaluation) — agent-vs-agent composition (Recipe 12), compatibility commands
- [Performance benchmarks](#performance-benchmarks)
- [Reproducibility checklist](#reproducibility-checklist)

## Setup

```bash
python -m pip install -e .               # core package (no ML required)
python -m pip install -e '.[rl]'         # PPO training and neural evaluation
python -m pip install -e '.[openspiel]'  # optional OpenSpiel adapter
python -m unittest discover -s tests -v  # verify the checkout
```

## Command index

| Goal | Entry point | Extra | Section |
| --- | --- | --- | --- |
| Human/AI console game, JSONL self-play logs | `python -m texas42` | — (`rl` for neural seats) | [Console game](#console-game) |
| Baseline comparison (HHH, CHH, HHR, …) | `python -m examples.policy_ablation_benchmark` | — | [Fixed-policy experiments](#fixed-policy-experiments) |
| Original heuristic-vs-random baseline | `python -m examples.heuristic_benchmark` | — | [Fixed-policy experiments](#fixed-policy-experiments) |
| Bid-versus-pass intervention | `python -m examples.counterfactual_bidding` | — | [Counterfactual bidding](#counterfactual-bidding) |
| Open-hand endgames and complete-deal bids | `python -m examples.endgame_oracle` | — | [Endgame oracle](#endgame-oracle) |
| Sample and label play positions for critic supervision | `python -m examples.oracle_labels` | — | [Oracle training labels](#oracle-training-labels) |
| Train legacy play-only PPO | `python -m examples.train_ppo` | `rl` | [Legacy trainer](#legacy-trainer-train_ppo) |
| Train independent-phase PPO / historical self-play | `python -m examples.train_selfplay_ppo` | `rl` | [Multi-phase trainer](#multi-phase-trainer-train_selfplay_ppo) |
| Phase-composed agent-vs-agent evaluation (canonical) | `python -m examples.evaluate_agent_vs_agent` | `rl` only for neural phases | [Agent-vs-agent](#agent-vs-agent-composition) |
| Legacy checkpoint evaluation | `python -m examples.evaluate_ppo` | `rl` | [Compatibility evaluation](#compatibility-evaluation) |
| Multi-opponent phase evaluation (anchors, archives) | `python -m examples.evaluate_selfplay_ppo` | `rl` | [Compatibility evaluation](#compatibility-evaluation) |
| Agent API demo (no arguments) | `python -m examples.agent_selfplay` | — | [Self-play examples](#self-play-examples) |
| OpenSpiel rollouts | `python -m examples.openspiel_selfplay` | `openspiel` | [Self-play examples](#self-play-examples) |
| Worker scaling, bootstrap, validation and arena profiles | `python -m examples.parallel_benchmark` and others | varies | [Performance benchmarks](#performance-benchmarks) |

## Shared conventions

### Duplicate deals and units

A duplicate **group** (`--deals N`) contains two games on the same absolute-seat deal
and shaker with ABAB/BABA partnership seating swapped. Thus `--deals 4096` normally
yields 8192 hand games. In `--episode match` mode the units are paired match
schedules, not individual hands. Bootstrap intervals resample **whole groups**. See
[arena methodology](arena.md) for metric and statistical definitions.

### Common arena flags

These flags recur across the arena, evaluation and fixed-policy commands. Per-command
defaults are listed in each command's table.

| Argument | Meaning |
| --- | --- |
| `--deals` | Number of paired duplicate groups, not number of games. |
| `--seed` | Deterministic deal/episode/agent seed. Choose fresh held-out values for final tests. |
| `--threads` | Maximum simulation processes: `1` sequential, `0` auto (capped at 8 and by CPU affinity, physical cores, quotas and task count), `N` up to N. |
| `--bootstrap-seed` | Independent bootstrap RNG seed. |
| `--bootstrap-samples` | Number of complete-group bootstrap resamples. |
| `--episode` | `hand` or `match`. |
| `--target-marks` | Marks needed to win a match; does not turn hand mode into a match. |
| `--output` | Optional detailed JSON report or benchmark profile path. |
| `--log-dir` | Readable `.log`/`.csv` sidecars; default `logs/experiments/<command>-<run-id>/`. Does not replace `--output`. |

### Built-in policy variants

Three letters name the bidding, trump and trick-play strategies. Every command that
accepts a variant name uses this table.

| Variant | Bidding | Trump | Trick play |
| --- | --- | --- | --- |
| HHH | Heuristic | Heuristic | Heuristic |
| HHR | Heuristic | Heuristic | Random |
| HRR | Heuristic | Random | Random |
| CHH | Conservative | Heuristic | Heuristic |
| CRR | Conservative | Random | Random |
| RRR | Random | Random | Random |

Conservative bidding passes when legal, otherwise makes the minimum bid. In training
and multi-phase evaluation, `anchor` means heuristic bid/trump plus the original
frozen play-only PPO network. See [Heuristic V1](simpleheuristic.md) and
[policy ablations](policy_ablation.md).

### Outputs and logs

| Location | Contents |
| --- | --- |
| `--output <path>.json` | Optional detailed JSON report; not a substitute for the log directory. |
| `logs/experiments/<command>-<run-id>/` | Auto-created readable `.log` and `.csv` files for experiment commands. |
| `runs/<run>/…` | Training configuration, progress, checkpoints and snapshots; see [legacy](#train_ppo-outputs) and [multi-phase](#training-artifacts-and-metrics) artifacts. |

## Console game

```bash
python -m texas42 --seed 42 --target-marks 7
# Human in seat 1, with independently selected policy bots
python -m texas42 --players human HHH CHH HHR --seed 42
# Opposing HHH/CHH partnerships, with a replayable match log
python -m texas42 --all-ai --players HHH CHH HHH CHH \
  --seed 42 --target-marks 1 --self-play-log logs/hhh-vs-chh.jsonl
# Default placeholder bots, logged
python -m texas42 --all-ai --seed 42 --target-marks 1 \
  --self-play-log logs/selfplay.jsonl
```

Console `--self-play-log` requires `--all-ai` and writes schema-5 JSONL; it is not
neural-network training. The console runs ordinary matches; use
[agent-vs-agent evaluation](#agent-vs-agent-composition) for duplicate deals,
parameter-equality checks and confidence intervals.

### Player selection

`--players P1 P2 P3 P4` selects each seat independently. Choices are `human`,
`placeholder` (the original always-pass random bot), `neural`, and the six
[built-in variants](#built-in-policy-variants). Seats 1/3 partner against seats 2/4.
Variant bots reuse the existing `HybridAgent` through `AgentPlayer`; each seat has its
own agent and random streams, reset once per match. Non-neural players require no ML
dependencies. `RRR` bids uniformly among legal actions, unlike `placeholder`. Multiple
`human` seats share one terminal, so avoid viewing another player's private prompts.

Without `--players`, the original defaults remain: human in seat 1 and three
placeholder bots, or four placeholders with `--all-ai`. Explicit selections take
precedence; `--all-ai` rejects any human selection. Four bot selections also run
without `--all-ai`, but logging still requires that flag.

### Neural console players

Install the `rl` extra. `neural` defaults to greedy neural bidding and play with
heuristic trump; the checkpoint must supply trained neural bid and play networks. For
an original play-only checkpoint, explicitly select `--player-N-bid heuristic`;
unavailable neural phases are never replaced silently.

```bash
# Play against neural bots; your partner in seat 3 remains HHH
python -m texas42 --players human neural HHH neural --seed 42 \
  --player-2-checkpoint runs/ppo-play-v1/checkpoints/best.pt --player-2-bid heuristic \
  --player-4-checkpoint runs/ppo-play-v1/checkpoints/best.pt --player-4-bid heuristic

# Hold joint PPO playing fixed on all seats; change only partnership bidding
python -m texas42 --all-ai --players neural neural neural neural --seed 42 \
  --player-1-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --player-2-checkpoint runs/ppo-joint-v1/checkpoints/best.pt --player-2-bid heuristic \
  --player-3-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --player-4-checkpoint runs/ppo-joint-v1/checkpoints/best.pt --player-4-bid heuristic \
  --self-play-log logs/console-bidding-ablation.jsonl

# Independent checkpoints and conservative bidding on one seat
python -m texas42 --players human neural HHH CHH --seed 42 \
  --player-2-bid-checkpoint runs/ppo-bidding-v1/checkpoints/best.pt \
  --player-2-play-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --player-4-play neural \
  --player-4-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt
```

Each neural phase resolves `--player-N-PHASE-checkpoint` before
`--player-N-checkpoint`; a checkpoint override requires a neural strategy.
Version-1 checkpoints support play only; version-2 checkpoints must record the
requested phase as trained and neural. Console output identifies the selected phases,
paths and SHA256 values. Frozen CPU weights are cached and shared across seats;
private agent state and phase RNG streams remain independent. Gameplay uses the same
`CompositeAgentFactory` as phase-composable evaluation, with one PyTorch inference
thread and no parameter updates.

### Console arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--players P1 P2 P3 P4` | Human + 3 placeholders | Seat types: `human`, `placeholder`, `neural`, or a variant. |
| `--all-ai` | Off | Run without humans; rejects `human` seats. |
| `--self-play-log [PATH]` | Off; `logs/selfplay.jsonl` if no path | Write schema-5 JSONL (overwrites). Requires `--all-ai`. |
| `--seed` | Random | Reproducible shuffle and bot seed. |
| `--target-marks` | `7` | Marks needed to win. |
| `--debug` | Off | Reveal every hand at the start of each hand. |
| `--ruleset-id` | `cadets_base_v1` | Versioned house rules (currently the only value). |
| `--player-N-checkpoint` | Unset | Default checkpoint for seat N's neural phases (N = 1–4). |
| `--player-N-bid` | From preset | `neural`, `heuristic`, `random`, `conservative`. |
| `--player-N-trump`, `--player-N-play` | From preset | `neural`, `heuristic`, `random`. |
| `--player-N-{bid,trump,play}-checkpoint` | Unset | Checkpoint override for that neural phase only. |

Phase options are rejected on `human` and `placeholder` seats. Console controls are
described in the [README](../README.md#console-controls).

### Self-play examples

```bash
python -m examples.agent_selfplay                              # no arguments
python -m examples.openspiel_selfplay --episode hand --seed 42  # needs pyspiel
```

`openspiel_selfplay` accepts `--episode` (`hand`/`match`, default `hand`),
`--target-marks` (default `7`) and `--seed` (default `42`). See [OpenSpiel](openspiel.md).

## Fixed-policy experiments

No optional dependencies are required.

### Policy ablation benchmarks

```bash
# Isolate bidding changes (HHH versus conservative bidding)
python -m examples.policy_ablation_benchmark \
  --policy-a HHH --policy-b CHH --deals 4096 \
  --seed 42 --threads 4 --bootstrap-samples 2000 \
  --output runs/hhh-vs-chh.json

# Isolate heuristic vs random playing given fixed auction/trump
python -m examples.policy_ablation_benchmark \
  --policy-a HHH --policy-b HHR --deals 2048 --threads 4

# Original heuristic versus random benchmark
python -m examples.heuristic_benchmark \
  --opponent random --deals 1024 --threads 4
```

### Counterfactual bidding

```bash
python -m examples.counterfactual_bidding \
  --policy-a HHH --policy-b CHH --target-policy HHH \
  --deals 4096 --seed 42 --threads 4 \
  --bootstrap-samples 2000 \
  --output runs/counterfactual-bidding.json
```

The experiment identifies the first voluntary bid chosen by the target policy in each
original hand and replays BID versus PASS from the same deal. It supports hand
episodes only. The result is conditional on the actions the policy actually bids; it
does not evaluate every legal bid or pass opportunity. See
[counterfactual bidding](counterfactual_bidding.md).

### Fixed-policy arguments

All three commands also take the [common arena flags](#common-arena-flags) with
defaults `--seed 42`, `--episode hand`, `--target-marks 7`, `--bootstrap-seed 0`,
`--bootstrap-samples 2000`, `--threads 1`.

| Command | Specific arguments (default) | `--deals` default |
| --- | --- | --- |
| `policy_ablation_benchmark` | `--policy-a` (HHH), `--policy-b` (CRR); any variant | 1000 |
| `heuristic_benchmark` | `--opponent` `random` or `heuristic` (random) | 1000 |
| `counterfactual_bidding` | `--policy-a` (HHH), `--policy-b` (CHH), `--target-policy` (HHH); target must be one of the two policies; hand episodes only | 128 |

## Endgame oracle

Exact perfect-information search; no optional dependencies. See the
[oracle guide](endgame_oracle.md) for value, proof and search semantics.

### Endgame mode

```bash
python -m examples.endgame_oracle --seed 42 --remaining 3 --max-nodes 100000
python -m examples.endgame_oracle --seed 7 --remaining 2 --partial-plays 3
# Prove the position and one action; other actions may remain unresolved
python -m examples.endgame_oracle --seed 1 --remaining 2 --partial-plays 1 \
  --output position_value --max-nodes 4
# Final-point endgames (the binary contract cutoff does not apply)
python -m examples.endgame_oracle --objective final_points --seed 1 --remaining 2
# Bounded smoke benchmark, consecutive seeds starting at zero
python -m examples.endgame_oracle --seed 0 --remaining 3 --samples 100 \
  --max-nodes 100000 --time-seconds 2
# Small-endgame point benchmark; per-position limits
python -m examples.endgame_oracle --objective final_points --seed 0 \
  --remaining 3 --samples 100 --max-nodes 100000 --time-seconds 2
# Optional independent root-action workers (startup often dominates tiny endgames)
python -m examples.endgame_oracle --objective final_points --seed 42 \
  --remaining 6 --workers 4 --max-nodes 1000000 --time-seconds 20
```

The default HHR policies generate a reachable ordinary contract with heuristic
bidding/trump and seeded random play. Omitting `--seed` chooses a fresh random seed;
every run prints its resolved seed for reproduction, and multiple samples use
consecutive seeds starting there. Node and time limits apply **per position**.

The command prints the contract, scores, trump, actor/team, four hands, partial
trick, legal global IDs/dominoes, exact values or bounds, proven optimal actions,
completion flags, nodes, cache hits and runtime. With multiple samples it prints
completion rates and mean/median/max statistics instead.

### Deal mode

```bash
# Default complete deal: 252 trees, 1048576 nodes/lead, 1800 seconds total
python -m examples.endgame_oracle --mode deal --seed 42 --workers 8
# Smaller bounded smoke run: nodes per opening lead, one global time limit
python -m examples.endgame_oracle --mode deal --seed 42 \
  --max-nodes 10000 --time-seconds 1
# Spawn bounded workers for 252 independent opening-lead trees
python -m examples.endgame_oracle --mode deal --seed 42 --workers 4 \
  --max-nodes 200000 --time-seconds 10
# Maximum-contract proof search with ordering and cancellation
python -m examples.endgame_oracle --mode deal --seed 42 --workers 4 \
  --output maximum_contract --move-ordering combined --max-nodes 2000 --time-seconds 10
# Threshold-search ablation: classify every lead, without bidder cancellation
python -m examples.endgame_oracle --mode deal --seed 42 --workers 4 \
  --output maximum_contract --move-ordering combined --no-proof-cancellation \
  --max-nodes 2000 --time-seconds 10
```

`--mode deal` uses the seeded arena deal directly, with no preliminary policy play.
All hands are visible: this is an open-hand benchmark, not a hidden-information bid
recommendation, and 42 points establishes a one-mark contract without optimizing mark
stake. It prints a trump table for every candidate bidder, point bounds or exact
values, proven lead IDs and complete optimal lead sets, bidder contract bounds,
proven trump/lead choices, completeness flags and aggregate statistics.

**Budgets.** There are **252 logical opening-lead tasks**, run by at most eight
processes. Each continuation gets `--max-nodes` (default **1,048,576**) with a fresh
counter, so aggregate nodes may exceed the per-lead cap (at most 264,241,152 with
defaults). The **entire analysis shares** `--time-seconds` (default **1,800**),
including validation, worker startup, collection and shutdown; the monotonic deadline
never resets per lead. Time limits are cooperative, so a transition or process
shutdown can cause a small overrun. Worker-local caches together stay within
`--max-cache-entries`.

**Progress.** The CLI flushes a live `[completed/252]` line as each task settles,
with bidder, trump, opening domino, status (`exact`, `contract_proven`,
`node_limit`, `time_limit`, `skipped_proven`, `cancelled_proven`,
`cancelled_deadline`), actual nodes, task runtime and global elapsed time. Explicitly
skipped tasks count as settled without implying they were searched. Progress can
arrive out of order; final tables retain canonical seat/trump/action-ID order. The
header reports requested/resolved workers and limits; search statistics, proof
summaries, cutoff counters, settled task counts and requested-output completeness
follow. See [parallel details](endgame_oracle.md#parallel-execution) and
[deal proof semantics](endgame_oracle.md#complete-deal-bidding-benchmark).

**Outputs and ordering.** `--output maximum_contract` uses exact point-threshold
proofs and stops a bidder's unnecessary alternatives once its maximum bid
classification is established. `--output all_action_values` retains exhaustive
alternative analysis subject to budgets and never skips work after a bid proof. These
are different completion obligations: the `Requested output complete` line
distinguishes them from the legacy status requiring exact points and the complete
optimal-pair set. Proven bid classes, exact point values and identified ties are
reported separately. `--move-ordering combined` can help larger positions but remains
opt-in because some workloads slow down. See
[proof and ordering semantics](endgame_oracle.md#maximum-contract-proof-mode) and the
[optimization ablations](endgame_oracle.md#ordering-and-proof-search-ablations).

**Parallel measurements.** For the documented smoke measurements, repeat the seed-42
six-domino recipe with `--remaining 2`, `3`, `4`, `6` and `--workers 1`, `2`, `4`,
`8`; the deal recipe allows 200,000 nodes per opening-lead tree and ten seconds
total. Historical whole-analysis node measurements used a different budget scope. See
[252-task measurements and startup costs](endgame_oracle.md#252-opening-lead-scheduler-benchmark);
small endgames are usually faster with one worker.

### Oracle arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--seed` | Random | Integer deal/policy seed, printed for reproduction; samples use consecutive seeds. |
| `--mode` | `endgame` | `endgame` or `deal`; deal analyzes 252 opening leads across 36 bidder/trump combinations. |
| `--workers` | `1` | `1` sequential; `0` auto; `N` requests up to N spawned CPU processes, limited by available CPUs, tasks and a maximum of 8. |
| `--objective` | `contract_success` | Endgame: `contract_success` or `final_points`. Deal mode always uses final points. |
| `--remaining` | `3` | Endgame only: 1–7 dominoes per player at the selected trick boundary. |
| `--partial-plays` | `0` | Endgame only: advance 0–3 additional legal moves after that boundary. |
| `--policy` | `HHR` | Endgame only: variant used to reach the position (HHH, HHR, HRR, CHH, CRR, RRR). |
| `--output` | `all_action_values` | Endgame: `all_action_values` or `position_value`. Deal: `all_action_values` (exact point analysis) or `maximum_contract` (bid-classification proof). |
| `--move-ordering` | `none` | `none`, `heuristic`, `tt`, or `combined`. Only traversal changes; reported IDs remain canonical. |
| `--no-proof-cancellation` | Off | Deal `maximum_contract` ablation: classify every lead without stopping other tasks after a bidder proof. Ignored otherwise. |
| `--max-nodes` | `1048576` | Nonnegative nodes per opening-lead continuation in deal mode; per whole endgame position otherwise. |
| `--time-seconds` | `1800` | Nonnegative finite seconds for the entire deal analysis, or one endgame position. |
| `--max-cache-entries` | `100000` | Nonnegative aggregate table capacity; parallel workers each get floor(capacity/workers). Zero disables caching. |
| `--samples` | `1` | Positive endgame position count; >1 selects benchmark summary. Deal mode requires 1. |

### Oracle training labels

`examples.oracle_labels` samples play positions from seeded rollouts and labels
them with the [position oracle](endgame_oracle.md#arbitrary-play-positions). The
output JSON is the `--oracle-dataset` for [Recipe 13](#recipe-13-local-versus-centralized-critic-and-oracle-auxiliary)
and also its label cache: rerunning with the same `--output`, budget and split
labels only new positions, and a different budget or split is rejected.

```bash
# Late-hand positions (4-6 completed tricks) give exact labels cheaply
python -m examples.oracle_labels --output runs/oracle-labels/late-v1.json \
  --deals 2048 --seed 42 --policy HHH --min-tricks 4 --max-tricks 6 --per-deal 2 \
  --max-nodes 200000 --time-seconds 10 --split 0.8,0.1,0.1 --split-seed 0
# Tiny smoke dataset
python -m examples.oracle_labels --output runs/oracle-labels/smoke.json \
  --deals 40 --per-deal 2 --min-tricks 3 --max-nodes 50000 --time-seconds 2
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--output` | Required | Dataset JSON path; also the label cache. |
| `--deals` | `64` | Seeded hands played to collect candidate positions. |
| `--seed` | `42` | Deal, agent and position-choice seed. |
| `--policy` | `HHH` | [Variant](#built-in-policy-variants) playing all four seats. |
| `--min-tricks`, `--max-tricks` | `4`, `6` | Game-stage window in completed tricks (0–6). |
| `--per-deal` | `1` | Maximum positions kept per deal. |
| `--max-positions` | Unset | Stop after this many positions. |
| `--max-nodes`, `--time-seconds` | `200000`, `10` | **Per action search** budget; an exhausted search stores valid bounds, not an exact label. |
| `--max-cache-entries` | `100000` | Transposition-table capacity, shared across one position's actions. |
| `--move-ordering` | `none` | `none`, `heuristic`, `tt` or `combined`. |
| `--split`, `--split-seed` | `0.8,0.1,0.1`, `0` | Train/validation/test fractions, assigned per **deal**: positions of one deal never cross splits. |

Each record stores the initial deal, shaker, action prefix and acting seat (enough
to replay the position), certified final-point bounds for the acting seat's team,
exactness, the bidders' contract proof status, nodes and seconds. The header stores
the schema, objective, ruleset, action-space, encoder and privileged-layout
versions, the search budget and the split; incompatible files are rejected. The
printed sampling and labeling times are the labeling cost, separate from training.

## PPO training overview

Install the `rl` extra first. Two trainers exist deliberately:

| Trainer | Use for | Docs |
| --- | --- | --- |
| `examples.train_ppo` | Reproducing the version-1 play-only baseline (heuristic bid/trump), optional fixed/mixed heuristic opponents. | [Legacy PPO](ppo_training.md) |
| `examples.train_selfplay_ppo` | New experiments: independently trained/frozen bid, trump and play phases, neural and historical opponents. | [Multi-phase PPO](multiphase_selfplay.md) |

Both train complete **hand** episodes; there is no training `--episode` or match-PPO
option. The recipes are research configurations, not proven optimal settings.

### Workflow finder

| Recipe | Purpose | Prerequisite |
| --- | --- | --- |
| [1: original play-only](#recipe-1-original-play-only-ppo) | Reproduce `train_ppo`; creates the original play checkpoint | No pretrained checkpoint |
| [2: bidding-only](#recipe-2-bidding-only-frozen-play) | Learn bidding, preserve pretrained play | Compatible version-1 play checkpoint |
| [3: modern playing-only](#recipe-3-playing-only-with-the-modern-trainer) | Learn play with heuristic bid/trump using the version-2 path | Compatible version-1 play checkpoint |
| [4: trump-only](#recipe-4-trump-only) | Learn trump, preserve joint bid/play | Version-2 checkpoint with trained bid/play |
| [5: all phases versus HHH](#recipe-5-all-three-phases-against-hhh) | Learn bid/trump/play against one fixed heuristic | Compatible version-1 play checkpoint |
| [6: historical self-play](#recipe-6-joint-training-with-historical-opponents) | New stage using frozen snapshots and fixed baselines | Recipe 5 checkpoint with all routed phases trained |
| [7: exact resume](#recipe-7-resume-and-multi-opponent-validation) | Restore optimizer, progress, RNGs and historical roster | Existing version-2 checkpoint and its dependencies |
| [8: mixed trainability](#recipe-8-mixed-phase-training) | Freeze any neural continuation or use a heuristic | Trained source for every frozen phase |
| [9: independent sources](#recipe-9-independent-phase-checkpoints) | Load each learner phase from a different file | Compatible trained phase checkpoints, same hidden width |
| [10: composed opponents](#recipe-10-composed-training-opponents) | Independent frozen opponent phases | JSON definition and compatible phase checkpoints |
| [11: archived opponent](#recipe-11-training-against-an-archived-checkpoint) | Train against one fixed historical policy | Existing immutable trained snapshot/checkpoint |
| [12: controlled evaluation](#recipe-12-held-out-phase-ablations) | Compare one phase while holding continuations constant | Trained checkpoint(s), independent final deal seed |
| [13: critic comparison](#recipe-13-local-versus-centralized-critic-and-oracle-auxiliary) | Local critic (A), centralized critic (B), centralized plus oracle auxiliary (C) | Original play checkpoint; an oracle dataset for C |
| [14: from scratch](#recipe-14-from-scratch-training-and-self-play) | Random initialization of any phase subset; HHH, current-policy self-play or historical self-play | **None**: no checkpoint or anchor |

Before running a recipe, confirm the named input files exist and satisfy its
trained-phase prerequisites. Output directories must be new unless resuming. An
output path in an example is not evidence that the checkpoint already exists.
Recipe 1 can produce `runs/ppo-play-v1/checkpoints/best.pt`; later recipes use that
file as a prerequisite. Archive filenames must be replaced with actual retained
snapshot paths: publication intervals do not guarantee an exact hand-number suffix.

## Legacy trainer: train_ppo

### Recipe 1: original play-only PPO

No pretrained checkpoint is needed. This creates the version-1 original play model
with heuristic learner bid/trump and HHH opponents. The explicit episode-v1 sampling
scheme makes the intended parallel stream clear; an archived legacy-v1 reproduction
must instead retain its original sequential scheme and all original
optimizer/software settings.

```bash
python -m examples.train_ppo \
  --total-hands 524288 --rollout-play-steps 4096 \
  --ppo-epochs 4 --minibatch-size 512 \
  --learning-rate 0.0003 --sampling-scheme episode-v1 --seed 42 --device auto \
  --threads 8 --eval-threads 1 --torch-threads 1 \
  --output-dir runs/ppo-parallel-v1
```

Stage 1 learns trick play only: bidding and trump stay heuristic, two network seats
share weights, and PPO optimizes after collecting full-hand rollouts. All rollout
workers use frozen policy weights; the optimizer updates only in the parent process.
Each completed hand contributes 14 neural-controlled play decisions, so a 4096-sample
target collects 293 hands (4102 play records) in a standard full rollout.

### Legacy resume and warm-start

```bash
# Exact resume of a legacy single-worker checkpoint (cumulative hand target)
python -m examples.train_ppo \
  --resume runs/ppo-play-v1/checkpoints/latest.pt \
  --total-hands 1048576 --seed 42 --device auto \
  --threads 1 --torch-threads 1 \
  --output-dir runs/ppo-play-v1

# Exact resume of a parallel episode-v1 checkpoint
python -m examples.train_ppo \
  --resume runs/ppo-parallel-v1/checkpoints/latest.pt \
  --total-hands 1048576 --seed 42 --device auto \
  --threads 8 --eval-threads 1 --torch-threads 1 \
  --sampling-scheme episode-v1 \
  --output-dir runs/ppo-parallel-v1

# Start a NEW parallel experiment from an old model's weights
python -m examples.train_ppo \
  --warm-start runs/ppo-play-v1/checkpoints/best.pt \
  --total-hands 524288 --seed 42 --device auto \
  --threads 8 --eval-threads 1 --torch-threads 1 \
  --output-dir runs/ppo-warmstart-v1
```

> **Important:** `--resume` restores model, optimizer, counters and RNGs, and
> `--total-hands` is the **final cumulative** budget. `--warm-start` copies model
> weights only and starts fresh counters, optimizer and RNGs. The two flags are
> mutually exclusive. Exact resume cannot change sampling scheme or optimizer device
> type. Legacy `train_ppo` does not adopt the saved configuration automatically:
> repeat all training hyperparameters unchanged except the supported
> execution/budget settings.

The original single-process checkpoints use legacy-v1 sampling. The trainer uses
episode-v1 by default whenever `--threads` is not 1. To switch legacy weights to
parallel rollout collection, use `--warm-start`, not `--resume`. Parallel episode-v1
checkpoints can vary worker counts while retaining episode-seeded sampling; pass
`--sampling-scheme episode-v1` explicitly when requesting `--threads 1`.

### Legacy mixed-opponent training

```bash
python -m examples.train_ppo --total-hands 524288 --train-opponent HHH \
  --threads 8 --eval-threads 1 --output-dir runs/ppo-vs-hhh
python -m examples.train_ppo --total-hands 524288 --train-opponent CHH \
  --eval-opponent HHH --threads 8 --eval-threads 1 --output-dir runs/ppo-vs-chh
python -m examples.train_ppo --total-hands 524288 --train-opponent HHR \
  --eval-opponent HHH --threads 8 --output-dir runs/ppo-vs-random-play
python -m examples.train_ppo --total-hands 524288 \
  --train-opponents HHH:0.7,CHH:0.2,HHR:0.1 --eval-opponents HHH,CHH,HHR \
  --selection-opponent HHH --threads 8 --eval-threads 1 --output-dir runs/ppo-mixed

# Held-out evaluation of the mixed run
python -m examples.evaluate_ppo --checkpoint runs/ppo-mixed/checkpoints/best.pt \
  --opponents HHH,CHH,HHR --deals 2048 --threads 4 --seed 12345 \
  --output runs/ppo-mixed/test.json --log-dir logs/experiments/ppo-mixed-test

# Resume the mixed run, or warm-start a new mixed run
python -m examples.train_ppo --resume runs/ppo-mixed/checkpoints/latest.pt \
  --total-hands 1048576 --train-opponents HHH:7,CHH:2,HHR:1 \
  --eval-opponents HHH,CHH,HHR --selection-opponent HHH \
  --threads 4 --eval-threads 1 --output-dir runs/ppo-mixed
python -m examples.train_ppo --warm-start runs/ppo-play-v1/checkpoints/best.pt \
  --total-hands 524288 --train-opponents HHH:0.5,CHH:0.3,HHR:0.2 \
  --eval-opponents HHH,CHH,HHR --selection-opponent HHH \
  --threads 8 --output-dir runs/ppo-mixed-warmstart
```

Keep saved optimizer settings and sampling scheme unchanged for exact resume.
Collection throughput for these opponents is measured by
[`opponent_benchmark`](#performance-benchmarks). Method and migration details:
[legacy PPO](ppo_training.md#configuring-training-and-evaluation-opponents).

### train_ppo arguments

These defaults apply to legacy `examples.train_ppo` only; the multi-phase trainer has
[different defaults](#train_selfplay_ppo-arguments).

| Argument | Default | Purpose |
| --- | --- | --- |
| `--total-hands` | `100000` | Cumulative training hands budget. |
| `--rollout-play-steps` | `4096` | Minimum learned play samples collected before each PPO update; complete hands. |
| `--ppo-epochs` | `4` | Optimization passes over each rollout. |
| `--minibatch-size` | `512` | PPO optimizer minibatch size. |
| `--learning-rate` | `0.0003` | Adam learning rate. |
| `--gamma` | `1.0` | Discount factor per controlled-seat decision. |
| `--gae-lambda` | `0.95` | Generalized advantage estimation parameter. |
| `--clip-epsilon` | `0.2` | PPO clipped surrogate ratio limit. |
| `--entropy-coef` | `0.01` | Policy entropy regularization weight. |
| `--value-coef` | `0.5` | Value loss weight. |
| `--max-grad-norm` | `0.5` | Maximum norm for gradient clipping. |
| `--target-kl` | `0.0` | Optional KL early-stop threshold; 0 disables. |
| `--hidden-size` | `256` | Width of both MLP hidden layers. |
| `--seed` | `42` | Training/deal seed. |
| `--device` | `auto` | CPU or CUDA optimizer device; auto selects CUDA when available. |
| `--sampling-scheme` | `auto` | `auto`, `legacy-v1` or `episode-v1`; governs action sampling streams. |
| `--threads` | `1` | PPO simulation worker request (`0` = auto). |
| `--eval-threads` | `1` | Validation workers; runtime-only, not in checkpoint training config. |
| `--torch-threads` | `1` | Main-process PyTorch CPU intra-op threads; distinct from workers. |
| `--train-opponent` / `--train-opponents` | HHH | One variant, or a `NAME:WEIGHT` mixture; mutually exclusive. |
| `--eval-opponent` / `--eval-opponents` | HHH | One or comma-separated variants; mutually exclusive. |
| `--selection-opponent` | HHH when present | Primary validation opponent for `best.pt`; must belong to the evaluation set. |
| `--eval-every-hands` | `5000` | Periodic validation interval; may overshoot by one rollout. |
| `--eval-deals` | `128` | Duplicate groups per periodic validation. |
| `--bootstrap-samples` | `200` | Bootstrap replicates per periodic validation. |
| `--checkpoint-every-hands` | `5000` | Periodic `latest.pt` saving interval. |
| `--output-dir` | `runs/ppo-play-v1` | Training config, progress and checkpoints. |
| `--log-dir` | Auto subdirectory | Readable `training.log` and `training.csv`. |
| `--resume` | Unset | Exact checkpoint continuation; mutually exclusive with `--warm-start`. |
| `--warm-start` | Unset | Initialize a new run from model weights only. |

### train_ppo outputs

| Location | Contents |
| --- | --- |
| `runs/<run>/config.json` | Resolved training configuration (including `total_hands`, worker request). |
| `runs/<run>/progress.jsonl` | Per-update metrics and full periodic validation reports. |
| `runs/<run>/checkpoints/latest.pt` | Most recently saved complete training state; use for resume. |
| `runs/<run>/checkpoints/best.pt` | Best fixed-schedule validation mean signed marks. |

## Multi-phase trainer: train_selfplay_ppo

Use for new experiments with independently configured phases, neural opponents, or
historical self-play. The recipes below continue the
[workflow finder](#workflow-finder) numbering; read the
[terminology](#hands-samples-and-optimizer-steps) first if counters are unfamiliar.

### Recipe 2: bidding-only (frozen play)

Stage A. Requires an existing compatible original play-only checkpoint, for example
Recipe 1's `best.pt`. Explicit modern states preserve that play network bitwise and
create new bid heads from its representation. Trump is heuristic. Set imitation hands
to zero to ablate the optional teacher initialization.

```bash
python -m examples.train_selfplay_ppo \
  --init-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --learner-bid neural-train --learner-bid-source new \
  --learner-trump heuristic --learner-play neural-frozen --learner-play-source anchor \
  --bid-bc-hands 4096 --opponent-pool HHH:0.5,anchor:0.5 \
  --eval-opponents HHH,CHH,anchor --selection-opponent anchor \
  --total-hands 262144 --rollout-decisions 1024 \
  --ppo-epochs 4 --minibatch-size 256 --learning-rate 0.0001 \
  --threads 8 --eval-threads 1 --device auto --seed 42 \
  --output-dir runs/ppo-bidding-v1 --log-dir logs/experiments/ppo-bidding-v1
```

Check `phase_samples.bid`, `frozen_phases_verified.play=true`, and the source SHA256.
For an independent parameter-equality check, use the
[controlled bidding evaluation](#hold-joint-playing-fixed-change-bidding). The older
equivalent routing syntax is `--learn-phases bid --play-policy frozen
--trump-policy heuristic`; it retains its own established initialization sequence.

### Recipe 3: playing-only with the modern trainer

Requires the original play checkpoint. Use this path to include play in a version-2
experiment that can later change phases/opponents. For an exact legacy method/seed
reproduction use [Recipe 1](#recipe-1-original-play-only-ppo) instead; neither path
guarantees identical training samples to the other.

```bash
python -m examples.train_selfplay_ppo \
  --init-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --learner-bid heuristic --learner-trump heuristic --learner-play neural-train \
  --opponent-pool HHH:1 --eval-opponents HHH --selection-opponent HHH \
  --total-hands 100000 --rollout-decisions 4096 --ppo-epochs 4 --minibatch-size 512 \
  --threads 4 --eval-threads 1 --seed 42 --device auto \
  --output-dir runs/ppo-modern-play --log-dir logs/experiments/ppo-modern-play
```

### Recipe 4: trump-only

Requires a version-2 joint checkpoint with trained neural bid/play and its original
anchor file. Neither bid nor play is updated; trump gets new heads. A defending
learner contributes zero samples, a declaring learner exactly one. The sample target
can require many hands; lower the hand cap for bounded short checks. Empty
capped/final batches skip optimization but still advance progress. Every hand
executes all seven tricks.

```bash
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --learner-bid neural-frozen --learner-trump neural-train --learner-trump-source new \
  --learner-play neural-frozen \
  --opponent-checkpoint pretrump=runs/ppo-joint-v1/checkpoints/best.pt \
  --opponent-pool pretrump:0.7,HHH:0.3 \
  --eval-opponents pretrump,HHH,anchor --selection-opponent pretrump \
  --total-hands 262144 --rollout-decisions 1024 --max-rollout-hands 4096 \
  --threads 8 --eval-threads 4 --device auto --seed 42 \
  --output-dir runs/ppo-trump-v1 --log-dir logs/experiments/ppo-trump-v1
```

Check both frozen bid/play fields, actual trump sample counts and `trump_*` metrics.
`pretrump` must route the same bidding/playing with heuristic trump to isolate that
phase in validation. If its checkpoint already routes neural trump, register a
composed heuristic-trump opponent instead, as in
[Recipe 10](#recipe-10-composed-training-opponents).

### Recipe 5: all three phases against HHH

Requires Recipe 1's original play checkpoint, not a pretrained bid or trump policy.
Both new phases transfer its trunk with fresh policy/value heads; playing starts from
the trained whole network. All three modules are optimized.

```bash
python -m examples.train_selfplay_ppo \
  --init-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --learner-bid neural-train --learner-bid-source new \
  --learner-trump neural-train --learner-trump-source new \
  --learner-play neural-train --learner-play-source init \
  --opponent-pool HHH:1 --eval-opponents HHH,CHH,anchor --selection-opponent HHH \
  --bid-bc-hands 0 --total-hands 262144 --rollout-decisions 4096 \
  --ppo-epochs 4 --minibatch-size 512 --learning-rate 0.0001 \
  --snapshot-every-hands 32768 --snapshot-pool-size 8 \
  --threads 8 --eval-threads 1 --device auto --seed 42 \
  --output-dir runs/ppo-allphase-hhh --log-dir logs/experiments/ppo-allphase-hhh
```

Each hand yields 2 bid + 14 play samples, and 0 or 1 trump sample (16–17 total).
Inspect all three phase counts/metrics, bid/contract distributions, and validation
against HHH, CHH and anchor. There are no frozen learner networks in this recipe.
Snapshots can be published even though HHH remains the sole sampled training
opponent. Before reusing `best.pt` as a whole frozen opponent, ensure metadata
records all three routed phases as trained; merely declaring a phase neural is
insufficient.

### Recipe 6: joint training with historical opponents

Stage B. Requires Recipe 5's `best.pt` and its immutable original anchor. This is a
**new stage**, because the opponent distribution changes. It copies the three phase
networks and uses fresh Adam/RNG/counters; it starts with an empty historical roster.
It does not resume Recipe 5's optimizer or import its snapshots.

```bash
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-allphase-hhh/checkpoints/best.pt \
  --learner-bid neural-train --learner-trump neural-train --learner-play neural-train \
  --opponent-pool HHH:0.2,anchor:0.2,self-snapshots:0.6 \
  --eval-opponents HHH,CHH,anchor --selection-opponent HHH \
  --snapshot-every-hands 32768 --snapshot-pool-size 8 \
  --total-hands 524288 --rollout-decisions 4096 \
  --ppo-epochs 4 --minibatch-size 512 --learning-rate 0.0001 \
  --threads 8 --eval-threads 1 --device auto --seed 42 \
  --output-dir runs/ppo-allphase-selfplay --log-dir logs/experiments/ppo-allphase-selfplay
```

Before publication, the self-snapshots portion also selects anchor (80% effective
anchor probability in this example). Thereafter it selects uniformly from active
frozen stage snapshots. HHH and anchor retain their fixed weights/rules, offering
stable diversity and forgetting comparisons; this is not a guarantee against
forgetting. Repeated checkpoints need not get monotonically stronger.

For the earlier **bid/play-only** Stage B workflow, initialize from Recipe 2 and set
bid/play `neural-train`, trump `heuristic`, and a new output directory such as
`runs/ppo-joint-v1`. Those explicit states keep the legacy named workflow usable.

### Recipe 7: resume and multi-opponent validation

Requires `latest.pt` and all recorded source/opponent/roster files. Use the same
output directory to append progress, or a new one to retain only continuation
progress locally. `--total-hands` is the cumulative target, not an increment.

```bash
python -m examples.train_selfplay_ppo \
  --resume runs/ppo-allphase-selfplay/checkpoints/latest.pt \
  --total-hands 1048576 --threads 4 --eval-threads 1 \
  --output-dir runs/ppo-allphase-selfplay --log-dir logs/experiments/ppo-allphase-selfplay

python -m examples.evaluate_selfplay_ppo \
  --checkpoint runs/ppo-bidding-v1/checkpoints/best.pt \
  --opponents HHH,CHH,HHR,anchor --deals 4096 --seed 12345 \
  --threads 4 --output runs/ppo-bidding-v1/heldout.json
```

| On resume you may change | Requires a new stage (`--init-checkpoint`, new directory) |
| --- | --- |
| Cumulative `--total-hands`, rollout `--threads` | Phase states/sources, architecture, optimizer settings |
| `--eval-threads`, output/log locations | Canonical opponent distribution, evaluation procedure, selection criterion |
| | CPU↔CUDA optimizer device type (not exact resume) |

Modern resume restores omitted stored settings; legacy `train_ppo` instead requires
you to repeat the saved configuration. `best.pt` is selected performance, `latest.pt`
is newest saved progress. Extending a run that ended with a shortened final rollout
resumes that saved state, but is not equivalent to having used the longer budget from
the beginning.

### Recipe 8: mixed-phase training

Bidding-train/play-frozen/trump-heuristic is [Recipe 2](#recipe-2-bidding-only-frozen-play);
trump-train with bid/play frozen is [Recipe 4](#recipe-4-trump-only). Here is bid/play
training with frozen **neural trump**. Requires a checkpoint with all three phases
trained, such as Recipe 5's `best.pt`.

```bash
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-allphase-hhh/checkpoints/best.pt \
  --learner-bid neural-train --learner-trump neural-frozen --learner-play neural-train \
  --opponent-pool HHH:0.5,anchor:0.5 --eval-opponents HHH,anchor --selection-opponent HHH \
  --total-hands 262144 --rollout-decisions 4096 --threads 4 --eval-threads 1 \
  --output-dir runs/ppo-fixed-trump --log-dir logs/experiments/ppo-fixed-trump
```

Only bid/play observations enter PPO (16 per hand); frozen trump influences the
trajectory but never enters the buffer/optimizer. Inspect its frozen verification
field. Other nonempty subsets can be spelled with the same three state flags.

### Recipe 9: independent phase checkpoints

Requires the named compatible checkpoints, a joint initialization checkpoint and its
anchor. All learner sources must share the initialization width. The example freezes
bidding/play from different stages while updating trained trump.

```bash
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --opponent-checkpoint bidding=runs/ppo-bidding-v1/checkpoints/best.pt \
  --opponent-checkpoint trump_archive=runs/ppo-trump-v1/checkpoints/best.pt \
  --learner-bid neural-frozen --learner-bid-source bidding \
  --learner-trump neural-train --learner-trump-source trump_archive \
  --learner-play neural-frozen --learner-play-source runs/ppo-joint-v1/checkpoints/best.pt \
  --opponent-pool HHH:1 --eval-opponents HHH,anchor --selection-opponent HHH \
  --total-hands 100000 --rollout-decisions 1024 --threads 4 --eval-threads 1 \
  --output-dir runs/ppo-independent-sources --log-dir logs/experiments/ppo-independent-sources
```

Sources are copied once, not followed as their original experiments train further.
Do not overwrite files referenced by this experiment. Snapshot files can replace the
named source paths when their metadata declares the needed trained phase.

### Recipe 10: composed training opponents

Requires a trained joint bid/play checkpoint and its original anchor. Create
`runs/pretrump-composition.json` after ensuring `runs/` exists:

```json
{
  "bid": {"strategy": "neural", "source": "pretrump"},
  "trump": {"strategy": "heuristic"},
  "play": {"strategy": "neural", "source": "pretrump"}
}
```

```bash
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --learner-bid neural-frozen --learner-trump neural-train --learner-trump-source new \
  --learner-play neural-frozen \
  --opponent-checkpoint pretrump=runs/ppo-joint-v1/checkpoints/best.pt \
  --opponent-composition joint_heuristic_trump=runs/pretrump-composition.json \
  --opponent-pool joint_heuristic_trump:0.7,HHH:0.3 \
  --eval-opponents joint_heuristic_trump,HHH --selection-opponent joint_heuristic_trump \
  --total-hands 262144 --rollout-decisions 1024 --threads 4 --eval-threads 1 \
  --output-dir runs/ppo-trump-composed --log-dir logs/experiments/ppo-trump-composed
```

**JSON format.** Exactly the keys `bid`, `trump`, `play`; each value has a `strategy`
and, for neural only, a `source`. Bid strategies are neural/heuristic/random/
conservative; trump/play are neural/heuristic/random. Sources accept `init`,
`anchor`, registered checkpoint names, or direct paths, including retained snapshots.
Paths are relative to the working directory, not to the JSON file. `new` is not an
opponent source: frozen opponents need trained networks. Different phase files and
widths are allowed for composed **opponent** inference if each is independently
compatible; learner initialization still requires one shared width.

A learner source initializes an optimizable copy, whereas this JSON configures a
read-only opponent factory. JSON content and resolved identities are saved in the
training configuration; exact resume does not require the original JSON file when
registration flags are omitted, but it does require all referenced checkpoint files.

### Recipe 11: training against an archived checkpoint

Requires Recipe 5's trained all-phase checkpoint and an existing immutable archive at
`runs/archive/historical.pt` (copy/reference an actual retained snapshot there;
retain its dependencies for resumption). Unlike `self-snapshots`, this explicitly
registered identity never changes its membership or weights.

```bash
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-allphase-hhh/checkpoints/best.pt \
  --learner-bid neural-train --learner-trump neural-train --learner-play neural-train \
  --opponent-checkpoint archived=runs/archive/historical.pt \
  --opponent-pool HHH:0.3,archived:0.7 \
  --eval-opponents HHH,archived,anchor --selection-opponent archived \
  --total-hands 262144 --rollout-decisions 4096 --threads 4 --eval-threads 1 \
  --output-dir runs/ppo-versus-archive --log-dir logs/experiments/ppo-versus-archive
```

### Recipe 13: local versus centralized critic and oracle auxiliary

Three matched runs: **A** keeps the existing per-phase local value heads, **B**
trains a separate centralized critic on privileged features, and **C** adds the
oracle auxiliary head. Actor architecture and initialization, the seed, opponent
pool, hand budget and validation schedule are identical; only the critic flags
differ, so the first rollout is identical in all three. Requires the original
play checkpoint and, for C, a dataset from [oracle training labels](#oracle-training-labels).
See [critic modes](multiphase_selfplay.md#critic-modes-returns-and-oracle-auxiliary-supervision)
for the method.

```bash
COMMON=(--init-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt
  --learner-bid neural-train --learner-bid-source new --learner-trump heuristic
  --learner-play neural-train --opponent-pool HHH:0.5,anchor:0.5
  --eval-opponents HHH,CHH,anchor --selection-opponent anchor
  --total-hands 262144 --rollout-decisions 4096 --ppo-epochs 4 --minibatch-size 512
  --learning-rate 0.0001 --seed 42 --threads 8 --eval-threads 1 --device auto)
python -m examples.train_selfplay_ppo "${COMMON[@]}" --critic-mode local \
  --output-dir runs/critic-A-local --log-dir logs/experiments/critic-A-local
python -m examples.train_selfplay_ppo "${COMMON[@]}" --critic-mode centralized \
  --output-dir runs/critic-B-central --log-dir logs/experiments/critic-B-central
python -m examples.train_selfplay_ppo "${COMMON[@]}" --critic-mode centralized \
  --oracle-aux-coef 0.5 --oracle-dataset runs/oracle-labels/late-v1.json \
  --output-dir runs/critic-C-oracle --log-dir logs/experiments/critic-C-oracle
```

Evaluate every run against the same fixed opponents on one held-out seed, then
round-robin the runs head to head (neural bid/play, heuristic trump on both sides):

```bash
for RUN in critic-A-local critic-B-central critic-C-oracle; do
  python -m examples.evaluate_selfplay_ppo --checkpoint runs/$RUN/checkpoints/best.pt \
    --opponents HHH,CHH,HHR,anchor --deals 4096 --seed 27182818 --threads 4 \
    --output runs/$RUN/heldout.json
done
for PAIR in "critic-A-local critic-B-central" "critic-A-local critic-C-oracle" \
            "critic-B-central critic-C-oracle"; do
  set -- $PAIR
  python -m examples.evaluate_agent_vs_agent \
    --agent-a-checkpoint runs/$1/checkpoints/best.pt --agent-a-bid neural --agent-a-play neural \
    --agent-b-checkpoint runs/$2/checkpoints/best.pt --agent-b-bid neural --agent-b-play neural \
    --deals 4096 --seed 27182818 --bootstrap-samples 2000 --threads 4 \
    --output runs/$1-vs-$2.json
done
```

Repeat with `latest.pt` and with further training seeds before drawing conclusions;
`best.pt` is selected against the anchor on the small validation schedule, so the
fixed-opponent and head-to-head reports above are the comparison, not the selection
score. Add `--return-estimator monte-carlo` to any run for the complete-hand return
baseline. A tiny smoke version: `--total-hands 48 --rollout-decisions 256
--minibatch-size 64 --eval-deals 8` with the smoke label dataset.

### Recipe 14: from-scratch training and self-play

Requires **no checkpoint at all**. `--from-scratch` randomly initializes every
`neural-train` phase (trunk, policy and value heads) from `--seed`, starts Adam,
counters and RNG streams fresh, and records `initialization: random`, no
initialization checkpoint and no original anchor. Without phase flags all three
phases train; with explicit `--learner-*` or `--learn-phases` flags, unmentioned
phases are heuristic. `neural-frozen` phases and `--learner-*-source` flags are
rejected (there is nothing trained to freeze or copy). Defaults become
`--opponent-pool HHH:1 --eval-opponents HHH,CHH --selection-opponent HHH`; `anchor`
is unavailable unless an `--anchor-checkpoint` is supplied explicitly. Random
weights are not a playing strength: early self-play policies can be very weak.

```bash
# A. All three phases from random weights against HHH
python -m examples.train_selfplay_ppo --from-scratch \
  --learner-bid neural-train --learner-trump neural-train --learner-play neural-train \
  --opponent-pool HHH:1 --eval-opponents HHH,CHH --selection-opponent HHH \
  --total-hands 262144 --rollout-decisions 4096 --threads 8 --seed 42 \
  --output-dir runs/ppo-scratch-hhh

# B. Immediate current-policy self-play from the first hand
python -m examples.train_selfplay_ppo --from-scratch \
  --learner-bid neural-train --learner-trump neural-train --learner-play neural-train \
  --opponent-pool self-current:1 --eval-opponents HHH,CHH --selection-opponent HHH \
  --total-hands 262144 --rollout-decisions 4096 --threads 8 --seed 42 \
  --output-dir runs/ppo-scratch-selfplay

# C. Mixed heuristic and self-play opponents, centralized critic
python -m examples.train_selfplay_ppo --from-scratch \
  --learner-bid neural-train --learner-trump neural-train --learner-play neural-train \
  --critic-mode centralized --return-estimator gae \
  --opponent-pool HHH:0.3,CHH:0.2,self-current:0.5 \
  --eval-opponents HHH,CHH --selection-opponent HHH \
  --total-hands 262144 --rollout-decisions 4096 --threads 8 --seed 42 \
  --output-dir runs/ppo-scratch-centralized

# D. Historical snapshot self-play without any pretrained checkpoint
python -m examples.train_selfplay_ppo --from-scratch \
  --learner-bid neural-train --learner-trump neural-train --learner-play neural-train \
  --opponent-pool self-snapshots:0.7,HHH:0.3 \
  --snapshot-every-hands 32768 --snapshot-pool-size 8 \
  --eval-opponents HHH,CHH --selection-opponent HHH \
  --total-hands 524288 --rollout-decisions 4096 --threads 8 --seed 42 \
  --output-dir runs/ppo-scratch-historical

# E. Only trick play from random weights; heuristic bidding and trump
python -m examples.train_selfplay_ppo --from-scratch \
  --learner-bid heuristic --learner-trump heuristic --learner-play neural-train \
  --opponent-pool HHH:1 --eval-opponents HHH --selection-opponent HHH \
  --total-hands 262144 --rollout-decisions 4096 --threads 8 --seed 42 \
  --output-dir runs/ppo-scratch-play
```

Resume, evaluate, or start a new stage from the result without any anchor:

```bash
python -m examples.train_selfplay_ppo --resume runs/ppo-scratch-selfplay/checkpoints/latest.pt \
  --total-hands 524288 --output-dir runs/ppo-scratch-selfplay
python -m examples.evaluate_selfplay_ppo --checkpoint runs/ppo-scratch-selfplay/checkpoints/best.pt \
  --deals 4096 --threads 4          # default opponents HHH,CHH,HHR when there is no anchor
python -m examples.evaluate_agent_vs_agent \
  --agent-a-checkpoint runs/ppo-scratch-selfplay/checkpoints/best.pt \
  --agent-a-bid neural --agent-a-trump neural --agent-a-play neural \
  --agent-b-bid heuristic --agent-b-trump heuristic --agent-b-play heuristic \
  --deals 4096 --threads 4
python -m examples.train_selfplay_ppo --init-checkpoint runs/ppo-scratch-hhh/checkpoints/best.pt \
  --learner-bid neural-train --learner-trump neural-train --learner-play neural-train \
  --opponent-pool HHH:0.3,self-snapshots:0.7 --total-hands 262144 --rollout-decisions 4096 \
  --threads 8 --output-dir runs/ppo-scratch-stage2
```

`--resume` restores weights, critic, Adam, RNGs, counters, the snapshot roster,
opponent distribution, selection settings and the random-initialization
provenance. `--init-checkpoint` starts a **new stage** (fresh Adam, counters and
roster); a scratch lineage stays anchor-free, so its defaults are the HHH ones above.

Recipe 12 (held-out evaluation) is in [Evaluation](#recipe-12-held-out-phase-ablations).

### train_selfplay_ppo arguments

Defaults apply to a new run; omitted settings on resume normally come from the saved
configuration. Values are checked against argparse, `Config` and `SelfPlayConfig`.

| Group | Arguments |
| --- | --- |
| [A. Budget and PPO optimization](#a-budget-and-ppo-optimization) | `--total-hands`, `--rollout-decisions`, `--max-rollout-hands`, `--ppo-epochs`, `--minibatch-size`, `--learning-rate`, `--gamma`, `--gae-lambda`, `--clip-epsilon`, `--entropy-coef`, `--value-coef`, `--max-grad-norm`, `--target-kl`, `--hidden-size` |
| [B. Learner phases](#b-learner-phases-and-legacy-compatibility) | `--learner-{bid,trump,play}`, `--learner-{bid,trump,play}-source`, `--learn-phases`, `--play-policy`, `--trump-policy` |
| [C. Entry points and sources](#c-checkpoint-entry-points-and-neural-sources) | `--init-play-checkpoint`, `--init-checkpoint`, `--anchor-checkpoint`, `--resume` |
| [D. Opponents and snapshots](#d-training-opponents-and-historical-snapshots) | `--opponent-pool`, `--opponent-checkpoint`, `--opponent-composition`, `--snapshot-every-hands`, `--snapshot-pool-size` |
| [E. Validation and selection](#e-validation-and-checkpoint-selection) | `--eval-opponents`, `--selection-opponent`, `--eval-every-hands`, `--eval-deals`, `--bootstrap-samples`, `--checkpoint-every-hands` |
| [F. Execution and diagnostics](#f-execution-and-diagnostics) | `--seed`, `--device`, `--threads`, `--eval-threads`, `--torch-threads`, `--output-dir`, `--log-dir`, `--forgetting-hands`, `--bid-bc-hands` |
| [G. Critic, returns and oracle auxiliary](#g-critic-returns-and-oracle-auxiliary) | `--critic-mode`, `--return-estimator`, `--oracle-aux-coef`, `--oracle-dataset` |

#### A. Budget and PPO optimization

| Argument | Type / accepted values | New-run default | Effect, interactions and consequences |
| --- | --- | --- | --- |
| `--total-hands` | Integer > 0 | `100000` | Cumulative completed training-hand budget, including earlier hands on resume. Finishes hands; never exceeds this budget. It is not a sample or update count. |
| `--rollout-decisions` | Integer > 0 | `1024` | Target number of **trainable learner actions** in one fixed-policy rollout. Frozen, heuristic and opponent actions do not count. Whole hands/waves may overshoot; the final budget-limited batch can be smaller. |
| `--max-rollout-hands` | Integer > 0 | `4096` | Caps hands per **trump-only** rollout, even if no samples occur. Has no effect when bid or play is trainable. Reaching the cap returns collected samples, possibly none, so another rollout can continue within the total budget. |
| `--ppo-epochs` | Integer > 0 | `4` | Passes over the collected batch; each pass reshuffles. Increases data reuse and potential policy drift, not environment hands. `--target-kl` can stop passes early. |
| `--minibatch-size` | Integer > 0 | `512` | Samples per Adam step. Last minibatch may be smaller; a batch smaller than this still runs. Influences optimizer-step count and mixed-phase composition. |
| `--learning-rate` | Finite float > 0 | `0.0001` | Adam rate for all enabled phase modules. There is no separate per-phase rate. The legacy trainer defaults to `0.0003`. |
| `--gamma` | Float in `[0,1]` | `1.0` | Discount per **that seat's trainable decision**, not every engine turn. Frozen/heuristic continuations create no extra discount steps. Changing trained phases changes this decision-time convention. |
| `--gae-lambda` | Float in `[0,1]` | `0.95` | GAE decay within a complete seat/hand trajectory. Lower values rely more on old value estimates; no GAE crosses seats or hands. |
| `--clip-epsilon` | Finite float in `(0,1)` | `0.2` | PPO probability-ratio clipping range `1±epsilon`; uses original legal masks/log probabilities. Does not guarantee a maximum KL change. |
| `--entropy-coef` | Finite float ≥ 0 | `0.01` | Subtracted legal-policy entropy weight in the loss. Zero disables it; it applies only to trained actions. No illegal/reserved logits contribute. |
| `--value-coef` | Finite float ≥ 0 | `0.5` | Weight on signed-mark return MSE. Zero removes critic regression from the combined loss, while GAE still uses old value predictions. |
| `--max-grad-norm` | Finite float > 0 | `0.5` | Gradient-norm clipping before Adam. Logged `gradient_norm` is measured **before** clipping. Frozen modules have no gradients. |
| `--target-kl` | Finite float ≥ 0 | `0.0` | Zero disables early stopping. Otherwise checks approximate KL before applying each minibatch and skips the violating minibatch and remaining passes. An excessively small threshold can produce a no-optimizer-steps error. |
| `--hidden-size` | Integer > 0, compatible with sources | Initialization checkpoint's width; argparse fallback `256` | Width of both hidden layers in every phase network. Omission adopts the loaded checkpoint width; explicit mismatch is rejected. All learner sources must match this width, encoder, ruleset and action vocabulary. |

##### Hands, samples and optimizer steps

A **hand** executes four auction decisions, one trump decision and 28 plays. A
**sample** is one enabled learner action. A **rollout** is the complete-hand
collection made before changing weights. A **minibatch** is a shuffled slice of that
rollout. An **epoch** visits its samples once; an **optimizer step** updates weights
using one minibatch. With `B` samples and no KL early stop, one PPO update uses
`ppo_epochs × ceil(B/minibatch_size)` optimizer steps; an empty batch has none.

For example, bidding-only needs 512 hands for a full 1024-sample batch; play-only
needs 74 hands and supplies 1036 samples. More hands do not imply proportionally more
updates if batch size, sparse trump samples, final truncation or learning phases
change. Samples are collected again from the updated policy next rollout.

#### B. Learner phases and legacy compatibility

| Argument | Accepted values | New-run default / resolution | Meaning |
| --- | --- | --- | --- |
| `--learner-bid` | `heuristic`, `neural-frozen`, `neural-train` | Unset; see rules below | Independent bidding route and optimizer membership. |
| `--learner-trump` | Same three states | Unset; see rules below | Only the winning bidder acts; training samples can be sparse. |
| `--learner-play` | Same three states | Unset; see rules below | Seven play actions per learner seat if trainable. |
| `--learner-bid-source` | [Source](#learner-sources) token or path | Unset → `init` for a neural phase | Resolve only the bidding initialization. |
| `--learner-trump-source` | Source token or path | Unset → `init` for a neural phase | Resolve only the trump initialization. |
| `--learner-play-source` | Source token or path | Unset → `init` for a neural phase | Resolve only the playing initialization. |
| `--learn-phases` | Nonempty distinct comma-separated subset of `bid,trump,play`, no spaces | `bid` | Legacy set of trainable neural phases; order canonicalizes to bid/trump/play. Explicit use alongside modern states must identify exactly the same trained set. |
| `--play-policy` | `frozen`, `trainable` | Unset | Legacy assertion: in modern syntax requires `neural-frozen` or `neural-train` play respectively; heuristic play satisfies neither. With legacy syntax it must agree with membership in `--learn-phases`. |
| `--trump-policy` | `heuristic`, `neural` | argparse `heuristic`; asserted only if explicitly supplied | Checks the effective trump route. Omission does not switch an inherited neural trump module to heuristic. Use explicit learner states for independent routing. |

| Effective state | Rollout action selection | In buffer / Adam |
| --- | --- | --- |
| `heuristic` | Existing deterministic HeuristicAgent V1 | No / no |
| `neural-frozen` | Greedy legal action from frozen network | No / no |
| `neural-train` | Masked stochastic action from the rollout's policy snapshot | Yes / yes |

**Modern omission rules.** If any learner state **or source** flag is supplied, the
modern configuration path is active. Unmentioned phases declared neural in the
initialization checkpoint default to `neural-frozen`; the others default to
`heuristic`. An explicit `--learn-phases` makes otherwise unspecified members
`neural-train`. There must ultimately be at least one trainable phase. A source flag
alone does not opt its phase into training. Resume instead inherits saved states and
sources before checking explicit overrides.

**Legacy construction.** Without any learner flags, `--learn-phases` defaults to
`bid`, enables neural learning for those phases, and retains existing neural
continuations from version-2 initialization (play remains neural for version 1).
Thus default version-1 initialization learns bid with frozen play and heuristic
trump; a version-2 file declaring neural trump can retain frozen neural trump. Do not
rely on the literal argparse trump default to disable it. Different initialization
paths are not promised to yield identical newly initialized weights or RNG
sequences; preserve the original syntax/environment for legacy reproduction.

**Constraints.** All nonempty trainable subsets are supported, including trump alone.
Every frozen neural phase must have a trained compatible source. All-frozen/heuristic
configurations belong to evaluation, not training. Conflicting explicit legacy and
modern choices fail before training; prefer spelling all three modern states for a
new experiment. An older run whose saved configuration used legacy syntax must resume
with that syntax, or initialize a new stage to adopt source flags.

#### C. Checkpoint entry points and neural sources

| Argument | Type / accepted values | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `--init-play-checkpoint` | Path to compatible version-1 checkpoint | Unset | Pretrained play-only initialization; rejects a version-2 model. New optimizer, counters, RNGs and roster. |
| `--init-checkpoint` | Path to compatible version-1 or version-2 checkpoint | Unset | New stage from model parameters. Does not import optimizer, old best score, progress or historical roster. |
| `--anchor-checkpoint` | Path to original version-1 play checkpoint | Inferred | Version-1 initialization uses that same file as anchor. Version-2 initialization/resume retains the original anchor identity from metadata; an override must match its SHA256. |
| `--resume` | Path to compatible version-2 checkpoint | Unset | Exact saved-stage continuation. Mutually exclusive with the two initialization flags. Restores omitted settings, model, optimizer, counters, RNGs and active roster. |
| `--from-scratch` | Flag | Off | Random initialization of every `neural-train` phase; loads no checkpoint and needs no anchor ([Recipe 14](#recipe-14-from-scratch-training-and-self-play)). An `--anchor-checkpoint` is optional and only registers `anchor` as an opponent. |

If no entry-point flag is supplied, the CLI loads
`runs/ppo-play-v1/checkpoints/best.pt`; it does **not** create a pretrained model.
Checkpoint-initialized experiments from a play-only file use it as the original
anchor, even if learner play is heuristic; version-2 stages keep the anchor
recorded in their lineage, and an anchor-free (scratch) lineage stays anchor-free.
The four entry-point flags are mutually exclusive. A renamed/moved anchor can be
supplied by `--anchor-checkpoint` with the same bytes, but other recorded immutable
dependencies still need to be accessible at their saved paths.

##### Learner sources

| Source | Resolution / initialization |
| --- | --- |
| `init` | The initialization checkpoint. Copy the full requested trained phase (trunk, policy and value heads). Only for `neural-train`, a missing trained phase instead gets representation transfer and fresh heads, explicitly recorded. On resume, explicit `init` refers to the original stage initialization identity, not the evolving resume file. |
| `anchor` | The immutable original play-only checkpoint. Valid for neural play only; bid/trump requests fail even when trainable. |
| (none, `--from-scratch`) | Recorded as initialization `random`: no source, the network's normal random initialization. Not accepted as a `--learner-*-source` value. |
| `new` | Trainable phases only. From the initialization file, copy the trained play trunk, or the first trained phase trunk in bid/trump/play order if play is unavailable. Initialize fresh policy/value heads even if the requested phase was already trained. Requires a trained representation. |
| Registered name | Register with `--opponent-checkpoint NAME=PATH`. Must identify a single checkpoint containing the requested trained neural phase; variants and composed policies are not phase sources. |
| Direct or archived path | Same trained-phase requirement as a registered name. An archived snapshot is a fixed file, not a live subscription to later snapshots. |

All heuristic phases reject sources; frozen phases cannot use `new` or untrained
weights. Version-1 output layers have bid/trump logits, but those are **not trained
bid/trump policies**. Version-2 sources must declare the requested phase in both
`neural_phases` and `trained_phases`. Source paths resolve to absolute paths and
SHA256 before training. Source parameters are copied into independent learner
modules; training never modifies the file. Exact resume verifies identities and
optimizer membership. Different phase sources may be supplied independently, subject
to learner-width compatibility. See
[mechanics](multiphase_selfplay.md#initialization-and-freezing).

#### D. Training opponents and historical snapshots

| Argument | Type / accepted values | Default | Meaning and interactions |
| --- | --- | --- | --- |
| `--opponent-pool` | Nonempty `NAME:WEIGHT,...`; unique registered names, `self-current`, `self-snapshots`; finite positive weights | `HHH:0.5,anchor:0.5` with an anchor, else `HHH:1` | One opponent controls both opposing seats for a complete hand. Ratios normalize to sorted exact rational probabilities; `HHH:7,anchor:3` equals `HHH:0.7,anchor:0.3`. Realized counts fluctuate. |
| `--opponent-checkpoint` | Repeatable `NAME=PATH` | No archives | Register an immutable compatible checkpoint for learner sources, training or validation. Names cannot duplicate built-ins/anchor or reserved `init`, `new`, `self-snapshots`, `self-current`. Active whole-checkpoint opponents need all routed neural phases trained. |
| `--opponent-composition` | Repeatable `NAME=JSON_PATH` | No compositions | Register a frozen phase-composable opponent. Source checkpoint registrations are resolved before JSON definitions; earlier composed names cannot serve as single phase sources. See [Recipe 10](#recipe-10-composed-training-opponents). |
| `--snapshot-every-hands` | Integer > 0 | `32768` | Publication threshold since last snapshot, checked at rollout/update boundaries. Publishes only following a nonempty batch when every routed neural phase is trained. Threshold may overshoot; not every exact multiple produces a filename. |
| `--snapshot-pool-size` | Integer > 0 | `8` | Maximum active historical policies. Evicts oldest membership, not files. The separate anchor is never evicted. No CLI switch disables publication; snapshots can be produced even if the current pool does not sample them. |

**Opponent names.** Built-in variants are defined in
[the variant table](#built-in-policy-variants). `anchor` uses heuristic bid/trump and
the original frozen neural play. Registered checkpoints restore their declared neural
phases and use heuristic fallbacks. Composed opponents specify phases independently,
including different neural files. All opponents remain frozen and use greedy neural
inference.

**`self-current`** seats a frozen copy of the learner's own actor weights, taken at
the start of each rollout (the same snapshot the learner collects with) and
refreshed for the next rollout after the PPO update. It samples stochastically from
the legal softmax on trainable phases, is greedy on frozen neural phases, keeps
heuristic phases heuristic, and uses its own per-seat action streams. Its decisions
never enter the PPO buffer. The progress log records the snapshot's SHA256.

**`self-snapshots`** first uses the **original anchor** if the active roster is
empty in an anchor-based experiment; it does not automatically use the stage's
initial joint model. In a from-scratch or anchor-free lineage it instead uses the
frozen current policy (as `self-current`) until a snapshot is published. Once
published policies exist it selects uniformly from the active roster using a
separate seed stream. Snapshots are published only when every routed neural phase
has been trained. A new stage starts with an empty roster; resume restores the
exact saved roster. Keep files immutable and preserve evicted archives needed by
older checkpoints. Both self-play modes optimize **one** learning partnership
against frozen opponents; they are not four simultaneously updated PPO players.

#### E. Validation and checkpoint selection

| Argument | Type / accepted values | Default | Meaning and interactions |
| --- | --- | --- | --- |
| `--eval-opponents` | Nonempty, unique comma-separated exact registry names; no separator whitespace | `HHH,CHH,anchor` with an anchor, else `HHH,CHH` | Separate reports using the same fixed paired schedules. May include registered archives/compositions, but not the dynamic `self-current`/`self-snapshots` entries (rejected); register a concrete snapshot to evaluate it. `anchor` without an anchor is an error. |
| `--selection-opponent` | One name present in `--eval-opponents` | `anchor` with an anchor, else `HHH` | `best.pt` maximizes this opponent's mean signed marks/hand, strictly greater than the previous best; ties retain the earlier best. Does not automatically change when training opponents or validation names change. |
| `--eval-every-hands` | Integer > 0 | `5000` | Validation threshold since last evaluation, checked after collection/update. May overshoot by a rollout; also evaluates at total-hand completion, including an empty final batch. |
| `--eval-deals` | Integer > 0 | `128` | Duplicate-deal **groups per opponent** per validation. Two swapped games/group in hand mode, so 128 means 256 games against each opponent. Increasing this changes the selection experiment/cost. |
| `--bootstrap-samples` | Integer > 0 | `200` | Complete-group percentile-bootstrap replicates in validation, not PPO minibatches. CLI cannot disable validation bootstrap with zero. Evaluation CLIs have their own defaults. |
| `--checkpoint-every-hands` | Integer > 0 | `5000` | Save `latest.pt` at threshold and final completion. Intermediate progress since last save is not guaranteed recoverable after interruption. Independent of `best.pt` selection/publication. |

> **HHH-only training/selection** needs **all three** of `--opponent-pool HHH:1`,
> `--eval-opponents HHH` and `--selection-opponent HHH`; changing the training pool
> alone leaves anchor selection and the three default validation opponents intact.

Training deals collect stochastic learner actions and are never used as validation
trajectories. Validation is greedy on a fixed separately seeded schedule. The final
held-out comparison uses a new seed and is not fed back into checkpoint selection.
Win rate averages games; signed marks averages completed hands from the policy
partnership's perspective; 95% percentile bounds come from whole-group bootstrap
resampling. This is an uncertainty estimate under the experiment's opponents/deals,
not proof of optimal play; [arena definitions](arena.md) own the statistical
semantics.

#### F. Execution and diagnostics

| Argument | Type / accepted values | Default | Meaning and interactions |
| --- | --- | --- | --- |
| `--seed` | Integer `0 ≤ seed < 2^63` | `42` | Experiment seed for separate deal, action, shuffle, opponent, snapshot, imitation and validation streams. Record it with software versions; it is not a hardware-independent bitwise guarantee. |
| `--device` | `auto`, `cpu`, `cuda` | `auto` | Main optimizer device; auto chooses CUDA when available. Explicit unavailable CUDA fails. Rollout/evaluation neural inference runs on CPU even when optimization uses CUDA. |
| `--threads` | Integer ≥ 0 | `1` | Maximum simulation processes; 1 is sequential, 0 auto. Auto caps at 8; both auto/positive requests respect CPU affinity, physical-core information, quotas and independent-task count. Scheduling does not seed decisions. |
| `--eval-threads` | Integer ≥ 0 | `1` | Independent validation worker request using the same safe resolver. Runtime-only; changing it does not alter training samples or exact-resume configuration. Small jobs may be faster sequentially. |
| `--torch-threads` | Integer > 0 | `1` | Main-process PyTorch intra-op CPU threads, not simulations. Worker numerical threads are 1 to prevent oversubscription. Changing this stored setting is not accepted for exact resume. |
| `--output-dir` | Directory path | `runs/ppo-selfplay` | Config/progress/checkpoints/snapshots. New stages reject an existing `progress.jsonl`; resume can use the same directory or a new directory while preserving dependency paths. |
| `--log-dir` | Directory path | `logs/experiments/<command>-<run-id>/` | Readable `training.log` and `training.csv`. Training log appends; CSV is regenerated from available progress after successful completion. Reuse explicitly to keep a resumed transcript together. |
| `--forgetting-hands` | Integer > 0 | `8` | Size in anchor-generated hands of the fixed play-observation KL dataset. KL(anchor play ‖ current play) is diagnostic, not a penalty. Must remain positive even with heuristic learner play; then KL is unavailable. |
| `--bid-bc-hands` | Integer ≥ 0 | `0` | Optional HHH bidding imitation before PPO; zero disables. Requires trainable bidding. Four masked-cross-entropy epochs use separate data/shuffle streams and a temporary optimizer. Logs loss/accuracy, including forced and voluntary subsets. Not rerun on resume; an explicit value must still match the saved configuration. |

Worker detection and CPU/CUDA boundaries are explained in
[parallel execution](parallel_execution.md). Use the same numerical backend for
determinism checks. Exact cross-GPU, CUDA-build or PyTorch-version reproducibility is
not guaranteed. Increasing workers, epochs or CPU numerical threads is not a
guarantee of higher hands/second.

#### G. Critic, returns and oracle auxiliary

| Argument | Accepted values | Default | Meaning and interactions |
| --- | --- | --- | --- |
| `--critic-mode` | `local`, `centralized` | `local` | `local` reproduces the existing per-phase value heads exactly. `centralized` trains a separate critic on privileged features (all remaining hands); actors still see only their own observation. Fixed for exact resume. |
| `--return-estimator` | `gae`, `monte-carlo` | `gae` | `monte-carlo` uses complete-hand returns (GAE with λ = 1) as a diagnostic baseline. Uses `--gamma` (default 1). |
| `--oracle-aux-coef` | Finite float ≥ 0 | `0` (disabled) | Weight of the critic's oracle-point interval loss. A positive value requires `--critic-mode centralized` and `--oracle-dataset`. |
| `--oracle-dataset` | Path from `examples.oracle_labels` | Unset | Its SHA256 is stored in the configuration and verified on resume; only the `train` split is optimized and the `validation` split is reported. |

When a centralized critic starts from a checkpoint without compatible critic
weights (every play-only or local-critic file), it is initialized fresh from its own
seed stream and the startup output says so; `--init-checkpoint` from a centralized
checkpoint copies its critic.

### Training artifacts and metrics

| Artifact | Contents / use |
| --- | --- |
| `OUTPUT/config.json` | Resolved stored configuration; initial sources and identities for explicit phases, opponent definitions, optimization settings. Legacy configurations can leave `learner_phases` empty; metadata/startup output records their effective states. |
| `OUTPUT/progress.jsonl` | One row per completed collection/update cycle; cumulative counters, per-rollout samples/rewards, timing and validation reports. Not an experience dataset or a per-action log. |
| `OUTPUT/checkpoints/best.pt` | Model/optimizer/counters/RNGs at the best primary validation score. Use for final evaluation or a new stage; it may be older than `latest.pt`. |
| `OUTPUT/checkpoints/latest.pt` | Most recent periodically saved completed cycle. Preferred exact-resume point. Restores `best_score` and validation/publication thresholds too. |
| `OUTPUT/snapshots/policy-NNNNNNNNN.pt` | Immutable published frozen policies; filenames use actual cumulative hands. Active roster bounded, archived files retained. |
| `LOG/training.log` | Invocation, resolved policies/SHA256, progress, validation summaries and timings. Stdout is mirrored as it happens; errors on stderr are not necessarily in this file. |
| `LOG/training.csv` | Flattened per-progress-row metrics, phase samples/losses, opponent counts, primary/per-opponent validation and worker/time fields. Generated after a successful command return; interrupted runs may have progress without a refreshed CSV. |

**Reading progress fields.**

- `hands`, `training_samples`, `updates` and `optimizer_steps` are different
  cumulative counts.
- Modern `play_samples` is a compatibility alias for **all** training samples; actual
  playing counts are `phase_samples.play` per rollout.
- `mean_reward` averages all collected hand rewards, including zero-sample hands;
  `training_by_opponent` separates counts and means.
- `metrics` contains aggregate and `bid_*`/`trump_*`/`play_*` losses, entropy, KL and
  clipping statistics. A phase absent from a batch has no optimized phase metrics;
  console fallback zeros are not evidence it trained.
- `frozen_phases_verified` must remain true for every frozen neural phase;
  `frozen_play_verified` is its play compatibility field. A heuristic phase is not
  reported as a verified pretrained network.
- `playing_anchor_kl` can be nonzero for frozen joint play because its reference is
  the **original anchor**, not the stage's selected play source. No KL regularizer is
  implemented.
- Critic fields: `critic_mode`, `return_estimator`, `critic_initialization`;
  `metrics.value_loss` (main critic) and `critic_gradient_norm` (centralized only);
  `value_diagnostics` holds `explained_variance` and per-phase return, advantage
  (before normalization) and value statistics. With oracle supervision,
  `metrics.oracle_aux_loss` is the normalized interval loss and
  `metrics.oracle_aux_point_error` its distance in points; `oracle_validation`
  repeats both on the deal-disjoint validation split, and `oracle_dataset` reports
  the dataset size, exact-label fraction, mean bound width and labeling cost.
- Performance fields separate collection, optimization, validation, checkpoint and
  update time. Throughputs distinguish hands from learned decisions. Worker inference
  and loading durations are sums across workers, not additive wall-time stages; RSS
  samples at hand boundaries do not measure continuous peak memory. Startup,
  imitation and fixed-dataset construction precede the reported training-loop
  runtime.

### Troubleshooting PPO configuration

| Symptom | Cause / action |
| --- | --- |
| Initialization file missing | The modern default expects a pre-existing play checkpoint. Complete Recipe 1 or select a valid explicit initialization file. |
| Unavailable pretrained phase | Version 1 supplies play only; version 2 must mark a neural phase trained. Choose a trained file for frozen policies, or `init`/`new` representation transfer for a trainable phase. |
| Anchor bid/trump rejected | The original anchor is play-only. Use a trained joint/archive source or `new` for trainable bid/trump. |
| No trainable phase / conflicting flags | A source alone does not enable training. Spell all three learner states and remove inconsistent `--learn-phases`/`--play-policy`/`--trump-policy` assertions. |
| Architecture mismatch | All learner sources must match the initial hidden width and game/feature/action metadata. Omit `--hidden-size` to infer the initial width; a width override cannot reshape pretrained weights. |
| Frozen checkpoint identity changed | A recorded file was overwritten or replaced. Restore the exact SHA256 file; use immutable copies. Intentional replacement is a new stage, not exact resume. |
| Missing historical dependency | Restore the checkpoint at its recorded path, including inactive archives needed by older resumes. A roster bound does not make archive files disposable. |
| Resume configuration mismatch | Keep phase/source/optimizer/opponent/validation settings unchanged, restore omitted modern settings, and use the same device type. For changed experiments use `--init-checkpoint` and a new directory. |
| Bidding-only legacy anchor comparison rejected | Old bid-only syntax enforces anchor-identical play. Use explicit modern learner states for frozen joint play, and select the appropriate same-play opponent. |
| Trump samples remain zero | Opponents are declaring. Inspect bidding and per-phase counts; use the hand cap/total budget to bound work. Empty batches skip updates. Do not invent trump samples or infer learning from printed zero losses. |
| Frozen play has nonzero anchor KL | Its selected frozen source may be a joint policy different from the original anchor. Check frozen verification and actual tensor equality to the selected source. |
| No PPO optimizer steps | A very tight `--target-kl` can stop all minibatches. Review KL threshold, finite diagnostics and software/backend consistency; empty buffers are already skipped separately. |
| Output run already exists | Resume `latest.pt` in that directory, or choose a fresh directory for a new stage. Do not overwrite provenance to bypass the check. |
| CSV absent after interruption | Read `progress.jsonl` and the streamed log; CSV export occurs on successful completion. The latest checkpoint, not the final printed hand, controls recoverable state. |

## Evaluation

| Command | Use for |
| --- | --- |
| `evaluate_agent_vs_agent` | **Canonical** for new experiments: independently composed bid/trump/play on each side, explicit trained-phase checks, shared-parameter assertions. |
| `evaluate_ppo` | Legacy checkpoint versus built-in variants; preserves the legacy result schema. |
| `evaluate_selfplay_ppo` | Version-2 checkpoint versus built-ins, `anchor` and registered archives. |

### Agent-vs-agent composition

Use `examples.evaluate_agent_vs_agent` for new phase ablations. See
[composition methodology](phase_composable_evaluation.md) for checkpoint validation,
parameter equality and interpretation.

#### Hold joint playing fixed, change bidding

```bash
python -m examples.evaluate_agent_vs_agent \
  --agent-a-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --agent-a-bid neural --agent-a-trump heuristic --agent-a-play neural \
  --agent-b-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --agent-b-bid heuristic --agent-b-trump heuristic --agent-b-play neural \
  --deals 4096 --seed 31415926 --bootstrap-seed 0 \
  --bootstrap-samples 2000 --threads 4 \
  --output runs/ppo-joint-v1/bidding-ablation.json \
  --log-dir logs/experiments/joint-bidding-ablation
```

Change only `--agent-b-bid conservative` to compare conservative versus neural bidding.

#### Independent checkpoint sources

```bash
python -m examples.evaluate_agent_vs_agent \
  --agent-a-bid neural \
  --agent-a-bid-checkpoint runs/ppo-bidding-v1/checkpoints/best.pt \
  --agent-a-trump heuristic --agent-a-play neural \
  --agent-a-play-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --agent-b-bid conservative --agent-b-trump random --agent-b-play neural \
  --agent-b-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --deals 4096 --seed 42 --threads 4
```

For joint PPO versus the original play-only PPO, use neural bid/play from the joint
checkpoint for A and heuristic bid/trump plus neural play from the original for B:

```bash
python -m examples.evaluate_agent_vs_agent \
  --agent-a-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --agent-a-bid neural --agent-a-play neural \
  --agent-b-checkpoint runs/ppo-play-v1/checkpoints/best.pt --agent-b-play neural \
  --deals 4096 --seed 31415926 --threads 4
```

Use immutable snapshot paths in either agent's checkpoint options to compare archived
policies. Each requested neural phase must have been trained.

#### Pure heuristic, conservative and random comparisons

```bash
# HHH against HHH: deterministic paired symmetry check
python -m examples.evaluate_agent_vs_agent --deals 64 --seed 42
# HHH against conservative bidding with heuristic trump/play
python -m examples.evaluate_agent_vs_agent --agent-b-bid conservative --deals 64
# HHH against heuristic bidding with random trump/play
python -m examples.evaluate_agent_vs_agent \
  --agent-b-trump random --agent-b-play random --deals 64
```

#### Recipe 12: held-out phase ablations

Prerequisites: the trained checkpoints in each comparison, including all requested
trained neural phases. Use final deal seeds independent of the fixed
training-validation stream, and keep evaluation results out of optimizer/checkpoint
selection. See [interpretation](phase_composable_evaluation.md#controlled-training-comparisons).

| Phase under test | Recipe |
| --- | --- |
| Bidding | [Identical joint play, heuristic versus neural bidding](#hold-joint-playing-fixed-change-bidding); trump heuristic on both sides. |
| Trump | [Trump-only checkpoint versus its frozen bid/play source](#held-out-trump-ablation). |
| Playing | Identical neural bid/trump from an all-phase policy, changing only play (below). |

```bash
python -m examples.evaluate_agent_vs_agent \
  --agent-a-checkpoint runs/ppo-allphase-selfplay/checkpoints/best.pt \
  --agent-a-bid neural --agent-a-trump neural --agent-a-play neural \
  --agent-b-checkpoint runs/ppo-allphase-selfplay/checkpoints/best.pt \
  --agent-b-bid neural --agent-b-trump neural --agent-b-play heuristic \
  --deals 4096 --seed 27182818 --bootstrap-seed 0 --bootstrap-samples 2000 --threads 4 \
  --output runs/ppo-allphase-selfplay/playing-ablation.json \
  --log-dir logs/experiments/allphase-playing-ablation
```

Require shared bidding/trump parameters YES and only play differing. To compare two
learned play policies, replace B's playing strategy by neural and add its
play-checkpoint override; keep bid/trump sources unchanged. A complete joint model
versus HHH comparison measures several interacting phase differences instead.

#### Held-out trump ablation

Requires [Recipe 4](#recipe-4-trump-only)'s trained checkpoint and the exact joint
bid/play source used there. If different phase sources were chosen, supply those same
sources for B.

```bash
python -m examples.evaluate_agent_vs_agent \
  --agent-a-checkpoint runs/ppo-trump-v1/checkpoints/best.pt \
  --agent-a-bid neural --agent-a-trump neural --agent-a-play neural \
  --agent-b-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --agent-b-bid neural --agent-b-trump heuristic --agent-b-play neural \
  --deals 4096 --seed 31415926 --bootstrap-seed 0 --bootstrap-samples 2000 --threads 4 \
  --output runs/ppo-trump-v1/held-out-trump.json --log-dir logs/experiments/held-out-trump
```

Require shared bidding and playing parameters YES and only trump differing.

#### evaluate_agent_vs_agent arguments

This is the complete argument surface. Evaluation is CPU inference; there is no
optimizer/device or trainability flag here.

| Argument | Type / accepted values | Default | Meaning |
| --- | --- | --- | --- |
| `--agent-a-bid`, `--agent-b-bid` | neural, heuristic, random, conservative | heuristic | Bidding strategy; conservative passes when legal, otherwise minimum bid. |
| `--agent-a-trump`, `--agent-b-trump` | neural, heuristic, random | heuristic | Independent trump strategy. |
| `--agent-a-play`, `--agent-b-play` | neural, heuristic, random | heuristic | Independent playing strategy. |
| `--agent-a-checkpoint`, `--agent-b-checkpoint` | Compatible checkpoint path | Unset | Default neural source for that agent; ignored when no phase needs it. Version 1 supplies play only. |
| `--agent-{a,b}-bid-checkpoint` | Trained bid source path | Unset | Overrides only that agent's neural bidding source. |
| `--agent-{a,b}-trump-checkpoint` | Trained trump source path | Unset | Overrides only that agent's neural trump source. |
| `--agent-{a,b}-play-checkpoint` | Trained play source path | Unset | Overrides only that agent's neural playing source. |
| `--stochastic-neural` | Flag | Off (greedy) | Seeded phase-local sampling, independent between seats/phases; still frozen inference. |
| `--deals` | Integer > 0 | `2048` | Complete duplicate groups; two games/group. |
| `--seed` | Integer | `42` | Arena deal/agent seed; choose independent final seeds. |
| `--bootstrap-seed` | Integer | `0` | Separate resampling seed. |
| `--bootstrap-samples` | Integer > 0 | `2000` | Paired group bootstrap replicates; intervals unavailable for fewer than two groups. |
| `--threads` | Integer ≥ 0 | `1` | Up to N simulation workers; 0 auto with CPU/task limits. |
| `--episode` | hand, match | hand | Paired hands or complete paired match schedules. Match metrics use actual played-hand counts. |
| `--target-marks` | Integer > 0 | `7` | Match termination target; does not turn hand mode into a match. |
| `--output` | Path | Unset | Full JSON arena report with composition identities and diagnostics. |
| `--log-dir` | Directory path | Auto under `logs/experiments` | Readable log, hand CSV and metric CSV; separate from JSON path. |

A neural phase needs its phase override or agent default file; a phase checkpoint on
a non-neural strategy is rejected. Different neural files are allowed per phase,
including archived snapshots. Each selected network must satisfy metadata and
trained-phase checks. Actual network tensors, not just paths/hashes, determine
shared-parameter assertions. See [composition semantics](phase_composable_evaluation.md).

### Compatibility evaluation

```bash
# Legacy primary test: unchanged heuristic bidding/trump; PPO versus HHH play
python -m examples.evaluate_ppo \
  --checkpoint runs/ppo-parallel-v1/checkpoints/best.pt \
  --opponent HHH --deals 4096 --seed 24681357 \
  --threads 4 --bootstrap-samples 2000 \
  --output runs/ppo-best-vs-hhh.json
# Different bidding baseline
python -m examples.evaluate_ppo \
  --checkpoint runs/ppo-parallel-v1/checkpoints/best.pt \
  --opponent CHH --deals 4096 --seed 24681357 --threads 4
# Random trick-play opponent
python -m examples.evaluate_ppo \
  --checkpoint runs/ppo-parallel-v1/checkpoints/best.pt \
  --opponent HHR --deals 4096 --seed 24681357 --threads 4

# Version-2 checkpoint against built-ins and the original anchor
python -m examples.evaluate_selfplay_ppo \
  --checkpoint runs/ppo-bidding-v1/checkpoints/best.pt \
  --opponents HHH,CHH,HHR,anchor --deals 4096 --seed 12345 \
  --threads 4 --output runs/ppo-bidding-v1/heldout.json
```

For version-1 checkpoints the A side is PPO play plus heuristic auction/trump;
version-2 sources retain their configured neural phases. Against HHH, bidding and
trump policies are identical and the contrast isolates the two trick-playing
policies; against CHH, the opponent's bidding also changes. Use new held-out seeds
for final testing; repeated selection on one test seed makes it part of validation.

#### Compatibility evaluation arguments

Both commands share `--deals 2048`, `--seed 12345`, `--bootstrap-seed 0`,
`--bootstrap-samples 2000`, `--threads 1`, optional `--output` and optional
`--log-dir`, with the same ranges as above. They evaluate hands only and do not
expose independent phase strategies or stochastic inference.

| Command | Argument | Accepted values / default | Meaning |
| --- | --- | --- | --- |
| `evaluate_ppo` | `--checkpoint` | Required version-1 or version-2 path | Auto-route the model's neural phases; other phases heuristic. |
| `evaluate_ppo` | `--opponent` | One built-in variant; HHH if both selectors omitted | Mutually exclusive with `--opponents`. |
| `evaluate_ppo` | `--opponents` | Unique comma-separated built-in variants | Separate reports; no archive or composition registration. |
| `evaluate_selfplay_ppo` | `--checkpoint` | Required version-2 path | Rejects version 1; use `evaluate_ppo` or composable evaluation for play-only files. |
| `evaluate_selfplay_ppo` | `--anchor-checkpoint` | Original anchor path; inferred from checkpoint config | Override must match original anchor SHA256. |
| `evaluate_selfplay_ppo` | `--opponents` | Unique registered names; default `HHH,CHH,HHR,anchor`, or `HHH,CHH,HHR` for an anchor-free checkpoint | Built-ins, anchor and explicitly registered archives. Requesting `anchor` from an anchor-free checkpoint fails. |
| `evaluate_selfplay_ppo` | `--opponent-checkpoint` | Repeatable `NAME=PATH` | Registers whole frozen checkpoint opponents. No opponent-composition option here. |

These compatibility learner routes do not add the canonical per-selected-phase
trained check: use files whose routed neural phases are already trained. For explicit
availability validation or any independent phase ablation, use
`evaluate_agent_vs_agent`. Existing single/multi-opponent JSON envelopes remain
unchanged; reports are never merged into an unlabeled multi-opponent score.

## Performance benchmarks

These measure throughput and correctness, not policy strength. Note the list syntax:
`parallel_benchmark --workers` is comma-separated; the others take space-separated
lists.

```bash
# Arena and PPO worker scaling
python -m examples.parallel_benchmark \
  --workers 1,2,4,8,0 --deals 2048 \
  --ppo-hands-per-update 256 --ppo-updates 3 \
  --device cpu --output runs/parallel-profile.json

# Original-vs-optimized group bootstrap
python -m examples.bootstrap_benchmark \
  --groups 128 1024 4096 16384 \
  --output runs/bootstrap-profile.json

# Full arena before/after statistical optimization
python -m examples.arena_statistics_benchmark \
  --deals 4096 --threads 8 \
  --output runs/arena-statistics-profile.json

# PPO checkpoint evaluation scaling
python -m examples.validation_benchmark \
  --checkpoint runs/ppo-parallel-v1/checkpoints/best.pt \
  --groups 128 256 1024 4096 \
  --workers 1 2 4 8 --opponents HHH CHH \
  --output runs/validation-profile.json

# Legacy fixed/mixed-opponent collection throughput
python -m examples.opponent_benchmark --workers 1 4 \
  --hands-per-update 256 --updates 3 --output runs/opponent-performance.json

# Fast pure-engine smoke test
python -m examples.arena_benchmark --deals 1024 --threads 4
```

| Command | Principal arguments and defaults | Extra |
| --- | --- | --- |
| `parallel_benchmark` | `--workers 1,2,4,8,0` (comma-separated); `--deals 2048`; `--ppo-hands-per-update 256`; `--ppo-updates 3`; `--seed 42`; `--device cpu`; `--arena-only`; `--output` | `rl` unless `--arena-only` |
| `bootstrap_benchmark` | `--groups 128 1024 4096 16384`; `--output`. Uses 200 samples at 128 groups, 2000 otherwise. | — |
| `arena_statistics_benchmark` | `--deals 4096`; `--threads 8`; `--output`. Legacy vs optimized arena bootstrap on complete evaluations. | — |
| `validation_benchmark` | `--checkpoint` (required); `--groups 128 256 1024 4096`; `--workers 1 2 4 8`; `--opponents HHH CHH`; `--legacy-small-check`; `--output` | `rl` |
| `opponent_benchmark` | `--workers 1 4`; `--hands-per-update 256`; `--updates 3` (after one warm-up); `--hidden-size 256`; `--output` | `rl` |
| `multiphase_benchmark` | `--checkpoint runs/ppo-play-v1/checkpoints/best.pt`; `--hands 64`; `--workers 1 4`; `--device cpu`; `--snapshot-dir runs/multiphase-benchmark-snapshots`; `--output` | `rl` |
| `arena_benchmark` | `--deals 30`; `--seed 42`; `--bootstrap-seed 123`; `--threads 1`. Random-versus-random arena. | — |

All benchmarks also accept `--log-dir`. For 128/256-group PPO validation, one worker
was fastest in the branch benchmark; for 4096 groups, four workers was fastest in that
benchmark. Test on your own machine. See [parallel execution](parallel_execution.md)
and [evaluation performance](evaluation_performance.md).

## Reproducibility checklist

**Recommended experiment series.**

1. Run one baseline/ablation schedule.
2. Train Stage-1 PPO with a fixed training seed and preserve periodic checkpoints.
3. Use the fixed 128-group validation schedule only for model selection.
4. Evaluate best and latest models against HHH, CHH and HHR on separate 4096-group
   held-out seeds; compare different checkpoints on the same held-out deals.
5. Retain JSON, CSV, checkpoint hashes and Git commit IDs.

**Common pitfalls.**

- Use the same `--seed`, policies, rules and `--deals` to compare runs with different
  worker counts; group ordering is deterministic.
- A 95% confidence interval refers to resampling duplicate-deal groups. It is not a
  guarantee of generalization to unseen opponents.
- Fixed 128-group validation is inexpensive but noisy; evaluate selected checkpoints
  on large independent held-out schedules.
- `--threads 0` means safe automatic worker selection, not unlimited threads.
- PPO optimization remains in the parent process; more rollout workers cannot
  eliminate optimizer, serialization, evaluation or checkpoint overhead.
- Changing sampling from legacy-v1 to episode-v1 alters the training trajectory.
  Prefer `--warm-start` for cross-scheme experiments.
- `--resume` budgets are cumulative; a changed experiment is a new stage, not a
  resume.
