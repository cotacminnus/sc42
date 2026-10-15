# SC42 COMMAND REFERENCE

*Texas 42 research workflows | CLI recipes, parameters, checkpoints and outputs*

Canonical commands for the current checkout. Method and compatibility details live in the linked topic guides.

> **Note:** All commands below are run from the repository root. Examples use bash/WSL2 line continuation (\). Replace checkpoint and output paths with your own.

## 1. Setup and fast lookup

```bash
# Core package (no ML required)
python -m pip install -e .
# PPO training and checkpoint evaluation
python -m pip install -e '.[rl]'
# Optional OpenSpiel adapter
python -m pip install -e '.[openspiel]'
# Verify the checkout
python -m unittest discover -s tests -v
```

| Goal | Entry point | Best use |
| --- | --- | --- |
| Human/AI console game | python -m texas42 | Interactive play; optional JSONL self-play logs |
| Baseline comparison | python -m examples.policy_ablation_benchmark | HHH, CHH, HHR, etc. in paired duplicate deals |
| Bid/pass experiment | python -m examples.counterfactual_bidding | Counterfactual voluntary bid intervention |
| Open-hand endgame analysis | python -m examples.endgame_oracle | Exact ordinary-contract minimax with search budgets |
| Train legacy play-only PPO | python -m examples.train_ppo | Reproducible baseline; fixed or mixed heuristic variants |
| Legacy checkpoint evaluation | python -m examples.evaluate_ppo | Held-out, paired checkpoint evaluation |
| Current PPO training | python -m examples.train_selfplay_ppo | Independent phase networks and historical opponents |
| Agent-vs-agent evaluation | python -m examples.evaluate_agent_vs_agent | Canonical new phase-ablation experiments |
| Multi-opponent phase evaluation | python -m examples.evaluate_selfplay_ppo | Compatibility command for anchors and archived policies |
| Scaling benchmark | python -m examples.parallel_benchmark | Worker-count throughput and correctness |
| Bootstrap benchmark | python -m examples.bootstrap_benchmark | Old vs. optimized confidence interval implementation |
| Validation benchmark | python -m examples.validation_benchmark | Evaluation scaling by group and worker count |

## 2. Copy-ready commands

### Perfect-information endgame oracle

```bash
python -m examples.endgame_oracle --seed 42 --remaining 3 --max-nodes 100000
python -m examples.endgame_oracle --seed 7 --remaining 2 --partial-plays 3
# Prove the position and one action; other actions may remain unresolved
python -m examples.endgame_oracle --seed 1 --remaining 2 --partial-plays 1 \
  --output position_value --max-nodes 4
# Bounded smoke benchmark, consecutive seeds starting at zero
python -m examples.endgame_oracle --seed 0 --remaining 3 --samples 100 \
  --max-nodes 100000 --time-seconds 2
```

No optional dependencies are required. The default HHR policies generate a
reachable ordinary contract with heuristic bidding/trump and seeded random play.
The command prints the contract, scores, trump, actor/team, four hands, partial
trick, legal global IDs/dominoes and exact values or bounds, proven optimal
actions, completion flags, nodes, cache hits and runtime. With multiple samples it
prints completion rates and mean/median/max statistics instead. See the
[oracle guide](endgame_oracle.md) for fixed bidding-team value and proof semantics.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--seed` | 42 | Integer deal/policy seed; samples use consecutive seeds. |
| `--remaining` | 3 | 1–7 dominoes per player at the selected trick boundary. Larger positions are permitted. |
| `--partial-plays` | 0 | Advance 0–3 additional legal moves after that boundary. |
| `--policy` | HHR | Existing built-in variant: HHH, HHR, HRR, CHH, CRR, RRR. |
| `--output` | all_action_values | All legal root action values, or position_value for root outcome plus one proven optimal action. |
| `--max-nodes` | Unset | Nonnegative per-position search-node budget. |
| `--time-seconds` | Unset | Nonnegative finite per-position soft deadline, including validation. |
| `--max-cache-entries` | 100000 | Nonnegative table capacity; zero disables caching. |
| `--samples` | 1 | Positive position count; more than one selects benchmark summary output. |
| `-h`, `--help` | Off | Print argument help. |

### 2.1 Console game, self-play log and engine smoke test

```bash
python -m texas42 --seed 42 --target-marks 7
# Human in seat 1, with independently selected policy bots
python -m texas42 --players human HHH CHH HHR --seed 42
# Opposing HHH/CHH partnerships, with a replayable match log
python -m texas42 --all-ai --players HHH CHH HHH CHH \
  --seed 42 --target-marks 1 --self-play-log logs/hhh-vs-chh.jsonl
python -m texas42 --all-ai --seed 42 --target-marks 1 \
  --self-play-log logs/selfplay.jsonl
python -m examples.agent_selfplay
python -m examples.openspiel_selfplay --episode hand --seed 42
```

The OpenSpiel example requires the optional OpenSpiel/pyspiel installation. The standalone agent_selfplay example has no command-line arguments. Console --self-play-log requires --all-ai and writes JSONL; it is not neural-network training.

Console `--players P1 P2 P3 P4` selects each seat independently. Choices are
`human`, `placeholder` (the original always-pass random bot), `neural`, and the six variants
in the table below. Seats 1/3 partner against seats 2/4. Variant bots reuse the
existing `HybridAgent` through `AgentPlayer`; each seat has its own agent and random
streams, reset once per match. Non-neural players require no ML dependencies. `RRR` bids uniformly
among legal actions, unlike `placeholder`. Multiple `human` seats share the same
terminal, so avoid viewing another player's private prompts when playing locally.

Without `--players`, the original defaults remain: human in seat 1 and three
placeholder bots, or four placeholders with `--all-ai`. Explicit player selections
take precedence over these defaults; `--all-ai` rejects any human selection. Four
bot selections also run without `--all-ai`, but logging still requires that flag.

#### Neural console players and phase ablations

Install the optional `rl` extra for neural players. `neural` defaults to greedy
neural bidding and play, with heuristic trump. The checkpoint must supply trained
neural bid and play networks. For an original play-only checkpoint, explicitly
select `--player-N-bid heuristic`; unavailable neural phases are never replaced silently.
For seat `N` (1–4), `--player-N-bid`, `--player-N-trump` and `--player-N-play` override
the preset (`neural` or a named variant). Bidding accepts `neural`, `heuristic`,
`random`, `conservative`; trump/play accept the first three. Phase options are
rejected on `human` and `placeholder` players.

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
`--player-N-checkpoint`. A checkpoint override requires a neural strategy.
Version-1 checkpoints support play only; version-2 checkpoints must record the
requested phase as trained and neural. Console output identifies the selected
phases, paths and SHA256 values. Frozen CPU weights are cached and shared across
seats; private agent state and phase RNG streams remain independent. Gameplay
uses the same `CompositeAgentFactory` as phase-composable evaluation, with one
PyTorch inference thread and no training or parameter updates.

The console runs ordinary matches; use the [agent-vs-agent evaluation commands](#agent-vs-agent-phase-composition)
for duplicate deals, actual parameter-equality checks and confidence intervals.

### 2.2 Policy baselines and ablations

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

| Variant | Bidding | Trump | Trick play |
| --- | --- | --- | --- |
| HHH | Heuristic | Heuristic | Heuristic |
| HHR | Heuristic | Heuristic | Random |
| HRR | Heuristic | Random | Random |
| CHH | Conservative | Heuristic | Heuristic |
| CRR | Conservative | Random | Random |
| RRR | Random | Random | Random |

> **Note:** A deal group contains two games with swapped A/B partnerships. Thus --deals 4096 normally yields 8192 hand games. In --episode match mode the units are paired match schedules, not individual hands.

### 2.3 Counterfactual bidding

```bash
python -m examples.counterfactual_bidding \
  --policy-a HHH --policy-b CHH --target-policy HHH \
  --deals 4096 --seed 42 --threads 4 \
  --bootstrap-samples 2000 \
  --output runs/counterfactual-bidding.json
```

The experiment identifies the first voluntary bid chosen by the target policy in each original hand, and replays BID versus PASS from the same deal. It supports hand episodes only. The result is conditional on the actions the policy actually bids; it does not evaluate every legal bid or pass opportunity.

### 2.4 Start legacy play-only PPO from scratch (parallel)

**Recipe 1 — original play-only PPO.** No pretrained checkpoint is needed. This
creates the version-1 original play model with heuristic learner bid/trump and HHH
opponents. The explicit episode-v1 sampling scheme makes the intended parallel
stream clear; an archived legacy-v1 reproduction must instead retain its original
sequential scheme and all original optimizer/software settings.

```bash
python -m examples.train_ppo \
  --total-hands 524288 --rollout-play-steps 4096 \
  --ppo-epochs 4 --minibatch-size 512 \
  --learning-rate 0.0003 --sampling-scheme episode-v1 --seed 42 --device auto \
  --threads 8 --eval-threads 1 --torch-threads 1 \
  --output-dir runs/ppo-parallel-v1
```

Stage 1 learns trick play only: bidding and trump stay heuristic, two network seats share weights, and PPO optimizes after collecting full-hand rollouts. All rollout workers use frozen policy weights; the optimizer updates only in the parent process.

### 2.5 Resume a run vs. warm-start a new experiment

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

> **Note:** IMPORTANT: --resume restores model, optimizer, counters and RNGs; --total-hands is the FINAL cumulative budget. --warm-start copies model weights only and starts fresh counters, optimizer and RNGs. The two flags are mutually exclusive. Exact resume cannot change sampling scheme or underlying optimizer device type. Keep all training hyperparameters unchanged except the explicitly supported execution/budget settings.

The original single-process PPO checkpoints use legacy-v1 sampling. The parallel trainer uses episode-v1 by default whenever --threads is not 1. To switch legacy weights to parallel rollout collection, use --warm-start, not --resume. Parallel episode-v1 checkpoints can vary worker counts while retaining episode-seeded sampling; use --sampling-scheme episode-v1 explicitly if requesting --threads 1.

### 2.6 Legacy checkpoint evaluation

```bash
# Primary test: unchanged heuristic bidding/trump; PPO versus HHH play
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
```

For version-1 checkpoints the A-side is PPO play plus heuristic auction/trump. Version-2 sources retain their configured neural phases. For HHH comparison, the bidding and trump policies are identical and the contrast isolates the two trick-playing policies. Use new held-out seeds for final testing; repeated selection on one test seed makes it part of validation.

### 2.7 Performance and statistical benchmarks

```bash
# Arena and PPO worker scaling. Comma-separated --workers.
python -m examples.parallel_benchmark \
  --workers 1,2,4,8,0 --deals 2048 \
  --ppo-hands-per-update 256 --ppo-updates 3 \
  --device cpu --output runs/parallel-profile.json

# Original-vs-optimized group bootstrap (space-separated --groups)
python -m examples.bootstrap_benchmark \
  --groups 128 1024 4096 16384 \
  --output runs/bootstrap-profile.json

# Full arena before/after statistical optimization
python -m examples.arena_statistics_benchmark \
  --deals 4096 --threads 8 \
  --output runs/arena-statistics-profile.json

# PPO checkpoint evaluation scaling; space-separated lists
python -m examples.validation_benchmark \
  --checkpoint runs/ppo-parallel-v1/checkpoints/best.pt \
  --groups 128 256 1024 4096 \
  --workers 1 2 4 8 --opponents HHH CHH \
  --output runs/validation-profile.json
```

For a faster pure-engine baseline, use python -m examples.arena_benchmark --deals 1024 --threads 4. For OpenSpiel integration, use the openspiel_selfplay example and optional dependency.

## 3. Argument reference

### 3.1 Shared arena / evaluation flags

| Argument | Default | Meaning / scope |
| --- | --- | --- |
| --deals | Varies (128–4096 examples) | Number of paired duplicate groups, not number of games. |
| --seed | 42 baseline; 12345 PPO evaluation | Deterministic deal/episode seed; choose held-out values for tests. |
| --threads | 1 | Maximum simulation processes; 0=auto, capped at 8 and CPU limits; 1=sequential. |
| --bootstrap-seed | 0 (arena/PPO); 123 arena_benchmark | Independent bootstrap RNG seed. |
| --bootstrap-samples | 2000 (most evaluation CLIs) | Number of complete-group bootstrap resamples. |
| --episode | hand | hand or match for baseline arena and phase-composable commands. |
| --target-marks | 7 | Marks required to win matches; counterfactual supports hand only. |
| --output | Unset | Optional JSON export or benchmark profile path. |
| --log-dir | Auto subdirectory | Readable .log / .csv experiment sidecars; does not replace JSON --output. |

### 3.2 Policy / special evaluation flags

| Command | Arguments | Defaults and constraints |
| --- | --- | --- |
| policy_ablation_benchmark | --policy-a, --policy-b | HHH and CRR; values: HHH/HHR/HRR/CHH/CRR/RRR. |
| heuristic_benchmark | --opponent | random (or heuristic). |
| counterfactual_bidding | --policy-a, --policy-b, --target-policy | HHH / CHH / HHH; target must be one of the two policy names. |
| evaluate_ppo | --checkpoint, --opponent / --opponents | Checkpoint required; default HHH; single variant or comma-separated variants, mutually exclusive. |
| arena_benchmark | --deals, --seed, --bootstrap-seed | 30 / 42 / 123; simple random-versus-random arena smoke test. |
| openspiel_selfplay | --episode, --target-marks, --seed | hand / 7 / 42; optional pyspiel required. |
| texas42 console | --debug, --seed, --target-marks, --ruleset-id | --ruleset-id currently cadets_base_v1; --debug reveals hands. |
| texas42 console | --all-ai, --self-play-log [PATH] | AI-only game and optional JSONL; --self-play-log requires --all-ai. |

### 3.3 PPO training configuration flags

This table is for **legacy `examples.train_ppo`**. The complete modern argument
reference and its different defaults are [below](#multi-phase-arguments).

| Argument | Default | Purpose |
| --- | --- | --- |
| --total-hands | 100000 | Cumulative training hands budget. |
| --rollout-play-steps | 4096 | Minimum learned play samples collected before each PPO update; complete hands. |
| --ppo-epochs | 4 | Optimization passes over each rollout. |
| --minibatch-size | 512 | PPO optimizer minibatch size. |
| --learning-rate | 0.0003 | Adam learning rate. |
| --gamma | 1.0 | Discount factor per controlled-seat decision. |
| --gae-lambda | 0.95 | Generalized advantage estimation parameter. |
| --clip-epsilon | 0.2 | PPO clipped surrogate ratio limit. |
| --entropy-coef | 0.01 | Policy entropy regularization weight. |
| --value-coef | 0.5 | Value loss weight. |
| --max-grad-norm | 0.5 | Maximum norm for gradient clipping. |
| --hidden-size | 256 | Width of both MLP hidden layers. |
| --seed | 42 | Training/deal seed. |
| --device | auto | CPU or CUDA optimizer device; auto selects CUDA when available. |
| --eval-every-hands | 5000 | Periodic validation interval; may overshoot by one rollout. |
| --eval-deals | 128 | Duplicate groups per periodic validation. |
| --checkpoint-every-hands | 5000 | Periodic latest.pt saving interval. |
| --bootstrap-samples | 200 | Bootstrap replicates per periodic validation. |
| --target-kl | 0.0 | Optional KL early-stop threshold; 0 disables. |
| --threads | 1 | PPO simulation worker request (0=auto). |
| --torch-threads | 1 | Main-process PyTorch CPU intra-op threads; distinct from workers. |
| --train-opponent / --train-opponents | HHH | One variant or NAME:WEIGHT mixture; mutually exclusive. |
| --eval-opponent / --eval-opponents | HHH | One or comma-separated variants; mutually exclusive. |
| --selection-opponent | HHH when present | Primary validation opponent for best.pt; must belong to evaluation set. |
| --sampling-scheme | auto | auto, legacy-v1 or episode-v1; governs action sampling streams. |
| --eval-threads | 1 | Validation workers; runtime-only, not in checkpoint training Config. |
| --output-dir | runs/ppo-play-v1 | Training config, progress and checkpoints. |
| --resume | Unset | Exact checkpoint continuation; mutually exclusive with warm-start. |
| --warm-start | Unset | Initialize a new run from model weights only. |
| --log-dir | Auto subdirectory | Readable training.log and training.csv exports. |

> **Note:** At Stage 1, each completed hand contributes 14 neural-controlled play decisions. A target of 4096 samples therefore collects 293 hands (4102 play records) in a standard full rollout. Training episodes contain 33 game decisions and seven tricks.

### 3.4 Benchmark-specific arguments

| Command | Principal arguments and defaults |
| --- | --- |
| parallel_benchmark | --workers 1,2,4,8,0 (comma-separated); --deals 2048; --ppo-hands-per-update 256; --ppo-updates 3; --seed 42; --device cpu; --arena-only; --output. |
| bootstrap_benchmark | --groups 128 1024 4096 16384 (space-separated); --output. Uses 200 samples at 128 groups, 2000 otherwise. |
| validation_benchmark | --checkpoint REQUIRED; --groups 128 256 1024 4096; --workers 1 2 4 8; --opponents HHH CHH; --legacy-small-check; --output. |
| arena_statistics_benchmark | --deals 4096; --threads 8; --output. Measures legacy vs optimized arena bootstrap on complete evaluations. |
| arena_benchmark | --deals 30; --seed 42; --bootstrap-seed 123; --threads 1. |

## 4. Outputs, reproducibility and common pitfalls

| Location or field | What it contains |
| --- | --- |
| runs/<run>/config.json | Resolved PPO training configuration (including total_hands, worker request). |
| runs/<run>/progress.jsonl | Per-update metrics and full periodic validation reports. |
| runs/<run>/checkpoints/latest.pt | Most recently saved complete training state; use for resume. |
| runs/<run>/checkpoints/best.pt | Checkpoint with best fixed-schedule validation mean signed marks. |
| logs/experiments/<command>-<run-id>/ | Auto-created readable .log and .csv files for experiment commands. |
| --output <path>.json | Optional detailed JSON report; not a substitute for log/CSV directory. |

- Use the same --seed, policies, rules and --deals to compare arena execution with different worker counts; group ordering is deterministic.
- A 95% confidence interval refers to resampling duplicate-deal groups. It is not a guarantee of generalization to unseen opponents.
- Fixed 128-group validation is inexpensive but noisy; evaluate selected checkpoints on large independent held-out schedules (e.g., 4096 groups).
- PPO versus HHH holds heuristic bidding and trump fixed. PPO versus CHH also changes the bidding behavior of the opponent.
- Use --threads 0 for safe automatic worker selection, not unlimited threads. The cap is eight; available physical cores, CPU affinity and quotas may lower it.
- For 128/256-group PPO validation, one worker was fastest in the branch benchmark; for 4096 groups, four workers was fastest in that benchmark. Test on your own machine.
- PPO optimization remains in the parent process; increasing rollout workers cannot eliminate optimizer, serialization, evaluation or checkpoint overhead.
- Changing episode sampling from legacy-v1 to episode-v1 alters training trajectory. Prefer warm-start for cross-scheme experiments.
- To inspect any installed CLI directly, append --help, e.g., python -m examples.train_ppo --help.

## 5. Research workflow recommendation

For a reproducible series of experiments: (1) run one baseline/ablation schedule; (2) train Stage-1 PPO with a fixed training seed; (3) preserve periodic checkpoints; (4) use the fixed 128-group schedule only for model selection; (5) evaluate best and latest models against HHH, CHH and HHR on separate 4096-group test seeds; (6) compare different checkpoints on the same held-out deals; (7) retain JSON, CSV, checkpoint hashes and Git commit IDs.

## Multi-phase PPO commands

Use `examples.train_selfplay_ppo` for new experiments with independently configured
phases, neural opponents, or historical self-play. Use `examples.train_ppo` to
reproduce the version-1 play-only baseline. Both train complete **hand** episodes;
there is no training `--episode` or match-PPO option. Install the `rl` extra first.
The following examples specify research configurations, not proven optimal settings.

### Workflow finder

| Recipe | Purpose | Prerequisite |
| --- | --- | --- |
| [1: original play-only](#24-start-legacy-play-only-ppo-from-scratch-parallel) | Reproduce `train_ppo`; creates the original play checkpoint | No pretrained checkpoint |
| [2: bidding-only](#stage-a-frozen-play-learned-bidding) | Learn bidding, preserve pretrained play | Compatible version-1 play checkpoint |
| [3: modern playing-only](#playing-only-with-the-modern-trainer) | Learn play with heuristic bid/trump using the version-2 path | Compatible version-1 play checkpoint |
| [4: trump-only](#independent-learner-phases-and-trump-only-training) | Learn trump, preserve joint bid/play | Version-2 checkpoint with trained bid/play |
| [5: all phases versus HHH](#all-three-phases-against-hhh) | Learn bid/trump/play against one fixed heuristic | Compatible version-1 play checkpoint |
| [6: historical self-play](#stage-b-joint-training-with-historical-opponents) | New stage using frozen snapshots and fixed baselines | Recipe 5 checkpoint with all routed phases trained |
| [7: exact resume](#resume-and-multi-opponent-validation) | Restore optimizer, progress, RNGs and historical roster | Existing version-2 checkpoint and its dependencies |
| [8: mixed trainability](#mixed-phase-training) | Freeze any neural continuation or use a heuristic | Trained source for every frozen phase |
| [9: independent sources](#independent-phase-checkpoints) | Load each learner phase from a different file | Compatible trained phase checkpoints, same hidden width |
| [10: composed opponents](#register-composed-training-opponents) | Independent frozen opponent phases | JSON definition and compatible phase checkpoints |
| [11: archived opponent](#training-against-an-archived-checkpoint) | Train against one fixed historical policy | Existing immutable trained snapshot/checkpoint |
| [12: controlled evaluation](#held-out-phase-ablations) | Compare one phase while holding continuations constant | Trained checkpoint(s), independent final deal seed |

Before running a recipe, confirm the named input files exist and satisfy its
trained-phase prerequisites. Output directories must be new unless resuming.
An output path in an example is not evidence that the checkpoint already exists.
Recipe 1 can produce `runs/ppo-play-v1/checkpoints/best.pt`; later recipes use that
file as a prerequisite. Archive filenames must be replaced with actual retained
snapshot paths: publication intervals do not guarantee an exact hand-number suffix.

### Multi-phase arguments

All arguments below belong to **`examples.train_selfplay_ppo`**. `-h`/`--help`
prints available flags. Defaults apply to a new run; omitted settings on resume
normally come from the saved configuration. Table values are checked against
argparse, `Config`, and `SelfPlayConfig`, rather than inferred from example commands.

#### A. Budget and PPO optimization

| Argument | Type / accepted values | New-run default | Effect, interactions and consequences |
| --- | --- | --- | --- |
| `--total-hands` | Integer > 0 | `100000` | Cumulative completed training-hand budget, including earlier hands on resume. Finishes hands; never exceeds this budget. It is not a sample or update count. |
| `--rollout-decisions` | Integer > 0 | `1024` | Target number of **trainable learner actions** in one fixed-policy rollout. Frozen, heuristic and opponent actions do not count. Whole hands/waves may overshoot; the final budget-limited batch can be smaller. |
| `--max-rollout-hands` | Integer > 0 | `4096` | Caps hands per **trump-only** rollout, even if no samples occur. Has no effect when bid or play is trainable. Reaching the cap returns collected samples, possibly none, so another rollout can continue within the total budget. |
| `--ppo-epochs` | Integer > 0 | `4` | Passes over the collected batch; each pass reshuffles. Increases data reuse and potential policy drift, not environment hands. `target-kl` can stop passes early. |
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

A **hand** executes four auction decisions, one trump decision and 28 plays.
A **sample** is one enabled learner action. A **rollout** is the complete-hand
collection made before changing weights. A **minibatch** is a shuffled slice of
that rollout. An **epoch** visits its samples once; an **optimizer step** updates
weights using one minibatch. With `B` samples and no KL early stop, one PPO update
uses `ppo_epochs × ceil(B/minibatch_size)` optimizer steps. An empty batch has none.
For example, bidding-only needs 512 hands for a full 1024-sample batch; play-only
needs 74 hands and supplies 1036 samples. More hands do not imply proportionally
more updates if batch size, sparse trump samples, final truncation or learning
phases change. Samples are collected again from the updated policy next rollout.

#### B. Learner phases and legacy compatibility

| Argument | Accepted values | New-run default / resolution | Meaning |
| --- | --- | --- | --- |
| `--learner-bid` | `heuristic`, `neural-frozen`, `neural-train` | Unset; see rules below | Independent bidding route and optimizer membership. |
| `--learner-trump` | Same three states | Unset; see rules below | Only the winning bidder acts; training samples can be sparse. |
| `--learner-play` | Same three states | Unset; see rules below | Seven play actions per learner seat if trainable. |
| `--learner-bid-source` | Source token or path | Unset → `init` for a neural phase | Resolve only the bidding initialization. |
| `--learner-trump-source` | Source token or path | Unset → `init` for a neural phase | Resolve only the trump initialization. |
| `--learner-play-source` | Source token or path | Unset → `init` for a neural phase | Resolve only the playing initialization. |
| `--learn-phases` | Nonempty distinct comma-separated subset of `bid,trump,play`, no spaces | `bid` | Legacy set of trainable neural phases; order canonicalizes to bid/trump/play. Explicit use alongside modern states must identify exactly the same trained set. |
| `--play-policy` | `frozen`, `trainable` | Unset | Legacy assertion: in modern syntax requires neural-frozen or neural-train play respectively; heuristic play satisfies neither. With legacy syntax it must agree with membership in learn-phases. |
| `--trump-policy` | `heuristic`, `neural` | argparse `heuristic`; asserted only if explicitly supplied | Checks the effective trump route. Omission does not switch an inherited neural trump module to heuristic. Use explicit learner states for independent routing. |

| Effective state | Rollout action selection | Included in buffer / Adam |
| --- | --- | --- |
| `heuristic` | Existing deterministic HeuristicAgent V1 | No / no |
| `neural-frozen` | Greedy legal action from frozen network | No / no |
| `neural-train` | Masked stochastic action from the rollout's policy snapshot | Yes / yes |

**Omission rules matter.** If any learner state **or source** flag is supplied,
the modern configuration path is active. Unmentioned phases declared neural in
the initialization checkpoint default to neural-frozen; the others default to
heuristic. An explicit `--learn-phases` makes otherwise unspecified members
neural-train. There must ultimately be at least one trainable phase. A source flag
alone does not opt its phase into training. Resume instead inherits saved states
and sources before checking explicit overrides.

Without any learner flags, legacy phase construction applies: learn-phases defaults
to bid, enables neural learning for those phases, and retains existing neural
continuations from version-2 initialization (play remains neural for version 1).
Thus default version-1 initialization learns bid with frozen play and heuristic
trump; a version-2 file declaring neural trump can retain frozen neural trump.
Do not rely on the literal argparse trump default to disable it. Different
initialization paths are not promised to yield identical newly initialized weights
or RNG sequences; preserve the original syntax/environment for legacy reproduction.

All nonempty trainable subsets are supported, including trump alone. Every
frozen neural phase must have a trained compatible source. All-frozen/heuristic
configurations belong to evaluation, not training. Conflicting explicit legacy
and modern choices fail before training; prefer spelling all three modern states
for a new experiment. An older run whose saved configuration used legacy syntax
must resume with that syntax, or initialize a new stage to adopt source flags.

#### C. Checkpoint entry points and neural sources

| Argument | Type / accepted values | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `--init-play-checkpoint` | Path to compatible version-1 checkpoint | Unset | Pretrained play-only initialization; rejects a version-2 model. New optimizer, counters, RNGs and roster. |
| `--init-checkpoint` | Path to compatible version-1 or version-2 checkpoint | Unset | New stage from model parameters. Does not import optimizer, old best score, progress or historical roster. |
| `--anchor-checkpoint` | Path to original version-1 play checkpoint | Inferred | Version-1 initialization uses that same file as anchor. Version-2 initialization/resume retains the original anchor identity from metadata; an override must match its SHA256. |
| `--resume` | Path to compatible version-2 checkpoint | Unset | Exact saved-stage continuation. Mutually exclusive with the two initialization flags. Restore omitted settings, model, optimizer, counters, RNGs and active roster. |

If no entry-point source is supplied, the CLI loads
`runs/ppo-play-v1/checkpoints/best.pt`; it does **not** create a pretrained model.
A compatible original play-only anchor is required even if learner play is heuristic.
The three entry-point flags are mutually exclusive. A renamed/moved anchor can be
supplied by `--anchor-checkpoint` with the same bytes, but other recorded immutable
dependencies still need to be accessible at their saved paths.

| Learner source | Resolution / initialization |
| --- | --- |
| `init` | The initialization checkpoint. Copy the full requested trained phase (trunk, policy and value heads). Only for neural-train, a missing trained phase instead gets representation transfer and fresh heads, explicitly recorded. On resume, explicit `init` refers to the original stage initialization identity, not the evolving resume file. |
| `anchor` | The immutable original play-only checkpoint. Valid for neural play only; bid/trump requests fail even when trainable. |
| `new` | Trainable phases only. From the initialization file, copy the trained play trunk, or the first trained phase trunk in bid/trump/play order if play is unavailable. Initialize fresh policy/value heads even if the requested phase was already trained. Requires a trained representation. |
| Registered name | Register with `--opponent-checkpoint NAME=PATH`. Must identify a single checkpoint containing the requested trained neural phase; variants and composed policies are not phase sources. |
| Direct or archived path | Same trained-phase requirement as a registered name. An archived snapshot is a fixed file, not a live subscription to later snapshots. |

All heuristic phases reject sources; frozen phases cannot use `new` or untrained
weights. Version-1 output layers have bid/trump logits, but those are **not trained
bid/trump policies**. Version-2 sources must declare the requested phase in both
neural_phases and trained_phases. Source paths resolve to absolute paths and SHA256
before training. Source parameters are copied into independent learner modules;
training never modifies the file. Exact resume verifies identities and optimizer
membership. Different phase sources may be supplied independently, subject to
learner-width compatibility. See [mechanics](multiphase_selfplay.md#initialization-and-freezing).

#### D. Training opponents and historical snapshots

| Argument | Type / accepted values | Default | Meaning and interactions |
| --- | --- | --- | --- |
| `--opponent-pool` | Nonempty `NAME:WEIGHT,...`; unique registered names, finite positive weights | `HHH:0.5,anchor:0.5` | One opponent controls both opposing seats for a complete hand. Ratios normalize to sorted exact rational probabilities; `HHH:7,anchor:3` equals `HHH:0.7,anchor:0.3`. Realized counts fluctuate. |
| `--opponent-checkpoint` | Repeatable `NAME=PATH` | No archives | Register an immutable compatible checkpoint for learner sources, training or validation. Names cannot duplicate built-ins/anchor or reserved `init`, `new`, `self-snapshots`. Active whole-checkpoint opponents need all routed neural phases trained. |
| `--opponent-composition` | Repeatable `NAME=JSON_PATH` | No compositions | Register a frozen phase-composable opponent. Source checkpoint registrations are resolved before JSON definitions; earlier composed names cannot serve as single phase sources. See [JSON recipe](#register-composed-training-opponents). |
| `--snapshot-every-hands` | Integer > 0 | `32768` | Publication threshold since last snapshot, checked at rollout/update boundaries. Publishes only following a nonempty batch when every routed neural phase is trained. Threshold may overshoot; not every exact multiple produces a filename. |
| `--snapshot-pool-size` | Integer > 0 | `8` | Maximum active historical policies. Evicts oldest membership, not files. The separate anchor is never evicted. No CLI switch disables publication; snapshots can be produced even if the current pool does not sample them. |

Built-in variants are defined [once above](#22-policy-baselines-and-ablations):
HHH is heuristic in all phases; CHH passes unless forced but retains heuristic
trump/play; HHR changes only play to random. HRR, CRR and RRR supply other fixed
ablations. `anchor` uses heuristic bid/trump and the original frozen neural play.
Registered checkpoints restore their declared neural phases and use heuristic
fallbacks. Composed opponents specify phases independently, including different
neural files. They always remain frozen and use greedy neural inference.

`self-snapshots` first uses the **original anchor** if the active roster is empty;
it does not automatically use the stage's initial joint model. Once published
policies exist it selects uniformly from the active roster using a separate
seed stream. A new stage starts with an empty roster; resume restores the exact
saved roster. Keep files immutable and preserve evicted archives needed by older
checkpoints. This is a learning partnership against frozen opponents, not four
simultaneously updated PPO players. No live learner opponent is implemented.

#### E. Validation and checkpoint selection

| Argument | Type / accepted values | Default | Meaning and interactions |
| --- | --- | --- | --- |
| `--eval-opponents` | Nonempty, unique comma-separated exact registry names; no separator whitespace | `HHH,CHH,anchor` | Separate reports using the same fixed paired schedules. May include registered archives/compositions, but not the dynamic `self-snapshots` pool entry; register a concrete snapshot to evaluate it. |
| `--selection-opponent` | One name present in eval-opponents | `anchor` | `best.pt` maximizes this opponent's mean signed marks/hand, strictly greater than the previous best; ties retain the earlier best. Does not automatically change when training opponents or validation names change. |
| `--eval-every-hands` | Integer > 0 | `5000` | Validation threshold since last evaluation, checked after collection/update. May overshoot by a rollout; also evaluates at total-hand completion, including an empty final batch. |
| `--eval-deals` | Integer > 0 | `128` | Duplicate-deal **groups per opponent** per validation. Two swapped games/group in hand mode, so 128 means 256 games against each opponent. Increasing this changes the selection experiment/cost. |
| `--bootstrap-samples` | Integer > 0 | `200` | Complete-group percentile-bootstrap replicates in validation, not PPO minibatches. CLI cannot disable validation bootstrap with zero. Evaluation CLIs have their own defaults. |
| `--checkpoint-every-hands` | Integer > 0 | `5000` | Save latest.pt at threshold and final completion. Intermediate progress since last save is not guaranteed recoverable after interruption. Independent of best.pt selection/publication. |

For HHH-only training/selection, supply **all three** of `--opponent-pool HHH:1`,
`--eval-opponents HHH`, and `--selection-opponent HHH`; changing the training pool
alone leaves anchor selection and the three default validation opponents intact.
Training deals collect stochastic learner actions and are never used as validation
trajectories. Validation is greedy on a fixed separately seeded schedule. The final
held-out comparison uses a new seed and is not fed back into checkpoint selection.

A duplicate group has ABAB and BABA partnership seating on the same absolute-seat
deal and shaker. Win rate averages games; signed marks averages completed hands
from the policy partnership's perspective. Bootstrap resamples **whole groups**,
including both swapped games, and interpolates 95% percentile bounds. It is an
uncertainty estimate under the experiment's opponents/deals, not proof of optimal
play. [Arena definitions](arena.md) own detailed metric/statistical semantics.

#### F. Execution, diagnostics and files

| Argument | Type / accepted values | Default | Meaning and interactions |
| --- | --- | --- | --- |
| `--seed` | Integer `0 ≤ seed < 2^63` | `42` | Experiment seed for separate deal, action, shuffle, opponent, snapshot, imitation and validation streams. Record it with software versions; it is not a hardware-independent bitwise guarantee. |
| `--device` | `auto`, `cpu`, `cuda` | `auto` | Main optimizer device; auto chooses CUDA when available. Explicit unavailable CUDA fails. Rollout/evaluation neural inference runs on CPU even when optimization uses CUDA. |
| `--threads` | Integer ≥ 0 | `1` | Maximum simulation processes; 1 is sequential, 0 auto. Auto caps at 8; both auto/positive requests respect CPU affinity, physical-core information, quotas and independent-task count. Scheduling does not seed decisions. |
| `--eval-threads` | Integer ≥ 0 | `1` | Independent validation worker request using the same safe resolver. Runtime-only; changing it does not alter training samples or exact-resume configuration. Small jobs may be faster sequentially. |
| `--torch-threads` | Integer > 0 | `1` | Main-process PyTorch intra-op CPU threads, not simulations. Worker numerical threads are 1 to prevent oversubscription. Changing this stored setting is not accepted for exact resume. |
| `--output-dir` | Directory path | `runs/ppo-selfplay` | Config/progress/checkpoints/snapshots. New stages reject an existing progress.jsonl; resume can use the same directory or a new directory while preserving dependency paths. |
| `--log-dir` | Directory path | `logs/experiments/<command>-<run-id>/` | Readable training.log and training.csv. Training log appends; CSV is regenerated from available progress after successful completion. Reuse explicitly to keep a resumed transcript together. |
| `--forgetting-hands` | Integer > 0 | `8` | Size in anchor-generated hands of the fixed play-observation KL dataset. KL(anchor play || current play) is diagnostic, not a penalty. Must remain positive even with heuristic learner play; then KL is unavailable. |
| `--bid-bc-hands` | Integer ≥ 0 | `0` | Optional HHH bidding imitation before PPO; zero disables. Requires trainable bidding. Four masked-cross-entropy epochs use separate data/shuffle streams and a temporary optimizer. Logs loss/accuracy, including forced and voluntary subsets. Not rerun on resume; an explicit value must still match the saved configuration. |

Worker detection and CPU/CUDA boundaries are explained in [parallel execution](parallel_execution.md).
Use the same numerical backend for determinism checks. Exact cross-GPU, CUDA-build
or PyTorch-version reproducibility is not guaranteed. Increasing workers, epochs
or CPU numerical threads is not a guarantee of higher hands/second.

| Artifact | Contents / use |
| --- | --- |
| `OUTPUT/config.json` | Resolved stored configuration; initial sources and identities for explicit phases, opponent definitions, optimization settings. Legacy configurations can leave learner_phases empty; metadata/startup output records their effective states. |
| `OUTPUT/progress.jsonl` | One row per completed collection/update cycle; cumulative counters, per-rollout samples/rewards, timing and validation reports. Not an experience dataset or a continuously written per-action log. |
| `OUTPUT/checkpoints/best.pt` | Model/optimizer/counters/RNGs at the best primary validation score. Use for final evaluation or a new stage; it may be older than latest.pt. |
| `OUTPUT/checkpoints/latest.pt` | Most recent periodically saved completed cycle. Preferred exact-resume point. Restores best_score and validation/publication thresholds too. |
| `OUTPUT/snapshots/policy-NNNNNNNNN.pt` | Immutable published frozen policies; filenames use actual cumulative hands. Active roster bounded, archived files retained. |
| `LOG/training.log` | Invocation, resolved policies/SHA256, progress, validation summaries and timings. Stdout is mirrored as it happens; errors on stderr are not necessarily in this file. |
| `LOG/training.csv` | Flattened per-progress-row metrics, phase samples/losses, opponent counts, primary/per-opponent validation and worker/time fields. Generated after a successful command return; interrupted runs may have progress without a refreshed CSV. |

Read `hands`, `training_samples`, `updates`, and `optimizer_steps` as different
cumulative counts. Modern `play_samples` is a compatibility alias for **all**
training samples; actual playing counts are `phase_samples.play` per rollout.
`mean_reward` averages all collected hand rewards, including zero-sample hands;
`training_by_opponent` separates counts and means. `metrics` contains aggregate
and `bid_*`/`trump_*`/`play_*` losses, entropy, KL and clipping statistics. A phase
absent from a batch has no optimized phase metrics; console fallback zeros are
not evidence it trained. `frozen_phases_verified` must remain true for every frozen
neural phase; `frozen_play_verified` is its play compatibility field. A heuristic
phase is not reported as a verified pretrained network. `playing_anchor_kl` can be
nonzero for frozen joint play because its reference is the **original anchor**,
not the stage's selected play source. No KL regularizer is implemented.

Performance fields separate collection, optimization, validation, checkpoint and
update time. Throughputs distinguish hands from learned decisions. Worker inference
and loading durations are sums across workers, not additive wall-time stages;
RSS samples at hand boundaries do not measure continuous peak memory. Startup,
imitation and fixed-dataset construction precede the reported training-loop runtime.

### Stage A: frozen play, learned bidding

**Recipe 2 — bidding-only.** Requires an existing compatible original play-only
checkpoint, for example Recipe 1's best.pt. Explicit modern states preserve that
play network bitwise and create new bid heads from its representation. Trump is
heuristic. Set imitation hands to zero to ablate the optional teacher initialization.

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
For an independent parameter-equality check, use the controlled bidding evaluation
below. The older equivalent routing syntax is `--learn-phases bid --play-policy frozen
--trump-policy heuristic`; it retains its own established initialization sequence.

### Playing-only with the modern trainer

**Recipe 3.** Requires the original play checkpoint. Use this path to include play
in a version-2 experiment that can later change phases/opponents. For an exact
legacy method/seed reproduction use Recipe 1 instead; neither path guarantees
identical training samples to the other.

```bash
python -m examples.train_selfplay_ppo \
  --init-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --learner-bid heuristic --learner-trump heuristic --learner-play neural-train \
  --opponent-pool HHH:1 --eval-opponents HHH --selection-opponent HHH \
  --total-hands 100000 --rollout-decisions 4096 --ppo-epochs 4 --minibatch-size 512 \
  --threads 4 --eval-threads 1 --seed 42 --device auto \
  --output-dir runs/ppo-modern-play --log-dir logs/experiments/ppo-modern-play
```

### Independent learner phases and trump-only training

**Recipe 4.** Requires a version-2 joint checkpoint with trained neural bid/play and
its original anchor file. Neither bid nor play is updated; trump gets new heads.
A defending learner contributes zero samples, a declaring learner exactly one.
The sample target can require many hands; lower the hand cap for bounded short
checks. Empty capped/final batches skip optimization but still advance progress.
Every hand executes all seven tricks.

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
composed heuristic-trump opponent instead, as in Recipe 10.

### All three phases against HHH

**Recipe 5.** Requires Recipe 1's original play checkpoint, not a pretrained bid or
trump policy. Both new phases transfer its trunk with fresh policy/value heads;
playing starts from the trained whole network. All three modules are optimized.

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
opponent. Before reusing best.pt as a whole frozen opponent, ensure metadata records
all three routed phases as trained; merely declaring a phase neural is insufficient.

### Stage B: joint training with historical opponents

**Recipe 6.** Requires Recipe 5's best.pt and its immutable original anchor.
This is a **new stage**, because the opponent distribution changes. It copies the
three phase networks and uses fresh Adam/RNG/counters; it starts with an empty
historical roster. It does not resume Recipe 5's optimizer or import its snapshots.

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
For the earlier **bid/play-only** Stage B workflow, initialize from Recipe 2 and
set bid/play neural-train, trump heuristic, and a new output directory such as
`runs/ppo-joint-v1`. Those explicit states keep the legacy named workflow usable.

### Resume and multi-opponent validation

**Recipe 7.** Requires latest.pt and all recorded source/opponent/roster files.
Use the same output directory to append progress, or a new one to retain only
continuation progress locally. Total hands is the cumulative target, not an increment.

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

Changing the cumulative budget or rollout worker request is supported. Evaluation
workers and output/log locations are runtime choices. Changing phase states/sources,
architecture, optimizer settings, canonical opponent distribution, evaluation
procedure or selection criterion requires a new stage; changing CPU/CUDA optimizer
device type is not exact resume. Modern resume restores omitted stored settings;
legacy train_ppo requires you to repeat saved configuration rather than adopting it
automatically. Best.pt is selected performance, latest.pt is newest saved progress.
Extending a run that ended with a shortened final rollout resumes that saved state,
but is not equivalent to having used the longer budget from the beginning.

### Mixed-phase training

**Recipe 8.** Bidding-train/play-frozen/trump-heuristic is Recipe 2; trump-train with
bid/play-frozen is Recipe 4. Here is bid/play training with frozen **neural trump**.
Requires a checkpoint with all three phases trained, such as Recipe 5's best.pt.

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

### Independent phase checkpoints

**Recipe 9.** Requires the three named compatible checkpoints, a joint initialization
checkpoint and its anchor. All learner sources must share the initialization width.
The example freezes bidding/play from different stages while updating trained trump.

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
Do not overwrite files referenced by this experiment. Snapshot files can replace
the named source paths when their metadata declares the needed trained phase.

### Register composed training opponents

**Recipe 10.** Requires a trained joint bid/play checkpoint and its original anchor.
Create `runs/pretrump-composition.json` after ensuring `runs/` exists:

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

JSON must have exactly bid/trump/play keys; each value has a strategy and, for
neural only, a source. Bid strategies are neural/heuristic/random/conservative;
trump/play are neural/heuristic/random. Sources accept init, anchor, registered
checkpoint names, or direct paths, including retained snapshots. Paths are relative
to the working directory, not to the JSON file. `new` is not an opponent source:
frozen opponents need trained networks. Different phase files and widths are
allowed for composed **opponent** inference if each is independently compatible;
learner initialization still requires one shared width.

A learner source initializes an optimizable copy, whereas this JSON configures a
read-only opponent factory. JSON content and resolved identities are saved in the
training configuration; exact resume does not require the original JSON file when
registration flags are omitted, but it does require all referenced checkpoint files.

### Training against an archived checkpoint

**Recipe 11.** Requires Recipe 5's trained all-phase checkpoint and an existing
immutable archive at `runs/archive/historical.pt` (copy/reference an actual retained
snapshot there; retain its dependencies for resumption). Unlike self-snapshots,
this explicitly registered identity never changes its membership or weights.

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

### Held-out phase ablations

**Recipe 12.** Prerequisites: the trained checkpoints in each comparison, including
all requested trained neural phases. Use final deal seeds independent of the fixed
training-validation stream. Keep evaluation results out of optimizer/checkpoint
selection. See [interpretation](phase_composable_evaluation.md#controlled-training-comparisons).

- Bidding: use [identical joint play with heuristic versus neural bidding](#hold-joint-playing-fixed-change-bidding). Trump is heuristic on both sides.
- Trump: use the [trump-only checkpoint versus its frozen bid/play source](#held-out-trump-ablation).
- Playing: use identical neural bid/trump from an all-phase policy, changing only play below.

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

Require shared bidding/trump parameters YES and only play differing. To compare
two learned play policies, replace B's playing strategy by neural and add its
play-checkpoint override; keep bid/trump sources unchanged. A complete joint model
versus HHH comparison measures several interacting phase differences instead.

### Held-out trump ablation

Requires Recipe 4's trained checkpoint and the exact joint bid/play source used
there. If different phase sources were chosen, supply those same sources for B.

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

### Troubleshooting PPO configuration

| Symptom | Cause / action |
| --- | --- |
| Initialization file missing | The modern default expects a pre-existing play checkpoint. Complete Recipe 1 or select a valid explicit initialization file. |
| Unavailable pretrained phase | Version 1 supplies play only; version 2 must mark a neural phase trained. Choose a trained file for frozen policies, or init/new representation transfer for a trainable phase. |
| Anchor bid/trump rejected | The original anchor is play-only. Use a trained joint/archive source or `new` for trainable bid/trump. |
| No trainable phase / conflicting flags | A source alone does not enable training. Spell all three learner states and remove inconsistent learn-phases/play-policy/trump-policy assertions. |
| Architecture mismatch | All learner sources must match the initial hidden width and game/feature/action metadata. Omit hidden-size to infer the initial width; a width override cannot reshape pretrained weights. |
| Frozen checkpoint identity changed | A recorded file was overwritten or replaced. Restore the exact SHA256 file; use immutable copies. Intentional replacement is a new stage, not exact resume. |
| Missing historical dependency | Restore the checkpoint at its recorded path, including inactive archives needed by older resumes. A roster bound does not make archive files disposable. |
| Resume configuration mismatch | Keep phase/source/optimizer/opponent/validation settings unchanged, restore omitted modern settings, and use the same device type. For changed experiments use init-checkpoint and a new directory. |
| Bidding-only legacy anchor comparison rejected | Old bid-only syntax enforces anchor-identical play. Use explicit modern learner states for frozen joint play, and select the appropriate same-play opponent. |
| Trump samples remain zero | Opponents are declaring. Inspect bidding and per-phase counts; use the hand cap/total budget to bound work. Empty batches skip updates. Do not invent trump samples or infer learning from printed zero losses. |
| Frozen play has nonzero anchor KL | Its selected frozen source may be a joint policy different from original anchor. Check frozen verification and actual tensor equality to the selected source. |
| No PPO optimizer steps | A very tight target-kl can stop all minibatches. Review KL threshold, finite diagnostics and software/backend consistency; empty buffers are already skipped separately. |
| Output run already exists | Resume latest.pt in that directory, or choose a fresh directory for a new stage. Do not overwrite provenance to bypass the check. |
| CSV absent after interruption | Read progress.jsonl and the streamed log; CSV export occurs on successful completion. Latest checkpoint, not the final printed hand, controls recoverable state. |

## Agent-vs-agent phase composition

Use this command for new phase ablations. See [composition methodology](phase_composable_evaluation.md)
for checkpoint validation, parameter equality and interpretation.

### Hold joint playing fixed, change bidding

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

### Independent checkpoint sources

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

Use immutable snapshot paths in either agent's checkpoint options to compare
archived policies. Each requested neural phase must have been trained.

### Pure heuristic, conservative and random comparisons

```bash
# HHH against HHH: deterministic paired symmetry check
python -m examples.evaluate_agent_vs_agent --deals 64 --seed 42
# HHH against conservative bidding with heuristic trump/play
python -m examples.evaluate_agent_vs_agent --agent-b-bid conservative --deals 64
# HHH against heuristic bidding with random trump/play
python -m examples.evaluate_agent_vs_agent \
  --agent-b-trump random --agent-b-play random --deals 64
```

### Composition arguments

This is the complete argument surface of `examples.evaluate_agent_vs_agent`.
Evaluation is CPU inference; there is no optimizer/device or trainability flag here.

| Argument | Type / accepted values | Default | Meaning |
| --- | --- | --- | --- |
| `--agent-a-bid`, `--agent-b-bid` | neural, heuristic, random, conservative | heuristic | Bidding strategy; conservative passes when legal, otherwise minimum bid. |
| `--agent-a-trump`, `--agent-b-trump` | neural, heuristic, random | heuristic | Independent trump strategy. |
| `--agent-a-play`, `--agent-b-play` | neural, heuristic, random | heuristic | Independent playing strategy. |
| `--agent-a-checkpoint`, `--agent-b-checkpoint` | Compatible checkpoint path | Unset | Default neural source for that agent; ignored when no phase needs it. Version 1 supplies play only. |
| `--agent-a-bid-checkpoint`, `--agent-b-bid-checkpoint` | Trained bid source path | Unset | Overrides only that agent's neural bidding source. |
| `--agent-a-trump-checkpoint`, `--agent-b-trump-checkpoint` | Trained trump source path | Unset | Overrides only that agent's neural trump source. |
| `--agent-a-play-checkpoint`, `--agent-b-play-checkpoint` | Trained play source path | Unset | Overrides only that agent's neural playing source. |
| `--stochastic-neural` | Boolean flag | Off (greedy) | Seeded phase-local sampling, independent between seats/phases; still frozen inference. |
| `--deals` | Integer > 0 | 2048 | Complete duplicate groups; two games/group. |
| `--seed` | Integer | 42 | Arena deal/agent seed; choose independent final seeds. |
| `--bootstrap-seed` | Integer | 0 | Separate resampling seed. |
| `--bootstrap-samples` | Integer > 0 | 2000 | Paired group bootstrap replicates; intervals unavailable for fewer than two groups. |
| `--threads` | Integer ≥ 0 | 1 | Up to N simulation workers; 0 auto with CPU/task limits. |
| `--episode` | hand, match | hand | Paired hands or complete paired match schedules. Match metrics use actual played-hand counts. |
| `--target-marks` | Integer > 0 | 7 | Match termination target; does not turn hand mode into a match. |
| `--output` | Path | Unset | Full JSON arena report with composition identities and diagnostics. |
| `--log-dir` | Directory path | Auto under logs/experiments | Readable log, hand CSV and metric CSV; separate from JSON path. |
| `-h` / `--help` | Boolean flag | Off | Print accepted flags. |

A neural phase needs its phase override or agent default file; a phase checkpoint
on a non-neural strategy is rejected. Different neural files are allowed per phase,
including archived snapshots. Each selected network must satisfy metadata and
trained-phase checks. Actual network tensors, not just paths/hashes, determine
shared-parameter assertions. See [composition semantics](phase_composable_evaluation.md).

### Compatibility checkpoint evaluation arguments

Both compatibility commands share `--deals 2048`, `--seed 12345`,
`--bootstrap-seed 0`, `--bootstrap-samples 2000`, `--threads 1`, optional `--output`,
optional `--log-dir`, and `--help`, with the same ranges as above. They evaluate
hands only and do not expose independent phase strategies or stochastic inference.

| Command / additional argument | Accepted values / default | Meaning |
| --- | --- | --- |
| evaluate_ppo `--checkpoint` | Required version-1 or version-2 path | Auto-route the model's neural phases; other phases heuristic. |
| evaluate_ppo `--opponent` | One built-in variant; default HHH if both selectors omitted | Mutually exclusive with opponents. |
| evaluate_ppo `--opponents` | Unique comma-separated built-in variants | Separate reports; no archive or composition registration. |
| evaluate_selfplay_ppo `--checkpoint` | Required version-2 path | Rejects version 1; use evaluate_ppo or composable evaluation for play-only files. |
| evaluate_selfplay_ppo `--anchor-checkpoint` | Original anchor path; inferred from checkpoint config | Override must match original anchor SHA256. |
| evaluate_selfplay_ppo `--opponents` | Unique registered names; default HHH,CHH,HHR,anchor | Built-ins, anchor and explicitly registered archives. |
| evaluate_selfplay_ppo `--opponent-checkpoint` | Repeatable NAME=PATH | Registers whole frozen checkpoint opponents. No opponent-composition option here. |

These compatibility learner routes do not add the canonical per-selected-phase
trained check: use files whose routed neural phases are already trained. For
explicit availability validation or any independent phase ablation, use
`evaluate_agent_vs_agent`. Existing single/multi-opponent JSON envelopes remain
unchanged; reports are never merged into an unlabeled multi-opponent score.

## Legacy mixed-opponent training recipes

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
```

```bash
python -m examples.evaluate_ppo --checkpoint runs/ppo-mixed/checkpoints/best.pt \
  --opponents HHH,CHH,HHR --deals 2048 --threads 4 --seed 12345 \
  --output runs/ppo-mixed/test.json --log-dir logs/experiments/ppo-mixed-test
```

```bash
python -m examples.train_ppo --resume runs/ppo-mixed/checkpoints/latest.pt \
  --total-hands 1048576 --train-opponents HHH:7,CHH:2,HHR:1 \
  --eval-opponents HHH,CHH,HHR --selection-opponent HHH \
  --threads 4 --eval-threads 1 --output-dir runs/ppo-mixed
python -m examples.train_ppo --warm-start runs/ppo-play-v1/checkpoints/best.pt \
  --total-hands 524288 --train-opponents HHH:0.5,CHH:0.3,HHR:0.2 \
  --eval-opponents HHH,CHH,HHR --selection-opponent HHH \
  --threads 8 --output-dir runs/ppo-mixed-warmstart
```

```bash
python -m examples.opponent_benchmark --workers 1 4 \
  --hands-per-update 256 --updates 3 --output runs/opponent-performance.json
```

Keep saved optimizer settings and sampling scheme unchanged for exact resume.
Method and migration details: [legacy PPO](ppo_training.md#configuring-training-and-evaluation-opponents).
