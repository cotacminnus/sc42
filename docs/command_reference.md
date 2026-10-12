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
| Train legacy play-only PPO | python -m examples.train_ppo | Reproducible baseline; fixed or mixed heuristic variants |
| Legacy checkpoint evaluation | python -m examples.evaluate_ppo | Held-out, paired checkpoint evaluation |
| Current PPO training | python -m examples.train_selfplay_ppo | Independent phase networks and historical opponents |
| Agent-vs-agent evaluation | python -m examples.evaluate_agent_vs_agent | Canonical new phase-ablation experiments |
| Multi-opponent phase evaluation | python -m examples.evaluate_selfplay_ppo | Compatibility command for anchors and archived policies |
| Scaling benchmark | python -m examples.parallel_benchmark | Worker-count throughput and correctness |
| Bootstrap benchmark | python -m examples.bootstrap_benchmark | Old vs. optimized confidence interval implementation |
| Validation benchmark | python -m examples.validation_benchmark | Evaluation scaling by group and worker count |

## 2. Copy-ready commands

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

```bash
python -m examples.train_ppo \
  --total-hands 524288 --rollout-play-steps 4096 \
  --ppo-epochs 4 --minibatch-size 512 \
  --learning-rate 0.0003 --seed 42 --device auto \
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

See [multi-phase training](multiphase_selfplay.md) for parameter transfer, credit assignment and checkpoint compatibility. Existing play-only commands remain available.

### Stage A: frozen play, learned bidding

```bash
python -m examples.train_selfplay_ppo \
  --init-play-checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --learn-phases bid --play-policy frozen --trump-policy heuristic \
  --bid-bc-hands 4096 --opponent-pool HHH:0.5,anchor:0.5 \
  --total-hands 262144 --rollout-decisions 1024 \
  --ppo-epochs 4 --minibatch-size 256 --learning-rate 0.0001 \
  --threads 8 --eval-threads 1 --device auto --seed 42 \
  --output-dir runs/ppo-bidding-v1
```

### Stage B: joint training with historical opponents

```bash
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-bidding-v1/checkpoints/best.pt \
  --learn-phases bid,play --trump-policy heuristic \
  --opponent-pool HHH:0.3,anchor:0.3,self-snapshots:0.4 \
  --snapshot-every-hands 32768 --snapshot-pool-size 8 \
  --total-hands 524288 --rollout-decisions 4096 \
  --ppo-epochs 4 --minibatch-size 512 --learning-rate 0.0001 \
  --threads 8 --eval-threads 1 --device auto --seed 42 \
  --output-dir runs/ppo-joint-v1
```

### Resume and multi-opponent validation

```bash
python -m examples.train_selfplay_ppo \
  --resume runs/ppo-joint-v1/checkpoints/latest.pt \
  --total-hands 1048576 --threads 4 --eval-threads 1 \
  --output-dir runs/ppo-joint-v1

python -m examples.evaluate_selfplay_ppo \
  --checkpoint runs/ppo-bidding-v1/checkpoints/best.pt \
  --opponents HHH,CHH,HHR,anchor --deals 4096 --seed 12345 \
  --threads 4 --output runs/ppo-bidding-v1/heldout.json
```

Add `--opponent-checkpoint archive=runs/ppo-joint-v1/snapshots/policy-000032768.pt`
and include `archive` in `--opponents` to evaluate an archived policy.

### Multi-phase arguments

| Entry point / flag | Meaning |
| --- | --- |
| `examples.train_selfplay_ppo` | Independent phase networks, frozen historical opponents, hand episodes. |
| `--init-play-checkpoint PATH` | Original play-only pretrained weights; default `runs/ppo-play-v1/checkpoints/best.pt`. |
| `--init-checkpoint PATH` | New stage from phase weights with fresh optimizer/RNG; exclusive with initialization/resume sources. |
| `--resume PATH` | Exact update-boundary continuation; omitted options restore saved configuration. |
| `--learn-phases bid` / `trump` / `play` / comma-separated subset | Legacy enabled neural PPO decisions; default bid. |
| `--learner-bid`, `--learner-trump`, `--learner-play` | Each accepts `heuristic`, `neural-frozen`, or `neural-train`. At least one must train. |
| `--learner-PHASE-source SOURCE` | `init` (default), `anchor`, registered checkpoint name, direct path, or `new`. Resolved path/SHA256 saved before training. |
| `--play-policy frozen\|trainable` | Optional assertion agreeing with learn-phases. |
| `--trump-policy heuristic\|neural` | Legacy assertion; omitted values follow enabled/inherited neural phases. |
| `--opponent-pool NAME:WEIGHT,...` | Default `HHH:0.5,anchor:0.5`; canonical ratios; existing variants, anchor, self-snapshots, registered checkpoints. |
| `--opponent-checkpoint NAME=PATH` | Repeatable frozen checkpoint registration. |
| `--opponent-composition NAME=JSON_PATH` | Repeatable frozen phase composition; registered name works in training pools and validation. |
| `--rollout-decisions N` | Enabled decision target, default 1024; complete hands, variable-trump bounded waves. |
| `--max-rollout-hands N` | Trump-only collection safeguard, default 4096; return a smaller or empty batch at this limit or the total-hand budget. |
| `--bid-bc-hands N` | Default 0; optional four-epoch HHH masked imitation initialization. |
| `--snapshot-every-hands N` / `--snapshot-pool-size N` | Default 32768 / 8; publish at update boundaries; bound active roster. |
| `--eval-opponents HHH,CHH,anchor` | Separate paired reports on the same fixed validation schedule. |
| `--selection-opponent anchor` | Select best.pt by this opponent's mean signed marks. |
| `--forgetting-hands N` | Default 8; fixed anchor-play observation dataset for KL diagnostics. |
| `--anchor-checkpoint PATH` | Original play anchor; inferred from source checkpoint when omitted. |
| `--threads N` / `--eval-threads N` | Simulation and validation worker requests, default 1 each. |
| `--output-dir PATH` / `--log-dir PATH` | Checkpoints/progress in runs; readable logs/CSV in dedicated logs directory. |
| `examples.evaluate_selfplay_ppo --checkpoint PATH --opponents HHH,CHH,HHR,anchor` | Held-out phase-aware evaluation, optional historical registrations and JSON `--output`. |
| `examples.multiphase_benchmark --hands 64 --workers 1 4` | Warmed collection/update/inference/load/evaluation measurements for four configurations. |

The usual PPO optimizer, device, budget, evaluation interval, and checkpoint
interval flags are supported. The new learning rate defaults to 0.0001. The old
`--rollout-play-steps`, variant-only training opponent flags, and legacy sampling
scheme flags belong to `train_ppo`, not the new entry point.

### Independent learner phases and trump-only training

When using `--learner-*`, unmentioned existing neural phases default to frozen;
other unmentioned phases use heuristics. Explicit legacy flags must agree with
the resolved states. With no new flags, the legacy initialization and RNG path
is retained. All sources must have the initialization checkpoint's hidden width.

`init` loads the requested trained phase when available. For `neural-train` only,
a missing phase in `init` receives a copied representation and fresh policy/value
heads, recorded as `representation` initialization. `new` explicitly requests
that initialization even when a trained phase exists. The representation comes
from trained play when available, otherwise the first trained phase in bid/trump/play
order. Named/direct sources must already supply the requested trained phase.
`anchor` supplies trained play only; it cannot supply neural bid or trump.
Heuristic phases reject sources. Frozen phases always require trained sources.

```bash
# Freeze joint bidding/play, learn only trump, select best.pt against pretrump
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --learner-bid neural-frozen --learner-trump neural-train --learner-play neural-frozen \
  --opponent-checkpoint pretrump=runs/ppo-joint-v1/checkpoints/best.pt \
  --opponent-pool pretrump:0.7,HHH:0.3 \
  --eval-opponents pretrump,HHH,anchor --selection-opponent pretrump \
  --total-hands 262144 --rollout-decisions 1024 --max-rollout-hands 4096 \
  --threads 8 --eval-threads 4 --output-dir runs/ppo-trump-v1

# Train bidding from an archive, preserve anchor play, use heuristic trump
python -m examples.train_selfplay_ppo \
  --init-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --opponent-checkpoint archive=runs/ppo-joint-v1/snapshots/policy-000032768.pt \
  --learner-bid neural-train --learner-bid-source archive \
  --learner-trump heuristic --learner-play neural-frozen --learner-play-source anchor \
  --output-dir runs/ppo-archived-bid

# Resume restores phase sources, optimizer, RNG and snapshot roster
python -m examples.train_selfplay_ppo \
  --resume runs/ppo-trump-v1/checkpoints/latest.pt \
  --total-hands 524288 --threads 4 --output-dir runs/ppo-trump-v1
```

Changing a phase source or trainability requires a new stage using
`--init-checkpoint`, with a new output directory. Resume verifies source SHA256;
it does not reinitialize the learned networks. Keep source files and historical
dependencies available. Zero-sample batches advance hand counts and logging,
skip PPO optimization, and still allow checkpoint saving and validation.

### Register composed training opponents

Create a JSON file such as `runs/pretrump-composition.json`:

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
  --learner-bid neural-frozen --learner-trump neural-train --learner-play neural-frozen \
  --opponent-checkpoint pretrump=runs/ppo-joint-v1/checkpoints/best.pt \
  --opponent-composition joint_heuristic_trump=runs/pretrump-composition.json \
  --opponent-pool joint_heuristic_trump:0.7,HHH:0.3 \
  --eval-opponents joint_heuristic_trump,HHH --selection-opponent joint_heuristic_trump \
  --output-dir runs/ppo-trump-composed
```

Every JSON file specifies `bid`, `trump`, and `play`. A neural phase requires a
`source`: `init`, `anchor`, a registered checkpoint name, or a checkpoint path.
Paths are relative to the working directory. Non-neural phases omit `source`;
they support the existing heuristic/random strategies and conservative bidding.
Independent source paths can include immutable historical snapshots. Frozen
opponents use greedy neural inference. `self-snapshots` remains a separate
pool entry, with the fixed anchor fallback until a trained snapshot exists.

### Held-out trump ablation

```bash
python -m examples.evaluate_agent_vs_agent \
  --agent-a-checkpoint runs/ppo-trump-v1/checkpoints/best.pt \
  --agent-a-bid neural --agent-a-trump neural --agent-a-play neural \
  --agent-b-checkpoint runs/ppo-joint-v1/checkpoints/best.pt \
  --agent-b-bid neural --agent-b-trump heuristic --agent-b-play neural \
  --deals 4096 --seed 31415926 --threads 4 \
  --output runs/ppo-trump-v1/held-out-trump.json
```

Require `Shared bidding parameters: YES`, `Shared playing parameters: YES`, and
only trump differing. This holds both frozen continuations constant. Use a held-out
seed rather than the validation schedule that selected the checkpoint.

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

| Argument | Default / meaning |
| --- | --- |
| `--agent-{a,b}-bid` | `heuristic`; neural, heuristic, random or conservative. |
| `--agent-{a,b}-trump`, `--agent-{a,b}-play` | `heuristic`; neural, heuristic or random. |
| `--agent-{a,b}-checkpoint PATH` | Default source for that agent's neural phases. |
| `--agent-{a,b}-{bid,trump,play}-checkpoint PATH` | Override only that phase's neural source. |
| `--stochastic-neural` | Optional phase-seeded sampling; default greedy inference. |
| Shared evaluation flags | `--deals 2048 --seed 42 --bootstrap-seed 0 --bootstrap-samples 2000 --threads 1`. |
| Hand/match flags | `--episode hand` or `--episode match`; `--target-marks 7`. |
| Outputs | `--output PATH` for JSON; `--log-dir PATH` for readable logs and CSV. |

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
