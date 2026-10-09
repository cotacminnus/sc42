# SC42 COMMAND REFERENCE

*Texas 42 research workflows | CLI recipes, parameters, checkpoints and outputs*

Source snapshot: cotacminnus/sc42 · evaluation-performance-optimization · commit 93dd899d (2026-10-09 UTC). Commands verified against the CLI source and project documentation on this branch; later revisions may differ.

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
| Train PPO | python -m examples.train_ppo | Play-only masked PPO against frozen HHH |
| Evaluate PPO | python -m examples.evaluate_ppo | Held-out, paired checkpoint evaluation |
| Scaling benchmark | python -m examples.parallel_benchmark | Worker-count throughput and correctness |
| Bootstrap benchmark | python -m examples.bootstrap_benchmark | Old vs. optimized confidence interval implementation |
| Validation benchmark | python -m examples.validation_benchmark | Evaluation scaling by group and worker count |

## 2. Copy-ready commands

### 2.1 Console game, self-play log and engine smoke test

```bash
python -m texas42 --seed 42 --target-marks 7
python -m texas42 --all-ai --seed 42 --target-marks 1 \
  --self-play-log logs/selfplay.jsonl
python -m examples.agent_selfplay
python -m examples.openspiel_selfplay --episode hand --seed 42
```

The OpenSpiel example requires the optional OpenSpiel/pyspiel installation. The standalone agent_selfplay example has no command-line arguments. Console --self-play-log requires --all-ai and writes JSONL; it is not neural-network training.

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

### 2.4 Start PPO from scratch (parallel)

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

### 2.6 Held-out PPO evaluation

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

The A-side is PPO play plus heuristic auction/trump. For HHH comparison, the bidding and trump policies are identical and the contrast isolates the two trick-playing policies. Use new held-out seeds for final testing; repeated selection on one test seed makes it part of validation.

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
| --episode | hand | hand or match for supported baseline arena commands. |
| --target-marks | 7 | Marks required to win matches; counterfactual supports hand only. |
| --output | Unset | Optional JSON export or benchmark profile path. |
| --log-dir | Auto subdirectory | Readable .log / .csv experiment sidecars; does not replace JSON --output. |

### 3.2 Policy / special evaluation flags

| Command | Arguments | Defaults and constraints |
| --- | --- | --- |
| policy_ablation_benchmark | --policy-a, --policy-b | HHH and CRR; values: HHH/HHR/HRR/CHH/CRR/RRR. |
| heuristic_benchmark | --opponent | random (or heuristic). |
| counterfactual_bidding | --policy-a, --policy-b, --target-policy | HHH / CHH / HHH; target must be one of the two policy names. |
| evaluate_ppo | --checkpoint, --opponent | Checkpoint required; opponent HHH (or CHH, HHR). |
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

This command reference reflects implementation status at commit 93dd899d. The latest optimization introduced a cached group bootstrap, independent PPO validation workers, and automatic readable log/CSV exports; earlier branches do not support all flags shown here.

Project sources: https://github.com/cotacminnus/sc42/tree/evaluation-performance-optimization · docs/ppo_training.md · docs/parallel_execution.md · docs/evaluation_performance.md · docs/policy_ablation.md · docs/counterfactual_bidding.md
