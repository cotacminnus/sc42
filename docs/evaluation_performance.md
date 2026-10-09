# Evaluation and statistical performance

Arena bootstrap caches a group's seat-averaged win and signed-mark sums and
counts, then resamples complete groups with the original `random.Random` stream.
Both policy columns share each sampled group index. Win rates pool games;
signed marks pool hands, including matches with differing lengths. Percentiles
still interpolate at `(samples - 1) * p`, at 2.5% and 97.5%. Fewer than two
independent groups yield unavailable intervals. A zero-game replicate makes that
policy's intervals unavailable; a zero-hand replicate makes its marks interval
unavailable. Detailed game records and ordinary metrics are retained.

`texas42.statistics` contains dependency-free scalar cluster resampling and
percentile helpers. The public single-policy `arena.bootstrap()` remains available.
Specialized counterfactual estimators retain their own statistical semantics.
Grouping changes floating-point addition order: rational seat averages can differ
at rounding precision, so comparisons involving heterogeneous seats allow 1e-12.
Ordinary partnership experiments have exact numerical agreement.

## PPO validation workers

`train_ppo --eval-threads N` controls validation independently of rollout
`--threads`. The default is 1; zero uses the same safe automatic cap as arena
simulation. It is a runtime option, outside checkpoint training configuration;
changing it does not change optimizer/RNG continuity or checkpoint selection.
`evaluate_ppo --threads N` keeps its existing meaning. Each evaluation process
initializes one frozen CPU model and shares it across independent seat agents.
Workers never receive CUDA models. Each validation currently creates a new pool;
small workloads can cost more in process startup than they save in simulation.

## Profiling and exports

Performance telemetry remains outside `ArenaReport.to_dict()`. Simulation wall
time includes process startup, model initialization, scheduling, serialization,
and shutdown. Additional counters report time to the first worker initializer,
summed worker model initialization and summed group execution time. These overlap
and must not be added together as wall-clock phases. Group statistics, ordinary
aggregation, and bootstrap sampling/percentiles are timed separately.

Experiment CLIs save readable `.log` files and CSV exports together in a dedicated
`logs/experiments/<command>-<run-id>/` directory by default. The directory is
created automatically and ignored by Git. Use `--log-dir DIRECTORY` to select a
particular location, for example:

```bash
python -m examples.policy_ablation_benchmark --policy-a HHH --policy-b CHH \
  --deals 1024 --output runs/results.json --log-dir logs/experiments/hhh-vs-chh
```

JSON stays at the requested `--output` location. The log directory contains
`results.log`, `results.csv`, and `results-metrics.csv`; CSV preserves resolved
hand diagnostics and policy metrics with intervals. Counterfactual CSV contains
one row per intervention. Training writes `training.log` and `training.csv` in
its log directory, while checkpoints, config, and `progress.jsonl` stay under
`--output-dir`. Logs include CLI arguments, seeds, resolved workers, timings,
metrics, final summaries and artifact paths. Reusing an explicit log directory
and basename replaces benchmark logs/CSVs; training logs append. Default run
subdirectories prevent accidental overwrites between separate executions.

Run statistical profiling on identical in-memory game fixtures:

```bash
python -m examples.bootstrap_benchmark --output runs/bootstrap-profile.json
python -m examples.validation_benchmark --checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --output runs/validation-profile.json
```

The bootstrap benchmark duplicates 32 real group fixtures with distinct group
IDs to measure scaling. Those duplicated fixtures are not a strength experiment.
The validation benchmark checks exact reports across worker counts, and times the
original object-list bootstrap on the same completed games. Its reference total
replaces only the optimized bootstrap phase; it is a measured component comparison,
not a second independent simulation run. No timing thresholds are CI assertions.

## Measured validation scaling

Measured locally on CPU in WSL2, using the existing `ppo-play-v1` best
checkpoint, HHH, seed 12345, and 2,000 bootstrap replicates. Values include
spawn/shutdown and statistics; they are individual measurements, not universal
worker-count recommendations. All 16 reports matched exactly.

| Duplicate groups | 1 worker | 2 workers | 4 workers | 8 workers |
|---:|---:|---:|---:|---:|
| 128 | 1.232 s | 2.670 s | 3.919 s | 7.296 s |
| 256 | 2.439 s | 3.358 s | 4.443 s | 8.014 s |
| 1024 | 10.042 s | 7.452 s | 7.120 s | 10.161 s |
| 4096 | 38.849 s | 23.608 s | 16.114 s | 17.567 s |

Sequential validation was fastest for 128 and 256 groups; four workers were
fastest for 1,024 and 4,096 groups. At 4,096 groups the retained legacy
bootstrap took 42.851 s on the same PPO game data; the optimized phase took
about 3.4–3.5 s. Process startup and shutdown remain significant. No GPU
utilization or per-worker memory measurements were collected.

## Before/after measurements

| Component | Before | After | Speedup |
|---|---:|---:|---:|
| Bootstrap, 128 groups / 200 replicates | 0.108 s | 0.010 s | 10.72x |
| Bootstrap, 1,024 groups / 2,000 replicates | 8.271 s | 0.782 s | 10.58x |
| Bootstrap, 4,096 groups / 2,000 replicates | 35.122 s | 3.358 s | 10.46x |
| Bootstrap, 16,384 groups / 2,000 replicates | 156.068 s | 19.557 s | 7.98x |
| Complete HHH vs. HHH arena, 4,096 groups / 8 workers | 48.305 s | 5.036 s | 9.59x |
| Complete HHH vs. CHH arena, 4,096 groups / 8 workers | 47.408 s | 5.328 s | 8.90x |
| PPO validation vs. HHH, 128 groups | 8.466 s | 1.218 s | 6.95x |

Complete arena rows are independently executed before/after runs with identical
reports. The small PPO validation row compares the original bootstrap with eight
workers to the optimized bootstrap with the new sequential validation default;
its speedup includes both statistical and process-overhead savings. Existing PPO
validation against CHH at 4,096 groups / eight workers took 17.932 s (14.417 s
simulation and 3.512 s statistics). All reported intervals matched their retained
reference exactly. Synthetic-fixture bootstrap timings are separate from these
complete simulations. Measurements include no claims of policy improvement.

For independent complete before/after arena runs, use:

```bash
python -m examples.arena_statistics_benchmark --deals 4096 --threads 8 \
  --output runs/arena-statistics-profile.json
python -m examples.validation_benchmark --checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --opponents HHH --groups 128 --workers 1 --legacy-small-check \
  --output runs/small-validation-profile.json
```
