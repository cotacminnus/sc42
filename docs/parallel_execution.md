# Parallel simulation

`--threads N` controls **simulation processes**, not Python threads. Texas 42's
engine is CPU-bound Python; processes avoid the GIL. Each hand's decisions depend
on previous actions, so a hand runs sequentially. Independent hands (training)
or complete duplicate groups (evaluation) can run concurrently.

## Worker resolution and shared infrastructure

`texas42.parallel.resolve_workers(requested, tasks=None)` rejects negatives,
booleans, and nonintegers. One means sequential. Zero means automatic, capped at
`AUTO_WORKER_CAP = 8`. Explicit positive requests are clamped to detected CPU
resources and available independent tasks. At least one worker is always returned.
Detection intersects logical CPU count, CPU affinity, Linux physical-core IDs
when readable, and available Linux cgroup CPU quotas. Quotas are rounded down
with a minimum of one. Missing information is ignored; if all information is
missing, the fallback is one. This is best-effort detection, not a universal
resource monitor. Explicit requests cannot override detected CPU limits.

`OrderedPool` uses portable multiprocessing **spawn**, bounded submission (at most
twice the worker count outstanding), ordered collection, and explicit shutdown.
A failure cancels queued work and waits for the bounded set of running tasks;
no partial batch is returned. Errors include task index and engine game context,
or PPO episode and worker process context. A user policy that never returns can
still hang a running task; there is no policy watchdog. Nested pools are rejected.
Use pools/trainers as context managers or call `close()` in a `finally` block.
Executable callers must use the standard `if __name__ == '__main__'` guard.
Interactive notebooks should invoke module CLIs rather than spawning from cells.

Spawn inherits numerical-library limits of one for OMP, MKL, OpenBLAS and NumExpr;
PPO workers explicitly set PyTorch intra-op threads to one. The temporary parent
environment change during spawn is restored and assumes one orchestration thread.
The ordinary engine does not import PyTorch or NumPy.

## Arena and counterfactual evaluation

```bash
python -m examples.policy_ablation_benchmark --policy-a HHH --policy-b CHH \
  --deals 4096 --seed 42 --threads 0
python -m examples.heuristic_benchmark --opponent heuristic --deals 4096 --threads 4
python -m examples.counterfactual_bidding --policy-a HHH --policy-b CHH \
  --target-policy HHH --deals 128 --threads 4
python -m examples.evaluate_ppo --checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --opponent HHH --deals 4096 --seed 114514 --threads 0
```

`compare_agents(..., threads=1, performance=None)` remains sequential by default.
Each job reconstructs the original schedule from the experiment seed and group
index and runs both swapped partnerships, including complete matches in match
mode. Group IDs, shaker rotation and absolute-seat agent seeds are unchanged.
Results are ordered by group index. The parent computes metrics and the existing
paired bootstrap only after all groups complete. No worker uses completion order,
PID, or wall time as a seed. All deterministic outcome JSON is unchanged across
worker counts, including stochastic policies with deterministic seeded factories.

`variant_factory()` now returns a picklable callable object. Custom factories
must be importable top-level functions/classes with picklable state and must
construct an agent deterministically from seat/seed (not a mutable cross-game
factory counter). Stateful agent memory remains per seat/episode. Closures
continue to work with `threads=1`, but parallel use raises a clear error.
Never replace an unsupported factory with another policy. PPO evaluation transfers
one CPU snapshot per worker initialization, constructs the frozen model once,
and shares its weights among fresh per-seat agents. It never sends CUDA models
or contexts to workers. Counterfactual jobs retain both orientations and all
BID/PASS replays in the original group; bootstrap analysis remains in the parent.

Arena CLIs print resolved workers, simulation time (including pool startup and
shutdown), aggregation/bootstrap time, total time and throughput. The optional
`performance` dict receives timing metadata separately from `ArenaReport`, so
outcome equality and deterministic exported JSON remain meaningful.

## Synchronous PPO collection

```bash
python -m examples.train_ppo --total-hands 524288 --rollout-play-steps 4096 \
  --ppo-epochs 4 --minibatch-size 512 --seed 42 --device auto \
  --threads 0 --torch-threads 1 --output-dir runs/ppo-parallel
```

`--threads` selects simulation workers; `--torch-threads` selects the main
process's PyTorch intra-op threads. These are distinct. Workers use one numerical
thread regardless of the main setting. Architecture A is implemented: frozen CPU
inference in workers, optimization only in the parent on its configured CPU/CUDA
device. No centralized GPU inference server is implemented.

The next batch is determined before submission: `ceil(rollout_play_steps / 14)`
complete hands, capped by remaining `total_hands`. Each episode index appears
exactly once. Contiguous chunks contain complete hands, and results are restored
to episode order. A persistent pool survives updates. Each rollout serializes a
detached CPU weight snapshot once, sends it per chunk (at most one chunk per
configured worker), and caches the loaded snapshot per worker by content hash.
The parent waits for every chunk before GAE or optimization. Workers have no
optimizer, so all samples are on-policy with respect to one frozen snapshot.

Each hand still plays 33 decisions, seven tricks, and 42 points. Only the two
learning seats' seven play observations are returned, with separate episode/seat
trajectories. Rewards remain zero except for each seat's last record, which gets
the engine's signed terminal marks. Parent-side GAE uses the existing per-seat
implementation without interleaving seats or crossing episodes. Collection checks
exact sample counts, unique episodes, complete hands, and immutable records.

Training logs include requested/resolved workers, backend, sampling scheme,
worker PIDs (diagnostic only), collection/optimization/evaluation/checkpoint times,
update/total wall time, hands/s and learned decisions/s. GPU operations synchronize
before timing optimization and copying snapshots. Pools close on normal completion,
interrupts and exceptions. Periodic evaluation currently uses a separate temporary
pool; rollout workers remain idle, so their memory remains allocated during it.

## RNG and checkpoints

There are two explicitly named schemes:

- `legacy-v1`: original persistent action generator. Default when `threads=1`.
  It requires sequential simulation and preserves old seed meaning.
- `episode-v1`: each hand samples on CPU with a fresh generator seeded by
  `stream_seed(seed, 'parallel-episode-v1', episode_index)`. Default for requests
  other than one, even if resource limits resolve to one worker. Assignment to
  workers does not affect the stream. Use `--sampling-scheme episode-v1 --threads 1`
  for a sequential reference to a parallel run.

On the same CPU software/configuration, episode-v1 samples and PPO updates are
tested equal across worker counts. It is intentionally **not** equivalent to the
legacy stream. GPU optimization and different library/hardware builds retain their
usual numerical reproducibility limitations. PID telemetry and timing will differ.

Checkpoints retain all model, optimizer, progress and RNG fields. An added
`execution` object records requested/resolved workers, backend, sampling scheme
and main PyTorch thread count. Older checkpoints without that object explicitly
migrate to legacy-v1: their old `config.threads` means **torch_threads**, and their
simulation count is one. They remain loadable for evaluation and exact sequential
resumption. Specify the original numerical-thread setting when resuming them.

Exact resume rejects sampling-scheme changes, different optimizer device types,
and other changed training hyperparameters. Episode-v1 permits different worker
counts, including one with the explicit episode-v1 setting. To start a new parallel
experiment from old weights, use `--warm-start checkpoint.pt`; this starts fresh
optimizer, counters and RNG states and does not claim continuity. `--resume` and
`--warm-start` are mutually exclusive. Keep the original hidden size.

## Measuring performance

```bash
python -m examples.parallel_benchmark --workers 1,2,4,8,0 --deals 2048 \
  --ppo-hands-per-update 256 --ppo-updates 3 --device cpu \
  --output /tmp/scaling.json
```

The benchmark asserts identical arena output, episode-v1 rollout records, and
final PPO parameters across worker counts. It reports arena time/speedup/efficiency,
warm PPO collection throughput, optimization sample throughput and full PPO wall
time. One initial full PPO update warms each persistent pool; warmed throughput
excludes that update, but full wall time includes it. Increase workloads to amortize
spawn overhead. `--arena-only` works without PyTorch. Run again with `--device cuda`
where available; do not overlap benchmarking jobs or compare loaded vs idle hosts.
Parent peak RSS is a cumulative process high-water mark, not a worker-memory
measurement. CPU/GPU utilization is not sampled.

More workers can be slower: spawning, imports, weight/trajectory serialization,
IPC, memory pressure and CPU contention can dominate. Small PPO batches have
few hands per worker and can be especially inefficient. The parent optimizer and
bootstrap are still sequential. CPU rollout inference avoids per-action GPU
synchronization but may become the bottleneck; a future batched inference service
can replace the rollout boundary without changing the buffer or engine.

## Observed CPU scaling run

One WSL2 run (Python 3.13.5, eight detected physical cores, CPU-only PyTorch)
used 2,048 duplicate groups with 200 bootstrap samples, and PPO batches of 256
hands with one warmup plus three measured updates. All outcome JSON, PPO sample
records and final PPO weights matched across worker counts.

| Request | Workers | Arena total seconds | Games/s | Total arena speedup | Warm PPO hands/s | Collection speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 6.650 | 615.9 | 1.00 | 196.4 | 1.00 |
| 2 | 2 | 4.698 | 871.9 | 1.42 | 359.3 | 1.83 |
| 4 | 4 | 3.752 | 1091.8 | 1.77 | 621.1 | 3.16 |
| 8 | 8 | 3.329 | 1230.4 | 2.00 | 800.6 | 4.08 |
| auto | 8 | 3.354 | 1221.4 | 1.98 | 841.3 | 4.28 |

These are individual timings, not hardware-independent guarantees. In this run,
arena simulation alone fell from 4.700 to 1.009 seconds, while bootstrap/aggregation
still took roughly 2 seconds. Full PPO wall time including warmup was 7.158 seconds
at one worker, 5.177 at four, and 5.786 at auto: eight workers improved warmed
collection but did not minimize end-to-end time for this short workload. PPO
optimization throughput stayed around 13,000 sample presentations/s. Parent peak
RSS rose from 374 to 453 MiB over the whole benchmark; worker RSS and utilization
were not measured. CUDA was unavailable, so no GPU speedup claim is made.


## Opponent pools and validation

PPO worker tasks now include a lightweight `OpponentPool` of canonical variant
probabilities. Each episode chooses its opponent using a separate seed derived
from the episode index; worker assignment cannot affect selection. Both opposing
seats use that identity throughout the hand. The learning seats retain heuristic
auction/trump and the frozen neural play snapshot. Every episode still returns
14 experiences, with independent per-seat GAE and terminal engine marks.
Default HHH collection consumes no additional random draws.

Fixed and mixed pools work with legacy-v1 or episode-v1; comparison across worker
counts requires episode-v1 as before. Checkpoints reject training-mixture or
selection-criterion changes during exact resume. Normalized ratio equivalence is
preserved. `--eval-threads` remains runtime-only. Multiple validation opponents
share the same predetermined schedules and are reported separately; only the
primary opponent determines `best.pt`. See [ppo_training.md](ppo_training.md)
for mixture, validation, warm-start and resume commands.
