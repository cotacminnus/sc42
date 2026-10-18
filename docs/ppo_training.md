# Legacy baseline: PPO trick play

The optional `research/ppo` package trains a feed-forward actor/critic using only
`encode(Observation)`. Bidding and trump selection remain the current heuristic
for the learning partnership. The opposing seats default to HHH and may use
any existing phase variant or a weighted mixture. Freezing these phases makes HHH the primary comparison for
trick-play quality and avoids uniform random overbidding confounding results.
No changes are made to rules, actions, observations, or schema-5 records.

## Scope and commands

This guide owns the original play-only method and version-1 checkpoint continuity.
Use [multi-phase PPO](multiphase_selfplay.md) for new independent bid/trump/play
experiments, phase freezing and historical neural opponents. Modern play-only
training is also possible, but has a different checkpoint architecture and sampling
scheme; matching its phase routes does not reproduce a version-1 experiment.
Copy-ready commands and all flag defaults are in the [command reference](command_reference.md#recipe-1-original-play-only-ppo).

Python 3.11+ is supported, subject to available PyTorch wheels for your Python
version/platform. Install `python -m pip install -e '.[rl]'`. This extra includes
PyTorch >=2.3 and NumPy >=1.24; ordinary `texas42` imports need neither. Install a
CPU or CUDA PyTorch build appropriate to your machine using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
OpenSpiel is not needed. Linux, WSL2 and Windows use the same Python code.

Repeat evaluation with `--opponent CHH` and `--opponent HHR` for complementary
comparisons. HHH keeps auction and trump rules equal and is the primary metric.
Evaluation uses greedy play, independent seat agents sharing immutable weights,
the unchanged arena's duplicate deals, and group-resampled bootstrap intervals.
Output includes checkpoint SHA256, win rate, signed marks, success/set rates,
sample size, intervals, and runtime. Small smoke runs do not establish strength.
Use final test seeds different from the fixed validation schedule used to select
checkpoints. Repeated checkpoint selection can overfit a validation schedule.

For baseline reproduction retain the exact committed heuristic version, optimizer
settings, opponent distribution, checkpoint generation and sampling scheme, not
only the root seed. `auto` with one requested simulation worker selects legacy-v1;
episode-v1 is explicitly available for worker-count reproducibility. Modern
multi-phase sampling uses its own multiphase-episode-v1 stream. See
[Recipe 1](command_reference.md#recipe-1-original-play-only-ppo)
and [modern playing-only](command_reference.md#recipe-3-playing-only-with-the-modern-trainer).
Legacy CLI resume requires repeating nondefault saved settings; unlike modern
resume, omitted arguments do not generally reconstruct the saved Config.

## Architecture and masking

The shared trunk is `FEATURE_SIZE -> 256 -> 256`, with ReLU after each linear
layer; heads output `NUM_ACTIONS` logits and one signed-mark value. These are
currently 387 features and 58 global actions. `--hidden-size` changes trunk width.
The actor-relative encoder supports all seats. Both learning seats share weights;
neither actor nor critic sees opponents' hidden hands or a privileged deal.

Illegal logits become negative infinity before constructing a categorical
policy. Empty masks are rejected. Sampling, log probabilities and entropy all
use this same legal distribution, and updates use the original stored masks.
Reserved IDs remain masked by the engine. Greedy evaluation breaks logit ties
by the smallest global ID. `PPOAgent` implements the ordinary agent protocol
using composition: heuristic bid/trump and neural play. Reset assigns its seat
without rewinding the optional stochastic inference generator.

## Complete hands and per-seat GAE

`Trainer.collect()` creates a fresh hand environment and four independent reset agents.
The learning partnership alternates between absolute teams each episode. Only
its two seats' play actions enter the buffer, normally 14 samples per hand.
The old network remains fixed for the entire rollout. Collection reaches the
requested play-step threshold and then finishes the current hand; the total-hand
budget is never exceeded. Every hand still has 33 decisions and seven tricks.

Each controlled seat has its own seven-observation trajectory. Rewards are zero
until its final recorded play, which receives the engine's terminal signed marks:
+marks for the scoring partnership, -marks for the other. This final reward is
attached after the whole hand completes, even if that seat acted earlier in the
last trick. A partnership's seats receive identical rewards. There is no trick
reward or early-decision shaping.

GAE runs backward separately within each seat/episode, with terminal bootstrap
zero: `delta = reward + gamma * next_value - value` and
`advantage = delta + gamma * lambda * next_advantage`. Targets are advantage plus
old value. It never crosses seats or episodes and does not reset at tricks.
For gamma below one, discounting is **per controlled seat decision**, not per
engine action. Thus intervening opponents' turns do not introduce extra discounts.

The completed trajectories are combined; advantages are normalized over the
batch with a guarded population standard deviation. Each epoch shuffles samples
using a dedicated generator. PPO minimizes the clipped surrogate plus value MSE
minus legal entropy, using detached old log probabilities and targets. The buffer
is cleared after updating. Model parameters actually change through Adam steps.

Optimizer defaults are listed in the [argument reference](command_reference.md#train_ppo-arguments).
All are CLI options (`--gamma`, `--gae-lambda`, `--clip-epsilon`, `--value-coef`,
`--entropy-coef`, `--max-grad-norm`, `--target-kl`). Non-finite losses or gradients
abort training. Logs report sample-weighted optimization diagnostics, including
pre-clipping gradient norm, approximate KL and clipping fraction. Optional KL
stopping skips a minibatch whose measured divergence already exceeds the limit.

## Checkpoints, progress, and reproducibility

`config.json` and `progress.jsonl` contain configuration, sample/hand counts,
optimizer steps, losses, rewards, runtime/throughput, and full validation reports.
`latest.pt` is saved at completed update boundaries after the configured hand
interval and at completion. `best.pt` is selected strictly by mean signed marks
against the designated selection opponent (default HHH) on a fixed validation
schedule, never by training reward.
Evaluation and save intervals can overshoot by up to a rollout because the policy
is not updated mid-rollout. Evaluation also runs at final completion.

Checkpoints contain model/optimizer states, configuration, training counters,
Python/NumPy/Torch CPU/CUDA RNG states, sampling/shuffling generator states,
feature schema, encoder layout, action version, dimensions, ruleset, stage and
opponent, Git SHA/dirty flag, and PyTorch version. Loads check compatibility and
strict parameter shapes. Loading uses explicit CPU mapping and
[`weights_only=True`](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
CUDA-trained parameters can always be evaluated on CPU. Generated runs and `.pt`
files are Git-ignored. Preserve the committed baseline and environment versions
for publication; a dirty working tree is explicitly recorded.

Resume uses the same training configuration, with permitted execution changes described
in [parallel_execution.md](parallel_execution.md), and verifies
the resolved device type, and restores optimizer and RNG state. Exact continuation
is tested at update boundaries. Extending a run whose last rollout was shortened
by its total-hand budget changes the batching relative to an uninterrupted longer
run. CPU inference can move devices; exact training resume cannot change the
sampling generator's device type.

Deal seeds are derived from experiment seed and episode index, separately from
sampling, minibatch and validation streams. Evaluation never trains and greedy
inference does not consume training randomness. Strict CPU tests use deterministic
PyTorch operations and one numerical thread (`--torch-threads` controls this). Sampling uses an
explicit generator via
[`torch.multinomial`](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html).
Exact bitwise results across PyTorch versions, CUDA builds and GPU hardware are
not guaranteed. `--device auto` selects CUDA when available; explicit unavailable
CUDA is rejected. GPU updates can be faster while CPU environment simulation
remains the bottleneck. Rollout records remain on CPU; minibatches move to device.

## Limits and next steps

Stage 1 is hand-only, synchronous and feed-forward, using fixed baseline variants
or a distribution of them. It may exploit those continuation policies, and its observation summary has no recurrent memory.
The critic is decentralized and sees the same public/private observation as the
actor. Winning against HHH is an empirical question, not a smoke-test criterion.

Independent phase training and frozen historical pools are implemented in the
[separate multi-phase trainer](multiphase_selfplay.md). They do not alter this
legacy baseline. Recurrent policies require
sequence batching and memory isolation. Centralized privileged critics would
require a separate explicit data path while retaining ordinary decentralized
execution. Match-level returns need their own lifecycle and GAE design. These extensions are not part of the play-only trainer.

## Parallel simulation

Use `--threads 0` for conservative auto-sized process collection, `--threads N`
for up to N workers, and `--torch-threads 1` for main-process numerical threads.
The sequential default preserves legacy sampling; parallel runs use versioned
per-episode CPU sampling. See [parallel_execution.md](parallel_execution.md) for
checkpoint migration, warm starts, exact resume rules, profiling, and benchmarks.

See [evaluation performance](evaluation_performance.md) for cached cluster bootstrap,
independent `--eval-threads` validation workers, profiling, and readable log/CSV exports.

## Configuring training and evaluation opponents

All six frozen phase variants are supported: HHH, CHH, HHR, HRR, CRR and RRR.
H denotes HeuristicAgent V1, C conservative bidding, and R uniform legal random
choices. See [policy_ablation.md](policy_ablation.md) for their exact policies.
Only the opposing partnership changes: PPO still controls play on its two seats,
with heuristic bidding and trump selection. The learning partnership rotates by
episode exactly as before. Rewards, per-seat GAE and PPO optimization are unchanged.

Fixed HHH training retains its original agent construction and sampling behavior.
Single-opponent and mixture arguments are mutually exclusive. See the
[mixed-opponent recipes](command_reference.md#legacy-mixed-opponent-training)
for HHH, CHH, HHR, mixed training, validation, resume and warm-start commands.

Names must be registered and unique; weights must be finite and strictly positive.
Weights need not sum to one: `HHH:7,CHH:2,HHR:1` is exactly equivalent to the
70/20/10 mixture. Pools are sorted by name and stored as normalized rational strings
(e.g. `7/10`) to preserve exact configuration equivalence on checkpoint reload.
The displayed probabilities are decimals. Empty pools and invalid entries fail
before training. Each hand draws one identity for both opposing seats; it never
switches within a hand. Realized proportions fluctuate.

`OpponentPool` is an immutable, spawn-safe policy specification. Its dedicated
`training-opponent-v1` stream depends only on training seed and episode index;
seat-local baseline RNG seeds use `training-opponent-agent-v1`. Fixed pools skip
sampling altogether. Deal, PPO action, minibatch and evaluation streams are
independent. Different opponents can alter subsequent auctions, legal plays and
action RNG consumption, but the initial deal for each episode stays identical.
Episode-v1 trajectories and updates agree across worker counts on the same CPU
numerical backend. Legacy-v1 retains its existing sequential action stream.

Training against one policy and sampling from a policy distribution are different
procedures. Validation against several policies yields separate labelled reports;
it does not create a blended score. `--eval-opponent` and `--eval-opponents` are
mutually exclusive. Validation uses the same duplicate-deal schedule and bootstrap
seed for every opponent, independently of training. `--selection-opponent` must
belong to that set; one opponent is automatically primary. With multiple opponents,
HHH is primary by default when present; otherwise an explicit choice is required.
`best.pt` still maximizes only that primary opponent's mean signed marks.

Standalone evaluation retains its single-opponent interface and adds an optional
multi-opponent mode:

Single-opponent JSON is unchanged. Multi-opponent JSON has `evaluations` and
`breakdowns` keyed by name, plus the checkpoint identifier. Hand and metric CSVs
include an opponent column. Training keeps existing primary `evaluation` and
`evaluation_performance` progress fields and adds `evaluations` and
`evaluation_performances` mappings. Per-update `training_by_opponent` records
counts, reward sums and means (undefined when zero), alongside the overall reward.
CSV columns expose these diagnostics, seeds, sample sizes, workers and selection
opponent. `rollout_performance` separates opponent assignment, construction,
simulation and main-process merging; worker times are sums, not additive wall
phases. See [evaluation_performance.md](evaluation_performance.md) for log paths.

### Checkpoint continuation and warm starts

Old checkpoints without opponent fields migrate to HHH training, HHH validation
and HHH selection. Their model, optimizer and saved RNG streams remain loadable.
Exact resume requires the same canonical training mixture and validation/selection
configuration. Changed distributions or selection criteria are rejected before
restoring training state, preventing reuse of an incomparable `best_score`.
Evaluation worker count remains an execution-only CLI option.

Match the remaining hyperparameters and sampling scheme for exact resume. A warm
start copies only model weights into a new optimizer, counters and RNG streams;
it is a new experiment. Neither mode changes older checkpoint files. These pools
support registered baseline variants only; frozen learned opponents and historical
snapshots belong to the [multi-phase path](multiphase_selfplay.md). Live self-play remains future work.
Performance against random agents, especially RRR's aggressive bidding, does not
establish competitive strength. Validate on held-out seeds against fixed policies.

Use `examples.opponent_benchmark` from the [command reference](command_reference.md#legacy-mixed-opponent-training)
for warmed collection measurements. The historical measurements below use identical
budgets and several measured updates; no strength claims follow.

### Historical CPU smoke validation and throughput

Four seed-42 smoke runs each completed 256 hands, 3,584 experiences and 14
updates on four rollout workers, with separate HHH/CHH/HHR validation. The
50/50 pool selected CHH for 138 hands and HHH for 118; the 70/20/10 pool
selected CHH for 57, HHH for 172 and HHR for 27. These short runs verify
integration and do not establish competitive strength. Older local HHH
checkpoints loaded and resumed; the next hand matched the original collector
exactly. Warm starting an older HHH network into mixed training and resuming
a mixed run with changed worker counts were also exercised.

Warmed throughput measurements used hidden size 256, seed 42, episode-v1,
one warm-up update followed by three measured 256-hand updates. Each run
trained 1,024 hands in total; the table covers the last 768.

| Pool | 1 worker, hands/s | 4 workers, hands/s |
|---|---:|---:|
| HHH:1 | 199.8 | 671.8 |
| CHH:1 | 203.2 | 612.6 |
| HHH:1,CHH:1 | 194.0 | 643.0 |
| HHH:7,CHH:2,HHR:1 | 188.1 | 627.1 |

Mixed assignment cost about 33–37 microseconds per hand; fixed assignment
about 1 microsecond. Agent construction ranged from about 5 microseconds for
HHH to 46 for CHH. Main-process merging took 6.6–8.3 milliseconds over three
measured updates. These are single-run measurements. Mixture assignment was
less than 1% of summed hand simulation time; overall throughput differences
also include different policies, game trajectories and measurement variability,
so they cannot be attributed solely to configuration overhead. CPU simulations
and PPO optimization remain the main work. No GPU performance claim is made.

## Related workflows

- [Multi-phase PPO](multiphase_selfplay.md): independent bid/trump/play networks and historical opponents.
- [Phase-composable evaluation](phase_composable_evaluation.md): canonical new two-agent experiments.
- [Arena](arena.md): metric definitions and paired statistical methodology.

Legacy `evaluate_many()` still accepts in-memory training models. It shares frozen
snapshot preparation and arena execution with phase-aware evaluation while preserving
`PPOAgent` inference and existing single/multi-opponent report schemas.
