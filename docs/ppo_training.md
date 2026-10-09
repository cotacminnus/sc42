# Stage 1: PPO trick play

The optional `research/ppo` package trains a feed-forward actor/critic using only
`encode(Observation)`. Bidding and trump selection remain the current heuristic
for all four seats. Freezing these phases makes HHH the primary comparison for
trick-play quality and avoids uniform random overbidding confounding results.
No changes are made to rules, actions, observations, or schema-5 records.

## Installation and commands

Python 3.11+ is supported, subject to available PyTorch wheels for your Python
version/platform. Install `python -m pip install -e '.[rl]'`. This extra includes
PyTorch >=2.3 and NumPy >=1.24; ordinary `texas42` imports need neither. Install a
CPU or CUDA PyTorch build appropriate to your machine using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
OpenSpiel is not needed. Linux, WSL2 and Windows use the same Python code.

```bash
python -m examples.train_ppo \
  --total-hands 100000 --rollout-play-steps 4096 \
  --ppo-epochs 4 --minibatch-size 512 --learning-rate 0.0003 \
  --seed 42 --device auto --output-dir runs/ppo-play-v1

python -m examples.evaluate_ppo \
  --checkpoint runs/ppo-play-v1/checkpoints/best.pt \
  --opponent HHH --deals 2048 --seed 12345 --output runs/ppo-play-v1/test.json
```

Repeat evaluation with `--opponent CHH` and `--opponent HHR` for complementary
comparisons. HHH keeps auction and trump rules equal and is the primary metric.
Evaluation uses greedy play, independent seat agents sharing immutable weights,
the unchanged arena's duplicate deals, and group-resampled bootstrap intervals.
Output includes checkpoint SHA256, win rate, signed marks, success/set rates,
sample size, intervals, and runtime. Small smoke runs do not establish strength.
Use final test seeds different from the fixed validation schedule used to select
checkpoints. Repeated checkpoint selection can overfit a validation schedule.

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

`Trainer.collect()` creates a fresh hand environment and four reset heuristics.
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

| Setting | Default |
| --- | --- |
| Adam learning rate | 0.0003 |
| Gamma / GAE lambda | 1.0 / 0.95 |
| PPO clip | 0.2 |
| Value / entropy coefficients | 0.5 / 0.01 |
| Maximum gradient norm | 0.5 |
| Epochs / minibatch | 4 / 512 |
| Rollout play samples | 4096 (rounded up to a complete hand) |
| Optional KL threshold | 0 (disabled) |

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
against frozen HHH on a fixed validation schedule, never by training reward.
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

```bash
python -m examples.train_ppo --resume runs/ppo-play-v1/checkpoints/latest.pt \
  --total-hands 200000 --seed 42 --device auto --output-dir runs/ppo-play-v1
```

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

Stage 1 is hand-only, synchronous, feed-forward, against one frozen opponent.
It may exploit that opponent, and its observation summary has no recurrent memory.
The critic is decentralized and sees the same public/private observation as the
actor. Winning against HHH is an empirical question, not a smoke-test criterion.

Future work can extend the same buffer to trump decisions, then bidding, then
joint phase training with clearly defined returns and masks. Mixed-opponent or
checkpoint-pool training would test generalization. Recurrent policies require
sequence batching and memory isolation. Centralized privileged critics would
require a separate explicit data path while retaining ordinary decentralized
execution. Match-level returns need their own lifecycle and GAE design. None of
these extensions is implemented here.

## Parallel simulation

Use `--threads 0` for conservative auto-sized process collection, `--threads N`
for up to N workers, and `--torch-threads 1` for main-process numerical threads.
The sequential default preserves legacy sampling; parallel runs use versioned
per-episode CPU sampling. See [parallel_execution.md](parallel_execution.md) for
checkpoint migration, warm starts, exact resume rules, profiling, and benchmarks.

See [evaluation performance](evaluation_performance.md) for cached cluster bootstrap,
independent `--eval-threads` validation workers, profiling, and readable log/CSV exports.
