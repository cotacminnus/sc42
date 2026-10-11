# Multi-phase PPO and historical self-play

The new `train_selfplay_ppo` entry point adds independent bidding, playing, and
optional trump actor-critics. The original `train_ppo` and version-1 checkpoints
retain their behavior. This is hand-level PPO, not simultaneous learner-versus-
learner training. Rules, the 387-feature encoder, 58 global actions, schema-5
records, partnership rewards, and duplicate-group bootstrap are unchanged.

## Parameters and phase routing

Each phase uses the existing `ActorCritic`: two ReLU hidden layers (checkpoint
width, normally 256), 58 logits, and a signed-mark value. Playing receives every
pretrained parameter exactly, including its trunk and critic. Bidding and trump
receive copies of the pretrained trunk but freshly initialized policy and value
heads: the original bid/trump output rows were untrained and are not reused.

`--learn-phases bid` freezes every playing parameter; no trunk is shared. It uses
neural bidding, **greedy frozen pretrained play**, and heuristic trump.
`--learn-phases bid,play` enables both independent modules. `play` alone is also
supported. Newly initialized phases and enabled learning decisions sample masked
policies; frozen neural continuations and opponents use greedy actions. All
networks retain global IDs and the authoritative observation masks. Reserved
contracts remain illegal. Ordinary actors and critics see no hidden hands.

`--learn-phases bid,trump,play --trump-policy neural` enables the prepared trump
module. Heuristic trump is the default; trump-only training is rejected. No claim
about effective neural trump training is made by the small integration tests.
`--play-policy frozen|trainable` is optional and must agree with learning phases.

## Stage A: learn bidding, preserve play

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

`anchor` is the immutable original play-only checkpoint with heuristic bid/trump.
Its absolute path and SHA256 are recorded. The width defaults to the initial
checkpoint's width; an explicit incompatible `--hidden-size` is rejected.

`--bid-bc-hands N` optionally initializes bidding with four epochs of masked
cross-entropy on HHH engine trajectories, using a separate data/shuffle stream.
Zero (default) disables imitation. Logs include loss, accuracy, sample counts,
and forced/voluntary accuracy separately; absent categories are unavailable.
Playing is untouched. This temporary Adam optimizer is discarded before PPO.
Imitating HHH is initialization, not evidence of optimal bidding.

## Stage B: joint training

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

`--init-checkpoint` transfers all phase weights into a **new experiment** with
fresh optimizer, counters, and RNG streams. It does not import an old snapshot
roster. It retains the original play anchor. Enabled phases become trainable;
other existing neural phases remain frozen. Existing heuristic variants are
available, and `--opponent-checkpoint historical=PATH` registers an additional
frozen policy usable as `historical:WEIGHT` in the pool and validation set.

## Whole-hand collection and GAE

A deterministic alternating partnership learns; both seats share phase weights
and retain separate chronological trajectories. Bid-only hands contribute two
samples. Bid/play hands contribute sixteen. A trainable trump adds one when the
learning team declares. `--rollout-decisions` counts actual enabled decisions,
always finishing hands and respecting `--total-hands`. Variable-trump collection
uses deterministic waves of at most eight hands, so a final wave can overshoot
the decision target. Boundaries do not depend on worker scheduling.

Each seat's last trained action receives the engine's final signed marks once,
even if that action was its auction call. Earlier rewards are zero. GAE traverses
that seat's entire enabled sequence, with terminal bootstrap zero. Gamma advances
per enabled seat decision, not per trick or opponent turn. No trajectories cross
seat/hand boundaries. All hands execute 33 decisions, seven tricks, and 42 points.

The existing clipped objective, batch advantage normalization, value loss,
entropy bonus, Adam, and gradient clipping are reused unchanged. Weighting is
ordinary per sample: playing therefore has more weight in joint batches. Logs
also report each phase's sample count, loss, entropy, approximate KL, and clipping
fraction; there is no implicit phase balancing or reward shaping.

## Frozen opponent pool and reproducibility

Mixture ratios normalize canonically. Opponent type, historical membership choice,
agent actions, deals, learning samples, shuffle, imitation, and evaluation use
separate versioned streams. An episode's opponent is fixed for the whole hand.
`self-snapshots` chooses uniformly from the active historical roster, falling back
to the anchor until the first snapshot. Snapshot publication occurs at update
boundaries; interval thresholds may overshoot by one rollout.

CPU workers cache frozen learner snapshots and neural opponents. Cached opponents
are bounded by the active roster and explicit policy registry. Each seat has an
independent agent; immutable parameters are shared locally. The main process alone
updates parameters, potentially on CUDA. Spawn pools persist across rollouts.
Worker-count changes preserve episode-seeded CPU records and updates. Exact GPU
reproducibility across builds/hardware is not promised.

The active roster holds at most `--snapshot-pool-size` policies. Original anchor
membership is separate. Published files under `runs/<run>/snapshots/` are immutable;
evicted files remain archived because older resumes may reference them. Archive
disk usage can grow even though active membership and worker caches are bounded.
Do not remove files needed by a checkpoint you intend to resume.

## Checkpoints, resume, and evaluation

Version-2 checkpoints record independent modules, neural/trained/learning/frozen
phase metadata, original anchor identity, architecture, optimizer, counters,
all existing RNG states, exact snapshot roster, opponent distribution, primary
selection opponent, compatibility versions, Git SHA, and dirty-worktree flag.
Version-1 inference remains supported. Migration initializes a new stage, never
pretending its previously untrained bidding outputs were learned.

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

Resume restores omitted options from the checkpoint. Explicit changes to phases,
architecture, optimizer settings, opponent identities, validation procedure, or
selection criterion are rejected. Budget extension and worker changes are allowed;
optimizer device changes are not exact continuations. Use initialization for new
stages. Checkpoint dependencies must remain present at their recorded paths.

`--eval-opponents HHH,CHH,anchor --selection-opponent anchor` evaluates each policy
on identical paired schedules and selects `best.pt` by the primary opponent's
mean signed marks, never an unlabeled average. Existing `evaluate_ppo --opponent
HHH` also handles version-2 policies. Explicit historical policies are available
through the new evaluation CLI's `--opponent-checkpoint NAME=PATH`.

Progress JSONL retains primary `evaluation` and adds separate reports, identities,
resolved-contract diagnostics, bid/pass counts, and declarer/defender outcomes.
Readable logs and CSV live under `logs/experiments/` or `--log-dir`. Contract timing
reports completed-trick certainty; hand mode always resolves within one hand.

## Forgetting and research interpretation

A fixed anchor-generated public observation dataset reports KL(anchor play ||
current play), with `--forgetting-hands` controlling its size. Bidding-only weights
must remain bitwise equal to the anchor; joint updates can change them. No KL
regularizer is implemented; no arbitrary coefficient is silently introduced.

For Stage A, compare learned bidding versus **heuristic bidding with the exact
same playing weights** (anchor). Use multiple training seeds, a fixed validation
schedule only for selection, then independent large held-out duplicate schedules.
Report intervals, contract values, bid origin, auction frequencies, roles, and
both SHA256 identities. Different auctions induce different downstream states;
conditional contract success rates are not causal bidding comparisons.

Joint historical self-play can overfit its finite opponent pool. Short smoke tests
establish mechanics, not stronger bidding, generalization, equilibrium, or Nash
optimality. Recurrent memory, live self-play, privileged critics, and opponent-pool
curricula remain future research. Benchmark warmed costs with
`python -m examples.multiphase_benchmark --hands 64 --workers 1 4`.

## Local smoke validation and measured performance

The implementation was exercised with the available 256-wide play checkpoint
(SHA256 `82810b346ecc93cf0ccc34494637fcc23182397ac6196880f298d557f2949a6c`).
Stage A ran 128 hands / 256 bidding samples / four updates, including 64 imitation
hands. Its playing parameters remained bitwise equal to the original. Stage B
ran 128 hands / 2048 samples / four updates with historical opponents; its active
roster retained the last two snapshots. A continuation restored that roster,
changed rollout workers from four to two, and reached 160 hands. Deterministic
unit fixtures compare exact records, advantages, updates, and parameters across
worker counts and across restored historical rounds.

A small independent 32-group/64-game Stage A evaluation against the same-play
anchor returned win rate 46.875% and signed marks -0.0625 per hand (95% paired
intervals approximately 42.2–52.4% and -0.156–0.048). This is integration evidence
and **does not establish improved bidding**.

A separate CPU benchmark measured one warmed 64-hand batch after a 64-hand
warmup, two PPO epochs, 64-sample minibatches, and 256-wide networks:

| Configuration | 1 worker, hands/s | 4 workers, hands/s |
| --- | ---: | ---: |
| Existing play-only PPO | 145.9 | 365.7 |
| Bidding-only, frozen neural play | 108.8 | 273.0 |
| Joint bid/play versus HHH | 96.4 | 179.5 |
| Joint bid/play versus historical neural policy | 55.8 | 132.4 |

Four-worker optimization wall times were respectively 0.133, 0.027, 0.298, and
0.251 seconds. Newly loaded historical opponents cost 0.136 seconds summed
across workers. Inference and load measurements are **summed worker durations**,
not additive wall-time phases. Historical neural play requires more inference
than heuristic opponents; these are different workloads, not pure framework
speedup comparisons. Single-batch timings are noisy and exclude warmed process
startup. CUDA was unavailable, so no GPU throughput or utilization is claimed.
Machine-readable measurements and readable log/CSV exports are in
`runs/multiphase-final-profile.json` and `logs/experiments/multiphase-final-profile/`.

The CPU host was an Intel i5-12600KF. Four-worker RSS samples were approximately
311–316 MiB per simulation process (sampled at hand boundaries, not continuous
peak-memory monitoring). Original rollout instrumentation does not report neural
inference/RSS separately, so those legacy baseline fields are unavailable rather
than estimated. Snapshot SHA256 covers the entire file: execution provenance can
change a file's hash even when model tensors and future sampled decisions agree.
