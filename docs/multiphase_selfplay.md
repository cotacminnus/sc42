# Multi-phase PPO and historical self-play

The current `train_selfplay_ppo` entry point adds independent bidding, playing, and
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

`--learn-phases bid,trump,play` enables all three modules; `trump` alone is supported.
No claim about effective neural trump training is made by the small integration tests.
`--play-policy frozen|trainable` is optional and must agree with learning phases.

For independent sources and routing, each learner phase can be `heuristic`,
`neural-frozen`, or `neural-train`. Frozen neural phases act greedily and are excluded
from Adam; trainable phases sample masked actions and enter PPO. `LearnerPhaseSpec`
records initialization status, immutable source identity, and trainability because
an evaluation `PhasePolicySpec` alone has no optimizer/initialization semantics.
Both paths use the same phase networks, rollout function, and PPO optimizer.
Legacy construction retains its original initialization RNG sequence; resolved
phase metadata describes both legacy and explicit configurations.

See [independent-phase commands](command_reference.md#independent-learner-phases-and-trump-only-training).
Sources resolve once at stage initialization. A trained phase copies the complete
actor/critic; a new trainable phase transfers only a trained trunk and receives
fresh heads. Frozen phases cannot use untrained sources. All sources must match
the stage's hidden width. The play-only anchor cannot supply neural bidding/trump.
An archived initialization never changes automatically when later snapshots appear.

## Stage A: learn bidding, preserve play

Use the [Stage A command](command_reference.md#stage-a-frozen-play-learned-bidding).

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

Use the [Stage B command](command_reference.md#stage-b-joint-training-with-historical-opponents).

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

Trump-only hands yield zero or one sample: only the learner's winning bidder has
a trajectory. Empty seat trajectories receive no rewards and do not enter GAE.
The collector stops at the actual decision target, total-hand budget, or
`max_rollout_hands` safeguard (4096 by default). An empty batch skips optimization,
advances progress, and remains resumable. No auction is altered to create samples.
Frozen neural parameters are checked bitwise after every rollout/update.

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

[Composed opponent registration](command_reference.md#register-composed-training-opponents)
reuses `CompositePolicySpec` and `CompositeAgentFactory` for independent frozen
phase sources. A `PolicySpec` stores the resolved composition and its digest;
each underlying checkpoint retains its own SHA256. Registry-only archives can
supply one trained phase without enabling their other networks. Complete frozen
checkpoint opponents must have every routed neural phase trained. Snapshot
publication waits until all routed phases have received training; empty batches
never publish snapshots. Source files are immutable dependencies of exact resume.

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

See [resume and validation commands](command_reference.md#resume-and-multi-opponent-validation).

Version-2 checkpoints record independent modules, neural/trained/learning/frozen
phase metadata, original anchor identity, architecture, optimizer, counters,
all existing RNG states, exact snapshot roster, opponent distribution, primary
selection opponent, compatibility versions, Git SHA, and dirty-worktree flag.
Optional version-2 fields now record resolved learner source/trainability,
initial trained-versus-representation status, named compositions, and optimizer
parameter names. Older version-2 files load with legacy initialization semantics.
Version-1 inference remains supported. Migration initializes a new stage, never
pretending its previously untrained bidding outputs were learned.

Resume restores omitted options from the checkpoint. Explicit changes to phases,
architecture, optimizer settings, opponent identities, validation procedure, or
selection criterion are rejected. Budget extension and worker changes are allowed;
optimizer device changes are not exact continuations. Use initialization for new
stages. Checkpoint dependencies must remain present at their recorded paths.

`--eval-opponents HHH,CHH,anchor --selection-opponent anchor` evaluates each policy
on identical paired schedules and selects `best.pt` by the primary opponent's
mean signed marks, never an unlabeled average. Existing `evaluate_ppo --opponent
HHH` also handles version-2 policies. Explicit historical policies are available
through `evaluate_selfplay_ppo`'s `--opponent-checkpoint NAME=PATH`.

Progress JSONL retains primary `evaluation` and adds separate reports, identities,
resolved-contract diagnostics, bid/pass counts, and declarer/defender outcomes.
Readable logs and CSV live under `logs/experiments/` or `--log-dir`. Contract timing
reports completed-trick certainty; hand mode always resolves within one hand.

## Forgetting and research interpretation

A fixed anchor-generated public observation dataset reports KL(anchor play ||
current play), with `--forgetting-hands` controlling its size. With heuristic play
the diagnostic is unavailable. Legacy Stage A playing weights stay equal to the
anchor; explicit configurations freeze their selected playing source, which may
be a different joint checkpoint. Joint updates can change those weights. No KL
regularizer is implemented; no arbitrary coefficient is silently introduced.

For Stage A, compare learned bidding versus **heuristic bidding with the exact
same playing weights** (anchor). Use multiple training seeds, a fixed validation
schedule only for selection, then independent large held-out duplicate schedules.
Report intervals, contract values, bid origin, auction frequencies, roles, and
both SHA256 identities. Different auctions induce different downstream states;
conditional contract success rates are not causal bidding comparisons.

For trump-only experiments, compare with the same frozen joint bidding/playing
networks and heuristic trump. The [held-out command](command_reference.md#held-out-trump-ablation)
checks actual parameter equality. Use independent final deals after selecting
checkpoints on the fixed validation schedule; a short smoke run tests mechanics.

Joint historical self-play can overfit its finite opponent pool. Short smoke tests
establish mechanics, not stronger bidding, generalization, equilibrium, or Nash
optimality. Recurrent memory, live self-play, privileged critics, and opponent-pool
curricula remain future research. Benchmark warmed costs with
the [multi-phase benchmark command](command_reference.md#multi-phase-arguments).

## Historical smoke validation and measured performance

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

## Evaluation ownership

[Phase-composable evaluation](phase_composable_evaluation.md) is the canonical
system for new agent comparisons, including holding the joint playing network
constant while varying bidding. `evaluate_selfplay_ppo` remains the compatibility
command for multi-opponent anchor/archive reports. Training validates in-memory
models through the same shared snapshot factory and arena execution; it does not
write a temporary checkpoint for evaluation.

Checkpoint serialization and RNG restoration are shared with the legacy trainer
in `checkpoint.py`; phase metadata and roster validation remain in
`selfplay_checkpoint.py`. The two collectors remain distinct because their sample
counts, RNG schemes and enabled-phase trajectories have different semantics.
