# Multi-phase PPO and historical self-play

Use `examples.train_selfplay_ppo` for new experiments that independently train or
freeze bidding, trump and playing, or use frozen learned opponents. The original
play-only `train_ppo` remains a separate reproducible baseline; see
[legacy PPO](ppo_training.md). Neither command trains multi-hand matches.

This guide explains mechanics and interpretation. The
[command reference](command_reference.md#multi-phase-trainer-train_selfplay_ppo) owns every flag,
default, prerequisite and copy-ready recipe. Start with original play training,
or [from random weights](#from-scratch-training-and-current-policy-self-play), then
choose a phase experiment, train against fixed or historical opponents, and
finish with independent [controlled evaluation](phase_composable_evaluation.md).
Multi-stage curricula (for example play → bidding → trump → joint) can be run
from one JSON file with [experiment pipelines](experiment_pipelines.md).
Rules, 387 actor-relative features, 58 global actions, schema-5 data and partnership
signed-mark rewards remain unchanged. Ordinary actors and critics see only their
acting player's immutable observation; neither gets opponent hands.

## Parameters and phase routing

Each phase has its own existing ActorCritic: two ReLU hidden layers of the selected
checkpoint width, a 58-logit policy head, and a signed-mark value head. Learner
phases share the input format, not a trainable trunk. Both seats of the learning
partnership share those parameters but keep separate agent/trajectory state.
A frozen playing trunk therefore cannot change through a bidding gradient.

| Learner state | Rollout behavior | Optimization and observations |
| --- | --- | --- |
| heuristic | HeuristicAgent V1 acts normally | No neural samples or parameter updates |
| neural-frozen | Greedy legal neural continuation | Parameters frozen, excluded from Adam; actions affect the game but not the PPO buffer |
| neural-train | Masked stochastic neural decisions | Features, original legal masks, actions, old log probabilities and values enter PPO |

At least one phase must train. All-frozen agents are valid in evaluation.
Reserved actions stay illegal. Policy sampling, log probabilities and entropy
use the legal distribution; updates retain each sample's original mask. Evaluation
is greedy by default. Training does not stop early when a contract becomes decided.

See [phase omission/conflict rules](command_reference.md#b-learner-phases-and-legacy-compatibility).
Modern unmentioned phases can inherit a frozen neural route from initialization;
spell all three states for an explicit research configuration. Legacy flags retain
the established initialization path and can inherit existing neural continuations
as well. A literal argparse default does not necessarily determine the effective
route. Equivalent routing under different initialization syntax is not promised
to reproduce identical newly initialized parameters or random streams.

## Initialization and freezing

The initial file fixes the learner architecture and identifies the immutable
original version-1 play anchor. A version-2 stage retains that original anchor,
even if its present playing network differs. The anchor has trained play only,
with heuristic bidding/trump when used as an opponent. A primary version-1
initialization must use that same checkpoint as its anchor.

[Source resolution](command_reference.md#c-checkpoint-entry-points-and-neural-sources)
turns init, anchor, a named archive, or a direct file into an absolute path/SHA256.
A trained neural phase copies its complete trunk, policy and critic parameters.
A trainable phase missing from init, or explicitly requesting new, transfers a
trained representation with freshly initialized policy/value heads. It prefers
trained play; otherwise it takes the first trained phase in bid/trump/play order.
Metadata records this as representation initialization, not a pretrained policy
for the new phase. The old play-only model's bidding/trump output rows are untrained.
A frozen phase must have an actually trained source; anchor cannot supply bid/trump.

Independent learner sources must match the initialization hidden width and current
encoder/ruleset/actions. Copying weights creates independent learner modules; it
does not subscribe to a changing checkpoint file. Trained/frozen/neural/learning
metadata controls reuse, rather than the mere presence of tensors in the file.

The CLI snapshots each frozen neural network's state at startup and verifies
bitwise equality after every collection/update cycle. A mismatch aborts. This is
recorded in frozen_phases_verified; heuristic phases have no pretrained-network
verification. A joint frozen playing source can differ from original anchor,
so nonzero anchor-play KL does not imply frozen weights changed.

## From-scratch training and current-policy self-play

`--from-scratch` ([Recipe 14](command_reference.md#recipe-14-from-scratch-training-and-self-play))
trains any nonempty subset of bid/trump/play from random weights in the same
trainer, model and PPO update. Each `neural-train` phase is recorded with the
learner initialization `random`: no source, no transferred trunk, the normal
random trunk and heads of the existing `PhaseActorCritic`, created from a named
`--seed` stream so equal seeds give identical networks. `trained_phases` starts
empty and a phase becomes trained only after an optimizer step updates it, so a
scratch checkpoint never claims an untrained policy and snapshots are published
only once every routed neural phase has trained. There is no anchor: `anchor` is
never registered (requests fail clearly), the play-forgetting KL is reported as
unavailable, and the checkpoint records `initialization: random`, the seed, and
`None` for the initialization checkpoint and original anchor instead of invented
paths or hashes. Resume, evaluation and later stages never need a play-only file.

Five workflows differ only in initialization and opponents:

| Workflow | Initialization | Opponents during training |
| --- | --- | --- |
| Scratch against HHH | Random | Fixed heuristics |
| Scratch current-policy self-play | Random | `self-current`: a frozen copy of the learner's own rollout weights, refreshed after every PPO update |
| Historical snapshot self-play | Random or checkpoint | `self-snapshots`: published, immutable earlier learners; before the first publication, the anchor (anchor-based runs) or the current policy (scratch and anchor-free runs) |
| Checkpoint-initialized PPO | Copied trained phases / representation transfer | Any mixture, including the original anchor |
| New stage vs exact resume | `--init-checkpoint` copies weights only (fresh Adam, counters, roster); `--resume` continues everything exactly | Unchanged on resume; free to change in a new stage |

`self-current` is synchronized self-play: at the start of each rollout the learner
weights are snapshotted once; the learning partnership samples its trainable
decisions from that snapshot as before, and the opposing partnership uses the same
frozen weights with its own seeded action streams (stochastic on trainable phases,
greedy on frozen ones, heuristic phases unchanged). Only the learning partnership's
decisions enter the buffer and receive gradients; the opposing partnership's
weights cannot change during the rollout and are refreshed for the next one. This
optimizes one partnership against a frozen copy of itself, not four simultaneously
updated players. It works from the first hand, when both sides are random. Random
initialization implies no playing strength, and early self-play can be very weak
and slow to improve; keep fixed heuristic validation opponents to measure progress.

## From initialization to the next training stage

The implementation follows this sequence:

1. **Load/validate initialization and anchor.** Check feature/action/ruleset metadata,
   model width, checkpoint version, and original anchor identity. Exact resume
   first reconstructs the saved configuration.
2. **Resolve learner and opponent specifications.** Determine phase states and
   concrete identities; validate requested trained phases and conflicting flags.
3. **Construct independent phase networks.** Transfer whole trained phases or
   representations with new heads. Heuristic routes use the existing agent.
4. **Set freezing and build Adam.** Only enabled neural parameters require
   gradients and enter the optimizer. New stages have fresh optimizer/counters/RNGs;
   resume restores all saved state after compatibility/dependency checks.
5. **Prepare optional diagnostics/initialization.** Bidding imitation, if requested,
   runs before PPO. A fixed anchor-generated play-observation dataset supports
   forgetting diagnostics; neither consumes rollout sampling or shuffle streams.
6. **Build the current frozen opponent pool.** The active snapshot roster is fixed
   for the whole upcoming rollout. Both opposing seats use the same chosen policy
   for a hand, with independent seat agents.
7. **Distribute one fixed learner snapshot.** CPU workers collect independently
   seeded complete hands. The learner partnership alternates by episode/seed;
   weights are fixed until all assigned work returns. Optimizer ownership stays
   in the main process, on CPU or CUDA.
8. **Record enabled learner actions.** Preserve encoded observations, original
   masks, selected global IDs, old log probabilities and values. Opponent and
   frozen/heuristic actions continue normally without becoming PPO samples.
9. **Complete each hand and attach terminal marks.** Each nonempty seat trajectory
   receives its partnership's signed marks on its own last trained action. Empty
   trajectories receive no fictitious action or reward record.
10. **Merge deterministically and calculate GAE.** Sort episodes; calculate advantages
    and returns separately for each seat/hand chronology; combine them into a batch.
11. **Optimize the batch.** Normalize advantages over the collected batch, shuffle
    each epoch, and apply the existing clipped PPO policy loss, value MSE and legal
    entropy with Adam/gradient clipping. Stop early on configured KL, or skip PPO
    entirely if the collected batch is empty.
12. **Record diagnostics and verify freezing.** Store actual phase sample counts,
    aggregate/per-phase optimization diagnostics, opponent rewards and timings.
    A phase is marked trained after its optimization metrics are produced (or
    bidding imitation); the marker does not certify policy strength.
13. **Publish an eligible historical snapshot.** At a completed cycle boundary,
    check the hand interval, nonempty collection and trained status of every
    routed neural phase. Snapshot publication occurs **before validation**. It
    affects subsequent rollouts, never hands already collected.
14. **Validate when due/final.** Greedy in-memory agents use the paired arena's
    fixed separately seeded schedule against each registered validation policy.
15. **Select best.pt.** Save only on a strictly higher primary opponent signed-mark
    mean. Other reports remain separate; ties retain the earlier best.
16. **Save latest.pt when due/final and append progress.** Saved counters include
    publication/validation thresholds, optimizer and RNG state. Progress rows and
    readable output describe the completed cycle; a successful CLI return exports CSV.
17. **Continue, resume or initialize a new stage.** Repeat with updated weights,
    resume the saved state after interruption, or intentionally reset the stage's
    optimizer/counters/roster when changing its experimental configuration.

This is synchronous, on-policy PPO against frozen continuations. Workers neither
update parameters nor share mutable per-seat policy state. There is no asynchronous
stale-policy collection, live learner-versus-learner update, privileged critic,
recurrent policy, phase balancing, or extra trick/early-decision reward.

## Whole-hand collection and per-seat GAE

A normal hand executes 33 engine decisions, seven tricks and distributes 42 points.
Training sample counts depend on which learner decisions are enabled, not the
number of engine transitions:

| Trainable phases | Samples per hand from the learning partnership |
| --- | --- |
| bid | 2 |
| play | 14 |
| trump | 0 or 1 |
| bid,play | 16 |
| bid,trump | 2 or 3 |
| trump,play | 14 or 15 |
| bid,trump,play | 16 or 17 |

Only the winning bidder chooses trump. A defending learner produces no trump
sample; when it declares, only that learner seat contributes one. A frozen trump
selection creates no training sample even if the learner chooses it.

[Rollout controls](command_reference.md#a-budget-and-ppo-optimization) count actual
trainable decisions and finish hands. When trump is enabled, collection advances
in deterministic waves of at most eight hands and checks the accumulated count
between waves. The final wave can exceed the decision target. The total-hand
budget always bounds work; an additional hand cap bounds each trump-only rollout.
An empty capped/final batch advances hand counters and rewards/logging, skips
optimization and snapshot publication, and remains resumable. It can still validate
and save a checkpoint. The current learner checkpoint may still contain an untrained
neural phase; such a phase cannot be used as a frozen opponent/phase source.

A seat's chronology can be BID → TRUMP (if applicable) → seven PLAYs, or just one
BID/TRUMP when other phases are frozen/heuristic. The final signed marks attach
only after the engine completes the hand, regardless of who makes the last engine
move. Earlier trained actions get zero immediate reward. Both partners receive
the same team return on their nonempty trajectories; defenders get the opposite
signed stake. Multi-mark contract stakes remain part of that return.

GAE traverses each nonempty seat sequence backward, using the next trained
observation's value and terminal bootstrap zero. It never crosses seat/hand
boundaries or resets after tricks. Gamma advances once per enabled seat decision;
frozen continuations and opponent turns are not separate discount steps. With
gamma below one, changing enabled phases changes the number of discount steps.
The value target is advantage plus old value, predicting signed marks, not merely
contract success. Advantage normalization is over the merged batch, not per phase.

## Critic modes, returns and oracle auxiliary supervision

Three independently selectable features let their effects be measured separately
([flags](command_reference.md#g-critic-returns-and-oracle-auxiliary),
[A/B/C recipe](command_reference.md#recipe-13-local-versus-centralized-critic-and-oracle-auxiliary),
[policy-only recipe](command_reference.md#recipe-15-policy-only-ppo-and-a-three-way-critic-comparison)).
All defaults reproduce the existing local critic and GAE bit for bit (verified by
comparing rollout actions, values, progress metrics and weights against the previous code).

`--critic-mode` selects one of three training modes for the same actors:

| Mode | Advantage baseline | Value loss | Extra trainable parameters |
| --- | --- | --- | --- |
| `none` (policy-only PPO) | None (zero): advantage = Monte Carlo return | None | None; actor value heads excluded |
| `local` (default) | Each phase network's own value head | Signed-mark MSE on the value heads | Value heads |
| `centralized` | Separate privileged critic | Signed-mark MSE on that critic (+ optional oracle head) | `CentralCritic` |

PPO itself does not require a learned critic: the clipped surrogate only needs an
advantage estimate for each sampled action. A learned value baseline reduces the
variance of that estimate (and GAE trades a little bias for further variance
reduction), which typically improves sample efficiency; without one, the gradient
is unbiased but noisier. Complete-hand Monte Carlo returns can be especially noisy
here because a hand's single terminal signed-mark reward is credited equally to
every decision of the seat: the opening bid, the trump choice and all seven plays
receive the same return, although later plays have much less influence on most
of the outcome variance than the deal itself. Which mode learns better in SC42 is
an empirical question; no ranking is claimed before matched experiments.

**Policy-only PPO (`none`).** Collection is unchanged (same actors, legal masks,
sampling streams and opponents; the actor initialization and the first rollout are
identical across all three modes for one seed). Each completed seat trajectory
receives `G_t = Σ_{k≥t} γ^(k−t) r_k`; with the default γ = 1 every decision gets the
terminal signed marks of that seat's partnership. Stored values are set to exactly
0.0 when the trajectory enters the buffer, so the advantage is `G_t` (a fixed zero
baseline), followed by the usual batch normalization. The loss is the clipped PPO
objective minus the entropy bonus only: there is no value regression and no
critic, the centralized critic is never constructed, and oracle supervision is
rejected (it trains the centralized critic). The actors keep their local value-head
tensors so checkpoints and inference stay compatible, but those heads have
`requires_grad=False`, receive no gradients and are not in the Adam parameter list;
they still produce an (ignored) output during the forward pass. `--value-coef` has no
effect. `none` requires the Monte Carlo estimator and selects it when
`--return-estimator` is omitted; an explicit `gae` is rejected. Progress rows report
`value_loss`, `*_value_loss`, `explained_variance` and `value_mean` as `null`
(unavailable, not zero); return and advantage statistics per phase remain.

**Return audit.** The existing per-seat trajectories already satisfy the reference-
team requirements: both learners belong to one partnership, each trajectory holds
one seat's decisions in one hand, the terminal reward is that seat's team signed
marks (partners equal, opponents opposite), and GAE never crosses seats, hands or
teams. Opponent turns and frozen/heuristic decisions are skipped rather than
discounted, and bid → trump → play are ordinary steps of one seat trajectory, so
a bid value bootstraps from the same seat's (same team's) next value. Hands always
complete inside a rollout, so no trajectory is truncated in training; `gae` and
`monte_carlo` nevertheless accept a same-team `bootstrap_value` for truncated
trajectories, and the buffer requires it whenever the last decision is not terminal.
`gamma` stays 1 by default for undiscounted hand-level signed marks.

**Return estimators.** `gae` is unchanged. `monte-carlo` assigns every decision the
complete discounted hand return (the final signed marks when gamma = 1), with
advantage = return − value: GAE with λ = 1, kept as a diagnostic baseline.

**Centralized critic.** `--critic-mode centralized` adds a separate `CentralCritic`
(`research/ppo/critic.py`): the same two-layer MLP width as the actor, a signed-mark
value head and a separate oracle head. Its input is
`texas42.privileged.encode_privileged` (509 features, layout version 1): the
acting seat's ordinary 387-feature vector (own hand, public bidding and play
history, contract, trump, scores, phase, voids, relative seats), plus all four
remaining hands by relative seat, the relative trick leader and next player, and
an explicit perspective field (the actor's team is always the reference). With the
public record of who played what, the remaining hands determine the full initial
deal, so the critic sees every input the non-recurrent seat policies have seen.
Approximations: opponent identity is not encoded, so values average over the
training opponent mixture; there are no recurrent policies, so no recurrent state
needs privileged isolation.

Actors are unchanged: one shared actor per phase across both learner seats,
separate bid/trump/play networks and all freezing controls. Privileged features are
computed beside each learner decision in the rollout (from
`SelfPlayEnv.remaining_hands()`, a privileged accessor that agents never receive)
and enter only the critic. The actor's own value heads are left untouched in this
mode. The critic predicts expected final signed marks **under the rollout
policies** from actual returns, not a perfect-information value. It has its own
initialization seed stream (actor initialization and rollouts stay matched across
modes), joins the Adam optimizer after the trainable actor parameters, and has its
gradient clipped separately so a large value gradient never rescales the actor's.

**Oracle auxiliary head (optional, off by default).** With `--oracle-aux-coef c > 0`
and an [offline label dataset](command_reference.md#oracle-training-labels), each
optimizer step also samples a batch of labelled positions (separate seed stream)
and adds `c · mean(relu(L − v)² + relu(v − U)²)`, where `v` is the oracle head's
prediction and `[L, U]` the certified double-dummy final-point bounds for the
acting team, all divided by 42. When `L = U` this is squared error. The loss
shapes the shared critic trunk only: main value targets remain actual rollout
returns, advantages are unchanged, no reward shaping or actor imitation is added,
and no search runs during training. Labels come from late-hand positions by
default because they are cheap to prove exactly; one open-hand deal is not a
realistic bidding success probability, so no bid-phase targets are produced.

Checkpoints record `critic_mode`, `return_estimator`, `critic_optimized` (false for
`none`), `initial_critic_mode` (the initialization checkpoint's mode; null from
scratch), the Adam parameter names, the critic weights and architecture, the
privileged layout version and how the critic was initialized; local and older
checkpoints (which record no mode) load unchanged as the local critic with GAE.
Exact resume rejects a different critic mode or effective estimator; a new stage
may change the mode, always with a fresh optimizer, and copies a centralized critic
only from a centralized source. Actor-only inference (`load_model`, evaluation,
console) never needs the critic, a dataset or the oracle.

## Opponent pools, archives and historical self-play

[Opponent arguments and recipes](command_reference.md#d-training-opponents-and-historical-snapshots)
cover six heuristic/random/conservative variants, the original anchor, complete
trained neural checkpoints and independently composed frozen policies. Positive
mixture ratios normalize canonically to sorted rational probabilities. One identity
is selected for the whole hand, with both opposing seats receiving independent
agents. Different strategies can alter auction choices, later legal moves and returns.

A named archive is a fixed policy; self-snapshots is a dynamically growing/bounded
historical roster. Before the roster has an eligible snapshot, its selection falls
back to original anchor, not the current stage's initial joint network; scratch
and anchor-free lineages fall back to the frozen current policy instead. Thereafter
selection is uniform over active members. Snapshots copy the learner at completed
update boundaries, remain frozen, and do not follow later optimizer changes.
The original anchor is separate and never evicted. The roster retains at most its
configured capacity; evicted files remain on disk because old resumes may need them.
Publication may overshoot its threshold and can occur even in HHH-only training.

Starting a [new historical stage](command_reference.md#recipe-6-joint-training-with-historical-opponents)
from an HHH-trained checkpoint copies its trained phase weights, not its historical
roster or Adam state. Using the same root seed is a new experiment, not an exact
continuation; counters and episode streams start again. Retain fixed validation
opponents alongside historical training opponents so metrics remain interpretable.

Composed opponents reuse the existing CompositeAgentFactory; no new phase router
or engine is involved. Neural files are loaded/cached per worker and shared as
frozen weights across separate seat delegates. A complete checkpoint opponent
must have all routed neural phases trained. A composed opponent/source can select
only valid trained phases from several files without enabling their other weights.
JSON specifications and concrete source identities are stored in config/checkpoints;
resume with omitted registration flags does not need the original JSON definition.

## Reproducibility and execution

Episode index/seed determine deals and learning-team assignment. Separate streams
cover learner action sampling, minibatch shuffle, opponent type/seat actions,
historical member selection, teacher data/shuffle, fixed forgetting data, validation
and bootstrap. PID, completion order and wall clock never determine game seeds.
Collected episodes return in deterministic order; fixed snapshots remain unchanged
for the entire rollout. CPU episode-seeded records and updates are tested across
resolved worker counts. Requests are upper bounds and can be clamped by available
CPUs/tasks. Exact bitwise portability across PyTorch/CUDA/hardware versions is not
promised. Preserve Git commit, dirty-worktree state and numerical-library versions.

Enabled learner phases share one learner-action generator within each hand;
changing the trained subset can change later sampling draws. Those categories
are independent of deal/opponent/validation streams, but training does not promise
separate learner RNGs per phase. Composable stochastic evaluation has its own
explicit per-phase streams.

Simulation uses bounded persistent spawn processes, with worker numerical threads
set to one. Inference is CPU-local; optimizer minibatches may move to CUDA in the
main process. Increasing worker count is not guaranteed to improve throughput,
especially with sparse trump decisions, small validation jobs or large neural pools.
[Parallel execution](parallel_execution.md) owns resource detection, profiling and
platform details. No OpenSpiel installation is needed for PPO.

## Checkpoints, resume, and evaluation

Version 1 retains its play-only weights/semantics. Version 2 records independent
phase modules, neural/trained/learning/frozen states, resolved source paths/SHA256,
representation-versus-trained initialization, original anchor identity, optimizer
parameter membership/state, counters, RNGs, opponent distribution/compositions,
exact snapshot roster, selection opponent and all game/encoder compatibility
versions. Loading uses CPU mapping and weights_only=True with strict dimensions.
Original checkpoint files are never rewritten by evaluation or stage initialization.

[Resume recipes](command_reference.md#recipe-7-resume-and-multi-opponent-validation) distinguish
an exact stage continuation from fresh initialization. Omitted modern settings
restore saved values; explicit changes undergo compatibility checks. Budget extension
and worker-count changes are permitted within the same sampling/numerical scheme.
Same optimizer device type is required. Phase/source, architecture, optimization,
opponent and validation/selection changes are new stages. Older version-2 files
without explicit learner specifications retain their legacy configuration path;
use legacy flags for their exact resume. Keep every recorded source/roster file
available and immutable. Extending a budget-truncated final batch is not identical
to training with the larger budget from the outset, because batch boundaries differ.

The validation opponent is independent of the training pool. Multiple opponents
receive distinct reports on identical duplicate schedules; best.pt optimizes only
one primary opponent's mean signed marks/hand. Training always validates the
current in-memory weights without creating a temporary checkpoint. latest.pt is
most recent periodically saved progress, not necessarily highest performance.
An interruption can lose work since that save; progress/log records alone cannot
restore optimizer state. See [artifact/metric reference](command_reference.md#training-artifacts-and-metrics).

Final evaluation uses held-out deal seeds and the
[phase-composable interface](phase_composable_evaluation.md), including actual
parameter equality checks. Multi-opponent anchor/archive reports remain available
through evaluate_selfplay_ppo. ABAB/BABA deal pairing and whole-group bootstrap
remain in the existing arena; see [statistical definitions](arena.md).

## Forgetting and scientific limitations

The fixed anchor-play observation dataset measures KL(anchor || current neural
play). It is diagnostic only, with no KL regularizer or privileged information.
Heuristic play makes this diagnostic unavailable. Optional bidding imitation
uses HHH observations including forced/voluntary states; it initializes behavior,
not optimal bidding. Its temporary Adam optimizer is discarded before PPO.

- **Sample imbalance:** joint batches have two bid, fourteen play, and at most one
  trump sample per hand. PPO losses are sample weighted, without phase-balanced
  minibatches, automatic weights, or per-phase advantage normalization. Phase
  diagnostics reveal effective sample sizes but do not correct them.
- **Sparse trump learning:** learner defenders supply no trump samples. The hand
  cap bounds work, not the probability of collecting useful samples. Tiny single-
  sample batches have zero normalized advantage; entropy/value learning can still
  act, but a successful optimizer call alone is not evidence of an effective policy.
- **Coupled dynamics:** changed bidding changes declared hands/stakes; trump/play
  change their profitability. Comparing success rates on differently selected
  contracts is not a causal bid/pass comparison, nor are phase effects additive.
- **Opponent overfitting:** HHH-only learning can exploit that specific heuristic.
  Hold fixed diverse evaluation policies and final deal seeds; random opponents
  with aggressive bidding are particularly weak measures of general strength.
- **Historical self-play:** snapshot diversity does not guarantee monotonic strength,
  convergence, protection against forgetting or equilibrium behavior. A bounded
  roster and its anchor fallback determine the actual opponent distribution.
- **Selection bias:** the primary opponent and fixed validation schedule determine
  which model becomes best.pt. Repeated selection can overfit that schedule; report
  independent final results and avoid retroactively selecting from held-out tests.

For controlled experiments, [hold continuation networks constant](command_reference.md#recipe-12-held-out-phase-ablations)
and check actual tensor equality for neural-vs-neural phases. Compare multiple
training seeds, sample counts, contracts/roles and uncertainty. Performance against
HHH or historical snapshots is an empirical result under those opponents, not a
proof of global optimality or Nash equilibrium. Short smoke tests verify integration.
Use [troubleshooting](command_reference.md#troubleshooting-ppo-configuration) when a
configuration or resume fails rather than bypassing phase/identity validation.

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

## Paired-partner training

`--paired-partners` (`SelfPlayConfig.partner_mode='paired'`) replaces the one
shared learner network with two, so the partners can develop complementary
habits. Shared mode remains the default and is unchanged.

| Aspect | Paired behaviour |
| --- | --- |
| Seats | Partner 1 plays the learning team's first seat (0 or 1); partner 2 its second (2 or 3). The learning team still alternates by episode, and the shaker rotates, so the seat is not a fixed bidding position. Specialization can only be conventional. |
| Initialization | Both partners start from identical weights: the second trainer repeats the first's seeded initialization. Differences come only from experience. |
| Collection | One rollout. Each learner seat acts and records with its own network; opponents, deals and the per-hand sampling stream are shared. With identical weights, records equal shared mode exactly (tested). |
| Optimization | Each partner runs the unchanged PPO update on its own seat's samples, with its own Adam state and minibatch-shuffle stream. Advantages are normalized per partner batch. |
| Samples | About half per network: about 8 per hand with bid+play, versus 16 for a shared network. `--rollout-decisions` counts both partners together. |
| Validation and best.pt | The pair validates as one team (seat-aware factory). `best.pt` is chosen by the pair's score and saved for both partners. |
| Files | `checkpoints/partner-1/{latest,best}.pt` and `partner-2/...`: ordinary version-2 checkpoints tagged `metadata.pairing`. Each loads alone wherever a checkpoint is accepted. |
| Resume | `--resume` with either partner file restores both after checking they belong together. `partner_mode` is part of the config and cannot change on resume. |
| Not yet supported | Historical self-snapshots (`self-snapshots` is rejected; no snapshots are published), `--critic-mode centralized` (the collector sends one critic, but each partner trainer would own and optimize its own) and `--critic-mode none` (policy-only PPO; untested with two partners, so rejected). Bidding imitation, frozen-phase verification and anchor KL run per partner; progress rows add a `partners` block with per-partner value diagnostics. |
| Supported with main's additions | `self-current` opponents: the current pair plays the opponent team, routed by the same seat rule. `--from-scratch`: both partners draw the same seeded random initialization. `--return-estimator monte-carlo` applies per partner buffer. |

Implementation: `PairedSelfPlayTrainer` in `selfplay_trainer.py` (partner 2 is
an ordinary `SelfPlayTrainer`); `save_pair`, `pair_paths` and `resume_pair` in
`selfplay_checkpoint.py`. For evaluation, see
[two-spec teams](phase_composable_evaluation.md#two-spec-teams-paired-partners-and-cross-play)
and the [recipes](command_reference.md#paired-partner-training-and-evaluation).

### One-seed study: base, shared and paired (262,144 hands)

**Setup.** One training seed (42), 256-wide networks, CPU. Outputs are in
`runs/study-*` (not committed).

| Model | Training | Bid | Trump | Play |
| --- | --- | --- | --- | --- |
| **Base** (`runs/study-play-v1`; also the anchor) | Legacy play-only PPO (`train_ppo`), 262,144 hands against HHH | Heuristic | Heuristic | Neural |
| **Shared** (`runs/study-shared`) | `train_selfplay_ppo`, initialized from Base, bid and play learned, 262,144 hands against HHH, 4,096-decision updates | Neural | Heuristic | Neural |
| **Paired** (`runs/study-paired`) | As Shared, with `--paired-partners`: two networks, about half the samples each | Neural, one network per partner | Heuristic | Neural, one network per partner |

Evaluation is greedy neural inference on held-out duplicate deals (seed 20261019).
The Paired model is always evaluated as its pair (partner 2 in each team's second
seat).

**Against fixed opponents** (2,048 deals, 4,096 games; signed marks/hand, 95% CI):

| Opponent | Shared | Paired |
| --- | --- | --- |
| HHH | +0.104 [+0.077, +0.132] | +0.064 [+0.039, +0.088] |
| CHH | +0.101 [+0.074, +0.127] | +0.062 [+0.037, +0.088] |
| HHR | +0.290 [+0.264, +0.314] | +0.253 [+0.226, +0.277] |
| HRR | +0.431 [+0.406, +0.456] | +0.410 [+0.384, +0.432] |
| CRR | +0.408 [+0.383, +0.433] | +0.387 [+0.362, +0.412] |
| RRR | +1.151 [+1.130, +1.173] | +1.148 [+1.126, +1.172] |
| Head-to-head (Paired vs Shared) | | −0.066 [−0.092, −0.041] |

**Round robin** (1,024 deals per pairing; row agent's signed marks/hand against
the column agent). Plain PIMC uses 32 samples and heuristic bidding.

| Row vs column | Base | Shared | Paired | HHH | RRR | Plain PIMC |
| --- | --- | --- | --- | --- | --- | --- |
| Base | – | −0.070 | −0.037 | +0.031 (n.s.) | +1.144 | −0.337 |
| Shared | +0.070 | – | +0.067 | +0.110 | +1.150 | −0.278 |
| Paired | +0.037 | −0.067 | – | +0.068 | +1.147 | −0.313 |
| HHH | −0.031 (n.s.) | −0.110 | −0.068 | – | +1.192 | −0.297 |
| Plain PIMC | +0.337 | +0.278 | +0.313 | +0.297 | +1.224 | – |

Standings by mean marks/hand against the others:

1. plain PIMC +0.490 (wins all five pairings);
2. Shared +0.224;
3. Paired +0.174;
4. Base +0.146;
5. HHH +0.137;
6. RRR −1.172.

All pairings except Base–HHH are significant.

**Readings:**

- **Learning the bid helps.** Shared improves on Base (+0.070 head-to-head, and
  +0.110 vs +0.031 against HHH).
- **At equal hands, Paired trails Shared** (−0.066). Each paired network sees about
  half the samples, so this is not yet a fair test of complementary partners. It
  needs equal samples per network and several seeds (roadmap step 8).
- **Search beats every trained model.** Plain PIMC beats all three trained models
  by 0.28–0.34 marks/hand. Model-guided PIMC is stronger still
  ([model-guided results](belief.md#model-guided-results)).
- **Caveats.** One seed and `best.pt` selected against the anchor's validation
  schedule. Treat these as evidence about these checkpoints, not about the
  methods.

## Search inside training

`--search-fraction P` adds Monte Carlo search to training. The theory, gates,
pilots and experiment plan are in [search in training](search_training.md); this
section covers operation.

```bash
python -m examples.train_selfplay_ppo --init-checkpoint runs/study-shared/checkpoints/best.pt \
  --learn-phases bid,play --total-hands 65536 --rollout-decisions 4096 --threads 8 \
  --search-fraction 0.01 --search-samples 16 --search-weight 0.25 --search-target cmpo \
  --output-dir runs/search-train --log-dir logs/experiments/search-train
```

| Aspect | Behaviour |
| --- | --- |
| Searched decisions | Each learner play decision with at least two legal moves, with probability P. One draw from a dedicated per-hand stream per learner play decision, so selection is worker-invariant and independent of the action streams. |
| Scheduling | Two passes. Hands record search requests; then every search runs as its own pool task, longest first, and the results are attached to their records. A few long searches therefore never leave the other workers idle. |
| The search | From the learner's own observation: uniform belief over consistent deals, `--search-samples` worlds, every legal move rolled out to the end of the hand with the current networks at all four seats (each seat's own network in paired mode), points objective. The true hidden hands are never used. |
| Effect on the hand | None. Search has its own seeded streams, so actions, records and rewards are bit-identical to a run without search. It only adds `Experience.search_q`. |
| The loss | Per update, targets fixed at θ_old: search advantages Â = Q̂ − Σ π_old Q̂, normalized by their π_old-weighted RMS. Each minibatch adds `--search-weight` × the mean per-state search loss: CMPO's KL(π̄ ‖ π_θ), or the all-action clipped surrogate (`--search-target advantage`). The value loss and GAE are unchanged. |
| Logged | Per update: `searched_states`, `search_loss`, `search_disagreement`, `search_abs_advantage`, `search_advantage_scale` (and `search_target_kl` for CMPO). Per rollout: `rollout_performance.search_seconds` (worker time), `search_wall_seconds` and `searched_decisions`. |
| Cost | Measured at P = 0.01: about 0.38 worker-s per search and about 23 searched states per 4,096-decision update. That is 11.3 ms per hand against 4.6 ms for ordinary training (2.5×), with the search pass at 62% worker efficiency. P = 0.005 costs about 1.75× ([measurement](search_training.md#9-compute-budget)). |
| Checkpoints and resume | Settings are config fields; checkpoints written before search existed load with search off. An explicit `--search-*` flag on `--resume` changes the setting. |
| Not yet supported | Stochastic (sampled) rollouts, searching bid or trump decisions, and leaf-truncated search (no cheap leaf passes the ranking gate). |

Implementation: `research/ppo/search.py` (`TrainingSearch`, `InMemoryGuide`,
`search_values`, `search_update_targets`, `search_minibatch_loss`),
`selfplay_rollout` (`collect_phase_hand(search=...)`, `SearchRequest`,
`search_chunk`, `attach_search_values`), and `SelfPlayTrainer._run_searches` and
`search_objective`. Tests: `tests/test_search_training.py`.

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
