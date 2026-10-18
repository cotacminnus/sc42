# Phase-composable agent evaluation

`examples.evaluate_agent_vs_agent` compares two independently configured policies
through the existing duplicate-deal arena. It changes neither training nor game
rules. Each agent selects bidding, trump and play separately. Defaults are all
heuristic; neural inference is greedy unless `--stochastic-neural` is supplied.

## Strategies and checkpoint resolution

For each of `--agent-a-*` and `--agent-b-*`, bidding accepts `neural`, `heuristic`,
`random` or `conservative`; trump/play accept the first three. Conservative bidding
passes when legal and otherwise chooses the minimum legal bid. Existing agents
implement all non-neural decisions. No checkpoint or PyTorch installation is
needed for purely non-neural comparisons.

Neural phases use `--agent-a-PHASE-checkpoint` before `--agent-a-checkpoint` (and
likewise for B). Missing sources fail clearly. A phase override for a non-neural
strategy is rejected; an unused default checkpoint is ignored. Version-1 PPO
checkpoints support **play only**. Version-2 sources must list the requested phase
in both `neural_phases` and `trained_phases`. Having a 58-output policy head does
not make untrained bidding/trump outputs usable. Schema, ruleset and action
compatibility checks and safe `weights_only=True` loading remain enforced.

## Hold joint-trained playing parameters constant

Run the [same-play bidding ablation](command_reference.md#hold-joint-playing-fixed-change-bidding).

The report must say `Shared playing parameters: YES` and identify bidding as the
only differing phase. This compares neural and heuristic bidding with identical
joint-trained play, rather than substituting the original play-only anchor.
Bidding can change downstream auctions and legal play trajectories.

## Independent checkpoint sources

See [independent-source commands](command_reference.md#independent-checkpoint-sources).

This holds no complete opponent policy constant: it evaluates the specified
compositions. To compare joint PPO against the original play-only PPO, select the
joint checkpoint as A's default with neural bid/play, and the original play-only
checkpoint as B's default with heuristic bid/trump and neural play.

To isolate conservative versus neural bidding with identical play, use the same-play
command and change only `--agent-b-bid conservative`. To evaluate archived
historical policies, substitute their immutable snapshot paths for either agent's
phase sources; trained-phase validation applies equally to archived files.
Neural trump requires a compatible checkpoint recording trained neural trump;
ordinary bid/play checkpoints do not supply it.

The same compositions can now serve as frozen training opponents through
[named JSON registration](command_reference.md#recipe-10-composed-training-opponents).
Training reuses this factory and its per-phase validation/cache; all opponent
parameters remain frozen. For a trump ablation, the learner can freeze the same
joint bidding and playing sources while training trump. Use the
[held-out trump comparison](command_reference.md#held-out-trump-ablation) to verify
that only trump differs. Source selection for a new training stage copies weights;
it never follows a moving snapshot file during training.

## Non-neural comparisons

The [non-neural recipes](command_reference.md#pure-heuristic-conservative-and-random-comparisons)
cover HHH symmetry checks, conservative bidding and random phase substitutions.

## Controlled training comparisons

Use the [held-out phase recipes](command_reference.md#recipe-12-held-out-phase-ablations)
after training; checkpoint selection uses a separate fixed validation schedule.
For bidding-only or trump-only learning, compare with the exact selected frozen
continuations, which may differ from original anchor. Checkpoint provenance alone
cannot establish that continuing playing parameters are identical.

| Learned phase under test | Hold constant | Required shared-network checks |
| --- | --- | --- |
| Bidding | Trump and play routes/sources | Playing YES; trump YES if both neural, otherwise identical chosen strategy |
| Trump | Bidding and play routes/sources | Bidding YES and playing YES |
| Playing | Bidding and trump routes/sources | Bidding/trump YES when both neural; otherwise identical chosen strategies |

Require only the tested phase in differing_phases. To compare two learned versions
of a phase, set both neural and override just B's source for that phase. To compare
with a heuristic, set B's tested phase heuristic and preserve its other sources.
Different legal states and downstream actions are expected after the phase under
test diverges. This controls policy components, not every realized game trajectory.

An all-phase joint model versus HHH varies multiple components. The same model
versus the original play-only anchor can also vary several components, including
playing weights. Those are useful complete-policy comparisons but do not isolate
one learned phase. Pure deterministic self-comparisons are paired symmetry checks.

Canonical phase-composable evaluation requires each selected neural phase to be
marked trained. An initialization/zero-sample
checkpoint containing an untrained routed phase is not a frozen learned source for
that phase. Version 1 can supply play only; version 2 supports its validated trained
subset. Greedy inference is the default across these evaluation commands.
evaluate_selfplay_ppo retains multiple named opponent reports and anchor identity
checks but does not accept the training CLI's opponent-composition JSON flag;
use evaluate_agent_vs_agent for independent phase selections, including archives.
The older checkpoint evaluation commands route neural_phases automatically and
do not impose the canonical per-selected-phase trained check on the learner.
Use them with checkpoints whose routed phases are all trained; use the canonical
command for explicit availability validation and component ablation.

## Reports, reproducibility and interpretation

JSON retains every existing arena field and adds `composition` and `breakdown`.
Readable logs, per-hand CSV and metric CSV go into `--log-dir`, defaulting to a
unique directory under `logs/experiments/`. CSV rows include each phase strategy,
resolved checkpoint path and SHA256. Reports retain contract origin, bid/pass
counts, declarer/defender outcomes, bid values, trump distributions and decision
trick timing. Match mode is optional with `--episode match --target-marks 7`.
The current normal contracts resolve within one hand; decision timing is reported
in completed tricks, not a new early-termination condition.

Parameter checks compare complete phase-network state tensors, including trunk,
policy and value parameters. Different file hashes can contain identical networks;
identical file hashes do not imply identical compositions. Non-neural parameter
comparisons show N/A. `shared_complete_checkpoint` means both agents' neural
sources identify the same single checkpoint; it does not assert that every phase
uses the same strategy. `only_one_phase_differs` compares strategies and actual
neural parameters; it does not compare stochastic realizations.

`CompositePolicySpec`, `PhasePolicySpec`, `CompositeAgentFactory`, and
`evaluate_composed()` in `research/ppo/composition.py` form the reusable API.
`resolve_policy_spec()` shares default/phase checkpoint resolution between the
arena CLI and [console player selection](command_reference.md#neural-console-players).
The console uses these same factories through `AgentPlayer` to combine human
players and independently composed bots, without changing arena methodology.
Factories are spawn-safe. `evaluate_matchups()` shares arena execution with the
legacy in-memory validators. Checkpoint loading/caching lives in `checkpoint.py`;
`opponents.py` owns file identities and named opponent registries. Each distinct checkpoint is verified and cached once
per worker; seats share frozen weights but retain separate delegate state and
RNGs. Reset assigns a seat without rewinding streams. Random streams follow the
existing HybridAgent scheme; stochastic neural streams use separate stable seeds
per phase. No global training RNG is consumed. CPU neural inference uses one
numerical thread; the calling process's thread setting is restored. Existing
play-only, multi-phase and ablation CLIs remain supported. Their argument parsing
and report envelopes remain compatibility adapters: `evaluate_ppo` preserves its
flat single-opponent JSON or named multi-opponent reports; `evaluate_selfplay_ppo`
retains anchor identity checks and archived opponent registration. Both retain
legacy `PPOAgent` inference, rather than changing an archived experiment's RNG
scheme to the explicit per-phase streams of this API.

See [arena methodology](arena.md#comparing-partnerships) for deal/seat coupling and
[uncertainty](arena.md#uncertainty) for complete-group bootstrap definitions.
Changing worker count preserves experimental results; runtime metadata is separate.
An evaluation estimates performance against the chosen continuation policies and
deal distribution. It does not prove globally optimal bidding or Nash equilibrium.
Success rates conditioned on different declared hands are not causal comparisons.

## Historical measured integration check

A local CPU check used the Stage B smoke checkpoint
`runs/multiphase-stage-b-smoke/checkpoints/best.pt` (SHA256
`e22952a2a02ef5d356fa596955f3c320273fbedea3402f2ad32eafa2af673dc8`),
256 duplicate groups, seed 31415926 and 200 bootstrap replicates. A used neural
bid/play; B used heuristic bid and the same neural play. Both used heuristic trump.
Full experimental JSON was identical with one and four workers. Playing tensors
were identical and bidding was the only configuration difference.

| Workers | Preparation | Arena simulation | Total including preparation | Simulation games/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.812 s | 3.225 s | 4.056 s | 158.8 |
| 4 | 0.766 s | 2.125 s | 2.910 s | 241.0 |

A won 50.0% of 512 games and averaged +0.00390625 signed marks per hand
(95% bootstrap CI −0.03916 to +0.04692). Every hand retained 33 decisions and
42 distributed points. This is an integration/performance smoke check using a
briefly trained model, not evidence of bidding improvement. Times are single-run
CPU wall measurements, including process startup; GPU performance was not measured.
Outputs are in `runs/composition-performance-{1,4}.json`; readable logs and CSV
are under `logs/experiments/composition-performance-{1,4}/`.
