# Phase-composable agent evaluation

`examples.evaluate_agent_vs_agent` compares two independently configured policies
through the existing duplicate-deal arena. It changes neither training nor game
rules. Each agent selects bidding, trump and play separately. Defaults are all
heuristic; neural inference is greedy unless `--stochastic-neural` is supplied.

## Agent catalogue

**The common interface.** Every agent implements `reset(seat)` and
`act(observation)`, which returns a permanent action ID 0–57. It sees only its own
`Observation`: its hand plus the public history (calls, contract, trump, played
dominoes, current trick, score). The engine supplies the legal mask. A hand asks
each seat for:

- one auction call;
- a trump choice (auction winner only);
- seven plays.

Agents are compositions of three phase delegates: bid, trump and play.

### Building blocks

| Component | Code | What it does |
| --- | --- | --- |
| Heuristic V1 | `texas42/agents/simpleheuristic.py` | Deterministic rules, described below |
| Random / conservative | `texas42/agents/random.py`, `conservative_random.py` | Random: uniform over legal moves. Conservative: pass when allowed, otherwise the minimum bid; uniform trump and play. |
| Neural phase network | `research/ppo/model.py`, loaded by `composition.phase_network` | The 387-feature encoding (`texas42/encoding.py`) feeds two 256-unit ReLU layers. A policy head gives 58 logits, legal-masked; evaluation plays the argmax. A value head predicts the actor team's final signed marks (PPO's baseline). Version-2 checkpoints hold one network per phase, and only the trained ones are used. Version 1 is play only. |
| PIMC search | `texas42/agents/pimc.py`, `texas42/belief.py` | For trump and play, described below. Bidding is always delegated to Heuristic V1. |

**Heuristic V1:**

- **Bid.**
  - Take the "best suit": most tiles, then highest rank, then count points.
  - Classify the hand from that suit's length and its counters:
    - 7 tiles: 5 marks;
    - 6 tiles and ≥5 counters: 2 marks;
    - 5 tiles and ≥5 counters: 37;
    - 6 tiles: 32;
    - 5 tiles: 30;
    - otherwise pass.
  - Open at the bid, or raise minimally while it is affordable; if forced, bid
    the minimum.
- **Trump.** The same best suit.
- **Play.**
  - Lead the strongest domino, trumps first.
  - If the partner is winning safely, give the most count.
  - If the partner is winning but could be overtaken, discard cheaply.
  - If an opponent is winning, win with the weakest winning domino when last to
    play or when the trick holds ≥5 counters.
  - Otherwise discard the cheapest domino.

**PIMC search**, per decision:

1. Build a belief from hard evidence (own hand, played dominoes, hand sizes,
   revealed voids). Count the consistent deals exactly; enumerate them when there
   are few, otherwise sample `--pimc-samples` of them uniformly.
2. In each deal, play every legal move and then roll the hand out with a rollout
   policy for all seats. Every move uses the same deals and seeds.
3. Pick the highest mean points.

Forced moves skip the search.

### Assembled agents

All flags are for agent A; agent B takes the same flags with `-b-`. Checkpoint
paths are the study runs' (`runs/` is not committed). Substitute your own.

| Agent | Bid | Trump | Play | `evaluate_agent_vs_agent` flags |
| --- | --- | --- | --- | --- |
| HHH (Heuristic V1) | Heuristic | Heuristic | Heuristic | none (the default) |
| HHR / HRR | Heuristic | Heuristic / random | Random | `--agent-a-play random` (HHR); add `--agent-a-trump random` for HRR |
| CHH / CRR | Conservative | Heuristic / random | Heuristic / random | `--agent-a-bid conservative` (CHH); add `--agent-a-trump random --agent-a-play random` for CRR |
| RRR | Random | Random | Random | `--agent-a-bid random --agent-a-trump random --agent-a-play random` |
| Plain PIMC | Heuristic | PIMC, HHH rollouts | PIMC, HHH rollouts | `--agent-a-trump pimc --agent-a-play pimc` |
| Base model (play-only, version 1; the anchor) | Heuristic | Heuristic | Neural | `--agent-a-checkpoint runs/study-play-v1/checkpoints/best.pt --agent-a-play neural` |
| Shared model (bid+play) | Neural | Heuristic | Neural | `--agent-a-checkpoint runs/study-shared/checkpoints/best.pt --agent-a-bid neural --agent-a-play neural` |
| Paired models (two networks) | Neural, own per partner | Heuristic | Neural, own per partner | as Shared with `--agent-a-checkpoint .../partner-1/best.pt --agent-a-partner-checkpoint .../partner-2/best.pt` |
| Distilled pilots (play network fine-tuned toward search) | Neural (as source) | Heuristic | Neural | as Shared with `--agent-a-checkpoint runs/model-guided/pilot/round2/distilled-c2.pt` ([pilots](search_training.md#62-one-step-distillation-pilot)) |
| Model-guided PIMC | Heuristic | PIMC, model rollouts | PIMC, model rollouts | `--agent-a-trump pimc --agent-a-play pimc --agent-a-pimc-model runs/study-shared/checkpoints/best.pt` |
| Model-guided PIMC with a value leaf | Heuristic | PIMC, short rollouts + value head | Same | add `--pimc-leaf value` or `value-turn`. Measured weaker; see [the ranking gate](search_training.md#6-evidence-the-move-ranking-gate). |

**How hidden hands are handled:**

- Heuristic and neural agents decide from the current observation alone. A
  network can only learn implicit beliefs from the encoded history, such as
  revealed voids and who played what.
- PIMC agents reason explicitly over concrete hypothetical deals.
- Model-guided PIMC also models how everyone continues by using the trained
  model. Optionally, `--pimc-likelihood model` weights deals by how well they
  explain the observed moves.

**Measured strength:** plain PIMC beats every trained model (study round robin;
see [the study](multiphase_selfplay.md#one-seed-study-base-shared-and-paired-262144-hands)).
Model-guided PIMC beats plain PIMC
([model-guided results](belief.md#model-guided-results)).

## Strategies and checkpoint resolution

For each of `--agent-a-*` and `--agent-b-*`, bidding accepts `neural`, `heuristic`,
`random` or `conservative`; trump/play accept `neural`, `heuristic`, `random` or
`pimc`. Conservative bidding passes when legal and otherwise chooses the minimum
legal bid. Existing agents implement all non-neural decisions. No checkpoint or
PyTorch installation is needed for purely non-neural comparisons.

`pimc` is the belief-based Monte Carlo agent (`texas42.agents.pimc.PIMCAgent`). It
samples hidden deals consistent with the actor's observation, tries every legal
move in each with Heuristic V1 rollouts, and picks the best expected points; see
[belief modelling](belief.md). It is not offered for bidding, where expected points
is a poor objective. `--pimc-samples` (default 32) sets sampled deals per decision;
it is recorded in the JSON `composition` block only when PIMC is used, and is a
run setting rather than part of the stored policy identity. PIMC costs roughly
0.1–2 s per decision, so prefer `--deals 64`–`128` with `--threads` rather than the
2048-deal default.

**Model-guided PIMC.** `--agent-X-pimc-model PATH` gives that agent's `pimc`
phases a trained checkpoint to guide the search. The model plays the simulated
continuations for every seat, with its trained phases neural and the rest
heuristic. Two run-level options apply to model-guided phases only; plain `pimc`
phases ignore them:
- `--pimc-likelihood model` weights hidden deals by the model's move probabilities.
- `--pimc-leaf value` stops rollouts after the next trick and scores the rest with
  the model's value head, using signed marks as the objective. `--pimc-leaf
  value-turn` stops at the searcher's own next decision instead.
  - Both need a guide trained with the local critic, whose value heads are
    fitted.
  - Both rank moves worse than full rollouts, so prefer the default (`none`) for
    strength; see [the ranking gate](search_training.md#6-evidence-the-move-ranking-gate).

The guide is stored as that phase's checkpoint in the composition JSON, and the
run options as `pimc_leaf`, `pimc_likelihood` and `pimc_epsilon`. A default
`--agent-X-checkpoint` never turns plain PIMC into model-guided PIMC. Method,
costs and measured accuracy are in [belief modelling](belief.md#model-guided-pimc).

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

## Two-spec teams: paired partners and cross-play

By default one composition plays both seats of a team. Any
`--agent-X-partner-*` flag gives the team's second seat (2 or 3) its own
composition, while the first composition keeps seat 0 or 1. The arena seats
agent A at 0/2 and then 1/3, so the first composition always plays the team's
lower seat. This matches paired-partner training, where partner 1 is that seat.

- **Paired models:** `--agent-a-checkpoint .../partner-1/best.pt --agent-a-partner-checkpoint .../partner-2/best.pt`.
- **Cross-play:** a trained model beside a heuristic partner (`--agent-a-partner-play heuristic`), or models from different runs or seeds side by side (roadmap step 8).
- **Inheritance:** unset partner phases inherit the first seat's strategy. Neural partner phases use partner checkpoints first, then the first seat's sources.
- **Output:** the console reports whether the partners share neural parameters. JSON gains `partners`, `partner_comparisons` and `partner_seat_rule` only when a partner is configured.

`research.ppo.composition.PairedAgentFactory` implements the seat rule for any
arena factories, and training validation uses it for in-memory paired models.

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
