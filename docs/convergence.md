# Convergence diagnostics

How much does a training lineage still change, against whom did it converge, and
does the change make it better or only different? This document defines the
convergence measures, explains why there are several, records what the existing
runs show, and lists the record discrepancies that would otherwise mislead an
analysis. Commands and arguments are in the
[command reference](command_reference.md#convergence-report).

Implementation:
- `research/ppo/convergence.py`: pure metric functions and the record reader.
- `examples/convergence_report.py`: the CLI.
- `tests/test_convergence.py`: the tests.

The report is offline and read-only. It plays no games, and it reads only what a
run recorded:
- `progress.jsonl` (version-1 or version-2 rows);
- `config.json`;
- `snapshots/*.pt`;
- `checkpoints/`.

It never imports the trainers or the opponent pool. A change in how training
opponents are chosen therefore does not change how their records are read.

## 1. Why one number is not enough

The motivating hypothesis has two parts:
- Training against a fixed opponent (a heuristic or a fixed model) converges
  faster.
- Historical self-play converges less, because changing opponents pull toward
  different ideal playstyles.

Rock-paper-scissors shows why "convergence" has to be qualified:
- **Fixed opponent.** Against a fixed "always rock" opponent, a learner converges
  quickly, to "always paper". That is convergence *to its opponent*: swap the
  opponent for scissors and it must start again.
- **Self-play.** The opponent is the learner's own past, so it can chase forever
  (R → P → S → R). Each step beats the previous policy, yet the sequence never
  settles.

Related work points the same way:

| Finding | Source | Consequence here |
| --- | --- | --- |
| **Self-play cycles in non-transitive games.** | Balduzzi et al. 2019, [Open-ended learning in symmetric zero-sum games](https://arxiv.org/abs/1901.08106); Mertikopoulos et al. 2018, [Cycles in adversarial regularized learning](https://arxiv.org/abs/1709.02738) | Compare each policy with the whole history, not just its predecessor: recurrence and cross-play cycles. |
| **Fictitious play: the average policy converges where the last iterate need not.** | Heinrich & Silver 2016, [NFSP](https://arxiv.org/abs/1603.01121) | Drift here is last-iterate drift, and is labelled so. |
| **Real games are transitive skill plus cycles.** | Czarnecki et al. 2020, [Spinning tops](https://arxiv.org/abs/2004.09468) | Keep strength (against a fixed panel) separate from behaviour change. |
| **Averages over agents are biased by redundant copies.** | Balduzzi et al. 2018, [Re-evaluating evaluation](https://arxiv.org/abs/1806.02643); Omidshafiei et al. 2019, [α-Rank](https://arxiv.org/abs/1903.01373) | Keep a fixed, reported panel. Deduplicate checkpoints by weights, not by file. |
| **Agents overfit the opponents they trained with** (joint-policy correlation). | Lanctot et al. 2017, [PSRO](https://arxiv.org/abs/1711.00832) | Report the gap between training and held-out opponents, and compare lineages across independent seeds. |
| **Greedy actions churn even when the value is stable.** | Schaul et al. 2022, [Policy churn](https://arxiv.org/abs/2206.00730) | Report the most-likely-move change rate. Expect a nonzero floor: compare lineages, not against zero. |
| **A learned best response estimates exploitability.** | Timbers et al. 2022, [Approximate exploitability](https://ijcai.org/proceedings/2022/484) | Recipe in section 5. |
| **Cooperative self-play learns private conventions; partners from other runs fail.** | Hu et al. 2020, [Other-Play](https://arxiv.org/abs/2003.02979); Bard et al. 2020, [Hanabi](https://arxiv.org/abs/1902.00506); Tian et al. 2020, [Joint Policy Search (bridge)](https://arxiv.org/abs/2008.06495); Lockhart et al. 2020, [Bridge bidding](https://arxiv.org/abs/2011.14124) | Report bid-phase drift on its own (bids are how partners signal), plus partner divergence and partner cross-play. |
| **Diverse past partners make agents robust.** | Strouse et al. 2021, [Fictitious Co-Play](https://arxiv.org/abs/2110.08176) | Slower convergence is not a defect by itself. Judge it together with held-out results. |
| **Frozen past copies guard against cycle-chasing and forgetting.** | Vinyals et al. 2019, [AlphaStar](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) | The final policy plays its own past (forgetting check). |
| **Behavioural diversity differs from response diversity.** | Liu et al. 2021, [Unifying behavioral and response diversity](https://arxiv.org/abs/2106.04958) | Two layers: drift on probe positions (behaviour) and cross-play outcomes (response). |

So the report communicates four things, one layer each:
1. **Exposure:** what the lineage converged against.
2. **Drift:** how much its behaviour still changes.
3. **Cross-play:** whether that change is progress or cycling.
4. **Strength and generalization:** whether it holds against opponents it did
   not train with.

## 2. Definitions

### Policies of a lineage

| Run kind | Ordered policies |
| --- | --- |
| Version-2, shared | The initial checkpoint (from `initial_checkpoint_path`, verified by SHA) at 0 hands, every `snapshots/policy-*.pt`, then `latest`. |
| Paired | Per partner: initial, best and latest. Paired runs publish no snapshots. |
| Version-1 | Best, then latest. |
| Derived checkpoints (e.g. distillation pilots) | Each compared with its derivation source; siblings are not a time series. |

Consecutive policies with identical weights count once. `best.pt` is a selection,
not a time point, so it joins the sequence only when there are no snapshots.

### Drift (behaviour)

**Probe set.**
- Heuristic V1 plays all four seats for `--probe-hands` hands (seed domain
  `convergence-probe-v1`).
- Every decision with at least two legal moves is kept, separately for bid, trump
  and play.
- No lineage's own play decides what is probed, so lineages are compared on the
  same positions.
- This differs from the in-training forgetting probe
  ([`playing_kl`](multiphase_selfplay.md#forgetting-and-scientific-limitations)),
  which uses the anchor's own play positions and play decisions only.

**Measures.** For each phase that two policies both trained:

| Measure | Definition |
| --- | --- |
| `consecutive_kl` | Mean KL(p<sub>i</sub> ‖ p<sub>i+1</sub>) over probe positions, on legal moves. |
| `consecutive_greedy_change` | Share of probe positions whose most likely move changes from p<sub>i</sub> to p<sub>i+1</sub>. |
| `kl_to_final`, `greedy_change_to_final` | The same, against the last policy. |
| `pairwise_kl` | KL(p<sub>i</sub> ‖ p<sub>j</sub>) for every pair. |
| `recurrent` | Policies (from the third) whose nearest earlier policy, by symmetric KL, is not their immediate predecessor. Counted only when both the step from the predecessor and the predecessor's distance to that earlier policy are at least 0.01 nats. Among near-identical earlier policies, which one is nearest means nothing. |
| Partner divergence (paired runs) | KL in both directions and most-likely-move disagreement between the two partners' final networks. |

**Reading drift.**
- **A converging best response** has decreasing `consecutive_kl`, no recurrence,
  and a most-likely-move change rate that settles at a low floor.
- **A cycle** has steady steps and recurrence.
- Snapshot spacing should be equal when comparing lineages; KL does not scale
  linearly with the number of hands between policies.

### Exposure (what it trained against)

Each training hand is classified by the SHA logged in `opponent_assignments`,
never by its pool name:

| Class | Rule |
| --- | --- |
| `heuristic` | No SHA and a built-in variant name. |
| `anchor` | The run's anchor SHA. |
| `snapshot` | The SHA of a file in `snapshots/` or of a recorded roster entry. |
| `checkpoint` | The SHA of a registered opponent checkpoint or composition. |
| `current` | An unmatched SHA under a dynamic name (`self-current`, `self-snapshots`). The learner's own rollout weights are logged as a payload digest, which never matches a file. |
| `unknown` | Anything else. It is reported, not guessed. |

**Relabelled hands** are hands whose pool name differs from what was actually
played. Example: `self-snapshots` hands that played the anchor before the first
snapshot existed. Version-1 runs record names only; they are all built-in
variants.

### Strength and generalization

**Recorded validations.**
- Each in-training validation records agent A's signed marks with its 95%
  bootstrap interval. The study runs used 128 duplicate deals, 256 games.
- `plateau_hands` is the first validation after which no validation differs from
  the last by more than two combined standard errors (each from its interval).
- It is descriptive. With intervals about ±0.1 marks it comes early and says
  little.

**Generalization gap.** The final validation against training opponents minus the
mean against held-out validation opponents. The absolute gap also reflects how
hard each opponent is, so compare gaps between lineages on the same panel.

### Cross-play (response)

`--crossplay-pipeline` writes an [`examples.run_pipeline`](experiment_pipelines.md)
configuration of duplicate `evaluate_agent_vs_agent` steps. All steps use the same
deals and seed:
- every later policy against every earlier one, within each run;
- every policy against each fixed panel opponent;
- each run's final policy against every other run's;
- mixed partnerships: one run's first-seat network with another run's
  second-seat network, against the first panel opponent.

Comparisons whose two sides have the same weights as an earlier comparison reuse
its step.

After the pipeline runs, `--crossplay-results` reports:

| Result | Meaning |
| --- | --- |
| Transitive progress | Later-vs-earlier pairs significantly better, worse or inconclusive (the 95% interval excludes 0, or does not). |
| Cycles | Policy triples a > b > c > a, each significant. |
| Final vs earlier | The forgetting check. |
| Panel table | Each policy against each panel opponent, with a plateau: more precise than the recorded validations. |
| Partner compatibility | Mixed partnership marks minus the mean of the two runs' own final pairs against the same opponent. Negative means their conventions do not combine. |

## 3. What the existing runs show

**Setup.**
- Probe: 256 Heuristic V1 hands, seed 20261101, giving 1,024 bid, 256 trump and
  4,607 play positions. Trump is learned by no run.
- Every run trained against HHH only, so none tests the self-play half of the
  hypothesis yet (section 5).
- Step KL is the mean over steps, with the range in brackets.

**Drift per run:**

| Run | Spacing | Play step KL | Play moves changed per step | Bid step KL | First → final (play) |
| --- | --- | --- | --- | --- | --- |
| `study-shared` (bid+play, 262,144 hands) | 32,768 | 0.168 [0.140, 0.214] | 15.7% [14.3, 17.5] | 0.042 [0.000, 0.155] | KL 0.651, 29.4% moves |
| `search-p5/control` (131,072 hands from shared best) | 16,384 | 0.107 [0.089, 0.142] | 13.4% [12.0, 15.2] | 0.057 [0.028, 0.102] | KL 0.364, 23.2% |
| `search-p5/search` (65,536 hands, search fraction 1%) | 16,384 | 0.143 [0.133, 0.163] | 16.8% [16.2, 17.4] | 0.031 [0.024, 0.044] | KL 0.373, 25.9% |
| `study-play-v1` (play only), best → latest | 65,248 | 0.530 | 27.7% | — | — |
| `study-paired`, partner 1 / partner 2 (initial → best → latest) | 163,840, then 98,304 | 0.323 / 0.397 | 21.0% / 23.5% | 0.001 / 0.002 (best → latest) | KL 0.509 / 0.623 |
| Distillation pilots, source → distilled | — | 0.005–0.008 | 2.6–3.6% | 0.000 | — |

**Findings:**
- **No HHH-trained lineage converges in behaviour.**
  - Play-step KL and the most-likely-move change rate stay level through training
    (shared 14–17% of probe moves per 32,768 hands; control 12–15% per 16,384).
  - Recorded validation marks plateau within their wide intervals almost
    immediately.
  - So strength stopped changing measurably while behaviour kept changing. This
    is the churn floor of PPO at a constant learning rate, not convergence to a
    best response.
  - This is the baseline a self-play arm must be compared with; zero is not the
    baseline.
- **No recurrence anywhere.** The changes look like a continuing walk, not a
  cycle. That is expected against a fixed opponent.
- **Bidding is much more stable than play.**
  - Shared bidding barely moved between the 32,768 and 163,840 snapshots (step
    KL ≤ 0.006, no most-likely-bid changes), then changed late (up to 0.155).
  - Paired bid networks barely changed between best and latest.
- **Paired partners specialized.** Their final play networks disagree on 30.4%
  of probe moves (KL 0.72), while their bid networks agree (KL 0.001). The
  partnership's conventions live in play, not bidding.
- **Search in training did not change the drift profile** at 1%: the search and
  control runs have similar step sizes.
- **`search-p5/search` and `search-p5/search-balanced` are one lineage.** Every
  snapshot has identical weights, so they are not two samples.

## 4. Record discrepancies

Each is commented at its source and detected by the report.

| Discrepancy | Where | How the report handles it |
| --- | --- | --- |
| `self-snapshots` plays the anchor (the fixed HHH-trained base model) until the first snapshot, but the logged name stays `self-snapshots`. | `HistoricalPool.choose`, `SelfPlayConfig.snapshot_fallback` | Classified by SHA; reported as relabelled hands. With the same seed, a `self-snapshots:1` arm is bit-identical to an `anchor:1` arm until the first snapshot. |
| `self-current` hands log a digest of the rollout weight payload, not a file SHA. | `selfplay_rollout.collect_phase_hand`, `phase_chunk` | Classified as `current`. |
| File SHA is not weight identity: `best.pt` saved in the same update as a snapshot repeats its weights. Seen in `study-shared` best = snapshot 229,376 and `search-p5/control` best = snapshot 114,688. Reruns that differ only in metadata also have equal weights (`search` vs `search-balanced`). | `CheckpointIdentity`; snapshot publication in `train_selfplay_ppo` | `weights_digest`; warnings within and across runs; cross-play steps deduplicated. |
| Paired runs publish no snapshots. | `PairedSelfPlayTrainer.publish_snapshot` | Drift uses initial, best and latest, with a warning. |
| Version-1 progress keeps metrics at the top level and records no opponent SHAs, anchor KL or snapshots. | `examples/train_ppo.py` | Read by name (all built-in variants), with a warning. |
| Distilled checkpoints inherit the source's config, progress and snapshots. Their recorded hands are the source's. | `examples/search_distillation_pilot.py` | `metadata['derivation']` names the source. Compared with the source; hands shown as unknown. |
| The forgetting probe uses the anchor's own play positions and never covers bidding. | `imitation.teacher_data`, `playing_kl` | The neutral probe set covers all phases. |
| `searched.pt` in a pilot directory is a search cache, not a checkpoint. | `search_distillation_pilot` | Skipped with a note. |

## 5. Testing the hypothesis

Three arms continue the shared study model with the exact `search-p5/control`
recipe. Only the training opponents change.

| Arm | Opponents |
| --- | --- |
| A, fixed heuristic | HHH. This is the existing `runs/search-p5/control`. |
| B, fixed model | The anchor (base play-only model, heuristic bid and trump). |
| C, historical self-play | Its own snapshots. Under the current fallback, hands 0–16,383 play the anchor. If the snapshot fallback changes, the report's exposure table shows what was actually played. |

Run B and C one after the other, not beside timing runs. Each takes about the
control's time.

```bash
python -m examples.train_selfplay_ppo --init-checkpoint runs/study-shared/checkpoints/best.pt \
  --anchor-checkpoint runs/study-play-v1/checkpoints/best.pt --learn-phases bid,play \
  --opponent-pool anchor:1 --eval-opponents HHH,CHH,anchor --selection-opponent HHH \
  --snapshot-every-hands 16384 --total-hands 131072 --rollout-decisions 4096 \
  --ppo-epochs 4 --minibatch-size 512 --learning-rate 0.0001 --eval-every-hands 16384 --eval-deals 128 \
  --threads 8 --torch-threads 4 --seed 7 --output-dir runs/convergence/anchor --log-dir logs/experiments/convergence-anchor
# Arm C: the same with --opponent-pool self-snapshots:1 --output-dir runs/convergence/snapshots
```

Then report on the three arms, and generate and run their cross-play (commands
in the [command reference](command_reference.md#convergence-report)). A second
seed per arm separates seed variation from the opponent effect.

**Predictions:**
- A and B show level or falling drift and no recurrence.
- C shows larger or recurrent drift, and possibly cross-play cycles.

The held-out gap and the panel table decide whether faster convergence is
overfitting.

**Approximate exploitability** (Timbers et al. 2022):
1. Train a fresh learner for a fixed budget against one frozen final policy:
   `--opponent-checkpoint target=PATH --opponent-pool target:1 --eval-opponents HHH,target`
   ([Recipe 11](command_reference.md#recipe-11-training-against-an-archived-checkpoint)).
2. Evaluate it against the target.

Its margin is a lower bound on how exploitable the target is, within that budget.
Compare targets with the same budget and seed.

## 6. Limitations

- **The probe is not on-policy.** Heuristic positions differ from each lineage's
  own state distribution, so KL on rarely visited states counts equally.
- **Drift is last-iterate drift.** A self-play pool may be stable while its
  latest member churns.
- **Recorded validations are small** (128 deals). Use the cross-play panel table
  for strength conclusions.
- **One seed per lineage so far.** Differences between single runs are not method
  effects.
