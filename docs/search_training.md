# Search inside training: feasibility study

**Status:** assessment and design only. No training code implements this yet. This
document helps reviewers decide whether, and how, to make Monte Carlo search part
of PPO training rather than only a decision-time tool. Its building blocks (belief
worlds, model-guided rollouts, value leaves) already exist and are documented in
[belief modelling](belief.md#model-guided-pimc).

**Verdict:** feasible, with a clear and low-risk integration path.

- **Integration path:** use search as a source of lower-variance, all-action
  advantages inside PPO's clipped objective (section 5), on a small fraction of
  play decisions.
- **Expected cost:** about 1.3–2.5× current training cost, measured (section 6).
- **One prerequisite is unmet.** On the current study model, the cheap value-leaf
  search ranks moves poorly. It lost to plain PIMC, while full model rollouts
  beat it. So either use full rollouts at a very small fraction (about 1%), or
  first build and verify a value leaf that ranks moves well (section 9).
- **Next step:** do not implement training integration until that check passes.
- **Decision-time model-guided search** (full rollouts) is already the strongest
  agent measured here: +0.34 marks/hand over its own guide model.

## Contents

1. [Question and success criteria](#1-question-and-success-criteria)
2. [Why search could improve training](#2-why-search-could-improve-training)
3. [How it could fail](#3-how-it-could-fail)
4. [Options compared](#4-options-compared)
5. [Recommended design: search advantages inside PPO](#5-recommended-design-search-advantages-inside-ppo)
6. [Compute budget](#6-compute-budget)
7. [Implementation sketch](#7-implementation-sketch)
8. [Experiment protocol](#8-experiment-protocol)
9. [Open questions](#9-open-questions)
10. [References](#10-references)

## 1. Question and success criteria

Can Monte Carlo search be folded into training so that the network improves faster
or further than with PPO alone, without collapsing or giving no benefit?

Success means all of the following:

- **Strength at equal compute.** Better held-out results than the plain PPO
  baseline at **equal CPU time**, not only at equal hands, against HHH, the shared
  baseline and the anchor, over at least 3 training seeds. Search makes each hand
  more expensive, so equal-hands comparisons flatter it.
- **Stability.** No entropy collapse and no growing KL to the anchor (the existing
  forgetting diagnostic). No drop in validation after an initial gain.
- **Affordability.** Training throughput stays within an agreed budget, for
  example no worse than 2× slower.

## 2. Why search could improve training

**Policy iteration.** A search that plays each legal move and then continues with
the current network π estimates Q<sub>π</sub>(s, a). Choosing among those values
is one step of policy improvement at that state. Training the network toward the
search's choice distills that improvement. Repeating the cycle is
**Expert Iteration** (ExIt; Anthony, Tian & Barber 2017). AlphaZero uses the same
loop, with search visit counts as policy targets.

**Lower-variance advantages.** PPO's GAE advantage for a decision comes from one
noisy trajectory: one deal, one future. The TRPO "vine" procedure (Schulman et al.
2015) instead branches every action from the same state with **common random
numbers**, giving much lower-variance advantage estimates. `texas42.belief.evaluate_actions`
already does exactly this: same worlds and same rollout seed for every legal move.

**Small action sets help.** A Texas 42 play decision has at most 7 legal moves, so
search can evaluate **all** of them. Danihelka et al. (2022) show that AlphaZero
can fail to improve its policy when search does not visit every root action. That
failure mode does not arise here.

**Search as a regularized update.** Sokota et al. (2023) show that decision-time
planning in imperfect-information games can be viewed as one step of a policy-update
algorithm (mirror descent, or magnetic mirror descent in adversarial games). This
supports training on search outputs as long as the update is **regularized**,
staying close to the current policy or a "magnet" policy, which is also the key to
stability (section 3).

## 3. How it could fail

| Failure | Mechanism | Safeguard |
| --- | --- | --- |
| Biased targets | PIMC's strategy fusion (Frank & Basin 1998; Long et al. 2010), belief errors, value-leaf errors. The policy is pulled toward the search's mistakes. | Use search advantages, not search argmax. Mix with environment returns (β < 1). Check calibration with `examples.search_accuracy`. |
| Entropy collapse | Sharp (argmax) targets make the policy deterministic too early, and exploration dies. | Soft, clipped targets (CMPO, Muesli; Hessel et al. 2021). Keep the entropy bonus. Monitor entropy per phase. |
| Hidden-information leakage | Q computed on the **true** deal gives the actor targets it could not know, teaching overconfident "seeing" play. | Search only over the learner's own belief worlds (the observation), never `remaining_hands()`. Privileged information stays in critics. |
| Self-confirmation | Search uses the current network for rollouts and leaves, so its errors can reinforce themselves. | Regularize toward the current policy or an anchor (piKL; Jacob et al. 2022). Track KL to the anchor. Delay search until after a warm-up. |
| No benefit at equal compute | Search makes each hand several times more expensive, so fewer hands fit in the budget. | Search only a fraction *p* of decisions. Use the value leaf. Compare at equal CPU time (section 8). |
| Partnership mismatch | Search assumes the partner follows the network: SPARTA's single-agent assumption (Lerer et al. 2020). | Consistent in self-play. Check cross-play with paired or other-seed partners (roadmap step 8). |

## 4. Options compared

| Option | What changes | On-policy PPO kept? | Collapse risk | Compute | Verdict |
| --- | --- | --- | --- | --- | --- |
| (a) ExIt distillation | Extra cross-entropy loss toward a search policy (softmax of Q̂) | Yes; separate supervised term | Medium: sharp targets | Search on a fraction of states | Viable; needs a temperature |
| (b) Search advantages in PPO ("vine PPO") | At a fraction of states, add every action's search advantage to PPO's surrogate | Yes: weighting by π<sub>old</sub>(a) makes it an exact expectation of the surrogate | Low with clipping | Same as (a) | **Recommended** |
| (c) Search as behavior policy | Act with search during collection | No; the data comes from the search policy, so PPO ratios are wrong | High | Search on every decision | Not recommended |
| (d) Test-time search only | Nothing in training | n/a | None | Only at evaluation | Baseline (exists) |
| (e) Decision-time RL fine-tuning | Fine-tune a copy of the network per decision (Fickinger et al. 2021) | n/a | Low | Very high per decision | Out of scope |
| (f) Privileged leaves or targets | Use main's oracle or centralized critic as a search leaf | Leaves only | High if used for actor targets: perfect-information bias | Oracle is slow; critic is cheap | Possible leaf, never an actor target |

Why (b) over (a): the two can be combined, but (b) works inside PPO's existing
clipped objective and trust region. The search simply supplies better advantage
estimates. Clipping already limits how far one update moves the policy, which
directly addresses collapse. The PPO identity behind (b):

```text
PPO surrogate expectation at state s, over the old policy's action distribution:
  E_{a~π_old}[ min(r(a)·Â(a), clip(r(a))·Â(a)) ] = Σ_a π_old(a) · min(r(a)·Â(a), clip(r(a))·Â(a))
with r(a) = π_θ(a|s) / π_old(a|s).
With search we know Â(a) for EVERY legal a, so we can sum over all actions,
weighted by π_old(a), instead of using only the one sampled action.
```

## 5. Recommended design: search advantages inside PPO

1. **Which decisions.** Learner decisions in the play phase, chosen with
   probability *p* (for example 2–5%) from a dedicated seeded stream, so results
   stay worker-count invariant. Bidding and trump come later (section 9).
2. **The search.** Model-guided PIMC from the learner's own `Observation`
   (`texas42.belief.evaluate_actions`):
   - the current rollout snapshot plays all seats;
   - belief worlds are uniform or model-weighted; never the true deal;
   - a rollout length chosen by measured evidence, not assumption (section 6).
     - **Full rollouts** use the points objective, which ranks moves best in our
       measurements but costs about 4.5× more.
     - **Truncated rollouts** (`until='trick'`) use the value head as leaf
       (`research.ppo.search.ValueLeaf`) with the marks objective. They are cheap,
       but the current value head ranks alternative moves poorly.
     - A **points-scale value head** would combine the two: an auxiliary head
       predicting the actor team's remaining points, trained alongside PPO.
   - *S* = 8–16 sampled worlds.
3. **Search advantages.** For each legal action a, take Q̂(a) from the search.
   Then V̂ = Σ<sub>a</sub> π<sub>old</sub>(a) Q̂(a), and Â<sub>s</sub>(a) = Q̂(a) − V̂.
   Normalize per batch, as PPO already does for GAE advantages.
4. **Loss.**
   - L = L<sub>PPO</sub>(GAE, sampled action) + λ · L<sub>search</sub>.
   - L<sub>search</sub> is the all-action clipped surrogate above, summed over legal
     actions with weights π<sub>old</sub>(a). It is averaged over searched states only.
   - **CMPO variant:** target π̄(a) ∝ π<sub>old</sub>(a)·exp(clip(Â<sub>s</sub>(a)/σ, −c, c))
     with L<sub>search</sub> = KL(π̄ ‖ π<sub>θ</sub>). This bounds how far the target
     moves from the current policy (Muesli).
5. **Value.** Keep PPO's value loss on environment returns. Optionally, a small
   auxiliary loss toward V̂.
6. **Safeguards and diagnostics** (logged per update):
   - entropy per phase;
   - KL to the anchor (the existing forgetting diagnostic);
   - search/policy disagreement rate: argmax Q̂ ≠ argmax π<sub>old</sub>;
   - mean |Â<sub>s</sub>|;
   - fraction of searched decisions;
   - search seconds;
   - the existing target-KL early stop.
   - Start with λ = 0.25 and *p* = 2%, and turn search on only after a warm-up of
     ordinary PPO.

## 6. Compute budget

**Measured search cost.** `examples.search_accuracy`, shared bid+play study model;
200 play decisions sampled uniformly across the hand; 16 sampled worlds; one CPU
thread (Windows development machine with 20 logical CPUs, other work running):

| Search mode | Mean s/search (model table) | Max | Mean s/search (heuristic table) |
| --- | --- | --- | --- |
| Heuristic rollouts, full hand | 0.148 | 1.22 | 0.160 |
| Model rollouts, full hand | 0.576 | 4.98 | 0.630 |
| Model rollouts + value leaf | 0.123 | 1.32 | 0.145 |

**Training throughput (measured).** The joint bid+play study runs collected about
**132 hands/s with 8 workers**. That is about 0.061 s of worker time per hand, with
about 14 learner play decisions per hand.

**Slowdown estimate.** Searching a fraction *p* of play decisions with the value
leaf adds about *p* × 14 × 0.13 s ≈ *p* × 1.8 s of worker time per hand:

| *p* | Added worker time per hand | Estimated throughput (8 workers) |
| --- | --- | --- |
| 1% | +0.018 s (+30%) | ≈ 100 hands/s |
| 2% | +0.036 s (+60%) | ≈ 83 hands/s |
| 5% | +0.091 s (+150%) | ≈ 53 hands/s |
| 10% | +0.18 s (+300%) | ≈ 33 hands/s |

**Conclusions:**

- **The leaf is cheap but ranks moves poorly; full rollouts rank well but cost
  more.**
  - Full model rollouts cost about 4.5× more than the value leaf (about +135%
    worker time at *p* = 1%).
  - **Measured strength:** value-leaf search lost to plain PIMC (−0.141
    marks/hand), while full model rollouts beat it (+0.094). The objective
    ablation (marks instead of points with full rollouts: +0.031) suggests the
    leaf itself causes most of the loss.
  - In training, search advantages need the right *ranking* of moves, so the
    current leaf is a weak teacher.
  - **Options:**
    1. Full rollouts at *p* ≈ 0.5–1%.
    2. A points-scale value head, then re-measure its ranking.
    3. Search only late-hand positions, where full rollouts are short and cheap.
  - **Gate before training:** check leaf-vs-full-rollout rank agreement first
    (section 9).
- **Search 1–5% of play decisions.** That keeps training within roughly 1.3–2.5×
  its current cost, which is feasible.
- **Searches are cheap per hand.** Even *p* = 5% gives about 0.7 searched decisions
  per hand, which is about 35 searched states per 4,096-decision PPO batch.
- **Weight the search loss so its signal isn't swamped.** Each searched state
  contributes every legal action, but that is still a small share of the batch.
  The search-loss weight λ and *p* must be tuned together.
- **Optimizations not yet applied:**
  - batching rollouts across worlds into one network call;
  - fewer sampled worlds (8);
  - searching only positions where the policy is uncertain (high entropy), which
    targets the samples most likely to be informative.

**Caveat.** These are single-thread timings on one machine with other work
running, using positions sampled uniformly within the hand. Early-hand positions
cost up to about 10× the mean. Treat the numbers as planning estimates.

## 7. Implementation sketch

Where the recommended design would plug into existing code. These are not built
yet; each item names its owner module.

| Step | Module | Change |
| --- | --- | --- |
| Choose searched decisions | `research/ppo/selfplay_rollout.py` (`collect_phase_hand`) | Seeded per-episode draw `stream_seed(seed, 'search-decisions-v1', episode)` per learner play decision |
| Run search | same | `evaluate_actions(obs, rollout_policies=…, leaf_evaluator=…, until='trick', objective='marks')`, using factories built from the **in-memory** rollout snapshot. This needs an in-memory twin of `search.model_factories` / `ValueLeaf` built on `PPOFactory` weights, because the learner has no checkpoint file mid-rollout. |
| Store results | `research/ppo/buffer.py` (`Experience`) | Optional `search_q: tuple[(action_id, q), ...]`, carried to `tensors()` as a 58-wide array plus mask |
| Loss | `research/ppo/trainer.py` (`Trainer.update`) | Extra `search_loss` term when present; per-phase metrics |
| Config and CLI | `selfplay_trainer.SelfPlayConfig`, `examples/train_selfplay_ppo.py` | `search_fraction` (default 0 = off), `search_samples`, `search_weight`, `search_target` ∈ {advantage, cmpo}, `search_warmup_hands`. Defaults reproduce current training exactly. |
| Checkpoint | `selfplay_checkpoint.py` | Config fields only; no format change |
| Tests | `tests/` | Fraction 0 is bit-identical to today; worker invariance with search; tiny seeded update; resume |

## 8. Experiment protocol

- **Baseline:** the joint bid+play recipe (shared mode) from the same initialization.
- **Treatments:**
  - *p* ∈ {2%, 5%};
  - λ ∈ {0.25, 1};
  - target ∈ {advantage, CMPO};
  - full rollouts vs value leaf, as one ablation.
- **Budgets:** compare at equal hands *and* at equal CPU time. Treatments get fewer
  hands in the equal-time comparison.
- **Seeds:** at least 3 training seeds per arm. Report means and spread across
  seeds, not only within-run bootstrap intervals.
- **Evaluation:** held-out duplicate arena vs HHH, the shared baseline and the
  anchor (`evaluate_agent_vs_agent`). Also `search_accuracy` on each final model,
  checking whether search-trained models become better search guides.
- **Decision rule:** adopt only if the equal-CPU comparison is significantly
  positive in at least 2 of 3 seeds, with stable entropy and anchor KL.

## 9. Open questions

- **Prerequisite: does a value leaf rank moves well?** On the study model, value-leaf
  search played worse than plain PIMC even though its value estimates were
  well-calibrated on average ([results](belief.md#model-guided-results)). Before
  using a leaf to produce training advantages:
  - extend `search_accuracy` with leaf-vs-full-rollout **rank agreement** over all
    legal moves (Spearman correlation and top-1 match);
  - require good agreement, or use full rollouts instead.
  - A head trained only as PPO's baseline sees mostly on-policy moves, so it may
    never learn to separate the alternatives it is asked to compare.
- **Weak early value heads.** A value leaf from a barely trained network may be
  worse than a heuristic rollout. A warm-up period, or full rollouts at first, may
  be needed.
- **Points-scale leaf.** A small auxiliary head predicting the actor team's
  remaining points would let truncated search keep the points objective, which
  ranks play moves better.
  - Main's centralized-critic oracle head predicts double-dummy points, but it
    uses privileged inputs.
  - An information-set points head would be the honest equivalent.
- **Bidding and trump.** Their effects are only known at the end of the hand, so
  searching them needs full rollouts, which is expensive. Start with play only.
- **Belief choice.** Model-weighted beliefs are more accurate when the model plays
  the table ([results](belief.md#model-guided-results)), but they cost a replay
  per world.
- **Paired partners.** Paired mode would need search per partner and pair-aware
  rollouts.
- **Privileged leaves.** Main's centralized critic predicts signed marks from
  privileged world features. On a hypothetical world it is a cheap
  "perfect-information" leaf, with PIMC-style bias. It is worth one ablation as a
  leaf, never as an actor target.

## 10. References

- Anthony, Tian & Barber, *Thinking Fast and Slow with Deep Learning and Tree Search* (Expert Iteration), NeurIPS 2017 — https://arxiv.org/abs/1705.08439
- Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022 — https://iclr.cc/virtual/2022/poster/6418
- Hessel et al., *Muesli: Combining Improvements in Policy Optimization* (CMPO), ICML 2021 — https://arxiv.org/abs/2104.06159
- Jacob et al., *Modeling Strong and Human-Like Gameplay with KL-Regularized Search* (piKL), ICML 2022 — https://arxiv.org/abs/2112.07544
- Sokota et al., *The Update-Equivalence Framework for Decision-Time Planning*, 2023 — https://arxiv.org/abs/2304.13138
- Fickinger et al., *Scalable Online Planning via Reinforcement Learning Fine-Tuning*, NeurIPS 2021 — https://arxiv.org/abs/2109.15316
- Schulman et al., *Trust Region Policy Optimization* (vine procedure), ICML 2015 — https://arxiv.org/abs/1502.05477
- Ross, Gordon & Bagnell, *A Reduction of Imitation Learning to No-Regret Online Learning* (DAgger), AISTATS 2011 — https://arxiv.org/abs/1011.0686
- Lerer et al., *Improving Policies via Search in Cooperative Partially Observable Games* (SPARTA), AAAI 2020 — https://ojs.aaai.org/index.php/AAAI/article/view/6208
- Schmid et al., *Player of Games*, 2021 — https://arxiv.org/abs/2112.03178
- Long et al., *Understanding the Success of Perfect Information Monte Carlo Sampling*, AAAI 2010 — https://ojs.aaai.org/index.php/AAAI/article/view/7562
- Rebstock et al., *Policy Based Inference in Trick-Taking Card Games*, 2019 — https://arxiv.org/abs/1905.10911
- Tesauro & Galperin, *On-line Policy Improvement using Monte-Carlo Search*, NIPS 1996 — https://papers.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html
