# Search inside training: theory, evidence and plan

**Status:** theory, gate measurements and an enumerated implementation plan. No
training code uses search yet. This document is the authoritative home for:

- why Monte Carlo search could become part of PPO training rather than only a
  decision-time tool;
- the theory behind that claim, with references and the conditions under which it
  holds;
- the measured evidence that gates it (section 6);
- the implementation plan (section 10).

The search building blocks (belief worlds, model-guided rollouts, value leaves)
already exist; see [belief modelling](belief.md#model-guided-pimc).

**Verdict:** well founded and feasible, but gated on search quality.

- **Theory (section 4).**
  - At a searched state, search estimates the advantage of **every** legal move:
    exactly the quantity PPO's surrogate optimizes, with far lower variance than
    one sampled trajectory.
  - Iterated with a regularized update, that is approximate policy iteration.
  - It should beat training without search followed by one test-time search, but
    only if the search **ranks moves correctly**. Errors in its ranking are what
    iteration compounds.
- **Evidence (section 6).**
  - **Cheap leaves fail the ranking gate.** The value-leaf search does not rank
    moves well enough for any of the three guides measured (play-only, shared
    bid+play, paired). It agrees with full rollouts on the best move less often
    than plain heuristic rollouts do.
  - **The viewpoint is a minor cause.** Section 4.6 predicts a bias from the
    leaf taking the trick winner's viewpoint. Stopping at the searcher's own turn
    instead removes that bias and helps only a little; it still fails the gate
    (section 6.1). The value head's own error dominates.
  - **Offline distillation helps, slowly** (sections 6.2–6.3).
    - One round gave +0.009 [+0.001, +0.018] marks/hand against the source
      model; a second round, guided by the improved model, gave +0.015
      [+0.005, +0.027] in total.
    - The gains add up roughly per round, but at about +0.005–0.009 each, with no
      change yet against HHH.
- **Plan (section 10).**
  - Build in-memory search guides (P1).
  - Then train with full model rollouts on about 0.5–1% of play decisions (about
    1.4–1.8× current cost, section 9).
  - A cheap leaf needs a better value estimate first (P2 option 3).
  - Expect a modest effect, and confirm it at equal compute (P5).
- **Decision-time search is already strong.** Model-guided search with full
  rollouts is the strongest agent measured, at 1,024 deals:
  - +0.045 [+0.009, +0.081] marks/hand over plain heuristic PIMC;
  - +0.38 over the base model;
  - +0.34 over its own guide model (128 deals).

## Contents

1. [Question and success criteria](#1-question-and-success-criteria)
2. [Which "Monte Carlo"?](#2-which-monte-carlo)
3. [Formal setup and notation](#3-formal-setup-and-notation)
4. [Theory](#4-theory)
5. [Related methods under other names](#5-related-methods-under-other-names)
6. [Evidence: the move-ranking gate](#6-evidence-the-move-ranking-gate)
7. [Options compared](#7-options-compared)
8. [Recommended design](#8-recommended-design)
9. [Compute budget](#9-compute-budget)
10. [Implementation plan](#10-implementation-plan)
11. [Open questions](#11-open-questions)
12. [References](#12-references)

## 1. Question and success criteria

Can Monte Carlo search be folded into training so that the network improves faster
or further than with PPO alone, without collapsing or giving no benefit? And will
the resulting model then be a better **search guide** than a model trained without
search?

Success means all of the following:

- **Strength at equal compute.** Better held-out results than the plain PPO
  baseline at **equal CPU time**, not only at equal hands. Compare against HHH, the
  shared baseline and the anchor, over at least 3 training seeds. Search makes each
  hand more expensive, so equal-hands comparisons flatter it.
- **Stability.** No entropy collapse and no growing KL to the anchor (the existing
  forgetting diagnostic). No drop in validation after an initial gain.
- **Affordability.** Training throughput stays within an agreed budget, for
  example no worse than 2× slower.
- **Better with search too.** Test-time search guided by the integrated model is
  at least as strong as test-time search guided by the PPO-only model.

## 2. Which "Monte Carlo"?

Methods with similar names sample different things. This table fixes the meaning
used below.

| Family | What is sampled | Examples | Role here |
| --- | --- | --- | --- |
| Monte Carlo **returns** | Whole episodes under the current policy; the return is one sample of a value | REINFORCE (Williams 1992); DouZero's "Deep Monte-Carlo" (Zha et al. 2021) | Already in training: PPO's GAE advantage for the sampled action (λ = 1 is a pure Monte Carlo return) |
| Monte Carlo **rollouts** (rollout algorithms) | From one state, each candidate action followed by a fixed base policy | Tesauro & Galperin 1996; Bertsekas 2020 | The evaluation step of model-guided PIMC |
| **Determinized / PIMC** search | Hidden information (worlds) drawn from a belief, then an evaluation inside each world | GIB for bridge (Ginsberg 2001); Skat (Buro et al. 2009); `texas42.belief` | The belief layer: worlds consistent with the searcher's own observation |
| Monte Carlo **tree** search | A selectively grown tree of sampled simulations | UCT; AlphaZero; MuZero; ISMCTS (Cowling et al. 2012) | Not used. A play decision has at most 7 legal moves, so a one-ply search that evaluates all of them is enough |

**"Search in training" in this plan** means the following, at a small sample of
the learner's own decisions:

1. Sample worlds from the learner's belief (its own observation only).
2. Evaluate **every** legal action by rollouts of the current policy, optionally
   truncated with a leaf value.
3. Use the per-action estimates as a training signal.

It does **not** mean acting with search during collection. It also does not mean
the Monte Carlo returns PPO already uses.

## 3. Formal setup and notation

```text
Seats i ∈ {0,1,2,3}; seat i's team is i mod 2; partners are i and i+2.
A hand has at most 33 decisions: 4 bids, 1 trump choice, up to 28 plays.
Each seat makes at most 9 of them: 1 bid, at most 1 trump choice, 7 plays.
Reward: 0 until the hand ends, then signed marks R_i for seat i's team.
  R_i = −R_(i+1): zero-sum between the teams, common payoff within a team.
No discounting (γ = 1); the horizon is finite.

I          seat i's information state = its Observation: own hand + public history.
w          a world: a full deal consistent with I (the other seats' hidden hands).
h          a full history: world + all actions so far.
π(a | I)   the policy: a distribution over the legal actions A(I).
           |A(I)| ≤ 7 in the play phase.
b(w | I)   the belief: posterior probability of world w given I,
           when every seat acts by π.

Q^π(w, a) = E[ R_i | world w, the history so far, seat i plays a,
                     then every seat follows π from its own observation ]
Q^π(I, a) = Σ_w b(w | I) · Q^π(w, a)          information-set action value
V^π(I)    = Σ_a π(a | I) · Q^π(I, a)
A^π(I, a) = Q^π(I, a) − V^π(I)                 advantage
J(π)      = E[ R_i ] from the start of the hand

Search estimate from S sampled worlds w_1..w_S ~ b̂(· | I), weights ω_s (Σ ω_s = 1):
Q̂(I, a) = Σ_s ω_s · G(w_s, a; ξ_s)
  G   = one rollout's return; when truncated, the return so far plus a leaf value
  ξ_s = that rollout's random stream
  The same (w_s, ξ_s) is used for every action a (common random numbers).
```

**Policy rollouts make the estimate information-consistent.** In each sampled
world, every seat's later moves depend only on that seat's own observation in the
world, never on the world itself. So Q̂ is a Monte Carlo estimate of the
information-set value Q<sup>π</sup>(I, a). This differs from double-dummy PIMC,
which solves each world with perfect-information play. That solving is what
causes *strategy fusion* (Frank & Basin 1998): planning a different future in
each world although the player could not tell the worlds apart. Policy rollouts
avoid strategy fusion in the evaluation. The remaining error sources are the
belief, the leaf and sampling noise (section 4.6).

## 4. Theory

### 4.1 One search is one policy-improvement step

**Intuition.** Q<sup>π</sup>(I, a) answers "how good is move a here, if everyone
(me included) then keeps playing π?". The policy's own value at I is an
*average* of those answers. Choosing the best answer therefore cannot be worse
than the average.

```text
V^π(I) = Σ_a π(a|I) Q^π(I,a)  ≤  max_a Q^π(I,a).

Let π⁺ play argmax_a Q^π(I,a) at I and π everywhere else. Then
V^{π⁺}(I) = max_a Q^π(I,a) = V^π(I) + max_a A^π(I,a)  ≥  V^π(I).
```

Applying the rule at every information state is the improvement step of policy
iteration. The *policy improvement theorem* then gives V<sup>π⁺</sup> ≥
V<sup>π</sup> everywhere (Sutton & Barto 2018, §4.2). Bertsekas calls the same
fact the "cost improvement" property of rollout. Test-time model-guided PIMC is
this operator applied at the current decision.

**Conditions in an imperfect-information, four-player game.** Each condition
maps to something measured or controlled in this repository.

1. **The other three seats keep playing π.** With them fixed, seat i faces a
   single-agent problem over its own information states. Thanks to perfect
   recall, b(w | I) does not depend on seat i's own policy, so the theorem applies
   to seat i. This is the guarantee of SPARTA's single-agent search (Lerer et al.
   2020). The partner shares the payoff, so the team improves too.
   - The guarantee is relative to *fixed* other seats. It says nothing about
     opponents that adapt.
2. **b is the true posterior under π.** A uniform belief, or a likelihood from a
   different model, averages over the wrong worlds. Belief accuracy is measured
   in [belief modelling](belief.md#model-guided-results).
3. **Q̂ must rank moves correctly; calibration is not enough.** The step acts
   through an argmax (or a soft version of one), so what matters is the *ordering*
   of the legal moves. Section 4.5 makes this precise as ranking regret, and
   section 6 measures it.

Measured instance: full model-rollout search beats its own guide model by **+0.34
marks/hand** ([results](belief.md#model-guided-results)).

### 4.2 From one decision to a whole policy: performance difference, CPI, TRPO, PPO

**Performance difference lemma** (Kakade & Langford 2002). In the finite-horizon,
undiscounted form, for seat i with the other seats fixed:

```text
J(π') − J(π) = E_{τ ~ π'} [ Σ_{t ∈ seat i's decisions} A^π(I_t, a_t) ]
```

**Intuition.** A new policy's gain is the sum of the *old* policy's advantages
for the moves the *new* policy makes, along the new policy's own games.

**The surrogate.** The expectation is over trajectories of π', which are unknown
before the update. CPI, TRPO and PPO all replace them with the old policy's
information-state distribution d<sub>t</sub><sup>π</sup>:

```text
L_π(π') = Σ_t E_{I ~ d_t^π} Σ_a π'(a|I) A^π(I,a)  =  Σ_t E_{I ~ d_t^π, a ~ π} [ r(a) A^π(I,a) ],
  r(a) = π'(a|I) / π(a|I).
```

**How wrong the surrogate can be.** Let δ = max<sub>I</sub> TV(π'(·|I), π(·|I)),
the largest total-variation change of the policy at any state. Let ε =
max |A<sup>π</sup>|, and let H<sub>i</sub> ≤ 9 be the number of seat i's decisions.
Then

```text
| J(π') − J(π) − L_π(π') |  ≤  2 ε δ² H_i².
```

Derivation:

1. Σ<sub>a</sub> π(a|I) A<sup>π</sup>(I,a) = 0. So the integrand
   f(I) = Σ<sub>a</sub> (π'−π)(a|I) A<sup>π</sup>(I,a), and |f| ≤ 2δε.
2. At each decision, the two policies can be coupled to differ with probability
   at most δ. So by the t-th decision, the state distributions differ by at most
   tδ in total variation.
3. Each term therefore errs by at most tδ · (max f − min f) ≤ 4tδ²ε.
4. Summing over t < H<sub>i</sub> gives the bound.

This is the finite-horizon analogue of TRPO's Theorem 1 (Schulman et al. 2015) and
of CPI's mixture bound. The error is *quadratic* in the size of the policy change.
That is why PPO keeps each update small (clipping, target-KL early stop), and why
first-order improvement is reliable here.

Two qualifications:

- With shared-parameter self-play, all four seats change together. The lemma then
  needs extra terms for the other seats' changes, and the horizon is up to 33
  decisions.
- The bound is a guide to step size, not a guarantee of self-play progress
  (section 4.5).

**Connection to search.** A searched state yields an estimate of
A<sup>π</sup>(I, a) for **every** legal a: exactly the quantity inside
L<sub>π</sub>. GAE yields a noisy estimate for the one sampled a. Search does not
change PPO's objective; it changes how well that objective is estimated.

### 4.3 Search advantages give lower-variance, all-action gradients

Three variance reductions, each with an established name:

**(1) Sum over actions instead of sampling one** (Rao–Blackwellization; Expected
Policy Gradients, Ciosek & Whiteson 2018; Mean Actor-Critic, Allen et al. 2017):

```text
g_1   = ∇θ log π_θ(a|I) · Â(I,a),   a ~ π_old            one sampled action (PPO, REINFORCE)
g_all = Σ_a ∇θ π_θ(a|I) · Â(I,a)  =  E_{a ~ π_θ}[ g_1 | I ]     all actions
Var(g_1) = Var(g_all) + E[ Var(g_1 | I) ]  ≥  Var(g_all)       law of total variance
```

This holds for the same advantage estimator Â, at θ = θ<sub>old</sub>, where
π<sub>θ</sub> = π<sub>old</sub>. Conditioning on I removes the action-sampling
noise at no cost in bias.

**(2) Average hidden worlds instead of using the one real deal.**

- GAE sees one deal and one future.
- Search averages S belief worlds. For independent worlds, the world-sampling
  variance falls as 1/S. With self-normalized likelihood weights it falls as
  roughly 1/ESS, the effective sample size reported by `texas42.belief`.
- **Trade-off:** the real deal is an exact sample from the true posterior, so
  GAE has no belief bias. Search has much lower variance but inherits the
  belief's bias (section 4.6).

**(3) Common random numbers for differences** (the TRPO "vine" procedure,
Schulman et al. 2015). Advantages are differences of action values:

```text
Var(Q̂_a − Q̂_b) = Var Q̂_a + Var Q̂_b − 2 Cov(Q̂_a, Q̂_b)
```

`evaluate_actions` uses the same worlds and rollout seeds for every legal move,
so the covariance is large and positive. Most outcome variance comes from the
deal, which all moves share. What remains is the noise that actually depends on
the move.

**The all-action clipped surrogate.** At searched states 𝒮 it applies PPO's
clipped objective to every legal action, weighted by π<sub>old</sub>:

```text
L_search(θ) = (1/|𝒮|) Σ_{I ∈ 𝒮} Σ_{a ∈ A(I)} π_old(a|I) · min( r_θ(a|I)·Â_s(I,a),
                                                              clip(r_θ(a|I), 1−ϵ, 1+ϵ)·Â_s(I,a) )
Â_s(I,a)    = Q̂(I,a) − Σ_b π_old(b|I) Q̂(I,b)
∇θ L_search at θ_old = (1/|𝒮|) Σ_I Σ_a ∇θ π_θ(a|I) · Â_s(I,a)       = g_all, averaged
```

- Weighting by π<sub>old</sub>(a) makes the sum the exact expectation, over the
  old policy's action distribution, of PPO's per-sample objective:
  E<sub>a~π<sub>old</sub></sub>[min(r Â, clip(r) Â)].
- **The baseline.** Because Σ<sub>a</sub> ∇π<sub>θ</sub>(a|I) = 0, subtracting
  the baseline Σ<sub>b</sub> π<sub>old</sub> Q̂ does not change the gradient at
  θ<sub>old</sub>. It does decide which actions are pushed up or down, and
  therefore which side of the clip applies.

**(4) More labels per decision.** GAE gives one advantage per decision. A searched
decision gives |A(I)| ≤ 7 advantages, each with lower variance.

**Caveat: greedy rollouts estimate a different policy's values.**

- Rollouts that follow each player's most likely move (`model_factories(greedy=True)`)
  estimate Q of *greedy(π)*, not of the stochastic π being trained.
- For improvement this is still valid: improving on a deterministic base policy
  is the classic rollout algorithm, and the variance is lower.
- But A<sup>greedy(π)</sup> is not the A<sup>π<sub>old</sub></sup> in PPO's
  surrogate. The plan makes this a setting (P3) and measures both (P5).

### 4.4 Regularized improvement: soft targets that cannot collapse

**The problem with hard targets.** Pure policy iteration (or ExIt with argmax
targets) moves the policy to a deterministic choice at each searched state. With a
noisy Q̂, that chases noise and collapses entropy.

**The fix: make the step size explicit.** Regularized improvement trades value
against distance from the current policy:

```text
π̄ = argmax_{q ∈ Δ(A(I))}  Σ_a q(a) Q̂(I,a) − η · KL(q ‖ π_old)
  ⇒  π̄(a) = π_old(a) · exp( Q̂(I,a) / η ) / Z      (Z normalizes; Q̂ may be replaced by Â_s)
```

Derivation: maximize with a Lagrange multiplier for Σ q = 1. Stationarity gives
log q(a) = log π<sub>old</sub>(a) + Q̂(a)/η − const.

Properties:

- **η sets the step size.**
  - η → 0 gives the argmax (policy iteration).
  - η → ∞ gives no change.
  - In between it is one mirror-descent (exponentiated-gradient) step with step
    size 1/η.
- **No move is ever zeroed.** π̄(a) > 0 wherever π<sub>old</sub>(a) > 0.
- **The step is bounded.** Clip the scaled advantages, x<sub>a</sub> =
  clip(Â<sub>s</sub>(a)/σ, −c, c), and use x in place of Q̂/η. Then:
  - π̄(a)/π<sub>old</sub>(a) ∈ [e<sup>−2c</sup>, e<sup>2c</sup>];
  - KL(π̄ ‖ π<sub>old</sub>) ≤ 2c. By Jensen, log Z ≥ E<sub>π<sub>old</sub></sub>[x],
    so KL = E<sub>π̄</sub>[x] − log Z ≤ max x − min x.

  This is **CMPO**, as used in Muesli (Hessel et al. 2021). One noisy search can
  neither eliminate a move nor move the policy far.

**The same operator under other names:**

- **MPO's E-step** (Abdolmaleki et al. 2018) and **V-MPO** (Song et al. 2019).
- **AWR's** exp(A/β) regression weights (Peng et al. 2019).
- **piKL** (Jacob et al. 2022), which regularizes toward a fixed *anchor* policy
  instead of π<sub>old</sub>.
- **MCTS as regularized policy optimization** (Grill et al. 2020). AlphaZero's
  visit counts approximately solve
  π̄ = argmax<sub>y</sub> [q<sup>T</sup>y − λ<sub>N</sub> KL(π<sub>θ</sub> ‖ y)],
  which uses the *reverse* KL.
  - The solution is π̄ = λ<sub>N</sub> π<sub>θ</sub> / (α − q), with α chosen so
    that π̄ sums to 1.
  - λ<sub>N</sub> = c·√(Σ<sub>b</sub> n<sub>b</sub>) / (|A| + Σ<sub>b</sub>
    n<sub>b</sub>), where n<sub>b</sub> is the visit count of action b. So
    λ<sub>N</sub> shrinks like 1/√N with the simulation count N: **the more
    search, the more its values are trusted.**
- **Update equivalence and magnetic mirror descent** (Sokota et al. 2023; MMD,
  Sokota et al. 2023b). Decision-time planning in imperfect-information games can
  be viewed as reproducing one step of such an update at the current decision.
  MMD adds a pull toward a fixed "magnet" policy ρ:

  ```text
  π_{t+1} ∝ [ π_t · ρ^{αη} · exp(η q_t) ]^{1/(1+αη)}
          = argmax_π  E_{a~π} q_t(a) − α KL(π, ρ) − (1/η) KL(π, π_t)
  ```

  The frozen play-only anchor is a natural magnet here. The existing anchor-KL
  diagnostic already measures the distance it would control.

**How this fits PPO.** At searched states, the CMPO target can enter as a KL
loss KL(π̄ ‖ π<sub>θ</sub>), alongside PPO's clipped loss on every state. This is
how Muesli combines the two.

### 4.5 Integrated versus separate training and search

**The improvement operator 𝓘.** Given a policy π, 𝓘(π) is the policy "search with
π (rollouts, likelihoods, leaf), then act on Q̂", either greedily or with the
regularized rule above.

**Separate (today):** train π<sub>PPO</sub> with PPO, then play
𝓘(π<sub>PPO</sub>). Combining 4.1 and 4.2:

```text
J(𝓘(π_PPO)) − J(π_PPO) = E_{τ ~ 𝓘(π_PPO)} Σ_t max_a A^{π_PPO}(I_t, a)  ≥  0
```

This holds with exact Q, a greedy 𝓘 and the other seats playing
π<sub>PPO</sub>. It is **one** application of the operator. The gain is large
exactly when π<sub>PPO</sub> is far from greedy with respect to its own Q, as
measured (+0.34 marks/hand).

**Integrated:** π<sub>k+1</sub> ≈ Π[𝓘(π<sub>k</sub>)], where Π means "fit the
network" (here, a regularized PPO update toward the search advantages).

- This is **approximate policy iteration** (API).
- Exact policy iteration reaches a policy that 𝓘 cannot improve in finitely many
  steps: a best response to the fixed other seats.
- API with per-iteration error ε stays within an error ball around it.
  Bertsekas & Tsitsiklis (1996) bound limsup ‖V* − V<sup>π<sub>k</sub></sup>‖ ≤
  2γε/(1−γ)<sup>2</sup> in the discounted case. Scherrer (2014) compares such
  bounds for API, CPI and related schemes.
- Here the per-seat horizon (≤ 9 decisions) plays the role of 1/(1−γ), so error
  amplification is mild.
- Test-time search then adds one more 𝓘 on top: play 𝓘(π<sub>K</sub>).

**Why integration should help.** Each reason is a separate mechanism with its own
evidence or test:

1. **Iterated vs single improvement.** Separate search stops after one step.
   Integrated training takes many, each from a better base.
2. **Amortization and generalization.** Search results are stored in the weights.
   - The network improves at every decision, including those too expensive to
     search at test time (early-hand positions, bidding).
   - It generalizes search results to states that were never searched.
   - This is the core argument of Expert Iteration (Anthony et al. 2017) and
     AlphaZero; Dual Policy Iteration (Sun et al. 2018) gives it an API analysis.
3. **A better guide makes a better search.** 𝓘's quality depends on π through
   the rollouts, the likelihood beliefs and the leaf. A better π therefore gives a
   better 𝓘(π).
   - Measured: model-guided search beat plain PIMC only when the model's rollouts
     were used ([results](belief.md#model-guided-results)).
   - P6 tests whether an integrated model is a better guide.
4. **More improvement per hand.** Search supervision has lower variance and gives
   |A| labels per searched state (section 4.3). Hamrick et al. (2021) found that,
   in model-based RL, planning helps most *inside learning* (better targets and
   better data). They also found that shallow trees with simple Monte Carlo
   rollouts perform about as well as more complex search in all but the hardest
   reasoning tasks, and that planning alone does not produce strong
   generalization. That is one reason the evaluation (P5) is held out.
5. **On-policy states.** The network is trained on states that it reaches itself,
   as in DAgger (Ross et al. 2011). It avoids the distribution shift of imitating
   an expert only on the expert's own states.

**When integration does not help:**

- **A biased operator compounds.** If 𝓘 consistently prefers wrong moves (belief
  mismatch, leaf bias, objective proxy), iterating it reinforces the bias. Separate
  search applies the bias only once.
  - API bounds scale with the per-iteration error. For classification-based
    policy iteration, that error is the **regret-weighted** classification error
    (Lazaric, Ghavamzadeh & Munos 2016):

    ```text
    E_I[ Q^π(I, a*(I)) − Q^π(I, â(I)) ],   a* = the truly best move,  â = the move the target picks
    ```

  - The ranking regret measured by `search_accuracy --rank-check` (section 6) is
    this quantity, with full rollouts standing in for the true Q<sup>π</sup>. That
    is why it is the gate.
- **Compute.** Searched hands cost more, so fewer hands fit the same budget. Only
  an equal-CPU comparison settles it (P5).
- **Self-play non-stationarity.** The improvement guarantees hold against *fixed*
  other seats.
  - In two-team zero-sum self-play, best-response-style improvement can cycle.
    Plain PPO self-play has the same problem; search does not cause it or cure it.
  - Historical opponents (`SelfPlayTrainer`) and the anchor KL are the existing
    mitigations.
  - Equilibrium-seeking search (ReBeL; Player of Games) is the principled
    alternative, but much heavier for four seats with 7-domino hands.

### 4.6 Bias of the search target, term by term

The error of Q̂ relative to the ideal Q<sup>π</sup>(I, a) splits into parts that
can be measured and fixed separately:

| Source | Mechanism | Measured by | Mitigation |
| --- | --- | --- | --- |
| Belief | b̂ ≠ b<sub>π</sub>. A uniform belief ignores bidding and play evidence. Likelihood weights are self-normalized importance sampling: O(1/S) bias, high variance when ESS is low. | P(true owner), log-loss, ESS (`search_accuracy`) | Model likelihood with ε-softening; more worlds; learned beliefs (roadmap step 7) |
| Rollout policy | Greedy rollouts evaluate greedy(π), not π (section 4.3) | P5 ablation | Sampled rollouts where on-policy advantages matter |
| **Leaf viewpoint** | See below | Truncation regret (section 6) | Own-turn leaf or world-value leaf (P2) |
| Objective proxy | Points are a denser proxy for the true objective, marks. Points rank play moves more sharply but can favour extra points in a hand that is already decided. | Objective regret (section 6) | Points for ranking, marks for the PPO return; or a points-scale leaf |
| Self-confirmation | 𝓘 uses π, so π's errors feed back into its own targets | Anchor KL, entropy, validation | Regularization (4.4); mixing with GAE (β < 1); warm-up |
| Sampling noise | Finite S | Repeat with other seeds | Common random numbers; S = 16; more worlds where policy entropy is high |

**Leaf viewpoint.** `ValueLeaf` scores a truncated rollout with the value head of
the seat **acting** at the leaf, V<sup>π</sup>(I<sub>actor</sub>). That is the
value from the actor's information. Unless the actor is the searcher, the actor
does not know the searcher's private hand:

```text
wanted:  E[ V^π(w, h_leaf)        | I, a ]       the world's value under π
leaf:    E[ V^π(I_actor(h_leaf))  | I, a ],      V^π(I_actor) = E[ V^π(w', h') | I_actor ]
```

- **When they agree.** By the tower property they agree when I<sub>actor</sub>
  *refines* I, i.e. it contains everything I does. That holds at the searcher's
  own later decisions (perfect recall).
- **When they disagree.** An opponent's or the partner's information does not
  contain the searcher's hand, so their value averages over hands the searcher
  knows are not real.
- **The predicted effect** (a hypothesis): a leaf from another seat's viewpoint
  credits the searcher only for strength that has become *public*. A strong
  domino still held is spread, in the actor's belief, across the other unseen
  hands. So the leaf should under-value keeping winners and over-value cashing
  them early.
- With `until='trick'`, whoever wins the trick acts at the leaf, so most leaves
  come from another seat's viewpoint.

**Two unbiased alternatives**, both information-legal because they read only
hypothetical worlds drawn from the searcher's own belief:

- **(a) Own-turn leaf.** Stop each rollout at the searcher's own next decision and
  use the searcher's value head there. The searcher's later information refines
  I, so E[V<sup>π</sup>(I'<sub>own</sub>) | I, a] = E[V<sup>π</sup>(w, h') | I, a].
  Cost: about one trick, like `until='trick'`. Implemented as `until='turn'`
  (`--pimc-leaf value-turn`). Measured: it ranks only a little better than the
  trick leaf and still fails the gate, because unbiased in principle is not
  accurate in practice: the value head's own error dominates (section 6.1).
- **(b) World-value leaf.** The centralized critic's main head
  (`research/ppo/critic.py`, `CentralCritic.forward`) predicts the actor team's
  signed marks from privileged world features (`texas42.privileged.encode_privileged`)
  under the rollout policies. That estimates V<sup>π</sup>(w, h) itself.
  - Its **oracle** head is different: it predicts double-dummy points, i.e.
    perfect-information play, and would reintroduce strategy-fusion bias.
  - Neither head may ever be an actor input or a target computed on the real deal.

**Failure modes and safeguards** (summary):

| Failure | Mechanism | Safeguard |
| --- | --- | --- |
| Biased targets | Belief, leaf and objective errors (table above); strategy fusion if a perfect-information leaf were used | Advantages, not argmax targets. Mix with environment returns. Gate on ranking regret (section 6). |
| Entropy collapse | Sharp targets make the policy deterministic too early | Soft, clipped targets (CMPO). Keep the entropy bonus. Monitor entropy per phase. |
| Hidden-information leakage | Q computed on the **true** deal gives the actor targets it could not know | Search only over the learner's own belief worlds, never `remaining_hands()`. Privileged information stays in critics and leaves on hypothetical worlds. |
| Self-confirmation | The search uses the current network | Regularize toward π<sub>old</sub> or the anchor (piKL, MMD). Track anchor KL. Warm up first. |
| No benefit at equal compute | Fewer hands fit in the budget | Search a fraction *p* of decisions. Use a cheap, verified leaf. Compare at equal CPU time. |
| Partnership mismatch | Search assumes the partner follows π (SPARTA's assumption) | Consistent in self-play. Check cross-play with paired or other-seed partners (roadmap step 8). |

## 5. Related methods under other names

All of the methods below combine sampled evaluation with policy learning. They
differ in what is searched, what the network learns from it, and whether the game
has hidden information.

| Method | Searched or sampled | What training uses | Relation to this plan |
| --- | --- | --- | --- |
| Policy iteration; rollout algorithms (Sutton & Barto 2018; Bertsekas 2020; Tesauro & Galperin 1996) | Rollouts of a base policy for each action | Exact Q (policy iteration); nothing (rollout) | Section 4.1. Test-time model-guided PIMC is a rollout algorithm |
| RCPI (Lagoudakis & Parr 2003); DPI (Lazaric et al. 2016) | Rollouts of every action at sampled states | A classifier on the best action; DPI weights each mistake by its regret | Hard-target ExIt; regret is the section 6 gate metric |
| CPI (Kakade & Langford 2002) | Q<sup>π</sup> estimates | Mixture update π' = (1−α)π + α·greedy | Conservative step; ancestor of trust regions |
| TRPO "vine" (Schulman et al. 2015) | All actions branched from rollout states, common random numbers | Surrogate with branched advantages | Option (b) is "vine PPO" plus belief sampling |
| EPG (Ciosek & Whiteson 2018); MAC (Allen et al. 2017) | Nothing: a learned critic Q(s, ·) | All-action gradient Σ<sub>a</sub> ∇π Q | Same gradient; Q comes from search instead of a critic |
| Expert Iteration (Anthony et al. 2017); AlphaZero (Silver et al. 2018); MuZero (Schrittwieser et al. 2020) | MCTS with the network as prior and value; MuZero also learns the model | Cross-entropy toward visit counts; value toward outcomes | The integrated loop, for perfect-information games |
| Gumbel AlphaZero / MuZero (Danihelka et al. 2022) | Sequential halving over sampled root actions | Improved policy from completed Q-values | Guaranteed improvement with few simulations; here all ≤ 7 moves are evaluated anyway |
| MCTS as regularized policy optimization (Grill et al. 2020) | MCTS | The exact regularized target π̄ | Section 4.4 |
| MPO, V-MPO, AWR, CMPO / Muesli | Learned Q or A (Muesli optionally adds a model) | exp(A/η)-weighted targets under a KL limit | The CMPO variant (P4) |
| piKL (Jacob et al. 2022); MMD (Sokota et al. 2023b) | Search regularized toward an anchor or magnet | Decision-time (piKL); training updates (MMD) | The anchor as regularizer |
| DAgger (Ross et al. 2011); policy distillation (Rusu et al. 2016) | Expert labels on the learner's own states | Supervised imitation | "Distill test-time search into the network" |
| Dyna (Sutton 1991) | Simulated experience from a model | Value and policy updates from simulated experience | Search in training is Dyna with the exact simulator plus belief sampling |
| DouZero (Zha et al. 2021) | Nothing: Monte Carlo *returns* of whole games | Q(s, a) regression; ε-greedy play | Same "Monte Carlo" name, but returns rather than search |
| PerfectDou (Yang et al. 2022) | Nothing: a privileged critic | Perfect-information distillation in actor-critic | Our centralized critic; a world-value leaf (P2) |
| Suphx (Li et al. 2020) | Run-time policy adaptation over sampled hidden tiles; oracle guiding | RL with privileged features dropped out over training | Test-time fine-tuning, option (e) |
| AlphaZe∗∗ (Blüml, Czech & Kersting 2023) | PIMC-style determinized search inside AlphaZero-like training | AlphaZero targets | **Closest precedent**: PIMC inside training for imperfect-information games |
| ReBeL (Brown et al. 2020); Player of Games (Schmid et al. 2021) | CFR search over public belief states | Value (and policy) networks trained on search outputs | Equilibrium-seeking, two-player zero-sum; heavy for four seats |
| SPARTA (Lerer et al. 2020); Learned Belief Search (Hu et al. 2021) | Single-agent search with a blueprint; LBS learns the belief | Nothing (test time) | The single-agent assumption (4.1); learned beliefs |
| Joint Policy Search (Tian et al. 2020) | Joint improvement for collaborating partners (bridge bidding) | Improved joint policy | Partnership bidding (future) |
| Skat PIMC with learned inference (Solinas, Rebstock & Buro 2019; Rebstock et al. 2019) | PIMC with worlds weighted by a network | Supervised inference model | The belief layer |
| RL fine-tuning (Fickinger et al. 2021) | Per-decision fine-tuning of a copy of the network | Test time only | Option (e) |

## 6. Evidence: the move-ranking gate

**Question.** Does a cheap search order the legal moves the way the
best-playing search does? Section 4.5 explains why this ranking (and its regret)
is the per-iteration error that training would compound.

**Setup** (`examples.search_accuracy --rank-check`):

- **Positions:** the same 150 play decisions for every guide. Heuristic V1 played
  all four seats for 75 hands, and 2 decisions with at least two legal moves were
  sampled per hand (4.0 legal moves on average).
- **Search:** 16 uniform-belief worlds per search, seed 20261022, the same worlds
  and rollout seeds for every mode at a position.
- **Guides:**
  - **base:** play-only (`runs/study-play-v1`);
  - **shared:** bid+play (`runs/study-shared`);
  - **paired:** the paired-partner pair (`runs/study-paired`, partner-1 at seats
    0/1 and partner-2 at seats 2/3, `--partner-model`).
- **Reference:** each guide's own full model rollouts with the points objective,
  the configuration that plays best ([results](belief.md#model-guided-results)).
  The truncation row uses full model rollouts with the marks objective instead,
  so that only truncation differs.
- **Noise scale:** a second full-rollout search with independent worlds and
  seeds, compared with the reference (the `noise` comparison of `--rank-check`).
  Two estimates of the same quantity differ only by sampling, so this shows how
  much disagreement 16 worlds alone produce.
  - It is not a strict ceiling. The candidates share the reference's worlds
    (common random numbers), and with greedy rollouts each world's result is
    deterministic. So a candidate's disagreement with the reference is a
    *systematic* difference, seen through only 16 worlds.
- **Random move:** the reference's regret for a uniformly random legal move, for
  scale.
- Outputs: `runs/model-guided/rank-{base,shared,paired}.json` (not committed).
  Timings in these runs are inflated (three runs in parallel); use section 9 for
  cost.

**Metrics:**

- **Concordance:** the share of move pairs ordered the same way, among pairs both
  searches order strictly.
- **Top-1:** how often both searches pick the same best move.
- **Spearman:** rank correlation, averaged over positions with at least 3 moves.
- **Regret:** the reference's value of its best move minus its value of the
  candidate's best move. Units are points, or marks for the truncation row.

| Guide | Candidate (vs reference) | Concordance | Top-1 | Spearman | Regret |
| --- | --- | --- | --- | --- | --- |
| base | Noise scale: full rollouts, new worlds | 70.0% | 62.0% | 0.557 | 0.652 |
| base | Heuristic rollouts (points) | 68.3% | 52.7% | 0.451 | 0.891 |
| base | Value leaf (marks) | 63.1% | 45.3% | 0.311 | 1.190 |
| base | Full rollouts, marks objective | 86.5% | 65.3% | 0.695 | 0.521 |
| base | Random legal move | – | – | – | 1.949 |
| shared | Noise scale: full rollouts, new worlds | 71.6% | 66.0% | 0.583 | 0.573 |
| shared | Heuristic rollouts (points) | 70.8% | 57.3% | 0.461 | 0.777 |
| shared | Value leaf (marks) | 62.3% | 44.0% | 0.327 | 0.981 |
| shared | Full rollouts, marks objective | 86.4% | 69.3% | 0.701 | 0.463 |
| shared | Random legal move | – | – | – | 2.014 |
| paired | Noise scale: full rollouts, new worlds | 69.4% | 58.0% | 0.520 | 0.728 |
| paired | Heuristic rollouts (points) | 70.9% | 53.3% | 0.508 | 0.807 |
| paired | Value leaf (marks) | 63.9% | 50.0% | 0.416 | 0.921 |
| paired | Full rollouts, marks objective | 85.4% | 68.7% | 0.715 | 0.407 |
| paired | Random legal move | – | – | – | 1.937 |

**Truncation alone.** Value leaf vs full model rollouts, both scored in marks:

| Guide | Concordance | Top-1 | Spearman | Regret (marks) | Random-move regret (marks) |
| --- | --- | --- | --- | --- | --- |
| base | 65.5% | 42.0% | 0.334 | 0.100 | 0.156 |
| shared | 61.8% | 37.3% | 0.263 | 0.114 | 0.170 |
| paired | 64.2% | 40.0% | 0.381 | 0.095 | 0.159 |

**Leaf minus heuristic rollouts**, both against the points reference. Paired
bootstrap over positions, 5,000 resamples, 95% intervals. This is a post-hoc
analysis of the per-position `q` rows in the JSON outputs, not part of the CLI:

| Guide | Regret difference (points) | Top-1 difference |
| --- | --- | --- |
| base | +0.30 [−0.03, +0.66] | −7.4 pp [−16.7, +2.7] |
| shared | +0.21 [−0.10, +0.52] | −13.4 pp [−23.3, −2.7] |
| paired | +0.12 [−0.22, +0.47] | −3.4 pp [−13.3, +6.7] |

**Readings:**

- **The value leaf fails the gate for every guide.** Its regret is higher and its
  top-1 agreement lower than those of plain heuristic rollouts, which need no
  network at all.
  - Each guide's interval alone is wide; only the shared guide's top-1 difference
    excludes zero.
  - But all six differences point the same way on the same positions, and the
    arena result agrees: leaf search lost to plain PIMC by −0.141 marks/hand.
- **Truncation is the main cause.** With the objective held fixed (marks), the
  leaf picks the same best move as full rollouts only 37–42% of the time. Its
  regret is 60–67% of a random move's: it recovers only about a third of the gap
  between a random move and the best one.
- **The objective matters less.** Scoring full rollouts in marks instead of
  points keeps 85–87% of pairwise orderings. This matches the arena ablation, where
  marks cost a little and the leaf cost more.
- **Training the guide differently does not fix it.** Play-only, shared bid+play
  and paired guides all show the same pattern. The paired pair's leaf is
  slightly better and the base model's worst, within noise. A failure common to
  three differently trained value heads points to a structural cause rather than
  one weak checkpoint. The leaf-viewpoint bias of section 4.6 is the leading
  candidate, and the P2 leaves test it directly.
- **Sixteen worlds give noisy rankings, and the leaf's error is not noise.**
  - Two independent full-rollout searches agree on the best move in only 58–66%
    of positions. Their regret against each other (0.57–0.73 points) is about a
    third of a random move's.
  - Heuristic rollouts sit near that noise scale: concordance 68–71% vs 69–72%.
  - The value leaf falls clearly below it (top-1 44–50%, regret 0.92–1.19),
    *although* it shares the reference's worlds. Its disagreement is a
    systematic error, not sampling noise.
- **Why this distinction matters for training.** Unbiased noise in Q̂ averages
  out over many PPO updates, as GAE noise already does. A systematic ranking
  error does not average out, and iteration compounds it (section 4.5).
  - Full rollouts are therefore usable training targets despite their noise,
    especially as soft advantages or CMPO targets rather than argmax labels
    (section 4.4).
  - The current leaf is not usable.

**Gate decision (P0).** The acting-seat value leaf (`ValueLeaf` with
`until='trick'`) must **not** supply training advantages. Training-time search
starts with full model rollouts and the points objective at *p* ≈ 0.5–1%
(1.4–1.8× current cost, section 9), or after a P2 leaf passes this same gate. Full model rollouts
remain the reference and the recommended decision-time search.

### 6.1 The own-turn leaf (P2a)

**Question.** Section 4.6 predicts that the leaf ranks poorly partly because it
takes the trick winner's viewpoint. Does stopping at the searcher's **own** next
decision, where the value head sees the searcher's own information, fix the
ranking?

**Setup.** As above: `--rank-check` now adds `model-rollouts+turn-leaf`, the same
value head with `until='turn'`, on the same positions, worlds and seeds. The
existing rows reproduced exactly.

| Guide | Candidate (vs reference) | Concordance | Top-1 | Spearman | Regret |
| --- | --- | --- | --- | --- | --- |
| base | Trick leaf vs full rollouts (marks) | 65.5% | 42.0% | 0.334 | 0.100 marks |
| base | **Own-turn leaf** vs full rollouts (marks) | 64.5% | 41.3% | 0.326 | 0.092 marks |
| base | **Own-turn leaf** vs full rollouts (points) | 63.2% | 44.0% | 0.288 | 0.942 |
| shared | Trick leaf vs full rollouts (marks) | 61.8% | 37.3% | 0.263 | 0.114 marks |
| shared | **Own-turn leaf** vs full rollouts (marks) | 63.8% | 40.0% | 0.314 | 0.078 marks |
| shared | **Own-turn leaf** vs full rollouts (points) | 65.1% | 46.0% | 0.368 | 0.887 |
| paired | Trick leaf vs full rollouts (marks) | 64.2% | 40.0% | 0.381 | 0.095 marks |
| paired | **Own-turn leaf** vs full rollouts (marks) | 68.9% | 42.7% | 0.405 | 0.082 marks |
| paired | **Own-turn leaf** vs full rollouts (points) | 65.9% | 42.0% | 0.375 | 0.981 |

For comparison, against the points reference heuristic rollouts score top-1
52.7%, 57.3% and 53.3%, with regret 0.891, 0.777 and 0.807 points (table above).

**Paired bootstrap** over positions (5,000 resamples, 95% intervals; post-hoc
analysis of the `q` rows):

| Guide | Own-turn − trick regret (marks) | Own-turn − heuristic regret (points) | Own-turn − heuristic top-1 |
| --- | --- | --- | --- |
| base | −0.008 [−0.038, +0.023] | +0.05 [−0.24, +0.36] | −8.7 pp [−18.7, +1.3] |
| shared | −0.036 [−0.062, −0.012] | +0.11 [−0.19, +0.44] | −11.3 pp [−22.0, −1.3] |
| paired | −0.013 [−0.044, +0.014] | +0.17 [−0.13, +0.50] | −11.3 pp [−20.0, −2.7] |

**Readings:**

- **The viewpoint matters, but only a little.** With the objective held fixed
  (marks), the own-turn leaf has lower regret than the trick leaf for every guide,
  by 8–32%. Only the shared guide's difference excludes zero.
- **It still fails the gate.** Against the best-playing search it picks the
  same best move less often than plain heuristic rollouts do (42–46% vs 53–57%;
  the interval excludes zero for two guides), and its regret is higher for every
  guide.
- **So the value head's own error dominates.** A head trained only as PPO's
  baseline, on the moves the policy actually takes, does not separate the
  alternatives well from either viewpoint. Section 4.6's bias is real but
  secondary.
- **Cost.** In these (concurrent) runs the own-turn leaf took 0.13–0.14
  s/search, against 0.09–0.10 for the trick leaf and 0.39–0.42 for full
  rollouts. It is still about 3× cheaper than full rollouts.

**Decision.** The arena check for this leaf is skipped, as pre-registered.
Neither value leaf may supply training advantages. A cheap leaf needs a better
value estimate (P2 option 3: a head trained to separate alternatives, such as a
points-scale or search-supervised head), not just a better stopping point.

### 6.2 One-step distillation pilot

**Question.** Before building training-time search: does **one** regularized
improvement step, distilled from search, make the network itself play better?
That is one iteration of the integrated loop of section 4.5, run offline.

**Setup.** `examples.search_distillation_pilot`
([command](command_reference.md#model-guided-pimc)).
It reproduces these results exactly: the search and the checkpoint are
bit-identical to the original scratch run. Outputs are in
`runs/model-guided/pilot/`. The target functions (`search_advantages`,
`cmpo_target`, `masked_kl`) live in `research/ppo/search.py`, where the training
search loss (P4) reuses them.

- **States:** the shared bid+play model at all four seats (greedy, its own
  evaluation distribution) for 1,000 hands, seed 20261025. 4 play decisions with
  at least two legal moves were sampled per hand, giving 4,000 states. Every
  tenth hand is held out (400 validation states).
- **Search:** full model rollouts, points objective, 16 uniform worlds, from each
  state's own observation. 0.43 s/search; 3.8 minutes with 8 processes.
- **Target:** CMPO (section 4.4), π̄ ∝ π<sub>old</sub>·exp(clip(Â/σ, −c, c)),
  with c ∈ {1, 2}. σ = 1.01 points, the π<sub>old</sub>-weighted RMS advantage.
- **Fine-tuning:**
  - only the play network;
  - loss KL(π̄ ‖ π<sub>θ</sub>) + (V<sub>θ</sub> − V<sub>old</sub>)<sup>2</sup>.
    The trunk is shared, and the second term keeps the value head's outputs.
  - Adam, learning rate 10<sup>−4</sup>, batches of 256, early stopping on
    validation KL (best epochs 8 and 9).
- **Evaluation:** `evaluate_agent_vs_agent`, 4,096 duplicate deals (8,192 games),
  held-out seed 20261026.
  - Both sides use neural bidding, heuristic trump and neural play.
  - The arena confirms that only the play network differs.
  - Intervals resample whole duplicate groups.

**Diagnostics** (from `distilled-c*.json`):

| | c = 1 | c = 2 |
| --- | --- | --- |
| Search's best move ≠ policy's choice (train / validation) | 61% / 66% | 61% / 66% |
| Target step KL(π̄ ‖ π<sub>old</sub>), mean (bound 2c) | 0.068 | 0.130 |
| Play choices changed after fine-tuning (train / validation) | 4.6% / 2.8% | 5.3% / 3.3% |
| KL to target, train (before → after) | 0.068 → 0.038 | 0.130 → 0.078 |
| KL to target, validation (before → after) | 0.086 → 0.084 | 0.160 → 0.153 |
| Value-head drift, validation RMSE (marks) | 0.020 | 0.023 |

**Results:**

| A | B | A marks/hand (95% CI) | A win rate |
| --- | --- | --- | --- |
| Distilled, c = 1 | Shared model | +0.005 [−0.002, +0.013] | 50.3% |
| Distilled, c = 2 | Shared model | **+0.009 [+0.001, +0.018]** | 50.5% |
| Distilled, c = 1 | HHH | +0.113 [+0.094, +0.132] | 55.6% |
| Shared model | HHH | +0.114 [+0.094, +0.132] | 55.7% |

**Readings:**

- **One step helps, slightly, against the policy it was computed from.**
  - Both point estimates are positive, and the larger step gains more.
  - The c = 2 interval excludes zero, which meets the pre-registered "promising"
    criterion, but narrowly. With two configurations tested, its lower bound
    (+0.001) would not survive a multiple-comparison correction. Treat this as
    weak positive evidence.
- **The step is small by construction.** The policy is peaked, so a clipped tilt
  of e<sup>±c</sup> rarely overturns its first choice. Play changed on only about
  3% of held-out decisions, so a small effect is expected.
- **Generalization from 3,600 states is weak.** The fit improves a lot on
  searched states (KL −44% / −40%) but only 3–4% on held-out ones.
- **No transfer to HHH** (+0.113 vs +0.114). This is consistent with section
  4.1: the improvement is guaranteed against the fixed policy whose values were
  searched, and HHH is a different opponent. The change is also too small to see
  against HHH at this resolution.
- **About noise.** The search disagrees with the policy on about 61% of states.
  A large part of that is the 16-world noise measured in section 6 (independent
  searches agree on the best move only about 60% of the time). Soft targets
  absorb it, but it limits how much one step can learn.

**Decision.** The mechanism points the right way, but one offline step from
3,600 states is worth only about +0.01 marks/hand. For comparison, test-time
search is worth +0.34. Any real gain must come from **iterating** and from many
more searched states:

- In-training search at *p* = 1% would search about 36 states per update, or
  about 37,000 per 1,024-update run (10× this pilot). It would also re-search
  from each improved policy.
- The cheapest next test of compounding is a second offline round: search again
  with the distilled model as guide, distill again, and compare round 2 with
  round 1 and with the shared model. That needs no new training code. (Done:
  section 6.3.)

### 6.3 Second distillation round: do the gains compound?

**Protocol.** Identical to round 1 except for the starting point and the data:

- The round-1 c = 2 model is both the policy being improved and the search
  guide. That is the integrated loop: a better policy plays the rollouts.
- Its states come from its **own** greedy self-play: 1,000 new hands, seed
  20261027, 4,000 states, 400 of them held out.
- Fine-tuning uses the same c = 2 CMPO step.
- Arena: the same 4,096 deals (seed 20261026), so round-1 and round-2 results
  are measured on identical deals.
- **Pre-registered rule:** the gains compound if round 2 beats round 1
  head-to-head and beats the shared model by more than round 1 did.

Outputs are in `runs/model-guided/pilot/round2/`.

**Diagnostics:**

- Early stopping chose epoch 2, so the step was smaller than in round 1. Play
  choices changed on 3.5% of training states and 1.5% of held-out states.
  KL(π<sub>new</sub> ‖ π<sub>old</sub>) on held-out states was 0.0044, half of
  round 1's 0.0089.
- The search still disagrees with the policy on 63–64% of states. That is no
  fewer than in round 1, which supports reading most of the disagreement as
  16-world search noise rather than policy error.

**Results:**

| A | B | A marks/hand (95% CI) | A win rate |
| --- | --- | --- | --- |
| Round 2 | Round 1 (c = 2) | +0.005 [−0.002, +0.014] | 50.3% |
| Round 2 | Shared model | **+0.015 [+0.005, +0.027]** | 50.8% |
| Round 1 (c = 2), for reference | Shared model | +0.009 [+0.001, +0.018] | 50.5% |
| Round 2 | HHH | +0.116 [+0.097, +0.135] | 55.8% |
| Shared model, for reference | HHH | +0.114 [+0.094, +0.132] | 55.7% |

**Readings:**

- **The gains add up, roughly.** Round 2's margin over the shared model
  (+0.015) is close to the sum of the two steps (+0.009 then +0.005). Its
  interval now clearly excludes zero, a more robust result than round 1's.
- **The rule is half met.** Round 2 does beat the shared model by more than
  round 1 did. But its head-to-head step over round 1 is positive with an
  interval that includes zero. This is evidence *consistent with* compounding,
  not proof of it.
- **Still no transfer to HHH.** +0.116 vs +0.114 is within noise. The
  improvements so far are against the self-play policy family that was searched,
  as section 4.1 predicts.
- **The rate is slow.** About +0.005–0.009 marks/hand per round of 3,600
  searched states, with the step size limited by early stopping and the
  policy's peakedness.
  - In-training search at *p* = 1% searches about 3,600 states every 100
    updates, roughly one pilot round's worth, so a 1,024-update run would supply
    about ten rounds of searched states.
  - If the per-round gain held, that would be on the order of +0.05–0.1
    marks/hand against the source policy. This is an extrapolation, not a
    measurement. PPO's own updates may overlap with or wash out part of it.

**Decision.** Building the training integration (P1, P3, P4) with full rollouts
at *p* ≈ 1% is justified as an experiment. The expected effect is modest and
must be confirmed at equal compute (P5). Cheaper ways to get larger steps per
searched state are open questions: more worlds per search to cut target noise,
and searching only high-entropy states.

## 7. Options compared

| Option | What changes | On-policy PPO kept? | Collapse risk | Compute | Verdict |
| --- | --- | --- | --- | --- | --- |
| (a) ExIt distillation | Extra cross-entropy loss toward a search policy (softmax of Q̂) | Yes; separate supervised term | Medium: sharp targets | Search on a fraction of states | Viable with a temperature; the CMPO KL target is its regularized form |
| (b) Search advantages in PPO ("vine PPO") | At a fraction of states, add every action's search advantage to PPO's surrogate | Yes: weighting by π<sub>old</sub>(a) makes it an exact expectation of the surrogate | Low with clipping | Same as (a) | **Recommended** |
| (c) Search as behavior policy | Act with search during collection | No; the data comes from the search policy, so PPO ratios are wrong | High | Search on every decision | Not recommended |
| (d) Test-time search only | Nothing in training | n/a | None | Only at evaluation | Baseline (exists) |
| (e) Decision-time RL fine-tuning | Fine-tune a copy of the network per decision (Fickinger et al. 2021; Suphx) | n/a | Low | Very high per decision | Out of scope |
| (f) Privileged leaves or targets | The centralized critic or oracle as a search leaf | Leaves only | High if used as actor targets: perfect-information bias | Critic is cheap; oracle is slow | The critic's main head is a candidate leaf (P2); never an actor target |

Why (b) over (a): the two can be combined, but (b) works inside PPO's existing
clipped objective and trust region (sections 4.2–4.3). The search simply supplies
better advantage estimates, and clipping already limits how far one update moves
the policy.

## 8. Recommended design

1. **Which decisions.** Learner decisions in the play phase, each chosen with
   probability *p* (for example 1–5%) from a dedicated seeded stream, so results
   stay worker-count invariant. Bidding and trump come later (section 11).
2. **The search.** Model-guided PIMC from the learner's own `Observation`
   (`texas42.belief.evaluate_actions`):
   - the current rollout snapshot plays all seats (pair-aware in paired mode);
   - belief worlds are uniform or model-weighted, never the true deal;
   - *S* = 8–16 sampled worlds;
   - the rollout length is chosen by the gate (section 6), not by assumption:
     - **full rollouts, points objective:** rank moves best; cost about 4.5× the
       leaf;
     - **a truncated rollout with a leaf** only if that leaf passes the ranking
       gate. Neither the acting-seat nor the own-turn value leaf does (sections 6
       and 6.1). Remaining candidates: the world-value leaf (section 4.6) and a
       value head trained to separate alternatives (P2 option 3).
3. **Search advantages.** For each legal action a: Q̂(a) from the search,
   V̂ = Σ<sub>a</sub> π<sub>old</sub>(a) Q̂(a), Â<sub>s</sub>(a) = Q̂(a) − V̂.
   Normalize per batch, as PPO already does for GAE advantages.
4. **Loss.**
   - L = L<sub>PPO</sub>(GAE, sampled action) + λ · L<sub>search</sub>.
   - L<sub>search</sub> is the all-action clipped surrogate (section 4.3), averaged
     over searched states only.
   - **CMPO variant:** target π̄(a) ∝ π<sub>old</sub>(a)·exp(clip(Â<sub>s</sub>(a)/σ, −c, c))
     with L<sub>search</sub> = KL(π̄ ‖ π<sub>θ</sub>). The step is bounded by
     KL ≤ 2c (section 4.4).
5. **Value.** Keep PPO's value loss on environment returns. Optionally add a small
   auxiliary loss toward V̂.
6. **Safeguards and diagnostics** (logged per update):
   - entropy per phase;
   - KL to the anchor (the existing forgetting diagnostic);
   - search/policy disagreement rate: argmax Q̂ ≠ argmax π<sub>old</sub>;
   - mean |Â<sub>s</sub>|;
   - fraction of searched decisions;
   - search seconds;
   - the existing target-KL early stop.

   Start with λ = 0.25, and *p* = 0.5–1% with full rollouts (or 2–5% with a leaf
   that has passed the gate). Turn search on only after a warm-up of ordinary
   PPO.

## 9. Compute budget

**Measured search cost.** `examples.search_accuracy`, shared bid+play study model;
200 play decisions sampled uniformly across the hand; 16 sampled worlds; one CPU
thread (Windows development machine with 20 logical CPUs, other work running):

| Search mode | Mean s/search (model table) | Max | Mean s/search (heuristic table) |
| --- | --- | --- | --- |
| Heuristic rollouts, full hand | 0.148 | 1.22 | 0.160 |
| Model rollouts, full hand | 0.576 | 4.98 | 0.630 |
| Model rollouts + value leaf | 0.123 | 1.32 | 0.145 |

**Training throughput (measured).** From `runs/study-shared/progress.jsonl`, the
shared bid+play study run: 262,144 hands in 1,024 updates of 4,096 learner
decisions, 8 collection workers, 4 torch threads.

| Quantity | Value |
| --- | --- |
| Hands per update | 256 |
| Learner decisions per hand | 16, of which 14 are play decisions |
| Collection | 1,108 s in total: 236.5 hands/s, i.e. 0.034 s of worker time per hand |
| Optimization | 2,385 s in total; it runs after each collection, not alongside it |
| Collection + optimization | **75 hands/s** (0.0133 s per hand); validation adds another 232 s over the run |

**Optimization, not collection, dominates training time.** Search runs only during
collection.

**Slowdown estimate.** Searching a fraction *p* of play decisions adds *p* × 14 ×
(seconds per search) of worker time per hand. That is spread over the 8 workers,
so it adds *p* × 14 × t / 8 of wall time per hand to the 0.0133 s:

| *p* | Leaf-cost search (t = 0.13 s) | Throughput | Full-rollout search (t = 0.6 s) | Throughput | Searched states per update |
| --- | --- | --- | --- | --- | --- |
| 0.5% | 1.09× | ≈ 69 hands/s | 1.39× | ≈ 54 hands/s | ≈ 18 |
| 1% | 1.17× | ≈ 64 hands/s | 1.79× | ≈ 42 hands/s | ≈ 36 |
| 2% | 1.34× | ≈ 56 hands/s | 2.58× | ≈ 29 hands/s | ≈ 72 |
| 5% | 1.85× | ≈ 41 hands/s | 4.9× | ≈ 15 hands/s | ≈ 179 |

The estimate assumes three things: search costs the same in a worker as in the
single-thread measurement, collection parallelizes across the 8 workers, and the
search loss adds negligible optimization time.

**Conclusions:**

- **Full rollouts fit a 2× budget up to *p* ≈ 1%.** That is about 1.8× today's
  training (*p* = 0.5% is about 1.4×). Restricting full rollouts to late-hand
  positions, where rollouts are short, would stretch the budget further.
- **A verified cheap leaf** would allow *p* ≈ 2–5% within roughly 1.3–1.9× the
  current cost.
- **Searched states are a small share of each update.** At *p* = 1% there are
  about 36 searched states among an update's 4,096 learner decisions (about 3,600
  of them play decisions). Each contributes every legal action, but the
  search-loss weight λ and *p* must be tuned together so that its signal is not
  swamped.
- **Optimizations not yet applied:**
  - batching rollouts across worlds into one network call;
  - fewer sampled worlds (8);
  - searching only positions where the policy is uncertain (high entropy), which
    targets the samples most likely to change the policy.

**Caveat.** These are single-thread timings on one machine with other work
running, using positions sampled uniformly within the hand. Early-hand positions
cost up to about 10× the mean. Treat the numbers as planning estimates.

## 10. Implementation plan

Each phase has an owner module, tests and an acceptance gate. A phase starts only
when the previous gate passes. Defaults must reproduce current training exactly
(search off).

**P0. Move-ranking gate (done).**
- `examples/search_accuracy.py --rank-check`; `--partner-model` for paired guides.
- Result: section 6.
- Acceptance: a candidate cheap search is usable for training only if its ranking
  regret against full model rollouts (points objective) is no larger than that of
  plain heuristic-rollout search, and its top-1 agreement is at least as high.

**P1. In-memory search guides.**
- `research/ppo/search.py`: build rollout factories and a leaf from in-memory
  network weights (the collector's rollout snapshot), not only from a
  `CheckpointIdentity`. The learner has no checkpoint file mid-rollout.
- Pair-aware in paired mode (already true of `model_factories` and `ValueLeaf`).
- Tests: in-memory and checkpoint-built guides give identical Q̂ for the same
  weights and seeds; picklable for spawn workers.

**P2. A cheap leaf that ranks (only if a leaf is wanted).** Candidates, in order:
1. **Own-turn leaf (done; fails the gate, section 6.1):** `until='turn'` in
   `texas42.belief.simulate` / `evaluate_actions` and `--pimc-leaf value-turn`,
   using the searcher's own value head (unbiased in principle by the tower
   property; section 4.6).
2. **World-value leaf:** the centralized critic's main head on hypothetical worlds
   (`encode_privileged(observation, world hands)`), for runs trained with
   `--critic-mode centralized`.
3. **Points-scale information-set head:** an auxiliary head predicting the actor
   team's remaining points. This changes the model format, so it needs explicit
   checkpoint versioning in `research/ppo/checkpoint.py`.
- Tests: leaf sign and phase rules (as for `ValueLeaf`); `until` semantics; no
  real-deal input.
- Acceptance: the P0 criterion with the new leaf, plus an arena check that leaf
  search is not worse than full-rollout search beyond the confidence interval.

**P3. Search in the collector.**
- `research/ppo/selfplay_rollout.py` (`collect_phase_hand`): for each learner play
  decision, a per-episode draw from `stream_seed(seed, 'search-decisions-v1',
  episode)` decides whether to search. Search over the learner's own belief
  worlds, never `remaining_hands()`.
- `research/ppo/buffer.py` (`Experience`): optional `search_q`, carried to
  `tensors()` as a 58-wide array plus mask.
- Config (`SelfPlayConfig`, `examples/train_selfplay_ppo.py`): `search_fraction`
  (default 0 = off), `search_samples`, `search_leaf`, `search_rollouts`
  ∈ {greedy, sampled}, `search_warmup_hands`.
- Tests: fraction 0 is bit-identical to today; 1-vs-N worker invariance with
  search on; searched decisions and Q̂ are deterministic given the seed.

**P4. The loss.**
- `research/ppo/trainer.py` (`Trainer.update`): `search_weight` λ,
  `search_target` ∈ {advantage, cmpo}, `search_clip` c. Per-phase metrics: entropy,
  anchor KL, disagreement rate, mean |Â<sub>s</sub>|, searched fraction, search
  seconds.
- Checkpoints (`selfplay_checkpoint.py`): config fields only; no format change.
- Tests: on hand-made batches, the gradient of L<sub>search</sub> at
  θ<sub>old</sub> equals the all-action gradient (section 4.3), and the CMPO target
  respects KL ≤ 2c. A tiny seeded update/resume fixture covers both generations.

**P5. Experiments.**
- **Baseline:** the joint bid+play recipe (shared mode) from the same
  initialization.
- **Treatments:** *p* ∈ {0.5%, 1%} with full rollouts (higher with a gated leaf);
  λ ∈ {0.25, 1}; target ∈ {advantage, CMPO}.
  Ablations: rollouts greedy vs sampled; full rollouts vs the P2 leaf.
- **Budgets:** compare at equal hands *and* at equal CPU time. Treatments get fewer
  hands in the equal-time comparison.
- **Seeds:** at least 3 training seeds per arm. Report means and spread across
  seeds, not only within-run bootstrap intervals.
- **Evaluation:** the held-out duplicate arena vs HHH, the shared baseline and the
  anchor (`evaluate_agent_vs_agent`).
- **Decision rule:** adopt only if the equal-CPU comparison is significantly
  positive in at least 2 of 3 seeds, with stable entropy and anchor KL.

**P6. Search on top of the integrated model.**
- Re-run `search_accuracy --rank-check` with each final model as guide, and the
  arena with model-guided PIMC (full rollouts) guided by the integrated model vs
  by the PPO-only model.
- This tests section 4.5's claims 1 and 3: J(𝓘(π<sub>int</sub>)) vs
  J(𝓘(π<sub>PPO</sub>)), and π<sub>int</sub> vs π<sub>PPO</sub> without search.
- **Preliminary, with the offline round-2 network as guide**
  ([1,024-deal results](belief.md#model-guided-results)):
  - head-to-head against shared-guided search: +0.027 [−0.004, +0.061]
    marks/hand;
  - against plain PIMC: +0.044 vs +0.045 for the shared guide, on the same
    deals.
  - So a +0.015 better network has not yet produced a measurably better search.
    Claim 3 needs a larger gain in the network itself.

## 11. Open questions

- **Which cheap leaf ranks well?** The own-turn leaf removes the viewpoint bias
  but still fails the gate, because the PPO value head's approximation error
  dominates (section 6.1).
  - The world-value leaf (centralized critic) is untested.
  - A head trained to separate alternative moves (for example on search values,
    or in points) is the remaining route.
- **How far do distilled improvements compound?** Two offline rounds gave +0.009,
  then +0.015 in total (sections 6.2–6.3).
  - Whether many rounds, or in-training search with about ten rounds' worth of
    searched states per run, approach a meaningful share of test-time search's
    +0.34 is the main open question.
  - So is whether the gains ever transfer to opponents outside the self-play
    family (HHH).
- **How many worlds per training search?**
  - At 16 worlds, two independent searches agree on the best move only about 60%
    of the time (section 6).
  - Soft targets tolerate that noise, but more worlds would sharpen every target,
    at linear cost.
  - Spending the budget on more worlds at fewer states vs fewer worlds at more
    states is a P5 ablation.
- **Weak early value heads.** A leaf from a barely trained network may be worse
  than a heuristic rollout. A warm-up period, or full rollouts at first, may be
  needed.
- **Bidding and trump.** Their effects are only known at the end of the hand, so
  searching them needs full rollouts, which is expensive. Start with play only.
- **Belief choice.** Model-weighted beliefs are more accurate when the model plays
  the table ([results](belief.md#model-guided-results)), but they cost a replay
  per world.
- **Paired partners.** Search guides and leaves are already pair-aware (section 6
  uses a paired guide). Paired training would need the per-partner rollout
  snapshots in the collector.
- **Equilibrium vs best response.** All guarantees here are against fixed other
  seats. Whether search-trained self-play cycles more or less than plain PPO
  self-play is open; held-out and cross-play evaluation are the checks.

## 12. References

**Policy iteration, rollouts and approximate policy iteration**

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press 2018 (policy improvement theorem, §4.2) — http://incompleteideas.net/book/the-book-2nd.html
- Bertsekas, *Rollout, Policy Iteration, and Distributed Reinforcement Learning*, Athena Scientific 2020 — https://web.mit.edu/dimitrib/www/dp_rollout_book.html
- Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming*, Athena Scientific 1996 (approximate policy iteration error bound).
- Tesauro & Galperin, *On-line Policy Improvement using Monte-Carlo Search*, NIPS 1996 — https://papers.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html
- Kakade & Langford, *Approximately Optimal Approximate Reinforcement Learning* (CPI, performance difference lemma), ICML 2002 — https://mlanthology.org/icml/2002/kakade2002icml-approximately/
- Lagoudakis & Parr, *Reinforcement Learning as Classification: Leveraging Modern Classifiers* (RCPI), ICML 2003 — https://mlanthology.org/icml/2003/lagoudakis2003icml-reinforcement
- Lazaric, Ghavamzadeh & Munos, *Analysis of Classification-based Policy Iteration Algorithms*, JMLR 17, 2016 — https://jmlr.org/papers/v17/10-364.html
- Scherrer, *Approximate Policy Iteration Schemes: A Comparison*, ICML 2014 — https://proceedings.mlr.press/v32/scherrer14.html
- Sun, Gordon, Boots & Bagnell, *Dual Policy Iteration*, NeurIPS 2018 — https://arxiv.org/abs/1805.10755
- Sutton, *Dyna, an Integrated Architecture for Learning, Planning, and Reacting*, 1991 — https://papersdb.cs.ualberta.ca/~papersdb/view_publication.php?pub_id=500

**Policy gradients, trust regions and variance reduction**

- Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE), Machine Learning 8, 1992.
- Schulman et al., *Trust Region Policy Optimization* (vine procedure), ICML 2015 — https://arxiv.org/abs/1502.05477
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 — https://arxiv.org/abs/1707.06347
- Ciosek & Whiteson, *Expected Policy Gradients*, AAAI 2018 — https://ojs.aaai.org/index.php/AAAI/article/view/11607
- Allen, Asadi, Roderick, Mohamed, Konidaris & Littman, *Mean Actor Critic*, 2017 — https://arxiv.org/abs/1709.00503

**Regularized policy improvement**

- Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation* (MPO), ICLR 2018 — https://arxiv.org/abs/1806.06920
- Song et al., *V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control*, ICLR 2020 — https://arxiv.org/abs/1909.12238
- Peng, Kumar, Zhang & Levine, *Advantage-Weighted Regression*, 2019 — https://arxiv.org/abs/1910.00177
- Hessel et al., *Muesli: Combining Improvements in Policy Optimization* (CMPO), ICML 2021 — https://arxiv.org/abs/2104.06159
- Grill et al., *Monte-Carlo Tree Search as Regularized Policy Optimization*, ICML 2020 — https://arxiv.org/abs/2007.12509
- Jacob et al., *Modeling Strong and Human-Like Gameplay with KL-Regularized Search* (piKL), ICML 2022 — https://arxiv.org/abs/2112.07544
- Sokota et al., *The Update-Equivalence Framework for Decision-Time Planning*, 2023 — https://arxiv.org/abs/2304.13138
- Sokota et al. (2023b), *A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games* (magnetic mirror descent), ICLR 2023 — https://arxiv.org/abs/2206.05825

**Search and learning combined**

- Anthony, Tian & Barber, *Thinking Fast and Slow with Deep Learning and Tree Search* (Expert Iteration), NeurIPS 2017 — https://arxiv.org/abs/1705.08439
- Silver et al., *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play* (AlphaZero), Science 2018 — https://arxiv.org/abs/1712.01815
- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), Nature 2020 — https://arxiv.org/abs/1911.08265
- Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022 — https://iclr.cc/virtual/2022/poster/6418
- Hamrick et al., *On the Role of Planning in Model-Based Deep Reinforcement Learning*, ICLR 2021 — https://arxiv.org/abs/2011.04021
- Ross, Gordon & Bagnell, *A Reduction of Imitation Learning to No-Regret Online Learning* (DAgger), AISTATS 2011 — https://arxiv.org/abs/1011.0686
- Rusu et al., *Policy Distillation*, ICLR 2016 — https://arxiv.org/abs/1511.06295
- Fickinger et al., *Scalable Online Planning via Reinforcement Learning Fine-Tuning*, NeurIPS 2021 — https://arxiv.org/abs/2109.15316

**Imperfect-information games**

- Frank & Basin, *Search in Games with Incomplete Information: A Case Study using Bridge Card Play* (strategy fusion), Artificial Intelligence 100, 1998.
- Ginsberg, *GIB: Imperfect Information in a Computationally Challenging Game*, JAIR 14, 2001.
- Buro, Long, Furtak & Sturtevant, *Improving State Evaluation, Inference, and Search in Trick-Based Card Games*, IJCAI 2009.
- Long et al., *Understanding the Success of Perfect Information Monte Carlo Sampling*, AAAI 2010 — https://ojs.aaai.org/index.php/AAAI/article/view/7562
- Cowling, Powley & Whitehouse, *Information Set Monte Carlo Tree Search*, IEEE TCIAIG 2012 — https://eprints.whiterose.ac.uk/75048
- Solinas, Rebstock & Buro, *Improving Search with Supervised Learning in Trick-Based Card Games*, AAAI 2019 — https://ojs.aaai.org/index.php/AAAI/article/view/3909
- Rebstock et al., *Policy Based Inference in Trick-Taking Card Games*, 2019 — https://arxiv.org/abs/1905.10911
- Lerer et al., *Improving Policies via Search in Cooperative Partially Observable Games* (SPARTA), AAAI 2020 — https://ojs.aaai.org/index.php/AAAI/article/view/6208
- Hu, Lerer, Brown & Foerster, *Learned Belief Search*, 2021 — https://arxiv.org/abs/2106.09086
- Tian, Gong & Jiang, *Joint Policy Search for Multi-agent Collaboration with Imperfect Information*, NeurIPS 2020 — https://arxiv.org/abs/2008.06495
- Brown, Bakhtin, Lerer & Gong, *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games* (ReBeL), NeurIPS 2020 — https://arxiv.org/abs/2007.13544
- Schmid et al., *Player of Games*, 2021 — https://arxiv.org/abs/2112.03178
- Blüml, Czech & Kersting, *AlphaZe∗∗: AlphaZero-like Baselines for Imperfect Information Games Are Surprisingly Strong*, Frontiers in AI 2023 — https://doi.org/10.3389/frai.2023.1014561
- Zha et al., *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*, ICML 2021 — https://arxiv.org/abs/2106.06135
- Yang et al., *PerfectDou: Dominating DouDizhu with Perfect Information Distillation*, NeurIPS 2022 — https://arxiv.org/abs/2203.16406
- Li et al., *Suphx: Mastering Mahjong with Deep Reinforcement Learning*, 2020 — https://arxiv.org/abs/2003.13590
