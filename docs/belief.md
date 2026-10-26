# Belief modelling over hidden dominoes

`texas42/belief.py` reasons about the dominoes one seat cannot see. From an
ordinary `Observation` it can:

1. derive what is **certain** about the hidden deal (hand sizes, revealed voids);
2. **count** the deals still possible, and give each unseen domino's exact
   probability of being in each hand;
3. **list** or **sample** those deals as hypothetical "worlds";
4. **weight** worlds by how well they explain the other players' moves;
5. **simulate** play forward in a world from a chosen sequence of moves;
6. **estimate the expected points** of every legal action, averaged over worlds.

This is roadmap step 7 ([roadmap](roadmap.md)). It does not depend on any
perfect-information solver. A future oracle can plug in as a leaf evaluator
(see [Using a perfect-information evaluator](#using-a-perfect-information-evaluator)).
The module is dependency-free and sits in `texas42/` beside the other game-level
tools. No learning code is involved.

## Contents

- [Why this is tractable in Texas 42](#why-this-is-tractable-in-texas-42)
- [Pipeline and data formats](#pipeline-and-data-formats)
- [Hard evidence: constraints](#1-hard-evidence-constraints)
- [Exact counting and marginals](#2-exact-counting-and-marginals)
- [Enumeration and exact uniform sampling](#3-enumeration-and-exact-uniform-sampling)
- [Rebuilding a simulator state for a world](#4-rebuilding-a-simulator-state-for-a-world)
- [Soft evidence: policy likelihoods](#5-soft-evidence-policy-likelihoods)
- [Simulation (rollouts)](#6-simulation-rollouts)
- [Evaluating actions](#7-evaluating-actions)
- [The ActionProbabilities agent query](#the-actionprobabilities-agent-query)
- [Measured behaviour](#measured-behaviour)
- [The PIMC agent](#the-pimc-agent)
- [Model-guided PIMC](#model-guided-pimc)
- [Choosing settings](#choosing-settings)
- [Limitations and next steps](#limitations-and-next-steps)
- [Research background](#research-background)

## Why this is tractable in Texas 42

The only hidden information in a hand is the initial deal. After the shuffle
there are no draws or other chance events, which is the same structure as bridge
and Skat. Every rule-based constraint on the hidden deal can be stated per domino
("seat s cannot hold this domino because s failed to follow one of its suits"),
so the number of consistent deals can be counted exactly with a small dynamic
program (DP).

Consistent deals from one seat's point of view, when nobody has revealed a void:

| Point in the hand | Unseen dominoes (per other seat) | Consistent deals |
| --- | --- | --- |
| Bidding / opening lead | 21 (7/7/7) | 399,072,960 |
| After trick 1 | 18 (6/6/6) | 17,153,136 |
| After trick 2 | 15 (5/5/5) | 756,756 |
| After trick 3 | 12 (4/4/4) | 34,650 |
| After trick 4 | 9 (3/3/3) | 1,680 |
| After trick 5 | 6 (2/2/2) | 90 |
| After trick 6 | 3 (1/1/1) | 6 |

Voids only reduce these numbers. Late in a hand every world can be listed,
so expectations become exact rather than estimated.

## Pipeline and data formats

Seats are absolute (0–3, team = seat % 2). Dominoes are permanent IDs 0–27
(`PLAY_ACTIONS[id]` decodes them). Actions are permanent global IDs.

```text
Observation                    the observer's own view (selfplay.py)
  │ hidden_constraints()
  ▼
HiddenConstraints              unknown IDs, open slots per seat, allowed seats per ID
  │ count_worlds(), tile_marginals()        exact integers / probabilities
  │ enumerate_worlds(), sample_world()      worlds
  ▼
PrivilegedState                remaining hands of all 4 seats (agents/base.py)
  │ world_state()              rebuild deal + replay public history via HandState.apply
  ▼
HandState                      a full engine state for ONE hypothetical world (state.py)
  │ build_belief()             -> Belief (worlds + normalized weights)
  │ simulate()                 -> Rollout (action IDs, probability, points)
  │ evaluate_actions()         -> ActionEvaluation (expected points per legal action)
```

### Privacy contract

Every function takes only the observer's `Observation` plus an explicit
`random.Random`. Worlds and `HandState`s created here are hypotheses generated
from public information, not the true deal. Passing them to perfect-information
tools therefore leaks nothing. Never pass the environment's real state in. Tests
read the true deal only to check answers.

### Reproducibility

All randomness comes from caller-supplied `random.Random` streams; there are no
module-level random calls. Equal inputs and equal seeds give equal outputs (this
is tested). `evaluate_actions` draws belief samples first, then one rollout seed
per world.

## 1. Hard evidence: constraints

`hidden_constraints(obs) -> HiddenConstraints`

Certain knowledge comes from two places:

- **Counting.** Every seat started with 7 dominoes, so seat *s* holds
  7 − (dominoes *s* has played) of the unseen ones.
- **Revealed voids.** Rules force a player to follow the led suit when able.
  Playing off-suit proves the player held none of that suit, and hands only
  shrink, so the void lasts for the rest of the hand. The computation reuses
  `encoding.revealed_voids`, and suit membership under trump comes from
  `rules.follows_suit`.

Bids and trump calls create no certainties under the base rules; any hand may
make any legal call. They are soft evidence (section 5).

| Field | Format | Meaning |
| --- | --- | --- |
| `observer` | int | Seat that observed |
| `observer_hand` | sorted IDs | Known exactly |
| `unknown` | sorted IDs | Neither in the observer's hand nor played |
| `slots` | 4 ints | Unseen dominoes each seat still holds (0 for the observer); sums to `len(unknown)` |
| `allowed` | tuple per unknown ID | Seats that may hold it |
| `voids` | set of `(seat, suit)` | Suit is 0–6 or `"doubles"` |
| `trump` | trump value | `None` for Follow Me or before trump |

An internally inconsistent observation raises `ValueError`.

## 2. Exact counting and marginals

`count_worlds(constraints) -> int`,
`tile_marginals(constraints) -> {id: (p0, p1, p2, p3)}`

**Method.** Unknown dominoes with the same set of allowed seats are
interchangeable. With three other seats there are at most 7 such groups. For
each group, the DP chooses how many of its *c* dominoes go to each allowed seat
(a *split*, k_s ≥ 0, summing to *c* and within each seat's remaining capacity),
and multiplies by the multinomial *c*! / ∏ k_s! ways to choose which dominoes:

```text
ways(g, capacity) = Σ_split multinomial(c_g; split) · ways(g+1, capacity − split)
ways(end, capacity) = 1 if capacity is all zero else 0
```

Python integers keep counts exact. This is the counting idea used by exact
Minesweeper and Battleship probability solvers.

**Why each consistent world is equally likely.** The shuffle is uniform, and
conditioning a uniform distribution on a logical event leaves it uniform over the
surviving outcomes. With no voids, the marginals reduce to the bridge
**vacant places** rule: an unseen domino is in seat *s* with probability
slots[s] / Σ slots (tested: 1/3 each during bidding).

**Marginals.** `tile_marginals` computes P(domino *d* in seat *s*) as the count
with *d* fixed in *s*, divided by the total count. That is the same DP with *d*
removed and slots[s] reduced by one. Rows sum to 1, and are 0 for the observer
and for disallowed seats.

**When to use.** Use the marginals as a baseline belief, as ground truth for
learned belief models (roadmap variants B/C), or to measure how fast play reveals
information (RQ2).

## 3. Enumeration and exact uniform sampling

`enumerate_worlds(constraints, *, limit=None)` yields every consistent world once,
in a deterministic order. This is the "keep the whole possibility set" approach of
StrangeFish in Reconnaissance Blind Chess. `limit` makes an accidental huge
enumeration fail fast.

`sample_world(constraints, rng)` draws one world **exactly uniformly**. It walks
the groups, picks a split in proportion to multinomial(split) × ways(remaining),
then shuffles the group's dominoes and deals them by that split. The probabilities
multiply out to exactly 1 / count for every world. There is no rejection loop and
no floating-point bias: `rng.randrange` is used on exact integers. A chi-square
test checks uniformity.

## 4. Rebuilding a simulator state for a world

`world_state(obs, world) -> HandState` rebuilds the hypothetical **initial**
deal: each seat's remaining hand plus what it has played. It recovers the shaker
(from the bid view, or the seat before the first caller). It then replays every
event from `history.public_history(obs.view)` through `HandState.apply`.

- The engine validates every move, so a world that contradicts the public record
  raises `ValueError` instead of silently diverging. No rule logic is duplicated.
- The function finally checks that it is the observer's turn with the same legal
  action IDs as the `Observation`. Tests check that every world reproduces the
  observation, apart from the dealt order within a hand, which carries no
  information.
- The returned state is private and mutable: step it, copy it, roll it out, or
  hand it to a solver.

## 5. Soft evidence: policy likelihoods

`build_belief(obs, *, rng, samples=64, enumerate_below=256, likelihood_policies=None, epsilon=0.1) -> Belief`
and `policy_likelihood(obs, world, policies, *, epsilon=0.1) -> float`

**Theory (Bayes' rule over deals).**

```text
P(world | public history) ∝ P(world) · Π over other players' decisions t of P(a_t | that player's view at t, in this world)
```

P(world) is uniform over consistent worlds. The product is the **likelihood**:
how probable the observed bids, trump call and plays would be if this world were
true, under an assumed model of how the other players choose. This is
Policy-Based Inference (Rebstock et al. 2019) and the bid inference of the Skat
program Kermit (Buro et al. 2009); the poker program Loki's "weight tables"
apply the same update.

**How it is computed.** `policy_likelihood` replays the world. At each other
player's decision it builds that player's `Observation` in the hypothetical world
and asks the modelled agent for `action_probabilities(obs)`, then multiplies in
the probability of the action actually taken. The observer's own decisions are
skipped: the observer's view is identical in every consistent world, so that
factor is a constant that cancels when weights are normalized.

**Why epsilon.** Deterministic models (HeuristicAgent) give probability 0 to any
move they would not make. One surprise would then eliminate the true world. Each
factor therefore becomes (1 − ε)·p + ε / |legal moves|. ε means "the real player
is not exactly my model". Note that ε = 1 does **not** give a flat belief: it
models players as choosing uniformly among their legal moves, and the number of
legal moves depends on the hidden hand. For a flat belief, omit
`likelihood_policies`.

**The `Belief` object.**

| Field | Meaning |
| --- | --- |
| `worlds`, `weights` | Hypotheses and normalized weights (sum to 1) |
| `world_count` | Exact count of hard-consistent worlds |
| `mode` | `"enumerated"`: every consistent world, so averages are exact. `"sampled"`: i.i.d. uniform draws, so averages are estimates (self-normalized importance sampling when weighted) |
| `likelihood` | `"uniform"` or `"policy"` |
| `effective_sample_size` | 1/Σw². Equals the number of worlds when weights are uniform; near 1 when one world dominates |
| `marginals()` | Weighted P(seat holds domino), same format as `tile_marginals` |

**When to use which.**

- **Uniform** is right when other players' choices are uninformative or untrusted.
  It is also the baseline for RQ2.
- **Policy** fits when a trusted model of the others exists, such as the same
  Heuristic or PPO policies used in evaluation. Bids and partner signals then
  shift the belief. It costs one replay per world.

## 6. Simulation (rollouts)

`simulate(state, *, rng, choices=(), policies=None, until="phase", track_probability=True) -> Rollout`

This plays forward from a **copy** of a `HandState`; the input is never changed.
This is the backgammon *rollout* (Tesauro & Galperin 1996), and the inner loop of
PIMC and of Recursive Monte Carlo search (Furtak & Buro 2013).

| Argument | Meaning |
| --- | --- |
| `choices` | Leading decisions as **indices** into the sorted legal action IDs at each step (0 = lowest ID). This walks a chosen "decision-tree branch". |
| `policies` | Four arena-style factories `factory(seat, seed) -> Agent` for the remaining steps. `None` means uniformly random legal moves. Agents are built fresh and see observations only from this point, so agents with memory lack earlier history. |
| `until` | `"trick"`: next completed trick. `"phase"`: bid/trump/play phase change (for play, the end of the hand). `"hand"`: end of the hand. `"turn"`: the seat that acted first is to act again (in `evaluate_actions`, the observer's own next decision). The hand ending always stops. |
| `track_probability` | Compute `rollout_probability` (policy steps then also call `action_probabilities`). |

| `Rollout` field | Meaning |
| --- | --- |
| `action_ids` | Every applied global action ID, in order (IDs stay meaningful across worlds; indices do not) |
| `explicit_steps` | How many came from `choices` |
| `rollout_probability` | Product of the modelled probabilities of the non-explicit steps (1/\|legal\| for random steps). Explicit choices are interventions ("what if I play X"), so they add no factor. `None` if not tracked or a policy cannot report probabilities. |
| `start_points`, `end_points`, `points_delta` | Team points (team 0, team 1), and the change |
| `stop_reason` | `"trick"`, `"phase"`, `"turn"` or `"hand"` |
| `hand_result` | Engine `HandResult` when the hand finished |
| `final_state` | Private `HandState` where it stopped |

The overall likelihood of a simulated path in a weighted belief is
(world weight) × `rollout_probability`.

## 7. Evaluating actions

`evaluate_actions(obs, *, rng, belief=None, rollout_policies=None, until="hand", leaf_evaluator=None, ...) -> ActionEvaluation`

**Method (PIMC with paired rollouts).** For every world *i* (weight *w_i*) and
every legal action *a*, the value is the points the observer's team gains after
playing *a* and rolling out. The estimate is Σ w_i · value_i(a). **The same worlds
and the same rollout seed are used for every action** ("common random numbers",
the variance reduction that made backgammon rollouts practical, and the same idea
as the arena's duplicate deals). Differences between actions then reflect the
actions, not deal luck.

**Objective.** The default is **expected points** gained by the observer's team.
Marks and contract-made rates are reported as diagnostics when every rollout
finishes the hand.

| `ActionEstimate` field | Meaning |
| --- | --- |
| `expected_points` | Weighted mean points for the observer's team |
| `std` | Spread across worlds: how much the answer depends on the deal |
| `standard_error` | Approximate estimator error: √Σ w²(v − mean)², which equals std/√n for n uniform samples. In enumerated mode with deterministic policies the mean is exact, and this reflects spread. |
| `advantage`, `advantage_standard_error` | Paired difference from the best action (≤ 0) and its error; usually far smaller than `standard_error` |
| `expected_signed_marks`, `contract_made_rate` | When every rollout finished the hand, else `None` |

`ActionEvaluation` adds `best_action_id` (ties go to the lowest ID),
`world_count`, `mode`, `worlds_used` and `effective_sample_size`.

**Known limitation (PIMC).** Averaging over worlds assumes that after this move
players act well in each world separately. This is *strategy fusion* and
*non-locality* (Frank & Basin 1998; Long et al. 2010). Rollouts with ordinary
policies, which see only their own observations, avoid the strongest form of
this. A perfect-information leaf evaluator reintroduces it after the stopping
point. Stopping later (`until="trick"` before solving) is the EPIMC remedy
(Arjonilla, Saffidine & Cazenave 2024).

### Using a perfect-information evaluator

`leaf_evaluator(state, observer_seat) -> float` must return the expected
**additional** points for the observer's team from `state`. It is called only
for rollouts that stopped before the hand ended. Because `state` is a
hypothetical world, a full-information solver (for example the oracle under
development) can be used without leaking hidden information. A typical setup:
`until="trick"`, with the solver's point value as the leaf.

## The ActionProbabilities agent query

`agents.base.ActionProbabilities` is an optional, **pure** query:

```python
def action_probabilities(self, obs: Observation) -> dict[int, float]: ...
```

It returns the distribution `act(obs)` samples from: global IDs (a subset of
`obs.legal_action_ids`) mapped to probabilities summing to 1. Like
`BidPreference`, it must not change memory, consume RNG, or call a stateful
`act`. Belief weighting and rollout probabilities depend on it.

| Agent | Distribution |
| --- | --- |
| `RandomAgent` | Uniform over legal IDs |
| `ConservativeRandomAgent` | One-hot bid (its deterministic rule, now a shared helper); uniform trump and play |
| `HeuristicAgent` | One-hot on its deterministic choice |
| `HybridAgent` | Delegates to the acting phase's policy |

Adding it changes no `act` behaviour; a test runs agents in lockstep to confirm
that querying consumes no randomness. A PPO policy can implement it as its
masked softmax.

## Measured behaviour

**Workload.** Two seeded hands (seeds 3 and 11) played by Heuristic agents; one
opening-lead position per trick; `evaluate_actions` with HHH rollouts, 64
samples, `enumerate_below=256`. Measured on one Windows 11 development machine,
Python 3.13, a single process, with other work running. These are rough
magnitudes, not benchmarks.

| Tricks done | Worlds | Mode | Legal actions | Time |
| --- | --- | --- | --- | --- |
| 0 | 399,072,960 | sampled (64) | 7 | 0.65–1.9 s |
| 2 | 16,632–252,252 | sampled | 5 | 0.33–1.1 s |
| 4 | 400–1,680 | sampled | 3 | 0.15–0.44 s |
| 5 | 36–90 | enumerated | 2 | 0.10–0.13 s |

Cost is roughly (worlds) × (legal actions) × (one rollout). Enumerating 1,680
worlds took about 16 s before rollouts used a cheap state copy and skipped
unused probability queries.

**Weight collapse with deterministic models.** With HHH likelihoods and ε = 0.1,
the effective sample size of 64 sampled worlds fell from about 60 at the opening
lead to as low as 1.1 by trick 3–4 in these hands. A deterministic model makes
most worlds explain the history poorly, and each unexplained move multiplies a
world's weight by about ε/|legal|. This is the weight degeneracy known from
particle filters and Loki-style weighting. Options: larger ε, more samples,
stochastic models, or resampling. Check `effective_sample_size` before trusting
policy-weighted estimates.

## The PIMC agent

`texas42.agents.pimc.PIMCAgent(seed, *, samples=32, enumerate_below=256)` wraps
`evaluate_actions` as an ordinary `Agent` (`reset(seat)`, `act(obs) -> int`).

| Phase | Decision |
| --- | --- |
| Bid | Delegated to `HeuristicAgent` (expected points is a poor bidding objective) |
| Trump, play | `evaluate_actions` with Heuristic V1 rollouts for all seats and a uniform belief, choosing `best_action_id` |
| Forced move | Returned directly; no search, no random draws |

- **Randomness:** one `random.Random(seed)` per instance supplies belief sampling
  and rollout seeds, so equal seeds and equal observation sequences give equal
  actions. `reset` does not rewind the stream.
- **Diagnostics:** `last_evaluation` keeps the latest `ActionEvaluation` for
  inspection.
- **No `action_probabilities`:** decisions consume RNG, so the agent cannot
  answer the pure query and cannot serve as a likelihood model.
- **Import path:** import it from `texas42.agents.pimc`, not the package root.
  `texas42.belief` imports `texas42.agents`, so a root-level import would be
  circular.

To compare it with heuristic or trained policies, use the `pimc` trump/play
strategy of `examples.evaluate_agent_vs_agent` ([composition](phase_composable_evaluation.md),
[commands](command_reference.md#belief-based-pimc-trump-and-play)), with
`--pimc-samples` and modest `--deals`. Expect a few seconds per hand.

**Measured strength.** In the study round robin (1,024 deals per pairing, 32
samples, heuristic bidding), plain PIMC beat every other agent:

| Opponent | PIMC's margin (marks/hand) |
| --- | --- |
| HHH | +0.297 |
| Base model | +0.337 |
| Shared model | +0.278 |
| Paired model | +0.313 |

Details are in [the study](multiphase_selfplay.md#one-seed-study-base-shared-and-paired-262144-hands).
Model-guided PIMC beats plain PIMC in turn (next section).

## Model-guided PIMC

Plain PIMC models every player as Heuristic V1. Its estimates are only as good
as that assumption. A trained checkpoint can replace the heuristic in three
independent places (`research/ppo/search.py`, wired through the `pimc`
composition strategy):

| Option | What the model does | Theory | Flag |
| --- | --- | --- | --- |
| Rollout policy | Plays every seat's simulated continuation | The estimate becomes Q<sub>π</sub>(s, a) for the model's policy π: "play a, then everyone plays like the model". Choosing the best such action is one step of policy improvement (Tesauro & Galperin 1996; Furtak & Buro 2013). | `--agent-X-pimc-model PATH` |
| Likelihood model | Weights hidden deals by P(observed move \| that player's view in the deal) | Policy-Based Inference (Rebstock et al. 2019) with a learned, stochastic policy | `--pimc-likelihood model` |
| Value leaf | Rollouts stop early; the value head estimates the rest. `value` stops after the next trick (the trick winner's viewpoint). `value-turn` stops at the searcher's own next decision (the searcher's viewpoint). Both rank moves worse than full rollouts ([ranking gate](search_training.md#6-evidence-the-move-ranking-gate)). | AlphaZero-style truncated search: a learned evaluation replaces the long tail of each rollout | `--pimc-leaf value` or `value-turn` |

### Data flow

```text
Observation ── build_belief(likelihood = model softmax, optional) ──> weighted worlds
  for each world, each legal move a (same rollout seed for every a):
    world_state ──> simulate(choices=(a,), policies = model at all four seats,
                             until = 'hand' | 'trick' | 'turn')
      hand finished ──> value = signed marks (or points)
      stopped early ──> value = ValueLeaf(state, observer)
          = value head of the acting seat's network on ITS observation,
            sign-flipped if that seat is on the other team
            ('turn' stops at the observer's own decision, so the actor is the observer)
  estimate(a) = belief-weighted mean; best move = argmax
```

- **Model as a player:** `search.model_policy_spec` turns a checkpoint into a full
  player. Trained phases are neural; the rest are Heuristic V1. For example, the
  study's shared model plays neural bid and play and heuristic trump, exactly as
  in training.
- **Rollout determinism:** model rollouts are greedy by default, which gives low
  variance. With `--stochastic-neural` they sample the model's softmax from
  seeded, seat-local streams.
- **Likelihood uses the full distribution:** belief weighting always uses the
  model's full softmax (`action_probabilities`), whichever rollout mode is set.

### Objective

- **Without a value leaf:** expected points, as in plain PIMC.
- **With `--pimc-leaf value` or `value-turn`:** signed marks
  (`evaluate_actions(objective='marks')`).
  - The guide must have trained its local value heads.
  - Centralized-critic and policy-only training leave them unfitted, so such
    checkpoints are rejected (`search.value_heads_trained`).
  - PPO value heads predict the acting team's final signed marks, so the leaf
    and the rollout outcomes share units.
  - Marks also suit trump decisions better than points, because they include
    whether the contract is made.

### Privacy

The value leaf builds another seat's `Observation` inside a hypothetical world.
That is a guess derived from public information, not the real hand, which keeps
the search honest. It is also exactly the input the network saw in training, so
the value is an information-set value rather than a perfect-information one.

### Inference speed and validity

- **Fast greedy path:** greedy neural decisions now take the masked argmax of the
  logits directly instead of building a `Categorical` distribution. This
  measured about 6× faster (≈160 µs vs ≈960 µs per call on one CPU thread), with
  identical actions on 4,800 real observations and in the regression test.
  Stochastic decisions are unchanged.
- **Probabilities in double precision:** `action_probabilities` computes the
  softmax in 64-bit precision.
- **Relaxed validity check:** `policy_likelihood`'s check now allows a 1e-6
  rounding tolerance, appropriate for learned policies.

### Measuring estimation accuracy

`python -m examples.search_accuracy --model PATH` plays hands with a fixed table
(the model at all seats, or `--table heuristic`) and samples play decisions. For
each one it scores:

- **Belief accuracy:** the probability on the true owner of each unseen domino.
  Hidden hands come from the documented privileged label channel and are never
  passed to search.
- **Value accuracy:** the estimate for the move actually played vs. the realized
  signed marks.
- **Cost:** per-search time.

The modes compared are uniform, heuristic-likelihood and model-likelihood
beliefs; and heuristic rollouts, model rollouts and model rollouts with a value
leaf. Measured results are in [Model-guided results](#model-guided-results).

`--rank-check` adds a **move-ranking** comparison. Calibration for the move
actually played does not show whether a search orders the *alternatives*
correctly, and that ordering is what chooses moves and what training on search
advantages would learn from. The check stores every legal move's estimate in
every mode and adds full rollouts scored in points, a replicate of them with
independent worlds (the noise scale) and the own-turn value leaf. For each
position it then
compares a candidate mode with a reference mode on the same worlds and seeds:
pairwise order concordance, top-1 agreement, Spearman rank correlation, and
**ranking regret**, the reference's value of its own best move minus its value
of the candidate's best move. `--partner-model` makes a paired-partner pair the
guide.

### Limitations

- **Imperfect guides:** a weak model is a weak guide. Model-guided search
  inherits its policy and value errors.
- **Cost:** model rollouts cost several times more than heuristic ones, even with
  the fast path. The value leaf recovers most of the difference.
- **Remaining PIMC flaws:** policy rollouts avoid the strongest form of strategy
  fusion (section 7). The value leaf is an information-set value, so it adds no
  perfect-information bias. But it is the **acting** seat's value, not the
  searcher's: whenever another seat acts at the leaf, it ignores the searcher's
  private hand ([leaf viewpoint](search_training.md#46-bias-of-the-search-target-term-by-term)).
  The [ranking check](#move-ranking-across-guide-models) measures the combined
  effect.
- **Run-level settings:** leaf and likelihood apply to every model-guided agent in
  a run. To compare two leaf settings, use separate runs against a common
  baseline.

### Model-guided results

**Setup:**
- Guide model: the shared bid+play study model (`runs/study-shared`, neural bid
  and play, heuristic trump, 262,144 training hands, one seed).
- 16 sampled worlds per search.
- Measured on one Windows machine.
- These are single-model, single-seed results. Treat them as evidence about this
  checkpoint, not general claims.

**Estimation accuracy** (`examples.search_accuracy`): 200 play decisions from 100
hands per table.

*Belief accuracy*: P(true owner) is the mean probability on each unseen domino's
true owner (higher is better). Log-loss uses 1% uniform smoothing.

| Belief | P(true owner), model table | Log-loss | ESS | P(true owner), heuristic table | Log-loss | ESS |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform (exact) | 0.403 | **0.965** | – | 0.377 | **1.016** | – |
| Heuristic likelihood | 0.459 | 1.474 | 4.6 | **0.497** | 1.272 | 7.0 |
| Model likelihood | **0.551** | 1.079 | 4.8 | 0.429 | 1.474 | 5.5 |

*Value estimates*: for the move actually played, vs. the realized signed marks.
Correlation is the comparable column; RMSE includes outcome noise.

| Rollouts | Corr. (model table) | RMSE | s/search | Corr. (heuristic table) | RMSE | s/search |
| --- | --- | --- | --- | --- | --- | --- |
| Heuristic, full hand | 0.584 | 0.817 | 0.148 | **0.625** | 0.799 | 0.160 |
| Model, full hand | 0.588 | 0.814 | 0.576 | 0.607 | 0.815 | 0.630 |
| Model + value leaf | **0.597** | 0.803 | **0.123** | 0.542 | 0.862 | 0.145 |

Readings:
- **A matching likelihood model is what makes beliefs more accurate.** Each
  weighted belief is most accurate when its likelihood model matches the table:
  model likelihood for the model table, heuristic likelihood for the heuristic
  table.
- **Weighted beliefs are overconfident at 16 samples.** With an effective sample
  size of about 5, they put more mass on the truth on average but sometimes miss
  it entirely. That is why uniform has the best log-loss. More samples or softer
  likelihoods (higher ε) would help.
- **Value accuracy also follows the table:** model rollouts and the value leaf
  are best on the model table, heuristic rollouts on the heuristic table.
- **The value leaf is fast and calibrated:** it is about 4.7× faster than full
  model rollouts, with similar calibration on the model table.

**Playing strength** (`evaluate_agent_vs_agent`): 128 duplicate deals (256 games)
per row, seed 20261022, heuristic bidding for every PIMC agent; A's signed
marks/hand with 95% paired bootstrap CI.

| A | B | A marks/hand | A win rate |
| --- | --- | --- | --- |
| Model-guided PIMC (full rollouts) | Plain PIMC | +0.094 [−0.016, +0.203] | 54.7% |
| Model-guided PIMC + value leaf | Plain PIMC | **−0.141** [−0.250, −0.031] | 43.4% |
| Model-guided PIMC + value leaf + model likelihood | Plain PIMC | **−0.133** [−0.234, −0.031] | 43.8% |
| Model-guided PIMC (full rollouts) | Shared model | **+0.340** [+0.234, +0.449] | 66.8% |
| Model-guided PIMC + value leaf | Shared model | +0.059 [−0.059, +0.176] | 53.1% |
| Plain PIMC | Shared model | **+0.160** [+0.047, +0.266] | 57.8% |

What the strength results show:
- **Full model rollouts are the strongest configuration.**
  - They beat the guide model itself by +0.34 marks/hand, about twice plain
    PIMC's margin over the same model (+0.16).
  - Head-to-head they lead plain PIMC by +0.094, but that interval includes 0 at
    128 deals. The 1,024-deal confirmation below settles it: smaller, but
    significant.

**Confirmation at 1,024 deals.** 2,048 games per row, fresh seed 20261030, the same
settings as above: heuristic bidding for every PIMC agent, 16 samples, full
rollouts, uniform belief. Outputs are in `runs/model-guided/arena-1024/`.

| A | B | A marks/hand | A win rate |
| --- | --- | --- | --- |
| Model-guided PIMC (shared guide) | Plain PIMC | **+0.045** [+0.009, +0.081] | 52.2% |
| Model-guided PIMC (shared guide) | Base model (heuristic bid and trump, neural play) | **+0.381** [+0.348, +0.415] | 69.0% |
| Model-guided PIMC (round-2 distilled guide) | Plain PIMC | **+0.044** [+0.006, +0.081] | 52.2% |
| Model-guided PIMC (round-2 distilled guide) | Model-guided PIMC (shared guide) | +0.027 [−0.004, +0.061] | 51.4% |

- **Model-guided search is the strongest agent measured.** It beats plain PIMC
  significantly, though by about half the 128-deal estimate. It beats the
  play-only base model by +0.38 marks/hand.
- **A better guide makes at most a slightly better search, so far.** The round-2
  distilled network beats its source by +0.015 on its own
  ([search in training](search_training.md#63-second-distillation-round-do-the-gains-compound)).
  As a guide:
  - it leads the shared-guided search head-to-head by +0.027, an interval that
    includes 0;
  - it ties it against plain PIMC (+0.044 vs +0.045 on the same deals).
- **The value leaf loses strength, despite accurate value estimates for the
  move actually played.** An objective ablation (64 deals, scratch script
  recorded in `runs/model-guided/objective-ablation.json`) separates the two
  changes the leaf mode makes:

  | Model-guided PIMC vs plain PIMC | Marks/hand |
  | --- | --- |
  | Full rollouts, points objective | +0.094 [−0.016, +0.203] (128 deals) |
  | Full rollouts, **marks** objective | +0.031 [−0.125, +0.188] (64 deals) |
  | Value leaf, marks objective | −0.141 [−0.250, −0.031] (128 deals) |

  - **Objective:** the point estimates suggest switching to marks costs a little.
    Marks mostly reflect whether the contract is made, so they separate play moves
    less sharply than points.
  - **Leaf:** truncating with the value head costs more. A value head can be
    calibrated on average yet rank *alternative* moves poorly. It was trained
    only as PPO's baseline on the moves the policy actually takes.
  - **Caveat:** the intervals overlap, so this split is indicative, not conclusive.
- **Practical guidance:**
  - For decision quality, use full model rollouts with the points objective.
  - Do not rely on the value leaf to rank moves (next section).
  - The value leaf's present strength is cheap value estimation;
    [search in training](search_training.md) discusses what this means for
    training-time search.

#### Move ranking across guide models

`search_accuracy --rank-check` compares how search modes order **all** legal
moves. It was run on the same 150 heuristic-table positions with three guides:
play-only base, shared bid+play, and the paired-partner pair (`--partner-model`).
The full tables, noise scale and bootstrap intervals are in
[the ranking gate](search_training.md#6-evidence-the-move-ranking-gate). In short:

- **The leaf ranks moves poorly, for every guide.** Value-leaf search picks the
  same best move as full model rollouts (points) in only 44–50% of positions.
  Plain heuristic rollouts manage 53–57%. The leaf's regret is also higher:
  0.92–1.19 vs 0.78–0.89 points, where a random legal move would cost about
  1.95.
- **Truncation, not the objective, is the main cause.** With both scored in
  marks, the leaf matches full model rollouts on the best move only 37–42% of
  the time.
- **The leaf's error is systematic, not sampling noise.** Two independent
  full-rollout searches agree on the best move in only 58–66% of positions at
  16 worlds. The leaf falls well below that, even though it shares the
  reference's worlds.
- **The guide's training doesn't change the picture.** The paired pair is
  slightly better and the base model worst, within noise.
- **The leaf's viewpoint is only a minor cause.** The leaf scores positions from
  the acting seat's viewpoint, not the searcher's
  ([leaf viewpoint](search_training.md#46-bias-of-the-search-target-term-by-term)).
  The own-turn leaf (`until='turn'`, `--pimc-leaf value-turn`) removes that bias
  and ranks only a little better: its best-move agreement with full rollouts is
  42–46%. It still fails the gate, so the value head's own error dominates
  ([own-turn results](search_training.md#61-the-own-turn-leaf-p2a)).

## Choosing settings

| Situation | Suggested settings |
| --- | --- |
| Quick move estimate | `samples=32–64`, HHH `rollout_policies`, uniform belief |
| Exact late-hand analysis | `enumerate_below` ≥ `count_worlds` (cost grows linearly) |
| Model-based inference | `likelihood_policies` = the policies actually used; ε 0.1–0.3; watch ESS |
| Plugging in an oracle | `until="trick"` plus `leaf_evaluator` |
| RQ2 baselines | Compare uniform vs policy beliefs (and later learned beliefs) via `marginals()` log-loss on logged deals (`TrainingCollector(include_initial_deal=True)`) |

## Limitations and next steps

- **Strength not yet measured.** The PIMC agent exists, but no multi-seed arena
  study against HHH or PPO has been run; smoke tests show mechanics only.
- **Single-hand objective.** Match marks and future hands are not modelled.
- **Strategy fusion** remains (see section 7); ISMCTS or EPIMC are possible
  extensions.
- **No learned beliefs yet.** A belief network would replace uniform sampling
  with model-guided sampling (Solinas et al. 2019); `Belief.marginals()` is the
  target format.
- **MCMC** sampling (Solinas et al. 2023) would only become useful for strongly
  weighted beliefs; hard constraints are already sampled exactly.
- **Copy helper.** `_copy_state` duplicates a private helper on the oracle branch.
  A shared `HandState.copy()` could replace both once that branch merges.

## Research background

- Ginsberg, *GIB: Imperfect Information in a Computationally Challenging Game*, JAIR 2001 — https://arxiv.org/abs/1106.0669
- Long, Sturtevant, Buro & Furtak, *Understanding the Success of Perfect Information Monte Carlo Sampling*, AAAI 2010 — https://ojs.aaai.org/index.php/AAAI/article/view/7562
- Buro, Long, Furtak & Sturtevant, *Improving State Evaluation, Inference, and Search in Trick-Based Card Games*, IJCAI 2009 — https://www.ijcai.org/Abstract/09/236
- Rebstock et al., *Policy Based Inference in Trick-Taking Card Games*, 2019 — https://arxiv.org/abs/1905.10911
- Solinas, Rebstock & Buro, *Improving Search with Supervised Learning in Trick-Based Card Games*, AAAI 2019 — https://arxiv.org/abs/1903.09604
- Solinas et al., *History Filtering in Imperfect Information Games*, 2023 — https://arxiv.org/abs/2311.14651
- Furtak, dissertation on recursive Monte Carlo search (Skat), 2013 — https://era.library.ualberta.ca/items/df9f1055-df09-43ed-910b-38da4e9d0792
- Arjonilla, Saffidine & Cazenave, *PIMC with Postponing Reasoning* (EPIMC), 2024 — https://arxiv.org/abs/2408.02380
- Cazenave, *The αμ Search Algorithm for Bridge* — https://arxiv.org/abs/1911.07960
- Cowling, Powley & Whitehouse, *Information Set Monte Carlo Tree Search*, IEEE TCIAIG 2012 — https://eprints.whiterose.ac.uk/75048
- Lerer et al., *Improving Policies via Search in Cooperative Partially Observable Games* (SPARTA), AAAI 2020 — https://ojs.aaai.org/index.php/AAAI/article/view/6208
- Silver & Veness, *Monte-Carlo Planning in Large POMDPs* (POMCP), NeurIPS 2010 — https://neurips.cc/virtual/2010/poster/2145
- Tesauro & Galperin, *On-line Policy Improvement using Monte-Carlo Search*, NIPS 1996 — https://papers.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html
- Loki opponent-model weight tables — https://webdocs.cs.ualberta.ca/~jonathan/PREVIOUS/Grad/pena/node26.html
- Reconnaissance Blind Chess information-set weighting — https://arxiv.org/html/2407.05864v1
- Vacant places — https://en.wikipedia.org/wiki/Vacant_Places
- Minesweeper exact probabilities — https://mrgris.com/projects/minesweepr/
