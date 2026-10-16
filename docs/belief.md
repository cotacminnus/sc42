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
| `until` | `"trick"`: next completed trick. `"phase"`: bid/trump/play phase change (for play, the end of the hand). `"hand"`: end of the hand. The hand ending always stops. |
| `track_probability` | Compute `rollout_probability` (policy steps then also call `action_probabilities`). |

| `Rollout` field | Meaning |
| --- | --- |
| `action_ids` | Every applied global action ID, in order (IDs stay meaningful across worlds; indices do not) |
| `explicit_steps` | How many came from `choices` |
| `rollout_probability` | Product of the modelled probabilities of the non-explicit steps (1/\|legal\| for random steps). Explicit choices are interventions ("what if I play X"), so they add no factor. `None` if not tracked or a policy cannot report probabilities. |
| `start_points`, `end_points`, `points_delta` | Team points (team 0, team 1), and the change |
| `stop_reason` | `"trick"`, `"phase"` or `"hand"` |
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
