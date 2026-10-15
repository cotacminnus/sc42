# Perfect-information endgame oracle

`texas42.oracle` solves ordinary-contract endgames and analyzes complete deals for
maximum makeable open-hand contracts. It is a separate analysis/search component,
with no PPO, belief sampling, policy adapter, or changes to observations and training. Engine rules,
legal action IDs, trick resolution and final scoring remain authoritative.

## Inputs and supported rules

Call `solve(full_state, *, output="all_action_values", limits=SearchLimits(...))`
with a complete legal `texas42.state.HandState` in the play phase. Bidding and
trump must already be fixed. The state must retain its completed trick history,
partial trick, scores, and four remaining hands in absolute seat order. The
oracle copies it and validates the play history by reconstructing the deal and
replaying engine transitions. Neither solving nor validation mutates the caller.

Only ordinary contracts under `cadets_base_v1` are supported: point bids 30–41
and mark bids 1–5, with numbered trump, doubles trump, or Follow Me/no trump.
Every mark bid targets 42 points; stakes do not change this binary objective.
Unresolved bidding/trump phases, special contracts, and unknown rulesets raise
clear errors. Inconsistent scores or histories are rejected rather than used to
justify an early result.

Partial tricks are supported, including positions where some players have no
dominoes left. The main workload is 1–3 dominoes per player. There is no solver
hand-size restriction; supply budgets for larger positions. This solves a hand's
contract, not a match objective involving future hands or accumulated match marks.

## Values and proof semantics

The reference team is always the bidding partnership:

- `+1`: the bidding team can force contract success under open-hand optimal play.
- `-1`: defenders can force contract failure under open-hand optimal play.

These are minimax outcomes, not win probabilities, heuristic scores, or signed
mark stakes. A node maximizes when the actual actor is on the bidding team and
minimizes otherwise. Trick winners determine the next actor; values are never
negated merely because a move was made.

`OracleResult` is an immutable result with these fields/properties:

| Member | Meaning |
| --- | --- |
| `status` | `exact` when the requested output is fulfilled, otherwise `budget_exceeded`. |
| `lower`, `upper` | Proven bounds on the root outcome, initially `[-1, +1]`. |
| `exact`, `value` | Root bounds coincide; `value` is that outcome, otherwise `None`. |
| `optimal_action` | One proven optimal global action ID, otherwise `None`. No tentative action is exposed. |
| `optimal_actions` | The complete set of optimal IDs, otherwise `None`. At a solved terminal state it is `()`. |
| `action_values` | One `ActionValue` per legal root action in global-ID order. Each has `action_id`, `lower`, `upper`, `exact`, `value`, and `searched`. |
| `all_action_values_complete` | Every legal root action has an exact value; independent of root exactness. |
| `principal_variation` | Currently always `None`; this implementation does not assemble or claim proof of a continuation. |
| `nodes`, `cache_hits`, `elapsed_seconds` | Entered search nodes, table hits, and monotonic elapsed time including copying/validation. |

`output="position_value"` stops when it proves the root outcome and one optimal
action. Other actions can remain unresolved even with `status="exact"`.
`output="all_action_values"` searches every root child with a full window while
sharing the cache. A budget can then leave `status="budget_exceeded"` alongside
an exact root value and a proven optimal action. For example, one proven `+1`
action at a MAX root establishes the root outcome while another action still
has bounds `[-1, +1]`. The complete optimal set remains `None`.

An unvisited action normally has `searched=False`, `value=None`, and the full
unknown outcome range. This is not a fabricated action evaluation. A special
case is an already decided contract: accumulated scores prove the outcome of
every legal continuation, so every action is exact even without visiting its
child. `searched` indicates a child was entered, not whether its value is proven.
The complete action set is exposed only once every action value is exact.

## Search, cache, and limits

`exhaustive_minimax(full_state) -> int` is the small reference implementation:
no pruning, caching, early score cutoffs or budgets; it evaluates engine terminal
results. Use it only for tiny positions. `solve` uses alpha–beta with deterministic
global-ID move ordering and no heuristic leaf evaluation.

Score cutoffs reuse the existing ordinary-contract certainty test: success once
the bidders have their target, failure once defenders have more than
`42 - target`. The 42-point total and monotone awarded scores make these proofs
valid even when a trick is partly played. The engine itself still plays all
seven tricks in ordinary games.

The per-call transposition table stores `EXACT`, `LOWER_BOUND`, and `UPPER_BOUND`
entries. Lookups tighten windows; stores classify against the original incoming
window. Bounds propagate through interrupted searches, but interrupted nodes are
not stored. FIFO eviction bounds the entry count; eviction changes work, not
answers. `max_cache_entries=0` disables caching.

Readable immutable keys include objective version, reference team, ruleset,
phase, contract, bidder, trump, actor, sorted remaining domino IDs by absolute
seat, ordered partial trick, completed trick count, awarded team scores, and
terminal result. Completed trick contents cannot affect future play once their
points are awarded; their count matters for engine termination. Auction history,
shaker and match marks cannot affect this fixed hand objective.

`SearchLimits(time_seconds=None, max_nodes=None, max_cache_entries=100_000)`
defaults to unlimited time/nodes and a bounded table. Nonnegative finite time and
integer node/cache limits are accepted, including zero. Nodes count the root,
terminal/score-cutoff nodes, and cache hits. Limits are checked on node entry;
time is a soft deadline, not an interrupt inside input validation or a single
engine transition. `monotonic()` supplies the clock. No incomplete leaf receives
a heuristic estimate, and no unknown action is recommended as proven optimal.

## Final-point objective

`solve_points(full_state, *, output="all_action_values", limits=None,
reference_team=None)` uses the same state validation, root-action analysis,
alpha–beta, engine transitions, budgets and bounded cache as `solve`. It returns
the same `OracleResult` shape, with integer final points in `0–42`. The reference
team defaults to the bidding partnership; explicitly select team `0` or `1` to
analyze the other partnership. At each node the actual actor's team determines
MAX or MIN, including across trick boundaries.

The binary contract cutoff is never applied to this objective: making a bid
does not fix the final point total. Instead, the awarded reference-team points
are a lower bound, and `42 - opposing_team_points` is an upper bound. These
bounds remain valid during partial tricks because unawarded trick points are
still available. Terminal scores make them coincide. Interrupted nodes retain
these bounds without heuristic evaluation. Objective, reference team and ruleset
are explicit in cache keys, so binary and point values cannot contaminate each
other even if tables are shared.

The existing `solve` API and its binary results are unchanged. Point results
can additionally prove a choice optimal while its precise value remains
unknown: a choice with bounds `[35, 40]` dominates every alternative whose upper
bound is at most 34. A complete optimal set can likewise be established without
solving strictly inferior alternatives. An alternative whose upper bound touches
35 still might tie and cannot be omitted from the proof obligation.

The terminal-only reference also accepts `objective="final_points"` and
`reference_team=0` or `1`. Its default remains contract success. It has no budget
and should only be used on tiny endgames.

## Complete-deal bidding benchmark

`analyze_deal(hands, *, ruleset_id=RULESET_ID, limits=None) -> DealAnalysis`
accepts four complete seven-domino hands of `Domino` values in absolute-seat
order. The native `EpisodeState.deal` validates the complete double-six set once,
including uniqueness, without mutating the input. Internally constructed states
are trusted and do not repeat endgame history reconstruction. Arena `Deal`
objects contain IDs; convert each ID using `PLAY_ACTIONS[id]` as the example does.

For each candidate bidder seat 0–3, the bidder leads, its partnership maximizes
final points, and defenders minimize them. The analysis examines all nine modes
from canonical `TRUMP_ACTIONS`: numbered `0–6`, `"doubles"`, and `None` (no trump).
Every mode starts from a fresh play state with zero scores, empty trick/history,
and seven dominoes per seat. The engine supplies the legal first-play IDs and
every subsequent transition.

A normal 30-point contract is an internal placeholder required by the engine.
Its target/stake affects neither point utility nor ordinary play legality.
Ordinary legality depends on hand, lead and trump; the point objective uses
awarded points, not the placeholder's terminal scoring team. Tests replay legal
play under different placeholder stakes and verify equal actions and point scores.

The bidder's value is the maximum over trumps of the maximum over first plays.
There is no auction, opponent bidding model, or optimization of mark stake:

| Guaranteed final points | Maximum makeable ordinary contract |
| --- | --- |
| Below 30 | None |
| 30–41 | Point bid equal to those points |
| 42 | `Bid(marks=1)` |

All supported mark stakes require 42 points. Guaranteeing 42 does **not** establish
five marks as an optimal stake.

Results expose proofs at three levels:

| Result | Members and semantics |
| --- | --- |
| Per first play (`ActionValue`) | `action_id`, `lower`, `upper`, `value`, `exact`, `status` (`exact`/`unresolved`) and `searched`. Unvisited actions initially have `[0, 42]`, no value, and `searched=False`. |
| Per trump (`TrumpAnalysis`) | `trump`, point bounds/value/exactness, `optimal_action`, `optimal_actions`, `action_values`, `all_action_values_complete`, and `status`. |
| Per bidder (`BidderAnalysis`) | `seat`, `team`, point bounds/value/exactness, contract fields below, `optimal_choice`, `optimal_choices`, `trumps`, `all_trump_values_complete`, `all_action_values_complete`, and `status`. Choices are immutable `TrumpLead(trump, action_id)` pairs. |
| Overall (`DealAnalysis`) | `bidders`, aggregate `nodes`, `cache_hits`, `elapsed_seconds`, `status`, and `all_alternative_values_complete`. |

For trump/bidder results, `exact` concerns only the point value. `status="exact"`
requires that value and the complete optimal-choice set; otherwise it is
`unresolved`. A non-`None` single choice is proven optimal, never tentative.
`optimal_actions`/`optimal_choices` is `None` until the full tie set is proven.
Unknown alternatives whose upper bounds are strictly below a proven optimum do
not prevent completion of that set. `all_*_values_complete` remains false if any
of those inferior alternatives has only bounds. At the overall level,
`status="exact"` requires an exact value and complete optimal pairs for all four
bidders; otherwise it is `budget_exceeded`. It does not imply every alternative
value is exact. No principal variation or contingent strategy is supplied.

Bidder contract fields deliberately distinguish classification from point value:

- `contract_exact` says whether the bounds imply one maximum contract class.
- `maximum_contract` is that `Bid`, or `None`. Pair it with `contract_exact` to
  distinguish a proven absence of a makeable contract from an unresolved maximum.
- `proven_contract` is the strongest bid established by the lower bound, if any.
- `possible_contract` is the highest bid still possible under the upper bound,
  which is not itself a makeability proof.

For example, bounds `[20, 29]` prove no ordinary contract is makeable while the
exact point total remains unknown: `contract_exact=True`, `maximum_contract=None`,
and `value=None`. Bounds `[30, 42]` prove 30 makeable but leave the maximum
unresolved; a one-mark contract is still possible.

### Scheduling and whole-analysis budgets

One `SearchLimits` and one bounded cache span **all 36** bidder/trump combinations.
Limits never reset between combinations or first plays. Time includes initial
validation and result assembly, with soft checks on search-node entry. Scheduling
is deterministic: seat order, canonical trump order, then global play-action ID.
Early combinations may consume the entire budget; later combinations still
appear explicitly unresolved. Nodes count search entries, not the 36 trump-setup
transitions. Cache retention is not required for correctness.

Each lead is searched with a window that preserves ties with the best proven
lower bound so far. Integer point utility permits `alpha=incumbent-1`: failing
low proves a lead strictly inferior, while a lead tying or exceeding the
incumbent is resolved. The same incumbent spans the bidder's earlier trumps.
This can leave suboptimal action/trump values bounded rather than exact; no
symmetry reduction or heuristic leaf evaluation is used.

The API defaults to unlimited time/nodes when no limits are supplied, matching
the endgame API. Full-hand analysis can be expensive: callers should pass limits.
The CLI's deal mode always defaults to an explicitly printed **100,000 nodes
and five seconds for the whole deal**, with configurable limits. This is not a
promise of completion. Endgame CLI budget defaults retain their existing behavior.

## Running and validation

See the [command reference](command_reference.md#perfect-information-endgame-oracle)
for copy-ready example/benchmark commands and all flags. The example uses arena
deal generation and existing ordinary policies, stops after legal engine play,
and explicitly prints all four hands. With no `--seed`, it chooses and prints a
random seed; supplying that seed reproduces the positions. It generates positions
rather than adding a new state file format: existing schema-5 serialization restores actor
observations, not arbitrary mutable full-state snapshots.

Focused verification from the repository root:

```bash
python -m unittest discover -s tests -p 'test_oracle*.py' -v
```

Tests compare exhaustive terminal-only minimax with every root action under
disabled, single-entry and normal caches across seeded reachable endgames. They
also cover ordinary mark contracts, all trump forms, both teams to act, forced
moves, tied optima, outcome-changing moves, same-team turns at trick boundaries,
partial tricks, terminal results, caller immutability, key separation, table
bound windows/eviction, invalid inputs, CLI output, and deterministic node-budget
interruption. A proven root with unresolved actions is explicitly exercised for
both bidding and defending actors.

Point tests compare both reference teams and actual acting teams with exhaustive
terminal minimax, including partial tricks, no trump and doubles. They cover
point bounds at deterministic node interruptions, objective/cache separation,
contract cutoffs not applying to points, strict-dominance/tie proofs, contract
classification, one-time deal validation, every candidate leader and trump,
whole-analysis budgets, input immutability, and bounded CLI execution. Synthetic
aggregation fixtures avoid expensive full-deal exhaustive searches.

## Bounded smoke benchmarks

### Contract-success endgames

Measured on 2026-10-14 with CPython 3.13.5, one process on an Intel Core
i5-12600KF under WSL2/Linux 6.6.87.2. Positions use the example's default HHR
policies, no partial trick, and consecutive seeds starting at zero. Every solve
requests all root action values, with 100,000 nodes, two seconds and 100,000
cache entries per position. Two- and three-domino samples contain 100 positions
each; the four-domino sample contains 20. The command reference's benchmark
recipe reproduces this workload by changing `--remaining` and `--samples`.

| Dominoes/player | Exact completion | Initially undecided | Nodes mean / median / max | Solve ms mean / median / max |
| --- | --- | --- | --- | --- |
| 2 | 100/100 (100%) | 47/100 | 14.06 / 1 / 51 | 0.534 / 0.394 / 1.363 |
| 3 | 100/100 (100%) | 65/100 | 106.57 / 96 / 493 | 1.942 / 1.755 / 8.011 |
| 4 | 20/20 (100%) | 14/20 | 699.90 / 516 / 3698 | 11.912 / 8.523 / 63.820 |

Cache hits (mean / median / max) were respectively `0 / 0 / 0`,
`6.56 / 0.5 / 62`, and `117.25 / 32.5 / 1179`. Every sample completed all action
values as well as the root. Timings include input copying/validation and exclude
position generation, printing, and interpreter startup. These small measurements
include score-decided positions and operating-system timing variability; they
are not a worst-case guarantee or evidence about hidden-information policy strength.

### Final-point endgames and complete-deal smoke run

On the same CPython 3.13.5/i5-12600KF/WSL2 environment, the point objective used
HHR-generated endgames, consecutive seeds starting at zero, no partial trick,
100,000 nodes/two seconds per position, and 100,000 cache entries. Every position
requested every root action value. Timings include validation and exclude deal
generation, startup and printing.

| Dominoes/player | Exact completion | Nodes mean / median / max | Solve ms mean / median / max |
| --- | --- | --- | --- |
| 2 | 100/100 (100%) | 41.21 / 41 / 73 | 1.020 / 1.001 / 1.780 |
| 3 | 100/100 (100%) | 360.15 / 325.5 / 1296 | 6.470 / 5.881 / 24.013 |
| 4 | 10/10 (100%) | 2734.30 / 1804.5 / 6796 | 50.732 / 32.318 / 129.897 |

Cache hits (mean / median / max) were `0 / 0 / 0`, `36.98 / 28 / 117`, and
`468.60 / 207.5 / 1899`, respectively. All action values completed in each sample.
These point workloads continue past already-decided contracts.

The seed-42 complete-deal CLI smoke run, with a **whole-analysis** limit of
10,000 nodes/one second and 100,000 cache entries, used exactly 10,000 nodes,
recorded 1573 cache hits, and returned `budget_exceeded` in 0.188214 seconds.
It proved 0/4 bidder values and 0/36 trump values; no optimal choice or maximum
contract was claimed. Later choices remained explicitly unresolved. This run
also succeeded with Python site packages disabled (`-S`), verifying that deal
analysis needs no optional ML dependencies. It demonstrates bounded interruption,
not full-deal tractability. No unbounded full-hand search was run.

Regression verification passed 237 repository tests, including 19 existing oracle
tests and 16 new point/deal tests. In addition, 1440 seeded comparisons against
the previous binary implementation matched bounds, action results, optimal
actions, node counts and cache hits (excluding elapsed time).

## Interpretation and limitations

An open-hand optimal action need not be optimal when hands are hidden. Its proof
assumes both partnerships know every remaining domino and can choose continuations
using that knowledge. Actual players must act on their own observations; they
cannot condition decisions on the unseen deal. This oracle supplies analysis or
explicitly privileged labels, not a replacement for those information constraints.

There is no hidden-information strategy, determinization, belief model, learned
ordering, persistent cache, or principal-variation output. Runtime can grow
quickly with larger hands; small smoke benchmarks do not establish full-hand
tractability or playing strength.
