# Perfect-information endgame oracle

`texas42.oracle` solves ordinary-contract endgames and analyzes complete deals for
maximum makeable open-hand contracts. It is a separate analysis/search component,
with no PPO, belief sampling, policy adapter, or changes to observations and training. Engine rules,
legal action IDs, trick resolution and final scoring remain authoritative.

## Inputs and supported rules

Call `solve(full_state, *, output="all_action_values", limits=SearchLimits(...), workers=1)`
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

With one worker, `output="position_value"` stops when it proves the root outcome and one optimal
action. Other actions can remain unresolved even with `status="exact"`.
`output="all_action_values"` searches every root child with a full window while
sharing the cache in sequential execution. A budget can then leave `status="budget_exceeded"` alongside
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

`SearchLimits(time_seconds=1800.0, max_nodes=1_048_576, max_cache_entries=100_000)`
defaults to 1,800 seconds (30 minutes) and 1,048,576 nodes **per tree**. Passing
`limits=None` uses those defaults; explicitly setting either time/node field to
`None` disables that individual limit. Nonnegative finite time and integer
node/cache limits are accepted, including zero. Nodes count the root,
terminal/score-cutoff nodes, and cache hits. Limits are checked on node entry;
time is a soft deadline, not an interrupt inside input validation or a single
engine transition. `monotonic()` supplies the clock. No incomplete leaf receives
a heuristic estimate, and no unknown action is recommended as proven optimal.

## Final-point objective

`solve_points(full_state, *, output="all_action_values", limits=None,
reference_team=None, workers=1)` uses the same state validation, root-action analysis,
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

`analyze_deal(hands, *, ruleset_id=RULESET_ID, limits=None, workers=1) -> DealAnalysis`
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

### Scheduling and per-tree budgets

Each of the **36 bidder/trump combinations** is one tree and receives the full
time and node allowance. A fresh search context resets its counter, deadline and
interruption flag; an exhausted earlier tree does not prevent later trees from
running. Its legal first-play branches share that tree's allowance. The same
scope applies with one worker or many. Cache capacity remains bounded across
the analysis, with entries reusable between trees.

Sequential scheduling is seat order, canonical trump order, then global play
action ID. The tree clock starts after the one-time deal validation, immediately
before constructing that tree's play-start state. Nodes count search entries,
not the trump-setup transition. Cache retention is not required for correctness.
Returned nodes/cache hits/elapsed time are aggregate run statistics, so they can
exceed a single tree's limits. With a finite node cap, at most `36 * max_nodes`
search nodes are permitted across the whole deal. There is no whole-run timeout.

Each lead is searched with a window that preserves ties with the best proven
lower bound so far. Integer point utility permits `alpha=incumbent-1`: failing
low proves a lead strictly inferior, while a lead tying or exceeding the
incumbent is resolved. The same incumbent spans the bidder's earlier trumps.
This can leave suboptimal action/trump values bounded rather than exact; no
symmetry reduction or heuristic leaf evaluation is used.

The API and CLI both default to **1,048,576 nodes and 1,800 seconds per tree**.
These are upper limits, not a promise of completion. A complete-deal analysis
can therefore run longer than 1,800 seconds in total. A `solve`/`solve_points`
endgame position is one tree, including all its legal root-action branches;
parallelizing those branches does not multiply the position's budget.

## Parallel execution

All three APIs accept the optional `workers` argument, defaulting to `1`.
`workers=1` retains sequential traversal, cache sharing, incumbent
windows, result structures, and proof semantics. Time/node limits now apply per
tree with every worker count. Parallelization
does not repair or redesign the underlying minimax algorithm, rules or objectives.

`workers=0` selects automatically; positive values request up to that many CPU
processes. Resolution uses the repository's `resolve_workers` affinity, physical
core and cgroup limits, with an additional maximum of eight oracle workers and
the number of independent tasks. `OrderedPool` supplies reusable **spawned**
processes, ordered results, failure propagation and shutdown. There is no thread
pool, recursive spawning, or new process-pool abstraction. One pool is reused for
the tasks in a single API call; separate API calls create separate pools.

### Tasks, state and caches

- Complete-deal analysis has 36 tasks, one per candidate bidder and canonical
  trump. A process receives the validated deal once, then reconstructs an
  independent play-start state for each assigned combination. Each task runs the
  existing `_solve_state` and `_Search`, including ordinary global-ID lead order.
  Because earlier trumps need not finish first, their incumbent bounds are not
  dependencies; each combination starts with the valid zero-point incumbent.
- `solve` and `solve_points` split at legal root-action children. The coordinator
  counts the root once, and each child uses the existing alpha–beta search with
  a full window. Actor/reference-team semantics remain unchanged. Parallel
  `position_value` may analyze all root actions instead of stopping after the
  first proof. Existing aggregation decides what is exact or optimal.

Only the input state is serialized once per worker, followed by small task
identifiers and node allocations. No search tree or mutable state is shared.
Already-decided roots, forced single-action roots, and zero budgets avoid starting
a process pool. Sequential execution is usually preferable for small endgames.

Each process retains a private transposition table across its assigned tasks.
`max_cache_entries` remains an **aggregate entry limit**: each process gets
`floor(max_cache_entries / resolved_workers)` entries. Any remainder is unused;
zero disables caching. Thus all worker tables together stay within the caller's
entry cap. This is an entry-count limit, not a byte limit on Python process RSS.
Process/import/serialization overhead is additional. There is no synchronized
shared table, so cross-worker cache reuse and earlier-trump pruning are lost;
aggregate node counts can increase relative to sequential execution.

### Per-tree node and time limits

For deal analysis, every dispatched bidder/trump task receives the full node
limit and starts a fresh monotonic deadline when the worker begins that task.
Queueing, process startup, and time spent on previous combinations do not consume
the next tree's time allowance. There is no node division or shared deadline
across these 36 trees. The pool reuses its processes and their bounded caches.

Root-action tasks in `solve`/`solve_points` are **branches of one tree**. Their
coordinator still counts the root once and reserves disjoint node allocations in
canonical batches of at most the worker count. It divides that position's
remaining nodes over pending branches and reclaims unused reservations for later
batches. A branch interrupted at its allocation is not restarted. The sum of
root plus branch nodes stays within `max_nodes`. This policy can leave some
budget unused; it does not change proof or heuristic-evaluation semantics.

Endgame branches share the position's original deadline, including validation
and worker startup. For either mode, workers check the applicable tree deadline
on search-node entry. Limits are cooperative, not hard process kills; a
transition or shutdown can overrun a very short deadline. Reported wall time
always includes the entire call, including validation, spawn, queueing and
shutdown, even though deal tree deadlines start later. Worker errors identify
the task and propagate as exceptions, with the existing cancellation/shutdown
behavior. `max_cache_entries` remains aggregate, rather than a per-tree allowance.

### Reproducibility and platforms

Exact values and complete optimal-action/choice sets must agree between sequential
and parallel executions. Canonical result ordering is independent of task finish
order. With restrictive budgets, worker counts, private-cache assignment and time
spent starting processes can change bounds, node counts and completion rates;
identical unfinished reports across worker counts or repeated parallel runs are
not promised. `workers=1` retains deterministic traversal with the new per-tree limits.

Spawn works on Windows, Linux and WSL2. Script callers must use a
`if __name__ == "__main__":` guard. Run the example as a module from an importable
checkout; interactive stdin/notebook entry points may not be importable by spawn.
Nested process pools are rejected by the existing worker infrastructure. No
PyTorch, NumPy, OpenSpiel or new runtime dependency is required. The CLI reports
requested/resolved workers and eligible independent tasks; these are capacity
figures, and an already-proven root may dispatch no work.

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
per-tree budgets, input immutability, and bounded CLI execution. Synthetic
aggregation fixtures avoid expensive full-deal exhaustive searches.

The pre-parallelization baseline passed all 237 repository tests. Parallel tests
add spawn execution, both actors/reference teams, all trumps, exact ties and
ordering, position-wide branch budgets (including zero), input immutability, cache caps/reuse,
worker errors/shutdown, and the unchanged one-worker path. Exact complete-deal
aggregation uses synthetic worker fixtures; real full-deal integration is tested
with bounded searches, not an unbounded 36-tree exhaustive run. New per-tree
regressions verify fresh sequential/worker deadlines and a full node allowance
for every bidder/trump tree, including after an earlier tree exhausts its budget.

## Bounded smoke benchmarks

The historical deal measurements below predate per-tree limits and used the old
whole-analysis budgets. They remain historical measurements, not predictions for
current deal runs. Endgame position budgets still cover one entire position, so
those workload definitions are unchanged apart from the new defaults.

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

### Parallelization smoke benchmark

Measured with CPython 3.13.5 on the same i5-12600KF/WSL2 environment, with eight
available physical CPUs. Each row is one timing observation in a verification
pass, not an averaged performance estimate. Endgames use seed 42, HHR play,
no partial trick, the final-point objective and all root action values. Limits
are one million nodes/20 seconds and 100,000 aggregate cache entries per call.
The complete deal uses seed 42 with 200,000 nodes/10 seconds and the same cache
cap. Input generation is excluded; validation, spawn and shutdown are included.
Repeat the [point command recipes](command_reference.md#perfect-information-endgame-oracle)
with `--workers 1`, `2`, `4`, `8` and the corresponding remaining hand size.
The historical deal rows used 200,000 nodes for the whole analysis. Current deal
commands give that allowance to each tree, so they do not reproduce those rows.

The workload column is dominoes per player, or `deal` for all 36 combinations.
Exact completion is 100% or 0% of the single analysis at that setting. Exact
endgame values, every action value and complete optimal-action sets were checked
against the one-worker result. Bounded deal results are not claimed equivalent
in traversal or incomplete bounds.

| Workload | Requested → resolved | Wall s | Speedup | Nodes | Nodes/s | Cache hits | Exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 → 1 | 0.0009 | 1.00× | 30 | 33462 | 0 | 100% |
| 2 | 2 → 2 | 0.0846 | 0.01× | 30 | 354 | 0 | 100% |
| 2 | 4 → 2 | 0.0834 | 0.01× | 30 | 360 | 0 | 100% |
| 2 | 8 → 2 | 0.0795 | 0.01× | 30 | 377 | 0 | 100% |
| 3 | 1 → 1 | 0.0076 | 1.00× | 430 | 56573 | 44 | 100% |
| 3 | 2 → 2 | 0.0827 | 0.09× | 430 | 5197 | 44 | 100% |
| 3 | 4 → 3 | 0.0869 | 0.09× | 458 | 5272 | 37 | 100% |
| 3 | 8 → 3 | 0.1038 | 0.07× | 430 | 4143 | 44 | 100% |
| 4 | 1 → 1 | 0.0308 | 1.00× | 1578 | 51224 | 165 | 100% |
| 4 | 2 → 2 | 0.1041 | 0.30× | 1712 | 16440 | 156 | 100% |
| 4 | 4 → 4 | 0.1013 | 0.30× | 1716 | 16939 | 155 | 100% |
| 4 | 8 → 4 | 0.0998 | 0.31× | 1716 | 17187 | 155 | 100% |
| 6 | 1 → 1 | 3.2326 | 1.00× | 158490 | 49029 | 37506 | 100% |
| 6 | 2 → 2 | 2.4125 | 1.34× | 180331 | 74749 | 39640 | 100% |
| 6 | 4 → 4 | 1.8108 | 1.79× | 192507 | 106310 | 40171 | 100% |
| 6 | 8 → 6 | 1.6704 | 1.94× | 219804 | 131586 | 43769 | 100% |
| deal | 1 → 1 | 4.2837 | 1.00× | 200000 | 46689 | 40102 | 0% |
| deal | 2 → 2 | 2.2863 | 1.87× | 200000 | 87477 | 28506 | 0% |
| deal | 4 → 4 | 1.2114 | 3.54× | 200000 | 165105 | 28506 | 0% |
| deal | 8 → 8 | 0.9609 | 4.46× | 200000 | 208131 | 28506 | 0% |

An empty-task diagnostic that initializes the oracle workers and dispatches
zero-node jobs measured 0.0824/0.0881/0.1193 seconds for 2/4/8 worker capacities,
including process startup and shutdown. The public zero-budget APIs skip that
startup entirely. Overhead dominates 2–4-domino positions: use one worker for
those workloads. The six-domino search benefits from parallelism despite visiting
about 39% more nodes at six workers. The bounded deal's 4.46× speedup measures
throughput, not completion: every configuration returned `budget_exceeded`.
Canonical batch waits, changed cache locality and reduced incumbent sharing
limit scaling; more workers do not guarantee improvement.

Cumulative **parent-only** peak RSS rose from roughly 24 MiB to 116 MiB during
the benchmark. This excludes child processes and is not a combined peak-memory
measurement; aggregate process RSS was not sampled. Cache-entry caps were tested
independently. OS scheduling/import timing remains a measurement limitation.

The pre-change full suite passed 237 tests; after parallelization it passed 247.
An additional 772 one-worker baseline comparisons matched the prior implementation
including node counts, cache hits and proof outputs, excluding elapsed time.
No underlying oracle correctness repair was included in this change.

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
