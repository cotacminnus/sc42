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
defaults to 1,800 seconds (30 minutes) and 1,048,576 nodes **per endgame position**.
Complete deals instead use nodes per opening lead and one shared time budget,
as described below. Passing
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

`analyze_deal(hands, *, ruleset_id=RULESET_ID, limits=None, workers=1, progress=None) -> DealAnalysis`
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

### Scheduling, node budgets and the global deadline

Complete-deal analysis creates **252 logical tasks**: four bidder seats × nine
canonical trumps × seven legal opening actions. A task is identified by
`(bidder_seat, trump, opening_action_id)`. It reconstructs an independent
play-start state, applies the opening through the engine, and calls the existing
`_Search.search` on that continuation with a full search window. No recursive
search algorithm, objective, rule, legal action order or proof definition changes.
The former 36-task scheduler's incumbent sharing between opening leads is no
longer available at this independent task boundary.

`SearchLimits()` defaults to **1,048,576 nodes per opening-lead continuation** and
**1,800 seconds for the entire deal analysis**. These scopes apply with one or
many workers. Each continuation has its own counter and exhaustion flag; its
first entered node counts as one. Trump setup and applying the chosen lead do
not count as search nodes. Node counters include terminal and cache-hit entries,
just as in endgame search. An earlier exhausted task cannot consume a later
one's node allowance. With finite `max_nodes`, aggregate nodes cannot exceed
`252 * max_nodes` (264,241,152 at the default); actual totals are reported.

One absolute `monotonic()` deadline starts at API entry, before validation and
process startup, and is passed unchanged to every worker. It covers validation,
startup, queueing, search, result collection, aggregation and shutdown. The
coordinator stops submitting at that deadline and cancels queued futures where
possible. Workers check the same deadline before constructing a continuation
and on search-node entry. Running tasks return their valid partial bounds;
never-started/cancelled tasks retain `[0, 42]`, `value=None`, `searched=False`.
Validation, engine transitions, callbacks, aggregation and process shutdown are
cooperative operations rather than hard interrupts, so elapsed wall time can
slightly exceed the configured limit. A deadline never resets per task.
`None` disables the respective API limit; CLI flags accept nonnegative finite
values, with zero providing an immediately bounded run.

The hierarchy is reconstructed from canonical bidder/trump/action-ID ordering,
regardless of completion order. Each trump maximizes its seven `ActionValue`
bounds using the existing `_choice_proof`/`_combine` rules, and
`_bidder_analysis` preserves the existing maximum-contract and tie semantics.
An exhausted task can still have exact bounds; exactness follows the bounds,
not merely its stop reason. No heuristic value or tentative optimum is inserted.

### Live progress

The API remains silent unless `progress=callback` is supplied. The callback
receives one immutable `DealProgress` per settled logical task, in the coordinator
process. It includes `completed`, `total` (252), `bidder`, canonical `trump`,
`action_id`, `status`, actual `nodes`/`cache_hits`, task `elapsed_seconds`
(excluding queueing), and `global_elapsed_seconds`. It is synchronous; expensive
callbacks consume the global time budget. Task statuses are:

- `exact`: the lead's point bounds coincide, including after interruption.
- `node_limit`: unresolved and its node allocation is exhausted (also zero nodes).
- `time_limit`: started, but unresolved when the shared deadline stops search.
- `not_started`: not submitted, or the worker observes expiry before tree setup.
- `cancelled`: submitted work cancelled before execution.

If both limits bind, an unresolved task at its node cap reports `node_limit`.
Every successful analysis invocation settles all 252 tasks exactly once, including
explicitly skipped tasks after expiry. Worker exceptions instead propagate; no
normal completion is fabricated. The CLI installs a callback that flushes a
`[completed/252]` line immediately with bidder, readable trump/domino, status,
nodes, task runtime and global elapsed time. Workers never print. The existing
bidder tables and proof summary follow those progress lines. A final `[252/252]`
means all tasks are accounted for; inspect statuses to learn which were searched.

## Parallel execution

All three APIs accept `workers=1` (sequential), `workers=0` (automatic), or a
positive requested CPU-process count. Resolution uses `resolve_workers` affinity,
physical-core and cgroup limits, capped at eight oracle processes and the number
of tasks. One persistent `OrderedPool` supplies **spawned** workers for each API
call; separate calls create separate pools. There are no nested pools or thread
based search workers. The existing `map` interface remains ordered;
`iter_completed` adds bounded completion-driven collection for deal tasks.

At most `workers` deal futures are in flight. When one settles, the coordinator
reports it and immediately refills the freed slot without waiting for the first
submitted task. Simultaneously available results are processed by submission
index. New tasks are submitted in canonical order until expiry, but worker
assignment and completion order can vary. Only the validated deal/configuration
is sent once per process; subsequent messages carry compact task IDs and node
allocations. No mutable hand or search tree is shared. Zero budgets avoid
process startup. One worker is generally best for tiny endgames.

Each worker retains a local transposition table between assigned tasks.
`max_cache_entries` remains an **aggregate entry cap**, each process receiving
`floor(max_cache_entries / resolved_workers)` entries. Remainders are unused;
zero disables caching. The cap does not bound interpreter/serialization memory
or bytes per entry. Objective/reference-team namespaces remain explicit in keys.
Cache eviction cannot change correctness. Process-local tables lose cross-worker
reuse, and independent full-window leads lose incumbent sharing, so aggregate
work may increase. Additional futures/IPC also cost time.

### Endgame compatibility

`solve()` and `solve_points()` retain their existing per-position budgets,
sequential traversal and optional parallel root-action implementation. Their root
is counted once. Parallel branches reserve disjoint node allocations in canonical
waves, reclaiming unused allocations for later waves; sum of root and child
nodes stays within the position's `max_nodes`. All branches share that position's
original deadline, including input validation and worker startup. An interrupted
allocation is not restarted. Existing result dataclasses, exact/proof semantics,
and the exhaustive reference remain unchanged. No underlying oracle correctness
repairs are included in this scheduling change. Worker exceptions identify the
task and propagate through pool cancellation and shutdown.

### Reproducibility and platforms

Exact values and complete optimal-action/choice sets must agree between sequential
and parallel executions. Canonical result ordering is independent of task finish
order. With restrictive budgets, worker counts, private-cache assignment and time
spent starting processes can change bounds, node counts and completion rates;
identical unfinished reports across worker counts or repeated parallel runs are
not promised. `workers=1` retains deterministic task/traversal order with per-lead node limits.
Time-limited runs can still vary with runtime and callback costs.

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
python -m unittest discover -s tests -v
git diff --check
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
per-lead budgets, input immutability, and bounded CLI execution. Synthetic
aggregation fixtures avoid expensive full-deal exhaustive searches.

The baseline for the 252-task change passed all 47 focused oracle tests. New
regressions cover 252 distinct legal keys and default allocations, a single
shared deadline including validation, cancelled/unstarted tasks, independent
node counters, completion-driven refill past a slow first task, reversed-result
aggregation, progress exactly once per task, and the unchanged endgame path.
Synthetic workers test full-deal aggregation and ties without exhaustive full-hand
searches. Real spawned searches exercise bounded full deals; tractable real
endgames are compared with terminal-only exhaustive minimax. No pre-existing
search correctness failure was observed or repaired in this task.

## Bounded smoke benchmarks

The historical deal measurements below predate the 252-task scheduler and used
old whole-analysis node budgets. They remain historical measurements, not predictions for
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
The historical deal rows used 200,000 nodes for the whole analysis. Current deal commands give that node allowance to each of 252 opening-lead
trees and share one time limit, so they do not reproduce those rows.

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

### 252-opening-lead scheduler benchmark

Measured on 2026-10-15 with CPython 3.13.5, Intel Core i5-12600KF,
eight available physical CPUs, WSL2/Linux 6.6.87.2. The previous 36-task code
is commit `3e55855`; the new code uses 252 independent continuations. Both
receive the identical arena deal (seed 42, shaker 2), final-point objective,
100,000 aggregate cache entries, and an equal **252,000-node maximum aggregate
allowance**. Specifically, the old scheduler gets 7,000 nodes per bidder/trump;
the new scheduler gets 1,000 per opening lead. Equal numbers on the CLI would
instead give the new scheduler seven times the potential nodes, so that is not
used for this comparison. Old counters also include the 36 trump roots; new
counters begin after each opening lead. Work distribution and pruning differ.

Both use a nonbinding 20-second allowance: per combination in the old code,
global in the new code. No measured call reached that time limit. Timing covers
validation, process startup, scheduling, result collection, aggregation and
shutdown; input generation and console printing are excluded. The new progress
callback appends events in memory. Two passes were run; the table reports the
second pass's actual counts and single timings, not a statistical speedup claim.
Speedup is relative to that scheduler's one-worker run.

| Tasks | Workers | Wall s | Speedup | Nodes | Nodes/s | Cache hits | Sampled sum RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 36 | 1 | 6.2736 | 1.00× | 252000 | 40169 | 36921 | 123 |
| 252 | 1 | 6.0332 | 1.00× | 251815 | 41739 | 30797 | 134 |
| 36 | 2 | 2.9866 | 2.10× | 252000 | 84376 | 36636 | 270 |
| 252 | 2 | 2.9267 | 2.06× | 251815 | 86042 | 27102 | 268 |
| 36 | 4 | 1.5059 | 4.17× | 252000 | 167345 | 36636 | 320 |
| 252 | 4 | 1.5454 | 3.90× | 251823 | 162951 | 25884 | 317 |
| 36 | 8 | 1.0778 | 5.82× | 252000 | 233814 | 36636 | 417 |
| 252 | 8 | 1.0125 | 5.96× | 251839 | 248730 | 25439 | 414 |

Every old row established 0/252 exact action values, 0/36 exact trump values,
and 0/4 exact bidder point values. Every new row established 1/252, 1/36 and 1/4,
respectively. All rows remain `budget_exceeded`: the one exact bidder value does
not prove its complete optimal-choice set. These are partial-analysis counts,
not a claim of full-deal exhaustive correctness or fast full-hand completion.
The new scheduler reaches 5.96× scaling at eight workers and is about 6% faster
than the old eight-worker run in this pass. At four workers it is about 3%
slower; in the first pass, two and four workers were slightly slower. Cache
assignment, reduced reuse and OS timing affect throughput and bounded results.

For the new scheduler, sum of task wall times divided by worker count × total
wall time was 99.6% / 88.1% / 83.0% / 72.7% at 1/2/4/8 workers. This is an
approximate occupied-worker fraction including OS waiting, **not CPU utilization**.
RSS is a 50-ms sampled sum across the coordinator, children and resource tracker;
it can miss peaks and double-count shared pages. Retained Python allocator memory
from earlier calls affects these observations. It is not a physical-memory cap.

A startup/coordination-dominated run used 7 nodes per old task versus 1 per new
task (both 252 total nodes), eight workers, and the same deal/cache/time settings.
It took **0.1688 seconds for 36 tasks versus 0.2157 for 252**; all action/trump/bidder
values remained unresolved and both had zero cache hits. That approximately
28% slowdown demonstrates the cost of smaller tasks and additional IPC. No
assumption that more tasks always improve speed is made.

The unchanged endgame path was also measured with HHR-generated seeds 0–4,
no partial trick, all point-action values, 100,000 nodes/five seconds per
position, and 100,000 cache entries. Each row totals five API calls; process
startup is included for each parallel call. Every sample completed exactly.

| Dominoes/player | Workers requested | Total wall s | Nodes | Cache hits | Exact |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | 0.0053 | 199 | 0 | 5/5 |
| 2 | 4 | 0.4390 | 199 | 0 | 5/5 |
| 3 | 1 | 0.0400 | 2149 | 150 | 5/5 |
| 3 | 4 | 0.4733 | 2241 | 127 | 5/5 |
| 4 | 1 | 0.2185 | 11854 | 1421 | 5/5 |
| 4 | 4 | 0.5739 | 13117 | 1294 | 5/5 |

Small endgames remain much faster sequentially. Resolved workers are capped by
legal root actions; requesting four does not force four when fewer exist.
To repeat the new deal workload, run the existing CLI with `--mode deal --seed 42
--max-nodes 1000 --time-seconds 20 --max-cache-entries 100000` and workers 1/2/4/8.
For the small-endgame sample use `--objective final_points --seed 0 --samples 5
--remaining 2 --max-nodes 100000 --time-seconds 5` (then remaining 3 and 4).

A separate eight-worker CLI deadline smoke run used seed 42, 100,000 nodes per
lead and **0.5 seconds total**. It returned in 0.6090 seconds including shutdown,
visited 104,404 actual nodes with 17,836 hits, and settled all 252 tasks with
explicit exact/time-limit/not-started progress. One action/trump/bidder point
value was exact; complete optimal sets remained unresolved. This was an
interruption demonstration, not a strict timing assertion; workers returned
partial results promptly and the final `[252/252]` accounted for skipped work.

Verification passed 52 focused oracle tests and 254 full repository tests,
including PPO and engine regressions. An additional 320 calls to `solve` and
`solve_points` matched commit `3e55855` on seeded 2–3-domino positions, both
output modes and partial tricks, including node/cache counts and proof fields
(excluding elapsed time). `git diff --check` passed. No unbounded full-hand
search or training was run for these measurements.

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
