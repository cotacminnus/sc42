# Perfect-information endgame oracle

`texas42.oracle` solves ordinary-contract endgames and analyzes complete deals for
maximum makeable open-hand contracts. It is a separate analysis/search component,
with no PPO, belief sampling, policy adapter, or changes to observations; training
uses it only through offline labels for an optional critic auxiliary head. Engine rules,
legal action IDs, trick resolution and final scoring remain authoritative.

## Inputs and supported rules

Call `solve(full_state, *, output="all_action_values", limits=SearchLimits(...), workers=1,
move_ordering="none")`
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
| `statistics` | Immutable `SearchStatistics` counters, described below; instrumentation does not affect proof equality. |

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
results. Use it only for tiny positions. `solve` uses alpha–beta with deterministic configurable move ordering
and no heuristic leaf evaluation. The default remains global action-ID order.

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

## Move ordering and instrumentation

All public search APIs accept `move_ordering`:

| Mode | Traversal |
| --- | --- |
| `none` (default) | Original legal global action-ID order. |
| `heuristic` | Deterministic Texas 42 priorities, breaking ties by global ID. |
| `tt` | A legal TT preferred action first, then canonical order. |
| `combined` | A legal TT preferred action first, then heuristic order. |

The heuristic uses `rules.is_trump`, `led_suit`, `suit_rank`, and `trick_winner`.
On lead it prefers strength/control, with trump and counter cost as tie-breakers.
Within a trick it prefers putting the actor's partnership ahead, feeding counters
to a winning partnership, and conserving strong trumps when possible. A partial
trick winner is provisional, never a guarantee about the rest of the trick.
Ranking is from the **actual actor's partnership** perspective at both MAX and
MIN nodes. Search still maximizes/minimizes the fixed reference utility. The
heuristic neither removes moves nor evaluates leaves. It uses only cheap local
features, without a second search or a hand-control simulation at each node.

TT entries append an optional `preferred_action`. It is the best bound-producing
child seen in that traversal, which may be a cutoff move rather than a proven
optimum. The action is checked against engine legal IDs before ordering. It is
never returned directly as a proven optimal action or principal variation.
EXACT/LOWER_BOUND/UPPER_BOUND classification still uses the original window;
interrupted or cancelled ancestors are not stored. Objective/reference-team/rule
key namespaces are unchanged. Ordering/output mode need not enter the key because
the cached utility is still the same point or binary minimax value/bound.
Externally reported root rows and optimal sets retain canonical ordering.

`none` remains the default: combined ordering substantially helps some larger
endgames, but the measured five-domino sample and one full-deal ablation were
slower. Choose it explicitly when appropriate rather than assuming it always wins.

`OracleResult.statistics` and `DealAnalysis.statistics` contain:

- `alpha_beta_cutoffs`: window cutoffs leaving legal children unsearched.
- `tt_cutoffs`: TT hits resolving a node or closing its window.
- `score_bound_cutoffs`: terminal/awarded-score proofs and exact bound closures
  that avoid exploring remaining children (including binary contract certainty).
- `ordering_calls`: nodes with multiple moves that invoke ordering.

Existing `nodes` and `cache_hits` remain authoritative actual-work totals.
Counters include repeated threshold queries and are summed from worker payloads,
without shared mutable counters. Ordering cost is compared through wall-time
ablations; there is no per-node timing instrumentation. Deal
`task_status_counts` is a canonical tuple of `(status, count)` pairs for all 252
settlements. Counts for exact, node-limited, deadline-limited and proof-cancelled
work do not imply full bidder/choice proof completion.

## Final-point objective

`solve_points(full_state, *, output="all_action_values", limits=None,
reference_team=None, workers=1, move_ordering="none")` uses the same state validation, root-action analysis,
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

## Arbitrary play positions

`evaluate_position(position, *, objective, reference_team, threshold, actions,
limits, total_time_seconds, move_ordering, cache)` reuses the same alpha-beta search
for any valid play position, without a replayable trick history. A `PlayPosition`
gives the remaining hands (domino IDs by absolute seat), the partial trick, its
leader, optional next player (verified), trump, bidder, optional bid and captured
points. Validation checks disjoint hands, turn order, hand sizes against one shared
trick count, captured points equal to completed tricks plus the counters of every
domino already gone, and that partial-trick followers either followed suit or are
still void in it. `position_from_state(state)` snapshots a live `HandState`. Only
ordinary contracts are accepted.

| Objective | Reference team | Result per action |
| --- | --- | --- |
| `final_points` | Default: the acting seat's team; 0 or 1 allowed | Certified `[lower, upper]` final team points |
| `contract` | Always the bidders (another team is rejected) | Bounds plus `proven_makeable` (lower ≥ threshold), `proven_unmakeable` (upper < threshold) or `unresolved` |

The contract threshold defaults to the bid's target points (30–41, or 42 for mark
bids) and is searched with a null window, so its bounds are proofs about the
threshold rather than exact values. Bounds always describe **final totals including
captured points**. `limits.max_nodes` and `limits.time_seconds` apply to **each
action's search separately**; `total_time_seconds` is one optional global deadline
for the call. Every row reports nodes and a termination reason (`exact`, `decided`,
`threshold_proven`, `node_limit`, `time_limit`, or `not_started` after the global
deadline). Exhausted searches keep valid bounds and are never relabelled exact;
`proven_optimal_actions` is given only when every legal action was evaluated and
the bounds prove the choice. A `cache` dictionary can be shared across calls because
keys include the ruleset, objective and reference team. This API feeds the
[offline critic labels](command_reference.md#oracle-training-labels); training never
calls it synchronously.

## Complete-deal bidding benchmark

`analyze_deal(hands, *, ruleset_id=RULESET_ID, limits=None, workers=1, progress=None,
move_ordering="none", output="all_action_values", proof_cancellation=True) -> DealAnalysis`
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
| Overall (`DealAnalysis`) | `bidders`, aggregate `nodes`, `cache_hits`, `elapsed_seconds`, legacy `status`, `all_alternative_values_complete`, requested `output`, `requested_output_complete`, `statistics`, and `task_status_counts`. |

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

### Maximum-contract proof mode

The default `output="all_action_values"` retains full-window continuation searches
for every opening lead. It never cancels a lead just because the bidder's maximum
contract is already known. Budgets can still leave individual values unresolved.

`output="maximum_contract"` instead requests only each bidder's highest makeable
ordinary contract. For each choice, the same point-valued alpha–beta engine asks
integer threshold questions: first whether 42 points are forceable using window
`[41, 42]`, then bisecting the remaining bid range with `[threshold-1, threshold]`.
Fail-high/low results are retained as bounds. Queries share one search context,
node counter, deadline and TT; the node budget does **not** reset per threshold.
Search can stop below 30 without determining the precise point score. No binary
contract-success cutoff or placeholder bid target is used for these point queries.

The coordinator computes `L_b = max(choice.lower)` and `U_b = max(choice.upper)`.
A maximum contract is proven exactly when `_makeable_bid(L_b) ==
_makeable_bid(U_b)`. One lower bound of 42 suffices because 42 is the domain
maximum. No-makeable-contract requires **every** alternative's upper bound below
30. For bids 30–41 the mapping is one-to-one, so a proven class also implies
an exact bidder point value; every alternative must be bounded above by that
optimum. Apparent strength or heuristic priority never establishes a proof.

`DealAnalysis.output` records the request; `requested_output_complete` means all
four contract classifications are proven in maximum-contract mode, or all
individual values are exact in all-action mode. The existing `status` remains
unchanged: it requires exact bidder points and complete optimal pairs. It can
therefore remain `budget_exceeded` even when `requested_output_complete=True`.
Each bidder's existing `contract_exact`, `maximum_contract`, point exactness,
single optimal choice, full optimal set and alternative-value completeness keep
their distinct meanings. In particular, proving 42 with one lead does not identify
all equally optimal trump/lead pairs. No unfinished alternative becomes exact.

In proof mode, deterministic task priorities favor concentrated trumps and strong
leads, interleaved across bidders so all seats receive promising initial work.
This affects scheduling only, independent of recursive `move_ordering`. The
main process owns aggregate proofs. Once a bidder is proven, the coordinator
stops submitting its unnecessary alternatives and cancels queued futures where
possible. Running workers poll a four-byte, coordinator-written shared-memory
flag array before starting and every 128 nodes. Flags only transition from zero
to one; stale reads may waste work but cannot establish a proof. There is no
shared transposition table or per-node synchronization/RPC. Partial bounds and
late completed results are preserved. Shared memory is closed/unlinked after
pool shutdown, including on exceptions.

`proof_cancellation=False` disables bidder-level skipping/cancellation for an
ablation, while retaining each task's threshold/classification search and proof
mode priorities. It has no effect on exhaustive mode, which never sets proof
stop flags. No task needed by `all_action_values` is cancelled after a bid proof.

### Scheduling, node budgets and the global deadline

Complete-deal analysis creates **252 logical tasks**: four bidder seats × nine
canonical trumps × seven legal opening actions. A task is identified by
`(bidder_seat, trump, opening_action_id)`. It reconstructs an independent
play-start state, applies the opening through the engine, and calls the existing
`_Search.search` on that continuation: full windows in exhaustive mode,
threshold windows in maximum-contract mode. Ordering and window selection change
search effort; utility, rules and proof definitions remain authoritative.
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
callbacks consume the global time budget. `bidder_contract_proven` is true once
per newly proven bidder; `maximum_contract` supplies the bid (or `None` for a
proven absence). The CLI prints that proof separately from task completion.
Task statuses are:

- `exact`: the lead's point bounds coincide, including after interruption.
- `contract_proven`: this searched lead's bid class is proven but its points are
  unresolved (below 30); it is not an exact action value.
- `node_limit`: unresolved and its node allocation is exhausted (also zero nodes).
- `time_limit`: started, but unresolved when the shared deadline stops search.
- `skipped_proven`: unnecessary after bidder proof, zero nodes and `searched=False`.
- `cancelled_proven`: a running task stopped on proof, with any sound partial
  bounds and actual visited nodes retained.
- `cancelled_deadline`: deadline prevented submission/start, or a queued future
  was cancelled; zero nodes and `searched=False`.

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
index. Exhaustive tasks are submitted in canonical order; proof mode uses the
deterministic bidder-interleaved priorities above. Worker assignment and
completion order can vary. Only the validated deal/configuration
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
sequential traversal when ordering is disabled, and optional parallel
root-action implementation. Their root
is counted once. Parallel branches reserve disjoint node allocations in canonical
waves, reclaiming unused allocations for later waves; sum of root and child
nodes stays within the position's `max_nodes`. All branches share that position's
original deadline, including input validation and worker startup. An interrupted
allocation is not restarted. Existing result dataclasses, exact/proof semantics,
and the exhaustive reference remain unchanged. Move ordering affects traversal only; neither heuristics nor cancellation
substitute estimated utilities for unresolved nodes. Worker exceptions identify the
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

See the [command reference](command_reference.md#endgame-oracle)
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
Repeat the [point command recipes](command_reference.md#endgame-oracle)
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

### Ordering and proof-search ablations

Measured on 2026-10-15 with CPython 3.13.5, Intel Core i5-12600KF,
eight available physical CPUs and WSL2/Linux 6.6.87.2. Each complete-deal row
uses four spawned workers, the arena deal for the printed seed with
`shaker=seed % 4`, 2,000 nodes per lead, one 10-second global deadline, and
100,000 aggregate cache entries. No row reached the time limit. Validation,
startup, coordination, cancellation and shutdown are timed; input generation
and printing are excluded. The progress callback collects events in memory.
These are single observations, not averaged performance guarantees.

| Configuration | Node ordering | Task objective | Bidder cancellation |
| --- | --- | --- | --- |
| Baseline | none | all_action_values | off |
| A | combined | all_action_values | off |
| B | none | maximum_contract | off |
| C | combined | maximum_contract | off |
| D | combined | maximum_contract | on |

B/C/D additionally use the proof mode's common deterministic task priorities;
C versus D isolates cancellation. Baseline versus A compares the same exhaustive
request. B versus C compares the same proof request without cancellation. A
proof request is **not** exhaustive solution of every alternative, so cross-mode
speedup alone does not establish a faster exhaustive solver.

| Seed | Config | Nodes | Wall s | Nodes/s | AB cutoffs | TT hits | TT cutoffs | Max contracts proven | Exact leads | Skipped by proof | Running proof cancellations | Speedup vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Baseline | 500239 | 3.3171 | 150806 | 37250 | 61575 | 60922 | 2/4 | 9/252 | 0 | 0 | 1.00× |
| 42 | A | 438274 | 3.2066 | 136677 | 43632 | 62761 | 61806 | 2/4 | 53/252 | 0 | 0 | 1.03× |
| 42 | B | 485182 | 3.2485 | 149355 | 80499 | 68768 | 68726 | 2/4 | 7/252 | 0 | 0 | 1.02× |
| 42 | C | 400684 | 2.9231 | 137076 | 59120 | 60972 | 60972 | 2/4 | 45/252 | 0 | 0 | 1.13× |
| 42 | D | 194200 | 1.4482 | 134101 | 28277 | 30130 | 30130 | 2/4 | 15/252 | 124 | 0 | 2.29× |
| 7 | Baseline | 503930 | 3.3186 | 151852 | 42987 | 69451 | 69139 | 0/4 | 1/252 | 0 | 0 | 1.00× |
| 7 | A | 495222 | 3.6380 | 136124 | 31788 | 92389 | 92017 | 3/4 | 16/252 | 0 | 0 | 0.91× |
| 7 | B | 499792 | 3.3854 | 147630 | 89630 | 78962 | 78858 | 0/4 | 0/252 | 0 | 0 | 0.98× |
| 7 | C | 477096 | 3.6507 | 130688 | 67433 | 88880 | 88818 | 3/4 | 15/252 | 0 | 0 | 0.91× |
| 7 | D | 189395 | 1.3898 | 136277 | 25940 | 39915 | 39915 | 3/4 | 4/252 | 150 | 2 | 2.39× |

Every row settled 252 tasks, but no row proved all four bidder contracts or
completed all individual values. With seed 42, D versus C used 51.5% fewer nodes
and was 2.02× faster. With seed 7 it used 60.3% fewer nodes and was 2.63× faster.
D retained fewer exact lead values because those are not required for its bid
proof objective. Combined exhaustive ordering was slightly faster for seed 42,
but about 10% slower for seed 7 despite fewer nodes. Node-limited task counts
were respectively 243/199/230/143/65 for seed 42 and 251/236/242/201/82 for seed 7
in Baseline/A/B/C/D order. Proof-classified but non-exact lead counts were
0/0/15/64/48 and 0/0/10/36/14. No task was deadline-limited in these rows.

Score-bound cutoff totals were 97,368/82,668/76,057/68,096/34,903 for seed 42
and 86,993/89,616/66,775/67,585/25,399 for seed 7. Ordering-call totals were
0/167,079/0/164,709/81,425 and 0/181,722/0/189,627/75,559. Additional TT queries
and heuristic calculations can offset pruning gains. Scheduling-dependent
private-cache assignment can also change bounded work between runs.

For fixed endgames, all ordering modes completed exact point values and identical
complete optimal-action sets. The following totals use HHR-generated trick-boundary
positions, 500,000 nodes/10 seconds per position, 100,000 cache entries, one worker,
and every root action. Sizes 2–4 use seeds 0–9, size 5 seeds 0–2, and size 6 seed 42.
No position reached either limit. Validation is included, input generation excluded.

| Dominoes/player | Samples | Ordering | Nodes | Total wall s | AB cutoffs | TT hits | Ordering calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 10 | none | 417 | 0.0107 | 2 | 0 | 0 |
| 2 | 10 | heuristic | 385 | 0.0123 | 2 | 0 | 64 |
| 2 | 10 | tt | 417 | 0.0103 | 2 | 0 | 64 |
| 2 | 10 | combined | 385 | 0.0103 | 2 | 0 | 64 |
| 3 | 10 | none | 3800 | 0.0728 | 116 | 300 | 0 |
| 3 | 10 | heuristic | 3110 | 0.0686 | 157 | 280 | 969 |
| 3 | 10 | tt | 3800 | 0.0813 | 116 | 300 | 1094 |
| 3 | 10 | combined | 3110 | 0.0737 | 157 | 280 | 969 |
| 4 | 10 | none | 27343 | 0.5375 | 2240 | 4686 | 0 |
| 4 | 10 | heuristic | 14230 | 0.3366 | 1309 | 2535 | 5737 |
| 4 | 10 | tt | 27343 | 0.5296 | 2240 | 4686 | 10608 |
| 4 | 10 | combined | 14230 | 0.3202 | 1309 | 2535 | 5737 |
| 5 | 3 | none | 58016 | 1.1385 | 5530 | 10542 | 0 |
| 5 | 3 | heuristic | 59820 | 1.4100 | 9517 | 12936 | 25533 |
| 5 | 3 | tt | 57986 | 1.1264 | 5529 | 10518 | 21182 |
| 5 | 3 | combined | 59820 | 1.3940 | 9517 | 12936 | 25533 |
| 6 | 1 | none | 158490 | 3.3532 | 19413 | 37506 | 0 |
| 6 | 1 | heuristic | 22331 | 0.5539 | 3450 | 4993 | 10633 |
| 6 | 1 | tt | 158467 | 3.3350 | 19417 | 37491 | 71056 |
| 6 | 1 | combined | 22331 | 0.5474 | 3450 | 4993 | 10633 |

Combined ordering reduced the six-domino sample from 158,490 to 22,331 nodes
and 3.3532 to 0.5474 seconds (6.13×). The five-domino sample instead used more
nodes and took 1.3940 versus 1.1385 seconds. TT-only ordering changed few nodes
on these full-window workloads because most table hits already resolve nodes;
threshold search can reuse bound-entry hints across successive windows.
Sub-millisecond timing differences in tiny samples are not evidence of gains.
The heterogeneous results are why `none` remains the default.

To reproduce the complete-deal ablations, use the maximum-contract command in
the command reference with seed 42 or 7, four workers, 2,000 nodes and ten seconds:
Baseline/A use `--output all_action_values` with `--move-ordering none/combined`;
B/C use `--output maximum_contract --no-proof-cancellation` with `none/combined`;
D uses `--output maximum_contract --move-ordering combined` with cancellation.
All share the same node, time and cache caps; repeated threshold queries consume
the same per-task counter. For endgames, use the existing `--samples`, `--remaining`,
`--objective final_points`, `--move-ordering` and explicit limits above.

An additional deadline CLI smoke used seed 42, two workers, TT ordering,
maximum-contract mode, 100,000 nodes per lead and 0.2 seconds global time. It
settled all 252 tasks in 0.2510 seconds including shutdown, with 9,962 nodes,
1,642 cache hits, one exact lead, two time-limited searches and 249
`cancelled_deadline` tasks. One bidder's contract was proven; the rest remained
unresolved. This demonstrates cooperative interruption, not a timing guarantee.
No unbounded full-hand search or model training was launched.

The optimization baseline at commit `346c762` passed all 52 existing oracle
tests. Verification after the changes passed **269 repository tests**, including
67 oracle tests (15 new optimization tests), plus `git diff --check`. New tests
cover all four orderings, legal/illegal TT hints, both acting/reference teams,
all nine trumps, partial tricks, bound classifications, interrupted search,
proof-skipped tasks, simultaneous/late results, real spawn/shared-memory stop
signals, node/time limits, and exception cleanup. Shared proof memory is verified
unlinked after worker failure. An additional 320 default-order endgame calls
matched the baseline including nodes, cache hits and proof fields (excluding
elapsed time and new instrumentation). Another 768 three-domino threshold
searches matched terminal-only exhaustive minimax with caching disabled,
single-entry and normal caches, both reference teams, partial tricks and
none/combined ordering. No pre-existing correctness failure was observed or
silently repaired.

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
