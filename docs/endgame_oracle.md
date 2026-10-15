# Perfect-information endgame oracle

`texas42.oracle` solves the remaining play of one ordinary contract with all four
hands visible. It is a separate analysis/search component, with no PPO, belief
sampling, policy adapter, or changes to observations and training. Engine rules,
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

## Running and validation

See the [command reference](command_reference.md#perfect-information-endgame-oracle)
for copy-ready example/benchmark commands and all flags. The example uses arena
deal generation and existing ordinary policies, stops after legal engine play,
and explicitly prints all four hands. It generates positions rather than adding
a new state file format: existing schema-5 serialization restores actor
observations, not arbitrary mutable full-state snapshots.

Focused verification from the repository root:

```bash
python -m unittest discover -s tests -p test_oracle.py -v
```

Tests compare exhaustive terminal-only minimax with every root action under
disabled, single-entry and normal caches across seeded reachable endgames. They
also cover ordinary mark contracts, all trump forms, both teams to act, forced
moves, tied optima, outcome-changing moves, same-team turns at trick boundaries,
partial tricks, terminal results, caller immutability, key separation, table
bound windows/eviction, invalid inputs, CLI output, and deterministic node-budget
interruption. A proven root with unresolved actions is explicitly exercised for
both bidding and defending actors.

## Bounded smoke benchmark

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
