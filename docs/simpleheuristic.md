# Simple Heuristic Agent V1

`from texas42.agents import HeuristicAgent` provides a deterministic, hand-crafted
baseline for future comparisons with learned policies. It uses only the acting
player's immutable observation and legal global action IDs. It implements
`reset(seat)` and `act(observation)` and works with `SelfPlayEnv`, `AgentPlayer`
and the arena. Each seat needs a fresh instance. Reset once per episode, retaining
that assignment across hands in a match. The heuristic has no RNG or learned state.

## Frozen V1 policy

Doubles compete directly with numbered suits 0–6 for both bidding and trump
selection. Choose the best suit by descending population, strongest
suit rank, then counter points within that suit. Exact ties favor earlier suits
in that order. Doubles count as numbered trump when they contain its number.
The desired opening bid is:

| Best suit | Desired bid |
| --- | --- |
| Seven trumps | Five marks |
| Six trumps with at least five counter points | Two marks |
| Five trumps with at least five counter points | 37 points |
| Six trumps with fewer than five counter points | 32 points |
| Five trumps with fewer than five counter points | 30 points |
| Four or fewer trumps | Pass |

These are targets, subject to engine legality. Open with the largest legal bid
at or below the target (so seven trumps opens at two marks, not five). When a bid
already exists, choose the smallest legal raise within the target. The first mark
bid is at most two marks; subsequent mark bids increase by one. Pass when there is
no affordable raise; if forced after three passes, choose the smallest legal bid.
V1 treats a partner's existing bid like an opponent's bid.

For trump selection, always choose the best suit defined above. This heuristic
no longer selects Follow Me, even when doubles dominate the hand: seven doubles
select doubles trump. Doubles can also win when their population ties a numbered
suit and their rank/counter tie-breakers win. For example, `66 55 44 60 61 32 10` ties doubles
and sixes at three tiles and the same top rank; the double-five counter selects
doubles. Regression tests cover all seven numbered selections, doubles dominance,
and deterministic ties. Follow Me remains legal in the engine, and the heuristic
can still play a hand where another agent selected it.

Trick priorities, always restricted to the engine's legal plays:

1. Lead the strongest trump; with no trump, lead the strongest available domino.
2. If the partner wins and the player is last, or the winning tile is classified
   unbeatable (highest trump double, or a double in Follow Me), contribute the
   most counters, breaking ties toward weaker tiles.
3. If the partner may be overtaken, discard the lowest-cost tile.
4. Against an opponent, take a trick with the weakest sufficient winner when
   last to play or when the current trick contains at least five counter points.
5. Otherwise discard by ascending counter value, power, then pip total.

Power compares trump status and suit rank; exact play ties retain legal-action
order. The policy does not infer hidden hands, track remaining trump, plan future
tricks, or protect a partner's auction. Its cross-suit power comparisons and
counter thresholds are simple approximations. Tests freeze these priorities;
benchmark results do not silently trigger policy tuning.

## Reproducible benchmarks

```bash
python -m examples.heuristic_benchmark --opponent random --deals 1000 --seed 42
python -m examples.heuristic_benchmark --opponent heuristic --deals 1000 --seed 42
python -m examples.heuristic_benchmark --opponent random --deals 20 --seed 42 \
  --episode match --target-marks 7 --bootstrap-seed 123 \
  --bootstrap-samples 2000 --output /tmp/heuristic-results.json
```

The benchmark calls `arena.compare_agents()` with independent seat instances.
A is always Heuristic V1; B is the selected opponent. Absolute-seat hands and
shaker remain fixed when partnerships swap. In match mode a group is an entire
predetermined schedule and the two matches can terminate after different numbers
of hands. Seed controls deals and random agents; bootstrap seed independently
controls resampling. Heuristic factories accept but do not use the seed.

Terminal output uses aligned tables rather than raw JSON. Runtime measures wall-clock
time for evaluation plus bootstrap, excluding formatting and file export; games/s
and decisions/s use that same duration. Compare timings using the same episode,
bootstrap settings, hardware, and Python environment. Timing is displayed only,
so exported results remain reproducible. `--output` still writes JSON for analysis.

Output includes games/groups, policy win rates, signed marks per played hand,
contracts declared, success rates, mean contract value **in marks**, bidding
frequency (non-pass calls per controlled seat turn), and average match length.
Undefined rates print `unavailable`. Breakdowns count successful/failed resolved
contracts, point versus mark bids, trump selections, and outcomes by final bid
value. Initial hand-strength categories and losing auction calls are not retained
in arena results; their outcome breakdown is intentionally omitted rather than
adding privileged tracking. JSON uses `ArenaReport.to_dict()` plus benchmark
labels and these derived breakdowns, without modifying training records.

The 95% percentile bootstrap resamples complete duplicate groups; fewer than two
groups has no interval. Small smoke runs are integration checks, not precise
estimates. See [arena.md](arena.md) for weighting and interval definitions.
Overlapping intervals are not a formal hypothesis test.

Heuristic versus heuristic uses identical deterministic policies. Each pair has
identical absolute-seat play and opposite policy labels: the paired outcome is
a symmetry and integration check, not evidence of competitive strength.
Beating uniform random play also does not establish competitive strength.
Future comparisons should freeze V1, use held-out schedules, report seeds and
versions, and compare stronger baselines without modifying this policy post hoc.
