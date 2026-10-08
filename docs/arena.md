# Reproducible policy evaluation

`texas42.arena` runs ordinary `Agent` implementations through `EpisodeState`.
It adds no rules, training dependencies, or changes to observation encodings.
Results are frozen dataclasses; the runner never prints. Run the baseline with:

```sh
python -m examples.arena_benchmark --deals 30 --seed 42 --bootstrap-seed 123
```

## Comparing partnerships

```python
from texas42.agents import RandomAgent
from texas42.arena import compare_agents

def factory(seat, seed):
    return RandomAgent(seed)

report = compare_agents(factory, factory, num_deals=100, seed=42,
                        bootstrap_seed=123, bootstrap_samples=2000)
print(report.summary())
```

Each factory receives `(seat, seed)` and must return a **fresh instance** on
every invocation. Shared immutable model weights are fine; mutable memory
belongs to that instance. Four distinct instances are required within a game.
The arena calls `reset(seat)` once, then only calls `act(observation)` on the
acting seat. Match hand boundaries do not reset memory. Factories are responsible
for deterministic initialization and avoiding external/global random sources.

Each deal is played with A at seats 0/2 and B at 1/3, then with labels swapped.
Absolute-seat hands and shaker remain identical. Each pair has a `group_id`;
its hands retain content-derived `deal_id` values. Shakers cycle across groups.
Changing decisions cannot consume or change the deal-generation RNG stream.
SHA-256-derived seeds separate deal, agent, and bootstrap randomness from one
another and from Python's process-randomized `hash()`.

Agent seeds are coupled by **absolute seat within each pair**, not by policy.
This controls initialization seeds; different policies may consume randomness
differently. Both policies play every seat in a pair. This controls deal and
partnership-position effects, but does not erase policy/seat interactions,
partnership conventions, or stochastic decision differences. Outcomes need not
be opposites. With identical deterministic seeded policies, the two games are
identical apart from labels: random-vs-random in the example therefore yields
50% paired wins and zero signed marks, often with degenerate intervals. This is
a consequence of seed coupling, not a general claim about random-agent win rates.
Use distinct policies and multiple experiment seeds for substantive comparisons.
Reproducibility assumes the same engine, factory code, and Python RNG behavior.

## Explicit games, rotations, and matches

```python
from texas42.arena import generate_schedule, run_game, policy_metrics, bootstrap

# Supply four different factories for heterogeneous partnerships if desired.
factories = (factory, factory, factory, factory)
labels = ('north', 'east', 'south', 'west')
schedule = generate_schedule(1, seed=12, shaker=0)
games = []
for rotation in range(4):
    games.append(run_game(
        factories[rotation:] + factories[:rotation], schedule,
        seating=labels[rotation:] + labels[:rotation], seed=91,
        game_id=f'rotation-{rotation}', group_id=schedule[0].deal_id))

# Complete match, with official shaker rotation and persistent agent memory.
target = 7
match_schedule = generate_schedule(2 * target - 1, seed=12)
match = run_game(factories, match_schedule, seating=labels, seed=91,
                 episode='match', target_marks=target)

paired_matches = compare_agents(factory, factory, num_deals=20, seed=42,
                                episode='match', target_marks=7)
```

Explicit permutations let four distinct policies occupy every seat. Rotate both
factories and labels together; keep the underlying schedule fixed. `run_game`
accepts any four per-seat factories, including heterogeneous partnerships.
Assign a common group ID to all rotations of one independent schedule. Separate
independent schedules need distinct group IDs. The default group ID is the first
deal's content ID. `compare_agents` handles these assignments automatically.

In match mode, `num_deals` means **paired match schedules**, not hands played.
All schedules are generated before any agent runs. A match can require at most
`2 * target_marks - 1` hands: every hand awards at least one mark to a team.
`run_game` requires this capacity and validates every scheduled shaker against
the official rotation. An early-finishing match leaves an unused suffix; its
swapped counterpart starts at the same schedule's beginning. The next pair uses
an independent schedule, unaffected by either match's length. `schedule_ids`
records the full schedule, while `hands` contains only played hands.

## Results and metric definitions

`GameEvaluation` records mode, target, assignments, agent seeds, schedule IDs,
played hands, final marks, winner, and terminal seat rewards. Hand-mode winner
means the scoring partnership. Match-mode winner and rewards come directly from
`EpisodeState`; intermediate match hand rewards are zero. `HandEvaluation`
contains shaker, deal ID, seating, decisions, non-pass bid counts, rewards, and
the engine's complete `HandResult`: bidder, contract kind and bid, trump, team
points, scoring team and marks. `contract_success` compares declaring and scoring
teams. Ordinary hands have 33 decisions.

`policy_metrics(games, label)` reports:

- Games played: episodes containing that policy, counted once per episode.
- Win rate: mean episode win indicator from that policy's perspective.
- Mean signed marks: marks awarded with positive sign on the policy's team and
  negative sign otherwise, averaged over **played hands**, not matches.
- Contract success: successes divided by auctions won by that policy.
- Mean contract value: marks at stake per auction won, not the point threshold.
  Point bids and one-mark bids both have a value of one mark.
- Bidding frequency: non-pass calls divided by bidding turns belonging to that
  policy. Winning the auction is counted separately in `contracts_declared`.

For homogeneous partnerships each policy counts once per game/hand. For labels
appearing on both opposing teams, outcomes are averaged over that label's seats;
a label on all seats receives win rate 0.5 and signed marks zero. This makes
heterogeneous seat assignments explicit without counting partners twice.
`ArenaReport` also exposes overall contract success and mean match length.
Undefined denominators return `None`, including success/value for never-declarers
and match length in hand-only experiments. These are not zero-percent estimates.

```python
import json
with open('arena-report.json', 'w') as output:
    json.dump(report.to_dict(), output, indent=2)
```

`to_dict()` includes derived success/length fields and preserves nested contract
values. For CSV, select fields from each game/hand into rows with `csv.DictWriter`;
retain group IDs and policy assignments so downstream analysis stays paired.
The arena report is separate from schema-5 training logs.

## Uncertainty

`bootstrap(games, policy, seed=123, samples=2000)` samples complete duplicate
groups with replacement, keeping all games and all played hands in each selected
group. It recomputes policy win rate and the ratio of signed marks to played
hands each time. Thus match-length differences remain part of the estimator.
The reported 95% interval uses the 2.5th and 97.5th percentiles of those estimates,
with linear interpolation between sorted samples. The bootstrap seed is separate
from experiment randomness and is recorded in `ArenaReport`.

Fewer than two independent groups, or an undefined bootstrap replicate, produces
an unavailable interval (`None`). Small datasets and few resamples give unstable
intervals; even a zero-width interval does not establish certainty. These
intervals characterize the sampled deal/seed schedule and fixed policies, not
variation across training runs. Overlapping intervals are not a formal
hypothesis test. Do not bootstrap individual games of a duplicate pair or
individual hands of a match as if independent.

## Safety boundary and future work

Invalid IDs (including bools, floats, reserved contracts, and wrong-phase IDs)
are rejected before transitions. Factory, reset, and decision exceptions raise
`AgentExecutionError` with game, hand, seat, phase, and the original exception as
its cause. A failed evaluation is abandoned; no action retries or partial report
are returned. There is no rollback, timeout, process isolation, or protection
against deliberately malicious Python agents. Schedules/results are immutable;
ordinary policies receive no schedule, engine state, or opponent hands.

A future privileged-oracle runner must call `PrivilegedAgent.act_privileged`
through a separate explicit boundary. It must label those results separately
from ordinary imperfect-information policy evaluations. PIMC, beliefs, heuristic
agents and learned policies can use the existing ordinary factories. Next steps
are a deterministic heuristic baseline, held-out deal schedules, checkpoint
metadata validation in learned-agent factories, and comparisons across multiple
training seeds. This module does not implement models, search, training, special
contracts, or an oracle.

## Resolved bid origin

`HandEvaluation.forced_bid: bool = False` records whether the final winning
contract originated in an engine-marked mandatory bid. The runner snapshots
`BidView.forced` before each non-pass bidding action and resolves the origin by
winning bidder and bid. Bookkeeping restarts every hand, including in matches.
Three passes with no bid force the shaker to bid; shaker identity itself does
not imply forced bidding, since the shaker can also voluntarily raise an
existing bid. A forced bid need not be 30 points.

This immutable evaluation-only field is included by `ArenaReport.to_dict()`;
`HandResult`, observations, training data, metrics, and bootstrap remain unchanged.
The default preserves existing hand-record construction, but old records without
origin data cannot retrospectively establish whether a contract was forced.

Both heuristic and policy-ablation benchmarks add `thirty_point_bids` to each
policy's JSON breakdown and a shared terminal table for forced/voluntary
successful/failed final 30-point contracts. Counts refer only to the resolved
contract, not superseded auction calls. Empty categories have N/A success rates.
Separating mandatory contracts from voluntary openings helps reveal selection
effects hidden by pooled 30-point statistics; it is observational and does not
establish which alternative action would have performed better.

## Contract decision timing

`HandEvaluation.contract_decided_at_trick` records the first completed trick
(1–7) after which a normal contract's outcome is mathematically certain. The
arena reads the engine's cumulative points only after a trick completes. For
target T, declarers succeed once they have at least T points; they fail once
defenders have more than 42−T points. Point bids use their stated target and
all mark bids require 42. Thus a 30-point contract fails at 13 defender points,
while a mark contract fails at the first defender point. Equality at the success
target counts; equality at 42−T defender points does not establish failure.

This is evaluation metadata, not an early-stop rule. Every hand still plays all
seven tricks, makes 33 decisions, and distributes all 42 points. Rewards are
issued at the same time as before. The first deciding trick is never overwritten,
including when the last trick also ends a match. No observation, training record,
agent, or scoring API changes. Older manually constructed evaluation records may
have None; their timing is unavailable rather than assumed to be seven.

Both benchmark JSON breakdowns include `decision_timing` per declaring policy:
contract and recorded counts, mean deciding trick, mean remaining tricks, and a
seven-bin distribution keyed by string trick numbers. The same summaries are
provided by outcome (successful/failed) and origin (forced/voluntary). Unknown
timings are excluded from means/distributions, with the `recorded` count explicit.
The shared terminal tables show these groups too; empty means are N/A.

Remaining complete tricks equal `7 - contract_decided_at_trick`; subsequent
domino plays equal four times that number. These counts have no reward or penalty.
Earlier success/failure reflects bid difficulty, deal and opposing policies as
well as play. Timing alone cannot measure optimal play or prove that resources
were wasted. The formula is explicitly limited to normal contracts; future
special contracts with different winning conditions require separate diagnostics.

Forced auction status and policy preference are distinct. `forced_bid` records
that the winning bidder could not legally pass; the optional
`would_pass_if_allowed` field records whether that policy would have passed if
permitted. True means compelled, false means willing, and None means unavailable
or not a forced call. The optional `BidPreference` protocol in `agents.base`
requires a pure diagnostic that cannot consume randomness or update memory.
See [policy_ablation.md](policy_ablation.md#policy-preference-in-a-forced-auction-slot)
for reporting and examples.
