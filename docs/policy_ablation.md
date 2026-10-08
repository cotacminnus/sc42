# Phase-wise baseline experiments

HeuristicAgent V1 is a frozen, hand-crafted baseline. Uniform RandomAgent bids
uniformly among legal calls, including large raises: frequent failed contracts
can dominate its opponent's win rate. Beating that opponent does not establish
competitive trick play. ConservativeRandomAgent passes whenever possible and
otherwise takes the lowest-strength legal bid. It selects trump and domino
plays uniformly. All legality comes from the existing observation and registry.
Neither existing baseline's strategy has changed.

## Policies and API

| Variant | Bidding | Trump | Playing |
| --- | --- | --- | --- |
| HHH | Heuristic V1 | Heuristic V1 | Heuristic V1 |
| HHR | Heuristic V1 | Heuristic V1 | Random |
| HRR | Heuristic V1 | Random | Random |
| CHH | Conservative | Heuristic V1 | Heuristic V1 |
| CRR | Conservative | Random | Random |
| RRR | Random | Random | Random |

```python
from texas42.agents import ConservativeRandomAgent, HybridAgent, variant_factory
from texas42.arena import compare_agents

agent = HybridAgent(bidding='heuristic', trump='random', playing='random', seed=42)
report = compare_agents(variant_factory('HHH'), variant_factory('CRR'),
                        num_deals=32, seed=42)
```

Factories create a new agent per seat; the runner calls `reset(seat)` once per
episode. Match hands retain state. All delegates use only immutable ordinary
observations, never opponents' hands or mutable engine objects. These agents
also support SelfPlayEnv and AgentPlayer through the unchanged Agent protocol.

ConservativeRandomAgent seeds `random.Random` at construction; bidding consumes
no random draws. Reset assigns the seat without restarting randomness. HybridAgent
uses `random.Random(seed).getrandbits(128)` three times, in bidding/trump/playing
order, even for deterministic phases. Each phase receives a separate delegate
and seed; resets do not rewind streams. Changing the trump strategy therefore
does not consume play's random draws. Different legal sets, acting orders and
histories can still cause divergent trajectories. Seeds and complete observation
sequences must agree to reproduce random choices. RRR is uniform random but is
not stream-identical to the original single-stream RandomAgent. Keep the original
heuristic benchmark for exact historical reproduction.

## Running experiments

```bash
python -m examples.policy_ablation_benchmark --policy-a HHH --policy-b CRR --deals 1024 --seed 42
python -m examples.policy_ablation_benchmark --policy-a HHH --policy-b HHR --deals 1024 --seed 42
python -m examples.policy_ablation_benchmark --policy-a HHR --policy-b HRR --deals 1024 --seed 42
python -m examples.policy_ablation_benchmark --policy-a HHH --policy-b CHH --deals 1024 --seed 42
python -m examples.policy_ablation_benchmark --policy-a HHH --policy-b RRR --deals 1024 --seed 42
```

HHH/CRR removes voluntary random overbidding. HHH/HHR primarily tests trick play;
HHR/HRR tests trump selection; HHH/CHH tests bidding. HHH/RRR preserves the
uniform-random comparison, subject to the RNG distinction above. Identical HHH
against HHH is a deterministic symmetry/integration check: absolute-seat play
is identical in each duplicate game, while the policy labels swap, giving
exactly 50% wins each. This is not an estimate of competitive strength.

Options include `--episode hand|match`, `--target-marks` (default 7),
`--bootstrap-seed` (default 0), `--bootstrap-samples` (default 2000), and
`--output results.json`. Hand mode is the default. The original
`examples.heuristic_benchmark` CLI is unchanged. No ML libraries are required.

Terminal output has tables for policy configurations, wins, signed marks,
auctions won, contract success, contract value, and bidding frequency. Resolved
contract tables give successful/failed counts, point/mark bid counts, trump
selection counts, and outcomes by final bid. Divide those counts by auctions
won for frequencies; failure rate is one minus success rate when defined.
Auctions won means contracts declared, not hands won: a failed declarer's
opponent receives the marks. Mean contract value is in marks, including for
point bids. A policy with no declared contracts has unavailable success rate
and mean value, not zero. Bidding frequency includes forced bids. We do not
record new action events for voluntary/forced/pass breakdowns.

Runtime measures evaluation plus bootstrap, excluding formatting and file export;
throughput uses that duration. Compare performance only with comparable hardware,
Python, episode settings and bootstrap counts. JSON includes report metrics,
seeds, policy configurations and contract breakdowns. Runtime remains terminal-only
so exported results are deterministic. Match output includes average hands per
match; signed marks per hand remain hand-weighted across differing match lengths.

## Interpretation and limits

The arena reuses predetermined absolute-seat hands and shaker within each pair,
then swaps partnerships. Per-seat seeds are coupled across the pair, and the
schedule RNG is independent of agents. Matches use the same precomputed schedule
but can terminate at different lengths. See [arena.md](arena.md) for definitions.

The dependency-free percentile bootstrap resamples complete duplicate groups,
not individual games, using a separate seed. It reports 95% intervals for win
rate and signed marks. Fewer than two groups yields unavailable intervals;
small samples can give degenerate or unstable intervals. Overlapping intervals
are not a hypothesis test, nor do small smoke runs establish significance.

A phase change alters subsequent observations, legal actions, and both teams'
choices. These comparisons control policy components but retain strategic
interactions; effects are neither additive nor causally independent. Use multiple
seeds, larger held-out deal schedules, and stronger opponents before drawing
strength conclusions. Do not tune V1 after seeing these results. Future learned
agents can use the same interface and arena without changing rules or schemas;
privileged oracle evaluation remains a separate future feature.
