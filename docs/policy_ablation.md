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
and mean value, not zero. Bidding frequency includes forced bids. Resolved 30-point contracts are also split by forced/voluntary origin below;
intermediate auction calls and pass rates are not counted in that breakdown.

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

## Forced versus voluntary 30-point contracts

The shared benchmark table and JSON `thirty_point_bids` breakdown separate
forced/voluntary successful/failed resolved 30-point contracts. A forced bid
comes from the engine's `BidView.forced`: after three passes with no bid, the
shaker must bid. This flag is captured before the action, then associated with
the final winning bidder and bid. Each arena hand exports `forced_bid` for any
contract value, including mandatory bids above 30. This is evaluation metadata;
scoring results and training records are unchanged.

A voluntary opening 30 and a mandatory 30 can reflect different hand selection
and bidding opportunities. Separating them makes the aggregate contract success
rate easier to interpret. Being the shaker alone is insufficient: the shaker
may voluntarily raise a previous bidder. Intermediate overbid calls are excluded.
Empty categories show N/A, not a zero success rate.

These are observational success rates. They do not show whether passing, raising,
or choosing another contract would have improved an individual hand: policies,
hands, and subsequent decisions differ across the categories. No counterfactual
or causal comparison follows from this breakdown alone.

## When the contract becomes certain

Both benchmark entry points report contract decision timing, assigned to the
resolved declaring policy, with successful/failed and forced/voluntary groups.
For normal target T the first completed trick with declarer points ≥ T or
defender points > 42−T decides the contract. Mark bids target all 42 points.
For example, 12 defender points do not defeat a 30 bid, but 13 do.

Deciding a contract is different from completing the hand: every hand still
plays all seven tricks and 33 decisions, with unchanged terminal scoring and
rewards. Mean remaining tricks measures `7 - deciding trick`; multiply by four
for subsequent domino plays. JSON adds `contract_decided_at_trick` per hand and
`decision_timing` summaries/distributions in each policy's breakdown. Unknown
legacy timings are excluded and empty means are N/A. See [arena.md](arena.md).

These diagnostics describe normal contracts only. Earlier decisions may reflect
bid difficulty or the deal, not better or worse trick play. They do not establish
optimal resource use or a causal phase contribution, and must not be applied
unchanged to future special contracts.

## Bid versus pass interventions

Resolved-contract origin reporting now covers all final bid values, including
forced bids above 30, with declaring-team signed marks. It counts winning
contracts, not all auction opportunities. The separate
[counterfactual bidding experiment](counterfactual_bidding.md) selects the first
voluntary target-policy bid and replays BID/PASS from the same deal and prefix.
It estimates paired acting-team mark differences under unchanged continuation
policies; it does not replace ordinary duplicate-deal arena metrics.

### Policy preference in a forced auction slot

`forced_bid` continues to mean that passing was illegal for the resolved bidder.
It does not mean that the policy wanted to pass. Evaluation now also records
`would_pass_if_allowed`: `true` identifies an unwilling bidder (the current
heuristic and conservative policies then take the minimum bid, 30), while
`false` identifies a willing bidder who happens to occupy the last auction slot.
A willing bidder can also choose 30; bid value alone cannot distinguish these cases.

The shared benchmark table splits forced contracts by final bid value and policy
preference, with successful and failed counts. Existing forced/voluntary tables
retain their auction-rule meaning. JSON includes `forced_bid_preferences` and the
per-hand preference field. `null` means unknown or inapplicable, not unwilling.

Heuristic and conservative agents provide the optional, side-effect-free
`would_pass_if_allowed(obs)` diagnostic; HybridAgent forwards its bidding
policy's answer. Random and uninstrumented custom agents report unknown. The
arena queries it before the decision, never calls `act` a second time, and does
not change observations, legal actions, RNG streams, or game rules. Wrappers
must forward the diagnostic to preserve this metadata. This describes the
policy's preference, not whether bidding was objectively better than passing.
