# Counterfactual bidding experiments

A declarer's success rate does not answer whether bidding was better than
passing. Passing may let a partner or opponent win the auction, change trump,
and change the acting team's ultimate marks. This experiment estimates a
one-action BID-versus-PASS intervention under specified continuation policies.
It does not change Heuristic V1 or any other policy.

## Running

```bash
python -m examples.counterfactual_bidding \
  --policy-a HHH --policy-b CHH --target-policy HHH \
  --deals 128 --seed 42 --bootstrap-samples 200 --output results.json
```

All six ablation variants are accepted for either policy and the target. The
target must be one of the configured policies. If both sides use the same name,
all seats are target seats. Each deal group produces two original hands, swapping
policies over the same absolute-seat hands and shaker. Schedules and per-seat
seeds use exactly the existing arena scheme. `--bootstrap-seed` defaults to 0;
`--target-marks` defaults to 7 and is retained in hand observations. Only hand
mode is supported; `--episode match` explicitly fails. A future match experiment
needs coherent prior scores, policy memory, schedules and a choice between
immediate hand marks and final match outcomes.

Programmatic use:

```python
from texas42.experiments.counterfactual_bidding import compare_bidding
report = compare_bidding('HHH', 'CHH', target_policy='HHH', num_deals=128,
                         seed=42, bootstrap_seed=0, bootstrap_samples=200)
data = report.to_dict()
```

`evaluate_hand(factories, deal, seating=..., target_policy=..., seed=...)` also
supports explicit fixtures and arbitrary ordinary-agent factories.

## Selection and replay

1. Run the original hand normally with the arena.
2. Select the first target-policy bidding decision where pass is legal and the
   chosen action is a legal non-pass bid. Forced bids are excluded using the
   existing legal actions and observation flag.
3. Recreate all four agents with the same seeds for each replay, from the same
   deal and shaker. Call the policies normally through the selected decision.
   Require observations (including legal IDs) and chosen actions to match the
   original prefix exactly. The complete BID replay must match the original
   trajectory and game result, including decision timing; any mismatch aborts.
4. In the PASS replay replace only the selected action with a legal pass. The
   policy is still called at that decision, consuming the same RNG draws and
   internal updates. All subsequent actions are ordinary policy decisions.

No mutable agents are shared across replays. Factory reuse is rejected, as are
illegal actions. Wrappers receive only ordinary immutable observations; the
arena owns the deal and engine. All branches finish seven tricks, 33 decisions
and 42 points. The original runner, legality, scoring and rewards are reused.
There is no new transition engine, early stopping, or general callback API.

Subsequent choices can diverge, including auction winner, trump and played
tricks. Random streams agree through the intervention but are not artificially
coupled afterward if histories or draw counts differ. A policy that remembers
its proposed action receives the same policy-side update at intervention; it
must use later public observations to see the applied action. This convention
is explicit and reproducible for the current agents.

Selection counts include original hands with no intervention, eligible hands,
selected interventions, and later eligible original-trajectory decisions skipped.
Hands with no intervention are not zero-effect samples. Later opportunities in
the PASS trajectory do not change selection. The format retains group and game
IDs so future work can evaluate multiple opportunities while preserving clusters.

## Outcomes and features

For the acting partnership, signed marks are positive marks awarded when it
wins, otherwise negative marks awarded. `delta_marks = BID marks - PASS marks`:
positive favors BID, negative favors PASS, zero means identical signed marks.
The PASS contract may belong to another team; comparison never substitutes
contract success for acting-team marks.

Raw records preserve each branch's complete HandEvaluation, acting-team win,
contract success, signed marks and public action IDs. Thus final bidder/bid,
trump, team points, rewards, origin and deciding trick remain inspectable.
Records contain deal/group/game IDs, acting seat/policy, shaker, one-based auction
position, structured original bid and global action IDs, per-seat seeds, and
whether both branches have the same winning partnership. The full deal and
other actors' observations are not exported. Complete played actions are public
post-hand information, so raw records are research outputs, not policy inputs.

Actor-only descriptors include its seven domino IDs, candidate suit/length,
doubles, hand/suit counter points, highest-trump ownership, current bid and
bidder's absolute partnership, whether its partner leads, scores and target.
The candidate suit is the most populated of 0–6 and doubles, breaking ties by
highest rank then counters, then enumeration order, as in V1. Follow Me is not
a candidate suit. Highest trump is the numbered double or 6–6 for doubles.
These are analysis labels only; no new observation encoding or policy input
has been introduced.

Overall and conditional tables show sample counts, mean/median delta, BID/PASS/
equal counts and fractions, and mean marks in each branch. JSON retains all
raw records plus summaries by proposed bid, suit length, doubles, suit counters,
highest-trump ownership and whether a partner leads. Empty estimates are None
in JSON and N/A in terminal output. Timings are terminal-only for deterministic
JSON exports. Calling `to_dict()` calculates summaries and bootstrap intervals.

The ordinary benchmarks additionally report `contract_origins` by **resolved**
final bid and forced/voluntary origin, with success/failure counts, success rate,
and declaring-team mean signed marks. Forced higher bids are classified from
recorded engine origin, not value. Existing 30-point and other keys remain.
These resolved-contract statistics differ from counterfactual opportunities:
a voluntary bid can be eligible even when it is subsequently overbid.

## Uncertainty and interpretation

The 95% percentile interval for mean paired delta uses a separate bootstrap
seed. Each draw selects a whole original duplicate-deal group with replacement,
retaining all its interventions. BID/PASS branches are never separate samples.
Means weight interventions, not groups; unequal group sizes remain intact.
For overall or conditional summaries, only groups containing a qualifying
sample contribute; fewer than two contributing groups yields N/A. Intervals
use linear interpolation of the bootstrap mean distribution. They describe
uncertainty conditional on selection, not eligibility prevalence or unobserved
categories. Sparse groups can produce unstable or degenerate intervals.

The estimate depends on deal distribution, opponents, trump/play policies,
policy memory and stochastic continuation. It is not proof of a Nash equilibrium,
game-theoretically optimal bidding, or an additive contribution from bidding.
Evaluating only states where HHH already chooses to bid does **not** tell us
whether passing is optimal where HHH currently passes. Future experiments could
sample broader legal voluntary states, test several candidate bids, repeat
stochastic continuations and retain multiple interventions per duplicate group.
No special contracts, hidden-information oracle, learning model or match-value
estimation is implemented.
