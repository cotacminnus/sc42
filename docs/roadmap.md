# Research roadmap

This project uses Texas 42 as a compact testbed for imperfect-information,
partnership self-play research. It asks two questions:

1. How do explicit belief modeling and privileged-information training affect
   performance, sample efficiency, and decision quality compared with agents that
   learn only from their own observations?
2. How much explicit hidden-information inference is needed for competitive play
   in a small four-player partnership trick-taking game?

Step numbers match the team's live planning document.

## Status

| # | Step | Status | Where |
| --- | --- | --- | --- |
| 1 | Game engine | Done | `models.py`, `rules.py`, `state.py`, `engine.py` |
| 2 | Self-play setup | Mostly done | `selfplay.py`, `training.py`, `encoding.py`, `serialization.py`, `history.py`, `collection.py` |
| — | OpenSpiel compatibility | Done | `openspiel.py`, [openspiel.md](openspiel.md) |
| 3 | Perfect-information oracle | Endgame, complete-deal and arbitrary-position bounds implemented | `texas42/oracle.py`, [oracle guide](endgame_oracle.md) |
| 4 | Common agent interface | Done | `agents/`: protocols, RandomAgent, AgentPlayer |
| 5 | Baseline agents | V1 baselines done | HeuristicAgent, RandomAgent, ConservativeRandomAgent, HybridAgent; [phase ablations](policy_ablation.md) |
| 6 | Self-play training | Play-only and multi-phase historical PPO implemented | [legacy baseline](ppo_training.md), [current training](multiphase_selfplay.md) |
| 7 | Hidden-information reasoning | Belief toolkit, PIMC agent and model-guided PIMC (model rollouts, model likelihoods, value leaf) implemented, with an accuracy experiment | [belief modelling](belief.md); [search in training](search_training.md): theory and plan, ranking gate measured, not implemented |
| 8 | Partnership behavior | Paired-partner training and two-spec team evaluation implemented; studies not started | [paired partners](multiphase_selfplay.md#paired-partner-training) |
| 9 | Bidding | Neural bid-only/joint PPO and counterfactual tools implemented; strength to evaluate | [multi-phase PPO](multiphase_selfplay.md), [counterfactuals](counterfactual_bidding.md) |
| 10 | Special contracts (Nello) | IDs reserved | `actions.py` IDs 55–57 |
| 11 | Evaluation | Paired arena and phase composition implemented | [methodology](arena.md), [agent composition](phase_composable_evaluation.md) |

Next research priorities are multi-seed bidding ablations with identical frozen
playing weights, then joint historical self-play on held-out schedules. Existing
infrastructure supports these experiments; recurrent memory, beliefs, oracle
execution and partnership cross-play remain separate research milestones.

## Known gaps in the current code

- `RandomPlayer` never bids, so `--all-ai` CLI logs are always a forced 30 by
  the shaker. Use `SelfPlayEnv` with random legal IDs for bidding data.
- Ordinary CLI logs contain only actor-visible observations and public history.
  `TrainingCollector(include_initial_deal=True)` can separately record complete
  initial deals for belief/oracle labels. Offline oracle labels for the
  centralized critic's auxiliary head are implemented (`examples.oracle_labels`);
  belief-model labels remain future work. Ordinary policies receive only `Observation`.
- OpenSpiel's `deal_counts` observer field is always zero (see the TODO in
  `openspiel.py`).
- `pyspiel` wheels target Linux and macOS. On Windows, use WSL2; OpenSpiel tests
  skip locally and run in CI.

## Step-by-step plan

### 2. Self-play setup: feature encoder

The 387-feature actor-relative encoder, schema-5 observation round trips,
chronological public history and opt-in privileged deal labels are implemented.
Episode validation distinguishes verified replay from partial consistency checks.
See [encoding](encoding.md) for the authoritative API and privacy boundary.
Recurrent features and belief/oracle label generation remain future work.

### 3. Oracle (perfect-information solver)

Given all four hands, trump, and the leader, return the points each team can
still take under perfect play. Seats 0/2 maximize team-0 points; seats 1/3
minimize them. A tested prototype uses:

- 28-bit masks for hands and per-trump lookup tables built from `rules.py`;
- alpha-beta search that continues across trick boundaries;
- a transposition table keyed by `(hands, leader)` storing lower/upper bounds.
  At a trick boundary, the remaining value does not depend on history.

Historical prototype observation (not a packaged SC42 benchmark): a full hand
from the opening lead took about 0.4 s median in pure Python. The original note
does not specify hardware or sample count, so this is not a reproducible performance claim. Use it for: decision regret (best move value minus chosen move value),
double-dummy contract checks, and as the inner solver of PIMC (step 5).
Caveat: the oracle sees hidden hands, so it penalizes moves that are correct
under uncertainty. Report regret on the true deal and averaged over
belief-consistent deals separately.

### 4. Common agent interface

Ordinary policies implement `reset(seat)` and `act(Observation) -> int`.
The native environment, console adapter and arena share observations and legal
IDs; each seat has isolated memory. See [common agents](../README.md#common-agents)
for lifecycle and usage. A separate privileged protocol exists, but an oracle
runner is not implemented. Search and belief policies should use ordinary
observations unless explicitly evaluated as privileged teachers.

### 5. Baselines

Frozen Heuristic V1, conservative random bidding, and composable phase policies
are implemented. HHH/HHR/HRR/CHH/CRR/RRR factories support reproducible duplicate
hand and match experiments with independent per-phase RNG streams. See
[policy_ablation.md](policy_ablation.md) for commands, methodology and limits.
A PIMC trump/play baseline (`texas42.agents.pimc`, strategy `pimc` in agent-vs-agent
composition) is implemented; see [belief modelling](belief.md#the-pimc-agent). Stage-1
PPO provides a learned trick-play baseline.

- **Random:** choose uniformly from `legal_action_ids` (bids too).
- **Heuristic:** bid on trump length, doubles, and counters; call the longest
  suit; lead trumps, give counters to a winning partner, otherwise play low.
- **PIMC (Perfect Information Monte Carlo):** sample hidden hands consistent
  with what is known (hand sizes, revealed voids via `rules.follows_suit`), solve
  each sample with the oracle, and play the move with the best average. Known
  weaknesses are strategy fusion and non-locality (Frank & Basin 1998;
  Long et al. 2010).

### 6. Self-play training

The [legacy play-only PPO](ppo_training.md) remains a reproducible baseline.
[Multi-phase PPO](multiphase_selfplay.md) supports independent bidding/play
networks, frozen pretrained play, optional heuristic imitation, historical neural
opponents, exact snapshot-roster resume, and phase diagnostics. Trump routing and
optional training are available; effective trump learning is an empirical goal.

Next experiments:

- Compare multiple Stage A seeds against heuristic bidding with identical play.
- Evaluate joint historical training against HHH, CHH and the original anchor on
  independent held-out schedules; track playing-policy forgetting.
- Compare the local critic, the centralized privileged critic and its optional
  oracle auxiliary head under matched budgets
  ([Recipe 13](command_reference.md#recipe-13-local-versus-centralized-critic-and-oracle-auxiliary)).
- Study opponent-pool curricula and cross-play partners before introducing live
  self-play, recurrent policies or match-level PPO.
- Deep Monte Carlo remains an alternative algorithm to compare under equal budgets.
- Search inside training (Monte Carlo search as a policy-improvement signal) has a
  [theory, evidence and plan document](search_training.md): the theoretical case
  (policy improvement, performance difference, all-action variance reduction,
  regularized improvement), related methods, measured compute and an enumerated
  plan (P0–P6). Its first gate, move ranking, has been measured for the base,
  shared and paired guides: the current value leaf fails it, so training-time
  search must start with full rollouts or a new leaf (P2). Not implemented yet.

Short smoke tests establish mechanics, not competitive gains or generalization.

### 7. Hidden-information reasoning

Compare, at equal training budget:

| Variant | Description |
| --- | --- |
| A | Observation only |
| B | Implicit beliefs: auxiliary head predicting each unseen domino's owner |
| C | Explicit beliefs: separate belief network whose output feeds the policy |
| D | Privileged critic: value function sees all hands, policy does not |
| + | Beliefs from B/C replace uniform sampling inside PIMC |

Measure belief log-loss on true deals, arena score per training step, and
oracle regret. The encoder's `voids` slice is a hand-crafted inference feature;
zero it to ablate.

Implemented in `texas42/belief.py` ([belief modelling](belief.md)):

- Hard constraints (hand sizes, revealed voids) and **exact** world counts and
  per-domino marginals via a small DP. Exact uniform sampling, and full
  enumeration for small belief sets. Every world is replayed through the engine.
- Optional policy-likelihood weighting (Bayes over observed actions), using the
  new pure `ActionProbabilities` agent query, with ε-softening and an
  effective-sample-size diagnostic.
- Rollouts from index-chosen decision branches until a trick, phase or hand ends.
- Paired PIMC estimates of expected points per legal action, with a leaf hook
  for a future perfect-information oracle (kept separate from this module).
- Model-guided PIMC (`research/ppo/search.py`): a trained checkpoint can play the
  rollouts, weight beliefs by its move probabilities, and score truncated
  rollouts with its value head. `examples.search_accuracy` measures belief and
  value accuracy against true hands and outcomes
  ([model-guided PIMC](belief.md#model-guided-pimc)).

Observed so far: deterministic Heuristic likelihoods collapse the effective
sample size to about 1 by trick 3–4, so policy-weighted beliefs need softer
models or resampling. Next: duplicate-arena
comparisons against HHH; uniform vs policy belief log-loss on logged deals
(`TrainingCollector(include_initial_deal=True)`); then learned beliefs (B/C).

### 8. Partnership behavior

Train several seeds per method. Compare self-play (a seed with itself) against
cross-play (seed i partnered with seed j). A large gap indicates arbitrary
private conventions; Other-Play is a known mitigation.

Paired-partner training (`--paired-partners`) trains two separate networks for
the learning team. Two-spec team evaluation (`--agent-X-partner-*`) supports
pair, mirrored and cross-play comparisons; see
[paired partners](multiphase_selfplay.md#paired-partner-training). Next: compare
paired vs shared training at equal hands over several seeds, and cross-play each
partner with heuristic and other-seed partners.

### 9. Bidding

Bidding-only and joint PPO are implemented. First-voluntary BID/PASS replay
experiments estimate action value under fixed continuation policies; see
[counterfactual bidding](counterfactual_bidding.md). Broader opportunity sampling,
repeated stochastic continuations and match-value experiments remain future work.

Evaluate learned bidding using the [same-play phase ablation](phase_composable_evaluation.md).
Contract success alone does not show that bidding was preferable to passing.
Oracle/PIMC strength labels are a possible future initialization, with calibration
against real imperfect-information play because double-dummy outcomes overestimate it.

### 10. Special contracts

Action IDs 55–57 and `ContractKind` already reserve Nil, Splash, and Plunge.
Decide the house rules, implement them in `HandState`, add a new `RULESET_ID`,
then mark the action implemented. See [action-space.md](action-space.md).

### 11. Evaluation

The [arena](arena.md) owns duplicate schedules, hand/match metrics, contract
origin/timing diagnostics and group bootstrap definitions. New two-agent
experiments use [phase composition](phase_composable_evaluation.md); legacy
checkpoint and multi-opponent validation interfaces remain supported.

Next: multi-seed held-out comparisons, partnership cross-play matrices, and
explicit privileged-oracle evaluation/regret. Confidence intervals over deal
groups do not capture variation across training seeds.

Shared spawn workers and deterministic merging are implemented; see
[parallel execution](parallel_execution.md). [Evaluation performance](evaluation_performance.md)
preserves measured bootstrap and validation benchmarks. Centralized batched
inference, asynchronous learning and multi-host execution remain future work.

## References

Python and tooling:

- [dataclasses](https://docs.python.org/3/library/dataclasses.html),
  [enum.StrEnum](https://docs.python.org/3/library/enum.html#enum.StrEnum),
  [random](https://docs.python.org/3/library/random.html),
  [argparse nargs](https://docs.python.org/3/library/argparse.html#nargs),
  [contextlib.ExitStack](https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack),
  [unittest discovery](https://docs.python.org/3/library/unittest.html#test-discovery),
  [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/),
  [GitHub Actions matrices](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)
- NumPy [views vs copies](https://numpy.org/doc/stable/user/basics.copies.html),
  [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html),
  [flatnonzero](https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html)
- PyTorch [docs](https://pytorch.org/docs/stable/),
  [masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html),
  [Categorical](https://pytorch.org/docs/stable/distributions.html#categorical);
  [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/);
  [Numba](https://numba.readthedocs.io/);
  [scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

OpenSpiel (see also [openspiel.md](openspiel.md)):

- [Paper](https://arxiv.org/abs/1908.09453), [documentation](https://openspiel.readthedocs.io/),
  [install](https://openspiel.readthedocs.io/en/latest/install.html),
  [Windows notes](https://openspiel.readthedocs.io/en/latest/windows.html)
- Algorithms worth trying:
  [ismcts.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/ismcts.py),
  [nfsp.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/nfsp.py),
  [pytorch/ppo.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/ppo.py),
  [evaluate_bots.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/evaluate_bots.py)

Search:

- [Transposition tables](https://www.chessprogramming.org/Transposition_Table),
  [MTD(f)](https://www.chessprogramming.org/MTD(f)),
  [DDS bridge double-dummy solver](https://github.com/dds-bridge/dds)

Papers:

- DouZero (Deep Monte Carlo): https://arxiv.org/abs/2106.06135
- PerfectDou (perfect-information distillation): https://arxiv.org/abs/2203.16406
- Suphx (oracle guiding): https://arxiv.org/abs/2003.13590
- Asymmetric actor-critic: https://arxiv.org/abs/1710.06542
- PPO: https://arxiv.org/abs/1707.06347
- Invalid action masking: https://arxiv.org/abs/2006.14171
- Other-Play: https://arxiv.org/abs/2003.02979
- Learned Belief Search: https://arxiv.org/abs/2106.09086
- Long et al., "Understanding the Success of Perfect Information Monte Carlo
  Sampling in Game Tree Search" (AAAI 2010)
- Frank & Basin, "Search in Games with Incomplete Information: A Case Study
  Using Bridge Card Play" (Artificial Intelligence, 1998)
