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
| 3 | Perfect-information oracle | Prototype only | planned `texas42/oracle.py` |
| 4 | Common agent interface | Done | `agents/`: protocols, RandomAgent, AgentPlayer |
| 5 | Baseline agents | Random only | `agents.RandomAgent` (all legal bids); legacy `players.RandomPlayer` |
| 6 | Self-play training | Not started | planned `research/` (PyTorch) |
| 7 | Hidden-information reasoning | Not started | planned `texas42/belief.py` |
| 8 | Partnership behavior | Not started | uses the arena (step 11) |
| 9 | Bidding | Not started | |
| 10 | Special contracts (Nello) | IDs reserved | `actions.py` IDs 55–57 |
| 11 | Evaluation | Done (ordinary agents) | `texas42/arena.py`, [arena.md](arena.md) |

Recommended build order puts the evaluation tools before training, so every
learned agent is measured on the same harness:

```text
4 agent interface -> 3 oracle -> 5 baselines (heuristic, PIMC) -> 11 arena
2 encoder ---------------------> 6 DMC / PPO  <--------------------- 11 arena
6 -> 7 beliefs -> 8 cross-play partners -> 9 bidding -> 10 Nello
```

## Known gaps in the current code

- `RandomPlayer` never bids, so `--all-ai` CLI logs are always a forced 30 by
  the shaker. Use `SelfPlayEnv` with random legal IDs for bidding data.
- Ordinary CLI logs contain only actor-visible observations and public history.
  `TrainingCollector(include_initial_deal=True)` can separately record complete
  initial deals for belief/oracle labels. Label generation and models remain
  future work; ordinary policies receive only `Observation`.
- OpenSpiel's `deal_counts` observer field is always zero (see the TODO in
  `openspiel.py`).
- `pyspiel` wheels target Linux and macOS. On Windows, use WSL2; OpenSpiel tests
  skip locally and run in CI.

## Step-by-step plan

### 2. Self-play setup: feature encoder

`texas42/encoding.py` turns an `Observation` into a fixed-length list of floats,
from the acting seat's point of view. See [encoding.md](encoding.md). Remaining
work: parallel rollout collection (`multiprocessing`). Schema-5 JSONL loading
now restores typed policy observations, and a separately versioned public
history preserves action chronology without changing the 387-dimensional
encoder. Opt-in `TrainingCollector` labels preserve the full initial deal
outside observations, including across hand boundaries in match mode. Tests
cover round trips, rotated-seat invariance, hidden-hand privacy, chronological
replay and incompatible versions. See [encoding.md](encoding.md) for APIs and
JSONL collection examples. `validation.validate_episode()` checks complete schema-5
episodes and distinguishes privileged verified replay from partial consistency
validation without independently recorded deals. Collector failures abort the
rollout until reset. No recurrent models or neural training are implemented.

### 3. Oracle (perfect-information solver)

Given all four hands, trump, and the leader, return the points each team can
still take under perfect play. Seats 0/2 maximize team-0 points; seats 1/3
minimize them. A tested prototype uses:

- 28-bit masks for hands and per-trump lookup tables built from `rules.py`;
- alpha-beta search that continues across trick boundaries;
- a transposition table keyed by `(hands, leader)` storing lower/upper bounds.
  At a trick boundary, the remaining value does not depend on history.

It solves a full hand from the opening lead in about 0.4 s (median) in pure
Python. Use it for: decision regret (best move value minus chosen move value),
double-dummy contract checks, and as the inner solver of PIMC (step 5).
Caveat: the oracle sees hidden hands, so it penalizes moves that are correct
under uncertainty. Report regret on the true deal and averaged over
belief-consistent deals separately.

### 4. Common agent interface

Implemented in `texas42/agents/`: ordinary `Agent.reset(seat)` and
`Agent.act(Observation) -> int`, uniform `RandomAgent`, and `AgentPlayer` for
`engine.Game`. `Player` retains its existing object-based decisions and gains
a no-op episode reset hook. Shared pure legality and observation construction
keep adapter inputs identical to native self-play inputs. The CLI's existing
human/random player defaults remain unchanged.

Use four distinct agent instances, allowing shared weights but isolated private
memory. Game resets once per match or standalone hand; native self-play runners
reset agents once per `env.reset()`, never per hand within a match. RandomAgent
continues its seeded RNG stream across resets. See the [README](../README.md#common-agents)
and run `python -m examples.agent_selfplay` for a complete hand.

Custom heuristic, neural, PIMC and belief policies implement the ordinary
protocol without depending on mutable engine state. Neural imports and training
remain outside the core interface. A separate `PrivilegedAgent` protocol accepts
an immutable `PrivilegedState` containing remaining hands; ordinary adapters
never provide it. The oracle and its runner remain future work.

The arena now accepts four per-seat factories, resets at episode boundaries,
and dispatches only acting-seat observations. Predetermined duplicate schedules
and paired bootstrap intervals support reproducible comparisons. Privileged
oracle execution remains a separate future path; see [arena.md](arena.md).

### 5. Baselines

- **Random:** choose uniformly from `legal_action_ids` (bids too).
- **Heuristic:** bid on trump length, doubles, and counters; call the longest
  suit; lead trumps, give counters to a winning partner, otherwise play low.
- **PIMC (Perfect Information Monte Carlo):** sample hidden hands consistent
  with what is known (hand sizes, revealed voids via `rules.follows_suit`), solve
  each sample with the oracle, and play the move with the best average. Known
  weaknesses are strategy fusion and non-locality (Frank & Basin 1998;
  Long et al. 2010).

### 6. Self-play training

- **Deep Monte Carlo** (DouZero): learn Q(state, action) from Monte Carlo hand
  returns, with ε-greedy exploration over legal IDs. Simple and stable; do it first.
- **PPO / actor-critic:** phase heads from `ACTION_REGISTRY.phase_slots()`
  (bid 21, trump 9, play 28) or one 58-way head. Mask illegal logits before sampling.
- Train against a pool of past checkpoints, and save `training_metadata()` with
  every checkpoint.

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

### 8. Partnership behavior

Train several seeds per method. Compare self-play (a seed with itself) against
cross-play (seed i partnered with seed j). A large gap indicates arbitrary
private conventions; Other-Play is a known mitigation.

### 9. Bidding

First regress hand strength (probability of reaching each bid) from oracle or
PIMC outcomes over sampled partner/opponent hands; fix that bidder while
training play; then fine-tune jointly with `episode="match"`. Double-dummy
results overestimate the bidder, so calibrate against real play.

### 10. Special contracts

Action IDs 55–57 and `ContractKind` already reserve Nil, Splash, and Plunge.
Decide the house rules, implement them in `HandState`, add a new `RULESET_ID`,
then mark the action implemented. See [action-space.md](action-space.md).

### 11. Evaluation

Implemented: dependency-free hand/match execution, deterministic complete deal
schedules, duplicate partnerships, explicit seat permutations, policy-perspective
metrics, and group-resampled 95% bootstrap intervals. Matches retain agent memory
and use official shaker rotation. Run `python -m examples.arena_benchmark`; see
[arena.md](arena.md) for metric definitions, seed coupling, exports, and limits.
Next: heuristic baselines, held-out schedules, learned-checkpoint evaluation and
cross-play matrices. Oracle execution/regret, model training, and advanced
statistical testing remain future work.

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
