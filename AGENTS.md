# SC42 Repository Guidelines for Coding Agents

## Purpose and design priorities

SC42 is a reproducible research codebase for the Texas 42 domino game. Favor correctness, comparability of experiments, small public APIs, and ease of maintenance over feature count or architectural novelty.

**Extend and reuse existing code by default.** Before adding a function, method, class, module, script, or document, find the closest existing implementation and its callers/tests. Prefer, in order:

1. Use the existing API unchanged.
2. Modify or parameterize the existing implementation without obscuring its responsibility.
3. Extract a small shared helper when there is demonstrated duplication and the helper has a coherent purpose.
4. Add a new abstraction only when it clearly improves structure, manages necessary state, or separates genuinely different responsibilities.

Do not introduce parallel implementations for the same concept or speculative general-purpose frameworks. Do not add a wrapper solely to rename an API or move code. A new class is not automatically better than a function; prefer the simpler maintainable form. Keep relevant behavior near its established owner. When a new abstraction is justified, explain why existing ones were insufficient and identify the duplication it eliminates.

Consolidation is not deletion for its own sake: preserve useful specialization, reproducible historical experiments, and backwards compatibility. Avoid combining algorithms merely because their interfaces resemble each other.

## Repository ownership and canonical interfaces

- `texas42/`: dependency-light game rules, immutable observations, legal actions, built-in agents, self-play environment, arena, and optional OpenSpiel integration. Learning algorithms belong outside this package.
- `research/ppo/`: optional PyTorch models, rollout collection, training, checkpoint compatibility, frozen policies, evaluation adapters, and phase composition.
- `examples/`: runnable experiment/benchmark CLIs. Prefer extending an existing command rather than creating another near-duplicate entry point.
- `tests/`: deterministic `unittest` regression and integration tests.
- `docs/`: authoritative technical explanations and command documentation.
- `runs/` and `logs/`: generated artifacts; do not commit checkpoints, logs, or benchmark outputs.

For new two-agent experiments, prefer `examples.evaluate_agent_vs_agent` and `research/ppo/composition.py`; reuse `texas42.arena.compare_agents` for paired evaluation. Preserve older `evaluate_ppo` and `evaluate_selfplay_ppo` behavior or provide compatibility adapters, including multi-opponent evaluation and legacy result schemas. Training still has two deliberately distinct paths: `train_ppo`/`Trainer` is the reproducible play-only baseline; `train_selfplay_ppo`/`SelfPlayTrainer` supports independent phase networks and historical opponents. Do not declare them interchangeable without equivalence tests.

Use `research/ppo/checkpoint.py` as the checkpoint compatibility boundary, retaining version-1 and version-2 semantics. Prefer shared serialization/RNG utilities over duplicating these mechanisms, but do not rewrite checkpoint formats during general cleanup.

## Before making changes

1. Read relevant code, `AGENTS.md`, tests, and documentation. Search for existing implementations and call sites (for example, `rg 'symbol|related_term'`).
2. Identify the canonical owner, compatibility constraints, and whether the proposed change is behavior-preserving or behavior-changing.
3. Make the smallest cohesive change. Avoid opportunistic refactors and unrelated new features.
4. Update or add tests for observable behavior; compare seeded baseline outputs when consolidating existing mechanisms.
5. Update the existing authoritative document rather than adding a new overlapping document.

For larger refactors, give a short before/after map of modules and responsibilities, explain retained legacy interfaces, and report any intentional behavior changes separately. Do not claim full equivalence without verifying it.

## Experimental invariants and compatibility

- Preserve Texas 42 rules, authoritative legal-action masks, permanent action IDs `0–57` (reserved `55–57` remain illegal), the 387-feature observation encoder, schema-5 records, compatibility metadata, and ruleset identifiers unless explicitly authorized to version them.
- Agents only receive their own public immutable observations, not hidden opponent hands. Reuse the engine for transitions and legality; do not duplicate rule logic in agents or evaluators.
- Preserve per-seat, per-hand PPO trajectories and correct partnership signed-mark rewards. Do not mix trajectories across seats or hands.
- Keep legacy play-only PPO model/checkpoint loading and exact resume intact. Multi-phase checkpoints must accurately record neural, trained, learning, and frozen phases. Do not silently treat untrained bid/trump logits as learned policies.
- Keep distinct training/evaluation/random phase streams and stable seeded behavior. A cleanup must not silently change sampling, optimizer steps, action selection, checkpoint selection, or worker-count invariance.
- Preserve ABAB/BABA seating swaps on identical absolute-seat deals and shaker. Bootstrap confidence intervals must resample whole duplicate groups. Signed marks per hand are a primary performance measure, with contract and role diagnostics retained.
- Preserve pure-heuristic evaluation without a PyTorch dependency; keep OpenSpiel and RL dependencies optional in core workflows.
- Never conflate historical self-play snapshots, the frozen original play-only `anchor`, or dynamically composed policies.

## Documentation: one home per topic

- `README.md`: concise project overview, setup, and links to current workflows.
- `docs/command_reference.md`: canonical copy-ready CLI usage and argument reference.
- `docs/ppo_training.md`: legacy play-only PPO method, checkpoint continuity, and baseline reproduction.
- `docs/multiphase_selfplay.md`: current independent-phase training, historical opponents, snapshots, and resume.
- `docs/phase_composable_evaluation.md`: current agent-vs-agent composition and phase-ablation recipes.
- `docs/arena.md`: duplicate-deal methodology, metrics, uncertainty, and statistical interpretation.
- `docs/roadmap.md`: research status, remaining questions, and milestones.

Other focused research documents may remain. Prefer cross-links to copying commands, metric definitions, or implementation explanations into several places. Clearly label archived or legacy procedures rather than quietly deleting history. Keep examples synchronized with actual CLI flags. Any significant modification, addition, and deletion of the code must be reflected in the documents after the changes have been implemented.

## Implementation and testing

Use Python 3.11+, idiomatic type annotations, `snake_case` functions/modules, `PascalCase` classes, and focused immutable specifications where appropriate. Follow surrounding style. Keep performance-sensitive rollout code lean; avoid repeated model loading, unnecessary tensor copies, and excessive abstractions in inner loops.

From the repository root:

```bash
python -m pip install -e .
python -m pip install -e '.[rl,openspiel]'  # when optional dependencies are needed
python -m unittest discover -s tests -v
git diff --check
```

When changing evaluation, add deterministic tests for legacy and composable configurations, result schemas, checkpoint availability, matched duplicate groups, and 1-vs-N worker invariance. When changing training/checkpoint internals, add tiny seeded rollout/update/resume fixtures for both generations. Include small CLI smoke tests; do not substitute short tests for claims about policy strength. Benchmark only when relevant and report the actual hardware/workload and measurement limits.

## Change reports and Git hygiene

Work on a focused feature branch; do not merge automatically. Include in the completion report: reused and removed duplication, any new APIs and their justification, compatibility preserved, tests actually run, measured effects if applicable, and remaining limitations. Avoid unrelated formatting-only churn, generated files, fictitious metrics, and undocumented behavior changes.