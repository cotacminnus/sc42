# Repository Guidelines

## Project Structure & Module Organization

`texas42/` contains the dependency-free engine, rules, immutable observations,
agents, self-play environment, evaluation arena, and optional OpenSpiel adapter.
`research/ppo/` contains optional PyTorch training, rollout, evaluation, and
checkpoint code. Keep learning algorithms outside the core engine. Executable
examples live in `examples/`, standard-library tests in `tests/`, and design and
command documentation in `docs/`. Generated artifacts belong in ignored `runs/`
and `logs/` directories.

## Build, Test, and Development Commands

Run commands from the repository root with Python 3.11 or newer:

- `python -m pip install -e .`: install the package and `tx42` CLI.
- `python -m pip install -e '.[rl,openspiel]'`: install optional PPO and OpenSpiel dependencies.
- `python -m texas42 --all-ai --seed 42 --target-marks 1`: run a reproducible console match.
- `python -m unittest discover -s tests -v`: run the complete available suite.
- `python -m examples.policy_ablation_benchmark --policy-a HHH --policy-b CHH --deals 32 --seed 42`: run a small duplicate-deal benchmark.

See `docs/command_reference.md` for training and evaluation commands.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations, `snake_case` functions/modules,
`PascalCase` classes, and uppercase constants. Prefer small pure helpers,
composition, protocols, and immutable value objects. No formatter or linter is
configured; follow neighboring code and run `git diff --check` before submission.

## Testing Guidelines

Use `unittest` and name modules `tests/test_<area>.py`. Add deterministic regression
tests for changed behavior, explicit numerical fixtures for statistical/PPO
calculations, and seeded integration tests. Optional dependency tests may skip
locally; verify them with the relevant extras installed. CI covers dependency-free,
OpenSpiel, and CPU PPO configurations. No numerical coverage threshold is configured.
Avoid timing assertions and unsupported claims of agent strength.

## Compatibility & Experiment Safety

Preserve permanent action IDs 0–57, illegal reserved IDs 55–57, the 387-feature
encoder, schema-5 records, and compatibility metadata. Reuse engine transitions
and legality instead of duplicating rules. Ordinary agents receive immutable
observations without hidden opponent hands. Keep deterministic seed streams,
duplicate-group bootstrap sampling, and per-seat PPO trajectories intact.
Write readable experiment logs and CSV exports under `logs/experiments/`.

## Commit & Pull Request Guidelines

Existing commits use short descriptions such as “optimized bootstrap process”;
no formal prefix convention is enforced. Use focused feature branches and do not
merge automatically. PRs should explain behavior, compatibility implications,
tests and skips, and measured benchmark results when relevant. Link related
issues and update affected documentation. Keep checkpoints and generated logs
out of commits.
