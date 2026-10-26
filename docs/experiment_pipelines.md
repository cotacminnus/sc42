# Experiment pipelines

An experiment pipeline is one JSON file listing an ordered set of existing SC42
commands, for example *train play → train bidding → train trump → train jointly →
evaluate*. `examples.run_pipeline` runs the steps one after another, passes each
stage's checkpoints to later stages, records full provenance, and can resume an
interrupted experiment without repeating finished work.

The runner adds no training or evaluation logic. Every step is an ordinary run of
one of the commands below, so their flags, validation, logs and reports are exactly
the ones documented in the [command reference](command_reference.md).

## Contents

- [Quick start](#quick-start)
- [Writing a pipeline](#writing-a-pipeline)
- [Configuration reference](#configuration-reference)
- [Referencing earlier steps](#referencing-earlier-steps)
- [Output layout](#output-layout)
- [Running and dry runs](#running-and-dry-runs)
- [The manifest](#the-manifest)
- [Resuming and recovering](#resuming-and-recovering)
- [Common workflows](#common-workflows)
- [Experimental design notes](#experimental-design-notes)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Quick start

```bash
# 1. Check the plan: validates everything, prints the exact commands, runs nothing.
python -m examples.run_pipeline --config configs/pipelines/progressive_training.json --dry-run
# 2. Run it.
python -m examples.run_pipeline --config configs/pipelines/progressive_training.json
# 3. If it was interrupted or a step failed, fix the cause and continue.
python -m examples.run_pipeline --config configs/pipelines/progressive_training.json --resume
```

Run from the repository root. Training and neural evaluation steps need the `rl`
extra (`python -m pip install -e '.[rl]'`); a missing extra is reported before
anything starts.

| Example in `configs/pipelines/` | What it does |
| --- | --- |
| `template.json` | Annotated starting point: a scratch stage, a second stage that also trains against the first as a named opponent, and an evaluation. |
| `progressive_training.json` | Curriculum from random weights: play → bidding (play frozen) → trump (bid/play frozen) → joint with a centralized critic and historical snapshots → head-to-head evaluation against the pre-joint model. No pretrained checkpoint or anchor. |
| `scratch_selfplay.json` | All phases from random weights against `HHH:0.3,self-current:0.7`, a second stage adding historical snapshots, then fixed-opponent and head-to-head evaluations. |
| `critic_comparison.json` | Three independent from-scratch branches that differ only in `critic-mode` (`none`, `local`, `centralized`), each evaluated on the same held-out deals. |
| `study_comparisons.json` | Evaluation only. The one-seed study's Shared and Paired models each against HHH, HHR, HRR, CHH, CRR and RRR, then Paired against Shared (13 comparisons). Needs the study checkpoints in `runs/study-*`, or your own paths. |
| `study_round_robin.json` | Evaluation only. Every pair of Base, Shared, Paired, RRR, HHH and plain PIMC (15 comparisons). |

These examples use full-size budgets (hundreds of thousands of hands, 8 workers).
For a quick trial, copy one and shrink `total-hands`, `rollout-decisions`,
`eval-deals`, `deals` and `threads`.

## Writing a pipeline

Start by copying the template:

```bash
cp configs/pipelines/template.json configs/pipelines/my-experiment.json
```

**1. Name the experiment and choose where its results go.** `output_root` must be
new for every run of a changed configuration; relative paths resolve from the
repository root.

```json
{
  "schema_version": 1,
  "name": "my-experiment",
  "output_root": "runs/pipelines/my-experiment",
  "steps": []
}
```

**2. Translate each command line into a step.** Take the command you would type,
drop `python -m examples.`, put the command name in `command`, and move each flag
into `args` without its leading `--`:

```bash
python -m examples.train_selfplay_ppo --from-scratch --learner-play neural-train \
  --opponent-pool HHH:1 --total-hands 65536 --seed 42
```

```json
{
  "id": "play",
  "command": "train_selfplay_ppo",
  "args": {
    "from-scratch": true,
    "learner-play": "neural-train",
    "opponent-pool": "HHH:1",
    "total-hands": 65536,
    "seed": 42
  }
}
```

Leave out `output-dir`, `output` and `log-dir`; the runner assigns each step its own
directory ([output layout](#output-layout)).

**3. Connect stages with references.** A later step uses an earlier step's
checkpoint through `${STEP.best}` or `${STEP.latest}`:

```json
{
  "id": "bidding",
  "command": "train_selfplay_ppo",
  "args": {
    "init-checkpoint": "${play.best}",
    "learner-bid": "neural-train",
    "learner-bid-source": "new",
    "learner-play": "neural-frozen",
    "opponent-pool": "HHH:1",
    "total-hands": 65536,
    "seed": 42
  }
}
```

**4. Add evaluations.** Evaluation steps reference the checkpoints they compare:

```json
{
  "id": "evaluation",
  "command": "evaluate_selfplay_ppo",
  "args": {"checkpoint": "${bidding.best}", "opponents": "HHH,CHH,HHR", "deals": 2048, "seed": 12345}
}
```

**5. Dry-run, then run.** Fix anything the dry run reports; it checks every step's
arguments with that command's own parser.

## Configuration reference

### Top level

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | Yes | Must be `1`. |
| `name` | Yes | Experiment name, shown in progress and stored in the manifest. |
| `output_root` | Yes | Directory for all step outputs and `pipeline_state.json`. |
| `steps` | Yes | Nonempty list; list order is execution order. |
| `description` | No | Free text for humans. |

Any other field is rejected.

### Steps

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Unique name: letters, digits, `_` and `-`, starting with a letter or digit. Becomes the step directory and the name used in references. |
| `command` | Yes | One of the allowlisted commands below. |
| `args` | No | The command's flags, written without `--`. |
| `description` | No | Free text for humans. |

| `command` | Kind | Underlying CLI |
| --- | --- | --- |
| `train_selfplay_ppo` | training | [Multi-phase trainer](command_reference.md#multi-phase-trainer-train_selfplay_ppo) |
| `train_ppo` | training | [Legacy trainer](command_reference.md#legacy-trainer-train_ppo) |
| `evaluate_agent_vs_agent` | evaluation | [Agent-vs-agent](command_reference.md#agent-vs-agent-composition) |
| `evaluate_selfplay_ppo`, `evaluate_ppo` | evaluation | [Compatibility evaluation](command_reference.md#compatibility-evaluation) |
| `oracle_labels` | dataset | [Oracle training labels](command_reference.md#oracle-training-labels) |

No other program can be run: there is no shell, no arbitrary module and no expression
evaluation.

### Argument values

| JSON value | Becomes | Example |
| --- | --- | --- |
| string | `--flag VALUE`, unchanged | `"opponent-pool": "HHH:0.3,self-current:0.7"` → `--opponent-pool HHH:0.3,self-current:0.7` |
| number | `--flag VALUE`, written exactly as in the file | `"learning-rate": 0.0001` → `--learning-rate 0.0001` |
| `true` | bare flag | `"from-scratch": true` → `--from-scratch` |
| `false` | omitted | `"stochastic-neural": false` → nothing |
| list of strings/numbers | the flag repeated | `"opponent-checkpoint": ["a=x.pt", "b=y.pt"]` → `--opponent-checkpoint a=x.pt --opponent-checkpoint b=y.pt` |

`null`, objects, empty lists and lists containing booleans, lists or objects are
rejected, as are keys written with `--` or in another case. Values are never
rewritten: seeds, budgets and opponent weights reach the command exactly as written.
Whether a flag exists, takes a value or conflicts with another flag is decided by the
command's own argparse definition.

## Referencing earlier steps

| Reference | Resolves to | Allowed for |
| --- | --- | --- |
| `${STEP.best}` | `STEP/checkpoints/best.pt`: the checkpoint selected on fixed validation opponents | training steps |
| `${STEP.latest}` | `STEP/checkpoints/latest.pt`: the most recently saved state | training steps |
| `${STEP.output}` | The step's JSON output: the evaluation report or the label dataset | evaluation and `oracle_labels` steps |

References may appear inside a value, which is how earlier checkpoints become
named opponents:

```json
"opponent-checkpoint": ["stage1=${stage1.best}"],
"opponent-pool": "HHH:0.5,stage1:0.5"
```

Rules:

- The referenced step must come **earlier** in the list; forward and self references
  are rejected, so cycles are impossible.
- A checkpoint reference must name a training step, and `.output` must name an
  evaluation or dataset step.
- When the dependent step starts, the file must exist. A missing `best.pt` is an
  error even if `latest.pt` exists; the runner never substitutes one for the other.
- The absolute path and SHA256 of every referenced file are recorded, and checked
  again on resume.
- External files given directly by path are checked and hashed in the same way
  before their step runs. Examples: an existing `init-checkpoint` or
  `oracle-dataset`, or an `evaluate_agent_vs_agent` `agent-a-checkpoint`,
  `agent-a-partner-checkpoint` (and their per-phase forms) or
  `agent-a-pimc-model`.

## Output layout

Each step owns `OUTPUT_ROOT/<id>/`. The runner adds these arguments unless the
step sets them:

| Step kind | Added arguments |
| --- | --- |
| training | `--output-dir OUTPUT_ROOT/<id>`, `--log-dir OUTPUT_ROOT/<id>/logs` |
| evaluation | `--output OUTPUT_ROOT/<id>/result.json`, `--log-dir OUTPUT_ROOT/<id>/logs` |
| dataset | `--output OUTPUT_ROOT/<id>/labels.json`, `--log-dir OUTPUT_ROOT/<id>/logs` |

For `progressive_training.json` this gives:

```text
runs/pipelines/progressive-v1/
├── pipeline_state.json          manifest (see below)
├── play/
│   ├── checkpoints/best.pt, latest.pt
│   ├── config.json, progress.jsonl
│   └── logs/training.log, training.csv
├── bidding/   (same as play)
├── trump/     (same as play)
├── joint/     (same as play, plus snapshots/policy-*.pt)
└── evaluation/
    ├── result.json
    └── logs/result.log, result.csv, result-metrics.csv
```

Explicit `output-dir`, `output` or `log-dir` values are respected. Two steps may
not share or nest output locations, and nothing may overwrite the manifest or the
summary.

**Comparison summary.** When a pipeline with `evaluate_agent_vs_agent` steps
completes, the runner writes `OUTPUT_ROOT/summary.md` and prints it.

- One table row per comparison, in step order: the step id, its `description`,
  agent A's signed marks per hand and win rate with their 95% intervals, and the
  games played.
- A resumed run that completes rewrites it.
- Other evaluation commands keep their own reports and are not summarized.

## Running and dry runs

| Flag | Effect |
| --- | --- |
| `--config PATH` | The pipeline file (required). |
| `--dry-run` | Validate the file, references, output locations and every step's arguments; print each exact command, its outputs, dependencies and external inputs. Creates no files and runs nothing; checkpoints of earlier steps appear as predicted paths. |
| `--resume` | Continue a previous run of the same configuration. |

Every step is first validated, then the steps run one at a time as separate
processes. Each command prints its normal output, between progress lines:

```text
Pipeline: progressive-v1

[1/5] play       TRAINING
...normal train_selfplay_ppo output...
[1/5] play       COMPLETED
[2/5] bidding    TRAINING
...
[5/5] evaluation COMPLETED

Pipeline completed successfully.
```

A step is completed only when its command exits with status 0 **and** its expected
outputs exist (`latest.pt` and `progress.jsonl` for training, the JSON output
otherwise). On the first failure the pipeline stops, nothing later runs, and
`run_pipeline` exits with the failing step's status. Configuration and validation
errors exit with status 2 before anything runs. A new (non-resume) run refuses an
existing `pipeline_state.json` or non-empty step outputs, so earlier results are
never overwritten.

## The manifest

`OUTPUT_ROOT/pipeline_state.json` is rewritten atomically after every state change.
It contains:

- the verbatim configuration, its path and SHA256;
- the git commit and whether the working tree had uncommitted changes, and the
  Python executable;
- for each step: its state (`pending`, `running`, `completed` or `failed`), the
  exact command, owned locations, produced outputs with SHA256, dependencies
  (references) and external inputs with paths and SHA256;
- for each attempt: start and finish times, exit status, mode (`initial`, `resume`
  or `retry`), the exact command used and, for resumed training, what was reconciled;
- the time and git commit of every `--resume`.

To reproduce an experiment, check out the recorded commit and run the configuration
stored in the manifest with a new `output_root`.

## Resuming and recovering

`--resume` needs the **identical** configuration: any change, even to a description,
is rejected. For a changed experiment use a new `output_root`. It then walks the
steps in order:

| Situation | What `--resume` does |
| --- | --- |
| Step completed | Skipped after re-checking its outputs and the dependencies it used. A missing or modified file stops the pipeline. |
| Training step interrupted or failed, with `latest.pt` | Continued exactly from `latest.pt` in the same directory, using that command's own resume (below). |
| Training step with partial outputs but no `latest.pt` | Stops and explains; remove that step's directory to restart it from its configured initialization. Training is never restarted silently. |
| Training step that never produced output | Started normally. |
| Evaluation or labeling step failed | Rerun completely (labeling reuses its own label cache). |
| Step not yet reached | Run normally. |

The two trainers resume differently, and the runner follows each:

- **`train_selfplay_ppo`** restores its configuration from the checkpoint, so it is
  called with `--resume latest.pt` plus only `total-hands`, `threads`, `eval-threads`,
  `device` and the output locations.
- **`train_ppo`** requires the saved configuration to be repeated, so it is called
  with the step's original arguments and `--resume latest.pt` in place of any
  `--warm-start`.

A run can stop after it wrote progress or snapshots newer than its last
`latest.pt`. Before resuming, those `progress.jsonl` rows and `snapshots/policy-*.pt`
files are moved to `*.pre-resume-<time>` backups; the resumed run recreates them
deterministically from the checkpoint, so history is never duplicated. The manifest
records the checkpoint's hand count and what was moved. In tests, an interrupted and
resumed stage produced weights identical to an uninterrupted run.

If the code changed since the pipeline started, `--resume` prints a warning with
both commits and continues.

## Common workflows

**Start from random weights.** Give the first training step `"from-scratch": true`.
No checkpoint or play-only anchor is needed, and later stages of that lineage stay
anchor-free; see [from-scratch training](command_reference.md#recipe-14-from-scratch-training-and-self-play).

**Progressive curriculum.** Each stage uses `init-checkpoint: "${previous.best}"`
and changes which phases are `neural-train`, `neural-frozen` or `heuristic`. Use
`learner-PHASE-source: "new"` for a phase that has never been trained, so it starts
with fresh heads on the trained trunk. See `progressive_training.json`.

**Self-play.** Use `self-current` (a frozen copy of the current learner, refreshed
each rollout) and/or `self-snapshots` (published earlier learners) in
`opponent-pool`, for example `"HHH:0.3,self-current:0.7"`. See `scratch_selfplay.json`.

**Train against an earlier stage.** Register it with
`"opponent-checkpoint": ["stage1=${stage1.best}"]` and use `stage1` in
`opponent-pool` or `eval-opponents`. See `template.json`.

**Choose the critic.** `"critic-mode"` takes `"none"` (policy-only PPO with Monte
Carlo returns and no value loss), `"local"` (the default; omit the key to keep it)
or `"centralized"` (`train_selfplay_ppo` only). Omit `"return-estimator"` to use
each mode's default; `none` with `"return-estimator": "gae"` fails when the step
starts. A pipeline step resumed after an interruption keeps its mode. For example:

```json
{
  "id": "policy-only",
  "command": "train_selfplay_ppo",
  "args": {
    "from-scratch": true,
    "critic-mode": "none",
    "opponent-pool": "HHH:0.3,self-current:0.7",
    "eval-opponents": "HHH,CHH",
    "selection-opponent": "HHH",
    "total-hands": 262144,
    "rollout-decisions": 4096,
    "threads": 8,
    "seed": 42
  }
}
```

**Compare critic modes.** Give each mode its own training step with identical
arguments and seed and no reference to another treatment, so the branches are
independent and start from identical actor weights; then evaluate each with the
same opponents, deals and seed. See `critic_comparison.json` and
[policy-only PPO](command_reference.md#recipe-15-policy-only-ppo-and-a-three-way-critic-comparison).

**Centralized critic with oracle supervision.** Add an `oracle_labels` step, then
`"critic-mode": "centralized"`, `"oracle-aux-coef": 0.5` and
`"oracle-dataset": "${labels.output}"` in the training step.

**Legacy baseline.** A `train_ppo` step produces a version-1 play checkpoint; a
later `train_selfplay_ppo` step can use `"init-play-checkpoint": "${legacy.best}"`,
and `evaluate_ppo` can evaluate it.

**Compare models.** End with `evaluate_agent_vs_agent` steps that reference the
checkpoints to compare, and `evaluate_selfplay_ppo` steps against fixed opponents.
Use evaluation seeds different from training.

**Evaluation-only comparison sets.** A pipeline needs no training step. Each step
can name existing checkpoints by path, and `summary.md` collects the results. For
example, one model against each fixed variant:

```json
{
  "id": "paired-vs-chh",
  "command": "evaluate_agent_vs_agent",
  "description": "Paired pair vs CHH",
  "args": {
    "agent-a-bid": "neural", "agent-a-play": "neural",
    "agent-a-checkpoint": "runs/study-paired/checkpoints/partner-1/best.pt",
    "agent-a-partner-checkpoint": "runs/study-paired/checkpoints/partner-2/best.pt",
    "agent-b-bid": "conservative",
    "deals": 2048, "seed": 20261019, "bootstrap-samples": 2000, "threads": 0
  }
}
```

`study_comparisons.json` (13 steps) and `study_round_robin.json` (15 steps) are
complete examples. The flags for every agent and opponent are in
[model comparisons](command_reference.md#model-comparisons-and-round-robins).

## Experimental design notes

- **New stage versus exact resume.** `init-checkpoint` and `init-play-checkpoint`
  start a new training stage: weights are copied, but the optimizer, counters, RNG
  streams and snapshot roster start fresh. Only `--resume` continues a stage exactly.
- **Curriculum versus ablation.** A curriculum feeds each stage into the next, so its
  stages are not independent comparisons. For an ablation, start every variant from
  the same checkpoint with matched seeds and budgets, then compare them with
  [controlled evaluation](command_reference.md#recipe-12-held-out-phase-ablations).
- **`best.pt` versus `latest.pt`.** `best.pt` is chosen on the small fixed validation
  schedule against `selection-opponent`; `latest.pt` is simply the newest state.
  Reference the one you mean, and evaluate final models on independent held-out seeds.
- **Training versus validation opponents.** `opponent-pool` decides who the learner
  trains against. `eval-opponents` and `selection-opponent` must be fixed policies;
  `self-current` and `self-snapshots` are rejected there because they change during
  training.

## Troubleshooting

| Message | Cause and fix |
| --- | --- |
| `Unsupported schema_version` / `Unknown pipeline fields` / `Step N: unknown fields` | A misspelled or extra field; only the documented fields are accepted. |
| `unsupported command` | `command` must be one of the six allowlisted names, without `examples.`. |
| `argument ... must be written without "--"` | Write `"total-hands"`, not `"--total-hands"`. |
| `has unsupported value type` / `is an empty list` | `null`, objects and empty lists are not arguments; omit the key instead. |
| `rejects its arguments: ...` | The command's own parser refused the step: unknown flag, wrong type, invalid choice or conflicting flags. The rest of the message is the command's error. |
| `must name an earlier step` / `names no step` / `malformed reference` | Fix the `${STEP.best}` / `${STEP.latest}` / `${STEP.output}` reference or the step order. |
| `does not exist at ...; only latest.pt exists` | The referenced stage saved no `best.pt`; reference `${STEP.latest}` if that is what you want. |
| `Output collision` | Two steps share or nest an output location; remove the explicit paths or make them distinct. |
| `pipeline_state.json exists; use --resume` / `already exists` | The output root holds an earlier run; resume it or choose a new `output_root`. |
| `The pipeline configuration changed since this run started` | Resume requires the identical file; restore it or use a new `output_root`. |
| `output was modified` / `output is missing` / `dependency ... changed after use` | A recorded file changed after it was produced or used; restore it, or rerun into a new `output_root`. |
| `has partial outputs but no latest.pt` | The step stopped before its first checkpoint; inspect, then remove its directory to restart that step. |
| `needs an optional dependency that is not installed` | Install the `rl` extra. |

## Limitations

- Steps run strictly one after another; there is no parallel execution of
  independent steps.
- `--resume` accepts no configuration changes at all.
- Validation imports each command's module to use its parser, which loads PyTorch
  for training and neural evaluation steps (no networks are built).
- If `best.pt` was saved after the last `latest.pt`, it is left in place on resume
  and may be overwritten as training replays deterministically.
