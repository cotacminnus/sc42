# Experiment visualization

`examples.visualize_experiments` reads completed training runs, evaluation reports
and search-accuracy reports, and writes selectable figures, CSV tables and an
offline HTML report. It only reads recorded data: it never runs training or
evaluation, never changes an input, and needs no server, database or notebook.

- [Install](#install)
- [Quick start](#quick-start)
- [Input files](#input-files)
- [Selecting figures](#selecting-figures)
- [Figure families](#figure-families)
- [Research-question presets](#research-question-presets)
- [Configuration JSON](#configuration-json)
- [Outputs](#outputs)
- [Comparing runs](#comparing-runs)
- [Reading uncertainty](#reading-uncertainty)
- [Missing data](#missing-data)
- [Limitations of current instrumentation](#limitations-of-current-instrumentation)

## Install

```bash
python -m pip install -e '.[viz]'    # NumPy, pandas, Matplotlib, Plotly
```

Training does not need this extra. Plotly is only needed for `--formats html`.

## Quick start

```bash
# Compare training runs
python -m examples.visualize_experiments \
  --run baseline=runs/ppo-baseline \
  --run centralized=runs/ppo-centralized \
  --plots learning,ppo,critic \
  --formats html,png \
  --output-dir runs/visualizations/critic-study

# Every completed step of a pipeline
python -m examples.visualize_experiments \
  --pipeline runs/pipelines/critic-comparison-v1 \
  --plots learning,evaluation,opponents \
  --formats html,png,pdf \
  --output-dir runs/visualizations/critic-comparison

# One evaluation report
python -m examples.visualize_experiments \
  --evaluation runs/pipelines/study-round-robin-v1/shared-vs-paired/result.json \
  --plots evaluation,contracts \
  --formats png,svg \
  --output-dir runs/visualizations/partnership-study
```

Preview what can be drawn, and why anything cannot, without writing a file:

```bash
python -m examples.visualize_experiments --pipeline DIR --plots all --dry-run
python -m examples.visualize_experiments --list-plots
```

### After `configs/pipelines/critic_comparison.json` finishes

Run the pipeline with `python -m examples.run_pipeline --config configs/pipelines/critic_comparison.json`, then:

```bash
python -m examples.visualize_experiments --config configs/visualization/critic_comparison.json
```

Or choose runs yourself:

```bash
python -m examples.visualize_experiments \
  --run none=runs/pipelines/critic-comparison-v1/no-critic \
  --run local=runs/pipelines/critic-comparison-v1/local-critic \
  --run central=runs/pipelines/critic-comparison-v1/centralized-critic \
  --plots learning,critic,ppo,resources \
  --formats html,png,svg \
  --output-dir runs/visualizations/critic-ablation
```

## Input files

| Source | Option | Read from |
| --- | --- | --- |
| Training run | `--run NAME=DIR` | `DIR/progress.jsonl` (canonical), `DIR/config.json`, hashes of `DIR/checkpoints/**/*.pt` |
| Flattened fallback | `--run NAME=DIR` | `DIR/logs/training.csv`, only when `progress.jsonl` is absent; nested diagnostics and episode types are then unavailable |
| Pipeline | `--pipeline DIR` | `DIR/pipeline_state.json`: completed `training` and `evaluation` steps only |
| Arena report | `--evaluation [NAME=]PATH` | `evaluate_agent_vs_agent`, `evaluate_selfplay_ppo` or `evaluate_ppo` result JSON |
| Arena CSV | `--evaluation [NAME=]PATH` | a metrics CSV (with its per-hand sibling if present) or a per-hand CSV, for reports without JSON |
| Search accuracy | `--search-accuracy [NAME=]PATH` | `search_accuracy.py --output` JSON |

Both training-log generations load: flat legacy `train_ppo` records and nested
multi-phase `train_selfplay_ppo` records, including paired-partner records, whose
learner metrics are plotted per partner.

Records a later `--resume` replayed are dropped with a warning (they reach or pass
the hand count where training restarted). `runtime_seconds` restarts at each resume,
so the runtime axis adds the segments and warns.

Pipeline step lookup tries `DIR/<step id>` first and the manifest's recorded location
second, so a moved tree still works. Steps that are not completed are listed, not used.
With several `--pipeline` options, experiment names gain a `<directory>:` prefix.

Game records make `progress.jsonl` very large; they are skipped while parsing and
only their counts are kept. On the development machine, every figure in all four
formats for a 7.5 GB pipeline took about a minute.

## Selecting figures

| Argument | Purpose |
| --- | --- |
| `--run NAME=DIR` | Add a training run (repeatable) |
| `--pipeline DIR` | Add a pipeline's completed steps (repeatable) |
| `--evaluation [NAME=]PATH` | Add an arena report (repeatable) |
| `--search-accuracy [NAME=]PATH` | Add a search-accuracy report (repeatable) |
| `--plots LIST` | Comma-separated families, or `all`; a list beginning with `+` (for example `+bidding,contracts`) extends a preset or configured selection |
| `--rq rq1\|rq2\|rq3` | Research-question preset |
| `--metrics LIST` | Keep only figures whose id contains a term or matches a glob, for example `entropy,learning_*_vs_hhh` |
| `--formats LIST` | `html`, `png`, `svg`, `pdf` (default `png`) |
| `--output-dir DIR` | Destination; must be outside every input run or pipeline directory |
| `--x-axis VALUE` | `hands` (default), `matches`, `updates` or `runtime` |
| `--smooth N` | Optional rolling mean over N updates on update-level curves; labelled in the caption; off by default |
| `--aggregate-seeds` | Add a between-seed mean and SD for runs that differ only in seed |
| `--annotate VALUE` | Matrix cell text: `none`, `value` (default), `ci` or `n` |
| `--report` / `--no-report` | Write `index.html`; default is on when `html` is a selected format |
| `--config PATH` | Load a [configuration file](#configuration-json) |
| `--list-plots` | Show families and required data |
| `--dry-run` | Show planned figures and availability; write nothing |

Simulated `hands` is the primary sample-efficiency axis. `--x-axis matches` requires
every selected run to record matches, so hand-objective runs are rejected rather than
drawn on an invented axis.

## Figure families

| Family | Shows | Figure ids |
| --- | --- | --- |
| `learning` | Held-out hand or match win rate and signed marks per hand by evaluation opponent, by experience and by runtime; training reward; final comparison of methods | `learning_*` |
| `ppo` | Policy loss, entropy, approximate KL, clip fraction, gradient norm; samples, entropy and KL per phase | `ppo_*` |
| `critic` | Value loss, explained variance, predicted value versus return, advantage spread, oracle auxiliary loss and validation error | `critic_*` |
| `bidding` | Call distribution, bid categories, winning contracts, forced and voluntary calls, high bids, attempts versus auctions won, contract success, trends | `bidding_*` |
| `match-bidding` | Bid and high-bid rate by score state and exact score, objective comparison, hands per match, match win rate | `match_bidding_*` |
| `opponents` | Exposure share and counts, reward and validation by opponent, configured versus realized distribution, distinct identities | `opponents_*` |
| `evaluation` | Win rate, signed marks, contract success, declarer versus defender, by-opponent views, match length | `evaluation_*` |
| `contracts` | Frequency, success by category, forced versus voluntary, decision timing, roles | `contracts_*` |
| `round-robin` | Agent matrices: signed marks per hand, and match win rate | `round_robin_*` |
| `partners` | First-seat by second-seat compatibility matrices and the original-versus-cross gap | `partners_*` |
| `belief` | True-owner probability, log-loss, effective sample size, by tricks completed | `belief_*` |
| `search` | Value error, move agreement, ranking concordance and regret, cost, search inside training | `search_*` |
| `resources` | Throughput, time split, evaluation time, search worker versus wall time, peak memory | `resources_*` |

Rendered axes carry units and captions give the sample sizes. Each experiment keeps
one colour in every figure (a colour-vision-checked palette; dashes or hatches
distinguish further series).

## Research-question presets

| Preset | Question | Families |
| --- | --- | --- |
| `rq1` | Belief modeling, privileged-information training, PPO sample efficiency | `learning`, `ppo`, `critic`, `evaluation`, `resources` |
| `rq2` | Necessity and effectiveness of explicit hidden-information inference | `belief`, `search`, `contracts`, `evaluation`, `resources` |
| `rq3` | Partnership coordination, specialization, unfamiliar partners | `partners`, `round-robin`, `bidding`, `opponents`, `evaluation` |

A preset selects figure families. It does not claim that the data answers the
question; `--dry-run` and the report's omitted-figure lists show what is missing.

## Configuration JSON

```bash
python -m examples.visualize_experiments --config configs/visualization/critic_comparison.json
```

```json
{
  "schema_version": 1,
  "name": "critic-study-figures",
  "runs": {"none": "runs/pipelines/critic-comparison-v1/no-critic"},
  "plots": ["learning", "ppo"],
  "formats": ["html", "png"],
  "x_axis": "hands",
  "output_dir": "runs/visualizations/critic-study"
}
```

Keys: `schema_version` (must be 1), `name`, `description`, `runs`, `evaluations` and
`search_accuracy` (name-to-path objects or path lists), `pipelines` (a list), `plots`
(list or `"all"`), `rq`, `metrics`, `formats`, `x_axis`, `output_dir`, `smooth`,
`aggregate_seeds`, `annotate` and `report`. Unknown keys are errors. Paths are relative
to the directory the command runs in. The file is independent of pipeline configuration.

Examples are in `configs/visualization/`. Figures are reproducible: the same
configuration and inputs give identical SVG figures and tables.

**Override rules:**

1. Selection options on the command line replace the file's value: `--formats`,
   `--x-axis`, `--output-dir`, `--metrics`, `--smooth`, `--annotate`, `--report`,
   `--aggregate-seeds`.
2. Inputs add: `--run`, `--pipeline`, `--evaluation` and `--search-accuracy` append to
   the file's inputs. A repeated experiment name is an error.
3. Figure selection: `--plots` not beginning with `+` replaces everything. `--rq` on
   the command line replaces the file's `plots` and `rq`. A list beginning with `+`
   extends the preset or the file's selection. In the file, `plots` beats `rq`.

## Outputs

```text
runs/visualizations/critic-study/
├── index.html                       (when requested)
├── figures/
│   ├── learning_hand_win_rate_vs_hhh.html / .png / .pdf
│   ├── ppo_entropy.png
│   ├── critic_explained_variance.png
│   └── plotly.min.js                (with html, shared; no CDN needed)
├── tables/
│   ├── normalized_training.csv      (one row per PPO update; blank is unavailable, not zero)
│   ├── normalized_evaluation.csv    (one row per policy and metric, with intervals and units)
│   ├── experiment_comparison.csv    (settings side by side)
│   └── <round-robin or partners figure id>.csv
└── visualization_manifest.json      (settings, inputs, figures, omissions, warnings, files)
```

Only the selected formats and figures are written. Rerunning into a directory removes
files listed by the earlier manifest and leaves anything else alone. The report has
navigation by family, experiment and checkpoint identities, metric definitions,
captions, links to static files, omissions and warnings. Move the whole directory
together, since pages link to one another by relative path.

## Comparing runs

Any number of runs can be compared, and the tool warns when they differ in reward
objective, training budget, opponent distribution, trained phases, evaluation
opponents, evaluation seeds or schedule, or network architecture. A comparison still
proceeds, because it can still be meaningful, and the differences appear in the
console, report and `experiment_comparison.csv`.

Quantities that are never drawn together:

- Hand win rate and match win rate.
- Signed marks per hand from hand evaluations and from match evaluations.
- Training reward under different units, such as signed marks per hand and terminal
  reward per match. Comparing them would need a common unit that the logs do not record.
- Marks-per-hand and match-win-rate cells in a matrix.
- Critic targets under different reward objectives: `critic_*` figures get an objective
  suffix (such as `critic_value_loss_match_win`) when objectives are mixed.

Critic labels include the return estimator. Local-GAE versus centralized-GAE isolates
critic architecture more directly than local-GAE versus critic-free Monte Carlo.
`best.pt` and `latest.pt` stay distinguishable: in-training curves are labelled as the
current policy, and report checkpoints are named from their hash or pipeline reference.

## Reading uncertainty

- Arena intervals are 95% bootstrap intervals over **complete duplicate groups**. The
  tool plots the recorded intervals and never recomputes them from flattened hands.
- Hands inside one match are not independent match outcomes. Figures report duplicate
  groups as the independent unit.
- Evaluation uncertainty (one policy, finite deals) and between-seed variability are
  separate. `--aggregate-seeds` shows individual runs plus a mean and a ±1 SD band over
  seeds, aligned by evaluation ordinal because seeds reach evaluations at slightly
  different hand counts. Repeated evaluations of one run are never treated as seeds.
- Update-level curves are raw unless `--smooth` is set. Bid, opponent and match trends
  pool recorded counts over 1/20 of the updates, keeping denominators; each caption says so.
- Evaluations are not interpolated. A line breaks where an evaluation is missing.
- Training reward against a changing self-play population is not absolute strength.
- Contract success is conditional on the contracts chosen, so it does not prove bidding
  quality. Differences between runs are observational unless the design isolates a cause.
- Search regret is measured against a reference search and is approximate unless an
  exact oracle certifies it. Timings from different hardware are not directly comparable.

See [arena methodology](arena.md) for the evaluation design.

## Missing data

An unavailable figure is skipped with a reason, the required file or metric is named,
and the remaining figures continue. Omissions appear in the console, the report and the
manifest. Within a figure, runs lacking a series are listed under "series left out".

| Case | Result |
| --- | --- |
| Critic-free PPO | Value loss, explained variance and predicted value are absent, never zero |
| Hand-objective training | Leading and trailing states show n/a; exact-score heatmaps are omitted |
| Ordinary PPO training | Belief figures need a `search_accuracy.py` report |
| One checkpoint | Partner matrices name the `evaluate_agent_vs_agent` comparisons to run |
| Older logs | Bidding, opponent identity or phase figures are omitted or list the run as left out |
| Categories with no contracts | Shown as n/a, which differs from zero success |
| Untested agent pairs | Grey matrix cells, not zero |

## Limitations of current instrumentation

- `high_bid_available_rate` is logged without a denominator, so it cannot be pooled and
  is not plotted. Pooled high-bid rates use all bid decisions.
- Bid counts by score state are rebuilt from logged rates and rounded to whole decisions.
- Partner relations come only from recorded checkpoint paths and hashes. Other pairs are
  left unclassified. The original-versus-cross gap is a point difference between
  separately evaluated cells, so it has no interval.
- Agent identity uses file hashes plus recorded parameter comparisons; copies whose bytes
  differ are merged only when a report records that their parameters are identical.
- Ordinary PPO logs have no belief marginals, and `search_accuracy.py` records no
  hidden-information size, so neither is plotted. Belief figures measure belief accuracy,
  not information exchange between partners.
- A `self-current` opponent has no fixed identity; identity figures report what was logged.
- In-training figures use the evaluation seeds and sizes of each run, so check the
  comparison table before reading small differences. Convergence reports are not integrated.
