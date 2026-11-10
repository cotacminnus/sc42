"""Figures and an offline HTML report from completed SC42 experiments (docs/visualization.md).

Read-only with respect to every input: it parses recorded logs and reports and
writes only under --output-dir. It never trains or evaluates anything.

    python -m examples.visualize_experiments --run baseline=runs/ppo-baseline --plots learning,ppo --formats png --output-dir runs/visualizations/x
    python -m examples.visualize_experiments --pipeline runs/pipelines/critic-comparison-v1 --rq rq1 --dry-run --output-dir runs/visualizations/x
    python -m examples.visualize_experiments --config configs/visualization/critic_comparison.json
    python -m examples.visualize_experiments --list-plots

Option precedence: a selection option on the command line replaces the value from
--config. Inputs (--run, --pipeline, --evaluation, --search-accuracy) add to those
in the file, and a repeated experiment name is an error. --plots replaces the
preset or file selection; a list beginning with + extends it instead.
"""
import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import sys

from research.analysis import FAMILIES, FORMATS, RESEARCH_PRESETS, RESEARCH_QUESTIONS, X_AXES

CONFIG_KEYS = {'schema_version', 'name', 'description', 'runs', 'pipelines', 'evaluations', 'search_accuracy', 'plots', 'rq', 'metrics', 'formats',
               'x_axis', 'output_dir', 'smooth', 'aggregate_seeds', 'report', 'annotate'}
ANNOTATIONS = ('none', 'value', 'ci', 'n')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', default=[], metavar='NAME=DIR', help='add a training experiment (repeatable)')
    parser.add_argument('--pipeline', action='append', default=[], metavar='DIR', help='add completed steps from a pipeline directory (repeatable)')
    parser.add_argument('--evaluation', action='append', default=[], metavar='[NAME=]PATH', help='add an arena result JSON, or metrics/per-hand CSV (repeatable)')
    parser.add_argument('--search-accuracy', action='append', default=[], metavar='[NAME=]PATH', help='add a search_accuracy.py JSON report (repeatable)')
    parser.add_argument('--plots', metavar='LIST', help="comma-separated families or 'all'; a list beginning with + extends the preset or configured selection")
    parser.add_argument('--rq', choices=sorted(RESEARCH_PRESETS), help='research-question preset of figure families')
    parser.add_argument('--metrics', metavar='LIST', help='only figures whose id contains one of these terms or matches one of these globs')
    parser.add_argument('--formats', metavar='LIST', help=f'comma-separated: {", ".join(FORMATS)} (default: png)')
    parser.add_argument('--output-dir', type=Path, help='visualization destination')
    parser.add_argument('--x-axis', choices=X_AXES, help='x-axis for training curves (default: hands)')
    parser.add_argument('--smooth', type=int, metavar='N', help='rolling mean over N updates on update-level curves, labelled in captions (default: off)')
    parser.add_argument('--aggregate-seeds', action=argparse.BooleanOptionalAction, default=None, help='add between-seed mean and SD for runs differing only in seed')
    parser.add_argument('--annotate', choices=ANNOTATIONS, help='matrix cell text (default: value)')
    parser.add_argument('--report', action=argparse.BooleanOptionalAction, default=None, help='write index.html (default: when html is a selected format)')
    parser.add_argument('--config', type=Path, help='reusable visualization configuration JSON')
    parser.add_argument('--list-plots', action='store_true', help='show available plots and required data, then exit')
    parser.add_argument('--dry-run', action='store_true', help='show planned plots and data availability without writing files')
    return parser


def list_plots() -> None:
    print('Figure families (select with --plots LIST or --plots all):\n')
    for name, (description, required) in FAMILIES.items():
        print(f'{name}\n  {description}\n  Requires: {required}')
    print('\nResearch-question presets (--rq): each selects figure families; it does not claim the data answers the question.')
    for name, families in RESEARCH_PRESETS.items():
        print(f'{name}: {", ".join(families)}\n  {RESEARCH_QUESTIONS[name]}')


def _read_config(path: Path, parser) -> dict:
    if not path.is_file():
        parser.error(f'Configuration file {path} does not exist')
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except ValueError as exc:
        parser.error(f'{path} is not valid JSON: {exc}')
    if not isinstance(config, dict):
        parser.error(f'{path} must hold a JSON object')
    if config.get('schema_version') != 1:
        parser.error(f'{path}: schema_version must be 1')
    if unknown := sorted(set(config) - CONFIG_KEYS):
        parser.error(f'{path}: unknown configuration keys: {", ".join(unknown)}')
    return config


def _split(values) -> list[str]:
    if isinstance(values, str):
        values = values.split(',')
    return [value.strip() for value in values if value.strip()]


def _named(value: str, default_name) -> tuple[str, Path]:
    left, separator, right = value.partition('=')
    if separator and left and not ({'/', '\\'} & set(left)):
        return left, Path(right)
    path = Path(value)
    return default_name(path), path


def _report_name(path: Path) -> str:
    return path.parent.name if path.stem == 'result' else path.stem


def _config_inputs(value) -> list[str]:
    """Config inputs are a NAME-to-path object, or a list of paths."""
    if isinstance(value, dict):
        return [f'{name}={path}' for name, path in value.items()]
    return list(value or [])


def resolve_families(args, config, parser) -> list[str]:
    """--plots beats --rq beats configuration; a list beginning with + extends whichever base applies."""
    rq = args.rq or config.get('rq')
    if rq is not None and rq not in RESEARCH_PRESETS:
        parser.error(f'Unknown research-question preset {rq!r}; choose from {", ".join(sorted(RESEARCH_PRESETS))}')
    base = list(RESEARCH_PRESETS[rq]) if rq and (args.rq or 'plots' not in config) else None
    if base is None and 'plots' in config:
        base = ['all'] if config['plots'] == 'all' else _split(config['plots'])
    requested = _split(args.plots) if args.plots else None
    if requested:
        names = [item.removeprefix('+') for item in requested]
        selection = (base or []) + names if requested[0].startswith('+') else names
    else:
        selection = base
    if not selection:
        parser.error('Select figures with --plots, --rq, or `plots` in --config (see --list-plots)')
    if 'all' in selection:
        return list(FAMILIES)
    unknown = sorted(set(selection) - set(FAMILIES))
    if unknown:
        parser.error(f'Unknown plot selection: {", ".join(unknown)}. Available: {", ".join(FAMILIES)}, all')
    return list(dict.fromkeys(selection))


def resolve_formats(args, config, parser) -> list[str]:
    requested = _split(args.formats) if args.formats else _split(config.get('formats') or ['png'])
    unknown = sorted(set(requested) - set(FORMATS))
    if unknown:
        parser.error(f'Unsupported format: {", ".join(unknown)}. Choose from {", ".join(FORMATS)}')
    if not requested:
        parser.error('Select at least one format')
    return list(dict.fromkeys(requested))


def _check_dependencies(formats, parser, dry_run) -> None:
    needed = ['numpy', 'pandas', 'matplotlib'] + (['plotly'] if 'html' in formats and not dry_run else [])
    if missing := [name for name in needed if find_spec(name) is None]:
        parser.error(f'Missing optional visualization dependencies: {", ".join(missing)}. Install them with: python -m pip install -e ".[viz]"')


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_plots:
        list_plots()
        return 0
    config = _read_config(args.config, parser) if args.config else {}
    families = resolve_families(args, config, parser)
    formats = resolve_formats(args, config, parser)
    x_axis = args.x_axis or config.get('x_axis', 'hands')
    if x_axis not in X_AXES:
        parser.error(f'Unsupported x-axis {x_axis!r}; choose from {", ".join(X_AXES)}')
    annotate = args.annotate or config.get('annotate', 'value')
    if annotate not in ANNOTATIONS:
        parser.error(f'Unsupported annotation {annotate!r}; choose from {", ".join(ANNOTATIONS)}')
    smooth = args.smooth if args.smooth is not None else config.get('smooth', 0)
    if not isinstance(smooth, int) or smooth < 0:
        parser.error('--smooth must be a nonnegative integer')
    output_dir = args.output_dir or (Path(config['output_dir']) if config.get('output_dir') else None)
    if output_dir is None and not args.dry_run:
        parser.error('--output-dir (or `output_dir` in --config) is required')
    _check_dependencies(formats, parser, args.dry_run)

    from research.analysis.loader import InputError, discover_pipeline, load_evaluation, load_search_accuracy, load_training
    from research.analysis.metrics import compare_experiments, X_COLUMNS
    from research.analysis.normalize import build_dataset
    from research.analysis.plots import Options, build_figures
    from research.analysis.report import write_outputs

    runs = _config_inputs(config.get('runs')) + args.run
    pipelines = [Path(p) for p in list(config.get('pipelines') or []) + args.pipeline]
    evaluations = _config_inputs(config.get('evaluations')) + args.evaluation
    searches = _config_inputs(config.get('search_accuracy')) + args.search_accuracy
    if not (runs or pipelines or evaluations or searches):
        parser.error('No inputs: give at least one of --run, --pipeline, --evaluation, --search-accuracy')
    training, reports, notes, input_dirs = [], [], [], []
    try:
        for item in runs:
            name, separator, path = item.partition('=')
            if not separator or not name or not path:
                parser.error(f'--run must be NAME=DIR, got {item!r}')
            training.append(load_training(name, Path(path)))
            input_dirs.append(Path(path))
        for directory in pipelines:
            contents = discover_pipeline(directory, f'{directory.name}:' if len(pipelines) > 1 else '')
            training += contents.training
            reports += contents.evaluations
            notes += [f'Pipeline {directory}: {item}' for item in contents.skipped]
            input_dirs.append(directory)
        for item in evaluations:
            name, path = _named(item, _report_name)
            reports.append(load_evaluation(name, path))
            if path.stem == 'result':
                input_dirs.append(path.parent)
        searches_loaded = [load_search_accuracy(*_named(item, lambda p: p.stem)) for item in searches]
    except InputError as exc:
        parser.error(str(exc))
    names = [source.name for source in training + reports + searches_loaded]
    if duplicates := sorted({name for name in names if names.count(name) > 1}):
        parser.error(f'Experiment names must be unique; repeated: {", ".join(duplicates)}')
    if output_dir is not None:
        resolved = output_dir.resolve()
        for directory in input_dirs:
            if resolved == directory.resolve() or resolved.is_relative_to(directory.resolve()):
                parser.error(f'--output-dir {output_dir} is inside the input directory {directory}; choose a separate destination so inputs stay unmodified')

    print(f'Loading {len(training)} training run(s), {len(reports)} evaluation report(s), {len(searches_loaded)} search report(s)...')
    data = build_dataset(training, reports, searches_loaded, notes)
    if training:
        column = X_COLUMNS[x_axis][0]
        incompatible = [name for name, frame in data.training.items() if column not in frame or frame[column].notna().sum() == 0]
        if incompatible:
            parser.error(f'--x-axis {x_axis} is incompatible with run(s) that do not record `{column}`: {", ".join(incompatible)}. '
                         + ('Hand-objective runs have no match counts; use --x-axis hands.' if x_axis == 'matches' else 'Choose another x-axis.'))
    comparison, warnings = compare_experiments(data)
    warnings = data.warnings + warnings
    options = Options(x_axis=x_axis, smooth=smooth, aggregate_seeds=bool(args.aggregate_seeds if args.aggregate_seeds is not None else config.get('aggregate_seeds', False)), annotate=annotate)
    metric_patterns = _split(args.metrics) if args.metrics else _split(config.get('metrics') or [])
    figures, omissions = build_figures(data, families, options, metric_patterns)
    name = config.get('name') or (output_dir.name if output_dir else 'sc42-visualization')
    report = args.report if args.report is not None else config.get('report', 'html' in formats)

    print(f'Figure families: {", ".join(families)}')
    if args.rq or (config.get('rq') and 'plots' not in config):
        print('Research-question presets select figures; they do not claim the available data answers the question.')
    for warning in warnings:
        print(f'WARNING: {warning}')
    for figure in figures:
        print(f'  plan {figure.id}: {figure.title}' if args.dry_run else f'  figure {figure.id}')
        for experiment, reason in figure.unavailable.items():
            print(f'    series left out, {experiment}: {reason}')
    for omission in omissions:
        print(f'  OMITTED {omission.id}: {omission.reason} (needs: {omission.required})')
    print(f'{len(figures)} figure(s) planned; {len(omissions)} omitted for missing data.')
    if args.dry_run:
        print(f'Dry run: no files written. Formats: {", ".join(formats)}; output directory: {output_dir}; HTML report: {"yes" if report else "no"}')
        return 0
    settings = dict(plots=families, formats=formats, x_axis=x_axis, smooth=smooth, aggregate_seeds=options.aggregate_seeds, annotate=annotate, metrics=metric_patterns, report=bool(report),
                    research_question=args.rq or config.get('rq'))
    manifest = write_outputs(data, figures, omissions, comparison, warnings, output_dir, settings, name=name, report=bool(report))
    print(f'Wrote {len(manifest["files"]) + 1} file(s) under {output_dir}' + (f'; open {output_dir / "index.html"}' if report else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
