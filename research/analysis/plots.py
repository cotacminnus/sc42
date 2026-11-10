"""Figure catalog and rendering.

Family builders turn a Dataset into backend-neutral `Figure` specifications, or
into `Omission` records explaining what is missing. Two renderers draw the same
specification: Matplotlib for PNG/SVG/PDF and Plotly for interactive HTML.
"""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import fnmatch
from pathlib import Path
import re

import matplotlib

matplotlib.use('Agg')  # Headless: figures are only ever written to files.
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from . import FAMILIES
from .metrics import (X_COLUMNS, aggregate_by_ordinal, parse_fraction, partner_matrices, partnership_gap, pooled_windows,
                      round_robin, seed_groups)
from .normalize import BID_CATEGORIES, CALL_CATEGORIES, PHASES, SCORE_STATES, TEAMS, Dataset

PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
DASHES = ('solid', 'dashed', 'dotted', 'dashdot')
MPL_DASHES = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.'}
HATCHES = ('', '//', '..', 'xx')
PLOTLY_HATCHES = {'': '', '//': '/', '..': '.', 'xx': 'x'}
MISSING_CELL = '#e6e6e3'
PROGRESS_WINDOWS = 20
LABEL_BARS_UP_TO = 18
HARDWARE_NOTE = ' Timings depend on the machine and worker count; do not compare across hardware.'
SELF_PLAY_NOTE = (' Training reward against a changing self-play population is not absolute playing strength.')


@dataclass
class Options:
    x_axis: str = 'hands'
    smooth: int = 0
    aggregate_seeds: bool = False
    annotate: str = 'value'  # Matrix cell text: 'none', 'value', 'ci' or 'n'.


@dataclass
class Series:
    label: str
    y: list
    x: list | None = None
    low: list | None = None
    high: list | None = None
    n: list | None = None  # Text per point: sample count, shown on bars and hover.
    key: str | None = None  # Identity that picks the color; defaults to the label.
    dash: str | None = None
    width: float = 1.5
    markers: bool = False
    pattern: int | None = None  # Hatch or dash variant, for several series of one experiment.


@dataclass
class Panel:
    kind: str  # 'line', 'bar', 'area', 'scatter' or 'heatmap'
    xlabel: str = ''
    ylabel: str = ''
    title: str = ''
    series: list[Series] = field(default_factory=list)
    categories: list[str] | None = None
    dodge: bool = True
    # Heatmap only.
    matrix: np.ndarray | None = None
    row_labels: list[str] | None = None
    column_labels: list[str] | None = None
    annotations: list[list[str]] | None = None
    outlines: dict[tuple[int, int], str] = field(default_factory=dict)
    center: float | None = None
    value_label: str = ''


@dataclass
class Figure:
    id: str
    family: str
    title: str
    caption: str
    panels: list[Panel]
    unavailable: dict[str, str] = field(default_factory=dict)  # Experiment left out -> why.
    table: pd.DataFrame | None = None


@dataclass
class Omission:
    id: str
    family: str
    reason: str
    required: str


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_') or 'x'


def _column(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name in frame and frame[name].notna().any():
        return frame[name]
    return None


def _total(frame: pd.DataFrame, name: str) -> float | None:
    column = _column(frame, name)
    return None if column is None else float(column.sum())


def _nan(value) -> float:
    return float('nan') if value is None else float(value)


def _ratio(top, bottom) -> float:
    return float('nan') if top is None or not bottom else top / bottom


def _count(value) -> str:
    return 'n not recorded' if value is None or value != value else f'n={value:,.0f}'


def _ordinals(experiments: Iterable[str]) -> list[int]:
    """Position of each entry among those of its experiment, for hatch and dash variants."""
    seen: dict[str, int] = {}
    result = []
    for experiment in experiments:
        result.append(seen.get(experiment, 0))
        seen[experiment] = result[-1] + 1
    return result


def _rolling(y: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy() if window > 1 else y


def _variants(frame: pd.DataFrame, column: str) -> list[tuple[str, str]]:
    """(label suffix, column): paired runs log each partner's learner metrics separately."""
    paired = [(f' [partner {i}]', f'partner_{i}:{column}') for i in (1, 2) if _column(frame, f'partner_{i}:{column}') is not None]
    if paired:
        return paired
    return [('', column)] if _column(frame, column) is not None else []


def _why_missing(data: Dataset, experiment: str, column: str) -> str:
    meta = data.meta(experiment)
    if meta.get('critic_mode') == 'none' and ('value' in column or 'explained_variance' in column):
        return 'critic-free PPO has no value function'
    return f'`{column}` is not recorded'


def _critic_label(data: Dataset, experiment: str) -> str:
    meta = data.meta(experiment)
    return f'{experiment} [critic: {meta["critic_mode"] or "not recorded"}; returns: {meta["return_estimator"] or "not recorded"}]'


def _plain(data: Dataset, experiment: str) -> str:
    return experiment


def _training_series(data: Dataset, column: str, options: Options, *, x_axis: str | None = None, label=_plain, dash=None,
                     suffix: str = '', transform=None) -> tuple[list[Series], dict[str, str]]:
    """One raw series per run (and partner); optional smoothing is applied only if requested."""
    x_column = X_COLUMNS[x_axis or options.x_axis][0]
    series, unavailable = [], {}
    for experiment, frame in data.training.items():
        variants = _variants(frame, column)
        if not variants:
            unavailable[experiment] = _why_missing(data, experiment, column)
            continue
        if _column(frame, x_column) is None:
            unavailable[experiment] = f'x-axis `{x_column}` is not recorded'
            continue
        for partner, name in variants:
            keep = frame[name].notna() & frame[x_column].notna()
            y = frame.loc[keep, name].to_numpy()
            if transform:
                y = transform(y)
            series.append(Series(label(data, experiment) + partner + suffix, _rolling(y, options.smooth).tolist(),
                                 x=frame.loc[keep, x_column].tolist(), key=experiment, dash=dash, width=0.7 if len(y) > 2000 else 1.2))
    return series, unavailable


def _smooth_note(options: Options) -> str:
    return f' Lines show a rolling mean over {options.smooth} updates (optional, --smooth).' if options.smooth > 1 else ''


def _line_figure(data: Dataset, options: Options, family: str, figure_id: str, title: str, column: str, ylabel: str, caption: str,
                 *, label=_plain) -> Figure | Omission:
    series, unavailable = _training_series(data, column, options, label=label)
    if not series:
        return Omission(figure_id, family, '; '.join(f'{k}: {v}' for k, v in unavailable.items()) or 'no training runs', f'progress.jsonl `{column}`')
    return Figure(figure_id, family, title, caption + _smooth_note(options),
                  [Panel('line', X_COLUMNS[options.x_axis][1], ylabel, series=series)], unavailable)


def _phase_figure(data: Dataset, options: Options, family: str, figure_id: str, title: str, pattern: str, ylabel: str, caption: str,
                  *, label=_plain) -> Figure | Omission:
    panels, unavailable = [], {}
    for phase in PHASES:
        series, missing = _training_series(data, pattern.format(phase=phase), options, label=label)
        unavailable.update({f'{k} ({phase})': v for k, v in missing.items()})
        if series:
            panels.append(Panel('line', X_COLUMNS[options.x_axis][1], ylabel, phase.title(), series))
    if not panels:
        return Omission(figure_id, family, '; '.join(sorted(set(unavailable.values()))) or 'no training runs',
                        f'progress.jsonl `{pattern.format(phase="<phase>")}`')
    return Figure(figure_id, family, title, caption + _smooth_note(options), panels, unavailable)


def _break_gaps(x: np.ndarray, *columns: np.ndarray) -> tuple:
    """Insert NaN where an evaluation is missing, so a line never bridges an absent measurement."""
    if len(x) < 3:
        return (x, *columns)
    steps = np.diff(x)
    breaks = np.flatnonzero(steps > 1.5 * np.median(steps)) + 1
    return tuple(np.insert(c.astype(float), breaks, np.nan) for c in (x, *columns))


def _training_evaluation(data: Dataset) -> pd.DataFrame:
    frame = data.evaluation
    return frame if frame.empty else frame[(frame['source'] == 'training') & (frame['policy'] == 'A')]


def _reports(data: Dataset) -> pd.DataFrame:
    frame = data.evaluation
    return frame if frame.empty else frame[(frame['source'] == 'report') & (frame['policy'] == 'A')]


EVALUATION_METRICS = {'hand_win_rate': ('Hand win rate', 'Hand win rate (fraction of hands)'),
                      'match_win_rate': ('Match win rate', 'Match win rate (fraction of matches)'),
                      'unknown_win_rate': ('Win rate (episode type not recorded)', 'Win rate (fraction of episodes)'),
                      'mean_signed_marks_per_hand': ('Mean signed marks per hand', 'Mean signed marks per hand (marks)')}


def _units_note(frame: pd.DataFrame) -> str:
    groups, games = frame['groups'].dropna(), frame['games'].dropna()
    parts = []
    if len(groups):
        parts.append(f'{groups.min():,.0f}-{groups.max():,.0f} duplicate groups per evaluation'
                     if groups.min() != groups.max() else f'{groups.iloc[0]:,.0f} duplicate groups per evaluation')
    if len(games):
        parts.append(f'{games.min():,.0f} games' if games.min() == games.max() else f'{games.min():,.0f}-{games.max():,.0f} games')
    seeds = sorted({str(s) for s in frame['eval_seed'].dropna()})
    if seeds:
        parts.append('evaluation seed ' + (seeds[0] if len(seeds) == 1 else f'varies ({len(seeds)} values)'))
    episodes = sorted(set(frame['episode']))
    parts.append('episode: ' + ', '.join(episodes))
    return '; '.join(parts) + '.'


def _evaluation_panels(data: Dataset, frame: pd.DataFrame, options: Options, axis: str, ylabel: str, title: str = '') -> tuple[Panel, dict]:
    x_column = X_COLUMNS[axis][0]
    series, unavailable = [], {}
    trajectories: dict[str, tuple] = {}
    for experiment, group in frame.groupby('experiment', sort=False):
        group = group[group[x_column].notna()] if x_column in group else group.iloc[0:0]
        if group.empty:
            unavailable[experiment] = f'x-axis `{x_column}` is not recorded for its evaluations'
            continue
        group = group.sort_values('source_line')
        x, y = group[x_column].to_numpy(dtype=float), group['value'].to_numpy(dtype=float)
        low, high = group['ci_low'].to_numpy(dtype=float), group['ci_high'].to_numpy(dtype=float)
        trajectories[experiment] = (x, y)
        drawn = _break_gaps(x, y, low, high)
        series.append(Series(experiment, drawn[1].tolist(), x=drawn[0].tolist(), low=drawn[2].tolist(), high=drawn[3].tolist(),
                             key=experiment, markers=len(x) <= 60))
    if options.aggregate_seeds:
        for names in seed_groups(data):
            present = [trajectories[name] for name in names if name in trajectories]
            if len(present) > 1:
                result = aggregate_by_ordinal(present)
                series.append(Series(f'mean of {result["seeds"]} seeds ({names[0]}...)', result['y'].tolist(), x=result['x'].tolist(),
                                     low=(result['y'] - result['sd']).tolist(), high=(result['y'] + result['sd']).tolist(),
                                     key=f'aggregate:{names[0]}', width=2.6))
    return Panel('line', X_COLUMNS[axis][1], ylabel, title, series), unavailable


def family_learning(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'learning'
    frame = _training_evaluation(data)
    if frame.empty:
        yield Omission('learning_evaluation', family, 'no in-training held-out evaluations were found',
                       'progress.jsonl `evaluations` or `evaluation`')
    else:
        aggregate_note = (' Thick line: mean over seeds at each evaluation ordinal; its band is +/-1 between-seed standard deviation '
                          '(separate from the per-run evaluation intervals).' if options.aggregate_seeds else '')
        common = ('Fixed-opponent evaluations from progress.jsonl. Bands are the recorded 95% bootstrap intervals over complete duplicate '
                  'groups. Line breaks mark missing evaluations; no values are interpolated.' + aggregate_note)
        for metric, (title, ylabel) in EVALUATION_METRICS.items():
            for episode, by_episode in frame[frame['metric'] == metric].groupby('episode', sort=True):
                for opponent, group in by_episode.groupby('opponent', sort=True):
                    # Signed marks from hand and match evaluations come from different schedules and stay separate.
                    prefix = '' if 'win_rate' in metric else f'{episode}_eval_'
                    base = f'learning_{prefix}{metric}_vs_{slug(opponent)}'
                    caption = f'{title} against {opponent} over training. {common} {_units_note(group)}'
                    panel, missing = _evaluation_panels(data, group, options, options.x_axis, ylabel)
                    yield Figure(base, family, f'{title} vs {opponent}', caption, [panel], missing)
                    if group['runtime_cumulative'].notna().any() if 'runtime_cumulative' in group else False:
                        panel, missing = _evaluation_panels(data, group, options, 'runtime', ylabel)
                        yield Figure(base + '_by_runtime', family, f'{title} vs {opponent} by runtime', caption + HARDWARE_NOTE, [panel], missing)
                    else:
                        yield Omission(base + '_by_runtime', family, 'cumulative runtime is not recorded', 'progress.jsonl `runtime_seconds`')
        names = data.training_runs()
        if len(names) < 2:
            yield Omission('learning_method_comparison', family, 'comparing training methods requires at least two training runs',
                           'two or more --run or pipeline training steps')
        else:
            for metric, (title, ylabel) in EVALUATION_METRICS.items():
                for episode, by_episode in frame[frame['metric'] == metric].groupby('episode', sort=True):
                    last = by_episode.sort_values('source_line').groupby(['experiment', 'opponent'], sort=False).tail(1)
                    opponents = sorted(set(last['opponent']))
                    series = []
                    for experiment in names:
                        group = last[last['experiment'] == experiment].set_index('opponent')
                        if not group.empty:
                            series.append(Series(experiment, [_nan(group['value'].get(o)) for o in opponents],
                                                 low=[_nan(group['ci_low'].get(o)) for o in opponents],
                                                 high=[_nan(group['ci_high'].get(o)) for o in opponents],
                                                 n=[_count(group['groups'].get(o)) + ' groups' for o in opponents], key=experiment))
                    prefix = '' if 'win_rate' in metric else f'{episode}_eval_'
                    hands = '; '.join(f'{e}: {h:,.0f} hands' for e, h in last.groupby('experiment', sort=False)['hands'].max().items())
                    yield Figure(f'learning_method_comparison_{prefix}{metric}', family, f'{title}: final in-training evaluation',
                                 f'Last in-training evaluation of each run (the current policy at that point, not best.pt). {hands}. '
                                 f'Whiskers are recorded 95% intervals over duplicate groups. {_units_note(last)}',
                                 [Panel('bar', 'Evaluation opponent', ylabel, series=series, categories=opponents)])
    objectives: dict[str, list[str]] = {}
    for experiment in data.training_runs():
        objectives.setdefault(data.meta(experiment)['reward_unit'], []).append(experiment)
    for unit, names in objectives.items():
        subset = Dataset(experiments=data.experiments, training={n: data.training[n] for n in names})
        figure = _line_figure(subset, options, family, f'learning_training_reward_{slug(unit)}', f'Training reward ({unit})', 'train_reward',
                              f'Mean training reward ({unit})', 'Mean rollout reward, one point per PPO update.' + SELF_PLAY_NOTE)
        yield figure


def family_ppo(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'ppo'
    caution = {'policy_loss': ' A falling policy loss does not by itself indicate stronger play.',
               'entropy': ' A rapid fall can indicate premature loss of exploration.',
               'approx_kl': ' Spikes indicate large policy changes between updates.',
               'clip_fraction': ' Fraction of samples where the clipped surrogate was active.',
               'gradient_norm': ''}
    for metric, (title, unit) in dict(policy_loss=('Policy loss', 'Policy loss (dimensionless)'), entropy=('Policy entropy', 'Entropy (nats)'),
                                      approx_kl=('Approximate KL', 'Approximate KL (nats)'),
                                      clip_fraction=('Clipped update fraction', 'Clip fraction (0-1)'),
                                      gradient_norm=('Gradient norm', 'Global L2 gradient norm')).items():
        yield _line_figure(data, options, family, f'ppo_{metric}', f'PPO {title.lower()}', f'ppo.{metric}', unit,
                           f'{title}, one point per PPO update.{caution[metric]}')
    series, unavailable = [], {}
    for experiment, frame in data.training.items():
        columns = {phase: _variants(frame, f'samples.{phase}') for phase in PHASES}
        if not any(columns.values()):
            unavailable[experiment] = '`phase_samples` is not recorded'
            continue
        partners = sorted({suffix for values in columns.values() for suffix, _ in values})
        for partner in partners:
            totals = [next((_total(frame, column) for suffix, column in columns[phase] if suffix == partner), None) for phase in PHASES]
            series.append(Series(experiment + partner, [_nan(v) for v in totals], n=[_count(v) for v in totals], key=experiment))
    if series:
        yield Figure('ppo_phase_samples', family, 'PPO samples by phase', 'Trainable decisions summed over all PPO updates. Absent phases were not trained or not recorded.',
                     [Panel('bar', 'Phase', 'Samples (decisions)', series=series, categories=list(PHASES))], unavailable)
    else:
        yield Omission('ppo_phase_samples', family, '`phase_samples` is not recorded in these logs (legacy play-only logs predate it)', 'progress.jsonl `phase_samples`')
    yield _phase_figure(data, options, family, 'ppo_phase_entropy', 'Per-phase policy entropy', 'ppo.{phase}_entropy', 'Entropy (nats)', 'Entropy per phase, one point per PPO update.')
    yield _phase_figure(data, options, family, 'ppo_phase_approx_kl', 'Per-phase approximate KL', 'ppo.{phase}_approx_kl', 'Approximate KL (nats)', 'Approximate KL per phase, one point per PPO update.')


def family_critic(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    """Critic targets are in reward units, so objectives with different units never share a figure."""
    groups: dict[str, list[str]] = {}
    for name in data.training_runs():
        groups.setdefault(data.meta(name)['reward_objective'], []).append(name)
    for objective, names in groups.items() or {'hand-marks': []}.items():
        subset = Dataset(experiments=data.experiments, training={name: data.training[name] for name in names})
        for item in _critic_figures(subset, options, 'terminal reward' if objective == 'match-win' else 'marks'):
            if len(groups) > 1:
                item.id += '_' + slug(objective)
            yield item


def _critic_figures(data: Dataset, options: Options, unit: str) -> Iterator[Figure | Omission]:
    family = 'critic'
    label = _critic_label
    estimator = ' Labels give each run\'s critic and return estimator: local-GAE versus centralized-GAE isolates critic architecture more directly than local-GAE versus critic-free Monte Carlo.'
    yield _line_figure(data, options, family, 'critic_value_loss', 'Critic value loss', 'ppo.value_loss', f'Value loss (MSE, squared {unit})',
                       'Value loss, one point per PPO update.' + estimator, label=label)
    yield _line_figure(data, options, family, 'critic_explained_variance', 'Critic explained variance', 'value.explained_variance', 'Explained variance (1 = perfect)',
                       'Explained variance of returns by the value predictions, one point per update.' + estimator, label=label)
    panels, unavailable = [], {}
    x_column, xlabel = X_COLUMNS[options.x_axis]

    def pooled_means(name: str, phase: str, suffix: str, dash: str | None) -> list[Series]:
        series = []
        column = f'value.{phase}.{name}'
        for experiment, frame in data.training.items():
            variants = _variants(frame, column)
            if not variants:
                unavailable[f'{experiment} ({phase})'] = _why_missing(data, experiment, column)
            for partner, full in variants:
                samples = full.removesuffix(name) + 'samples'
                if samples in frame and x_column in frame:
                    pooled = pooled_windows(frame, x_column, (frame[full] * frame[samples]).to_numpy(), samples, PROGRESS_WINDOWS)
                    if pooled:
                        series.append(Series(label(data, experiment) + partner + suffix, pooled['y'], x=pooled['x'], n=[_count(v) + ' decisions' for v in pooled['n']], key=experiment, dash=dash, markers=True))
        return series

    for phase in PHASES:
        lines = pooled_means('return_mean', phase, ' - empirical return', None) + pooled_means('value_mean', phase, ' - predicted value', 'dashed')
        if lines:
            panels.append(Panel('line', xlabel, f'Mean per decision ({unit})', phase.title(), lines))
    if panels:
        yield Figure('critic_value_vs_return', family, 'Predicted value versus empirical return', f'Per-phase rollout means, weighted by decisions and pooled over 1/{PROGRESS_WINDOWS} of the updates.' + estimator, panels, unavailable)
    else:
        yield Omission('critic_value_vs_return', family, '`value_diagnostics` is not recorded', 'progress.jsonl `value_diagnostics.<phase>.value_mean`, `return_mean`')
    yield _phase_figure(data, options, family, 'critic_advantage_std', 'Per-phase advantage standard deviation', 'value.{phase}.advantage_std', f'Advantage std ({unit})',
                        'Standard deviation of advantages before normalization. For critic-free PPO the advantage is the Monte Carlo return.' + estimator, label=label)
    series, missing = _training_series(data, 'ppo.oracle_aux_loss', options, label=label, suffix=' - training loss')
    validation, _ = _training_series(data, 'oracle_validation.loss', options, label=label, dash='dashed', suffix=' - validation loss')
    if series or validation:
        panels = [Panel('line', X_COLUMNS[options.x_axis][1], 'Oracle interval loss (dimensionless)', series=series + validation)]
        errors, _ = _training_series(data, 'ppo.oracle_aux_point_error', options, label=label, suffix=' - training error')
        held, _ = _training_series(data, 'oracle_validation.point_error', options, label=label, dash='dashed', suffix=' - validation error')
        if errors or held:
            panels.append(Panel('line', X_COLUMNS[options.x_axis][1], 'Oracle error (points outside interval)', series=errors + held))
        yield Figure('critic_oracle_auxiliary', family, 'Oracle auxiliary head', 'Interval loss on training batches and on the deal-disjoint validation split, when recorded.', panels, missing)
    else:
        yield Omission('critic_oracle_auxiliary', family, 'no run used an oracle auxiliary head', 'progress.jsonl `metrics.oracle_aux_loss`, `oracle_validation` (--oracle-aux-coef > 0)')


def _bidding_runs(data: Dataset, column: str) -> list[tuple[str, pd.DataFrame]]:
    return [(name, frame) for name, frame in data.training.items() if column in frame]


def _team_bars(data: Dataset, figure_id: str, family: str, title: str, ylabel: str, categories: list[str], read, caption: str,
               probe: str = 'bidding.learner.decisions', xlabel: str = 'Category', required: str = 'progress.jsonl `bidding`') -> Figure | Omission:
    runs = _bidding_runs(data, probe)
    if not runs:
        return Omission(figure_id, family, 'no run records per-decision bidding diagnostics (older logs predate them)', required)
    panels = []
    for team in TEAMS:
        series = []
        for name, frame in runs:
            values, counts = zip(*(read(frame, team, category) for category in categories))
            series.append(Series(name, list(values), n=list(counts), key=name))
        panels.append(Panel('bar', xlabel, ylabel, f'{team.title()} partnership', series, categories=categories))
    unavailable = {name: 'per-decision bidding diagnostics are not recorded' for name in data.training if name not in dict(runs)}
    return Figure(figure_id, family, title, caption, panels, unavailable)


def _rate_reader(numerator, denominator):
    """A reader returning (top/bottom, 'top/bottom'); an undefined rate is NaN, not zero."""
    def read(frame, team, category):
        top, bottom = _total(frame, numerator(team, category)), _total(frame, denominator(team, category))
        if bottom is None or bottom == 0:
            return float('nan'), 'n=0'
        return (top or 0.0) / bottom, f'{top or 0:,.0f}/{bottom:,.0f}'
    return read


def _sum_reader(frame, team, names):
    values = [_total(frame, name) for name in names]
    return None if all(v is None for v in values) else sum(v or 0.0 for v in values)


def _high_rate(frame, team, _):
    top = _sum_reader(frame, team, [f'bidding.{team}.calls.42', f'bidding.{team}.calls.2+ marks'])
    bottom = _total(frame, f'bidding.{team}.decisions')
    return (float('nan'), 'n=0') if not bottom else ((top or 0) / bottom, f'{top or 0:,.0f}/{bottom:,.0f}')


def _bid_share(frame, team, category):
    total = _sum_reader(frame, team, [f'bidding.{team}.calls.{c}' for c in BID_CATEGORIES])
    count = _total(frame, f'bidding.{team}.calls.{category}')
    return (float('nan'), 'n=0') if not total else ((count or 0) / total, f'{count or 0:,.0f}/{total:,.0f}')


def _won_share(frame, team, category):
    total = _sum_reader(frame, team, [f'contracts_won.{team}.{c}' for c in BID_CATEGORIES])
    count = _total(frame, f'contracts_won.{team}.{category}')
    return (float('nan'), 'n=0') if not total else ((count or 0) / total, f'{count or 0:,.0f}/{total:,.0f}')


def _forced_or_voluntary(frame, team, category):
    if category == 'Forced share of bid decisions':
        return _rate_reader(lambda t, c: f'bidding.{t}.forced_calls', lambda t, c: f'bidding.{t}.decisions')(frame, team, category)
    return _rate_reader(lambda t, c: f'bidding.{t}.voluntary_passes', lambda t, c: f'bidding.{t}.voluntary_decisions')(frame, team, category)


def _progress_panels(data: Dataset, options: Options, numerator, denominator, ylabel: str, teams=TEAMS, runs=None) -> list[Panel]:
    x_column, xlabel = X_COLUMNS[options.x_axis]
    panels = []
    for team in teams:
        series = []
        for name, frame in (runs if runs is not None else data.training.items()):
            pooled = pooled_windows(frame, x_column, numerator(frame, team), denominator(frame, team), PROGRESS_WINDOWS)
            if pooled:
                series.append(Series(name, pooled['y'], x=pooled['x'], n=[_count(v) for v in pooled['n']], key=name, markers=True))
        if series:
            panels.append(Panel('line', xlabel, ylabel, f'{team.title()} partnership', series))
    return panels


def family_bidding(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'bidding'
    denominator_note = ' Totals over all rollouts; bars are annotated with counts/denominators, and undefined categories show n/a.'
    decisions = lambda team, category: f'bidding.{team}.decisions'
    yield _team_bars(data, 'bidding_call_distribution', family, 'Auction call distribution', 'Share of bid decisions (0-1)', list(CALL_CATEGORIES),
                     _rate_reader(lambda t, c: f'bidding.{t}.calls.{c}', decisions), 'Share of every bid decision by call.' + denominator_note, xlabel='Call')
    yield _team_bars(data, 'bidding_bid_categories', family, 'Bid attempt categories', 'Share of non-pass calls (0-1)', list(BID_CATEGORIES), _bid_share,
                     'Calls of 30, 31-41, 42 and 2+ marks among non-pass calls: attempts, not contracts.' + denominator_note, xlabel='Bid category')
    yield _team_bars(data, 'bidding_winning_contracts', family, 'Winning contract categories', 'Share of resolved contracts (0-1)', list(BID_CATEGORIES), _won_share,
                     'Categories of the contracts each partnership actually won in rollouts (`rollout_bids`), which differ from bid attempts.' + denominator_note,
                     probe='contracts_won.learner.30', xlabel='Contract category', required='progress.jsonl `rollout_bids`')
    yield _team_bars(data, 'bidding_voluntary_forced', family, 'Forced and voluntary bidding', 'Rate (0-1)', ['Forced share of bid decisions', 'Voluntary pass rate'], _forced_or_voluntary,
                     'Forced calls have no legal pass; the voluntary pass rate covers decisions where pass was legal.' + denominator_note, xlabel='Quantity')
    yield _team_bars(data, 'bidding_high_bid_frequency', family, 'High-bid frequency', 'Share of bid decisions (0-1)', ['42 or 2+ marks'], _high_rate,
                     'Calls of 42 or 2+ marks over all bid decisions.' + denominator_note, xlabel='Calls')
    yield _team_bars(data, 'bidding_attempts_vs_auctions_won', family, 'Bid attempts versus auctions won', 'Auctions won per attempt (0-1)', list(BID_CATEGORIES),
                     _rate_reader(lambda t, c: f'bidding.{t}.attempts.{c}.auction_won', lambda t, c: f'bidding.{t}.attempts.{c}.calls'),
                     'For each attempt category, the share of attempts whose partnership won the auction.' + denominator_note, probe='bidding.learner.attempts.30.calls', xlabel='Bid category')
    yield _team_bars(data, 'bidding_contract_success', family, 'Contract success rate by bid category', 'Made contracts per contract (0-1)', list(BID_CATEGORIES),
                     _rate_reader(lambda t, c: f'bidding.{t}.contracts.{c}.made', lambda t, c: f'bidding.{t}.contracts.{c}.contracts'),
                     'Success is conditional on the contracts each policy chose to bid and won, so it does not by itself measure bidding quality.' + denominator_note,
                     probe='bidding.learner.contracts.30.contracts', xlabel='Contract category')
    bid_numerator = lambda frame, team: (frame[f'bidding.{team}.decisions'] - frame[f'bidding.{team}.calls.pass']).to_numpy()
    high_numerator = lambda frame, team: (frame[f'bidding.{team}.calls.42'].fillna(0) + frame[f'bidding.{team}.calls.2+ marks'].fillna(0)).to_numpy()
    runs = [(n, f) for n, f in data.training.items() if f'bidding.learner.calls.pass' in f]
    caption = f'Each point pools counts over 1/{PROGRESS_WINDOWS} of the recorded updates; x is the end of that window.'
    for figure_id, title, numerator in (('bidding_rate_progress', 'Bid rate over training', bid_numerator), ('bidding_high_bid_progress', 'High-bid rate over training', high_numerator)):
        panels = _progress_panels(data, options, numerator, lambda f, t: f'bidding.{t}.decisions', 'Rate per bid decision (0-1)', runs=runs)
        yield (Figure(figure_id, family, title, caption, panels) if panels else Omission(figure_id, family, 'no run records bid-call counts', 'progress.jsonl `bidding`'))


def _state_read(field_name):
    return _rate_reader(lambda t, c: f'bidding.{t}.{c}.{field_name}', lambda t, c: f'bidding.{t}.{c}.decisions')


def family_match_bidding(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'match-bidding'
    hand_note = (' Runs with the hand-marks objective have no match score: their auctions are all recorded as tied, so leading and trailing are undefined (n/a).')
    probe = 'bidding.learner.tied.decisions'
    yield _team_bars(data, 'match_bidding_by_score_state', family, 'Bid frequency by score state', 'Bids per bid decision (0-1)', list(SCORE_STATES), _state_read('bids'),
                     'Bid rate when the partnership leads, is tied or trails.' + hand_note, probe=probe, xlabel='Score state', required='progress.jsonl `bidding.learner.by_score_state`')
    yield _team_bars(data, 'match_bidding_high_by_score_state', family, 'High-bid frequency by score state', 'High bids per bid decision (0-1)', list(SCORE_STATES), _state_read('high_bids'),
                     '42 or 2+ marks calls by score state.' + hand_note, probe=probe, xlabel='Score state', required='progress.jsonl `bidding.learner.by_score_state`')
    for experiment, teams in data.by_match_score.items():
        scores = teams.get('learner') or {}
        figure_id = f'match_bidding_exact_score_{slug(experiment)}'
        if len(scores) <= 1:
            yield Omission(figure_id, family, f'{experiment} has no exact match-score variation (hand-objective auctions are all 0-0 or the field is absent)', 'progress.jsonl `bidding.learner.by_match_score`')
            continue
        size = 1 + max(max(int(part) for part in key.split('-')) for key in scores)
        matrix, text = np.full((size, size), np.nan), [['' for _ in range(size)] for _ in range(size)]
        for key, (decisions, bids, _) in scores.items():
            own, other = (int(part) for part in key.split('-'))
            matrix[own, other] = bids / decisions
            text[own][other] = f'{bids / decisions:.2f}\n({decisions})' if options.annotate != 'none' else ''
        yield Figure(figure_id, family, f'Learner bid rate by exact match score: {experiment}',
                     'Cells give the bid rate and (decisions), summed over training; blank cells were never observed.',
                     [Panel('heatmap', "Opponents' marks", "Learner partnership's marks", matrix=matrix, row_labels=[str(i) for i in range(size)],
                            column_labels=[str(i) for i in range(size)], annotations=text, center=None, value_label='Bid rate (0-1)')])
    objectives = {name: data.meta(name)['reward_objective'] for name, frame in data.training.items() if probe in frame}
    if len(set(objectives.values())) < 2:
        yield Omission('match_bidding_objective_comparison', family, 'requires runs with both match-win and hand-marks objectives recording score-state bidding', '--run for each objective with `bidding.*.by_score_state`')
    else:
        series = []
        categories = list(SCORE_STATES)
        for name, objective in objectives.items():
            frame = data.training[name]
            values, counts = zip(*(_state_read('bids')(frame, 'learner', state) for state in categories))
            series.append(Series(f'{name} [{objective}]', list(values), n=list(counts), key=name))
        yield Figure('match_bidding_objective_comparison', family, 'Learner bid rate: match-win versus hand-marks', 'Observational comparison of the two objectives; settings may differ, see the comparison table.' + hand_note,
                     [Panel('bar', 'Score state', 'Bids per bid decision (0-1)', series=series, categories=categories)])
    runs = [(name, frame) for name, frame in data.training.items() if 'rollout_matches' in frame]
    match_caption = f'Each point pools rollouts over 1/{PROGRESS_WINDOWS} of the recorded updates.'
    if not runs:
        for figure_id in ('match_bidding_hands_per_match', 'match_bidding_rollout_match_win_rate'):
            yield Omission(figure_id, family, 'no run used the match-win objective', 'progress.jsonl `rollout_matches`, `mean_hands_per_match`, `match_win_rate`')
    else:
        x_column, xlabel = X_COLUMNS[options.x_axis]
        series, wins = [], []
        for name, frame in runs:
            pooled = pooled_windows(frame, x_column, 'rollout_hands', 'rollout_matches', PROGRESS_WINDOWS)
            won = pooled_windows(frame, x_column, (frame['match_win_rate'] * frame['rollout_matches']).to_numpy(), 'rollout_matches', PROGRESS_WINDOWS)
            if pooled:
                series.append(Series(name, pooled['y'], x=pooled['x'], n=[_count(v) + ' matches' for v in pooled['n']], key=name, markers=True))
            if won:
                wins.append(Series(name, won['y'], x=won['x'], n=[_count(v) + ' matches' for v in won['n']], key=name, markers=True))
        yield Figure('match_bidding_hands_per_match', family, 'Mean hands per match over training', match_caption, [Panel('line', xlabel, 'Hands per match', series=series)])
        yield Figure('match_bidding_rollout_match_win_rate', family, 'Rollout match win rate', match_caption + SELF_PLAY_NOTE, [Panel('line', xlabel, 'Match win rate (fraction of matches)', series=wins)])
    frame = _training_evaluation(data)
    rows = frame[frame['metric'] == 'match_win_rate'] if not frame.empty else frame
    if rows.empty:
        yield Omission('match_bidding_heldout_match_win_rate', family, 'no in-training match evaluations were recorded', 'progress.jsonl match-mode `evaluations` (--eval-match-pairs)')
    else:
        for opponent, group in rows.groupby('opponent', sort=True):
            panel, missing = _evaluation_panels(data, group, options, options.x_axis, EVALUATION_METRICS['match_win_rate'][1])
            yield Figure(f'match_bidding_heldout_match_win_rate_vs_{slug(opponent)}', family, f'Held-out match win rate vs {opponent}',
                         'Complete-match evaluations; bands are recorded 95% intervals over duplicate match groups. ' + _units_note(group), [panel], missing)


def _opponent_names(frame: pd.DataFrame) -> list[str]:
    return sorted({column.split('.')[1] for column in frame.columns if column.startswith('opp.') and column.endswith('.hands')})


def family_opponents(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'opponents'
    runs = [(name, frame) for name, frame in data.training.items() if _opponent_names(frame)]
    required = 'progress.jsonl `training_by_opponent`'
    if not runs:
        for figure_id in ('opponents_exposure_share', 'opponents_exposure_counts', 'opponents_training_reward'):
            yield Omission(figure_id, family, 'no training run recorded per-opponent statistics', required)
    x_column, xlabel = X_COLUMNS[options.x_axis]
    for name, frame in runs:
        opponents = _opponent_names(frame)
        hands = frame[[f'opp.{o}.hands' for o in opponents]].fillna(0).to_numpy()
        valid = frame[x_column].notna().to_numpy() if x_column in frame else np.zeros(len(frame), dtype=bool)
        if not valid.any():
            yield Omission(f'opponents_exposure_share_{slug(name)}', family, f'x-axis `{x_column}` is not recorded', f'progress.jsonl `{x_column}`')
            continue
        hands, x = hands[valid], frame.loc[valid, x_column].to_numpy()
        splits = np.array_split(np.arange(len(x)), min(PROGRESS_WINDOWS, len(x)))
        totals = np.array([hands[s].sum(axis=0) for s in splits])
        with np.errstate(invalid='ignore', divide='ignore'):
            shares = (totals.T / totals.sum(axis=1)).T
        keep = totals.sum(axis=1) > 0
        x_points = [float(x[s[-1]]) for s, k in zip(splits, keep) if k]
        series = [Series(o, shares[keep, i].tolist(), x=x_points, key=o) for i, o in enumerate(opponents)]
        yield Figure(f'opponents_exposure_share_{slug(name)}', family, f'Training opponent share: {name}', f'Share of training hands by opponent, pooled over 1/{PROGRESS_WINDOWS} of the updates. '
                     '`self-current` and `self-snapshots` are mutable populations, not single policies.', [Panel('area', xlabel, 'Share of training hands (0-1)', series=series)])
    if runs:
        opponents = sorted({o for _, frame in runs for o in _opponent_names(frame)})
        panels = []
        for unit in ('hands', 'matches'):
            series = []
            for name, frame in runs:
                values = [_total(frame, f'opp.{o}.{unit}') for o in opponents]
                if any(v is not None for v in values):
                    series.append(Series(name, [_nan(v) for v in values], n=[_count(v) for v in values], key=name))
            if series:
                panels.append(Panel('bar', 'Training opponent', f'Training {unit} (count)', series=series, categories=opponents))
        yield Figure('opponents_exposure_counts', family, 'Training exposure by opponent', 'Totals over all rollouts.', panels)
        grouped: dict[str, list] = {}
        for name, frame in runs:
            objective = data.meta(name)['reward_unit']
            unit = 'matches' if data.meta(name)['reward_objective'] == 'match-win' else 'hands'
            values, counts = [], []
            for opponent in opponents:
                reward, denominator = _total(frame, f'opp.{opponent}.reward_sum'), _total(frame, f'opp.{opponent}.{unit}')
                values.append(_ratio(reward, denominator))
                counts.append(_count(denominator) + f' {unit}')
            grouped.setdefault(objective, []).append(Series(name, values, n=counts, key=name))
        panels = [Panel('bar', 'Training opponent', f'Mean training reward ({unit})', series=series, categories=opponents) for unit, series in grouped.items()]
        yield Figure('opponents_training_reward', family, 'Mean training reward by opponent', 'Reward sums over denominators, by reward unit; units never share an axis.' + SELF_PLAY_NOTE, panels)
    frame = _training_evaluation(data)
    if frame.empty:
        yield Omission('opponents_validation', family, 'no in-training held-out evaluations were found', 'progress.jsonl `evaluations`')
    else:
        for metric, (title, ylabel) in EVALUATION_METRICS.items():
            for episode, by_episode in frame[frame['metric'] == metric].groupby('episode', sort=True):
                last = by_episode.sort_values('source_line').groupby(['experiment', 'opponent'], sort=False).tail(1)
                opponents = sorted(set(last['opponent']))
                series = []
                for experiment, group in last.groupby('experiment', sort=False):
                    group = group.set_index('opponent')
                    series.append(Series(experiment, [_nan(group['value'].get(o)) for o in opponents], low=[_nan(group['ci_low'].get(o)) for o in opponents],
                                         high=[_nan(group['ci_high'].get(o)) for o in opponents], n=[_count(group['groups'].get(o)) + ' groups' for o in opponents], key=experiment))
                identities = last.dropna(subset=['opponent_identity'])
                checkpoint = '; identity ' + ', '.join(sorted({f'{r.opponent}={str(r.opponent_identity)[:12]}' for r in identities.itertuples()})) if len(identities) else ''
                prefix = '' if 'win_rate' in metric else f'{episode}_eval_'
                yield Figure(f'opponents_validation_{prefix}{metric}', family, f'Fixed validation: {title.lower()}', f'Final in-training evaluation against each fixed opponent{checkpoint}. Whiskers are recorded 95% intervals. {_units_note(last)}',
                             [Panel('bar', 'Evaluation opponent', ylabel, series=series, categories=opponents)])
    for name in data.training_runs():
        figure_id = f'opponents_configured_vs_realized_{slug(name)}'
        configs, assigned = data.opponent_configs.get(name) or [], data.assignments.get(name) or {}
        if not configs:
            yield Omission(figure_id, family, f'{name}: configured opponent distribution is not recorded', 'progress.jsonl `train_opponents`')
            continue
        configured = {o: parse_fraction(p) for o, p in configs[-1][1]}
        frame = data.training[name]
        realized = {o: sum(counts.values()) for o, counts in assigned.items()} or {o: _total(frame, f'opp.{o}.hands') or 0 for o in _opponent_names(frame)}
        if not realized:
            yield Omission(figure_id, family, f'{name}: realized opponents are not recorded (older log)', 'progress.jsonl `opponent_assignments` or `training_by_opponent`')
            continue
        opponents = sorted(set(configured) | set(realized))
        total = sum(realized.values())
        changed = ' The configuration changed during training; bars show the last one.' if len(configs) > 1 else ''
        source_name = 'assignments' if assigned else 'hands'
        yield Figure(figure_id, family, f'Configured versus realized opponents: {name}', f'Realized share of {total:,.0f} recorded {source_name}. Snapshot opponents can be unavailable until the first snapshot exists.{changed}',
                     [Panel('bar', 'Training opponent', 'Share (0-1)', series=[
                         Series('configured', [_nan(configured.get(o)) for o in opponents], key='configured'),
                         Series('realized', [realized.get(o, 0) / total for o in opponents], n=[_count(realized.get(o)) for o in opponents], key='realized')], categories=opponents)])
    if not any(data.assignments.values()):
        yield Omission('opponents_identities', family, 'no run records per-episode opponent identities', 'progress.jsonl `opponent_assignments`')
    else:
        opponents = sorted({o for counts in data.assignments.values() for o in counts})
        series, notes = [], []
        for name, assigned in data.assignments.items():
            if assigned:
                distinct = [len({i for i in assigned.get(o, {}) if i}) for o in opponents]
                unknown = [sum(c for i, c in assigned.get(o, {}).items() if not i) for o in opponents]
                series.append(Series(name, distinct, n=[f'{d} ids; {u} unidentified' for d, u in zip(distinct, unknown)], key=name))
        for name, assigned in data.assignments.items():
            for opponent, counts in assigned.items():
                if None in counts and len(counts) == 1 and opponent.startswith('self'):
                    notes.append(f'{name}/{opponent}: no checkpoint identity recorded')
        limitation = (' Limitation: ' + '; '.join(notes) + '.') if notes else ''
        yield Figure('opponents_identities', family, 'Distinct recorded opponent identities', 'Distinct checkpoint hashes per opponent name. `self-current` and `self-snapshots` are populations that change during training, so one name is not one immutable policy.' + limitation,
                     [Panel('bar', 'Training opponent', 'Distinct identities (count)', series=series, categories=opponents)])


def _report_label(row) -> str:
    return f'{row.experiment} vs {row.opponent}'


def _categorical_report_figure(figure_id: str, family: str, title: str, ylabel: str, rows: pd.DataFrame, caption: str) -> Figure:
    """One bar per report and opponent, coloured by experiment."""
    categories = [_report_label(r) for r in rows.itertuples()]
    series = []
    for experiment, group in rows.groupby('experiment', sort=False):
        positions = {c: i for i, c in enumerate(categories)}
        y, low, high, n = [float('nan')] * len(categories), [float('nan')] * len(categories), [float('nan')] * len(categories), [''] * len(categories)
        for row in group.itertuples():
            i = positions[_report_label(row)]
            y[i], low[i], high[i] = float(row.value), _nan(row.ci_low), _nan(row.ci_high)
            n[i] = _count(row.groups) + ' groups' if row.groups == row.groups else ''
        series.append(Series(experiment, y, low=low, high=high, n=n, key=experiment))
    return Figure(figure_id, family, title, caption, [Panel('bar', 'Report', ylabel, series=series, categories=categories, dodge=False)])


def family_evaluation(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'evaluation'
    frame = _reports(data)
    if frame.empty:
        yield Omission('evaluation_comparison', family, 'no arena evaluation report was supplied (in-training evaluations appear in `learning`)', '--evaluation PATH or pipeline evaluation steps')
        return
    caption = 'Agent A perspective, with the 95% bootstrap intervals recorded by the arena over complete duplicate groups. '
    for metric, (title, ylabel) in EVALUATION_METRICS.items():
        for episode, group in frame[frame['metric'] == metric].groupby('episode', sort=True):
            suffix = '' if 'win_rate' in metric else f'{episode}_'
            group = group.drop_duplicates(['experiment', 'opponent'])
            yield _categorical_report_figure(f'evaluation_{suffix}{metric}', family, f'{title} ({episode} evaluation)', ylabel, group, caption + _units_note(group))
            opponents = sorted(set(group['opponent']))
            experiments = list(dict.fromkeys(group['experiment']))
            if len(experiments) > 1 and len(opponents) < len(group) or len(opponents) > 1 and len(experiments) == 1:
                series = []
                for experiment in experiments:
                    by = group[group['experiment'] == experiment].set_index('opponent')
                    series.append(Series(experiment, [_nan(by['value'].get(o)) for o in opponents], low=[_nan(by['ci_low'].get(o)) for o in opponents],
                                         high=[_nan(by['ci_high'].get(o)) for o in opponents], n=[_count(by['groups'].get(o)) + ' groups' for o in opponents], key=experiment))
                yield Figure(f'evaluation_{suffix}{metric}_by_opponent', family, f'{title} by opponent ({episode} evaluation)', caption + _units_note(group),
                             [Panel('bar', 'Opponent', ylabel, series=series, categories=opponents)])
    success = frame[frame['metric'] == 'contract_success_rate'].drop_duplicates(['experiment', 'opponent'])
    if success.empty:
        yield Omission('evaluation_contract_success', family, 'reports do not record contract success rates', 'report `metrics[].contract_success_rate`')
    else:
        yield _categorical_report_figure('evaluation_contract_success', family, 'Contract success rate', 'Contracts made per contract declared (0-1)', success,
                                         'Agent A. The arena records no interval for this rate. Success is conditional on the contracts chosen and won. ' + _units_note(success))
    roles = data.roles[data.roles['policy'] == 'A'] if not data.roles.empty else data.roles
    if roles.empty:
        yield Omission('evaluation_declarer_defender', family, 'reports do not record declarer and defender roles', 'report `breakdown.<policy>.roles`')
    else:
        panels = []
        for column, label in (('win_rate', 'Win rate in role (fraction of hands)'), ('mean_signed_marks', 'Mean signed marks per hand (marks)')):
            series = []
            groups = list(roles.groupby(['experiment', 'opponent'], sort=False))
            for ((experiment, opponent), group), pattern in zip(groups, _ordinals(key[0] for key, _ in groups)):
                by = group.set_index('role')
                categories = ['declarer', 'defender']
                series.append(Series(f'{experiment} vs {opponent}', [_nan(by[column].get(r)) for r in categories], n=[_count(by['hands'].get(r)) + ' hands' for r in categories], key=experiment, pattern=pattern))
            panels.append(Panel('bar', 'Role', label, series=series, categories=categories))
        yield Figure('evaluation_declarer_defender', family, 'Declarer and defender performance', 'Agent A, hands by partnership role. Roles are conditional on auctions, so role differences are observational.', panels)
    lengths = frame[frame['metric'] == 'average_match_length'].drop_duplicates(['experiment', 'opponent'])
    if lengths.empty:
        yield Omission('evaluation_match_length', family, 'no report records a complete-match average length', 'report with episode=match and `average_match_length`')
    else:
        yield _categorical_report_figure('evaluation_match_length', family, 'Average match length', 'Hands per match', lengths, 'Mean hands per complete match. ' + _units_note(lengths))


def _contract_reports(data: Dataset) -> list[tuple[str, pd.DataFrame]]:
    frame = data.contracts
    if frame.empty:
        return []
    frame = frame[frame['policy'] == 'A']
    return [(f'{experiment} vs {opponent}', group, experiment) for (experiment, opponent), group in frame.groupby(['experiment', 'opponent'], sort=False)]


def family_contracts(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'contracts'
    reports = _contract_reports(data)
    required = 'report `breakdown`/`breakdowns`, or per-hand records'
    if not reports:
        for figure_id in ('contracts_frequency', 'contracts_success_by_category', 'contracts_forced_voluntary'):
            yield Omission(figure_id, family, 'no contract breakdowns were found', required)
    else:
        categories = [c for c in BID_CATEGORIES if c in set(data.contracts['category'])] + (['other'] if 'other' in set(data.contracts['category']) else [])
        frequency, success = [], []
        patterns = _ordinals(experiment for _, _, experiment in reports)
        for (label, group, experiment), pattern in zip(reports, patterns):
            by = group.groupby('category')[['contracts', 'successful']].sum()
            total = by['contracts'].sum()
            frequency.append(Series(label, [by['contracts'].get(c, 0) / total if total else float('nan') for c in categories],
                                    n=[f'{by["contracts"].get(c, 0):,.0f}/{total:,.0f}' for c in categories], key=experiment, pattern=pattern))
            success.append(Series(label, [by['successful'][c] / by['contracts'][c] if by['contracts'].get(c, 0) else float('nan') for c in categories],
                                  n=[f'{by["successful"].get(c, 0):,.0f}/{by["contracts"].get(c, 0):,.0f}' for c in categories], key=experiment, pattern=pattern))
        yield Figure('contracts_frequency', family, 'Contract frequency by category', 'Agent A contracts by category, with counts/denominators.', [Panel('bar', 'Contract category', 'Share of contracts (0-1)', series=frequency, categories=categories)])
        yield Figure('contracts_success_by_category', family, 'Contract success by category', 'Agent A. n/a marks categories with no contracts, which is different from zero success. Success is conditional on contracts chosen.',
                     [Panel('bar', 'Contract category', 'Contracts made per contract (0-1)', series=success, categories=categories)])
        origins = ['forced', 'voluntary']
        if 'origin' in data.contracts and set(data.contracts['origin']) <= {'unknown'}:
            yield Omission('contracts_forced_voluntary', family, 'contract origin was not recorded', 'report `breakdown.<policy>.contract_origins`')
        else:
            shares, made = [], []
            for (label, group, experiment), pattern in zip(reports, patterns):
                by = group.groupby('origin')[['contracts', 'successful']].sum()
                total = by['contracts'].sum()
                shares.append(Series(label, [by['contracts'].get(o, 0) / total if total else float('nan') for o in origins], n=[f'{by["contracts"].get(o, 0):,.0f}/{total:,.0f}' for o in origins], key=experiment, pattern=pattern))
                made.append(Series(label, [by['successful'][o] / by['contracts'][o] if by['contracts'].get(o, 0) else float('nan') for o in origins],
                                   n=[f'{by["successful"].get(o, 0):,.0f}/{by["contracts"].get(o, 0):,.0f}' for o in origins], key=experiment, pattern=pattern))
            yield Figure('contracts_forced_voluntary', family, 'Forced versus voluntary contracts', 'Agent A. Forced contracts are bids made because passing was illegal.',
                         [Panel('bar', 'Origin', 'Share of contracts (0-1)', series=shares, categories=origins), Panel('bar', 'Origin', 'Contracts made per contract (0-1)', series=made, categories=origins)])
    timing = data.timing[data.timing['policy'] == 'A'] if not data.timing.empty else data.timing
    if timing.empty:
        yield Omission('contracts_decision_timing', family, 'contract decision timing was not recorded', 'report `breakdown.<policy>.decision_timing` or per-hand `contract_decided_at_trick`')
    else:
        tricks = sorted(set(timing['trick']))
        series, notes = [], []
        groups = list(timing.groupby(['experiment', 'opponent'], sort=False))
        for ((experiment, opponent), group), pattern in zip(groups, _ordinals(key[0] for key, _ in groups)):
            counts = group.set_index('trick')['count']
            recorded, contracts = group['recorded'].iloc[0], group['contracts'].iloc[0]
            series.append(Series(f'{experiment} vs {opponent}', [counts.get(t, 0) / recorded if recorded else float('nan') for t in tricks], n=[f'{counts.get(t, 0):,.0f}/{recorded:,.0f}' for t in tricks], key=experiment, pattern=pattern))
            if recorded != contracts:
                notes.append(f'{experiment} vs {opponent}: recorded for {recorded:,.0f} of {contracts:,.0f} contracts')
        yield Figure('contracts_decision_timing', family, 'Contract decision timing by trick', 'Trick at which the contract outcome was decided; only positions where timing was recorded.' + (' ' + '; '.join(notes) + '.' if notes else ''),
                     [Panel('bar', 'Trick', 'Share of recorded contracts (0-1)', series=series, categories=[str(t) for t in tricks])])
    roles = data.roles[data.roles['policy'] == 'A'] if not data.roles.empty else data.roles
    if roles.empty:
        yield Omission('contracts_roles', family, 'declaring and defending outcomes were not recorded', 'report `breakdown.<policy>.roles`')
    else:
        panels = []
        for column, label in (('hands', 'Hands in role (count)'), ('wins', 'Hands won in role (count)')):
            series = []
            groups = list(roles.groupby(['experiment', 'opponent'], sort=False))
            for ((experiment, opponent), group), pattern in zip(groups, _ordinals(key[0] for key, _ in groups)):
                by = group.set_index('role')
                categories = ['declarer', 'defender']
                series.append(Series(f'{experiment} vs {opponent}', [_nan(by[column].get(r)) for r in categories], n=[_count(by[column].get(r)) for r in categories], key=experiment, pattern=pattern))
            panels.append(Panel('bar', 'Role', label, series=series, categories=categories))
        yield Figure('contracts_roles', family, 'Declaring versus defending outcomes', 'Agent A partnership. Counts, not rates, so sample sizes stay visible.', panels)


def _cell_text(matrix: dict, row: int, column: int, annotate: str) -> str:
    value = matrix['value'][row, column]
    if value != value:
        return ''
    if annotate == 'none':
        return ''
    text = f'{value:+.3f}'
    if annotate == 'ci' and matrix['low'][row, column] == matrix['low'][row, column]:
        text += f'\n[{matrix["low"][row, column]:+.2f}, {matrix["high"][row, column]:+.2f}]'
    if annotate == 'n' and matrix['groups'][row, column] == matrix['groups'][row, column]:
        text += f'\nn={matrix["groups"][row, column]:,.0f}'
    return text


def _matrix_table(labels_row, labels_column, matrix: dict) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(labels_row):
        for j, c in enumerate(labels_column):
            rows.append({'row': r, 'column': c, **{k: matrix[k][i, j] for k in ('value', 'low', 'high', 'groups', 'games') if k in matrix}})
    return pd.DataFrame(rows)


ROUND_ROBIN_METRICS = {'hand_signed_marks': ('mean_signed_marks_per_hand', 'hand', 'Mean signed marks per hand (row agent)', 0.0),
                       'match_signed_marks': ('mean_signed_marks_per_hand', 'match', 'Mean signed marks per hand in matches (row agent)', 0.0),
                       'match_win_rate': ('match_win_rate', 'match', 'Match win rate (row agent)', 0.5)}


def _episode_data(data: Dataset, episode: str) -> Dataset:
    """A view holding only the composed reports of one episode, so matrices cannot mix them."""
    frame = data.evaluation
    names = set(frame[(frame['source'] == 'report') & (frame['episode'] == episode)]['experiment'])
    return Dataset(evaluation=frame[frame['experiment'].isin(names)], pairings={k: v for k, v in data.pairings.items() if k in names})


def family_round_robin(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'round-robin'
    required = 'two or more evaluate_agent_vs_agent reports with `composition`'
    if len(data.pairings) < 2:
        yield Omission('round_robin', family, f'found {len(data.pairings)} evaluate_agent_vs_agent report(s); a matrix needs at least two', required)
        return
    for name, (metric, episode, label, center) in ROUND_ROBIN_METRICS.items():
        figure_id = f'round_robin_{name}'
        matrix = round_robin(_episode_data(data, episode), metric)
        if matrix is None:
            yield Omission(figure_id, family, f'no {episode}-episode composed report records `{metric}`', required)
            continue
        labels = matrix['labels']
        text = [[_cell_text(matrix, i, j, options.annotate) for j in range(len(labels))] for i in range(len(labels))]
        notes = []
        if matrix['duplicates']:
            notes.append('Same weights under several names, counted once: ' + '; '.join(f'{k} = {", ".join(v)}' for k, v in matrix['duplicates'].items()) + '.')
        if matrix['repeated']:
            notes.append('Superseded repeats of a cell (largest sample shown; none averaged): ' + ', '.join(matrix['repeated']) + '.')
        if np.isfinite(np.diag(matrix['value'])).any():
            notes.append('Diagonal cells compare a model with itself: a symmetry check, not evidence of strength.')
        groups = matrix['groups'][np.isfinite(matrix['groups'])]
        size = f'{groups.min():,.0f}-{groups.max():,.0f} duplicate groups per cell' if len(groups) else 'groups not recorded'
        yield Figure(figure_id, family, f'Round robin: {label.lower()}', f'{label} against the column agent; {size}. Grey cells were not tested and are not zero. {" ".join(notes)}'.strip(),
                     [Panel('heatmap', 'Agent B (column)', 'Agent A (row)', matrix=matrix['value'], row_labels=labels, column_labels=labels, annotations=text, center=center, value_label=label)],
                     table=_matrix_table(labels, labels, matrix))


def family_partners(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'partners'
    required = 'evaluate_agent_vs_agent reports with --agent-a-partner-checkpoint, for each first-seat x second-seat pair'
    results = partner_matrices(data) if data.pairings else []
    if not results:
        known = sorted({team['first_label'] for p in data.pairings.values() for team in p['teams'].values() if team['first_label']})
        comparisons = ('Run evaluate_agent_vs_agent with --agent-a-checkpoint X --agent-a-partner-checkpoint Y against one fixed opponent, for every first-seat X and second-seat Y'
                       + (f', e.g. among: {", ".join(known)}' if known else '') + '.')
        yield Omission('partners_compatibility', family, f'no report has an explicit mixed partnership. {comparisons}', required)
        return
    for result in results:
        rows, base = result['rows'], f'partners_{slug(result["opponent"])}_{result["episode"]}'
        for metric, label, center in (('mean_signed_marks_per_hand', 'Partnership mean signed marks per hand', 0.0), ('match_win_rate', 'Partnership match win rate', 0.5)):
            if metric == 'match_win_rate' and result['episode'] != 'match':
                continue
            size = (len(result['first_labels']), len(result['second_labels']))
            matrix = {name: np.full(size, np.nan) for name in ('value', 'low', 'high', 'groups', 'games')}
            outlines = {}
            for row in rows:
                if not row.get(metric):
                    continue
                i, j = result['first_index'][row['first_key']], result['second_index'][row['second_key']]
                values = row[metric]
                matrix['value'][i, j] = values['value']
                for name, key in (('low', 'ci_low'), ('high', 'ci_high'), ('groups', 'groups'), ('games', 'games')):
                    matrix[name][i, j] = _nan(values[key])
                if row['relation'] != 'unknown':
                    outlines[(i, j)] = row['relation']
            if not np.isfinite(matrix['value']).any():
                continue
            gap = partnership_gap(rows, metric)
            gap_text = (f'Original minus cross-partnership mean = {gap["gap"]:+.3f} ({gap["original"]} original cells, {gap["cross"]} cross cells; a point difference between separately evaluated cells).'
                        if gap['gap'] is not None else f'No gap: {gap["original"]} original and {gap["cross"]} cross cells are established by the recorded metadata.')
            text = [[_cell_text(matrix, i, j, options.annotate) for j in range(size[1])] for i in range(size[0])]
            table = _matrix_table(result['first_labels'], result['second_labels'], matrix)
            relation = {(r['first'], r['second']): r['relation'] for r in rows}
            table['relation'] = [relation.get((a, b), 'not evaluated') for a, b in zip(table['row'], table['column'])]
            table['opponent'], table['episode'] = result['opponent'], result['episode']
            yield Figure(f'{base}_{slug(metric)}', family, f'{label} vs {result["opponent"]}', f'Rows are first-seat networks and columns are second-seat networks. Solid outlines: original partnerships established by recorded checkpoints; dashed: cross-play. Unmarked cells have no established relation. Grey cells were not evaluated. {gap_text}',
                         [Panel('heatmap', 'Second-seat partner (column)', 'First-seat partner (row)', matrix=matrix['value'], row_labels=result['first_labels'], column_labels=result['second_labels'], annotations=text, outlines=outlines, center=center, value_label=label)],
                         table=table)


BELIEF_ORDER = ('uniform', 'heuristic-likelihood', 'model-likelihood')


def _report_series(sources, modes, read, unit_note=None) -> list[Series]:
    series = []
    for source in sources:
        values = [_nan(read(source, mode)) for mode in modes]
        series.append(Series(source.name, values, key=source.name, n=[unit_note(source, mode) if unit_note else None for mode in modes]))
    return series


def family_belief(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'belief'
    required = 'search_accuracy.py JSON (--search-accuracy PATH)'
    if not data.search:
        yield Omission('belief_quality', family, 'no search_accuracy.py report was supplied. Ordinary PPO training does not record explicit belief marginals.', required)
        return
    caption = ('Positions come from fixed tables of whole hands, so positions within one game are not independent. This measures belief accuracy, not information exchange between partners.')
    get = lambda name: (lambda s, mode: (s.data['summary']['beliefs'].get(mode) or {}).get(name))
    notes = lambda s, mode: f"n={s.data['summary']['positions']} positions"
    panels = [Panel('bar', 'Belief method', 'P(true owner) (0-1)', 'True-owner probability', _report_series(data.search, BELIEF_ORDER, get('p_true'), notes), categories=list(BELIEF_ORDER)),
              Panel('bar', 'Belief method', 'Smoothed log-loss (nats)', 'Log-loss', _report_series(data.search, BELIEF_ORDER, get('log_loss'), notes), categories=list(BELIEF_ORDER))]
    yield Figure('belief_method_comparison', family, 'Belief accuracy by method', caption, panels)
    yield Figure('belief_true_owner_probability', family, 'True-owner probability', caption, [panels[0]])
    yield Figure('belief_log_loss', family, 'Smoothed belief log-loss', caption + ' Log-loss uses the 1% uniform smoothing recorded by the report.', [panels[1]])
    ess_modes = [m for m in BELIEF_ORDER if any((s.data['summary']['beliefs'].get(m) or {}).get('mean_ess') is not None for s in data.search)]
    if ess_modes:
        yield Figure('belief_effective_sample_size', family, 'Effective sample size', 'Uniform marginals are exact and have no sample size, so they are not plotted.', [Panel('bar', 'Belief method', 'Mean effective sample size (worlds)', series=_report_series(data.search, ess_modes, get('mean_ess'), notes), categories=ess_modes)])
    else:
        yield Omission('belief_effective_sample_size', family, 'no report records effective sample size', 'search_accuracy.py JSON `summary.beliefs.*.mean_ess`')
    panels = []
    for metric, ylabel in (('p_true', 'P(true owner) (0-1)'), ('log_loss', 'Smoothed log-loss (nats)')):
        series = []
        for source in data.search:
            rows = [row for row in source.data.get('rows') or () if row.get('tricks_completed') is not None]
            tricks = sorted({row['tricks_completed'] for row in rows})
            for mode in BELIEF_ORDER:
                values, counts = [], []
                for trick in tricks:
                    selected = [row['beliefs'][mode][metric] for row in rows if row['tricks_completed'] == trick and mode in row['beliefs']]
                    values.append(float(np.mean(selected)) if selected else float('nan'))
                    counts.append(_count(len(selected)))
                if tricks:
                    series.append(Series(f'{source.name} - {mode}', values, x=tricks, n=counts, key=f'{mode}', markers=True))
        if series:
            panels.append(Panel('line', 'Tricks completed before the decision (count)', ylabel, series=series))
    if panels:
        yield Figure('belief_by_tricks_completed', family, 'Belief accuracy by tricks completed', 'Means over per-position records, grouped by the recorded `tricks_completed`. ' + caption, panels)
    else:
        yield Omission('belief_by_tricks_completed', family, 'per-position records do not include `tricks_completed`', 'search_accuracy.py JSON `rows[].tricks_completed`')


def family_search(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'search'
    required = 'search_accuracy.py JSON (--search-accuracy PATH)'
    if not data.search:
        yield Omission('search_accuracy', family, 'no search_accuracy.py report was supplied', required)
    else:
        modes = list(dict.fromkeys(mode for source in data.search for mode in source.data['summary'].get('values', {})))
        get = lambda name: (lambda s, mode: (s.data['summary']['values'].get(mode) or {}).get(name))
        caption = 'Value accuracy (versus realized signed marks, one noisy sample per position) is separate from ranking accuracy: a search can predict outcomes well and rank moves poorly. Compare modes with one another, not with zero.'
        notes = lambda s, mode: f"n={s.data['summary']['positions']}"
        yield Figure('search_value_error', family, 'Search value error', caption, [Panel('bar', 'Search mode', 'RMSE (marks)', 'RMSE', _report_series(data.search, modes, get('rmse'), notes), categories=modes), Panel('bar', 'Search mode', 'Bias (marks)', 'Bias', _report_series(data.search, modes, get('bias'), notes), categories=modes)])
        yield Figure('search_played_move_agreement', family, 'Agreement with the played move', "Share of positions where the search's best move equals the table's move; it is not agreement with a reference search.", [Panel('bar', 'Search mode', 'Agreement (0-1)', series=_report_series(data.search, modes, get('agreement'), notes), categories=modes)])
        series = []
        for source in data.search:
            values = source.data['summary'].get('values', {})
            series.append(Series(source.name, [float(values[m]['rmse']) for m in modes if m in values], x=[float(values[m]['mean_seconds']) for m in modes if m in values], n=[m for m in modes if m in values], key=source.name))
        yield Figure('search_accuracy_vs_cost', family, 'Search accuracy versus computation', 'Mean wall-clock seconds per search against value RMSE; points are labelled by mode.' + HARDWARE_NOTE, [Panel('scatter', 'Mean search time (seconds per position)', 'RMSE (marks)', series=series)])
        ranked = [s for s in data.search if s.data['summary'].get('ranking')]
        if not ranked:
            for figure_id in ('search_top1_agreement', 'search_rank_concordance', 'search_ranking_regret'):
                yield Omission(figure_id, family, 'reports were produced without --rank-check', 'search_accuracy.py --rank-check')
        else:
            pairs = list(dict.fromkeys(pair for s in ranked for pair in s.data['summary']['ranking']))
            rank = lambda name: (lambda s, pair: (s.data['summary']['ranking'].get(pair) or {}).get(name))
            reference = 'Pairs compare a candidate with a reference search (noise pair: independent replicate of the reference). Reference search is approximate, not an oracle.'
            yield Figure('search_top1_agreement', family, 'Top-1 move agreement with the reference search', reference, [Panel('bar', 'Candidate vs reference', 'Top-1 agreement (0-1)', series=_report_series(ranked, pairs, rank('top1')), categories=pairs)])
            yield Figure('search_rank_concordance', family, 'Pairwise ranking concordance', reference, [Panel('bar', 'Candidate vs reference', 'Concordant move pairs (0-1)', series=_report_series(ranked, pairs, rank('concordance')), categories=pairs)])
            panels = []
            for unit in ('marks', 'points'):
                unit_pairs = [p for p in pairs if any((s.data['summary']['ranking'].get(p) or {}).get('regret_units') == unit for s in ranked)]
                if unit_pairs:
                    panels.append(Panel('bar', 'Candidate vs reference', f'Regret ({unit}, reference-measured)', series=_report_series(ranked, unit_pairs, rank('regret')), categories=unit_pairs))
            yield Figure('search_ranking_regret', family, 'Approximate search ranking regret', 'Cost of the candidate\'s top move as measured by the reference search; approximate unless an exact oracle certifies it.', panels)
    has_search = any('rollout.search_seconds' in frame for frame in data.training.values())
    if not has_search:
        for figure_id in ('search_training_overhead', 'search_training_trajectories'):
            yield Omission(figure_id, family, 'no training run used --search-fraction', 'progress.jsonl `metrics.search_*`, `rollout_performance.search_*`')
        return
    worker, missing = _training_series(data, 'rollout.search_seconds', options, suffix=' - worker seconds')
    wall, _ = _training_series(data, 'rollout.search_wall_seconds', options, dash='dashed', suffix=' - wall seconds')
    yield Figure('search_training_overhead', family, 'Search overhead during PPO training', 'Worker-seconds sum over processes; wall seconds are elapsed time, so they are not interchangeable.' + HARDWARE_NOTE, [Panel('line', X_COLUMNS[options.x_axis][1], 'Search time per update (seconds)', series=worker + wall)], missing)
    panels = []
    for column, ylabel in (('ppo.search_loss', 'Search loss'), ('ppo.search_disagreement', 'Disagreement with policy (0-1)'), ('ppo.search_abs_advantage', 'Mean |advantage| (points)')):
        series, _ = _training_series(data, column, options)
        if series:
            panels.append(Panel('line', X_COLUMNS[options.x_axis][1], ylabel, series=series))
    yield (Figure('search_training_trajectories', family, 'Search targets during training', 'Search-target disagreement and search-loss trajectories.', panels) if panels else Omission('search_training_trajectories', family, 'search diagnostics are not recorded', 'progress.jsonl `metrics.search_*`'))


def family_resources(data: Dataset, options: Options) -> Iterator[Figure | Omission]:
    family = 'resources'
    yield _line_figure(data, options, family, 'resources_collection_hands_per_second', 'Collection throughput', 'collection_hands_per_second', 'Hands per second (wall clock)', 'Rollout collection throughput.' + HARDWARE_NOTE)
    yield _line_figure(data, options, family, 'resources_decisions_per_second', 'Training decisions per second', 'collection_decisions_per_second', 'Decisions per second (wall clock)', 'Rollout collection decisions.' + HARDWARE_NOTE)
    categories = ['collection_seconds', 'optimization_seconds', 'evaluation_seconds', 'checkpoint_seconds']
    series, unavailable = [], {}
    for name, frame in data.training.items():
        values = [_total(frame, c) for c in categories]
        if all(v is None for v in values):
            unavailable[name] = 'timing fields are not recorded'
        else:
            series.append(Series(name, [_nan(v) / 3600 for v in values], n=[f'{(v or 0) / 3600:.2f} h' for v in values], key=name))
    if series:
        yield Figure('resources_time_split', family, 'Training time by activity', 'Wall-clock seconds summed over updates.' + HARDWARE_NOTE, [Panel('bar', 'Activity', 'Hours (wall clock)', series=series, categories=[c.removesuffix('_seconds') for c in categories])], unavailable)
    else:
        yield Omission('resources_time_split', family, 'timing fields are not recorded', 'progress.jsonl `collection_seconds`, `optimization_seconds`')
    evaluated = Dataset(experiments=data.experiments, training={n: f[f['evaluation_seconds'] > 0] for n, f in data.training.items() if 'evaluation_seconds' in f and (f['evaluation_seconds'] > 0).any()})
    yield (_line_figure(evaluated, options, family, 'resources_evaluation_seconds', 'Evaluation time', 'evaluation_seconds', 'Evaluation time (seconds, wall clock)', 'Only updates that ran evaluations.' + HARDWARE_NOTE)
           if evaluated.training else Omission('resources_evaluation_seconds', family, 'no evaluation time was recorded', 'progress.jsonl `evaluation_seconds`'))
    worker, missing = _training_series(data, 'rollout.search_seconds', options, suffix=' - worker seconds')
    wall, _ = _training_series(data, 'rollout.search_wall_seconds', options, dash='dashed', suffix=' - wall seconds')
    yield (Figure('resources_search_worker_vs_wall', family, 'Search worker versus wall time', 'Worker-seconds sum over processes and can exceed wall time when searches run in parallel.' + HARDWARE_NOTE, [Panel('line', X_COLUMNS[options.x_axis][1], 'Search time per update (seconds)', series=worker + wall)], missing)
           if worker or wall else Omission('resources_search_worker_vs_wall', family, 'no training run used --search-fraction', 'progress.jsonl `rollout_performance.search_*`'))
    memory, missing = _training_series(data, 'rollout.worker_peak_rss_bytes', options, transform=lambda y: y / 2**20)
    yield (Figure('resources_worker_peak_memory', family, 'Worker peak memory', 'Largest worker resident set size reported in each rollout.' + HARDWARE_NOTE, [Panel('line', X_COLUMNS[options.x_axis][1], 'Peak RSS (MiB)', series=memory)], missing)
           if memory else Omission('resources_worker_peak_memory', family, 'worker memory was not recorded', 'progress.jsonl `rollout_performance.worker_peak_rss_bytes`'))


BUILDERS = {'learning': family_learning, 'ppo': family_ppo, 'critic': family_critic, 'bidding': family_bidding, 'match-bidding': family_match_bidding, 'opponents': family_opponents,
            'evaluation': family_evaluation, 'contracts': family_contracts, 'round-robin': family_round_robin, 'partners': family_partners, 'belief': family_belief, 'search': family_search,
            'resources': family_resources}
assert set(BUILDERS) == set(FAMILIES)


def build_figures(data: Dataset, families: Iterable[str], options: Options, patterns: Iterable[str] = ()) -> tuple[list[Figure], list[Omission]]:
    """Specifications for the selected families; `patterns` optionally filters figure ids."""
    patterns = list(patterns)

    def selected(figure_id):
        return not patterns or any(fnmatch.fnmatch(figure_id, p) or p in figure_id for p in patterns)

    figures, omissions, seen = [], [], set()
    for family in families:
        for item in BUILDERS[family](data, options):
            base, suffix = item.id, 1
            while item.id in seen:  # Distinct names can share a slug; keep every figure.
                suffix += 1
                item.id = f'{base}_{suffix}'
            seen.add(item.id)
            if selected(item.id):
                (figures if isinstance(item, Figure) else omissions).append(item)
    return figures, omissions


class Styles:
    """Stable color and dash for each identity; color follows the entity across figures."""

    def __init__(self, names: Iterable[str]):
        self.order: dict[str, int] = {}
        for name in names:
            self.index(name)

    def index(self, key: str) -> int:
        return self.order.setdefault(key, len(self.order))

    def color(self, key: str) -> str:
        return PALETTE[self.index(key) % len(PALETTE)]

    def pattern(self, key: str) -> int:
        return self.index(key) // len(PALETTE) % len(DASHES)

    def variant(self, series: Series) -> int:
        """Dash or hatch: an explicit per-series variant, else the entity's overflow pattern."""
        return (series.pattern % len(DASHES)) if series.pattern is not None else self.pattern(series.key or series.label)


def _style_axes(ax) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, axis='y', color='#e4e3df', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _compact_tick(value: float, _=None) -> str:
    size = abs(value)
    return f'{value / 1e6:g}M' if size >= 1e6 else f'{value / 1e3:g}k' if size >= 1e3 else f'{value:g}'


def _draw_line(ax, panel: Panel, styles: Styles) -> None:
    for s in panel.series:
        key = s.key or s.label
        linestyle = MPL_DASHES[s.dash or DASHES[styles.variant(s)]]
        ax.plot(s.x, s.y, color=styles.color(key), linestyle=linestyle, linewidth=s.width, label=s.label, marker='o' if s.markers else None, markersize=3)
        if s.low is not None:
            ax.fill_between(s.x, s.low, s.high, color=styles.color(key), alpha=0.15, linewidth=0)


def _draw_area(ax, panel: Panel, styles: Styles) -> None:
    ax.stackplot(panel.series[0].x, *[s.y for s in panel.series], labels=[s.label for s in panel.series], colors=[styles.color(s.key or s.label) for s in panel.series], edgecolor='white', linewidth=0.6)
    ax.set_ylim(0, 1)


def _draw_scatter(ax, panel: Panel, styles: Styles) -> None:
    for s in panel.series:
        ax.scatter(s.x, s.y, color=styles.color(s.key or s.label), label=s.label, s=30, zorder=3)
        for x, y, text in zip(s.x, s.y, s.n or ()):
            ax.annotate(text, (x, y), xytext=(4, 3), textcoords='offset points', fontsize=7)


def _draw_bar(ax, panel: Panel, styles: Styles) -> None:
    positions = np.arange(len(panel.categories))
    count = len(panel.series) if panel.dodge else 1
    width = 0.8 / count
    total_bars = len(panel.categories) * count
    for index, s in enumerate(panel.series):
        offset = (index - (count - 1) / 2) * width if count > 1 else 0
        key = s.key or s.label
        y = np.array(s.y, dtype=float)
        hatch = HATCHES[styles.variant(s) % len(HATCHES)] or None
        bars = ax.bar(positions + offset, np.nan_to_num(y), width * 0.92, color=styles.color(key), hatch=hatch, label=s.label, zorder=2)
        if s.low is not None:
            low, high = np.array(s.low, dtype=float), np.array(s.high, dtype=float)
            valid = np.isfinite(low) & np.isfinite(high) & np.isfinite(y)
            ax.errorbar(positions[valid] + offset, y[valid], yerr=[y[valid] - low[valid], high[valid] - y[valid]], fmt='none', ecolor='#3a3a38', elinewidth=0.9, capsize=2, zorder=3)
        for i, bar in enumerate(bars):
            if not np.isfinite(y[i]):
                if panel.dodge or len(panel.series) == 1:
                    ax.annotate('n/a', (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=6.5, color='#6d6c68')
            elif s.n and s.n[i] and total_bars <= LABEL_BARS_UP_TO:
                top = max(y[i], np.nan_to_num(s.high[i]) if s.high is not None and s.high[i] == s.high[i] else y[i])
                ax.annotate(s.n[i], (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=6, rotation=90 if count > 2 or (count > 1 and len(s.n[i]) > 10) else 0)
    ax.set_xticks(positions, panel.categories, rotation=20 if len(panel.categories) > 3 or max(len(c) for c in panel.categories) > 14 else 0, ha='right' if len(panel.categories) > 3 else 'center')
    ax.axhline(0, color='#55544f', linewidth=0.7, zorder=1)


def _heatmap_scale(panel: Panel):
    finite = panel.matrix[np.isfinite(panel.matrix)]
    if panel.center is None:
        return 'Blues', None, float(finite.min()), float(finite.max())
    reach = max(float(np.abs(finite - panel.center).max()), 1e-6)
    return 'RdBu', TwoSlopeNorm(panel.center, panel.center - reach, panel.center + reach), panel.center - reach, panel.center + reach


def _draw_heatmap(fig, ax, panel: Panel) -> None:
    cmap_name, norm, low, high = _heatmap_scale(panel)
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad(MISSING_CELL)
    masked = np.ma.masked_invalid(panel.matrix)
    image = ax.imshow(masked, cmap=cmap, norm=norm, vmin=None if norm else low, vmax=None if norm else high, aspect='auto')
    ax.set_xticks(range(len(panel.column_labels)), panel.column_labels, rotation=35, ha='right')
    ax.set_yticks(range(len(panel.row_labels)), panel.row_labels)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    for i, row in enumerate(panel.annotations or ()):
        for j, text in enumerate(row):
            if text:
                intensity = (panel.matrix[i, j] - low) / (high - low) if high > low else 0.5
                dark_cell = abs(intensity - 0.5) > 0.3 if panel.center is not None else intensity > 0.6
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color='white' if dark_cell else '#1a1a19')
    for (i, j), relation in panel.outlines.items():
        ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor='#000000', linewidth=1.6, linestyle='-' if relation == 'original' else '--'))
    fig.colorbar(image, ax=ax, label=panel.value_label, shrink=0.8)


matplotlib.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.titlesize': 11,
                            'svg.hashsalt': 'sc42', 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})


def render_static(figure: Figure, styles: Styles, paths: dict[str, Path]) -> None:
    """Write PNG/SVG/PDF for the formats keyed in `paths`; output bytes are reproducible."""
    panels = figure.panels
    heat = panels[0].kind == 'heatmap'
    if heat:
        size = max(5.0, 1.1 * len(panels[0].column_labels) + 2.6), max(3.6, 0.6 * len(panels[0].row_labels) + 2.0)
    else:
        size = (min(4.6 * len(panels), 13.0) if len(panels) > 1 else 6.8), 4.2
    fig, axes = plt.subplots(1, len(panels), figsize=size, layout='constrained', squeeze=False)
    drawers = {'line': _draw_line, 'area': _draw_area, 'scatter': _draw_scatter, 'bar': _draw_bar}
    for ax, panel in zip(axes[0], panels):
        if panel.kind == 'heatmap':
            _draw_heatmap(fig, ax, panel)
        else:
            _style_axes(ax)
            drawers[panel.kind](ax, panel, styles)
            if panel.kind in ('line', 'area') and ax.get_xlim()[1] >= 1e4:
                ax.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        if panel.title:
            ax.set_title(panel.title)
    labels, handles = [], []
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
    if len(labels) > 1:
        fig.legend(handles, labels, loc='outside lower center', ncol=min(3, len(labels)), frameon=False)
    fig.suptitle(figure.title)
    metadata = dict(png={'Software': None}, svg={'Date': None, 'Creator': None}, pdf={'CreationDate': None, 'ModDate': None, 'Producer': None, 'Creator': None})
    for extension, path in paths.items():
        fig.savefig(path, dpi=200 if extension == 'png' else None, metadata=metadata[extension])
    plt.close(fig)


def _plotly_series(panel: Panel, styles: Styles, go, row: int, column: int, figure, shown: set) -> None:
    for s in panel.series:
        key = s.key or s.label
        color = styles.color(key)
        legend = dict(name=s.label, legendgroup=s.label, showlegend=s.label not in shown)
        shown.add(s.label)
        at = dict(row=row, col=column)
        if panel.kind == 'bar':
            error = None
            if s.low is not None:
                y, low, high = (np.array(values, dtype=float) for values in (s.y, s.low, s.high))
                error = dict(type='data', array=np.nan_to_num(high - y).tolist(), arrayminus=np.nan_to_num(y - low).tolist(), color='#3a3a38', thickness=1)
            figure.add_trace(go.Bar(x=panel.categories, y=[None if v != v else v for v in s.y], error_y=error, text=s.n, marker=dict(color=color, pattern=dict(shape=PLOTLY_HATCHES[HATCHES[styles.variant(s) % len(HATCHES)]])), **legend), **at)
        elif panel.kind == 'area':
            figure.add_trace(go.Scatter(x=s.x, y=s.y, mode='lines', stackgroup='one', line=dict(color=color, width=0.5), **legend), **at)
        elif panel.kind == 'scatter':
            figure.add_trace(go.Scatter(x=s.x, y=s.y, mode='markers+text', text=s.n, textposition='top center', marker=dict(color=color, size=8), **legend), **at)
        else:
            line = dict(color=color, width=s.width, dash={'solid': 'solid', 'dashed': 'dash', 'dotted': 'dot', 'dashdot': 'dashdot'}[s.dash or DASHES[styles.variant(s)]])
            y = [None if v != v else v for v in s.y]
            x = [None if v != v else v for v in s.x]
            if s.low is not None:
                low, high = [None if v != v else v for v in s.low], [None if v != v else v for v in s.high]
                figure.add_trace(go.Scatter(x=x, y=high, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False, legendgroup=s.label), **at)
                figure.add_trace(go.Scatter(x=x, y=low, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=_with_alpha(color, 0.15), hoverinfo='skip', showlegend=False, legendgroup=s.label), **at)
            figure.add_trace(go.Scatter(x=x, y=y, mode='lines+markers' if s.markers else 'lines', line=line, marker=dict(size=5), text=s.n, **legend), **at)


def _with_alpha(color: str, alpha: float) -> str:
    return f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha})'


def render_interactive(figure: Figure, styles: Styles, path: Path) -> None:
    """Self-contained chart; a shared plotly.min.js sits beside it, so no CDN is used."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = figure.panels
    plotly_figure = make_subplots(rows=1, cols=len(panels), subplot_titles=[p.title for p in panels] if any(p.title for p in panels) else None)
    shown: set[str] = set()
    for column, panel in enumerate(panels, 1):
        if panel.kind == 'heatmap':
            _, _, low, high = _heatmap_scale(panel)
            text = [[cell.replace('\n', '<br>') for cell in row] for row in panel.annotations] if panel.annotations else None
            colorscale = 'RdBu' if panel.center is not None else 'Blues'
            matrix = [[None if v != v else float(v) for v in row] for row in panel.matrix]
            plotly_figure.add_trace(go.Heatmap(z=matrix, x=panel.column_labels, y=panel.row_labels, text=text, texttemplate='%{text}', colorscale=colorscale, zmid=panel.center, zmin=None if panel.center is not None else low,
                                               zmax=None if panel.center is not None else high, colorbar=dict(title=panel.value_label), hoverongaps=False), row=1, col=column)
            plotly_figure.update_yaxes(autorange='reversed', row=1, col=column)
            for (i, j), relation in panel.outlines.items():
                plotly_figure.add_shape(type='rect', x0=j - 0.48, x1=j + 0.48, y0=i - 0.48, y1=i + 0.48, line=dict(color='black', width=2, dash='solid' if relation == 'original' else 'dash'), row=1, col=column)
        else:
            _plotly_series(panel, styles, go, 1, column, plotly_figure, shown)
        plotly_figure.update_xaxes(title_text=panel.xlabel, row=1, col=column)
        plotly_figure.update_yaxes(title_text=panel.ylabel, row=1, col=column)
    plotly_figure.update_layout(title=figure.title, template='simple_white', barmode='overlay' if panels[0].kind == 'bar' and not panels[0].dodge else 'group', plot_bgcolor='white' if panels[0].kind != 'heatmap' else MISSING_CELL, font=dict(family='DejaVu Sans, Arial, sans-serif', size=12), height=520)
    plotly_figure.write_html(path, include_plotlyjs='directory', full_html=True, div_id=figure.id, config=dict(responsive=True, displaylogo=False))
