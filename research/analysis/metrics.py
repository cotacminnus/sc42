"""Comparisons and aggregates over normalized tables.

Nothing here resamples hands or recomputes an arena interval: uncertainty comes
from the intervals the arena recorded on complete duplicate groups. Aggregates
pool recorded counts, so denominators stay visible.
"""
from fractions import Fraction
import json

import numpy as np
import pandas as pd

from .normalize import Dataset

X_COLUMNS = {'hands': ('hands', 'Simulated hands (cumulative count)'),
             'matches': ('matches', 'Completed matches (cumulative count)'),
             'updates': ('updates', 'PPO updates (cumulative count)'),
             'runtime': ('runtime_cumulative', 'Cumulative training runtime (seconds)')}
_SEED_INDEPENDENT = {'seed', 'device', 'threads', 'torch_threads', 'eval_threads', 'output_dir', 'log_dir'}


def _shown(value) -> str:
    return 'not recorded' if value is None else json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def compare_experiments(data: Dataset) -> tuple[pd.DataFrame, list[str]]:
    """One row per compared setting, one column per experiment, and a warning per difference."""
    names = data.training_runs()
    meta = data.experiments
    fields = (('Reward objective', lambda m: m['reward_objective']),
              ('Critic mode', lambda m: m['critic_mode']),
              ('Return estimator', lambda m: m['return_estimator']),
              ('Trained phases', lambda m: m['trained_phases']),
              ('Opponent distribution', lambda m: m['opponent_configuration']),
              ('Evaluation opponents', lambda m: m['eval_opponents']),
              ('Evaluation schedule', lambda m: dict(every_hands=m['eval_every_hands'], deals=m['eval_deals'],
                                                     match_pairs=m['eval_match_pairs'])),
              ('Evaluation seeds', lambda m: m['eval_seeds']),
              ('Network architecture', lambda m: dict(hidden_size=m['hidden_size'], partner_mode=m['partner_mode'])),
              ('Rollout size', lambda m: m['rollout']),
              ('Training seed', lambda m: m['seed']))
    rows, warnings = [], []
    for label, read in fields:
        values = {name: read(meta[name]) for name in names}
        differs = len({_shown(value) for value in values.values()}) > 1
        rows.append(dict(setting=label, differs=differs, **{name: _shown(value) for name, value in values.items()}))
        if differs and label != 'Training seed':
            warnings.append(f'Compared training runs differ in {label.lower()}: '
                            + '; '.join(f'{name}={_shown(value)}' for name, value in values.items()))
    budgets = {name: meta[name]['final_hands'] for name in names}
    known = [value for value in budgets.values() if value]
    unequal = bool(known) and (max(known) - min(known)) / max(known) > 0.01
    rows.append(dict(setting='Training budget (final hands)', differs=unequal, **{name: _shown(value) for name, value in budgets.items()}))
    if len(names) > 1 and unequal:
        warnings.append('Compared training runs differ in training budget (final hands): '
                        + '; '.join(f'{name}={value:,.0f}' for name, value in budgets.items() if value is not None)
                        + '. Sample-efficiency comparisons need equal simulated-hand budgets.')
    rows.append(dict(setting='Final matches', differs=False, **{name: _shown(meta[name]['final_matches']) for name in names}))
    if len(names) < 2:
        warnings = []
    reports = [name for name, m in meta.items() if m['kind'] == 'evaluation']
    for label, key in (('Report schema', 'schema'), ('Episode types', 'episodes'), ('Report evaluation seeds', 'eval_seeds'), ('Report opponents', 'opponents'), ('Checkpoint', 'checkpoint_id')):
        if reports:
            values = {name: _shown(meta[name][key]) for name in reports}
            rows.append(dict(setting=label, differs=len(set(values.values())) > 1, **values))
    if len(reports) > 1:
        for label, key in (('seeds', 'eval_seeds'), ('episode types', 'episodes')):
            distinct = {_shown(meta[name][key]) for name in reports}
            if len(distinct) > 1:
                warnings.append(f'Compared evaluation reports differ in {label}: '
                                + '; '.join(f'{name}={_shown(meta[name][key])}' for name in reports))
    return pd.DataFrame(rows), warnings


def seed_groups(data: Dataset) -> list[list[str]]:
    """Runs whose configurations differ only in seed; groups of two or more only."""
    groups: dict[str, list[str]] = {}
    for name in data.training_runs():
        config = {k: v for k, v in data.meta(name)['config'].items() if k not in _SEED_INDEPENDENT}
        if config:
            groups.setdefault(json.dumps(config, sort_keys=True, default=str), []).append(name)
    return [names for names in groups.values() if len({data.meta(n)['seed'] for n in names}) > 1]


def aggregate_by_ordinal(trajectories: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Mean and between-seed standard deviation at each evaluation ordinal all seeds reached.

    Evaluations land at slightly different hand counts per seed, so seeds align
    by ordinal rather than by interpolating onto a common axis; x is the mean
    position of that ordinal.
    """
    length = min(len(x) for x, _ in trajectories)
    xs = np.array([x[:length] for x, _ in trajectories])
    ys = np.array([y[:length] for _, y in trajectories])
    return dict(x=xs.mean(axis=0), y=ys.mean(axis=0), sd=ys.std(axis=0, ddof=1), seeds=len(trajectories), points=length)


def pooled_windows(frame: pd.DataFrame, x_column: str, numerator: str | np.ndarray, denominator: str, windows: int = 20) -> dict | None:
    """Pooled rate over equal windows of updates that recorded a denominator.

    Each window sums counts, so a rollout with few decisions weighs less. x is
    the window's final position. Returns None if the columns were not recorded.
    """
    if denominator not in frame or x_column not in frame:
        return None
    top = frame[numerator] if isinstance(numerator, str) else pd.Series(numerator, index=frame.index)
    bottom = frame[denominator]
    keep = bottom.notna() & (bottom > 0) & top.notna() & frame[x_column].notna()
    if not keep.any():
        return None
    x, numerators, denominators = frame.loc[keep, x_column].to_numpy(), top[keep].to_numpy(), bottom[keep].to_numpy()
    edges = np.array_split(np.arange(len(x)), min(windows, len(x)))
    return dict(x=[float(x[e[-1]]) for e in edges],
                y=[float(numerators[e].sum() / denominators[e].sum()) for e in edges],
                n=[float(denominators[e].sum()) for e in edges])


def parse_fraction(text) -> float | None:
    try:
        return float(Fraction(str(text)))
    except (ValueError, ZeroDivisionError):
        return None


def _union(parent: dict, a, b) -> None:
    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    parent[find(a)] = find(b)


def _team_key(team: dict) -> tuple:
    first, second = team['first_key'], team['second_key']
    return (first, None if second == first else second)


def _evaluation_values(data: Dataset, experiment: str, metric: str) -> dict:
    """Recorded (value, low, high, groups, games) by policy for one report."""
    frame = data.evaluation
    if frame.empty:
        return {}
    rows = frame[(frame['experiment'] == experiment) & (frame['metric'] == metric)]
    return {row['policy']: {key: row[key] for key in ('value', 'ci_low', 'ci_high', 'groups', 'games', 'eval_seed')}
            for _, row in rows.iterrows()}


def round_robin(data: Dataset, metric: str) -> dict | None:
    """Agent matrix for one metric, from composed reports; untested cells stay NaN.

    The caller passes one episode's metric name, so hand and match results
    never share a matrix. Agents are identified by strategy and checkpoint hash
    (plus recorded parameter identity), so copies of one model merge rather
    than count as separate evidence.
    """
    parent: dict = {}
    entries = []
    for experiment, pairing in data.pairings.items():
        values = _evaluation_values(data, experiment, metric)
        if not values:
            continue
        a, b = pairing['teams']['A'], pairing['teams']['B']
        key_a, key_b = _team_key(a), _team_key(b)
        identical = pairing['comparisons'].get('phase_policies_identical') or {}
        if identical and all(identical.values()) and a['second_key'] is None and b['second_key'] is None:
            _union(parent, key_a, key_b)
        for policy, row_team, column_team in (('A', key_a, key_b), ('B', key_b, key_a)):
            if policy in values:
                entries.append((row_team, column_team, experiment, values[policy], a if policy == 'A' else b))
    if not entries:
        return None

    def root(key):
        while parent.get(key, key) != key:
            key = parent[key]
        return key

    aliases: dict = {}
    for row_team, column_team, experiment, _, team in entries:
        aliases.setdefault(root(row_team), set()).add(team['label'])
    for experiment, pairing in data.pairings.items():
        for team in pairing['teams'].values():
            aliases.setdefault(root(_team_key(team)), set()).add(team['label'])
    keys = sorted({root(key) for key in aliases if any(root(e[0]) == key or root(e[1]) == key for e in entries)},
                  key=lambda key: sorted(aliases[key])[0])
    labels = [sorted(aliases[key])[0] for key in keys]
    index = {key: position for position, key in enumerate(keys)}
    size = len(keys)
    matrix = {name: np.full((size, size), np.nan) for name in ('value', 'low', 'high', 'groups', 'games')}
    cells: dict = {}
    repeated = []
    for row_team, column_team, experiment, values, _ in entries:
        cell = (index[root(row_team)], index[root(column_team)])
        if cell in cells and cells[cell][1]['games'] >= values['games']:
            repeated.append(experiment)
            continue
        if cell in cells:
            repeated.append(cells[cell][0])
        cells[cell] = (experiment, values)
    for (row, column), (_, values) in cells.items():
        matrix['value'][row, column] = values['value']
        for name, key in (('low', 'ci_low'), ('high', 'ci_high'), ('groups', 'groups'), ('games', 'games')):
            if values[key] is not None and values[key] == values[key]:
                matrix[name][row, column] = values[key]
    return dict(labels=labels, **matrix, repeated=sorted(set(repeated)),
                duplicates={labels[index[k]]: sorted(aliases[k]) for k in keys if len(aliases[k]) > 1},
                reports=len({experiment for experiment, _ in cells.values()}))


def _single_checkpoint(checkpoints: list[dict]) -> str | None:
    return checkpoints[0].get('path', '').replace('\\', '/') if len(checkpoints) == 1 else None


def partnership_relation(first: dict, second: dict | None) -> str:
    """'original', 'cross' or 'unknown'; only recorded checkpoint paths and hashes decide.

    Original: partner-1/X with partner-2/X of one paired run, or one non-paired
    network at both seats. Cross: networks from different training runs, or a
    paired network beside itself. Anything else is unknown.
    """
    a = _single_checkpoint(first)
    if a is None:
        return 'unknown'
    b = a if second is None else _single_checkpoint(second)
    if b is None:
        return 'unknown'
    run_a, _, tail_a = a.rpartition('/checkpoints/')
    run_b, _, tail_b = b.rpartition('/checkpoints/')
    if not run_a or not run_b:
        return 'unknown'
    partner = tail_a.startswith('partner-')
    if a == b or (first and second and first[0].get('sha256') == second[0].get('sha256')):
        return 'cross' if partner else 'original'
    if run_a != run_b:
        return 'cross'
    if partner and tail_b.startswith('partner-') and tail_a.split('/', 1)[1:] == tail_b.split('/', 1)[1:] and tail_a != tail_b:
        return 'original'
    return 'unknown'


def partner_matrices(data: Dataset) -> list[dict]:
    """Compatibility matrices, one per (opponent, episode): a shared opponent keeps cells comparable."""
    entries: dict = {}
    for experiment, pairing in data.pairings.items():
        for policy, opposing in (('A', 'B'), ('B', 'A')):
            team, other = pairing['teams'][policy], pairing['teams'][opposing]
            mixed = team['second_key'] is not None and team['second_key'] != team['first_key']
            frame = data.evaluation[(data.evaluation['experiment'] == experiment) & (data.evaluation['policy'] == policy)]
            if frame.empty:
                continue
            episode = frame['episode'].iloc[0]
            row = dict(experiment=experiment, first=team['first_label'], second=team['second_label'] or team['first_label'],
                       first_key=team['first_key'], second_key=team['second_key'] or team['first_key'], mixed=mixed,
                       relation=partnership_relation(team['first_checkpoints'], team['second_checkpoints'] if team['second_key'] is not None else None))
            for metric in ('mean_signed_marks_per_hand', 'match_win_rate'):
                row[metric] = _evaluation_values(data, experiment, metric).get(policy)
            entries.setdefault((other['label'], episode), []).append(row)
    results = []
    for (opponent, episode), rows in sorted(entries.items(), key=lambda item: item[0]):
        if not any(r['mixed'] for r in rows):
            continue
        first = sorted({(r['first'], r['first_key']) for r in rows}, key=lambda item: item[0])
        second = sorted({(r['second'], r['second_key']) for r in rows}, key=lambda item: item[0])
        results.append(dict(opponent=opponent, episode=episode, rows=rows,
                            first_labels=[f[0] for f in first], second_labels=[s[0] for s in second],
                            first_index={f[1]: i for i, f in enumerate(first)},
                            second_index={s[1]: i for i, s in enumerate(second)}))
    return results


def partnership_gap(rows: list[dict], metric: str) -> dict:
    """Mean of original cells minus mean of cross cells; a point difference between evaluations."""
    values = {relation: [r[metric]['value'] for r in rows if r['relation'] == relation and r.get(metric)]
              for relation in ('original', 'cross', 'unknown')}
    gap = (float(np.mean(values['original']) - np.mean(values['cross']))
           if values['original'] and values['cross'] else None)
    return dict(gap=gap, original=len(values['original']), cross=len(values['cross']), unknown=len(values['unknown']))
