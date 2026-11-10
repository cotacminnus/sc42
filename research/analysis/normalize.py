"""Turn loaded SC42 artifacts into the normalized tables behind every figure.

Absent values stay absent: a metric that a record does not carry is NaN in the
tables and is never converted to zero. Original values are kept unmodified,
with the source file and line of each training record.
"""
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from .loader import EvaluationSource, SearchSource, TrainingSource

PHASES = ('bid', 'trump', 'play')
TEAMS = ('learner', 'opponent')
BID_CATEGORIES = ('30', '31-41', '42', '2+ marks')
CALL_CATEGORIES = ('pass',) + BID_CATEGORIES
SCORE_STATES = ('leading', 'tied', 'trailing')
LEGACY_METRICS = ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'gradient_norm')
TOP_LEVEL_FIELDS = ('hands', 'matches', 'updates', 'optimizer_steps', 'training_samples', 'play_samples',
                    'runtime_seconds', 'hands_per_second', 'collection_seconds', 'optimization_seconds',
                    'evaluation_seconds', 'checkpoint_seconds', 'update_seconds', 'collection_hands_per_second',
                    'collection_decisions_per_second', 'mean_hands_per_match', 'match_win_rate', 'hand_win_rate',
                    'mean_signed_marks_per_hand', 'mean_terminal_reward', 'rollout_matches', 'rollout_hands',
                    'playing_anchor_kl', 'selection_score', 'target_marks')
REWARD_UNITS = {'hand-marks': 'signed marks per hand', 'match-win': 'terminal reward per match (+1/-1)'}
# Value-function outputs that do not exist for critic-free PPO, whatever an old log wrote.
CRITIC_ONLY_SUFFIXES = ('value_loss', 'explained_variance', 'value_mean')


@dataclass
class Dataset:
    experiments: dict[str, dict] = field(default_factory=dict)  # Metadata by experiment, in input order.
    training: dict[str, pd.DataFrame] = field(default_factory=dict)  # Columnar: one row per update.
    evaluation: pd.DataFrame = field(default_factory=pd.DataFrame)  # Long: one row per metric and policy.
    contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    timing: pd.DataFrame = field(default_factory=pd.DataFrame)
    roles: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairings: dict[str, dict] = field(default_factory=dict)  # Composed-report teams by experiment.
    search: list[SearchSource] = field(default_factory=list)
    by_match_score: dict[str, dict] = field(default_factory=dict)  # experiment -> team -> 'a-b' -> counts.
    assignments: dict[str, dict] = field(default_factory=dict)  # experiment -> opponent -> identity -> episodes.
    opponent_configs: dict[str, list] = field(default_factory=dict)  # experiment -> [(first hands, config)].
    warnings: list[str] = field(default_factory=list)
    pending: dict[str, list[dict]] = field(default_factory=lambda: dict(evaluation=[], contracts=[], timing=[], roles=[]))

    def training_runs(self) -> list[str]:
        return list(self.training)

    def meta(self, experiment: str) -> dict:
        return self.experiments[experiment]


class _Columns:
    """Sparse numeric columns; memory stays small on logs with tens of thousands of updates."""

    def __init__(self):
        self.rows = 0
        self.values: dict[str, tuple[array, array]] = {}

    def add(self, key: str, value) -> None:
        if isinstance(value, (bool, int, float)) and value == value:  # NaN is not a measurement.
            indices, numbers = self.values.setdefault(key, (array('q'), array('d')))
            indices.append(self.rows - 1)
            numbers.append(float(value))

    def add_mapping(self, prefix: str, mapping) -> None:
        for key, value in (mapping or {}).items():
            self.add(prefix + str(key), value)

    def frame(self, keep: np.ndarray) -> pd.DataFrame:
        data = {}
        for key, (indices, numbers) in self.values.items():
            column = np.full(self.rows, np.nan)
            column[np.frombuffer(indices, dtype=np.int64)] = np.frombuffer(numbers, dtype=np.float64)
            data[key] = column[keep]
        return pd.DataFrame(data)


def _flatten_learner(columns: _Columns, prefix: str, metrics, value_diagnostics, phase_samples, critic_mode) -> None:
    for key, value in (metrics or {}).items():
        if not (critic_mode == 'none' and key.endswith(CRITIC_ONLY_SUFFIXES)):
            columns.add(f'{prefix}ppo.{key}', value)
    columns.add_mapping(f'{prefix}samples.', phase_samples)
    for key, value in (value_diagnostics or {}).items():
        if isinstance(value, dict):
            for name, number in value.items():
                if not (critic_mode == 'none' and name.endswith(CRITIC_ONLY_SUFFIXES)):
                    columns.add(f'{prefix}value.{key}.{name}', number)
        elif critic_mode != 'none':
            columns.add(f'{prefix}value.{key}', value)


def _flatten_bidding(columns: _Columns, row: dict, score_totals: dict) -> None:
    for team, counts in (row.get('rollout_bids') or {}).items():
        columns.add_mapping(f'contracts_won.{team}.', counts)
    for team, stats in (row.get('bidding') or {}).items():
        prefix = f'bidding.{team}.'
        decisions, forced = stats.get('decisions'), stats.get('forced_calls')
        columns.add(prefix + 'decisions', decisions)
        columns.add(prefix + 'forced_calls', forced)
        columns.add_mapping(prefix + 'calls.', stats.get('calls'))
        # Rates are logged with denominators derivable from counts, so aggregation can pool counts.
        if decisions is not None and forced is not None and stats.get('voluntary_pass_rate') is not None:
            voluntary = decisions - forced
            columns.add(prefix + 'voluntary_decisions', voluntary)
            columns.add(prefix + 'voluntary_passes', round(stats['voluntary_pass_rate'] * voluntary))
        for state, values in (stats.get('by_score_state') or {}).items():
            n = values.get('decisions')
            columns.add(f'{prefix}{state}.decisions', n)
            if n:
                if values.get('bid_rate') is not None:
                    columns.add(f'{prefix}{state}.bids', round(values['bid_rate'] * n))
                if values.get('high_bid_rate') is not None:
                    columns.add(f'{prefix}{state}.high_bids', round(values['high_bid_rate'] * n))
        for category, values in (stats.get('bid_attempts') or {}).items():
            columns.add(f'{prefix}attempts.{category}.calls', values.get('calls'))
            columns.add(f'{prefix}attempts.{category}.auction_won', values.get('auction_won'))
        for category, values in ((stats.get('contracts') or {}).get('by_category') or {}).items():
            columns.add(f'{prefix}contracts.{category}.contracts', values.get('contracts'))
            columns.add(f'{prefix}contracts.{category}.made', values.get('made'))
        for score, values in (stats.get('by_match_score') or {}).items():
            totals = score_totals[team].setdefault(score, [0, 0, 0])
            n = values['decisions']
            totals[0] += n
            totals[1] += round(values['bid_rate'] * n)
            totals[2] += round(values['high_bid_rate'] * n)


def _flatten_record(columns: _Columns, row: dict, critic_mode, score_totals, assignments) -> None:
    columns.add_mapping('', {key: row.get(key) for key in TOP_LEVEL_FIELDS})
    columns.add('train_reward', row.get('mean_reward', row.get('mean_signed_reward')))
    metrics = row.get('metrics')
    if metrics is None:
        metrics = {key: row.get(key) for key in LEGACY_METRICS}
    partners = row.get('partners')
    if partners:
        for name, stats in partners.items():
            _flatten_learner(columns, f'{name}:', stats.get('metrics'), stats.get('value_diagnostics'),
                             stats.get('phase_samples'), critic_mode)
    else:
        _flatten_learner(columns, '', metrics, row.get('value_diagnostics'), row.get('phase_samples'), critic_mode)
    columns.add_mapping('oracle_validation.', row.get('oracle_validation'))
    columns.add_mapping('rollout.', row.get('rollout_performance'))
    columns.add('workers', (row.get('execution') or {}).get('resolved_workers'))
    _flatten_bidding(columns, row, score_totals)
    for opponent, stats in (row.get('training_by_opponent') or {}).items():
        for key in ('hands', 'matches', 'reward_sum', 'wins'):
            columns.add(f'opp.{opponent}.{key}', stats.get(key))
    per_row = Counter()
    for entry in row.get('opponent_assignments') or ():
        _, opponent, identity = entry
        per_row[opponent] += 1
        assignments[opponent][identity] += 1
    for opponent, count in per_row.items():
        columns.add(f'assigned.{opponent}', count)


def _win_metric(episode: str) -> str:
    return f'{episode}_win_rate'


def _evaluation_rows(report: dict, base: dict, *, units: dict | None = None) -> list[dict]:
    """Long rows, one per (policy, metric); no value is computed here, only read."""
    games = report.get('_games') or {}
    episode = games.get('episode') or base.get('episode') or 'unknown'
    groups, sample_hands = games.get('groups'), games.get('hands')
    if units:
        groups = units.get('duplicate_match_groups', groups)
        sample_hands = units.get('hands', sample_hands)
    common = dict(base, episode=episode, groups=groups, sample_hands=sample_hands, target_marks=games.get('target_marks'),
                  bootstrap_samples=report.get('bootstrap_samples'), eval_seed=report.get('seed'),
                  bootstrap_seed=report.get('bootstrap_seed'))
    intervals = {row.get('policy'): row for row in report.get('intervals') or ()}
    rows = []
    for metric in report.get('metrics') or ():
        policy = metric.get('policy')
        interval = intervals.get(policy, {})
        shared = dict(common, policy=policy, games=metric.get('games_played'))

        def add(name, value, unit, bounds=None):
            if value is not None:
                rows.append(dict(shared, metric=name, value=value, sample_unit=unit,
                                 ci_low=bounds['low'] if bounds else None, ci_high=bounds['high'] if bounds else None))

        add(_win_metric(episode), metric.get('win_rate'), 'matches' if episode == 'match' else 'hands',
            interval.get('win_rate'))
        add('mean_signed_marks_per_hand', metric.get('mean_signed_marks'), 'hands', interval.get('mean_signed_marks'))
        add('contract_success_rate', metric.get('contract_success_rate'), 'contracts')
        add('contracts_declared', metric.get('contracts_declared'), 'contracts')
        add('bidding_frequency', metric.get('bidding_frequency'), 'bid opportunities')
        add('mean_contract_value', metric.get('mean_contract_value'), 'marks per contract')
        if episode == 'match' and policy == 'A':
            add('average_match_length', report.get('average_match_length'), 'hands per match')
    return rows


def _superseded(hands: np.ndarray) -> np.ndarray:
    """True for rows a later resume replayed: their hands reach a later row's restart point."""
    keep = np.ones(len(hands), dtype=bool)
    lowest = np.inf
    for index in range(len(hands) - 1, -1, -1):
        if hands[index] >= lowest:
            keep[index] = False
        elif hands[index] == hands[index]:
            lowest = hands[index]
    return ~keep


def _cumulative_runtime(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Runtime continuing across resumes, which restart the recorded clock from zero."""
    result, offset, resets = raw.copy(), 0.0, 0
    for index in range(1, len(raw)):
        if raw[index] < raw[index - 1]:
            offset, resets = offset + raw[index - 1], resets + 1
        result[index] = raw[index] + offset
    return result, resets


def _setting(row, config, key):
    return row.get(key) if row.get(key) is not None else config.get(key)


def add_training(data: Dataset, source: TrainingSource) -> None:
    config, name = source.config, source.name
    columns = _Columns()
    score_totals = {team: {} for team in TEAMS}
    assignments = defaultdict(Counter)
    configs, evaluation_rows, lines = [], [], []
    last, seeds, seen_flattened = {}, {}, False
    for line, row in source.rows():
        columns.rows += 1
        lines.append(line)
        critic = _setting(row, config, 'critic_mode')
        _flatten_record(columns, row, critic, score_totals, assignments)
        opponents = row.get('train_opponents')
        if opponents is not None and (not configs or configs[-1][1] != opponents):
            configs.append((row.get('hands'), opponents))
        last = {key: row.get(key) for key in ('critic_mode', 'return_estimator', 'reward_objective', 'selection_opponent',
                                              'anchor_sha256', 'budget_unit', 'budget_target')}
        seen_flattened = seen_flattened or bool(row.get('_flattened'))
        reports = row.get('evaluations')
        if not reports and row.get('evaluation'):
            opponent = row.get('selection_opponent') or config.get('selection_opponent') or 'unknown'
            reports = {opponent: row['evaluation']}
        for opponent, report in (reports or {}).items():
            identity = (row.get('evaluation_identities') or {}).get(opponent)
            units = (row.get('evaluation_sample_units') or {}).get(opponent)
            base = dict(experiment=name, source='training', source_path=str(source.progress or source.csv_path),
                        source_line=line, hands=row.get('hands'), matches=row.get('matches'), updates=row.get('updates'),
                        opponent=opponent, opponent_identity=identity, checkpoint_name='in-training (current policy)')
            if seen_flattened:
                base['episode'] = 'match' if row.get('evaluation_average_match_length') is not None else 'unknown'
            evaluation_rows += _evaluation_rows(report, base, units=units)
            seeds.setdefault(opponent, report.get('seed'))
    if not columns.rows:
        data.warnings.append(f'{name}: no training records in {source.progress or source.csv_path}.')
    hands = np.full(columns.rows, np.nan)
    if 'hands' in columns.values:
        indices, numbers = columns.values['hands']
        hands[np.frombuffer(indices, dtype=np.int64)] = np.frombuffer(numbers, dtype=np.float64)
    dropped = _superseded(hands) if columns.rows else np.zeros(0, dtype=bool)
    if dropped.any():
        data.warnings.append(f'{name}: {int(dropped.sum())} record(s) replayed by a later resume were excluded '
                             '(superseded rows reach or pass the hand count where training restarted).')
    frame = columns.frame(~dropped)
    source_lines = np.asarray(lines, dtype=np.int64)[~dropped]
    frame['source_line'] = source_lines
    if 'runtime_seconds' in frame:
        valid = frame['runtime_seconds'].notna().to_numpy()
        cumulative = np.full(len(frame), np.nan)
        cumulative[valid], resets = _cumulative_runtime(frame['runtime_seconds'].to_numpy()[valid])
        frame['runtime_cumulative'] = cumulative
        if resets:
            data.warnings.append(f'{name}: recorded runtime restarted {resets} time(s) (resumed training); runtime axes '
                                 'add the segments together and exclude time between segments.')
    data.training[name] = frame

    def resolved(key, default, reason):
        value = _setting(last, config, key)
        return (value, 'recorded') if value is not None else (default, reason)

    objective, objective_source = resolved('reward_objective', 'hand-marks', 'assumed: log predates the match objective')
    has_value_loss = 'ppo.value_loss' in frame or 'partner_1:ppo.value_loss' in frame
    critic, critic_source = resolved('critic_mode', 'local' if has_value_loss else None,
                                     'inferred: value_loss recorded, log predates critic modes')
    estimator, estimator_source = resolved('return_estimator', 'gae' if critic == 'local' else None,
                                           'inferred: log predates return estimators')
    phases = config.get('learn_phases')
    phases_source = 'recorded'
    if phases is None and config:
        phases, phases_source = ['play'], 'inferred: legacy play-only training'
    source_files = dict(progress=str(source.progress) if source.progress else None,
                        training_csv=str(source.csv_path) if source.csv_path else None,
                        config=str(source.config_path) if source.config_path else None)
    final = lambda column: float(frame[column].dropna().iloc[-1]) if column in frame and frame[column].notna().any() else None
    data.experiments[name] = dict(
        experiment=name, kind='training', path=str(source.directory), source_files=source_files, seed=config.get('seed'),
        reward_objective=objective, reward_objective_source=objective_source, reward_unit=REWARD_UNITS.get(objective, objective),
        critic_mode=critic, critic_mode_source=critic_source, return_estimator=estimator,
        return_estimator_source=estimator_source, trained_phases=phases, trained_phases_source=phases_source,
        partner_mode=config.get('partner_mode', 'shared'),
        opponent_configuration=configs[-1][1] if configs else config.get('train_opponents'),
        eval_opponents=config.get('evaluation_names') or config.get('eval_opponents'),
        eval_every_hands=config.get('eval_every_hands'), eval_deals=config.get('eval_deals'),
        eval_match_pairs=config.get('eval_match_pairs'), eval_seeds=seeds,
        hidden_size=config.get('hidden_size'), rollout=config.get('rollout_decisions', config.get('rollout_play_steps')),
        final_hands=final('hands'), final_matches=final('matches'), final_updates=final('updates'),
        final_runtime_seconds=final('runtime_cumulative'), budget_unit=last.get('budget_unit'),
        anchor_sha256=last.get('anchor_sha256'), checkpoints=source.checkpoints, selection_opponent=last.get('selection_opponent'),
        config=config)
    data.by_match_score[name] = score_totals
    data.assignments[name] = {opponent: dict(counts) for opponent, counts in assignments.items()}
    data.opponent_configs[name] = configs
    data.warnings += source.notes
    if evaluation_rows:
        kept = set(source_lines.tolist())
        evaluation_rows = [row for row in evaluation_rows if row['source_line'] in kept]
        if 'runtime_cumulative' in frame:
            runtime = dict(zip(source_lines.tolist(), frame['runtime_cumulative'].tolist()))
            for row in evaluation_rows:
                row['runtime_cumulative'] = runtime[row['source_line']]
    data.pending['evaluation'].extend(evaluation_rows)


LETTERS = {'neural': 'N', 'heuristic': 'H', 'random': 'R', 'conservative': 'C', 'pimc': 'P'}
SPEC_PHASES = ('bidding', 'trump', 'playing')


def _checkpoint_name(path: str) -> str:
    """Readable name from a recorded path: <run>:<relative checkpoint>, on POSIX or Windows paths."""
    parts = PurePosixPath(path.replace('\\', '/')).with_suffix('').parts
    if 'checkpoints' in parts:
        index = len(parts) - 1 - parts[::-1].index('checkpoints')
        if index:
            return parts[index - 1] + ':' + '/'.join(parts[index + 1:])
    return '/'.join(parts[-2:])


def _spec_checkpoints(spec: dict) -> list[dict]:
    found = []
    for phase in SPEC_PHASES:
        checkpoint = (spec.get(phase) or {}).get('checkpoint')
        if checkpoint and checkpoint not in found:
            found.append(checkpoint)
    return found


def agent_identity(spec: dict | None, composition: dict) -> tuple | None:
    """Identity by strategy and weights hash, so a copied checkpoint is the same model."""
    if spec is None:
        return None
    phases = tuple(((spec.get(phase) or {}).get('strategy'),
                    ((spec.get(phase) or {}).get('checkpoint') or {}).get('sha256')) for phase in SPEC_PHASES)
    pimc = (composition.get('pimc_samples'), composition.get('pimc_leaf'), composition.get('pimc_likelihood'))
    return phases + (pimc if any(strategy == 'pimc' for strategy, _ in phases) else (None, None, None),)


def agent_label(spec: dict | None) -> str | None:
    if spec is None:
        return None
    letters = ''.join(LETTERS.get((spec.get(phase) or {}).get('strategy'), '?') for phase in SPEC_PHASES)
    checkpoints = _spec_checkpoints(spec)
    if not checkpoints:
        return letters
    return letters + ' ' + '+'.join(_checkpoint_name(c.get('path') or c.get('sha256', '?')) for c in checkpoints)


def _pairing(source: EvaluationSource) -> dict:
    composition = source.composition or {}
    agents, partners = composition.get('agents') or {}, composition.get('partners') or {}
    teams = {}
    for name in ('A', 'B'):
        first, second = agents.get(name), partners.get(name)
        checkpoints = _spec_checkpoints(first or {})
        partner_checkpoints = _spec_checkpoints(second or {})
        teams[name] = dict(first_key=agent_identity(first, composition), second_key=agent_identity(second, composition),
                           first_label=agent_label(first), second_label=agent_label(second),
                           first_checkpoints=checkpoints, second_checkpoints=partner_checkpoints)
        label = teams[name]['first_label'] or name
        teams[name]['label'] = label if second is None else f'{label} + {teams[name]["second_label"]}'
        teams[name]['key'] = (teams[name]['first_key'], teams[name]['second_key'])
    return dict(teams=teams, comparisons=composition.get('comparisons') or {},
                partner_comparisons=composition.get('partner_comparisons') or {},
                neural_inference=composition.get('neural_inference'))


def category_of(contract: str) -> str:
    """Bucket a recorded contract ('points:33', 'marks:2'), matching the rollout categories."""
    kind, _, value = contract.partition(':')
    if not value.isdigit():
        return 'other'
    number = int(value)
    if kind == 'points':
        return '30' if number == 30 else '31-41' if number < 42 else '42'
    return '42' if number == 1 else '2+ marks'


def _truth(value) -> bool | None:
    if isinstance(value, str):
        value = {'true': True, 'false': False}.get(value.strip().lower())
    return value if isinstance(value, bool) else None


def breakdown_from_hands(rows: list[dict]) -> dict:
    """Contract breakdown from per-hand records, for reports that recorded none."""
    result = {}
    for row in rows:
        policy = row.get('declaring_policy')
        success, forced = _truth(row.get('success')), _truth(row.get('forced_bid'))
        if policy in (None, '') or success is None:
            continue
        stats = result.setdefault(policy, dict(contract_origins={}, decision_timing=dict(contracts=0, recorded=0, distribution={})))
        points, marks = row.get('bid_points'), row.get('bid_marks')
        key = f'points:{int(float(points))}' if points not in (None, '') else f'marks:{int(float(marks))}'
        origin = 'unknown' if forced is None else 'forced' if forced else 'voluntary'
        entry = stats['contract_origins'].setdefault(key, {}).setdefault(origin, dict(contracts=0, successful=0, failed=0))
        entry['contracts'] += 1
        entry['successful' if success else 'failed'] += 1
        timing = stats['decision_timing']
        timing['contracts'] += 1
        trick = row.get('contract_decided_at_trick')
        if trick not in (None, ''):
            timing['recorded'] += 1
            trick = str(int(float(trick)))
            timing['distribution'][trick] = timing['distribution'].get(trick, 0) + 1
    return result


def add_evaluation(data: Dataset, source: EvaluationSource, checkpoint_names: dict[str, str]) -> None:
    pairing = _pairing(source) if source.composition else None
    if pairing:
        data.pairings[source.name] = pairing
    contracts, timing, roles, episodes, seeds = [], [], [], set(), set()
    for opponent, report in source.reports.items():
        label = pairing['teams']['B']['label'] if pairing else opponent
        reference = next((r.rsplit('.', 1)[1] for r in source.references), None)
        known = checkpoint_names.get(source.checkpoint_id or '')
        base = dict(experiment=source.name, source='report', source_path=str(source.path), source_line=None, hands=None, matches=None,
                    updates=None, opponent=label, opponent_identity=source.opponent_identities.get(opponent),
                    checkpoint_id=source.checkpoint_id,
                    checkpoint_name=known or (f'{reference}.pt (pipeline reference)' if reference else None))
        rows = _evaluation_rows(report, base, units=source.sample_units.get(opponent))
        if pairing:
            for row in rows:
                row['agent'] = pairing['teams'][row['policy']]['label']
        data.pending['evaluation'].extend(rows)
        for row in rows:
            episodes.add(row['episode'])
            seeds.add(row['eval_seed'])
        breakdown = source.breakdowns.get(opponent) or breakdown_from_hands(source.hand_rows.get(opponent, []))
        common = dict(experiment=source.name, opponent=label)
        for policy, stats in breakdown.items():
            origins = stats.get('contract_origins') or {key: {'unknown': dict(contracts=v.get('successful', 0) + v.get('failed', 0), **v)}
                                                        for key, v in (stats.get('contract_values') or {}).items()}
            for contract, by_origin in origins.items():
                for origin, values in by_origin.items():
                    contracts.append(dict(common, policy=policy, contract=contract, category=category_of(contract), origin=origin,
                                          contracts=values['contracts'], successful=values['successful'], failed=values['failed']))
            recording = stats.get('decision_timing') or {}
            for trick, count in (recording.get('distribution') or {}).items():
                timing.append(dict(common, policy=policy, trick=int(trick), count=count, recorded=recording.get('recorded'),
                                   contracts=recording.get('contracts')))
            for role, values in (stats.get('roles') or {}).items():
                roles.append(dict(common, policy=policy, role=role, **{key: values.get(key) for key in (
                    'hands', 'wins', 'signed_marks_sum', 'win_rate', 'mean_signed_marks')}))
    data.pending['contracts'].extend(contracts)
    data.pending['timing'].extend(timing)
    data.pending['roles'].extend(roles)
    data.experiments[source.name] = dict(
        experiment=source.name, kind='evaluation', path=str(source.path), schema=source.schema, sidecars=source.sidecars,
        checkpoint_id=source.checkpoint_id, anchor_id=source.anchor_id, references=source.references,
        opponents=[pairing['teams']['B']['label']] if pairing else list(source.reports),
        agent=pairing['teams']['A']['label'] if pairing else None, episodes=sorted(episodes),
        eval_seeds=sorted(s for s in seeds if s is not None), neural_inference=(pairing or {}).get('neural_inference'))
    data.warnings += source.notes


def add_search(data: Dataset, source: SearchSource) -> None:
    payload = source.data
    data.search.append(source)
    model = payload.get('model') or {}
    data.experiments[source.name] = dict(
        experiment=source.name, kind='search-accuracy', path=str(source.path), checkpoint_id=model.get('sha256'),
        table=payload.get('table'), games=payload.get('games'), samples=payload.get('samples'),
        positions=(payload.get('summary') or {}).get('positions'), eval_seeds=[payload.get('seed')],
        rank_check=bool(payload.get('rank_check')))


def build_dataset(training: list[TrainingSource], evaluations: list[EvaluationSource], searches: list[SearchSource],
                  notes: list[str] = ()) -> Dataset:
    data = Dataset(warnings=list(notes))
    for source in training:
        add_training(data, source)
    checkpoint_names = {sha: f'{source.name}:{name}.pt' for source in training for name, sha in source.checkpoints.items()}
    for source in evaluations:
        add_evaluation(data, source, checkpoint_names)
    for source in searches:
        add_search(data, source)
    for table, rows in data.pending.items():
        setattr(data, table, pd.DataFrame(rows))
    data.pending = {}
    return data


def _json_field(value) -> str | None:
    return None if value is None else json.dumps(value, separators=(',', ':'))


def normalized_training(data: Dataset, name: str) -> pd.DataFrame:
    """One run's update table with its experiment metadata columns, for CSV export."""
    meta = data.experiments[name]
    identity = {'experiment': name, 'training_seed': meta['seed'], 'reward_objective': meta['reward_objective'],
                'critic_mode': meta['critic_mode'], 'return_estimator': meta['return_estimator'],
                'trained_phases': _json_field(meta['trained_phases']),
                'opponent_configuration': _json_field(meta['opponent_configuration']),
                'source_file': meta['source_files']['progress'] or meta['source_files']['training_csv']}
    frame = data.training[name]
    return frame.assign(**identity)[list(identity) + list(frame.columns)]
