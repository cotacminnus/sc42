"""Read-only access to SC42 experiment artifacts; standard library only.

Every function here opens files for reading and nothing else. Full game records
dominate `progress.jsonl` (gigabytes on long runs) and `result.json`, and only
their counts are needed, so `strip_games` removes them from the JSON text before
parsing and leaves a `_games` summary.
"""
from collections.abc import Iterator
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

MANIFEST = 'pipeline_state.json'
_GAMES_END = re.compile(r'\]\s*,\s*"metrics"\s*:')
_GROUP_ID = re.compile(r'"group_id":\s*"([^"]*)"')
_GAME_ID = re.compile(r'"game_id":\s*"')
_DEAL_ID = re.compile(r'"deal_id":\s*"')
_EPISODE = re.compile(r'"episode":\s*"(\w+)"')
_TARGET = re.compile(r'"target_marks":\s*(\d+)')


class InputError(ValueError):
    """An input path or file that cannot be used; reported to the user."""


def _summary(games: int, groups: int, hands: int, episode: str | None, target: int | None) -> dict:
    # Reports written before episodes were recorded hold one hand per game unless matches were played.
    inferred = episode is None
    if inferred:
        episode = 'match' if hands > games else 'hand'
    return dict(games=games, groups=groups, hands=hands, episode=episode, target_marks=target,
                episode_inferred=inferred)


def summarize_games(games: list[dict]) -> dict:
    """The `_games` summary of parsed game records."""
    episode = games[0].get('episode') if games else None
    target = games[0].get('target_marks') if games else None
    return _summary(len(games), len({g.get('group_id', g.get('game_id')) for g in games}),
                    sum(len(g.get('hands', ())) for g in games), episode, target)


def _summarize_games_text(body: str) -> dict:
    episode, target = _EPISODE.search(body), _TARGET.search(body)
    return _summary(len(_GAME_ID.findall(body)), len(set(_GROUP_ID.findall(body))), len(_DEAL_ID.findall(body)),
                    episode[1] if episode else None, int(target[1]) if target else None)


def strip_games(text: str) -> str:
    """Replace each report's `"games": [...]` with a `"_games": {...}` count summary."""
    pieces, position = [], 0
    while (start := text.find('"games":', position)) >= 0:
        opening = text.index('[', start)
        end = _GAMES_END.search(text, opening)
        if end is None:
            raise ValueError('games array has no following metrics field')
        pieces += [text[position:start], '"_games": ', json.dumps(_summarize_games_text(text[opening:end.start()]))]
        position = end.start() + 1
    pieces.append(text[position:])
    return ''.join(pieces)


def _strip_parsed(value):
    if isinstance(value, dict):
        if isinstance(value.get('games'), list):
            value['_games'] = summarize_games(value.pop('games'))
        for item in value.values():
            _strip_parsed(item)
    elif isinstance(value, list):
        for item in value:
            _strip_parsed(item)
    return value


def loads_without_games(text: str):
    """Parse report JSON without materializing games, falling back to a full parse."""
    if '"games":' not in text:
        return json.loads(text)
    try:
        return json.loads(strip_games(text))
    except ValueError:
        return _strip_parsed(json.loads(text))


def file_sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        raise InputError(f'Cannot read {path}: {exc}') from exc
    except ValueError as exc:
        raise InputError(f'{path} is not valid JSON: {exc}') from exc


@dataclass
class TrainingSource:
    """One training directory. Records stream through `rows()` because logs can exceed memory."""
    name: str
    directory: Path
    progress: Path | None
    csv_path: Path | None
    config_path: Path | None
    config: dict
    checkpoints: dict[str, str] = field(default_factory=dict)  # 'best', 'partner-1/best', ... -> SHA-256
    notes: list[str] = field(default_factory=list)

    def rows(self) -> Iterator[tuple[int, dict]]:
        """(1-based source line, record with games replaced by counts)."""
        if self.progress is None:
            yield from _rows_from_training_csv(self.csv_path)
            return
        with self.progress.open(encoding='utf-8') as source:
            for number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield number, loads_without_games(line)
                    except ValueError as exc:
                        raise InputError(f'{self.progress}:{number} is not valid JSON: {exc}') from exc


def _number(text):
    if text in (None, ''):
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _rows_from_training_csv(path: Path) -> Iterator[tuple[int, dict]]:
    """Rebuild the record subset that the flattened training.csv retains."""
    with path.open(newline='', encoding='utf-8') as source:
        for number, raw in enumerate(csv.DictReader(source), 2):
            row = {key: _number(value) for key, value in raw.items()}
            record, evaluations, metrics = {}, {}, {}
            for key, value in row.items():
                if value is None:
                    continue
                if key.startswith('ppo_'):
                    metrics[key[4:]] = value
                elif key.startswith('samples_'):
                    record.setdefault('phase_samples', {})[key[8:]] = value
                elif key.startswith('rollout_') and key not in ('rollout_matches', 'rollout_hands', 'rollout_decisions'):
                    record.setdefault('rollout_performance', {})[key[8:]] = value
                elif match := re.match(r'eval_(.+?)_(games_played|win_rate|mean_signed_marks|contract_success_rate|'
                                       r'contracts_declared|seed|bootstrap_seed|bootstrap_samples)(_low|_high)?$', key):
                    evaluations.setdefault(match[1], {})[match[2] + (match[3] or '')] = value
                elif not key.startswith(('train_', 'eval_', 'evaluation_', 'training_opponent')):
                    record[key] = value
            record['metrics'] = metrics
            if evaluations:
                record['evaluations'] = {opponent: _report_from_flat(values) for opponent, values in evaluations.items()}
                record['_flattened'] = True
            yield number, record


def _report_from_flat(values: dict) -> dict:
    interval = {name: ({'low': values[name + '_low'], 'high': values[name + '_high']}
                       if name + '_low' in values and name + '_high' in values else None)
                for name in ('win_rate', 'mean_signed_marks')}
    metric = {key: values.get(key) for key in ('games_played', 'win_rate', 'mean_signed_marks', 'contract_success_rate',
                                               'contracts_declared')}
    return dict(seed=values.get('seed'), bootstrap_seed=values.get('bootstrap_seed'),
                bootstrap_samples=values.get('bootstrap_samples'),
                metrics=[dict(metric, policy='A')], intervals=[dict(interval, policy='A')])


def load_training(name: str, directory: Path) -> TrainingSource:
    directory = Path(directory)
    if not directory.is_dir():
        raise InputError(f'Training directory {directory} does not exist')
    progress = directory / 'progress.jsonl'
    flattened = directory / 'logs' / 'training.csv'
    if not flattened.is_file():
        flattened = next((p for p in (directory / 'training.csv',) if p.is_file()), None)
    if not progress.is_file():
        if flattened is None:
            hint = ' (it holds pipeline_state.json: use --pipeline)' if (directory / MANIFEST).is_file() else ''
            raise InputError(f'{directory} has no progress.jsonl or training.csv{hint}')
        progress = None
    config_path = directory / 'config.json'
    config = _read_json(config_path) if config_path.is_file() else {}
    source = TrainingSource(name, directory, progress, flattened, config_path if config else None, config)
    if not config:
        source.notes.append(f'{name}: no config.json; settings are limited to fields in the training records.')
    if progress is None:
        source.notes.append(f'{name}: progress.jsonl is missing, so only the flattened training.csv is available; '
                            'nested diagnostics (bidding, value, opponents) and episode types are unavailable.')
    root = directory / 'checkpoints'
    for checkpoint in sorted(root.glob('**/*.pt')) if root.is_dir() else ():
        source.checkpoints[checkpoint.relative_to(root).with_suffix('').as_posix()] = file_sha256(checkpoint)
    return source


@dataclass
class EvaluationSource:
    """One arena result: `reports` maps an opponent label to a report without games."""
    name: str
    path: Path
    schema: str  # 'composed', 'multi-opponent', 'single-opponent', or 'csv'
    reports: dict[str, dict]
    breakdowns: dict[str, dict]
    checkpoint_id: str | None = None
    anchor_id: str | None = None
    opponent_identities: dict = field(default_factory=dict)
    composition: dict | None = None
    sample_units: dict = field(default_factory=dict)
    sidecars: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)  # Pipeline checkpoint references, e.g. 'local.best'.
    notes: list[str] = field(default_factory=list)
    hand_rows: dict[str, list[dict]] = field(default_factory=dict)  # Only when no breakdown was recorded.


def _hand_rows(games: list[dict]) -> list[dict]:
    """Per-hand rows in the schema of the arena's per-hand CSV."""
    rows = []
    for game in games:
        for hand in game['hands']:
            result = hand['result']
            bid = result.get('bid') or {}
            rows.append(dict(group_id=game.get('group_id'), game_id=game.get('game_id'),
                             declaring_policy=hand['seating'][result['bidder']], bid_points=bid.get('points'),
                             bid_marks=bid.get('marks'), success=hand.get('contract_success'),
                             forced_bid=hand.get('forced_bid'),
                             contract_decided_at_trick=hand.get('contract_decided_at_trick')))
    return rows


def _sidecars(path: Path) -> list[str]:
    logs = path.parent / 'logs'
    candidates = [logs / f'{path.stem}.csv', logs / f'{path.stem}-metrics.csv', logs / f'{path.stem}.log']
    return [str(p) for p in candidates if p.is_file()]


def _evaluation_from_csv(name: str, path: Path) -> EvaluationSource:
    """Metrics CSV (with its sibling per-hand CSV if present), or a per-hand CSV alone."""
    with path.open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    if not rows or not ({'win_rate', 'declaring_policy'} & set(rows[0])):
        raise InputError(f'{path} is neither an arena metrics CSV nor a per-hand CSV')
    if 'declaring_policy' in rows[0]:
        hand_path, metric_rows = path, None
        sibling = path.with_name(path.stem + '-metrics.csv')
        if sibling.is_file():
            with sibling.open(newline='', encoding='utf-8') as source:
                metric_rows = list(csv.DictReader(source))
        hands = rows
    else:
        metric_rows = rows
        hand_path = path.with_name(path.name.removesuffix('-metrics.csv') + '.csv')
        hands = []
        if hand_path.is_file():
            with hand_path.open(newline='', encoding='utf-8') as source:
                hands = list(csv.DictReader(source))
    grouped_hands: dict[str, list[dict]] = {}
    for row in hands:
        grouped_hands.setdefault(row.get('opponent') or row.get('policy_b') or 'B', []).append(row)
    reports: dict[str, dict] = {}
    for row in metric_rows or ():
        opponent = row.get('opponent') or row.get('policy_b') or 'B'
        report = reports.setdefault(opponent, dict(metrics=[], intervals=[]))
        values = {key: _number(value) for key, value in row.items()}
        report.update(seed=values.get('experiment_seed'), bootstrap_seed=values.get('bootstrap_seed'),
                      bootstrap_samples=values.get('bootstrap_samples'))
        report['metrics'].append({key: values.get(key) for key in (
            'policy', 'games_played', 'win_rate', 'mean_signed_marks', 'contracts_declared', 'contract_success_rate',
            'mean_contract_value', 'bidding_frequency')})
        report['intervals'].append({'policy': values.get('policy'), **{
            metric: ({'low': values[metric + '_low'], 'high': values[metric + '_high']}
                     if values.get(metric + '_low') is not None else None)
            for metric in ('win_rate', 'mean_signed_marks')}})
    for opponent, opponent_hands in grouped_hands.items():
        games = {row['game_id'] for row in opponent_hands}
        groups = {row['group_id'] for row in opponent_hands}
        reports.setdefault(opponent, dict(metrics=[], intervals=[]))['_games'] = _summary(
            len(games), len(groups), len(opponent_hands), None, None)
    source = EvaluationSource(name, path, 'csv', reports, {}, hand_rows=grouped_hands,
                              sidecars=[str(hand_path)] if hands and hand_path != path else [])
    checkpoint = next((row['checkpoint_id'] for row in metric_rows or rows if row.get('checkpoint_id')), None)
    source.checkpoint_id = checkpoint
    if metric_rows is None:
        source.notes.append(f'{name}: per-hand CSV without its metrics CSV; recorded intervals are unavailable, '
                            'so only contract figures can use it.')
    return source


def load_evaluation(name: str, path: Path) -> EvaluationSource:
    path = Path(path)
    if not path.is_file():
        raise InputError(f'Evaluation report {path} does not exist')
    if path.suffix.lower() == '.csv':
        return _evaluation_from_csv(name, path)
    try:
        text = path.read_text(encoding='utf-8')
        data = loads_without_games(text)
    except (OSError, ValueError) as exc:
        raise InputError(f'Cannot read evaluation report {path}: {exc}') from exc
    if not isinstance(data, dict) or not ('metrics' in data or 'evaluations' in data):
        raise InputError(f'{path} is not an arena report (expected `metrics` or `evaluations`)')
    if 'evaluations' in data:
        source = EvaluationSource(name, path, 'multi-opponent', data['evaluations'], data.get('breakdowns') or {},
                                  sample_units=data.get('sample_units') or {})
    elif data.get('composition'):
        source = EvaluationSource(name, path, 'composed', {'B': data}, {'B': data.get('breakdown') or {}},
                                  composition=data['composition'],
                                  sample_units={'B': data['sample_units']} if data.get('sample_units') else {})
    else:
        source = EvaluationSource(name, path, 'single-opponent', {'B': data}, {'B': data.get('breakdown') or {}})
    source.checkpoint_id = data.get('checkpoint_id')
    source.anchor_id = data.get('anchor_id')
    source.opponent_identities = data.get('opponent_identities') or {}
    source.sidecars = _sidecars(path)
    missing = [opponent for opponent in source.reports if not source.breakdowns.get(opponent)]
    if missing:  # Old reports: contract figures fall back to the hands themselves.
        full = json.loads(text)
        for opponent in missing:
            report = full if source.schema != 'multi-opponent' else full['evaluations'][opponent]
            source.hand_rows[opponent] = _hand_rows(report.get('games') or [])
    return source


@dataclass
class SearchSource:
    name: str
    path: Path
    data: dict


def load_search_accuracy(name: str, path: Path) -> SearchSource:
    path = Path(path)
    if not path.is_file():
        raise InputError(f'Search accuracy report {path} does not exist')
    data = _read_json(path)
    if not isinstance(data, dict) or 'summary' not in data or 'beliefs' not in (data.get('summary') or {}):
        raise InputError(f'{path} is not a search_accuracy.py report (expected `summary.beliefs`)')
    return SearchSource(name, path, data)


@dataclass
class PipelineContents:
    training: list[TrainingSource] = field(default_factory=list)
    evaluations: list[EvaluationSource] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def discover_pipeline(directory: Path, prefix: str = '') -> PipelineContents:
    """Completed training and evaluation steps of a pipeline, from its manifest only.

    A tree that moved after running still works: each step is looked up beside
    the manifest first, then at the location the manifest recorded.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise InputError(f'{directory} has no {MANIFEST}; is it a pipeline output_root?')
    manifest = _read_json(manifest_path)
    contents = PipelineContents()
    for step in manifest.get('steps', ()):
        step_id, kind, state = step['id'], step.get('kind'), step.get('state')
        name = prefix + step_id
        if state != 'completed':
            contents.skipped.append(f'{name}: step is {state}, not completed')
            continue
        recorded = step.get('locations') or {}
        step_directory = directory / step_id
        if not step_directory.is_dir() and recorded.get('directory'):
            step_directory = Path(recorded['directory'])
        try:
            if kind == 'training':
                contents.training.append(load_training(name, step_directory))
            elif kind == 'evaluation':
                path = step_directory / 'result.json'
                if not path.is_file() and recorded.get('output'):
                    path = Path(recorded['output'])
                evaluation = load_evaluation(name, path)
                evaluation.references = [d['reference'] for d in step.get('dependencies', ()) if d.get('reference')]
                contents.evaluations.append(evaluation)
            else:
                contents.skipped.append(f'{name}: {kind} step has no figures')
        except InputError as exc:
            contents.skipped.append(f'{name}: {exc}')
    return contents
