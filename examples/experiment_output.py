"""Human-readable execution logs and flat CSV sidecars for experiment CLIs."""
import csv
import hashlib
from functools import wraps
from dataclasses import asdict
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout
from time import time_ns


DEFAULT_LOG_DIRECTORY = Path('logs/experiments')


def add_output_arguments(parser):
    parser.add_argument('--log-dir', type=Path,
                        help='Directory for readable logs and CSV exports '
                             '(default: logs/experiments/<command>-<run-id>/)')


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        return self.log.write(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def csv_rows(path, rows):
    """Union of scalar keys; complex experimental metadata remains in JSON."""
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as sink:
        writer = csv.DictWriter(sink, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def training_rows(progress_rows, seed):
    rows = []
    for progress_row in progress_rows:
        row = {k: v for k, v in progress_row.items() if not isinstance(v, (dict, list, tuple))}
        row.update(experiment_seed=seed)
        if 'evaluation' in progress_row:
            metric = progress_row['evaluation']['metrics'][0]
            interval = progress_row['evaluation']['intervals'][0]
            row.update(evaluation_win_rate=metric['win_rate'],
                       evaluation_mean_signed_marks=metric['mean_signed_marks'],
                       evaluation_workers=progress_row.get('evaluation_performance', {}).get('workers'))
            for name in ('win_rate', 'mean_signed_marks'):
                ci = interval[name]
                for endpoint in ('low', 'high'):
                    row['evaluation_' + name + '_' + endpoint] = ci[endpoint] if ci else None
        rows.append(row)
    return rows


def record_output(function):
    """Save actual terminal output and CSV records after a successful execution.

    Timing and invocation metadata stay in the readable log, outside deterministic
    ArenaReport JSON. --log-dir contains every readable log and CSV export;
    --output selects only their basename and the separate JSON destination.
    """
    @wraps(function)
    def wrapped(argv=None):
        arguments = list(sys.argv[1:] if argv is None else argv)
        def option(name):
            for i, arg in enumerate(arguments):
                if arg == name and i + 1 < len(arguments):
                    return arguments[i + 1]
                if arg.startswith(name + '='):
                    return arg.split('=', 1)[1]
            return None
        name = Path(function.__code__.co_filename).stem
        training = name == 'train_ppo'
        training_directory = Path(option('--output-dir') or 'runs/ppo-play-v1')
        log_directory = Path(option('--log-dir') or DEFAULT_LOG_DIRECTORY / f'{name}-{time_ns()}')
        log_directory.mkdir(parents=True, exist_ok=True)
        if option('--output'):
            output = Path(option('--output'))
            output.parent.mkdir(parents=True, exist_ok=True)
        base = log_directory / ('training' if training else
                                Path(option('--output') or name + '.json').name)
        log_path, csv_path = base.with_suffix('.log'), base.with_suffix('.csv')
        # An unconventional JSON filename must never be overwritten by a sidecar.
        if log_path == base:
            log_path = base.with_name(base.name + '.readable.log')
        if csv_path == base:
            csv_path = base.with_name(base.name + '.hands.csv')
        with log_path.open('a' if training else 'w', encoding='utf-8') as sink:
            with redirect_stdout(Tee(sys.stdout, sink)):
                print('Experiment arguments: ' + ' '.join(arguments))
                result = function(argv)
        metrics = None
        if training:
            progress = training_directory / 'progress.jsonl'
            rows = []
            config = json.loads((training_directory / 'config.json').read_text())
            with progress.open(encoding='utf-8') as source:
                progress_rows = ({k: v for k, v in json.loads(line).items()} for line in source)
                rows = training_rows(progress_rows, config['seed'])
        elif hasattr(result, 'games'):
            rows = []
            for game in result.games:
                for index, hand in enumerate(game.hands, 1):
                    r = hand.result
                    rows.append(dict(experiment_seed=result.seed, bootstrap_seed=result.bootstrap_seed,
                        bootstrap_samples=result.bootstrap_samples, group_id=game.group_id, game_id=game.game_id,
                        deal_id=hand.deal_id, hand=index, seating='/'.join(hand.seating),
                        shaker=hand.shaker, bidder=r.bidder, declaring_policy=hand.seating[r.bidder],
                        bid_points=r.bid.points, bid_marks=r.bid.marks, trump=r.trump,
                        success=hand.contract_success, forced_bid=hand.forced_bid,
                        would_pass_if_allowed=hand.would_pass_if_allowed,
                        points_team_0=r.team_points[0], points_team_1=r.team_points[1],
                        scoring_team=r.scoring_team, marks_awarded=r.marks_awarded,
                        decisions=hand.decisions, contract_decided_at_trick=hand.contract_decided_at_trick))
            metrics = []
            for metric, interval in zip(result.metrics, result.intervals):
                row = dict(vars(metric)) if hasattr(metric, '__dict__') else {
                    name: getattr(metric, name) for name in metric.__dataclass_fields__}
                for name in ('win_rate', 'mean_signed_marks'):
                    ci = getattr(interval, name)
                    row[name + '_low'] = ci.low if ci else None
                    row[name + '_high'] = ci.high if ci else None
                row.update(experiment_seed=result.seed, bootstrap_seed=result.bootstrap_seed,
                           bootstrap_samples=result.bootstrap_samples)
                metrics.append(row)
        elif isinstance(result, list):
            def flatten(row, prefix=''):
                flat = {}
                for key, value in row.items():
                    name = prefix + key
                    if isinstance(value, dict):
                        flat.update(flatten(value, name + '_'))
                    elif not isinstance(value, (list, tuple)):
                        flat[name] = value
                return flat
            rows = [flatten(row) for row in result]
        else:
            # Export raw interventions without rerunning specialized bootstrap
            # aggregation in CounterfactualReport.to_dict().
            data = {'hands': [asdict(hand) for hand in result.hands
                              if hand.intervention is not None]}
            rows = []
            for hand in data.get('hands', []):
                intervention = hand.get('intervention')
                if intervention is None:
                    continue
                row = {k: v for k, v in intervention.items()
                       if not isinstance(v, (dict, list, tuple))}
                row.update(experiment_seed=result.seed, bootstrap_seed=result.bootstrap_seed,
                           bid_points=intervention['original_bid']['points'],
                           bid_marks=intervention['original_bid']['marks'])
                for name, value in intervention['features'].items():
                    if not isinstance(value, (dict, list, tuple)):
                        row['feature_' + name] = value
                for branch in ('bid', 'passed'):
                    outcome = intervention[branch]
                    for name in ('signed_marks', 'acting_team_won', 'contract_success'):
                        row[branch + '_' + name] = outcome[name]
                    hand = outcome['hand']
                    resolved = hand['result']
                    row[branch + '_contract_decided_at_trick'] = hand['contract_decided_at_trick']
                    for name in ('bidder', 'trump', 'scoring_team', 'marks_awarded'):
                        row[branch + '_' + name] = resolved[name]
                rows.append(row)

        labels = {
            'policy_ablation_benchmark': (option('--policy-a') or 'HHH', option('--policy-b') or 'CRR'),
            'heuristic_benchmark': ('heuristic_v1', option('--opponent') or 'random'),
            'evaluate_ppo': ('PPO', option('--opponent') or 'HHH'),
            'arena_benchmark': ('Random', 'Random'),
        }.get(name)
        if labels:
            context = dict(policy_a=labels[0], policy_b=labels[1])
            if option('--checkpoint'):
                with Path(option('--checkpoint')).open('rb') as source:
                    context['checkpoint_id'] = hashlib.file_digest(source, 'sha256').hexdigest()
            for row in rows:
                row.update(context)
            if metrics is not None:
                for row in metrics:
                    row.update(context, policy_name=labels[0 if row['policy'] == 'A' else 1])
        if metrics is not None:
            csv_rows(base.with_name(base.stem + '-metrics.csv'), metrics)
        csv_rows(csv_path, rows)
        summary = ''
        if hasattr(result, 'games'):
            summary = ('\nFinal experiment summary\n' + result.summary() +
                       f'\nExperiment seed: {result.seed}; bootstrap seed: {result.bootstrap_seed}; '
                       f'replicates: {result.bootstrap_samples}\n')
            if labels:
                summary += f'A: {labels[0]}; B: {labels[1]}\n'
        elif hasattr(result, 'hands') and not training:
            deltas = [h.intervention.delta_marks for h in result.hands if h.intervention is not None]
            mean = f'{sum(deltas) / len(deltas):.4f}' if deltas else 'N/A'
            summary = (f'\nFinal experiment summary: {len(result.hands)} hands; '
                       f'{len(deltas)} interventions; mean delta marks {mean}\n')
        footer = summary + f'Readable log: {log_path}\nCSV records: {csv_path}\n'
        with log_path.open('a', encoding='utf-8') as sink:
            sink.write(footer)
        print(footer, end='')
        return result
    return wrapped
