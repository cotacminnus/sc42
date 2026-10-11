"""Evaluate a Stage-1 PPO checkpoint through the existing duplicate-deal arena."""
from examples.experiment_output import record_output, add_output_arguments
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from texas42.parallel import resolve_workers
from examples.benchmark_output import format_performance
import torch
from examples.benchmark_output import format_report, format_breakdown
from examples.heuristic_benchmark import breakdown
from research.ppo.checkpoint import load_model
from research.ppo.evaluation import evaluate_many
from research.ppo.opponents import evaluation_opponents
from texas42.agents import VARIANTS


@record_output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    opponents = parser.add_mutually_exclusive_group()
    opponents.add_argument('--opponent', choices=VARIANTS)
    opponents.add_argument('--opponents', help='Comma-separated validation variants')
    parser.add_argument('--deals', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    try:
        resolve_workers(args.threads, args.deals)
        names = evaluation_opponents(tuple(n.strip() for n in args.opponents.split(','))
                                     if args.opponents is not None else (args.opponent or 'HHH',))
    except ValueError as exc:
        parser.error(str(exc))
    performance = {}
    torch.set_num_threads(1)
    model, _ = load_model(args.checkpoint)
    started = perf_counter()
    reports = evaluate_many(model, opponents=names, deals=args.deals, seed=args.seed,
                           bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples,
                           threads=args.threads, performance=performance)
    elapsed = perf_counter() - started
    identifier = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(f'Checkpoint: {args.checkpoint} (SHA256 {identifier})')
    details_by_opponent = {}
    for name, report in reports.items():
        print(format_performance(performance[name]))
        phases = ','.join(getattr(model, 'neural_phases', ('play',)))
        print(f'A: PPO {phases}; remaining phases heuristic; B: {name}')
        print(format_report(report, {'A': 'PPO play', 'B': name}, performance[name]['evaluation_total_seconds']))
        details = breakdown(report)
        details_by_opponent[name] = details
        print(format_breakdown(details))
        for policy, values in details.items():
            n = values['successful'] + values['failed']
            rate = f"{values['failed'] / n:.1%}" if n else 'N/A'
            print(f'{policy} contract set rate: {rate}')
    print(f'Total checkpoint evaluation: {elapsed:.3f}s')
    if args.output:
        if len(reports) == 1:
            name = names[0]
            data = dict(reports[name].to_dict(), checkpoint_id=identifier, breakdown=details_by_opponent[name])
        else:
            data = dict(checkpoint_id=identifier, evaluations={name: report.to_dict() for name, report in reports.items()},
                        breakdowns=details_by_opponent)
        args.output.write_text(json.dumps(data, indent=2) + '\n')
    return reports[names[0]] if len(reports) == 1 else reports


if __name__ == '__main__':
    main()
