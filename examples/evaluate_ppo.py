"""Evaluate a Stage-1 PPO checkpoint through the existing duplicate-deal arena."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from examples.benchmark_output import format_report, format_breakdown
from examples.heuristic_benchmark import breakdown
from research.ppo.checkpoint import load_model
from research.ppo.evaluation import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--opponent', choices=('HHH', 'CHH', 'HHR'), default='HHH')
    parser.add_argument('--deals', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    model, _ = load_model(args.checkpoint)
    started = perf_counter()
    report = evaluate(model, opponent=args.opponent, deals=args.deals, seed=args.seed,
                      bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples)
    elapsed = perf_counter() - started
    identifier = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(f'Checkpoint: {args.checkpoint} (SHA256 {identifier})')
    print(f'A: PPO play + heuristic auction/trump; B: {args.opponent}')
    print(format_report(report, {'A': 'PPO play', 'B': args.opponent}, elapsed))
    details = breakdown(report)
    print(format_breakdown(details))
    for policy, values in details.items():
        n = values['successful'] + values['failed']
        rate = f"{values['failed'] / n:.1%}" if n else 'N/A'
        print(f'{policy} contract set rate: {rate}')
    if args.output:
        args.output.write_text(json.dumps(dict(report.to_dict(), checkpoint_id=identifier,
                                             breakdown=details), indent=2) + '\n')
    return report


if __name__ == '__main__':
    main()
