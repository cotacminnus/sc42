"""Duplicate-deal comparisons of frozen phase policies, without ML dependencies."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from texas42.parallel import resolve_workers
from examples.benchmark_output import format_performance

from examples.benchmark_output import format_report, format_breakdown, table
from examples.heuristic_benchmark import breakdown
from texas42.agents import VARIANTS, variant_factory
from texas42.arena import compare_agents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-a', choices=VARIANTS, default='HHH')
    parser.add_argument('--policy-b', choices=VARIANTS, default='CRR')
    parser.add_argument('--deals', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode', choices=('hand', 'match'), default='hand')
    parser.add_argument('--target-marks', type=int, default=7)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        resolve_workers(args.threads, args.deals)
    except ValueError as exc:
        parser.error(str(exc))
    performance = {}
    started = perf_counter()
    report = compare_agents(variant_factory(args.policy_a), variant_factory(args.policy_b),
                            num_deals=args.deals, seed=args.seed, episode=args.episode,
                            target_marks=args.target_marks, bootstrap_seed=args.bootstrap_seed,
                            bootstrap_samples=args.bootstrap_samples, threads=args.threads, performance=performance)
    elapsed = perf_counter() - started
    print(format_performance(performance))
    labels = {p: f'{v} (bid/trump/play: {"/".join(VARIANTS[v])})'
              for p, v in (('A', args.policy_a), ('B', args.policy_b))}
    print(format_report(report, labels, elapsed))
    print('Contracts = auctions won, not hands won. Failure rate = 1 - success rate (when defined).')
    details = breakdown(report)
    print(format_breakdown(details))
    def rate(count, total):
        return f'{count / total:.1%}' if total else 'N/A'
    print('\nResolved-contract frequencies (denominator: auctions won)')
    print(table(('Policy', 'Failure rate', 'Point bids', 'Mark bids'), [
        (policy, rate(row['failed'], row['successful'] + row['failed']),
         rate(row['bid_types'].get('points', 0), row['successful'] + row['failed']),
         rate(row['bid_types'].get('marks', 0), row['successful'] + row['failed']))
        for policy, row in details.items()]))
    if args.policy_a == args.policy_b and args.policy_a in ('HHH', 'CHH'):
        print('Identical deterministic policies: paired symmetry check, not evidence of strength.')
    print('Phase effects interact; differences are not additive or causally independent.')
    print('Performance against random opponents does not establish competitive strength.')
    if args.output:
        data = report.to_dict()
        data['benchmark'] = {'policy_a': args.policy_a, 'policy_b': args.policy_b,
                             'phase_configurations': {p: VARIANTS[v] for p, v in
                                 (('A', args.policy_a), ('B', args.policy_b))},
                             'episode': args.episode, 'target_marks': args.target_marks}
        data['breakdown'] = details
        args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'JSON results saved to {args.output}')
    return report


if __name__ == '__main__':
    main()
