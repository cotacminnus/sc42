"""Reproducible V1 baseline comparison; no optional dependencies required."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from examples.benchmark_output import format_report, format_breakdown

from texas42.agents import HeuristicAgent, RandomAgent
from texas42.arena import compare_agents


def heuristic_factory(seat, seed):
    return HeuristicAgent()


def random_factory(seat, seed):
    return RandomAgent(seed)


def breakdown(report):
    """Summarize resolved contracts, not unobserved bidder hand categories."""
    result = {}
    for policy in ('A', 'B'):
        hands = [h for g in report.games for h in g.hands
                 if h.seating[h.result.bidder] == policy]
        categories = {}
        for hand in hands:
            bid = hand.result.bid
            category = f'points:{bid.points}' if bid.points is not None else f'marks:{bid.marks}'
            row = categories.setdefault(category, {'successful': 0, 'failed': 0})
            row['successful' if hand.contract_success else 'failed'] += 1
        result[policy] = {
            'successful': sum(h.contract_success for h in hands),
            'failed': sum(not h.contract_success for h in hands),
            'bid_types': dict(Counter('points' if h.result.bid.points is not None else 'marks'
                                      for h in hands)),
            'trumps': dict(Counter('Follow Me' if h.result.trump is None else str(h.result.trump)
                                   for h in hands)),
            'contract_values': categories,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opponent', choices=('random', 'heuristic'), default='random')
    parser.add_argument('--deals', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode', choices=('hand', 'match'), default='hand')
    parser.add_argument('--target-marks', type=int, default=7)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    started = perf_counter()
    report = compare_agents(heuristic_factory,
                            random_factory if args.opponent == 'random' else heuristic_factory,
                            num_deals=args.deals, seed=args.seed, episode=args.episode,
                            target_marks=args.target_marks, bootstrap_seed=args.bootstrap_seed,
                            bootstrap_samples=args.bootstrap_samples)
    elapsed = perf_counter() - started
    print(format_report(report, {'A': 'Heuristic V1',
                                'B': 'Random' if args.opponent == 'random' else 'Heuristic V1'}, elapsed))
    details = breakdown(report)
    print(format_breakdown(details))
    if args.opponent == 'heuristic':
        print('Identical deterministic policies: paired symmetry/integration check, not evidence of strength.')
    else:
        print('Performance against uniform random play does not establish competitive strength.')
    if args.output:
        data = report.to_dict()
        data['benchmark'] = {'policy_a': 'heuristic_v1', 'policy_b': args.opponent,
                             'episode': args.episode, 'target_marks': args.target_marks}
        data['breakdown'] = details
        args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'JSON results saved to {args.output}')
    return report


if __name__ == '__main__':
    main()
