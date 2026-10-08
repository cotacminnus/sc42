"""Reproducible V1 baseline comparison; no optional dependencies required."""
import argparse
from collections import Counter
import json
from pathlib import Path

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
    report = compare_agents(heuristic_factory,
                            random_factory if args.opponent == 'random' else heuristic_factory,
                            num_deals=args.deals, seed=args.seed, episode=args.episode,
                            target_marks=args.target_marks, bootstrap_seed=args.bootstrap_seed,
                            bootstrap_samples=args.bootstrap_samples)
    print(f'A: Heuristic V1; B: {args.opponent}')
    print(report.summary())
    for metric in report.metrics:
        def number(value):
            return 'unavailable' if value is None else f'{value:.3f}'
        print(f'{metric.policy}: contracts {metric.contracts_declared}; '
              f'success rate {number(metric.contract_success_rate)}; '
              f'mean contract value (marks) {number(metric.mean_contract_value)}; '
              f'bidding frequency {number(metric.bidding_frequency)}')
    if args.episode == 'match':
        print(f'Average match length: {report.average_match_length:.3f} hands')
    details = breakdown(report)
    print('Resolved-contract breakdown: ' + json.dumps(details, sort_keys=True))
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
    return report


if __name__ == '__main__':
    main()
