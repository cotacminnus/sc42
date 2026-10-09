"""Reproducible V1 baseline comparison; no optional dependencies required."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from texas42.parallel import resolve_workers
from examples.benchmark_output import format_performance

from examples.benchmark_output import format_report, format_breakdown

from texas42.agents import HeuristicAgent, RandomAgent
from texas42.arena import compare_agents


def heuristic_factory(seat, seed):
    return HeuristicAgent()


def random_factory(seat, seed):
    return RandomAgent(seed)


def timing_summary(hands):
    """Unknown legacy timing is excluded explicitly, never imputed as trick seven."""
    values = [h.contract_decided_at_trick for h in hands
              if h.contract_decided_at_trick is not None]
    return {
        'contracts': len(hands),
        'recorded': len(values),
        'mean_decided_trick': sum(values) / len(values) if values else None,
        'mean_remaining_tricks': sum(7 - v for v in values) / len(values) if values else None,
        'distribution': {str(i): values.count(i) for i in range(1, 8)},
    }


def breakdown(report):
    """Summarize resolved contracts, not unobserved bidder hand categories."""
    result = {}
    for policy in ('A', 'B'):
        hands = [h for g in report.games for h in g.hands
                 if h.seating[h.result.bidder] == policy]
        categories = {}
        origins = {}
        preferences = {}
        thirty = {origin: {"successful": 0, "failed": 0}
                  for origin in ("forced", "voluntary")}
        for hand in hands:
            bid = hand.result.bid
            category = f'points:{bid.points}' if bid.points is not None else f'marks:{bid.marks}'
            row = categories.setdefault(category, {'successful': 0, 'failed': 0})
            outcome = 'successful' if hand.contract_success else 'failed'
            row[outcome] += 1
            origin = 'forced' if hand.forced_bid else 'voluntary'
            if hand.forced_bid:
                preference = ('unknown' if hand.would_pass_if_allowed is None else
                              'would_pass' if hand.would_pass_if_allowed else 'would_bid')
                counts = preferences.setdefault(category, {}).setdefault(
                    preference, {'successful': 0, 'failed': 0})
                counts[outcome] += 1
            origin_row = origins.setdefault(category, {}).setdefault(origin,
                {'contracts': 0, 'successful': 0, 'failed': 0, 'signed_marks_sum': 0})
            origin_row['contracts'] += 1
            origin_row[outcome] += 1
            origin_row['signed_marks_sum'] += hand.result.marks_awarded * (1 if hand.contract_success else -1)
            if bid.points == 30:
                thirty['forced' if hand.forced_bid else 'voluntary'][outcome] += 1
        for origin_rows in origins.values():
            for counts in origin_rows.values():
                counts['success_rate'] = counts['successful'] / counts['contracts']
                counts['mean_signed_marks'] = counts.pop('signed_marks_sum') / counts['contracts']
        result[policy] = {
            'successful': sum(h.contract_success for h in hands),
            'failed': sum(not h.contract_success for h in hands),
            'bid_types': dict(Counter('points' if h.result.bid.points is not None else 'marks'
                                      for h in hands)),
            'trumps': dict(Counter('Follow Me' if h.result.trump is None else str(h.result.trump)
                                   for h in hands)),
            'contract_values': categories,
            'contract_origins': origins,
            'forced_bid_preferences': preferences,
            'thirty_point_bids': thirty,
            'decision_timing': {
                **timing_summary(hands),
                'by_outcome': {name: timing_summary([h for h in hands if h.contract_success == success])
                               for name, success in (('successful', True), ('failed', False))},
                'by_origin': {name: timing_summary([h for h in hands if h.forced_bid == forced])
                              for name, forced in (('forced', True), ('voluntary', False))},
            },
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
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        resolve_workers(args.threads, args.deals)
    except ValueError as exc:
        parser.error(str(exc))
    performance = {}
    started = perf_counter()
    report = compare_agents(heuristic_factory,
                            random_factory if args.opponent == 'random' else heuristic_factory,
                            num_deals=args.deals, seed=args.seed, episode=args.episode,
                            target_marks=args.target_marks, bootstrap_seed=args.bootstrap_seed,
                            bootstrap_samples=args.bootstrap_samples, threads=args.threads, performance=performance)
    elapsed = perf_counter() - started
    print(format_performance(performance))
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
