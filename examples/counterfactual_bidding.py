"""Paired voluntary BID versus PASS experiments under fixed continuation policies."""
from examples.experiment_output import record_output, add_output_arguments
import argparse
import json
from pathlib import Path
from time import perf_counter
from texas42.parallel import execution_summary

from examples.benchmark_output import table
from texas42.agents import VARIANTS
from texas42.experiments.counterfactual_bidding import compare_bidding


def format_results(data, elapsed):
    def number(value):
        return 'N/A' if value is None else f'{value:.3f}'
    def bid_label(key):
        _, kind, value = key.split('.')
        return f'{value} {kind}'
    def row(label, stats):
        ci = stats['mean_delta_ci95']
        interval = 'N/A' if ci is None else f"{ci['low']:.3f} to {ci['high']:.3f}"
        return (label, stats['samples'], number(stats['mean_delta_marks']),
                stats['bid_better'], stats['pass_better'], stats['equal'], interval)
    headers = ('Proposed bid / group', 'Samples', 'Mean Δ marks', 'Bid better', 'Pass better', 'Equal', '95% CI')
    overall = data['overall']
    lines = ['Counterfactual bidding outcomes',
        f"A: {data['policy_a']} | B: {data['policy_b']} | Target: {data['target_policy']}",
        f"Seed: {data['seed']} | Bootstrap seed: {data['bootstrap_seed']} | Samples: {data['bootstrap_samples']}",
        f"{data['total_hands']} original hands | {data['eligible_hands']} eligible | "
        f"{data['ineligible_hands']} without intervention | {data['skipped_eligible_decisions']} later opportunities skipped",
        f"BID replay checks passed: {data['selected_interventions']}",
        f"Runtime (evaluation + analysis): {elapsed:.3f} s",
        table(headers, [row('Overall', overall), *(row(bid_label(k), v) for k, v in data['by_bid'].items())]),
        f"Median Δ: {number(overall['median_delta_marks'])} | Mean BID marks: {number(overall['mean_bid_marks'])} | "
        f"Mean PASS marks: {number(overall['mean_pass_marks'])}",
        'Fractions BID better / PASS better / equal: ' + ' / '.join(number(overall[k]) for k in
            ('fraction_bid_better', 'fraction_pass_better', 'fraction_equal'))]
    for name, groups in data['by_hand_strength'].items():
        lines.extend(['', name, table(headers, [row(k, v) for k, v in groups.items()])])
    lines.append('N/A = unavailable. Intervals resample whole contributing duplicate groups.')
    lines.append('Conditional on the first voluntary bid selected by the target policy; not globally optimal bidding.')
    return '\n'.join(lines)


@record_output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-a', choices=VARIANTS, default='HHH')
    parser.add_argument('--policy-b', choices=VARIANTS, default='CHH')
    parser.add_argument('--target-policy', choices=VARIANTS, default='HHH')
    parser.add_argument('--deals', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode', choices=('hand', 'match'), default='hand')
    parser.add_argument('--target-marks', type=int, default=7)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if args.episode != 'hand':
        parser.error('match episodes are not supported; this experiment estimates immediate hand marks')
    try:
        print(execution_summary(args.threads, args.deals))
    except ValueError as exc:
        parser.error(str(exc))
    started = perf_counter()
    report = compare_bidding(args.policy_a, args.policy_b, target_policy=args.target_policy,
        num_deals=args.deals, seed=args.seed, episode=args.episode, target_marks=args.target_marks,
        bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples, threads=args.threads)
    data = report.to_dict()
    print(format_results(data, perf_counter() - started))
    if args.output:
        args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'JSON results saved to {args.output}')
    return report


if __name__ == '__main__':
    main()
