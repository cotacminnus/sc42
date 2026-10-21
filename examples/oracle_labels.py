"""Sample play positions from seeded rollouts and label them with the bounded oracle.

Output is one versioned JSON dataset for --oracle-dataset in train_selfplay_ppo.
Rerunning with the same --output, budget and split reuses cached labels.
"""
import argparse
from pathlib import Path
from time import perf_counter
from examples.experiment_output import record_output, add_output_arguments
from research.ppo.oracle_dataset import dataset_sha256, label_positions, sample_positions
from texas42.agents import VARIANTS
from texas42.oracle import SearchLimits


def fractions(text):
    values = tuple(float(v) for v in text.split(','))
    if len(values) != 3:
        raise argparse.ArgumentTypeError('expected TRAIN,VALIDATION,TEST fractions')
    return values


def build_parser():
    """The command's argparse definition (also used by examples.run_pipeline to validate)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='dataset JSON path (also the label cache)')
    parser.add_argument('--deals', type=int, default=64, help='seeded hands to sample from')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', choices=tuple(VARIANTS), default='HHH', help='rollout policy on all seats')
    parser.add_argument('--min-tricks', type=int, default=4, help='earliest stage: completed tricks (0-6)')
    parser.add_argument('--max-tricks', type=int, default=6, help='latest stage: completed tricks (0-6)')
    parser.add_argument('--per-deal', type=int, default=1, help='maximum positions kept per deal')
    parser.add_argument('--max-positions', type=int, help='stop after this many positions')
    parser.add_argument('--max-nodes', type=int, default=200_000, help='per action search')
    parser.add_argument('--time-seconds', type=float, default=10.0, help='per action search')
    parser.add_argument('--max-cache-entries', type=int, default=100_000)
    parser.add_argument('--move-ordering', choices=('none', 'heuristic', 'tt', 'combined'), default='none')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--split', type=fractions, default=(0.8, 0.1, 0.1),
                        help='deal-level TRAIN,VALIDATION,TEST fractions')
    add_output_arguments(parser)
    return parser


@record_output
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        limits = SearchLimits(args.time_seconds, args.max_nodes, args.max_cache_entries)
        started = perf_counter()
        positions = sample_positions(deals=args.deals, seed=args.seed, policy=args.policy,
                                     min_tricks=args.min_tricks, max_tricks=args.max_tricks,
                                     per_deal=args.per_deal, max_positions=args.max_positions)
        sampling = perf_counter() - started
        dataset = label_positions(positions, args.output, limits=limits, move_ordering=args.move_ordering,
                                  split_seed=args.split_seed, fractions=args.split)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    summary = dataset.summary()
    print(f'Positions: {summary["positions"]} from {summary["deals"]} deals; splits {summary["splits"]}')
    print(f'Exact labels: {summary["exact_fraction"]:.1%}; mean bound width {summary["mean_bound_width_points"]:.3f} points')
    print(f'Sampling {sampling:.3f}s; labeling {summary["labeling_seconds"]:.3f}s and {summary["labeling_nodes"]} nodes '
          f'(cumulative, including {dataset.reused} cached labels)')
    print(f'Dataset: {args.output}\nSHA256: {dataset_sha256(args.output)}')
    return [dict(split=r['split'], deal_seed=r['position']['deal_seed'], seat=r['position']['seat'],
                 tricks_done=r['position']['tricks_done'], lower=r['lower'], upper=r['upper'],
                 exact=r['exact'], contract_status=r['contract_status'], nodes=r['nodes'],
                 elapsed_seconds=r['elapsed_seconds']) for r in dataset.records]


if __name__ == '__main__':
    main()
