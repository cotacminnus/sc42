"""Complete before/after arena runs, replacing only the statistical algorithm."""
from examples.experiment_output import record_output, add_output_arguments
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from texas42.arena import compare_agents
from texas42.agents import variant_factory
from examples.bootstrap_benchmark import reference_bootstrap
from examples.benchmark_output import table


def legacy(games, policies, *, seed, samples, performance=None):
    started = perf_counter()
    result = tuple(reference_bootstrap(games, p, seed=seed, samples=samples) for p in policies)
    if performance is not None:
        performance.update(group_statistics_seconds=0.0,
                           bootstrap_sampling_seconds=perf_counter() - started)
    return result


@record_output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deals', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', type=Path)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    print('Experiment seed: 42; bootstrap seed: 0; samples: 2000', flush=True)
    rows = []
    for opponent in ('HHH', 'CHH'):
        before_performance, after_performance = {}, {}
        with patch('texas42.arena.bootstrap_policies', legacy):
            before = compare_agents(variant_factory('HHH'), variant_factory(opponent),
                                    num_deals=args.deals, threads=args.threads,
                                    performance=before_performance)
        after = compare_agents(variant_factory('HHH'), variant_factory(opponent),
                               num_deals=args.deals, threads=args.threads,
                               performance=after_performance)
        assert before.to_dict() == after.to_dict()
        row = dict(opponent=opponent, groups=args.deals, before=before_performance,
                   after=after_performance, speedup=before_performance['total_seconds'] /
                   after_performance['total_seconds'], exact_agreement=True)
        rows.append(row)
        print(table(('Opponent', 'Before (s)', 'After (s)', 'Speedup', 'Exact'),
                    [(opponent, f"{before_performance['total_seconds']:.3f}",
                      f"{after_performance['total_seconds']:.3f}", f"{row['speedup']:.2f}x", True)]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
    return rows


if __name__ == '__main__':
    main()
