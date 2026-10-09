"""Measure legacy versus pooled bootstrap on identical in-memory evaluations."""
from examples.experiment_output import record_output, add_output_arguments
import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from texas42.arena import (compare_agents, policy_metrics, PolicyIntervals,
                           ConfidenceInterval, bootstrap_policies, _seed)
from texas42.agents import variant_factory
from texas42.statistics import percentile_interval
from examples.benchmark_output import table


def reference_bootstrap(games, policy, *, seed=0, samples=2000):
    groups = {}
    for game in games:
        groups.setdefault(game.group_id, []).append(game)
    if len(groups) < 2:
        return PolicyIntervals(policy, None, None)
    groups = tuple(groups.values())
    rng = random.Random(_seed(seed, 'bootstrap'))
    wins, marks = [], []
    for _ in range(samples):
        selected = [game for _ in groups for game in rng.choice(groups)]
        metric = policy_metrics(selected, policy)
        if metric.win_rate is None:
            return PolicyIntervals(policy, None, None)
        wins.append(metric.win_rate)
        marks.append(metric.mean_signed_marks)
    return PolicyIntervals(policy, ConfidenceInterval(*percentile_interval(wins)),
                           ConfidenceInterval(*percentile_interval(marks)))


@record_output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', type=int, nargs='+', default=[128, 1024, 4096, 16384])
    parser.add_argument('--output', type=Path)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    # Replicate real immutable game rows under new group IDs. This benchmarks
    # statistical scaling, not an independently simulated strength experiment.
    fixture = compare_agents(variant_factory('RRR'), variant_factory('HHH'),
                             num_deals=32, seed=42, bootstrap_samples=1).games
    print('Fixture seed: 42; bootstrap seed: 0; complete group resampling', flush=True)
    rows = []
    for count in args.groups:
        games = tuple(replace(fixture[2 * (i % 32) + j], group_id=str(i))
                      for i in range(count) for j in range(2))
        samples = 200 if count == 128 else 2000
        started = perf_counter()
        old = tuple(reference_bootstrap(games, p, samples=samples) for p in ('A', 'B'))
        before = perf_counter() - started
        started = perf_counter()
        new = bootstrap_policies(games, ('A', 'B'), samples=samples)
        after = perf_counter() - started
        assert old == new, (old, new)
        rows.append(dict(groups=count, samples=samples, before_seconds=before,
                         after_seconds=after, speedup=before / after, exact_agreement=True))
        print(table(('Groups', 'Replicates', 'Before (s)', 'After (s)', 'Speedup', 'Exact'),
                    [(count, samples, f'{before:.3f}', f'{after:.3f}', f'{before / after:.2f}x', True)]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
    return rows


if __name__ == '__main__':
    main()
