"""Run with python -m examples.arena_benchmark; no optional dependencies."""
import argparse
from time import perf_counter

from examples.benchmark_output import format_report, format_performance
from texas42.parallel import resolve_workers

from texas42.agents import RandomAgent
from texas42.arena import compare_agents


def random_factory(seat: int, seed: int) -> RandomAgent:
    return RandomAgent(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deals', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap-seed', type=int, default=123)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    try:
        resolve_workers(args.threads, args.deals)
    except ValueError as exc:
        parser.error(str(exc))
    performance = {}
    started = perf_counter()
    report = compare_agents(random_factory, random_factory, num_deals=args.deals,
                            seed=args.seed, bootstrap_seed=args.bootstrap_seed,
                            threads=args.threads, performance=performance)
    elapsed = perf_counter() - started
    print(format_performance(performance))
    print(format_report(report, {'A': 'Random', 'B': 'Random'}, elapsed))


if __name__ == '__main__':
    main()
