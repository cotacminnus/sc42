"""Run with python -m examples.arena_benchmark; no optional dependencies."""
import argparse

from texas42.agents import RandomAgent
from texas42.arena import compare_agents


def random_factory(seat: int, seed: int) -> RandomAgent:
    return RandomAgent(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deals', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap-seed', type=int, default=123)
    args = parser.parse_args()
    report = compare_agents(random_factory, random_factory, num_deals=args.deals,
                            seed=args.seed, bootstrap_seed=args.bootstrap_seed)
    print(report.summary())


if __name__ == '__main__':
    main()
