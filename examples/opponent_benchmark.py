"""Warm collection throughput for fixed and mixed opponents (not strength tests)."""
from examples.experiment_output import record_output, add_output_arguments
import argparse
import json
from pathlib import Path
from time import perf_counter
from research.ppo.opponents import parse_opponents
from research.ppo.trainer import Config, Trainer
from examples.benchmark_output import table


SPECIFICATIONS = ('HHH:1', 'CHH:1', 'HHH:1,CHH:1', 'HHH:7,CHH:2,HHR:1')


@record_output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands-per-update', type=int, default=256)
    parser.add_argument('--updates', type=int, default=3, help='Measured updates following one warm-up update')
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--output', type=Path)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if args.hands_per_update < 1 or args.updates < 1:
        parser.error('hands-per-update and updates must be positive')
    rows = []
    print('CPU, seed 42, episode-v1; summed worker times are not wall-clock phases', flush=True)
    for spec in SPECIFICATIONS:
        for workers in args.workers:
            config = Config(total_hands=args.hands_per_update * (args.updates + 1),
                            rollout_play_steps=args.hands_per_update * 14,
                            ppo_epochs=2, minibatch_size=256, hidden_size=args.hidden_size,
                            threads=workers, device='cpu', sampling_scheme='episode-v1',
                            train_opponents=parse_opponents(spec))
            counts = {}
            with Trainer(config) as trainer:
                wall_started = perf_counter()
                elapsed = assignment = construction = simulation = merging = optimization = 0.0
                for update in range(args.updates + 1):
                    buffer, _ = trainer.collect()
                    if update:
                        elapsed += trainer.collection_seconds
                        p = trainer.rollout_performance
                        assignment += p['assignment_seconds']
                        construction += p['agent_construction_seconds']
                        simulation += p['simulation_seconds']
                        merging += p['merging_seconds']
                        for name, stats in trainer.opponent_metrics.items():
                            counts[name] = counts.get(name, 0) + stats['hands']
                    trainer.update(buffer)
                    if update:
                        optimization += trainer.optimization_seconds
                wall = perf_counter() - wall_started
                measured_hands = args.updates * args.hands_per_update
                row = dict(configuration=spec, seed=42, requested_workers=workers,
                           resolved_workers=trainer.workers, measured_hands=measured_hands,
                           trained_hands=trainer.hands, samples=trainer.samples,
                           hands_per_second=measured_hands / elapsed, collection_seconds=elapsed,
                           assignment_worker_seconds=assignment, construction_worker_seconds=construction,
                           simulation_worker_seconds=simulation, merging_seconds=merging,
                           optimization_seconds=optimization, total_seconds=wall, counts=counts)
                rows.append(row)
                print(table(('Opponent pool', 'Workers', 'Hands/s', 'Assignment*', 'Construction*', 'Simulation*', 'Merging'),
                            [(spec, trainer.workers, f'{measured_hands / elapsed:.1f}', f'{assignment:.4f}',
                              f'{construction:.4f}', f'{simulation:.3f}', f'{merging:.4f}')]), flush=True)
    print('Mixed-policy trajectories differ. Throughput differences are not solely assignment overhead.')
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
    return rows


if __name__ == '__main__':
    main()
