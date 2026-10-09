"""Profile checkpoint validation, including startup and the legacy bootstrap."""
from examples.experiment_output import record_output, add_output_arguments
import argparse
import json
import hashlib
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from examples.arena_statistics_benchmark import legacy
import torch
from research.ppo.checkpoint import load_model
from research.ppo.evaluation import evaluate
from examples.bootstrap_benchmark import reference_bootstrap
from examples.benchmark_output import table


@record_output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--groups', type=int, nargs='+', default=[128, 256, 1024, 4096])
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--opponents', nargs='+', default=['HHH', 'CHH'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--legacy-small-check', action='store_true',
                        help='Also measure actual legacy 8-worker versus new 1-worker validation')
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    model, _ = load_model(args.checkpoint)
    identifier = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(f'Checkpoint SHA256: {identifier}; evaluation seed: 12345; bootstrap seed: 0; samples: 2000', flush=True)
    rows = []
    for opponent in args.opponents:
        for count in args.groups:
            expected = None
            old_seconds = None
            for workers in args.workers:
                performance = {}
                started = perf_counter()
                report = evaluate(model, opponent=opponent, deals=count, threads=workers,
                                  seed=12345, bootstrap_samples=2000, performance=performance)
                wall = perf_counter() - started
                if expected is None:
                    expected = report.to_dict()
                    started = perf_counter()
                    reference = tuple(reference_bootstrap(report.games, p, samples=2000) for p in ('A', 'B'))
                    old_seconds = perf_counter() - started
                    assert reference == report.intervals
                else:
                    assert expected == report.to_dict()
                before = wall - performance['bootstrap_sampling_seconds'] - performance['group_statistics_seconds'] + old_seconds
                row = dict(checkpoint_id=identifier, seed=12345, bootstrap_seed=0, bootstrap_samples=2000,
                           opponent=opponent, groups=count, requested_workers=workers,
                           **{k: v for k, v in performance.items() if k != 'requested_workers'},
                           wall_seconds=wall, reference_bootstrap_seconds=old_seconds,
                           reference_total_seconds=before, speedup=before / wall, exact_agreement=True)
                rows.append(row)
                print(table(('Opponent', 'Groups', 'Workers', 'Startup', 'Model init*', 'Simulation', 'Bootstrap', 'Wall'),
                    [(opponent, count, workers, f"{performance['worker_startup_seconds']:.3f}",
                      f"{performance['model_initialization_seconds']:.3f}",
                      f"{performance['simulation_seconds']:.3f}",
                      f"{performance['bootstrap_seconds']:.3f}", f'{wall:.3f}')]), flush=True)
                if args.output:
                    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    if args.legacy_small_check:
        with patch('texas42.arena.bootstrap_policies', legacy):
            started = perf_counter()
            old = evaluate(model, deals=128, threads=8, seed=12345, bootstrap_samples=2000)
            before = perf_counter() - started
        started = perf_counter()
        new = evaluate(model, deals=128, threads=1, seed=12345, bootstrap_samples=2000)
        after = perf_counter() - started
        assert old.to_dict() == new.to_dict()
        rows.append(dict(component='small_validation', groups=128, before_seconds=before,
                         after_seconds=after, speedup=before / after, exact_agreement=True))
        print(f'Actual legacy 8-worker validation: {before:.3f}s; optimized sequential: '
              f'{after:.3f}s; {before / after:.2f}x', flush=True)
        if args.output:
            args.output.write_text(json.dumps(rows, indent=2) + '\n')
    print('* Model initialization is summed worker time, not an additive wall-clock phase.')
    return rows


if __name__ == '__main__':
    main()
