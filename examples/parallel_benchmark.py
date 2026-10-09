"""Scaling measurements with correctness checks, including warm persistent PPO workers."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
from examples.benchmark_output import table
from texas42.agents import variant_factory
from texas42.arena import compare_agents
from texas42.parallel import resolve_workers


def peak_parent_memory():
    try:
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024**2 if sys.platform == 'darwin' else 1024)
    except (ImportError, AttributeError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', default='1,2,4,8,0')
    parser.add_argument('--deals', type=int, default=2048)
    parser.add_argument('--ppo-hands-per-update', type=int, default=256)
    parser.add_argument('--ppo-updates', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--arena-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        counts = [int(s) for s in args.workers.split(',')]
        for workers in counts: resolve_workers(workers)
        if min(args.deals, args.ppo_hands_per_update, args.ppo_updates) <= 0:
            raise ValueError('Workload counts must be positive')
    except ValueError as exc:
        parser.error(str(exc))
    # Always establish the sequential reference regardless of requested row order.
    counts = list(dict.fromkeys([1, *counts]))
    rows, reference, ppo_reference = [], None, None
    for workers in counts:
        perf = {}
        report = compare_agents(variant_factory('HHH'), variant_factory('CHH'), num_deals=args.deals,
                                seed=args.seed, bootstrap_samples=200, threads=workers, performance=perf)
        outcomes = report.to_dict()
        if reference is None: reference = outcomes
        if outcomes != reference: raise RuntimeError('Arena outcomes changed across workers')
        row = dict(perf, arena_equal=True, games_per_second=report.games_played / perf['total_seconds'],
                   decisions_per_second=report.games_played * 33 / perf['total_seconds'])
        if not args.arena_only:
            from research.ppo.trainer import Config, Trainer
            import torch
            config = Config(total_hands=(args.ppo_updates + 1) * args.ppo_hands_per_update,
                            rollout_play_steps=14 * args.ppo_hands_per_update,
                            threads=workers, torch_threads=1, sampling_scheme='episode-v1',
                            seed=args.seed, device=args.device)
            started = perf_counter()
            digests, collection, optimization, pids = [], [], [], set()
            with Trainer(config) as trainer:
                for update in range(args.ppo_updates + 1):
                    buffer, _ = trainer.collect()
                    digests.append(hashlib.sha256(pickle.dumps(buffer.records)).hexdigest())
                    trainer.update(buffer)
                    pids.update(trainer.worker_pids)
                    if update:  # First full update warms imports, model, and persistent pool.
                        collection.append(trainer.collection_seconds)
                        optimization.append(trainer.optimization_seconds)
                parameter_digest = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                                          for p in trainer.model.parameters())).hexdigest()
                row.update(device=str(trainer.device), ppo_workers=trainer.workers)
            signature = (digests, parameter_digest)
            if ppo_reference is None: ppo_reference = signature
            if signature != ppo_reference: raise RuntimeError('PPO samples/weights changed across workers')
            row.update(ppo_equal=True, ppo_total_seconds=perf_counter() - started,
                       warm_collection_seconds=sum(collection), warm_optimization_seconds=sum(optimization),
                       warm_hands_per_second=args.ppo_hands_per_update * args.ppo_updates / sum(collection),
                       update_samples_per_second=14 * args.ppo_hands_per_update * args.ppo_updates / sum(optimization),
                       worker_processes_seen=len(pids), cuda_available=torch.cuda.is_available())
        row['parent_peak_rss_mib'] = peak_parent_memory()
        rows.append(row)
        row['arena_speedup'] = rows[0]['total_seconds'] / row['total_seconds']
        row['arena_efficiency'] = row['arena_speedup'] / row['workers']
        if not args.arena_only:
            row['ppo_collection_speedup'] = rows[0]['warm_collection_seconds'] / row['warm_collection_seconds']
            row['ppo_collection_efficiency'] = row['ppo_collection_speedup'] / row['ppo_workers']
        print(f"Completed requested={workers}, resolved={row['workers']}; correctness matched", flush=True)
    print(table(('Requested', 'Workers', 'Arena s', 'Games/s', 'Speedup', 'Efficiency', 'Warm PPO hands/s', 'PPO speedup'), [
        (r['requested_workers'], r['workers'], f"{r['total_seconds']:.3f}", f"{r['games_per_second']:.1f}",
         f"{r['arena_speedup']:.2f}", f"{r['arena_efficiency']:.2f}",
         f"{r['warm_hands_per_second']:.1f}" if 'warm_hands_per_second' in r else 'N/A',
         f"{r['ppo_collection_speedup']:.2f}" if 'ppo_collection_speedup' in r else 'N/A') for r in rows]))
    print('Arena timing includes spawn/shutdown and bootstrap. PPO warm collection excludes its first update.')
    print('Parent peak RSS is a cumulative high-water mark, not per-worker memory. CPU/GPU utilization is not sampled.')
    if args.output:
        args.output.write_text(json.dumps(dict(configuration=vars(args) | {'output': str(args.output)}, results=rows), indent=2) + '\n')
    return rows


if __name__ == '__main__':
    main()
