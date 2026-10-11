"""Measured warmed rollout/update costs; performance is not policy-strength evidence."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from examples.experiment_output import record_output, add_output_arguments
from research.ppo.checkpoint import load_model
from research.ppo.trainer import Config, Trainer
from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
from research.ppo.opponents import PolicySpec
from research.ppo.selfplay_evaluation import evaluate_phases
from texas42.agents import VARIANTS


@record_output
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/ppo-play-v1/checkpoints/best.pt'))
    p.add_argument('--hands', type=int, default=64)
    p.add_argument('--workers', type=int, nargs='+', default=[1, 4])
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    p.add_argument('--output', type=Path)
    p.add_argument('--snapshot-dir', type=Path, default=Path('runs/multiphase-benchmark-snapshots'))
    add_output_arguments(p)
    a = p.parse_args(argv)
    if a.hands <= 0:
        p.error('hands must be positive')
    initial, _ = load_model(a.checkpoint)
    anchor = PolicySpec.checkpoint('anchor', a.checkpoint)
    policies = tuple(PolicySpec(n) for n in VARIANTS) + (anchor,)
    results = []
    print('Mode             Workers    Hands/s   Decisions/s  Inference  Opponent load  Optimize  Evaluation')
    for workers in a.workers:
        for mode, learn in (('play-legacy', ('play',)), ('bid-frozen-play', ('bid',)),
                            ('joint-heuristic', ('bid', 'play')), ('joint-historical', ('bid', 'play'))):
            common = dict(total_hands=a.hands * 2, hidden_size=initial.hidden_size, ppo_epochs=2,
                          minibatch_size=64, seed=42, device=a.device, threads=workers)
            if mode == 'play-legacy':
                t = Trainer(Config(**common, rollout_play_steps=a.hands * 14, sampling_scheme='episode-v1'))
                t.model.load_state_dict(initial.state_dict())
            else:
                t = SelfPlayTrainer(SelfPlayConfig(**common, learn_phases=learn,
                    rollout_decisions=a.hands * (2 if learn == ('bid',) else 16), anchor=anchor,
                    opponent_policies=policies,
                    opponent_probabilities=((('self-snapshots', '1'),) if mode == 'joint-historical' else (('HHH', '1'),))), initial)
            try:
                b, _ = t.collect(); t.update(b)
                if mode == 'joint-historical':
                    spec = t.publish_snapshot(a.snapshot_dir / f'workers-{workers}')
                before = t.hands
                b, _ = t.collect(); samples = len(b); t.update(b)
                evaluation_seconds = 0.0
                if mode != 'play-legacy':
                    started = perf_counter()
                    evaluate_phases(t.model, (anchor,), deals=2, bootstrap_samples=5)
                    evaluation_seconds = perf_counter() - started
                row = dict(mode=mode, requested_workers=workers, resolved_workers=t.workers,
                    hands=t.hands - before, decisions=samples,
                    hands_per_second=(t.hands - before) / t.collection_seconds,
                    decisions_per_second=samples / t.collection_seconds,
                    collection_seconds=t.collection_seconds, optimization_seconds=t.optimization_seconds,
                    inference_seconds=t.rollout_performance.get('neural_inference_seconds'),
                    policy_decision_seconds=t.rollout_performance.get('inference_seconds'),
                    opponent_loading_seconds=t.rollout_performance.get('opponent_loading_seconds'),
                    evaluation_seconds=evaluation_seconds,
                    worker_peak_rss_bytes=t.rollout_performance.get('worker_peak_rss_bytes'))
                results.append(row)
                def fmt(value): return f'{value:.3f}s' if value is not None else 'N/A'
                print(f'{mode:19} {t.workers:3} {row["hands_per_second"]:10.1f} {row["decisions_per_second"]:13.1f} '
                      f'{fmt(row["inference_seconds"]):>10} {fmt(row["opponent_loading_seconds"]):>14} '
                      f'{t.optimization_seconds:8.3f}s {evaluation_seconds:10.3f}s')
            finally:
                t.close()
    if a.output:
        a.output.write_text(json.dumps(results, indent=2) + '\n')
    return results


if __name__ == '__main__':
    main()
