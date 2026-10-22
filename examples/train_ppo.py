"""Train masked PPO trick play against configured frozen variants using complete hand episodes."""
from examples.experiment_output import record_output, add_output_arguments, add_config_arguments
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
from time import perf_counter
from research.ppo.trainer import Config, Trainer, stream_seed
from research.ppo.checkpoint import save, resume, load_model
from research.ppo.evaluation import evaluate_many
from research.ppo.opponents import parse_opponents, evaluation_opponents, select_primary
from texas42.agents import VARIANTS
from texas42.parallel import resolve_workers


OPPONENT_FIELDS = ('train_opponents', 'eval_opponents', 'selection_opponent')


def build_parser():
    """The command's argparse definition (also used by examples.run_pipeline to validate)."""
    parser = argparse.ArgumentParser(description=__doc__)
    default = Config()
    add_config_arguments(parser, default, exclude=OPPONENT_FIELDS)
    training = parser.add_mutually_exclusive_group()
    training.add_argument('--train-opponent', choices=VARIANTS)
    training.add_argument('--train-opponents', help='NAME:WEIGHT,NAME:WEIGHT,...')
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument('--eval-opponent', choices=VARIANTS)
    validation.add_argument('--eval-opponents', help='Comma-separated opponent names')
    parser.add_argument('--selection-opponent', choices=VARIANTS)
    parser.add_argument('--eval-threads', type=int, default=1,
                        help='Validation simulation workers, independent of rollout workers (0: auto)')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/ppo-play-v1'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--resume', type=Path)
    source.add_argument('--warm-start', type=Path)
    add_output_arguments(parser)
    return parser


@record_output
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolve_workers(args.eval_threads, args.eval_deals)
        opponents = parse_opponents(args.train_opponents) if args.train_opponents is not None else ((args.train_opponent or 'HHH', '1'),)
        eval_names = evaluation_opponents(tuple(n.strip() for n in args.eval_opponents.split(','))
                         if args.eval_opponents is not None else (args.eval_opponent or 'HHH',))
        primary = select_primary(eval_names, args.selection_opponent)
        config = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if f.name not in OPPONENT_FIELDS},
                        train_opponents=opponents, eval_opponents=eval_names, selection_opponent=primary)
        trainer = Trainer(config)
        if args.resume:
            resume(args.resume, trainer)
        if args.warm_start:
            model, _ = load_model(args.warm_start)
            if model.hidden_size != config.hidden_size:
                raise ValueError('Warm-start hidden-size must match the checkpoint')
            trainer.model.load_state_dict(model.state_dict())
    except ValueError as exc:
        parser.error(str(exc))
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'progress.jsonl').exists() and not args.resume:
        parser.error('Output directory already contains a run; choose another directory or --resume')
    (directory / 'config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    checkpoints = directory / 'checkpoints'
    started = perf_counter()
    initial_hands = trainer.hands
    execution = trainer.execution_metadata()
    print('Execution configuration', flush=True)
    print(f'Training seed: {config.seed}; validation seed: {stream_seed(config.seed, "validation")}; '
          f'validation bootstrap seed: {stream_seed(config.seed, "validation-bootstrap")}', flush=True)
    print(f'Device: {trainer.device}\nRequested workers: {config.threads}\nResolved workers: {trainer.workers}\n'
          f'PyTorch CPU threads: {config.torch_threads}\nParallel backend: {execution["backend"]}\n'
          f'Rollout sampling: {trainer.sampling_scheme}', flush=True)
    print(f'Validation workers: {resolve_workers(args.eval_threads, config.eval_deals)} (requested {args.eval_threads})', flush=True)
    print(f'PPO trick-play training: heuristic auction/trump, neural partnership versus {trainer.opponents.describe()}', flush=True)
    print(f'Evaluation opponents: {", ".join(config.eval_opponents)}; checkpoint selection: {config.selection_opponent}', flush=True)
    try:
        while trainer.hands < config.total_hands:
            update_started = perf_counter()
            hands_before = trainer.hands
            buffer, reward = trainer.collect()
            metrics = trainer.update(buffer)
            elapsed = perf_counter() - started
            row = dict(hands=trainer.hands, play_samples=trainer.samples, updates=trainer.updates,
                       optimizer_steps=trainer.optimizer_steps, mean_signed_reward=reward,
                       runtime_seconds=elapsed, hands_per_second=(trainer.hands - initial_hands) / elapsed, **metrics)
            print(f"Hands {trainer.hands}/{config.total_hands} | samples {trainer.samples} | update {trainer.updates} | "
                  f"policy {metrics['policy_loss']:.4f} value {metrics['value_loss']:.4f} entropy {metrics['entropy']:.4f} "
                  f"KL {metrics['approx_kl']:.5f} clip {metrics['clip_fraction']:.3f} grad {metrics['gradient_norm']:.3f} | "
                  f"{row['hands_per_second']:.1f} hands/s", flush=True)
            row.update(execution=execution, worker_pids=trainer.worker_pids,
                       episode_start=hands_before, episode_stop=trainer.hands,
                       train_opponents=config.train_opponents, training_by_opponent=trainer.opponent_metrics,
                       rollout_performance=trainer.rollout_performance, selection_opponent=config.selection_opponent,
                       collection_seconds=trainer.collection_seconds,
                       optimization_seconds=trainer.optimization_seconds,
                       evaluation_seconds=0.0, checkpoint_seconds=0.0,
                       collection_hands_per_second=(trainer.hands - hands_before) / trainer.collection_seconds,
                       collection_decisions_per_second=14 * (trainer.hands - hands_before) / trainer.collection_seconds)
            for name, stats in trainer.opponent_metrics.items():
                reward_text = f'{stats["mean_signed_reward"]:+.4f}' if stats['hands'] else 'N/A'
                print(f'Train {name}: {stats["hands"]} hands, mean reward {reward_text}', flush=True)
            final = trainer.hands == config.total_hands
            if trainer.hands - trainer.last_eval >= config.eval_every_hands or final:
                evaluation_started = perf_counter()
                evaluation_performance = {}
                reports = evaluate_many(trainer.model, opponents=config.eval_opponents,
                                  deals=config.eval_deals, seed=stream_seed(config.seed, 'validation'),
                                  bootstrap_seed=stream_seed(config.seed, 'validation-bootstrap'),
                                  bootstrap_samples=config.bootstrap_samples, threads=args.eval_threads,
                                  performance=evaluation_performance)
                row['evaluation_seconds'] = perf_counter() - evaluation_started
                row['evaluation_performances'] = evaluation_performance
                row['evaluations'] = {name: report.to_dict() for name, report in reports.items()}
                report = reports[config.selection_opponent]
                # Preserve legacy primary fields for existing analysis scripts.
                row['evaluation_performance'] = evaluation_performance[config.selection_opponent]
                row['evaluation'] = row['evaluations'][config.selection_opponent]
                metric = report.metrics[0]
                trainer.last_eval = trainer.hands
                for name, outcome in reports.items():
                    m = outcome.metrics[0]
                    print(f'Evaluation {name}: win {m.win_rate:.1%}, signed marks {m.mean_signed_marks:.4f}; '
                          f'{config.eval_deals} duplicate groups', flush=True)
                    print(outcome.summary(), flush=True)
                print(f'Checkpoint selection opponent: {config.selection_opponent}', flush=True)
                if metric.mean_signed_marks > trainer.best_score:
                    trainer.best_score = metric.mean_signed_marks
                    checkpoint_started = perf_counter()
                    save(checkpoints / 'best.pt', trainer)
                    row['checkpoint_seconds'] += perf_counter() - checkpoint_started
            if trainer.hands - trainer.last_checkpoint >= config.checkpoint_every_hands or final:
                trainer.last_checkpoint = trainer.hands
                checkpoint_started = perf_counter()
                save(checkpoints / 'latest.pt', trainer)
                row['checkpoint_seconds'] += perf_counter() - checkpoint_started
                print(f'Checkpoint: {checkpoints / "latest.pt"}', flush=True)
            row['update_seconds'] = perf_counter() - update_started
            print(f"Workers: {trainer.workers} | collected hands {trainer.hands - hands_before} | "
                  f"collection {row['collection_seconds']:.3f}s | optimization {row['optimization_seconds']:.3f}s | "
                  f"evaluation {row['evaluation_seconds']:.3f}s | checkpoint {row['checkpoint_seconds']:.3f}s | "
                  f"total update {row['update_seconds']:.3f}s | "
                  f"collection {row['collection_hands_per_second']:.1f} hands/s, "
                  f"{row['collection_decisions_per_second']:.1f} learned decisions/s", flush=True)
            with (directory / 'progress.jsonl').open('a') as sink:
                # Include this update's validation/checkpoint overhead in saved throughput.
                row['runtime_seconds'] = perf_counter() - started
                row['hands_per_second'] = (trainer.hands - initial_hands) / row['runtime_seconds']
                sink.write(json.dumps(row, allow_nan=False) + '\n')
    finally:
        trainer.close()
    print(f'Training complete: {trainer.hands} hands, {trainer.samples} play samples, '
          f'{trainer.updates} updates; best held-out signed marks {trainer.best_score:.4f}; '
          f'total elapsed {perf_counter() - started:.3f}s', flush=True)
    return trainer


if __name__ == '__main__':
    main()
