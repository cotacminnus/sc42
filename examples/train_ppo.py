"""Train masked PPO trick play against frozen HHH using complete hand episodes."""
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
from time import perf_counter
from research.ppo.trainer import Config, Trainer, stream_seed
from research.ppo.checkpoint import save, resume
from research.ppo.evaluation import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = Config()
    for field in fields(Config):
        value = getattr(default, field.name)
        parser.add_argument('--' + field.name.replace('_', '-'), type=type(value), default=value)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/ppo-play-v1'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    try:
        config = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
        trainer = Trainer(config)
        if args.resume:
            resume(args.resume, trainer)
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
    print('PPO trick-play training: heuristic auction/trump, neural partnership versus HHH', flush=True)
    while trainer.hands < config.total_hands:
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
        final = trainer.hands == config.total_hands
        if trainer.hands - trainer.last_eval >= config.eval_every_hands or final:
            report = evaluate(trainer.model, deals=config.eval_deals,
                              seed=stream_seed(config.seed, 'validation'),
                              bootstrap_seed=stream_seed(config.seed, 'validation-bootstrap'),
                              bootstrap_samples=config.bootstrap_samples)
            metric = report.metrics[0]
            row['evaluation'] = report.to_dict()
            trainer.last_eval = trainer.hands
            print(f'Evaluation HHH: win {metric.win_rate:.1%}, signed marks {metric.mean_signed_marks:.4f}; '
                  f'{config.eval_deals} duplicate groups', flush=True)
            print(report.summary(), flush=True)
            if metric.mean_signed_marks > trainer.best_score:
                trainer.best_score = metric.mean_signed_marks
                save(checkpoints / 'best.pt', trainer)
        if trainer.hands - trainer.last_checkpoint >= config.checkpoint_every_hands or final:
            trainer.last_checkpoint = trainer.hands
            save(checkpoints / 'latest.pt', trainer)
            print(f'Checkpoint: {checkpoints / "latest.pt"}', flush=True)
        with (directory / 'progress.jsonl').open('a') as sink:
            # Include this update's validation/checkpoint overhead in saved throughput.
            row['runtime_seconds'] = perf_counter() - started
            row['hands_per_second'] = (trainer.hands - initial_hands) / row['runtime_seconds']
            sink.write(json.dumps(row, allow_nan=False) + '\n')
    return trainer


if __name__ == '__main__':
    main()
