"""Independent-phase PPO with frozen play and immutable historical self-play."""
import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys
from time import perf_counter
import torch
from examples.experiment_output import record_output, add_output_arguments, add_config_arguments
from research.ppo.trainer import Config
from research.ppo.checkpoint import load_model
from research.ppo.multiphase import phases, PhaseActorCritic
from research.ppo.opponents import PolicySpec, historical_pool, policy_registry
from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
from research.ppo.selfplay_checkpoint import save, resume, config_from_dict
from research.ppo.selfplay_evaluation import evaluate_phases, frozen_model, evaluation_diagnostics
from research.ppo.imitation import clone_bidding, teacher_data, playing_kl
from research.ppo.seeds import stream_seed
from texas42.parallel import resolve_workers


@record_output
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config(learning_rate=1e-4)
    skipped = {'train_opponents', 'eval_opponents', 'selection_opponent', 'rollout_play_steps', 'sampling_scheme'}
    add_config_arguments(parser, defaults, exclude=skipped)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--init-play-checkpoint', type=Path)
    source.add_argument('--init-checkpoint', type=Path, help='New stage: weights only, fresh optimizer/RNG')
    source.add_argument('--resume', type=Path, help='Exact update-boundary continuation')
    parser.add_argument('--anchor-checkpoint', type=Path)
    parser.add_argument('--learn-phases', default='bid')
    parser.add_argument('--play-policy', choices=('frozen', 'trainable'), help='Validated against learn-phases')
    parser.add_argument('--trump-policy', choices=('heuristic', 'neural'), default='heuristic')
    parser.add_argument('--opponent-pool', default='HHH:0.5,anchor:0.5')
    parser.add_argument('--opponent-checkpoint', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--eval-opponents', default='HHH,CHH,anchor')
    parser.add_argument('--selection-opponent', default='anchor')
    parser.add_argument('--rollout-decisions', type=int, default=1024)
    parser.add_argument('--snapshot-every-hands', type=int, default=32768)
    parser.add_argument('--snapshot-pool-size', type=int, default=8)
    parser.add_argument('--bid-bc-hands', type=int, default=0)
    parser.add_argument('--forgetting-hands', type=int, default=8)
    parser.add_argument('--eval-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/ppo-selfplay'))
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    arguments = list(sys.argv[1:] if argv is None else argv)
    explicit = {arg.split('=')[0] for arg in arguments if arg.startswith('--')}
    directory = args.output_dir
    if (directory / 'progress.jsonl').exists() and not args.resume:
        parser.error('Run already exists; choose another output directory or --resume')
    try:
        resolve_workers(args.eval_threads)
        source = args.resume or args.init_checkpoint or args.init_play_checkpoint or Path('runs/ppo-play-v1/checkpoints/best.pt')
        initial, data = load_model(source)
        if args.init_play_checkpoint and isinstance(initial, PhaseActorCritic):
            raise ValueError('--init-play-checkpoint requires an original play-only checkpoint')
        saved = config_from_dict(data['config']) if data['metadata']['checkpoint_version'] == 2 else None
        anchor_path = args.anchor_checkpoint or (Path(saved.anchor.path) if saved else source)
        anchor_spec = PolicySpec.checkpoint('anchor', anchor_path)
        anchor = frozen_model(anchor_spec)
        if saved is None and PolicySpec.checkpoint('initial', source).sha256 != anchor_spec.sha256:
            raise ValueError('Play-only initialization must use the same checkpoint as its anchor')
        if isinstance(anchor, PhaseActorCritic):
            raise ValueError('Anchor must be the original play-only policy, not a multi-phase checkpoint')
        if saved and saved.anchor.sha256 != anchor_spec.sha256:
            raise ValueError('New stage must retain its original play anchor identity')
        policies = tuple(policy_registry(anchor_spec, args.opponent_checkpoint,
                                         reserved=('self-snapshots',)).values())
        enabled = phases(args.learn_phases)
        if args.play_policy and (args.play_policy == 'trainable') != ('play' in enabled):
            raise ValueError('play-policy must agree with learn-phases')
        if (args.trump_policy == 'neural') != ('trump' in enabled):
            raise ValueError('Neural trump requires trump in learn-phases; heuristic trump cannot be trained')
        pool = historical_pool(args.opponent_pool, policies, anchor_spec)
        base = {f.name: getattr(args, f.name) for f in fields(Config) if f.name not in skipped}
        if '--hidden-size' not in explicit:
            base['hidden_size'] = initial.hidden_size
        config = SelfPlayConfig(**base, learn_phases=enabled, opponent_probabilities=pool.probabilities,
            opponent_policies=tuple(policies), anchor=anchor_spec,
            evaluation_names=tuple(args.eval_opponents.split(',')), primary_opponent=args.selection_opponent,
            rollout_decisions=args.rollout_decisions, snapshot_every_hands=args.snapshot_every_hands,
            snapshot_pool_size=args.snapshot_pool_size, bid_bc_hands=args.bid_bc_hands,
            forgetting_hands=args.forgetting_hands, sampling_scheme='episode-v1')
        if args.resume:
            if saved is None:
                raise ValueError('Resume requires a multi-phase checkpoint; use --init-play-checkpoint for migration')
            # Omitted options restore saved settings; explicit changes undergo strict validation.
            changes = {f.name: getattr(config, f.name) for f in fields(SelfPlayConfig)
                       if '--' + f.name.replace('_', '-') in explicit}
            for flag, key in (('--opponent-pool', 'opponent_probabilities'),
                              ('--opponent-checkpoint', 'opponent_policies'),
                              ('--eval-opponents', 'evaluation_names'), ('--selection-opponent', 'primary_opponent')):
                if flag in explicit:
                    changes[key] = getattr(config, key)
            config = replace(saved, **changes)
        if config.learn_phases == ('bid',) and isinstance(initial, PhaseActorCritic):
            if not all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(
                    initial.networks['play'].parameters(), anchor.parameters())):
                raise ValueError('Bidding-only anchor comparisons require identical pretrained playing weights; initialize from play-only or Stage A')
        trainer = SelfPlayTrainer(config, initial)
        if args.resume:
            resume(args.resume, trainer)
        elif args.bid_bc_hands:
            bc = clone_bidding(trainer, args.bid_bc_hands)
            print(f'Bidding imitation: {bc["samples"]} samples; loss {bc["loss"]:.4f}; accuracy {bc["accuracy"]:.1%}')
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    checkpoints = directory / 'checkpoints'
    started, initial_hands = perf_counter(), trainer.hands
    validation_seed = stream_seed(config.seed, 'selfplay-validation-v1')
    bootstrap_seed = stream_seed(config.seed, 'selfplay-validation-bootstrap-v1')
    reference = teacher_data(config.seed, config.forgetting_hands, anchor=anchor)
    frozen_play_state = {k: v.detach().cpu().clone() for k, v in trainer.model.networks['play'].state_dict().items()}
    print(f'Multi-phase PPO: learn {",".join(config.learn_phases)}, neural {",".join(trainer.model.neural_phases)}')
    print(f'Anchor: {config.anchor.path}\nAnchor SHA256: {config.anchor.sha256}')
    print(f'Device: {trainer.device}; workers: {trainer.workers}; torch threads: {config.torch_threads}; validation workers: {resolve_workers(args.eval_threads)}')
    print(f'Primary selection opponent: {config.primary_opponent}; training pool: {config.opponent_probabilities}')
    try:
        while trainer.hands < config.total_hands:
            before, update_started = trainer.hands, perf_counter()
            buffer, reward = trainer.collect()
            sample_count = len(buffer)
            metrics = trainer.update(buffer)
            frozen_play_verified = None
            if 'play' not in config.learn_phases:
                frozen_play_verified = all(torch.equal(v.detach().cpu(), frozen_play_state[k])
                    for k, v in trainer.model.networks['play'].state_dict().items())
                if not frozen_play_verified:
                    raise RuntimeError('Frozen playing parameters changed during bidding training')
            final = trainer.hands == config.total_hands
            row = dict(hands=trainer.hands, play_samples=trainer.samples, training_samples=trainer.samples,
                updates=trainer.updates, optimizer_steps=trainer.optimizer_steps, mean_reward=reward,
                phase_samples=trainer.phase_counts, training_by_opponent=trainer.opponent_metrics,
                train_opponents=config.opponent_probabilities, opponent_assignments=trainer.opponent_assignments,
                selection_opponent=config.primary_opponent, metrics=metrics, bc_metrics=trainer.bc_metrics,
                anchor_sha256=trainer.anchor_sha256, frozen_play_verified=frozen_play_verified, playing_anchor_kl=playing_kl(trainer.model, anchor, reference),
                collection_seconds=trainer.collection_seconds, optimization_seconds=trainer.optimization_seconds,
                rollout_performance=trainer.rollout_performance, execution=trainer.execution_metadata(),
                collection_hands_per_second=(trainer.hands - before) / trainer.collection_seconds,
                collection_decisions_per_second=sample_count / trainer.collection_seconds,
                evaluation_seconds=0.0, checkpoint_seconds=0.0)
            print(f'Update {trainer.updates}: {trainer.hands}/{config.total_hands} hands; samples {trainer.phase_counts}; reward {reward:+.4f}')
            for phase in config.learn_phases:
                print(f'  {phase}: policy {metrics.get(phase + "_policy_loss", 0):.5f}, entropy {metrics.get(phase + "_entropy", 0):.4f}, KL {metrics.get(phase + "_approx_kl", 0):.6f}')
            if trainer.hands - trainer.last_snapshot >= config.snapshot_every_hands:
                spec = trainer.publish_snapshot(directory / 'snapshots')
                print(f'Historical snapshot: {spec.name} SHA256 {spec.sha256}')
            if trainer.hands - trainer.last_eval >= config.eval_every_hands or final:
                evaluation_started = perf_counter()
                telemetry = {}
                specs = tuple(next(p for p in config.opponent_policies if p.name == n) for n in config.evaluation_names)
                reports = evaluate_phases(trainer.model, specs, deals=config.eval_deals, seed=validation_seed,
                    bootstrap_seed=bootstrap_seed, bootstrap_samples=config.bootstrap_samples,
                    threads=args.eval_threads, performance=telemetry)
                row['evaluation_seconds'] = perf_counter() - evaluation_started
                row['evaluations'] = {n: r.to_dict() for n, r in reports.items()}
                row['evaluation_performances'] = telemetry
                row['evaluation_diagnostics'] = {n: evaluation_diagnostics(r) for n, r in reports.items()}
                row['evaluation_identities'] = {s.name: s.sha256 for s in specs}
                row['evaluation'] = row['evaluations'][config.primary_opponent]
                row['evaluation_performance'] = telemetry[config.primary_opponent]
                trainer.last_eval = trainer.hands
                for name, report in reports.items():
                    print(f'Evaluation {name}\n{report.summary()}')
                score = reports[config.primary_opponent].metrics[0].mean_signed_marks
                if score > trainer.best_score:
                    trainer.best_score = score
                    saving_started = perf_counter(); save(checkpoints / 'best.pt', trainer)
                    row['checkpoint_seconds'] += perf_counter() - saving_started
            if trainer.hands - trainer.last_checkpoint >= config.checkpoint_every_hands or final:
                trainer.last_checkpoint = trainer.hands
                saving_started = perf_counter(); save(checkpoints / 'latest.pt', trainer)
                row['checkpoint_seconds'] += perf_counter() - saving_started
            row['update_seconds'] = perf_counter() - update_started
            row['runtime_seconds'] = perf_counter() - started
            row['hands_per_second'] = (trainer.hands - initial_hands) / row['runtime_seconds']
            print(f'Collection {trainer.collection_seconds:.3f}s ({row["collection_hands_per_second"]:.1f} hands/s); optimization {trainer.optimization_seconds:.3f}s; validation {row["evaluation_seconds"]:.3f}s; anchor play KL {row["playing_anchor_kl"]:.6g}')
            with (directory / 'progress.jsonl').open('a') as sink:
                sink.write(json.dumps(row, allow_nan=False) + '\n')
    finally:
        trainer.close()
    print(f'Complete: {trainer.hands} hands, {trainer.samples} training decisions, {trainer.updates} updates; elapsed {perf_counter() - started:.3f}s')
    return trainer


if __name__ == '__main__':
    main()
