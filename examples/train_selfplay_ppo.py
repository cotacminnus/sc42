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
from research.ppo.multiphase import phases, PhaseActorCritic, PHASES, resolve_learner_phases
from research.ppo.opponents import PolicySpec, CheckpointIdentity, historical_pool, policy_registry
from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer, PairedSelfPlayTrainer
from research.ppo.selfplay_checkpoint import save, resume, config_from_dict, save_pair, resume_pair
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
    for phase in PHASES:
        parser.add_argument(f'--learner-{phase}', choices=('heuristic', 'neural-frozen', 'neural-train'))
        parser.add_argument(f'--learner-{phase}-source',
                            help='init, anchor, registered checkpoint, file path, or new (fresh heads)')
    parser.add_argument('--opponent-pool', default='HHH:0.5,anchor:0.5')
    parser.add_argument('--opponent-checkpoint', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--opponent-composition', action='append', default=[], metavar='NAME=JSON_PATH')
    parser.add_argument('--eval-opponents', default='HHH,CHH,anchor')
    parser.add_argument('--selection-opponent', default='anchor')
    parser.add_argument('--rollout-decisions', type=int, default=1024)
    parser.add_argument('--max-rollout-hands', type=int, default=4096,
                        help='maximum hands in a trump-only rollout, even when no samples occur')
    parser.add_argument('--snapshot-every-hands', type=int, default=32768)
    parser.add_argument('--snapshot-pool-size', type=int, default=8)
    parser.add_argument('--bid-bc-hands', type=int, default=0)
    parser.add_argument('--forgetting-hands', type=int, default=8)
    parser.add_argument('--paired-partners', action='store_true',
                        help='train two separate learner networks: partner 1 plays the team\'s first seat '
                             '(0/1), partner 2 its second seat (2/3); saved as checkpoints/partner-1 and partner-2')
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
        if args.resume and saved is None:
            raise ValueError('Resume requires a multi-phase checkpoint; use --init-play-checkpoint for migration')
        anchor_path = args.anchor_checkpoint or (Path(saved.anchor.path) if saved else source)
        anchor_spec = PolicySpec.checkpoint('anchor', anchor_path)
        anchor = frozen_model(anchor_spec)
        if saved is None and PolicySpec.checkpoint('initial', source).sha256 != anchor_spec.sha256:
            raise ValueError('Play-only initialization must use the same checkpoint as its anchor')
        if isinstance(anchor, PhaseActorCritic):
            raise ValueError('Anchor must be the original play-only policy, not a multi-phase checkpoint')
        if saved and saved.anchor.sha256 != anchor_spec.sha256:
            raise ValueError('New stage must retain its original play anchor identity')
        identity = CheckpointIdentity.from_path(source)
        initialization_identity = (CheckpointIdentity(data['metadata']['initial_checkpoint_path'],
            data['metadata']['initial_checkpoint_sha256']) if args.resume and
            data['metadata'].get('initial_checkpoint_path') else identity)
        registry = policy_registry(anchor_spec, args.opponent_checkpoint,
            reserved=('self-snapshots', 'init', 'new'), compositions=args.opponent_composition, initial=initialization_identity)
        if args.resume and not {'--opponent-checkpoint', '--opponent-composition'} & explicit:
            registry = {p.name: p for p in saved.opponent_policies}
        policies = tuple(registry.values())
        enabled = phases(args.learn_phases) if not args.resume or '--learn-phases' in explicit else saved.learn_phases
        phase_flags = {f'--learner-{p}{suffix}' for p in PHASES for suffix in ('', '-source')}
        configured = bool(phase_flags & explicit)
        learner = saved.learner_phases if args.resume else ()
        if configured:
            existing_neural = getattr(initial, 'neural_phases', ('play',))
            states = []
            for phase in PHASES:
                default = 'neural-frozen' if phase in existing_neural else 'heuristic'
                if '--learn-phases' in explicit and phase in enabled:
                    default = 'neural-train'
                if args.resume and saved.learner_phases:
                    default = saved.learner_phases[PHASES.index(phase)].state
                states.append(getattr(args, 'learner_' + phase) or default)
            selected = tuple(p for p, state in zip(PHASES, states) if state == 'neural-train')
            if '--learn-phases' in explicit and selected != enabled:
                raise ValueError('--learn-phases conflicts with --learner-* trainability; use one consistent configuration')
            if not selected:
                raise ValueError('Training requires at least one neural-train learner phase')
            enabled = phases(selected)
            sources = [getattr(args, 'learner_' + p + '_source') for p in PHASES]
            if args.resume:
                # Omitted sources preserve original initialization provenance, not the resume file.
                learner = tuple(saved.learner_phases)
                if not learner:
                    raise ValueError('Use legacy flags to resume a legacy configuration; initialize a new stage for learner source options')
                resolved = []
                for i, state in enumerate(states):
                    if sources[i] is None:
                        resolved.append(replace(learner[i], state=state))
                    else:
                        probe_states = [s.state for s in learner]
                        probe_states[i] = state
                        probe_sources = [('new' if s.initialization == 'representation' else s.source.path)
                                         if s.source else None for s in learner]
                        probe_sources[i] = sources[i]
                        resolved.append(resolve_learner_phases(probe_states, probe_sources, registry, initialization_identity)[i])
                learner = tuple(resolved)
            else:
                learner = resolve_learner_phases(states, sources, registry, identity)
        if learner:
            if args.play_policy and learner[2].state != 'neural-' + ('train' if args.play_policy == 'trainable' else 'frozen'):
                raise ValueError('--play-policy conflicts with learner play state')
            if '--trump-policy' in explicit and (args.trump_policy == 'neural') != (learner[1].state != 'heuristic'):
                raise ValueError('--trump-policy conflicts with learner trump state')
        else:
            if args.play_policy and (args.play_policy == 'trainable') != ('play' in enabled):
                raise ValueError('play-policy must agree with learn-phases')
            if '--trump-policy' in explicit and (args.trump_policy == 'neural') != (
                    'trump' in enabled or 'trump' in getattr(initial, 'neural_phases', ())):
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
            forgetting_hands=args.forgetting_hands, sampling_scheme='episode-v1',
            learner_phases=learner, max_rollout_hands=args.max_rollout_hands,
            partner_mode='paired' if args.paired_partners else 'shared')
        if args.resume:
            if saved is None:
                raise ValueError('Resume requires a multi-phase checkpoint; use --init-play-checkpoint for migration')
            # Omitted options restore saved settings; explicit changes undergo strict validation.
            changes = {f.name: getattr(config, f.name) for f in fields(SelfPlayConfig)
                       if '--' + f.name.replace('_', '-') in explicit}
            for flag, key in (('--opponent-pool', 'opponent_probabilities'),
                              ('--opponent-checkpoint', 'opponent_policies'),
                              ('--opponent-composition', 'opponent_policies'),
                              ('--eval-opponents', 'evaluation_names'), ('--selection-opponent', 'primary_opponent'),
                              ('--paired-partners', 'partner_mode')):
                if flag in explicit:
                    changes[key] = getattr(config, key)
            if configured:
                changes.update(learner_phases=learner, learn_phases=enabled)
            config = replace(saved, **changes)
        if not config.learner_phases and config.learn_phases == ('bid',) and isinstance(initial, PhaseActorCritic):
            if not all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(
                    initial.networks['play'].parameters(), anchor.parameters())):
                raise ValueError('Bidding-only anchor comparisons require identical pretrained playing weights; initialize from play-only or Stage A')
        active = {n for n, _ in config.opponent_probabilities} | set(config.evaluation_names) | {'anchor'}
        for spec in config.opponent_policies:
            if spec.name not in active:
                continue  # An archive used only as a phase source need not expose every phase.
            factory = spec.factory()
            if hasattr(factory, 'prepare'):
                factory.prepare()
        # Paired mode (saved in the config, so resume keeps it): two learner
        # networks. `learners` lists every network-owning trainer, partner 1 first.
        paired = config.partner_mode == 'paired'
        trainer = (PairedSelfPlayTrainer if paired else SelfPlayTrainer)(config, initial)
        learners = (trainer, trainer.partner) if paired else (trainer,)
        if args.resume:
            (resume_pair if paired else resume)(args.resume, trainer)
        elif args.bid_bc_hands:
            for index, learner in enumerate(learners, start=1):
                bc = clone_bidding(learner, args.bid_bc_hands)
                who = f' (partner {index})' if paired else ''
                print(f'Bidding imitation{who}: {bc["samples"]} samples; loss {bc["loss"]:.4f}; accuracy {bc["accuracy"]:.1%}')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    def save_checkpoint(name):
        # Shared: checkpoints/<name>. Paired: checkpoints/partner-{1,2}/<name>.
        return save_pair(checkpoints, name, trainer) if paired else save(checkpoints / name, trainer)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    checkpoints = directory / 'checkpoints'
    started, initial_hands = perf_counter(), trainer.hands
    validation_seed = stream_seed(config.seed, 'selfplay-validation-v1')
    bootstrap_seed = stream_seed(config.seed, 'selfplay-validation-bootstrap-v1')
    reference = teacher_data(config.seed, config.forgetting_hands, anchor=anchor)
    # One snapshot per learner network; frozen phases must stay bitwise equal in every partner.
    frozen_states = [{p: {k: v.detach().cpu().clone() for k, v in t.model.networks[p].state_dict().items()}
                      for p in t.model.neural_phases if p not in config.learn_phases} for t in learners]
    print(f'Multi-phase PPO: learn {",".join(config.learn_phases)}, neural {",".join(trainer.model.neural_phases)}')
    if paired:
        print('Partners: paired (partner 1 = team first seat 0/1, partner 2 = second seat 2/3; '
              'no historical snapshots)')
    for phase, spec in zip(PHASES, trainer.learner_phases):
        print(f'  {phase}: {spec.state}; initialization={spec.initialization}; source={spec.source}')
    print(f'Anchor: {config.anchor.path}\nAnchor SHA256: {config.anchor.sha256}')
    print(f'Device: {trainer.device}; workers: {trainer.workers}; torch threads: {config.torch_threads}; validation workers: {resolve_workers(args.eval_threads)}')
    print(f'Primary selection opponent: {config.primary_opponent}; training pool: {config.opponent_probabilities}')
    try:
        while trainer.hands < config.total_hands:
            before, update_started = trainer.hands, perf_counter()
            # Paired collect() returns one buffer per partner; update() trains each separately.
            buffer, reward = trainer.collect()
            sample_count = sum(len(b) for b in buffer) if paired else len(buffer)
            trainer.optimization_seconds = 0.0
            metrics = trainer.update(buffer) if sample_count else {}
            frozen_verified = {p: all(torch.equal(v.detach().cpu(), states[p][k])
                for t, states in zip(learners, frozen_states)
                for k, v in t.model.networks[p].state_dict().items()) for p in frozen_states[0]}
            if not all(frozen_verified.values()):
                raise RuntimeError('Frozen phase parameters changed during training')
            frozen_play_verified = frozen_verified.get('play')
            final = trainer.hands == config.total_hands
            total_samples = sum(t.samples for t in learners)  # Both partners when paired.
            row = dict(hands=trainer.hands, play_samples=total_samples, training_samples=total_samples,
                updates=trainer.updates, optimizer_steps=trainer.optimizer_steps, mean_reward=reward,
                phase_samples=trainer.phase_counts, training_by_opponent=trainer.opponent_metrics,
                train_opponents=config.opponent_probabilities, opponent_assignments=trainer.opponent_assignments,
                selection_opponent=config.primary_opponent, metrics=metrics, bc_metrics=trainer.bc_metrics,
                anchor_sha256=trainer.anchor_sha256, frozen_play_verified=frozen_play_verified,
                learner_phases=[asdict(s) for s in trainer.learner_phases],
                frozen_phases_verified=frozen_verified,
                playing_anchor_kl=playing_kl(trainer.model, anchor, reference) if 'play' in trainer.model.neural_phases else None,
                collection_seconds=trainer.collection_seconds, optimization_seconds=trainer.optimization_seconds,
                rollout_performance=trainer.rollout_performance, execution=trainer.execution_metadata(),
                collection_hands_per_second=(trainer.hands - before) / trainer.collection_seconds,
                collection_decisions_per_second=sample_count / trainer.collection_seconds,
                evaluation_seconds=0.0, checkpoint_seconds=0.0)
            if paired:
                # Top-level fields keep partner 1 / combined values; per-partner detail here.
                # metrics is {'partner_1': ..., 'partner_2': ...} (a partner without samples is absent).
                row['partners'] = {f'partner_{i}': dict(
                    samples=t.samples, updates=t.updates, optimizer_steps=t.optimizer_steps,
                    phase_samples=trainer.partner_phase_counts[i - 1],
                    metrics=metrics.get(f'partner_{i}', {}),
                    playing_anchor_kl=playing_kl(t.model, anchor, reference) if 'play' in t.model.neural_phases else None)
                    for i, t in enumerate(learners, start=1)}
            print(f'Update {trainer.updates}: {trainer.hands}/{config.total_hands} hands; samples {trainer.phase_counts}; reward {reward:+.4f}')
            groups = ([(f'partner {i} ', metrics.get(f'partner_{i}', {})) for i in (1, 2)]
                      if paired else [('', metrics)])
            for label, values in groups:
                for phase in config.learn_phases:
                    print(f'  {label}{phase}: policy {values.get(phase + "_policy_loss", 0):.5f}, entropy {values.get(phase + "_entropy", 0):.4f}, KL {values.get(phase + "_approx_kl", 0):.6f}')
            if (not paired and sample_count and set(trainer.model.neural_phases) <= set(trainer.model.trained_phases)
                    and trainer.hands - trainer.last_snapshot >= config.snapshot_every_hands):
                spec = trainer.publish_snapshot(directory / 'snapshots')
                print(f'Historical snapshot: {spec.name} SHA256 {spec.sha256}')
            if trainer.hands - trainer.last_eval >= config.eval_every_hands or final:
                evaluation_started = perf_counter()
                telemetry = {}
                specs = tuple(next(p for p in config.opponent_policies if p.name == n) for n in config.evaluation_names)
                # Paired: the two partners validate together as one team (seat-aware factory).
                reports = evaluate_phases(trainer.model, specs, deals=config.eval_deals, seed=validation_seed,
                    bootstrap_seed=bootstrap_seed, bootstrap_samples=config.bootstrap_samples,
                    threads=args.eval_threads, performance=telemetry,
                    partner_model=trainer.partner.model if paired else None)
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
                    saving_started = perf_counter(); save_checkpoint('best.pt')
                    row['checkpoint_seconds'] += perf_counter() - saving_started
            if trainer.hands - trainer.last_checkpoint >= config.checkpoint_every_hands or final:
                trainer.last_checkpoint = trainer.hands
                saving_started = perf_counter(); save_checkpoint('latest.pt')
                row['checkpoint_seconds'] += perf_counter() - saving_started
            row['update_seconds'] = perf_counter() - update_started
            row['runtime_seconds'] = perf_counter() - started
            row['hands_per_second'] = (trainer.hands - initial_hands) / row['runtime_seconds']
            kl_text = 'N/A (heuristic play)' if row['playing_anchor_kl'] is None else f'{row["playing_anchor_kl"]:.6g}'
            print(f'Collection {trainer.collection_seconds:.3f}s ({row["collection_hands_per_second"]:.1f} hands/s); optimization {trainer.optimization_seconds:.3f}s; validation {row["evaluation_seconds"]:.3f}s; anchor play KL {kl_text}')
            with (directory / 'progress.jsonl').open('a') as sink:
                sink.write(json.dumps(row, allow_nan=False) + '\n')
    finally:
        trainer.close()
    print(f'Complete: {trainer.hands} hands, {sum(t.samples for t in learners)} training decisions, {trainer.updates} updates; elapsed {perf_counter() - started:.3f}s')
    if paired:
        print(f'Paired models: {checkpoints / "partner-1"} (seat 0/1) and {checkpoints / "partner-2"} (seat 2/3)')
    return trainer


if __name__ == '__main__':
    main()
