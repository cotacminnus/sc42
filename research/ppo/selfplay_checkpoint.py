"""Version-2 phase checkpoints; version-1 play weights retain their original meaning."""
from dataclasses import asdict
from pathlib import Path
import torch
from .checkpoint import metadata as legacy_metadata, capture, restore_rng
from .multiphase import phases
from .opponents import PolicySpec


def spec_from_dict(value):
    return PolicySpec(**value)


def validate(data):
    m = data.get('metadata', {})
    expected = legacy_metadata()
    for key in ('feature_size', 'num_actions', 'feature_schema_version', 'encoder_layout_version',
                'action_space_version', 'ruleset_id'):
        if m.get(key) != expected[key]:
            raise ValueError(f'Incompatible multi-phase checkpoint {key}')
    if m.get('checkpoint_version') != 2 or m.get('architecture') != 'independent-actor-critic-v1':
        raise ValueError('Incompatible phase checkpoint architecture/version')
    neural, learn = phases(m['neural_phases']), phases(data['config']['learn_phases'])
    if (not set(learn) <= set(neural) or tuple(m.get('learn_phases', ())) != learn
            or tuple(m.get('frozen_phases', ())) != tuple(p for p in neural if p not in learn)):
        raise ValueError('Incompatible phase configuration')
    if m.get('opponent_selection_scheme') != 'multiphase-opponent-v1' or m.get('snapshot_selection_scheme') != 'historical-snapshot-v1':
        raise ValueError('Incompatible historical opponent sampling scheme')
    trained = phases(m['trained_phases']) if m['trained_phases'] else ()
    if not set(trained) <= set(neural):
        raise ValueError('Incompatible trained-phase metadata')
    if m.get('original_play_sha256') != data['config']['anchor']['sha256']:
        raise ValueError('Incompatible original play identity')
    if data.get('execution', {}).get('sampling_scheme') != 'multiphase-episode-v1':
        raise ValueError('Incompatible multi-phase sampling scheme')
    if m.get('checkpoint_selection_opponent') != data['config']['primary_opponent']:
        raise ValueError('Incompatible checkpoint selection provenance')
    m.setdefault('network_configuration', {p: dict(hidden_size=data['config']['hidden_size'],
        feature_size=expected['feature_size'], num_actions=expected['num_actions']) for p in ('bid', 'trump', 'play')})
    if m.get('network_configuration') != {p: dict(hidden_size=data['config']['hidden_size'],
            feature_size=expected['feature_size'], num_actions=expected['num_actions']) for p in ('bid', 'trump', 'play')}:
        raise ValueError('Incompatible phase network dimensions')
    # Validate and canonicalize specifications, including historical membership.
    config = config_from_dict(data['config'])
    if 'learner_phases' in m:
        from .multiphase import LearnerPhaseSpec, PHASES
        specs = tuple(LearnerPhaseSpec(**s) for s in m['learner_phases'])
        if (len(specs) != 3 or tuple(p for p,s in zip(PHASES,specs) if s.state != 'heuristic') != neural
                or tuple(p for p,s in zip(PHASES,specs) if s.state == 'neural-train') != learn):
            raise ValueError('Incompatible learner phase metadata')
    if config.learner_phases:
        from .multiphase import PHASES
        if tuple(p for p, s in zip(PHASES, config.learner_phases) if s.state != 'heuristic') != neural:
            raise ValueError('Incompatible resolved learner phases')
        if m.get('learner_phases') != [asdict(s) for s in config.learner_phases]:
            raise ValueError('Incompatible learner source provenance')
    for spec in data['snapshots']:
        spec_from_dict(spec)


def config_from_dict(data):
    from .selfplay_trainer import SelfPlayConfig
    values = dict(data)
    values['anchor'] = spec_from_dict(values['anchor'])
    values['opponent_policies'] = tuple(spec_from_dict(s) for s in values['opponent_policies'])
    for name in ('learn_phases', 'evaluation_names', 'eval_opponents'):
        values[name] = tuple(values[name])
    return SelfPlayConfig(**values)


def save(path, trainer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Capture once; no intermediate version-1 file or optimizer/RNG reload.
    data = capture(trainer)
    neural = trainer.model.neural_phases
    for key in ('trained_phase', 'training_opponent', 'training_opponents', 'opponent_selection_scheme'):
        data['metadata'].pop(key, None)
    data['metadata'].update(checkpoint_version=2, architecture='independent-actor-critic-v1',
        neural_phases=neural, learn_phases=trainer.config.learn_phases,
        network_configuration={p: dict(hidden_size=trainer.model.hidden_size, feature_size=data['metadata']['feature_size'],
            num_actions=data['metadata']['num_actions']) for p in ('bid', 'trump', 'play')},
        trained_phases=getattr(trainer.model, 'trained_phases', ('play',)), frozen_phases=tuple(p for p in neural if p not in trainer.config.learn_phases),
        original_play_sha256=trainer.anchor_sha256,
        initial_checkpoint_sha256=trainer.initial_checkpoint_sha256,
        initial_checkpoint_path=trainer.initial_checkpoint_path, checkpoint_selection_opponent=trainer.config.primary_opponent,
        opponent_distribution=trainer.config.opponent_probabilities,
        opponent_selection_scheme='multiphase-opponent-v1',
        snapshot_selection_scheme='historical-snapshot-v1')
    data['snapshots'] = [asdict(spec) for spec in trainer.snapshots]
    data['progress']['last_snapshot'] = trainer.last_snapshot
    data['bc_metrics'] = trainer.bc_metrics
    data['metadata']['learner_phases'] = [asdict(s) for s in trainer.learner_phases]
    data['metadata']['optimizer_parameters'] = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    if getattr(trainer, 'pairing', None):
        # Paired-partner provenance only. The file stays an ordinary version-2
        # checkpoint: evaluation, opponents and stage initialization load it alone.
        data['metadata']['pairing'] = dict(trainer.pairing)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def resume(path, trainer):
    from .checkpoint import read
    data = read(path)
    if data['metadata']['checkpoint_version'] != 2:
        raise ValueError('Play-only checkpoints initialize a new stage; they cannot exactly resume multi-phase training')
    saved = asdict(config_from_dict(data['config']))
    current = asdict(trainer.config)
    for key, value in saved.items():
        if key not in ('total_hands', 'device', 'threads') and current[key] != value:
            raise ValueError(f'Exact resume configuration mismatch: {key}; initialize a new stage instead')
    if tuple(data['metadata']['neural_phases']) != trainer.model.neural_phases:
        raise ValueError('Exact resume neural phase architecture mismatch')
    if trainer.device.type != data['sampling_device']:
        raise ValueError('Exact resume requires the same optimizer device type')
    if trainer.config.total_hands < data['progress']['hands']:
        raise ValueError('total_hands precedes saved progress')
    membership = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    if data['metadata'].get('optimizer_parameters', membership) != membership:
        raise ValueError('Exact resume optimizer phase membership mismatch')
    trainer.model.load_state_dict(data['model'], strict=True)
    trainer.model.trained_phases = tuple(data['metadata']['trained_phases'])
    trainer.optimizer.load_state_dict(data['optimizer'])
    trainer.snapshots = [spec_from_dict(s) for s in data['snapshots']]
    # Ensure every immutable dependency still exists and has its saved identity.
    from .checkpoint import load_frozen
    active = {n for n, _ in trainer.config.opponent_probabilities} | set(trainer.config.evaluation_names)
    for spec in (trainer.config.anchor, *trainer.config.opponent_policies, *trainer.snapshots):
        for identity in spec.identities():
            load_frozen(identity, verify=True)
        if spec.kind != 'variant' and (spec.name in active or spec in trainer.snapshots or spec == trainer.config.anchor):
            spec.factory().prepare()
    for spec in trainer.config.learner_phases:
        if spec.source:
            load_frozen(spec.source, verify=True)
    for key, value in data['progress'].items():
        setattr(trainer, key, value)
    trainer.bc_metrics = data.get('bc_metrics')
    trainer.initial_checkpoint_sha256 = data['metadata'].get('initial_checkpoint_sha256')
    trainer.initial_checkpoint_path = data['metadata'].get('initial_checkpoint_path')
    if 'learner_phases' in data['metadata']:
        from .multiphase import LearnerPhaseSpec
        trainer.learner_phases = tuple(LearnerPhaseSpec(**s) for s in data['metadata']['learner_phases'])
    restore_rng(data['rng'], trainer)


# Paired-partner checkpoints. A pair is two ordinary version-2 files saved side
# by side as <checkpoints>/partner-1/<name> and <checkpoints>/partner-2/<name>,
# each tagged with metadata['pairing']. Either file can be used alone anywhere a
# checkpoint is accepted ("paired model 1/2"); exact resume needs both.
PARTNER_DIRECTORIES = ('partner-1', 'partner-2')


def save_pair(directory, name, trainer):
    """Save a PairedSelfPlayTrainer's two partners with the existing save()."""
    directory = Path(directory)
    save(directory / PARTNER_DIRECTORIES[0] / name, trainer)
    save(directory / PARTNER_DIRECTORIES[1] / name, trainer.partner)
    return tuple(directory / folder / name for folder in PARTNER_DIRECTORIES)


def pair_paths(path):
    """Given either partner file, return (partner-1 path, partner-2 path)."""
    from .checkpoint import read
    path = Path(path)
    pairing = read(path)['metadata'].get('pairing')
    if not pairing or pairing.get('mode') != 'paired-v1':
        raise ValueError(f'{path} is not a paired-partner checkpoint')
    root = path.parent.parent
    return tuple(root / folder / path.name for folder in PARTNER_DIRECTORIES)


def resume_pair(path, trainer):
    """Exactly resume both partners of a PairedSelfPlayTrainer from one saved pair.

    Checks that the two files are partners 1 and 2 of the same run (identical
    configuration and hand count) before restoring each with resume().
    """
    from .checkpoint import read
    first, second = pair_paths(path)
    a, b = read(first), read(second)
    if [a['metadata'].get('pairing', {}).get('partner'), b['metadata'].get('pairing', {}).get('partner')] != [1, 2]:
        raise ValueError('Paired checkpoint files are not partners 1 and 2')
    if a['config'] != b['config'] or a['progress']['hands'] != b['progress']['hands']:
        raise ValueError('Paired checkpoint files come from different runs or update boundaries')
    resume(first, trainer)
    resume(second, trainer.partner)
    return first, second
