"""Version-2 phase checkpoints; version-1 play weights retain their original meaning."""
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .checkpoint import metadata as legacy_metadata, save as legacy_save
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
    if not set(phases(m['trained_phases'])) <= set(neural):
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
    config_from_dict(data['config'])
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
    # Reuse the established optimizer/RNG/Git capture, then atomically publish v2.
    staging = path.with_name(path.name + '.capture')
    legacy_save(staging, trainer)
    data = torch.load(staging, map_location='cpu', weights_only=True)
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
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    temporary.replace(path)
    staging.unlink()


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
    trainer.model.load_state_dict(data['model'], strict=True)
    trainer.model.trained_phases = tuple(data['metadata']['trained_phases'])
    trainer.optimizer.load_state_dict(data['optimizer'])
    trainer.snapshots = [spec_from_dict(s) for s in data['snapshots']]
    # Ensure every immutable dependency still exists and has its saved identity.
    from .selfplay_evaluation import frozen_model
    for spec in (trainer.config.anchor, *trainer.config.opponent_policies, *trainer.snapshots):
        if spec.kind == 'checkpoint':
            frozen_model(spec)
    for key, value in data['progress'].items():
        setattr(trainer, key, value)
    trainer.bc_metrics = data.get('bc_metrics')
    trainer.initial_checkpoint_sha256 = data['metadata'].get('initial_checkpoint_sha256')
    trainer.initial_checkpoint_path = data['metadata'].get('initial_checkpoint_path')
    rng = data['rng']
    random.setstate(rng['python'])
    state = rng['numpy']
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(rng['cpu'])
    if trainer.device.type == 'cuda' and rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])
    trainer.sampling.set_state(rng['sampling'])
    trainer.shuffle.set_state(rng['shuffle'])
