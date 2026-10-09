"""Version-checked, weights-only checkpoints and exact update-boundary resumption."""
from dataclasses import asdict
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from texas42.actions import ACTION_SPACE_VERSION, NUM_ACTIONS
from texas42.encoding import FEATURE_SIZE
from texas42.features import FEATURE_SCHEMA_VERSION, ENCODER_LAYOUT_VERSION
from texas42.rulesets import RULESET_ID
from .model import ActorCritic


def metadata():
    return dict(checkpoint_version=1, feature_size=FEATURE_SIZE, num_actions=NUM_ACTIONS,
                feature_schema_version=FEATURE_SCHEMA_VERSION, encoder_layout_version=ENCODER_LAYOUT_VERSION,
                action_space_version=ACTION_SPACE_VERSION, ruleset_id=RULESET_ID,
                training_opponent='HHH', trained_phase='play')


def save(path, trainer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    numpy_state = np.random.get_state()
    data = dict(metadata=metadata(), execution=trainer.execution_metadata(), config=asdict(trainer.config), model=trainer.model.state_dict(),
                optimizer=trainer.optimizer.state_dict(), git_sha=sha, git_dirty=dirty,
                torch_version=str(torch.__version__), progress={name: getattr(trainer, name) for name in
                ('hands', 'samples', 'updates', 'optimizer_steps', 'best_score', 'last_eval', 'last_checkpoint')},
                rng=dict(python=random.getstate(), numpy=[numpy_state[0], numpy_state[1].tolist(),
                         *numpy_state[2:]], cpu=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                         sampling=trainer.sampling.get_state(), shuffle=trainer.shuffle.get_state()),
                sampling_device=trainer.device.type)
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def read(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in metadata().items():
        if data.get('metadata', {}).get(key) != value:
            raise ValueError(f'Incompatible checkpoint {key}: expected {value!r}')
    return data


def load_model(path, device='cpu'):
    data = read(path)
    # Loading an evaluation model must not perturb training's global RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = ActorCritic(data['config']['hidden_size'])
    model.load_state_dict(data['model'], strict=True)
    return model.to(device).eval(), data


def resume(path, trainer):
    data = read(path)
    current = asdict(trainer.config)
    saved = dict(data['config'])
    if 'execution' not in data:
        # Stage-1 checkpoints used "threads" for PyTorch intra-op parallelism.
        saved['torch_threads'] = saved.get('threads', 1)
        saved['threads'] = 1
        scheme = 'legacy-v1'
    else:
        scheme = data['execution']['sampling_scheme']
    if scheme != trainer.sampling_scheme:
        raise ValueError('Sampling scheme mismatch: exact resume cannot change streams; '
                         'use --warm-start for a new experiment, or --sampling-scheme ' + scheme)
    if scheme == 'legacy-v1' and trainer.config.threads != 1:
        raise ValueError('Legacy exact resume requires threads=1')
    for key, value in saved.items():
        if key not in ('total_hands', 'device', 'threads', 'sampling_scheme') and current[key] != value:
            raise ValueError(f'Resume configuration mismatch: {key}')
    if data['sampling_device'] != trainer.device.type:
        raise ValueError('Exact resumption requires the same sampling device type; CPU inference is always supported')
    if current['total_hands'] < data['progress']['hands']:
        raise ValueError('total_hands precedes checkpoint progress')
    trainer.model.load_state_dict(data['model'])
    trainer.optimizer.load_state_dict(data['optimizer'])
    for key, value in data['progress'].items():
        setattr(trainer, key, value)
    rng = data['rng']
    random.setstate(rng['python'])
    ns = rng['numpy']
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(rng['cpu'])
    if rng['cuda'] and trainer.device.type == 'cuda':
        torch.cuda.set_rng_state_all(rng['cuda'])
    trainer.sampling.set_state(rng['sampling'])
    trainer.shuffle.set_state(rng['shuffle'])
