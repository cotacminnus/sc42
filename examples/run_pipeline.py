"""Run a JSON-defined sequence of existing SC42 training/evaluation commands.

Each step runs one allowlisted command (examples.<command>) as a separate child
process with an explicit argument list; the command's own argparse and domain
validation stay authoritative. Steps run strictly in order, stop at the first
failure, and record provenance in OUTPUT_ROOT/pipeline_state.json, which
--resume uses to skip verified completed steps and continue an interrupted
training step through that command's own exact-resume mechanism.

    python -m examples.run_pipeline --config configs/pipelines/progressive_training.json [--dry-run | --resume]

See docs/command_reference.md#experiment-pipelines for the JSON format.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
STATE_FILE = 'pipeline_state.json'
REPOSITORY = Path(__file__).resolve().parents[1]

# The only executable entry points. kind decides automatic outputs, references and resume.
COMMANDS = {
    'train_ppo': 'training',
    'train_selfplay_ppo': 'training',
    'evaluate_ppo': 'evaluation',
    'evaluate_selfplay_ppo': 'evaluation',
    'evaluate_agent_vs_agent': 'evaluation',
    'oracle_labels': 'dataset',
}
STATUS = {'training': 'TRAINING', 'evaluation': 'EVALUATING', 'dataset': 'LABELING'}
OUTPUT_FILES = {'evaluation': 'result.json', 'dataset': 'labels.json'}
# Arguments holding an input file that must exist (references or external paths).
INPUT_FLAGS = re.compile(r'^(init-checkpoint|init-play-checkpoint|resume|warm-start|anchor-checkpoint|checkpoint|'
                         r'oracle-dataset|agent-[ab](-(bid|trump|play))?-checkpoint)$')
NAMED_INPUT_FLAGS = ('opponent-checkpoint',)  # NAME=PATH
REFERENCE = re.compile(r'\$\{([A-Za-z0-9_-]+)\.(best|latest|output)\}')
STEP_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
ARGUMENT = re.compile(r'^[a-z0-9][a-z0-9-]*$')
TOP_LEVEL = {'schema_version', 'name', 'output_root', 'steps', 'description'}
STEP_FIELDS = {'id', 'command', 'args', 'description'}


class Number:
    """A JSON non-integer number kept as its exact source text (1e-1 stays "1e-1")."""
    __slots__ = ('text',)

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return f'Number({self.text!r})'

    def __eq__(self, other):
        return isinstance(other, Number) and other.text == self.text

    def __hash__(self):
        return hash(self.text)


class PipelineError(ValueError):
    """Invalid configuration, references, outputs or manifest; nothing is run."""


@dataclass
class Step:
    index: int
    id: str
    command: str
    args: dict
    kind: str = field(init=False)
    directory: Path = field(init=False)

    def __post_init__(self):
        self.kind = COMMANDS[self.command]

    @property
    def module(self):
        return 'examples.' + self.command


@dataclass
class Pipeline:
    name: str
    output_root: Path
    steps: list
    config: dict
    sha256: str
    source: Path
    text: str | None = None  # Original JSON text, saved verbatim in the manifest.


def sha256_file(path) -> str:
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def repository_path(value) -> Path:
    """Paths inside a pipeline resolve from the repository root, never the JSON file."""
    path = Path(value)
    return (path if path.is_absolute() else REPOSITORY / path).resolve()


def load_config(path) -> Pipeline:
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise PipelineError(f'Cannot read pipeline configuration {path}: {exc}') from exc
    try:
        config = json.loads(text, parse_float=Number)
    except json.JSONDecodeError as exc:
        raise PipelineError(f'Malformed JSON in {path}: {exc}') from exc
    pipeline = parse_config(config, path)
    pipeline.text = text
    return pipeline


def parse_config(config, source=Path('<memory>')) -> Pipeline:
    if not isinstance(config, dict):
        raise PipelineError('A pipeline must be a JSON object')
    unknown = set(config) - TOP_LEVEL
    if unknown:
        raise PipelineError(f'Unknown pipeline fields: {", ".join(sorted(unknown))}')
    for key in ('schema_version', 'name', 'output_root', 'steps'):
        if key not in config:
            raise PipelineError(f'Missing required pipeline field: {key}')
    if type(config['schema_version']) is not int or config['schema_version'] != SCHEMA_VERSION:
        raise PipelineError(f'Unsupported schema_version {config["schema_version"]!r}; expected {SCHEMA_VERSION}')
    for key in ('name', 'output_root'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise PipelineError(f'{key} must be a nonempty string')
    if not isinstance(config['steps'], list) or not config['steps']:
        raise PipelineError('steps must be a nonempty list')
    steps, seen = [], set()
    for index, raw in enumerate(config['steps']):
        if not isinstance(raw, dict):
            raise PipelineError(f'Step {index + 1} must be a JSON object')
        unknown = set(raw) - STEP_FIELDS
        if unknown:
            raise PipelineError(f'Step {index + 1}: unknown fields {", ".join(sorted(unknown))}')
        for key in ('id', 'command'):
            if key not in raw:
                raise PipelineError(f'Step {index + 1}: missing required field {key}')
        sid = raw['id']
        if not isinstance(sid, str) or not STEP_ID.match(sid):
            raise PipelineError(f'Step {index + 1}: id must match {STEP_ID.pattern}')
        if sid in seen:
            raise PipelineError(f'Duplicate step id: {sid}')
        seen.add(sid)
        if raw['command'] not in COMMANDS:
            raise PipelineError(f'Step {sid}: unsupported command {raw["command"]!r}; '
                                f'allowed: {", ".join(COMMANDS)}')
        args = raw.get('args', {})
        if not isinstance(args, dict):
            raise PipelineError(f'Step {sid}: args must be a JSON object')
        steps.append(Step(index, sid, raw['command'], args))
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'), default=str)
    pipeline = Pipeline(config['name'], repository_path(config['output_root']), steps, config,
                        hashlib.sha256(canonical.encode()).hexdigest(), Path(source))
    for step in steps:
        step.directory = pipeline.output_root / step.id
        to_argv(step)  # Shape errors surface before anything runs.
        for _, value in _strings(step):
            _references(value, step, pipeline)
    _check_collisions(pipeline)
    return pipeline


def _scalar(value):
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, Number)):
        return str(value)
    if isinstance(value, float):  # Only from in-memory configs; JSON files give Number.
        return repr(value)
    return None


def to_argv(step) -> list:
    """JSON args -> CLI tokens: true is a flag, false omits it, lists repeat the flag."""
    argv = []
    for key, value in step.args.items():
        if not isinstance(key, str) or not ARGUMENT.match(key):
            raise PipelineError(f'Step {step.id}: argument {key!r} must be written without "--", '
                                'in lowercase-hyphen form')
        flag = '--' + key
        if value is True:
            argv.append(flag)
        elif value is False:
            continue
        elif isinstance(value, list):
            if not value:
                raise PipelineError(f'Step {step.id}: {key} is an empty list; omit it instead')
            for item in value:
                text = _scalar(item)
                if text is None:
                    raise PipelineError(f'Step {step.id}: {key} list items must be strings or numbers')
                argv += [flag, text]
        else:
            text = _scalar(value)
            if text is None:
                raise PipelineError(f'Step {step.id}: {key} has unsupported value type '
                                    f'{type(value).__name__}; use a string, number, boolean or list')
            argv += [flag, text]
    return argv


def _strings(step):
    for key, value in step.args.items():
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, str):
                yield key, item


def _references(text, step, pipeline):
    """Validated (step, artifact) references inside one string; nothing is evaluated."""
    found = []
    for match in REFERENCE.finditer(text):
        target, artifact = match.groups()
        source = next((s for s in pipeline.steps if s.id == target), None)
        if source is None:
            raise PipelineError(f'Step {step.id}: reference ${{{target}.{artifact}}} names no step')
        if source.index >= step.index:
            raise PipelineError(f'Step {step.id}: reference ${{{target}.{artifact}}} must name an earlier step')
        if artifact in ('best', 'latest') and source.kind != 'training':
            raise PipelineError(f'Step {step.id}: {target} is not a training step; use ${{{target}.output}}')
        if artifact == 'output' and source.kind == 'training':
            raise PipelineError(f'Step {step.id}: {target} is a training step; use .best or .latest')
        found.append((match.group(0), source, artifact))
    if '${' in REFERENCE.sub('', text):
        raise PipelineError(f'Step {step.id}: malformed reference in {text!r}; '
                            'use ${step.best}, ${step.latest} or ${step.output}')
    return found


def artifact_path(step, artifact) -> Path:
    if artifact == 'output':
        return output_paths(step)['output']
    return output_paths(step)['directory'] / 'checkpoints' / f'{artifact}.pt'


def output_paths(step) -> dict:
    """Owned locations: explicit arguments win; otherwise per-step defaults."""
    args = step.args
    if step.kind == 'training':
        directory = repository_path(args['output-dir']) if isinstance(args.get('output-dir'), str) else step.directory
        log = repository_path(args['log-dir']) if isinstance(args.get('log-dir'), str) else directory / 'logs'
        return dict(directory=directory, logs=log)
    output = (repository_path(args['output']) if isinstance(args.get('output'), str)
              else step.directory / OUTPUT_FILES[step.kind])
    log = repository_path(args['log-dir']) if isinstance(args.get('log-dir'), str) else step.directory / 'logs'
    return dict(directory=step.directory, output=output, logs=log)


def _check_collisions(pipeline):
    owned = []
    for step in pipeline.steps:
        for name, path in output_paths(step).items():
            if name != 'directory' or step.kind == 'training':
                owned.append((path, step.id, name))
    for i, (a, sa, na) in enumerate(owned):
        for b, sb, nb in owned[i + 1:]:
            if sa != sb and (a == b or a in b.parents or b in a.parents):
                raise PipelineError(f'Output collision: {sa} {na} {a} overlaps {sb} {nb} {b}')
    state = pipeline.output_root / STATE_FILE
    for path, sid, name in owned:
        if path == state:
            raise PipelineError(f'Step {sid} {name} collides with the pipeline manifest')


def resolve_args(step, pipeline, *, require=True):
    """Substitute references; return (args dict, dependencies). require=False predicts paths."""
    resolved, dependencies = {}, []
    for key, value in step.args.items():
        items = value if isinstance(value, list) else [value]
        out = []
        for item in items:
            if isinstance(item, str):
                for token, source, artifact in _references(item, step, pipeline):
                    path = artifact_path(source, artifact)
                    if require:
                        if not path.is_file():
                            other = 'latest' if artifact == 'best' else 'best'
                            hint = (f'; only {other}.pt exists' if artifact != 'output'
                                    and artifact_path(source, other).is_file() else '')
                            raise PipelineError(f'Step {step.id}: {token} does not exist at {path}{hint}')
                        dependency = dict(reference=token[2:-1], path=str(path), sha256=sha256_file(path))
                    else:
                        dependency = dict(reference=token[2:-1], path=str(path), sha256=None)
                    if dependency not in dependencies:
                        dependencies.append(dependency)
                    item = item.replace(token, str(path))
            out.append(item)
        resolved[key] = out if isinstance(value, list) else out[0]
    return resolved, dependencies


def external_inputs(step):
    """Input files named directly in this step; references are dependencies instead."""
    inputs = []
    for key, value in step.args.items():
        for item in (value if isinstance(value, list) else [value]):
            if not isinstance(item, str) or REFERENCE.search(item):
                continue
            if INPUT_FLAGS.match(key):
                inputs.append((key, item))
            elif key in NAMED_INPUT_FLAGS and '=' in item:
                inputs.append((key, item.split('=', 1)[1]))
    return inputs


def command_argv(step, args) -> list:
    """Complete child argument list, adding owned output locations when omitted."""
    argv = to_argv(Step(step.index, step.id, step.command, args))
    paths, present = output_paths(step), set(args)
    if step.kind == 'training':
        if 'output-dir' not in present:
            argv += ['--output-dir', str(paths['directory'])]
    elif 'output' not in present:
        argv += ['--output', str(paths['output'])]
    if 'log-dir' not in present:
        argv += ['--log-dir', str(paths['logs'])]
    return argv


def validate_with_parser(step, argv):
    """Parse with the command's own argparse definition; nothing is executed."""
    module = importlib.import_module(step.module)
    error = io.StringIO()
    try:
        with contextlib.redirect_stderr(error):
            module.build_parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code:
            message = error.getvalue().strip().splitlines()
            raise PipelineError(f'Step {step.id}: {step.command} rejects its arguments: '
                                f'{message[-1] if message else exc}') from None


def child(step, argv):
    return [sys.executable, '-m', step.module, *argv]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def git_state():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPOSITORY, text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return dict(sha=sha, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(sha=None, dirty=None)


def write_manifest(pipeline, manifest):
    manifest['updated'] = now()
    path = pipeline.output_root / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def new_manifest(pipeline):
    return dict(manifest_version=MANIFEST_VERSION, schema_version=SCHEMA_VERSION, name=pipeline.name,
                config_path=str(pipeline.source.resolve()), config_sha256=pipeline.sha256,
                config=json.loads(json.dumps(pipeline.config, default=str), parse_float=float)
                if pipeline.text is None else json.loads(pipeline.text), output_root=str(pipeline.output_root), git=git_state(),
                python=sys.executable, created=now(),
                steps=[dict(id=s.id, command=s.command, kind=s.kind, state='pending', attempts=[],
                            outputs={}, dependencies=[], inputs=[],
                            locations={k: str(v) for k, v in output_paths(s).items()}) for s in pipeline.steps])


def produced_outputs(step):
    """Expected outputs after a successful run, with SHA256 identities."""
    paths = output_paths(step)
    if step.kind == 'training':
        checkpoints = paths['directory'] / 'checkpoints'
        if not (checkpoints / 'latest.pt').is_file() or not (paths['directory'] / 'progress.jsonl').is_file():
            raise PipelineError(f'Step {step.id} exited successfully but produced no latest.pt/progress.jsonl')
        return {name: dict(path=str(checkpoints / f'{name}.pt'), sha256=sha256_file(checkpoints / f'{name}.pt'))
                for name in ('best', 'latest') if (checkpoints / f'{name}.pt').is_file()}
    if not paths['output'].is_file():
        raise PipelineError(f'Step {step.id} exited successfully but wrote no {paths["output"]}')
    return dict(output=dict(path=str(paths['output']), sha256=sha256_file(paths['output'])))


def verify_outputs(record):
    for name, item in record['outputs'].items():
        path = Path(item['path'])
        if not path.is_file():
            raise PipelineError(f'Completed step {record["id"]}: {name} output is missing: {path}')
        if sha256_file(path) != item['sha256']:
            raise PipelineError(f'Completed step {record["id"]}: {name} output was modified: {path}')


# ---------------------------------------------------------------------------
# Training-step recovery
# ---------------------------------------------------------------------------

def checkpoint_hands(path) -> int:
    import torch  # Only needed to recover an interrupted training step.
    return int(torch.load(path, map_location='cpu', weights_only=True)['progress']['hands'])


def reconcile(step, stamp):
    """Trim history written after latest.pt so resumed training never duplicates it.

    progress.jsonl rows with more hands than the checkpoint, and historical
    snapshots published after it, are moved to *.pre-resume-<stamp> backups;
    the resumed run regenerates them deterministically from the checkpoint.
    """
    directory = output_paths(step)['directory']
    latest = directory / 'checkpoints' / 'latest.pt'
    hands = checkpoint_hands(latest)
    report = dict(checkpoint_hands=hands, discarded_progress_rows=0, moved_snapshots=[])
    progress = directory / 'progress.jsonl'
    if progress.is_file():
        rows = [line for line in progress.read_text(encoding='utf-8').splitlines() if line.strip()]
        keep = [line for line in rows if json.loads(line)['hands'] <= hands]
        if len(keep) != len(rows):
            backup = progress.with_name(f'progress.jsonl.pre-resume-{stamp}')
            os.replace(progress, backup)
            progress.write_text(''.join(line + '\n' for line in keep), encoding='utf-8')
            report.update(discarded_progress_rows=len(rows) - len(keep), progress_backup=str(backup))
    snapshots = directory / 'snapshots'
    if snapshots.is_dir():
        later = [p for p in sorted(snapshots.glob('policy-*.pt')) if int(p.stem.split('-')[1]) > hands]
        if later:
            backup = directory / f'snapshots.pre-resume-{stamp}'
            backup.mkdir()
            for path in later:
                os.replace(path, backup / path.name)
            report['moved_snapshots'] = [str(backup / p.name) for p in later]
    return report


def resume_argv(step, args):
    """The command's own exact-resume invocation from this step's latest.pt.

    train_selfplay_ppo restores omitted settings from the checkpoint, so only
    the cumulative budget and runtime execution flags are repeated.
    train_ppo requires the saved configuration to be repeated, so the original
    arguments are reused with --resume replacing any initialization source.
    """
    latest = output_paths(step)['directory'] / 'checkpoints' / 'latest.pt'
    if step.command == 'train_selfplay_ppo':
        kept = {k: v for k, v in args.items()
                if k in ('total-hands', 'threads', 'eval-threads', 'device', 'output-dir', 'log-dir')}
    else:
        kept = {k: v for k, v in args.items() if k not in ('resume', 'warm-start')}
    kept['resume'] = str(latest)
    return command_argv(step, kept)


def has_partial_outputs(step):
    paths = output_paths(step)
    if step.kind == 'training':
        directory = paths['directory']
        return any((directory / name).exists() for name in ('progress.jsonl', 'checkpoints', 'snapshots'))
    return False


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def plan(pipeline):
    """Resolved argv per step with predicted references, validated by each parser."""
    rows = []
    for step in pipeline.steps:
        args, dependencies = resolve_args(step, pipeline, require=False)
        argv = command_argv(step, args)
        validate_with_parser(step, argv)
        rows.append((step, argv, dependencies, external_inputs(step)))
    return rows


def print_plan(pipeline, rows):
    print(f'Pipeline: {pipeline.name} (dry run; nothing is executed)')
    print(f'Configuration: {pipeline.source} SHA256 {pipeline.sha256}')
    print(f'Output root: {pipeline.output_root}')
    for step, argv, dependencies, inputs in rows:
        print(f'\n[{step.index + 1}/{len(pipeline.steps)}] {step.id} ({step.command})')
        print('  command: ' + shlex.join(child(step, argv)))
        for name, path in output_paths(step).items():
            print(f'  {name}: {path}')
        for d in dependencies:
            print(f'  depends on ${{{d["reference"]}}} -> {d["path"]} (produced by an earlier step)')
        for key, value in inputs:
            path = repository_path(value)
            print(f'  input --{key} {path} ({"exists" if path.is_file() else "MISSING"})')


def label(step, total, width, text):
    print(f'[{step.index + 1}/{total}] {step.id:<{width}} {text}', flush=True)


def run(pipeline, *, resume=False, runner=subprocess.run):
    """Execute or resume; returns the process exit status (0 on success)."""
    rows = plan(pipeline)  # Validate every step before running any of them.
    state = pipeline.output_root / STATE_FILE
    if resume:
        if not state.is_file():
            raise PipelineError(f'No manifest to resume at {state}')
        manifest = json.loads(state.read_text(encoding='utf-8'))
        if manifest.get('manifest_version') != MANIFEST_VERSION or manifest.get('name') != pipeline.name:
            raise PipelineError('Manifest does not belong to this pipeline')
        if manifest.get('config_sha256') != pipeline.sha256:
            raise PipelineError('The pipeline configuration changed since this run started; resume requires the '
                                'identical configuration (use a new output_root for a changed experiment)')
        if [s['id'] for s in manifest['steps']] != [s.id for s in pipeline.steps]:
            raise PipelineError('Manifest steps do not match the configuration')
        current = git_state()
        if current['sha'] != manifest['git']['sha']:
            print(f'Warning: code changed since the pipeline started ({manifest["git"]["sha"]} -> {current["sha"]})')
        manifest.setdefault('resumes', []).append(dict(time=now(), git=current))
    else:
        if state.exists():
            raise PipelineError(f'{state} exists; use --resume, or choose a new output_root')
        for step in pipeline.steps:
            for name, path in output_paths(step).items():
                if name != 'logs' and path.exists() and (path.is_file() or any(path.iterdir())):
                    raise PipelineError(f'Step {step.id} {name} already exists: {path}; '
                                        'choose a new output_root or remove it')
        manifest = new_manifest(pipeline)
    write_manifest(pipeline, manifest)
    total, width = len(pipeline.steps), max(len(s.id) for s in pipeline.steps)
    print(f'Pipeline: {pipeline.name}\n', flush=True)
    for (step, _, _, _), record in zip(rows, manifest['steps']):
        if record['state'] == 'completed':
            verify_outputs(record)
            for dependency in record['dependencies']:
                if sha256_file(dependency['path']) != dependency['sha256']:
                    raise PipelineError(f'Step {step.id}: dependency {dependency["path"]} changed after use')
            label(step, total, width, 'SKIPPED (completed; outputs verified)')
            continue
        args, dependencies = resolve_args(step, pipeline)
        inputs = []
        for key, value in external_inputs(step):
            path = repository_path(value)
            if not path.is_file():
                raise PipelineError(f'Step {step.id}: --{key} file does not exist: {path}')
            inputs.append(dict(argument=key, path=str(path), sha256=sha256_file(path)))
        attempt = dict(started=now(), mode='initial')
        argv = command_argv(step, args)
        if record['state'] in ('running', 'failed') or has_partial_outputs(step):
            latest = output_paths(step)['directory'] / 'checkpoints' / 'latest.pt'
            if step.kind == 'training' and latest.is_file():
                attempt.update(mode='resume', reconciliation=reconcile(step, datetime.now().strftime('%Y%m%d%H%M%S')))
                argv = resume_argv(step, args)
                label(step, total, width, f'RESUMING from {latest}')
            elif step.kind == 'training' and has_partial_outputs(step):
                record['state'] = 'failed'
                write_manifest(pipeline, manifest)
                raise PipelineError(f'Step {step.id} has partial outputs but no latest.pt to resume from: '
                                    f'{output_paths(step)["directory"]}. Inspect it, then remove that directory '
                                    'to restart this step from its configured initialization.')
            else:
                attempt['mode'] = 'retry'  # Evaluation/labeling reruns completely; no training progressed.
        attempt['argv'] = child(step, argv)
        record.update(state='running', argv=child(step, command_argv(step, args)),
                      dependencies=dependencies, inputs=inputs)
        record['attempts'].append(attempt)
        write_manifest(pipeline, manifest)
        label(step, total, width, STATUS[step.kind])
        try:
            code = runner(child(step, argv), cwd=REPOSITORY).returncode
        except KeyboardInterrupt:
            attempt.update(finished=now(), exit_code=None, interrupted=True)
            record['state'] = 'failed'
            write_manifest(pipeline, manifest)
            raise
        attempt.update(finished=now(), exit_code=code)
        if code == 0:
            try:
                record['outputs'] = produced_outputs(step)
            except PipelineError as exc:
                attempt['error'] = str(exc)
                code = 1
        if code != 0:
            record['state'] = 'failed'
            write_manifest(pipeline, manifest)
            label(step, total, width, f'FAILED (exit status {code})')
            print(f'\nPipeline stopped at step {step.id}. Fix the cause, then rerun with --resume.')
            return code
        record['state'] = 'completed'
        write_manifest(pipeline, manifest)
        label(step, total, width, 'COMPLETED')
    print('\nPipeline completed successfully.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True, help='pipeline JSON file')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='validate and print the plan without running anything')
    mode.add_argument('--resume', action='store_true', help='continue from OUTPUT_ROOT/pipeline_state.json')
    args = parser.parse_args(argv)
    try:
        pipeline = load_config(args.config)
        if args.dry_run:
            print_plan(pipeline, plan(pipeline))
            return 0
        return run(pipeline, resume=args.resume)
    except PipelineError as exc:
        parser.exit(2, f'{parser.prog}: error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
