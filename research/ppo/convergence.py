"""Convergence diagnostics for recorded training lineages (docs/convergence.md).

Reads only what a run recorded: progress.jsonl (version-1 or version-2 rows),
config.json, snapshots/*.pt and checkpoints/. It never imports the trainers or
the opponent pool, so a change in how training opponents are chosen does not
change how their records are read: every training hand is classified by the
SHA it was logged with, not by the pool name it was drawn under.

Four layers, each answering one question (docs/convergence.md, section 2):
  drift      How much does behaviour still change? Per phase, on a fixed
             neutral probe set: KL between policies, greedy-move churn, and
             recurrence (a policy closer to an older policy than to its
             predecessor, as in a rock-paper-scissors cycle).
  exposure   What did the learner actually train against?
  strength   Recorded validation marks per opponent, a plateau estimate, and
             the gap between training and held-out opponents.
  cross-play Snapshot-vs-snapshot duplicate results: transitive progress,
             significant 3-cycles, and the final policy against its past.
The metric functions take plain tensors, lists and dicts, so tests can feed
them synthetic sequences.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import torch

from texas42.agents import HeuristicAgent, VARIANTS
from texas42.encoding import encode, FEATURE_SIZE
from texas42.actions import NUM_ACTIONS
from texas42.selfplay import SelfPlayEnv
from .multiphase import PHASES
from .opponents import CheckpointIdentity, DYNAMIC_OPPONENTS, SELF_CURRENT, SELF_SNAPSHOTS
from .search import masked_kl, masked_log_softmax
from .seeds import stream_seed

PROBE_DOMAIN = 'convergence-probe-v1'
OPPONENT_CLASSES = ('heuristic', 'anchor', 'snapshot', 'checkpoint', 'current', 'unknown')
Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Policy drift on a neutral probe set
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_positions(seed, hands):
    """Heuristic V1 at all four seats; every decision with at least two legal moves, by phase.

    Neutral: no lineage's own play decides which positions are probed (unlike
    imitation.teacher_data(anchor=...), which probes the anchor's games, play
    only). Its own seed domain keeps the BC and anchor-KL streams unchanged.
    """
    features = {p: [] for p in PHASES}
    masks = {p: [] for p in PHASES}
    for episode in range(hands):
        env = SelfPlayEnv(seed=stream_seed(seed, PROBE_DOMAIN, episode))
        obs = env.reset()
        agents = [HeuristicAgent() for _ in range(4)]
        for seat, agent in enumerate(agents):
            agent.reset(seat)
        while True:
            if len(obs.legal_action_ids) > 1:
                features[str(obs.phase)].append(encode(obs))
                masks[str(obs.phase)].append(obs.action_mask)
            transition = env.step(agents[obs.seat].act(obs))
            if transition.terminated:
                break
            obs = transition.observation
    return {p: dict(features=torch.tensor(features[p], dtype=torch.float32).reshape(-1, FEATURE_SIZE),
                    masks=torch.tensor(masks[p], dtype=torch.bool).reshape(-1, NUM_ACTIONS)) for p in PHASES}


@torch.no_grad()
def policy_log_probs(network, probe):
    """Masked log-policy of one phase network on one probe phase."""
    distribution, _ = network.distribution(probe['features'], probe['masks'])
    return masked_log_softmax(distribution.logits, probe['masks'])


MIN_RECURRENCE_KL = 0.01  # Nats; smaller steps are a stationary policy wandering, not a cycle.


def drift(log_probs, masks, *, min_step_kl=MIN_RECURRENCE_KL):
    """Change across an ordered sequence of policies (oldest first) on one probe phase.

    log_probs: one [states, actions] masked log-policy per policy. Returns:
      consecutive_kl / consecutive_greedy_change: KL(p_i || p_i+1) and the share
        of states whose most likely move changes, for each step;
      kl_to_final / greedy_change_to_final: the same against the last policy;
      pairwise_kl: KL(p_i || p_j) for every pair;
      recurrent: each policy (from the third) whose nearest earlier policy, by
        symmetric KL, is not its immediate predecessor. Counted only when both
        the step from the predecessor and the distance from the predecessor to
        that earlier policy are at least min_step_kl: among nearly identical
        earlier policies, which one is nearest means nothing. A fixed
        opponent's best response converges without recurrence; a
        rock-paper-scissors chase (R -> P -> S -> R) returns to an earlier policy.
    """
    count = len(log_probs)
    if not count or not len(masks):
        return None
    kl = [[float(masked_kl(log_probs[i], log_probs[j], masks).mean()) if i != j else 0.0
           for j in range(count)] for i in range(count)]
    greedy = [p.argmax(-1) for p in log_probs]

    def changed(i, j):
        return float((greedy[i] != greedy[j]).float().mean())

    def symmetric(i, j):
        return (kl[i][j] + kl[j][i]) / 2

    recurrent = []
    for k in range(2, count):
        nearest = min(range(k), key=lambda j: (symmetric(k, j), -j))  # Ties go to the latest.
        previous = symmetric(k, k - 1)
        if nearest != k - 1 and previous >= min_step_kl and symmetric(nearest, k - 1) >= min_step_kl:
            recurrent.append(dict(index=k, nearest_earlier=nearest, ratio=symmetric(k, nearest) / previous))
    return dict(states=int(len(masks)),
                consecutive_kl=[kl[i][i + 1] for i in range(count - 1)],
                consecutive_greedy_change=[changed(i, i + 1) for i in range(count - 1)],
                kl_to_final=[kl[i][-1] for i in range(count)],
                greedy_change_to_final=[changed(i, count - 1) for i in range(count)],
                pairwise_kl=kl, recurrent=recurrent)


def weights_digest(path):
    """SHA-256 of a checkpoint's weights alone: tensor names, dtypes, shapes and bytes.

    Unlike CheckpointIdentity (the file SHA), it is equal for files that differ
    only in metadata, such as best.pt saved in the same update as a snapshot.
    """
    state = torch.load(path, map_location='cpu', weights_only=True)['model']
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(f'{name}|{tensor.dtype}|{tuple(tensor.shape)}|'.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Training exposure and recorded strength
# ---------------------------------------------------------------------------

def classify_opponent(name, sha256, identities):
    """(class, identity label) of one logged training opponent.

    identities maps the file SHAs a run recorded to (class, label). A missing
    SHA is a built-in variant. An unmatched SHA under a dynamic name is the
    learner's own rollout weights (logged as a payload digest, never a file);
    anything else unmatched is reported as unknown rather than guessed.
    """
    if sha256 is None:
        return ('heuristic', name) if name in VARIANTS else ('unknown', name)
    if sha256 in identities:
        return identities[sha256]
    if name in DYNAMIC_OPPONENTS:
        return 'current', 'learner rollout weights'
    return 'unknown', f'{name} {sha256[:12]}'


def opponent_exposure(rows, identities, window_hands=None):
    """Share of training hands per opponent class and identity, overall and per window.

    relabelled lists hands whose pool name differs from the identity actually
    played, e.g. self-snapshots hands that played the anchor before the first
    snapshot existed. Version-1 rows record names only (all built-in variants).
    """
    assignments = [tuple(a) for row in rows for a in row.get('opponent_assignments') or ()]
    if not assignments:
        counts = Counter()
        for row in rows:  # Version-1: training_by_opponent per update, variants only.
            for name, stats in (row.get('training_by_opponent') or {}).items():
                counts[name] += int(stats.get('hands', 0))
        if not counts:
            return None
        total = sum(counts.values())
        return dict(hands=total, identities_recorded=False,
                    by_class={'heuristic': 1.0} if all(n in VARIANTS for n in counts) else None,
                    by_identity=[dict(cls='heuristic' if n in VARIANTS else 'unknown', label=n, name=n,
                                      hands=h, share=h / total) for n, h in sorted(counts.items())],
                    windows=[], relabelled=[])
    identity_hands, class_hands, windows, relabelled = {}, Counter(), {}, {}
    for episode, name, sha in assignments:
        cls, label = classify_opponent(name, sha, identities)
        class_hands[cls] += 1
        entry = identity_hands.setdefault((cls, label, name), dict(hands=0, first=episode, last=episode))
        entry['hands'] += 1
        entry['first'], entry['last'] = min(entry['first'], episode), max(entry['last'], episode)
        if window_hands:
            windows.setdefault(episode // window_hands, Counter())[cls] += 1
        expected = (name == label or (cls == 'snapshot' and name == SELF_SNAPSHOTS)
                    or (cls == 'current' and name == SELF_CURRENT))
        if not expected:
            entry = relabelled.setdefault((name, cls, label), dict(hands=0, first=episode, last=episode))
            entry['hands'] += 1
            entry['first'], entry['last'] = min(entry['first'], episode), max(entry['last'], episode)
    total = len(assignments)
    return dict(
        hands=total, identities_recorded=True,
        by_class={c: class_hands[c] / total for c in OPPONENT_CLASSES if class_hands[c]},
        by_identity=[dict(cls=c, label=l, name=n, share=v['hands'] / total, **v)
                     for (c, l, n), v in sorted(identity_hands.items(), key=lambda kv: kv[1]['first'])],
        windows=[dict(start=w * window_hands, stop=(w + 1) * window_hands,
                      by_class={c: n / sum(counts.values()) for c, n in sorted(counts.items())})
                 for w, counts in sorted(windows.items())],
        relabelled=[dict(name=n, cls=c, label=l, **v) for (n, c, l), v in relabelled.items()])


def agent_a_marks(report):
    """dict(marks, low, high, games) for agent A of an ArenaReport.to_dict(), or None if malformed.

    The same schema is recorded by in-training validation and written by
    examples.evaluate_agent_vs_agent (result.json).
    """
    try:
        metrics = next(m for m in report['metrics'] if m['policy'] == 'A')
        interval = next(i for i in report['intervals'] if i['policy'] == 'A')['mean_signed_marks']
        return dict(marks=metrics['mean_signed_marks'], low=interval['low'], high=interval['high'],
                    games=metrics.get('games_played'))
    except (KeyError, TypeError, StopIteration):
        return None


def evaluation_points(rows):
    """{opponent: [dict(hands, marks, low, high, games)]} from recorded validations (agent A)."""
    series = {}
    for row in rows:
        reports = row.get('evaluations')
        if reports is None and row.get('evaluation') is not None:
            reports = {row.get('selection_opponent') or 'HHH': row['evaluation']}  # Oldest rows.
        for name, report in (reports or {}).items():
            result = agent_a_marks(report)
            if result is not None:
                series.setdefault(name, []).append(dict(hands=row.get('hands'), **result))
    return series


def plateau(points):
    """Descriptive plateau of an ordered (hands, marks, low, high) series.

    plateau_hands is the first point after which no point differs from the
    last by more than two combined standard errors (each taken from its 95%
    interval). With wide intervals it comes early: read it with the interval
    widths, not as a test.
    """
    if not points:
        return None
    se = [(p['high'] - p['low']) / (2 * Z95) for p in points]
    last = len(points) - 1

    def close(j):
        return abs(points[j]['marks'] - points[last]['marks']) <= 2 * math.hypot(se[j], se[last])

    start = last
    while start > 0 and close(start - 1):
        start -= 1
    best = max(range(len(points)), key=lambda j: points[j]['marks'])
    return dict(plateau_hands=points[start]['hands'], best_hands=points[best]['hands'],
                best_marks=points[best]['marks'], final_marks=points[last]['marks'],
                final_low=points[last]['low'], final_high=points[last]['high'], evaluations=len(points))


def generalization_gap(final_marks, training_names):
    """Training-opponent minus held-out-opponent marks at the final validation.

    The absolute gap also reflects how hard each opponent is; compare gaps
    between lineages evaluated on the same panel (joint-policy correlation,
    Lanctot et al. 2017), not with zero.
    """
    trained = [n for n in final_marks if n in training_names]
    held_out = [n for n in final_marks if n not in training_names]
    if not held_out:
        return None
    held_marks = sum(final_marks[n] for n in held_out) / len(held_out)
    training_marks = sum(final_marks[n] for n in trained) / len(trained) if trained else None
    return dict(training=trained, held_out=held_out, training_marks=training_marks, held_out_marks=held_marks,
                gap=None if training_marks is None else training_marks - held_marks)


# ---------------------------------------------------------------------------
# Cross-play
# ---------------------------------------------------------------------------

def crossplay_summary(results, order, panel=(), hands=None):
    """Summarize duplicate results among one lineage's policies (oldest first).

    results maps (A, B) to (marks, low, high), A's signed marks per hand with
    its 95% interval; the mirror (B, A) is implied (duplicate play is zero-sum
    between the two agents). Significant means the interval excludes 0.
      transitive_progress: later-vs-earlier pairs that are significantly
        better, worse or inconclusive;
      cycles: policy triples a > b > c > a, all three significant;
      final_vs_earlier: the last policy against each earlier one (forgetting);
      panel: each policy against each fixed panel opponent, with a plateau.
    """
    def get(a, b):
        if (a, b) in results:
            return results[(a, b)]
        if (b, a) in results:
            marks, low, high = results[(b, a)]
            return -marks, -high, -low
        return None

    def sign(result):
        return 0 if result is None else 1 if result[1] > 0 else -1 if result[2] < 0 else 0

    progress = Counter()
    for i, j in combinations(range(len(order)), 2):
        result = get(order[j], order[i])
        if result is not None:
            progress[{1: 'later_better', -1: 'later_worse', 0: 'inconclusive'}[sign(result)]] += 1
    cycles = []
    for a, b, c in combinations(order, 3):
        for x, y, z in ((a, b, c), (a, c, b)):
            if sign(get(x, y)) > 0 and sign(get(y, z)) > 0 and sign(get(z, x)) > 0:
                cycles.append([x, y, z])
    final = order[-1] if order else None
    forgetting = [dict(opponent=o, marks=r[0], low=r[1], high=r[2])
                  for o in order[:-1] if (r := get(final, o)) is not None]
    panels = {}
    for name in panel:
        points = [dict(policy=p, hands=(hands or {}).get(p), marks=r[0], low=r[1], high=r[2])
                  for p in order if (r := get(p, name)) is not None]
        panels[name] = dict(points=points, plateau=plateau(points) if points else None)
    return dict(policies=list(order), transitive_progress=dict(progress), cycles=cycles,
                final_vs_earlier=forgetting, panel=panels)


# ---------------------------------------------------------------------------
# Recorded lineages
# ---------------------------------------------------------------------------

def _hands_from_name(path):
    try:
        return int(Path(path).stem.split('-')[-1])
    except ValueError:
        return None


def _describe(role, path):
    """One checkpoint's identity, weights, recorded hands and provenance, or None if not a checkpoint."""
    from .checkpoint import read
    from .search import trained_phases
    try:
        data = read(path)
    except Exception:  # Not a checkpoint (e.g. a cached search result) or incompatible.
        return None
    identity = CheckpointIdentity.from_path(path)
    metadata = data.get('metadata', {})
    progress = data.get('progress') if isinstance(data.get('progress'), dict) else {}
    hands = _hands_from_name(path) if role == 'snapshot' else progress.get('hands')
    if metadata.get('derivation'):
        hands = None  # Copied from the source's progress; the derived model has no training hands.
    return dict(role=role, label=Path(path).stem if role == 'snapshot' else role, path=identity.path,
                sha256=identity.sha256, weights=weights_digest(path), hands=hands,
                version=metadata.get('checkpoint_version'), trained=list(trained_phases(identity)),
                derivation=metadata.get('derivation'),
                initial=(metadata.get('initial_checkpoint_path'), metadata.get('initial_checkpoint_sha256')),
                anchor=metadata.get('original_play_sha256'), roster=data.get('snapshots') or [])


def lineage_policy(path, role='checkpoint'):
    """One checkpoint described like a lineage's policies (identity, weights, trained phases)."""
    policy = _describe(role, Path(path))
    if policy is None:
        raise ValueError(f'{path} is not a readable checkpoint')
    return policy


def _read_rows(directory):
    path = directory / 'progress.jsonl'
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def lineage(directory):
    """Normalize one recorded run into policies, sequences, identities and warnings.

    Handles version-1 runs (best/latest, play only), version-2 shared runs
    (initial checkpoint + snapshots + latest), paired runs (per partner:
    initial, best and latest; they publish no snapshots) and directories of
    derived checkpoints (flagged from metadata['derivation']).
    """
    directory = Path(directory).resolve()
    rows = _read_rows(directory)
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    paired = (directory / 'checkpoints' / 'partner-1').is_dir()
    candidates = [('snapshot', p) for p in sorted((directory / 'snapshots').glob('policy-*.pt'))]
    if paired:
        candidates += [(f'partner-{i}-{name}', directory / 'checkpoints' / f'partner-{i}' / f'{name}.pt')
                       for i in (1, 2) for name in ('best', 'latest')]
    else:
        candidates += [(name, directory / 'checkpoints' / f'{name}.pt') for name in ('best', 'latest')]
    if not any(path.is_file() for _, path in candidates):
        candidates = [(p.stem, p) for p in sorted(directory.glob('*.pt'))]
    warnings, policies = [], []
    for role, path in candidates:
        if not path.is_file():
            continue
        described = _describe(role, path)
        if described is None:
            warnings.append(f'{path.name} is not a readable checkpoint; skipped')
        else:
            policies.append(described)
    if not policies:
        raise ValueError(f'No checkpoints found in {directory}')
    version = max(p['version'] or 0 for p in policies)
    snapshots = [p for p in policies if p['role'] == 'snapshot']
    latest = next((p for p in policies if p['role'] in ('latest', 'partner-1-latest')), policies[-1])

    # A derived checkpoint starts from its derivation source, not from the
    # source run's own initial checkpoint (which its copied metadata names).
    derivation = latest['derivation'] or {}
    initial_path, initial_sha = ((derivation.get('source_path'), derivation.get('source_sha256'))
                                 if derivation else latest['initial'])
    initial = None
    if initial_path:
        candidate = Path(initial_path)
        if candidate.is_file() and CheckpointIdentity.from_path(candidate).sha256 == initial_sha:
            initial = _describe('initial', candidate)
            if initial is not None:
                initial['hands'] = 0
        else:
            warnings.append(f'initial checkpoint {initial_path} is missing or changed; drift starts later')

    def unique(sequence):
        kept = []
        for policy in sequence:
            if kept and policy['weights'] == kept[-1]['weights']:
                continue  # The same weights twice (e.g. latest = the last snapshot) is no step.
            kept.append(policy)
        return kept

    def timeline(roles):
        return unique(([initial] if initial else []) + sorted(
            (p for p in policies if p['role'] in roles), key=lambda p: p['hands'] or 0))

    derived = all(p['derivation'] for p in policies)
    if derived:
        # Derived siblings (e.g. distillation pilots) are not a time series:
        # each is compared with its derivation source.
        if initial is not None:
            initial.update(role='source', label='source', hands=None)
        if len({p['derivation'].get('source_sha256') for p in policies}) > 1:
            warnings.append("derived checkpoints have different sources; all are compared with the last one's")
        sequences = {p['label']: unique(([initial] if initial else []) + [p]) for p in policies}
    elif paired:
        sequences = {f'partner-{i}': timeline({f'partner-{i}-best', f'partner-{i}-latest'}) for i in (1, 2)}
        warnings.append('paired run: no snapshots are published, so drift uses initial, best and latest only')
    elif snapshots:
        sequences = {'policy': unique(([initial] if initial else []) + snapshots
                                      + [p for p in policies if p['role'] == 'latest'])}
    else:
        sequences = {'policy': timeline({'best', 'latest'} | {p['role'] for p in policies})}

    by_weights = {}
    for policy in policies + ([initial] if initial else []):
        by_weights.setdefault(policy['weights'], []).append(policy)
    for group in by_weights.values():
        if len(group) > 1:
            names = ' = '.join(Path(p['path']).relative_to(directory).as_posix()
                               if Path(p['path']).is_relative_to(directory) else p['path'] for p in group)
            warnings.append(f'same weights under different file SHAs: {names}')
    for policy in policies:
        if policy['derivation']:
            d = policy['derivation']
            warnings.append(f'{policy["label"]} is derived ({d.get("kind")}) from {str(d.get("source_sha256"))[:12]}; '
                            'its config, progress and snapshots describe the source run')
    if rows and not any('opponent_assignments' in r for r in rows):
        warnings.append('version-1 progress: training opponents are recorded by name only (no SHAs)')

    identities = {}
    anchor_sha = ((config.get('anchor') or {}).get('sha256') or latest.get('anchor')
                  or next((r.get('anchor_sha256') for r in rows if r.get('anchor_sha256')), None))
    for spec in config.get('opponent_policies') or ():
        if spec.get('sha256') and spec.get('name') != 'anchor':
            identities[spec['sha256']] = ('checkpoint', spec['name'])
    for policy in policies:
        for spec in policy['roster']:
            if spec.get('sha256'):
                # Recorded paths may be Windows paths; the stem is what names the file.
                identities[spec['sha256']] = ('snapshot', PureWindowsPath(spec.get('path') or spec['name']).stem)
    for policy in snapshots:
        identities[policy['sha256']] = ('snapshot', policy['label'])
    if anchor_sha:
        identities[anchor_sha] = ('anchor', 'anchor')
    training_names = [name for name, _ in (config.get('opponent_probabilities') or config.get('train_opponents')
                                           or (rows[-1].get('train_opponents') if rows else None) or ())]
    return dict(directory=str(directory), version=version, paired=paired, derived=derived, rows=rows, config=config,
                policies=policies, initial=initial, sequences=sequences, identities=identities,
                anchor_sha256=anchor_sha, training_names=training_names,
                snapshot_every_hands=config.get('snapshot_every_hands'), warnings=warnings)


def phase_networks(policy):
    """{phase: network} for each phase the checkpoint trained."""
    from .composition import PhasePolicySpec, phase_network
    spec = PhasePolicySpec('neural', CheckpointIdentity(policy['path'], policy['sha256']))
    return {phase: phase_network(spec, phase) for phase in policy['trained']}


def analyze(run, probe):
    """Every offline metric of one lineage(): drift, partner divergence, exposure, strength."""
    drifts = {}
    for sequence_name, sequence in run['sequences'].items():
        networks = [phase_networks(p) for p in sequence]
        for phase in PHASES:
            members = [(p, n[phase]) for p, n in zip(sequence, networks) if phase in n]
            if len(members) < 2 or not len(probe[phase]['masks']):
                continue
            result = drift([policy_log_probs(net, probe[phase]) for _, net in members], probe[phase]['masks'])
            key = phase if len(run['sequences']) == 1 else f'{sequence_name} {phase}'
            drifts[key] = dict(policies=[p['label'] for p, _ in members],
                               hands=[p['hands'] for p, _ in members], **result)
    divergence = {}
    if run['paired']:
        last = [run['sequences'][f'partner-{i}'][-1] for i in (1, 2)]
        first, second = (phase_networks(p) for p in last)
        for phase in PHASES:
            if phase in first and phase in second and len(probe[phase]['masks']):
                result = drift([policy_log_probs(first[phase], probe[phase]),
                                policy_log_probs(second[phase], probe[phase])], probe[phase]['masks'])
                divergence[phase] = dict(kl_1_to_2=result['pairwise_kl'][0][1], kl_2_to_1=result['pairwise_kl'][1][0],
                                         greedy_disagreement=result['consecutive_greedy_change'][0],
                                         states=result['states'])
    exposure = opponent_exposure(run['rows'], run['identities'],
                                 run['snapshot_every_hands'] if run['version'] == 2 else None)
    warnings = list(run['warnings'])
    for entry in (exposure or {}).get('relabelled', ()):
        played = entry['cls'] if entry['label'] == entry['cls'] else f'{entry["cls"]} {entry["label"]}'
        warnings.append(f'{entry["hands"]:,} hands drawn as {entry["name"]} played {played} '
                        f'(episodes {entry["first"]:,}-{entry["last"]:,})')
    strength = {name: dict(points=points, **(plateau(points) or {}))
                for name, points in evaluation_points(run['rows']).items()}
    final = {name: s['final_marks'] for name, s in strength.items() if 'final_marks' in s}
    return dict(directory=run['directory'], version=run['version'], paired=run['paired'],
                policies=[{k: v for k, v in p.items() if k not in ('roster', 'initial')} for p in run['policies']],
                initial=None if run['initial'] is None else run['initial']['path'],
                warnings=warnings, drift=drifts, partner_divergence=divergence, exposure=exposure,
                strength=strength, training_opponents=run['training_names'],
                generalization=generalization_gap(final, run['training_names']))
