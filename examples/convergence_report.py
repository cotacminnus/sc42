"""Convergence report for recorded training runs (docs/convergence.md).

Offline and read-only: no game is played and nothing in the runs changes.
For each --run NAME=DIR it reports
  drift      per-phase policy change between the run's policies (initial
             checkpoint, snapshots, latest; best and latest for version-1 and
             paired runs) on a neutral Heuristic V1 probe set: KL, the share
             of probe positions whose most likely move changed, and recurrence;
  exposure   the opponents actually trained against, classified by the SHA
             each training hand logged (not by pool name);
  strength   recorded validation marks per opponent, a descriptive plateau,
             and the training-minus-held-out generalization gap;
  warnings   record discrepancies: identical weights under different file
             SHAs, derived checkpoints, relabelled opponent hands, missing
             snapshots or opponent identities.

Cross-play (the response layer) needs games, so it is a separate two-step
recipe through examples.run_pipeline:
  1. --crossplay-pipeline OUT.json writes a pipeline of duplicate
     evaluate_agent_vs_agent steps: each run's policies against each other and
     against a fixed --panel, every run's final policy against every other
     run's, and mixed partnerships (one run's first seat with another's
     second seat) against the first panel opponent.
  2. After python -m examples.run_pipeline --config OUT.json, the same command
     with --crossplay-results OUT.json (and the same --run, --panel and
     --crossplay-policies) adds transitive progress, cycles, forgetting and
     partner compatibility.
"""
import argparse
import json
import re
from pathlib import Path

from examples.experiment_output import add_output_arguments, record_output

STEP_SAFE = re.compile(r'[^A-Za-z0-9_-]+')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='NAME=DIR',
                   help='a training output directory (repeat to compare runs)')
    p.add_argument('--probe-hands', type=int, default=256,
                   help='Heuristic V1 hands whose positions (two or more legal moves) form the probe set')
    p.add_argument('--seed', type=int, default=20261101, help='probe positions and cross-play deals')
    p.add_argument('--output', type=Path, help='JSON report')
    cross = p.add_argument_group('cross-play')
    mode = cross.add_mutually_exclusive_group()
    mode.add_argument('--crossplay-pipeline', type=Path, metavar='OUT.json',
                      help='write an examples.run_pipeline configuration of cross-play steps')
    mode.add_argument('--crossplay-results', type=Path, metavar='PIPELINE.json',
                      help='summarize a completed --crossplay-pipeline run')
    cross.add_argument('--panel', default='HHH,CHH', help='fixed built-in opponents (comma-separated variants)')
    cross.add_argument('--panel-checkpoint', action='append', default=[], metavar='NAME=PATH',
                       help='fixed checkpoint opponent; plays its trained phases, heuristic otherwise')
    cross.add_argument('--crossplay-policies', type=int, default=0,
                       help='evenly spaced policies per run, first and last included (0: all)')
    cross.add_argument('--deals', type=int, default=1024)
    cross.add_argument('--bootstrap-samples', type=int, default=2000)
    cross.add_argument('--pipeline-name', default='convergence-crossplay-v1')
    cross.add_argument('--pipeline-root', help='output root of the generated pipeline (default runs/pipelines/NAME)')
    add_output_arguments(p)
    return p


def named(entries, flag, parser):
    pairs = []
    for entry in entries:
        name, sep, path = entry.partition('=')
        if not sep or not name or not path:
            parser.error(f'{flag} expects NAME=PATH, got {entry!r}')
        pairs.append((name, Path(path)))
    if len({n for n, _ in pairs}) != len(pairs):
        parser.error(f'duplicate {flag} names')
    return pairs


# ---------------------------------------------------------------------------
# Cross-play pipeline
# ---------------------------------------------------------------------------

def _relative(path):
    from examples.run_pipeline import REPOSITORY
    path = Path(path)
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(path)


def _thin(sequence, count):
    if not count or len(sequence) <= count:
        return list(sequence)
    if count == 1:
        return [sequence[-1]]
    picks = sorted({round(i * (len(sequence) - 1) / (count - 1)) for i in range(count)})
    return [sequence[i] for i in picks]


def crossplay_players(run, count):
    """[(label, first-seat policy, second-seat policy, hands)] oldest first.

    Shared and version-1 runs play one network at both seats; a paired run
    plays partner 1 at the first seat and partner 2 at the second.
    """
    if run['paired']:
        first, second = run['sequences']['partner-1'], run['sequences']['partner-2']
        by_role = {p['role'].removeprefix('partner-2-'): p for p in second}
        players = [(p['role'].removeprefix('partner-1-'), p, by_role.get(p['role'].removeprefix('partner-1-')), p['hands'])
                   for p in first]
        players = [(label, a, b, hands) for label, a, b, hands in players if b is not None]
    elif run['derived']:  # Siblings derived from one source: the source first, then each.
        players = [(p['label'], p, p, p['hands']) for p in ([run['initial']] if run['initial'] else [])
                   + [p for p in run['policies']]]
    else:
        players = [(p['label'], p, p, p['hands']) for p in next(iter(run['sequences'].values()))]
    return _thin(players, count)


def _agent(side, first, second):
    args = {f'agent-{side}-checkpoint': _relative(first['path'])}
    if second['path'] != first['path']:
        args[f'agent-{side}-partner-checkpoint'] = _relative(second['path'])
    from research.ppo.multiphase import PHASES
    for phase in PHASES:
        if phase in first['trained'] and phase in second['trained']:
            args[f'agent-{side}-{phase}'] = 'neural'
    return args


def _panel_agent(side, entry):
    from texas42.agents import VARIANTS
    from research.ppo.multiphase import PHASES
    if entry['kind'] == 'variant':
        return {f'agent-{side}-{phase}': strategy for phase, strategy in zip(PHASES, VARIANTS[entry['name']])
                if strategy != 'heuristic'}
    return _agent(side, entry['policy'], entry['policy'])


def crossplay_plan(runs, panel, count, deals, seed, bootstrap_samples, name, root=None):
    """(pipeline configuration, entries): entries map each comparison to its step id.

    A comparison whose two sides have the same weights as an earlier one
    reuses that step (identical files under different names play once).
    """
    common = dict(deals=deals, seed=seed, **{'bootstrap-samples': bootstrap_samples}, threads=0)
    steps, entries, played = [], [], {}

    def key(first, second):
        return first['weights'], second['weights']

    def add(kind, run, a_label, b_label, a_key, b_key, args, step_name, description):
        identity = (a_key, b_key)
        if identity not in played:
            step_id = STEP_SAFE.sub('-', step_name).strip('-')
            played[identity] = step_id
            steps.append(dict(id=step_id, command='evaluate_agent_vs_agent', description=description,
                              args={**args, **common}))
        entries.append(dict(kind=kind, run=run, a=a_label, b=b_label, step=played[identity]))

    players = {n: crossplay_players(r, count) for n, r in runs.items()}
    for run_name, sequence in players.items():
        for j, (late, a1, a2, _) in enumerate(sequence):
            for early, b1, b2, _ in sequence[:j]:
                add('within', run_name, late, early, key(a1, a2), key(b1, b2),
                    {**_agent('a', a1, a2), **_agent('b', b1, b2)}, f'{run_name}-{late}-vs-{early}',
                    f'{run_name}: {late} (A) vs {early} (B)')
            for entry in panel:
                if entry['kind'] == 'checkpoint' and entry['policy']['weights'] in (a1['weights'], a2['weights']):
                    continue  # The panel model itself: a mirror match carries no information.
                add('panel', run_name, late, entry['name'], key(a1, a2), ('panel', entry['name']),
                    {**_agent('a', a1, a2), **_panel_agent('b', entry)}, f'{run_name}-{late}-vs-{entry["name"]}',
                    f'{run_name}: {late} (A) vs fixed {entry["name"]} (B)')
    finals = {n: s[-1] for n, s in players.items() if s}
    names = list(finals)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            (_, a1, a2, _), (_, b1, b2, _) = finals[x], finals[y]
            if key(a1, a2) != key(b1, b2):
                add('cross', None, x, y, key(a1, a2), key(b1, b2), {**_agent('a', a1, a2), **_agent('b', b1, b2)},
                    f'final-{x}-vs-{y}', f'final {x} (A) vs final {y} (B)')
    if panel and len(names) > 1:
        reference = panel[0]
        for x in names:
            for y in names:
                (_, a1, _, _), (_, _, b2, _) = finals[x], finals[y]
                if x != y and key(a1, b2) != key(a1, a1):
                    add('mixed', None, f'{x}+{y}', reference['name'], key(a1, b2), ('panel', reference['name']),
                        {**_agent('a', a1, b2), **_panel_agent('b', reference)}, f'mixed-{x}-{y}-vs-{reference["name"]}',
                        f'mixed partnership: {x} first seat + {y} second seat (A) vs fixed {reference["name"]} (B)')
    config = dict(schema_version=1, name=name, output_root=root or f'runs/pipelines/{name}',
                  description=('Cross-play for docs/convergence.md, generated by examples.convergence_report '
                               f'from runs {", ".join(runs)}; {deals} duplicate deals per step, seed {seed}. '
                               'Summarize with the same command and --crossplay-results.'),
                  steps=steps)
    return config, entries


def crossplay_results(config_path, runs, panel, args):
    """Read a completed cross-play pipeline into per-run summaries, cross-run and partner results."""
    from examples import run_pipeline as rp
    from research.ppo.convergence import agent_a_marks, crossplay_summary
    pipeline = rp.load_config(config_path)
    expected, entries = crossplay_plan(runs, panel, args.crossplay_policies, args.deals, args.seed,
                                       args.bootstrap_samples, pipeline.name, pipeline.config['output_root'])
    steps = {s.id: s for s in pipeline.steps}
    agent_args = {s['id']: {k: v for k, v in s['args'].items() if k.startswith('agent-')} for s in expected['steps']}
    for step_id, wanted in agent_args.items():
        if step_id not in steps or {k: v for k, v in steps[step_id].args.items() if k.startswith('agent-')} != wanted:
            raise ValueError(f'{config_path} does not match these --run/--panel/--crossplay-policies settings '
                             f'(step {step_id}); regenerate it or pass the settings used to create it')
    results, missing = {}, set()
    for step_id in agent_args:
        path = rp.output_paths(steps[step_id])['output']
        try:
            result = agent_a_marks(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            result = None
        if result is None:
            missing.add(step_id)
        else:
            results[step_id] = (result['marks'], result['low'], result['high'])
    summary = dict(pipeline=str(config_path), missing_steps=sorted(missing), runs={}, cross=[], partners=[])
    for run_name, run in runs.items():
        sequence = crossplay_players(run, args.crossplay_policies)
        pairs = {(e['a'], e['b']): results[e['step']] for e in entries
                 if e['run'] == run_name and e['step'] in results}
        summary['runs'][run_name] = crossplay_summary(pairs, [label for label, *_ in sequence],
                                                      [p['name'] for p in panel],
                                                      {label: hands for label, _, _, hands in sequence})
    index = {(e['kind'], e['run'], e['a'], e['b']): e['step'] for e in entries}
    for e in entries:
        if e['kind'] == 'cross' and e['step'] in results:
            marks, low, high = results[e['step']]
            summary['cross'].append(dict(a=e['a'], b=e['b'], marks=marks, low=low, high=high))
    # Partner compatibility: a mixed partnership against the reference opponent,
    # minus the mean of the two runs' own final pairs against the same opponent.
    own = {}
    for run_name, run in runs.items():
        sequence = crossplay_players(run, args.crossplay_policies)
        step = index.get(('panel', run_name, sequence[-1][0], panel[0]['name'])) if sequence and panel else None
        if step in results:
            own[run_name] = results[step][0]
    for e in entries:
        if e['kind'] == 'mixed' and e['step'] in results:
            x, y = e['a'].split('+', 1)
            marks, low, high = results[e['step']]
            reference = (own[x] + own[y]) / 2 if x in own and y in own else None
            summary['partners'].append(dict(pair=e['a'], opponent=e['b'], marks=marks, low=low, high=high,
                                            own_pairs_mean=reference,
                                            compatibility=None if reference is None else marks - reference))
    return summary


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _hands(value):
    return 'N/A' if value is None else f'{value:,}'


def print_run(name, report):
    kind = 'paired partners' if report['paired'] else 'shared/single network'
    print(f'\n== {name}: {report["directory"]} (checkpoint version {report["version"]}, {kind})')
    for warning in report['warnings']:
        print(f'  ! {warning}')
    exposure = report['exposure']
    if exposure:
        classes = ', '.join(f'{c} {s:.1%}' for c, s in (exposure['by_class'] or {}).items())
        print(f'Trained against ({exposure["hands"]:,} hands{"" if exposure["identities_recorded"] else ", names only"}): '
              f'{classes or "see identities"}')
        for entry in exposure['by_identity']:
            span = (f', episodes {entry["first"]:,}-{entry["last"]:,}' if 'first' in entry else '')
            print(f'  {entry["cls"]:<10} {entry["label"]:<28} {entry["share"]:6.1%}{span}')
    else:
        print('Trained against: not recorded')
    for key, d in report['drift'].items():
        print(f'Drift {key} ({d["states"]:,} probe positions): KL(old || new) and most-likely-move changes')
        for i in range(len(d['consecutive_kl'])):
            print(f'  {d["policies"][i]:>18} -> {d["policies"][i + 1]:<18} {d["consecutive_kl"][i]:8.4f}  '
                  f'{d["consecutive_greedy_change"][i]:6.1%}')
        if len(d['policies']) > 2:
            print(f'  {"first":>18} -> {"final":<18} {d["kl_to_final"][0]:8.4f}  {d["greedy_change_to_final"][0]:6.1%}')
        recurrent = ', '.join(f'{d["policies"][r["index"]]} nearest {d["policies"][r["nearest_earlier"]]} '
                              f'(x{r["ratio"]:.2f} of the step)' for r in d['recurrent'])
        print(f'  recurrence: {recurrent or "none"}')
    for phase, d in report['partner_divergence'].items():
        print(f'Partner divergence {phase} ({d["states"]:,} positions): KL(1 || 2) {d["kl_1_to_2"]:.4f}, '
              f'KL(2 || 1) {d["kl_2_to_1"]:.4f}, most likely move differs {d["greedy_disagreement"]:.1%}')
    for opponent, s in report['strength'].items():
        if 'final_marks' in s:
            print(f'Validation vs {opponent} ({s["evaluations"]} points, {s["points"][-1]["games"]} games each): final '
                  f'{s["final_marks"]:+.3f} [{s["final_low"]:+.3f}, {s["final_high"]:+.3f}]; best {s["best_marks"]:+.3f} '
                  f'at {_hands(s["best_hands"])}; plateau from {_hands(s["plateau_hands"])} hands')
    gap = report['generalization']
    if gap:
        training = ('N/A' if gap['training_marks'] is None else f'{",".join(gap["training"])} {gap["training_marks"]:+.3f}')
        print(f'Generalization (final validation): training {training}; held-out {",".join(gap["held_out"])} '
              f'{gap["held_out_marks"]:+.3f}; gap ' + ('N/A' if gap['gap'] is None else f'{gap["gap"]:+.3f}'))


def print_crossplay(summary):
    print(f'\n== Cross-play: {summary["pipeline"]}')
    if summary['missing_steps']:
        print(f'  ! {len(summary["missing_steps"])} steps have no result yet: {", ".join(summary["missing_steps"][:6])}'
              + (' ...' if len(summary['missing_steps']) > 6 else ''))
    for run_name, s in summary['runs'].items():
        progress = s['transitive_progress']
        print(f'{run_name}: later vs earlier policies: {progress.get("later_better", 0)} significantly better, '
              f'{progress.get("later_worse", 0)} worse, {progress.get("inconclusive", 0)} inconclusive; '
              f'significant 3-cycles: {len(s["cycles"])}')
        for cycle in s['cycles']:
            print(f'  cycle: {" > ".join(cycle)} > {cycle[0]}')
        for row in s['final_vs_earlier']:
            print(f'  final vs {row["opponent"]:<18} {row["marks"]:+.3f} [{row["low"]:+.3f}, {row["high"]:+.3f}]')
        for opponent, table in s['panel'].items():
            for point in table['points']:
                print(f'  {point["policy"]:<18} vs {opponent:<6} {point["marks"]:+.3f} '
                      f'[{point["low"]:+.3f}, {point["high"]:+.3f}]')
            if table['plateau']:
                print(f'  vs {opponent}: plateau from {_hands(table["plateau"]["plateau_hands"])} hands')
    for row in summary['cross']:
        print(f'final {row["a"]} vs final {row["b"]}: {row["marks"]:+.3f} [{row["low"]:+.3f}, {row["high"]:+.3f}]')
    for row in summary['partners']:
        compatibility = ('N/A' if row['compatibility'] is None else f'{row["compatibility"]:+.3f} vs own pairs')
        print(f'mixed {row["pair"]} vs {row["opponent"]}: {row["marks"]:+.3f} [{row["low"]:+.3f}, {row["high"]:+.3f}]; '
              f'{compatibility}')


@record_output
def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.probe_hands < 1 or args.deals < 1 or args.crossplay_policies < 0 or args.bootstrap_samples < 1:
        p.error('--probe-hands, --deals and --bootstrap-samples must be positive; --crossplay-policies >= 0')
    from texas42.agents import VARIANTS
    from research.ppo import convergence
    names = [n.strip() for n in args.panel.split(',') if n.strip()]
    if any(n not in VARIANTS for n in names):
        p.error(f'--panel accepts built-in variants: {", ".join(VARIANTS)}')
    try:
        runs = {name: convergence.lineage(path) for name, path in named(args.run, '--run', p)}
        panel = [dict(kind='variant', name=n) for n in names]
        for name, path in named(args.panel_checkpoint, '--panel-checkpoint', p):
            policy = convergence.lineage_policy(path)
            panel.append(dict(kind='checkpoint', name=name, policy=policy))
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if len({e['name'] for e in panel}) != len(panel):
        p.error('duplicate panel names')
    probe = convergence.probe_positions(args.seed, args.probe_hands)
    print(f'Probe set: {args.probe_hands} Heuristic V1 hands, seed {args.seed}: '
          + ', '.join(f'{phase} {len(d["masks"]):,}' for phase, d in probe.items()) + ' positions')
    reports = {name: convergence.analyze(run, probe) for name, run in runs.items()}
    seen = {}
    for name, run in runs.items():
        for policy in run['policies']:
            seen.setdefault(policy['weights'], []).append(f'{name}/{policy["label"]}')
    shared = [labels for labels in seen.values() if len({l.split('/')[0] for l in labels}) > 1]
    for name, report in reports.items():
        print_run(name, report)
    for labels in shared:
        print(f'! same weights across runs: {" = ".join(labels)}')
    document = dict(settings=dict(seed=args.seed, probe_hands=args.probe_hands,
                                  probe_positions={k: len(v['masks']) for k, v in probe.items()}),
                    runs=reports, shared_weights_across_runs=shared)
    if args.crossplay_pipeline:
        from examples import run_pipeline as rp
        config, _ = crossplay_plan(runs, panel, args.crossplay_policies, args.deals, args.seed,
                                   args.bootstrap_samples, args.pipeline_name, args.pipeline_root)
        try:
            rp.plan(rp.parse_config(config, args.crossplay_pipeline))  # Validates every step's arguments.
        except rp.PipelineError as exc:
            p.error(str(exc))
        args.crossplay_pipeline.parent.mkdir(parents=True, exist_ok=True)
        args.crossplay_pipeline.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        print(f'\nCross-play pipeline: {args.crossplay_pipeline} ({len(config["steps"])} steps). Run it with\n'
              f'  python -m examples.run_pipeline --config {args.crossplay_pipeline}')
    if args.crossplay_results:
        from examples.run_pipeline import PipelineError
        try:
            document['crossplay'] = crossplay_results(args.crossplay_results, runs, panel, args)
        except (ValueError, PipelineError) as exc:
            p.error(str(exc))
        print_crossplay(document['crossplay'])
    if args.output:
        args.output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    # record_output flattens a returned list into the CSV export: one row per drift step.
    return [dict(run=name, sequence=key, before=d['policies'][i], after=d['policies'][i + 1],
                 hands_before=d['hands'][i], hands_after=d['hands'][i + 1], kl=d['consecutive_kl'][i],
                 greedy_change=d['consecutive_greedy_change'][i], states=d['states'])
            for name, report in reports.items() for key, d in report['drift'].items()
            for i in range(len(d['consecutive_kl']))]


if __name__ == '__main__':
    main()
