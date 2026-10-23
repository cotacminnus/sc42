"""Evaluate phase-aware bidding/play policies against variants and frozen anchors."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from examples.experiment_output import record_output, add_evaluation_arguments
from examples.benchmark_output import format_report, format_breakdown
from research.ppo.checkpoint import load_model
from research.ppo.multiphase import PhaseActorCritic
from research.ppo.opponents import NO_ANCHOR, PolicySpec, policy_registry
from research.ppo.selfplay_evaluation import evaluate_phases, evaluation_diagnostics, frozen_model


def build_parser():
    """The command's argparse definition (also used by examples.run_pipeline to validate)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--anchor-checkpoint', type=Path)
    p.add_argument('--opponents', help='default HHH,CHH,HHR,anchor; HHH,CHH,HHR for anchor-free checkpoints')
    p.add_argument('--opponent-checkpoint', action='append', default=[], metavar='NAME=PATH')
    add_evaluation_arguments(p)
    return p


@record_output
def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    try:
        model, data = load_model(a.checkpoint)
        if not isinstance(model, PhaseActorCritic):
            raise ValueError('Use evaluate_ppo for original play-only checkpoints')
        learner = PolicySpec.checkpoint('learner', a.checkpoint)
        recorded = data['config'].get('anchor')
        if recorded is None and a.anchor_checkpoint is not None:
            raise ValueError('This checkpoint has no original play anchor to override')
        # Anchor-free (from-scratch) lineages register no anchor; never substitute another policy.
        anchor = (PolicySpec.checkpoint('anchor', a.anchor_checkpoint or recorded['path'])
                  if recorded is not None else None)
        if anchor is not None and anchor.sha256 != data['metadata']['original_play_sha256']:
            raise ValueError('Anchor differs from the original pretrained playing policy')
        registry = policy_registry(anchor, a.opponent_checkpoint)
        names = tuple(n.strip() for n in (a.opponents or ('HHH,CHH,HHR,anchor' if anchor else 'HHH,CHH,HHR')).split(','))
        if 'anchor' in names and anchor is None:
            raise ValueError(NO_ANCHOR)
        if len(set(names)) != len(names) or any(n not in registry for n in names):
            raise ValueError('Unknown or duplicate evaluation opponent')
        specs = tuple(registry[n] for n in names)
        telemetry, started = {}, perf_counter()
        reports = evaluate_phases(model, specs, deals=a.deals, seed=a.seed, bootstrap_seed=a.bootstrap_seed,
            bootstrap_samples=a.bootstrap_samples, threads=a.threads, performance=telemetry)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    same_play = None if anchor is None else all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(
        model.networks['play'].parameters(), frozen_model(anchor).parameters()))
    print(f'Anchor comparison shares identical playing parameters: '
          f'{"N/A (no original anchor)" if same_play is None else same_play}')
    print(f'Learner SHA256: {learner.sha256}\nOriginal anchor SHA256: {anchor.sha256 if anchor else "none"}')
    details = {}
    for name, report in reports.items():
        print(f'Opponent {name}; identity {registry[name].sha256 or "heuristic variant"}')
        print(format_report(report, {'A': 'neural ' + ','.join(model.neural_phases), 'B': name},
                            telemetry[name]['evaluation_total_seconds']))
        details[name] = evaluation_diagnostics(report)
        print(format_breakdown(details[name]))
        for policy in ('A', 'B'):
            d = details[name][policy]
            print(f'{policy}: {d["bid_calls"]} bids, {d["pass_calls"]} passes')
            for role, stats in d['roles'].items():
                rate = f"{stats['win_rate']:.1%}" if stats['win_rate'] is not None else 'N/A'
                marks = f"{stats['mean_signed_marks']:+.3f}" if stats['mean_signed_marks'] is not None else 'N/A'
                print(f'  {role}: {stats["hands"]} hands; win {rate}; signed marks {marks}')
    print(f'Total evaluation: {perf_counter() - started:.3f}s')
    if a.output:
        a.output.write_text(json.dumps(dict(checkpoint_id=learner.sha256, anchor_id=anchor.sha256 if anchor else None,
            opponent_identities={s.name: s.sha256 for s in specs}, same_playing_parameters=same_play,
            evaluations={n: r.to_dict() for n, r in reports.items()}, breakdowns=details), indent=2) + '\n')
    return reports


if __name__ == '__main__':
    main()
