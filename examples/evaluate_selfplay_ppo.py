"""Evaluate phase-aware bidding/play policies against variants and frozen anchors."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from examples.experiment_output import record_output, add_output_arguments
from examples.benchmark_output import format_report, format_breakdown
from research.ppo.checkpoint import load_model
from research.ppo.multiphase import PhaseActorCritic
from research.ppo.opponents import PolicySpec
from research.ppo.selfplay_evaluation import evaluate_phases, evaluation_diagnostics, frozen_model
from texas42.agents import VARIANTS


@record_output
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--anchor-checkpoint', type=Path)
    p.add_argument('--opponents', default='HHH,CHH,HHR,anchor')
    p.add_argument('--opponent-checkpoint', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--deals', type=int, default=2048)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--bootstrap-seed', type=int, default=0)
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', type=Path)
    add_output_arguments(p)
    a = p.parse_args(argv)
    try:
        model, data = load_model(a.checkpoint)
        if not isinstance(model, PhaseActorCritic):
            raise ValueError('Use evaluate_ppo for original play-only checkpoints')
        learner = PolicySpec.checkpoint('learner', a.checkpoint)
        anchor = PolicySpec.checkpoint('anchor', a.anchor_checkpoint or data['config']['anchor']['path'])
        if anchor.sha256 != data['metadata']['original_play_sha256']:
            raise ValueError('Anchor differs from the original pretrained playing policy')
        registry = {n: PolicySpec(n) for n in VARIANTS}
        registry['anchor'] = anchor
        for entry in a.opponent_checkpoint:
            n, path = entry.split('=', 1)
            if n in registry or not n:
                raise ValueError('Duplicate/empty opponent name')
            registry[n] = PolicySpec.checkpoint(n, path)
        names = tuple(n.strip() for n in a.opponents.split(','))
        if len(set(names)) != len(names) or any(n not in registry for n in names):
            raise ValueError('Unknown or duplicate evaluation opponent')
        specs = tuple(registry[n] for n in names)
        telemetry, started = {}, perf_counter()
        reports = evaluate_phases(model, specs, deals=a.deals, seed=a.seed, bootstrap_seed=a.bootstrap_seed,
            bootstrap_samples=a.bootstrap_samples, threads=a.threads, performance=telemetry)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    same_play = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(
        model.networks['play'].parameters(), frozen_model(anchor).parameters()))
    print(f'Anchor comparison shares identical playing parameters: {same_play}')
    print(f'Learner SHA256: {learner.sha256}\nOriginal anchor SHA256: {anchor.sha256}')
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
        a.output.write_text(json.dumps(dict(checkpoint_id=learner.sha256, anchor_id=anchor.sha256,
            opponent_identities={s.name: s.sha256 for s in specs}, same_playing_parameters=same_play,
            evaluations={n: r.to_dict() for n, r in reports.items()}, breakdowns=details), indent=2) + '\n')
    return reports


if __name__ == '__main__':
    main()
