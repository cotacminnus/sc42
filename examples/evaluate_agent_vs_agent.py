"""Duplicate-deal evaluation of two independently composed phase policies."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from examples.benchmark_output import format_report, format_breakdown, format_performance, table
from examples.experiment_output import record_output, add_evaluation_arguments
from research.ppo.composition import (CheckpointIdentity, PhasePolicySpec, CompositePolicySpec,
                                     CompositeAgentFactory, comparison_metadata, evaluate_composed)
from research.ppo.diagnostics import evaluation_diagnostics


def policy_spec(args, agent):
    prefix = f'agent_{agent}_'
    policies = []
    for phase in ('bid', 'trump', 'play'):
        strategy = getattr(args, prefix + phase)
        override = getattr(args, prefix + phase + '_checkpoint')
        if override is not None and strategy != 'neural':
            raise ValueError(f'Agent {agent.upper()} {phase}: checkpoint override requires neural strategy')
        path = override or getattr(args, prefix + 'checkpoint')
        identity = CheckpointIdentity.from_path(path) if strategy == 'neural' and path else None
        policies.append(PhasePolicySpec(strategy, identity))
    return CompositePolicySpec(agent.upper(), *policies)


@record_output
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for agent in ('a', 'b'):
        p.add_argument(f'--agent-{agent}-checkpoint', type=Path)
        for phase in ('bid', 'trump', 'play'):
            choices = ('neural', 'heuristic', 'random', 'conservative') if phase == 'bid' else ('neural', 'heuristic', 'random')
            p.add_argument(f'--agent-{agent}-{phase}', choices=choices, default='heuristic')
            p.add_argument(f'--agent-{agent}-{phase}-checkpoint', type=Path)
    add_evaluation_arguments(p, seed=42)
    p.add_argument('--stochastic-neural', action='store_true', help='Seeded sampling instead of greedy neural inference')
    p.add_argument('--episode', choices=('hand', 'match'), default='hand')
    p.add_argument('--target-marks', type=int, default=7)
    args = p.parse_args(argv)
    started = perf_counter()
    try:
        a, b = (policy_spec(args, agent) for agent in ('a', 'b'))
        # Validate availability in the parent so unsupported neural phases fail clearly.
        for spec in (a, b):
            CompositeAgentFactory(spec).prepare()
        metadata = comparison_metadata(a, b)
    except (ValueError, OSError, ImportError) as exc:
        p.error(str(exc))
    for spec in (a, b):
        print(f'Agent {spec.name}: {spec.label()}')
        for phase, policy in spec.items():
            checkpoint = f' [{policy.checkpoint.sha256}] {policy.checkpoint.path}' if policy.checkpoint else ''
            print(f'  {phase.title()}: {policy.strategy.title()}{checkpoint}')
    for phase, same in metadata['neural_parameters_identical'].items():
        print(f'Shared {dict(bid="bidding", trump="trump", play="playing")[phase]} parameters: ' + ('N/A (not both neural)' if same is None else 'YES' if same else 'NO'))
    print(f'Shared complete checkpoint: {metadata["shared_complete_checkpoint"]}')
    print(f'Only one phase differs: {metadata["only_one_phase_differs"]}; differing phases: {", ".join(metadata["differing_phases"]) or "none"}')
    preparation_seconds = perf_counter() - started
    performance = {}
    try:
        report = evaluate_composed(a, b, stochastic=args.stochastic_neural,
            num_deals=args.deals, seed=args.seed, bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples, threads=args.threads,
            episode=args.episode, target_marks=args.target_marks, performance=performance)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    elapsed = perf_counter() - started
    print(f'Configuration/checkpoint preparation: {preparation_seconds:.3f} s')
    print(format_performance(performance))
    print(format_report(report, {'A': a.label(), 'B': b.label()}, elapsed))
    details = evaluation_diagnostics(report)
    print(format_breakdown(details))
    print('\nAuction calls and partnership roles')
    print(table(('Policy', 'Bids', 'Passes', 'Role', 'Hands', 'Win rate', 'Signed marks'), [
        (name, d['bid_calls'], d['pass_calls'], role, stats['hands'],
         f'{stats["win_rate"]:.1%}' if stats['win_rate'] is not None else 'N/A',
         f'{stats["mean_signed_marks"]:+.3f}' if stats['mean_signed_marks'] is not None else 'N/A')
        for name, d in details.items() for role, stats in d['roles'].items()]))
    print('Contracts = auctions won, not hands won. Phase effects interact; this comparison does not establish global optimality.')
    if (all(metadata['phase_policies_identical'].values()) and not args.stochastic_neural
            and all(s.strategy != 'random' for _, s in a.items())):
        print('Identical deterministic policies: paired symmetry check, not evidence of strength.')
    if args.output:
        data = report.to_dict()
        data.update(composition=dict(agents={'A': a.to_dict(), 'B': b.to_dict()},
            comparisons=metadata, neural_inference='stochastic' if args.stochastic_neural else 'greedy',
            sampling_scheme='composite-phase-v1'), breakdown=details)
        args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'JSON results saved to {args.output}')
    return report


if __name__ == '__main__':
    main()
