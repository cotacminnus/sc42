"""Duplicate-deal evaluation of two independently composed phase policies."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from examples.benchmark_output import format_report, format_breakdown, format_performance, table
from examples.experiment_output import record_output, add_evaluation_arguments
from research.ppo.composition import (PIMC_LEAVES, PIMC_LIKELIHOODS, STRATEGIES, CompositeAgentFactory,
                                     comparison_metadata, evaluate_composed, resolve_policy_spec,
                                     strategy_label)
from texas42.agents.pimc import DEFAULT_PIMC_SAMPLES
from texas42.belief import DEFAULT_EPSILON
from research.ppo.diagnostics import evaluation_diagnostics

PHASES = ('bid', 'trump', 'play')


def policy_spec(args, agent):
    prefix = f'agent_{agent}_'
    return resolve_policy_spec(agent.upper(),
        tuple(getattr(args, prefix + phase) for phase in PHASES),
        getattr(args, prefix + 'checkpoint'),
        tuple(getattr(args, prefix + phase + '_checkpoint') for phase in PHASES),
        pimc_checkpoint=getattr(args, prefix + 'pimc_model', None))


def partner_spec(args, agent, first):
    """Second-seat (seat 2/3) composition for one agent, or None for a mirrored team.

    Used for paired-partner models, or for cross-play such as a trained model
    beside a heuristic partner. An unset partner phase inherits the first
    seat's strategy. A neural partner phase loads, in order: its own
    --agent-X-partner-PHASE-checkpoint, then --agent-X-partner-checkpoint, and
    only when neither partner checkpoint flag is given, the first seat's sources.
    A partner 'pimc' phase uses the agent's --agent-X-pimc-model (if any).
    """
    prefix = f'agent_{agent}_partner_'
    strategies = tuple(getattr(args, prefix + phase) for phase in PHASES)
    default = getattr(args, prefix + 'checkpoint')
    overrides = tuple(getattr(args, prefix + phase + '_checkpoint') for phase in PHASES)
    if all(value is None for value in (*strategies, default, *overrides)):
        return None
    strategies = tuple(own or inherited.strategy for own, (_, inherited) in zip(strategies, first.items()))
    if default is None:
        default = getattr(args, f'agent_{agent}_checkpoint')
        inherited = tuple(getattr(args, f'agent_{agent}_{phase}_checkpoint') if strategy == 'neural' else None
                          for phase, strategy in zip(PHASES, strategies))
        overrides = tuple(own or source for own, source in zip(overrides, inherited))
    return resolve_policy_spec(f'{agent.upper()} partner', strategies, default, overrides,
                               pimc_checkpoint=getattr(args, f'agent_{agent}_pimc_model', None))


def describe(title, spec):
    print(f'{title}: {spec.label()}')
    for phase, policy in spec.items():
        checkpoint = f' [{policy.checkpoint.sha256}] {policy.checkpoint.path}' if policy.checkpoint else ''
        print(f'  {phase.title()}: {strategy_label(policy.strategy)}{checkpoint}')


def team_label(spec, partner):
    return spec.label() if partner is None else f'{spec.label()} + partner {partner.label()}'


def build_parser():
    """The command's argparse definition (also used by examples.run_pipeline to validate)."""
    p = argparse.ArgumentParser(description=__doc__)
    for agent in ('a', 'b'):
        p.add_argument(f'--agent-{agent}-checkpoint', type=Path)
        p.add_argument(f'--agent-{agent}-partner-checkpoint', type=Path,
                       help=f'default neural source for agent {agent.upper()}\'s second seat (2/3), '
                            'e.g. partner-2 of a paired training run')
        p.add_argument(f'--agent-{agent}-pimc-model', type=Path,
                       help=f'trained checkpoint guiding agent {agent.upper()}\'s pimc phases (model-guided '
                            'search: the model plays rollouts; see --pimc-leaf/--pimc-likelihood)')
        for phase in PHASES:
            p.add_argument(f'--agent-{agent}-{phase}', choices=STRATEGIES[phase], default='heuristic')
            p.add_argument(f'--agent-{agent}-{phase}-checkpoint', type=Path)
            p.add_argument(f'--agent-{agent}-partner-{phase}', choices=STRATEGIES[phase],
                           help='second-seat strategy; defaults to the first seat\'s')
            p.add_argument(f'--agent-{agent}-partner-{phase}-checkpoint', type=Path)
    add_evaluation_arguments(p, seed=42)
    p.add_argument('--stochastic-neural', action='store_true', help='Seeded sampling instead of greedy neural inference')
    p.add_argument('--pimc-samples', type=int, default=DEFAULT_PIMC_SAMPLES,
                   help='sampled hidden deals per PIMC decision (trump/play strategy "pimc"); '
                        'cost grows linearly, so pair large values with few --deals')
    p.add_argument('--pimc-leaf', choices=PIMC_LEAVES, default='none',
                   help='model-guided pimc only: "value" stops rollouts after the next trick and scores '
                        'the rest with the model value head (signed-marks objective, much faster); '
                        '"value-turn" stops at the searcher\'s own next decision instead, so the value '
                        'head takes the searcher\'s viewpoint')
    p.add_argument('--pimc-likelihood', choices=PIMC_LIKELIHOODS, default='uniform',
                   help='model-guided pimc only: "model" weights hidden deals by the model\'s move '
                        'probabilities for the observed bids and plays')
    p.add_argument('--pimc-epsilon', type=float, default=DEFAULT_EPSILON,
                   help='likelihood softening for unexpected moves, in [0, 1] (with --pimc-likelihood model)')
    p.add_argument('--episode', choices=('hand', 'match'), default='hand')
    p.add_argument('--target-marks', type=int, default=7)
    return p


@record_output
def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    started = perf_counter()
    if args.pimc_samples < 1:
        p.error('--pimc-samples must be positive')
    if not 0.0 <= args.pimc_epsilon <= 1.0:
        p.error('--pimc-epsilon must be between 0 and 1')
    # Run-level PIMC settings, applied by every CompositeAgentFactory below.
    pimc = dict(pimc_samples=args.pimc_samples, pimc_leaf=args.pimc_leaf,
                pimc_likelihood=args.pimc_likelihood, pimc_epsilon=args.pimc_epsilon)
    try:
        a, b = (policy_spec(args, agent) for agent in ('a', 'b'))
        # Optional second-seat compositions; None keeps one spec for both seats.
        a2, b2 = partner_spec(args, 'a', a), partner_spec(args, 'b', b)
        teams = {'A': (a, a2), 'B': (b, b2)}
        for name, specs in teams.items():
            if (getattr(args, f'agent_{name.lower()}_pimc_model') is not None
                    and not any(s.strategy == 'pimc' for spec in specs if spec for _, s in spec.items())):
                raise ValueError(f'--agent-{name.lower()}-pimc-model needs a pimc phase for agent {name}')
        guided = any(s.strategy == 'pimc' and s.checkpoint is not None
                     for spec in (a, b, a2, b2) if spec for _, s in spec.items())
        if (args.pimc_leaf != 'none' or args.pimc_likelihood != 'uniform') and not guided:
            raise ValueError('--pimc-leaf/--pimc-likelihood apply to model-guided pimc; '
                             'add --agent-X-pimc-model for an agent with a pimc phase')
        # Validate availability in the parent so unsupported neural phases fail clearly.
        for spec in (a, b, a2, b2):
            if spec is not None:
                CompositeAgentFactory(spec, **pimc).prepare()
        metadata = comparison_metadata(a, b)
        partners = {'A': (a, a2), 'B': (b, b2)}
        partner_metadata = {name: comparison_metadata(first, second) if second else None
                            for name, (first, second) in partners.items()}
    except (ValueError, OSError, ImportError) as exc:
        p.error(str(exc))
    for name, (spec, partner) in partners.items():
        describe(f'Agent {name}' + (' first seat (0/1)' if partner else ''), spec)
        if partner:
            describe(f'Agent {name} partner seat (2/3)', partner)
            same = partner_metadata[name]['neural_parameters_identical']
            shared = ', '.join(f'{phase} {"YES" if value else "NO"}' for phase, value in same.items() if value is not None)
            print(f'  Partner shares neural parameters with first seat: {shared or "N/A (no common neural phase)"}')
    for phase, same in metadata['neural_parameters_identical'].items():
        print(f'Shared {dict(bid="bidding", trump="trump", play="playing")[phase]} parameters: ' + ('N/A (not both neural)' if same is None else 'YES' if same else 'NO'))
    print(f'Shared complete checkpoint: {metadata["shared_complete_checkpoint"]}')
    print(f'Only one phase differs: {metadata["only_one_phase_differs"]}; differing phases: {", ".join(metadata["differing_phases"]) or "none"}')
    uses_pimc = any(s.strategy == 'pimc' for spec in (a, b, a2, b2) if spec for _, s in spec.items())
    if uses_pimc:
        print(f'PIMC: {args.pimc_samples} sampled deals per decision (roughly 0.1-2 s per decision; '
              'see docs/belief.md)')
        print('  Plain pimc phases: Heuristic V1 rollouts, uniform belief, expected points')
        if guided:
            leaf = {'value': 'rollouts stop after the next trick, model value head scores the rest (signed marks)',
                    'value-turn': "rollouts stop at the searcher's next decision, model value head scores "
                                  'the rest from its own view (signed marks)'}.get(
                args.pimc_leaf, 'full-hand rollouts (expected points)')
            belief = (f'model-weighted belief (epsilon {args.pimc_epsilon})'
                      if args.pimc_likelihood == 'model' else 'uniform belief')
            print(f'  Model-guided pimc phases: model rollouts; {leaf}; {belief}')
    preparation_seconds = perf_counter() - started
    performance = {}
    try:
        report = evaluate_composed(a, b, a_partner=a2, b_partner=b2, stochastic=args.stochastic_neural,
            **pimc, num_deals=args.deals, seed=args.seed,
            bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples,
            threads=args.threads, episode=args.episode, target_marks=args.target_marks,
            performance=performance)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    elapsed = perf_counter() - started
    print(f'Configuration/checkpoint preparation: {preparation_seconds:.3f} s')
    print(format_performance(performance))
    print(format_report(report, {'A': team_label(a, a2), 'B': team_label(b, b2)}, elapsed))
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
            and a2 is None and b2 is None
            and all(s.strategy not in ('random', 'pimc') for _, s in a.items())):
        print('Identical deterministic policies: paired symmetry check, not evidence of strength.')
    if args.output:
        data = report.to_dict()
        data.update(composition=dict(agents={'A': a.to_dict(), 'B': b.to_dict()},
            comparisons=metadata, neural_inference='stochastic' if args.stochastic_neural else 'greedy',
            sampling_scheme='composite-phase-v1'), breakdown=details)
        if uses_pimc:
            data['composition']['pimc_samples'] = args.pimc_samples
        if guided:
            # Model-guided search settings; the guide checkpoints are in each agent's
            # phase specification (strategy 'pimc' with a checkpoint).
            data['composition'].update(pimc_leaf=args.pimc_leaf, pimc_likelihood=args.pimc_likelihood,
                                       pimc_epsilon=args.pimc_epsilon)
        if a2 or b2:
            # Present only for paired/cross-play teams, so mirrored-team JSON is unchanged.
            data['composition']['partners'] = {name: second.to_dict() if second else None
                                               for name, (_, second) in partners.items()}
            data['composition']['partner_comparisons'] = partner_metadata
            data['composition']['partner_seat_rule'] = 'first spec: seat 0/1; partner spec: seat 2/3'
        args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'JSON results saved to {args.output}')
    return report


if __name__ == '__main__':
    main()
