"""Reachable open-hand analysis using ordinary policies and engine transitions."""
from __future__ import annotations

import argparse
from secrets import randbits
from statistics import mean, median

from texas42.actions import PLAY_ACTIONS
from texas42.agents import VARIANTS, variant_factory
from texas42.arena import generate_schedule
from texas42.models import trump_name
from texas42.oracle import DealAnalysis, DealProgress, OracleResult, SearchLimits, analyze_deal, solve, solve_points
from texas42.parallel import AUTO_WORKER_CAP, resolve_workers
from texas42.phases import Phase
from texas42.players import format_hand
from texas42.selfplay import observation_from_view
from texas42.state import HandState


def generate_endgame(seed: int, remaining: int, *, policy: str = 'HHR',
                     partial_plays: int = 0) -> HandState:
    """Stop legal policy play at a trick boundary, then optionally advance 0–3 moves."""
    if type(remaining) is not int or remaining not in range(1, 8):
        raise ValueError('remaining must be 1–7')
    if type(partial_plays) is not int or partial_plays not in range(4):
        raise ValueError('partial_plays must be 0–3')
    deal = generate_schedule(1, seed, shaker=seed % 4)[0]
    state = HandState([[PLAY_ACTIONS[i] for i in hand] for hand in deal.hands],
                      deal.shaker, (0, 0), 7)
    factory = variant_factory(policy)
    agents = [factory(seat, seed + seat) for seat in range(4)]
    for seat, agent in enumerate(agents):
        agent.reset(seat)
    while state.phase != Phase.PLAY or len(state.completed) < 7 - remaining:
        obs = observation_from_view(state.view(), ruleset_id=state.ruleset_id)
        state.apply(agents[obs.seat].act(obs))
    for _ in range(partial_plays):
        obs = observation_from_view(state.view(), ruleset_id=state.ruleset_id)
        state.apply(agents[obs.seat].act(obs))
    return state


def print_position(state: HandState, result: OracleResult, *, objective: str = 'contract_success') -> None:
    contract = state.contract
    print(f'Contract: {contract.bid}, bidder seat {contract.bidder}, bidding team {contract.bidder % 2}')
    print(f'Trump: {trump_name(state.trump)}; scores: {state.points}')
    print(f'To act: seat {state.seat}, team {state.seat % 2}')
    for seat, hand in enumerate(state.hands):
        print(f'Seat {seat}: {format_hand(hand)}')
    print('Partial trick:', ' '.join(f'{seat}:{domino}' for seat, domino in state.trick) or '(empty)')
    if objective == 'final_points':
        print('Values: guaranteed final points for the bidding team (0–42); all hands visible')
    else:
        print('Values: +1 bidding team can force success; -1 defenders can force failure')
    for row in result.action_values:
        value = f'{row.value:+d} exact' if row.exact else f'[{row.lower:+d}, {row.upper:+d}] unresolved'
        print(f'  {row.action_id:2d} {PLAY_ACTIONS[row.action_id]}: {value}; searched={row.searched}')
    print(f'Root: value={result.value}, bounds=[{result.lower:+d}, {result.upper:+d}], exact={result.exact}')
    print(f'Proven optimal action: {result.optimal_action}; complete optimal set: {result.optimal_actions}')
    print(f'Status: {result.status}; all action values complete: {result.all_action_values_complete}')
    print(f'Nodes: {result.nodes}; cache hits: {result.cache_hits}; runtime: {result.elapsed_seconds:.6f}s')


def print_deal(hands, result: DealAnalysis) -> None:
    print('Open-hand bidding benchmark: all hands visible; not a hidden-information bid recommendation.')
    print('42 points establishes a one-mark contract; mark stake is not optimized.')
    for seat, hand in enumerate(hands):
        print(f'Seat {seat}: {format_hand(hand)}')
    for bidder in result.bidders:
        print(f'\nBidder seat {bidder.seat}, team {bidder.team} (bidder leads)')
        print('Trump | Points/bounds | Proven lead ID | Complete optimal lead IDs | Proof | All values exact')
        for trump in bidder.trumps:
            points = str(trump.value) if trump.exact else f'[{trump.lower}, {trump.upper}]'
            print(f'{trump_name(trump.trump)} | {points} | {trump.optimal_action} | '
                  f'{trump.optimal_actions} | {trump.status} | {trump.all_action_values_complete}')
        contract = str(bidder.maximum_contract) if bidder.maximum_contract is not None else 'none'
        if not bidder.contract_exact:
            contract = 'unresolved'
        print(f'Best points: {bidder.value}, bounds=[{bidder.lower}, {bidder.upper}]; '
              f'maximum makeable contract: {contract}; contract exact: {bidder.contract_exact}')
        print(f'Proven makeable contract: {bidder.proven_contract}; '
              f'highest still possible: {bidder.possible_contract}')
        choice = bidder.optimal_choice
        proven = f'{trump_name(choice.trump)} / {PLAY_ACTIONS[choice.action_id]} ({choice.action_id})' if choice else None
        choices = ('; '.join(f'{trump_name(c.trump)} / {PLAY_ACTIONS[c.action_id]} ({c.action_id})'
                            for c in bidder.optimal_choices) if bidder.optimal_choices is not None else None)
        print(f'Proven optimal trump/lead: {proven}; complete optimal pairs: {choices}')
        print(f'Bidder proof: {bidder.status}; all trump values exact: {bidder.all_trump_values_complete}; '
              f'all action values exact: {bidder.all_action_values_complete}')
    print(f'\nStatus: {result.status}; all alternative values complete: {result.all_alternative_values_complete}')
    print(f'Nodes: {result.nodes}; cache hits: {result.cache_hits}; runtime: {result.elapsed_seconds:.6f}s')


def print_progress(event: DealProgress) -> None:
    print(f'[{event.completed}/{event.total}] Bidder {event.bidder} | '
          f'Trump {trump_name(event.trump)} | Lead {PLAY_ACTIONS[event.action_id]} | '
          f'{event.status} | nodes={event.nodes} | {event.elapsed_seconds:.3f}s | '
          f'total={event.global_elapsed_seconds:.3f}s', flush=True)
    if event.bidder_contract_proven:
        contract = str(event.maximum_contract) if event.maximum_contract is not None else 'no makeable contract'
        print(f'Bidder {event.bidder}: maximum contract proven ({contract})', flush=True)


def main(argv: list[str] | None = None) -> None:
    defaults = SearchLimits()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int,
                        help='Deal/policy seed; random when omitted, printed for reproduction')
    parser.add_argument('--mode', choices=('endgame', 'deal'), default='endgame')
    parser.add_argument('--workers', type=int, default=1,
                        help='Spawn workers: 1 sequential, 0 auto; at most eight and available CPUs')
    parser.add_argument('--objective', choices=('contract_success', 'final_points'), default='contract_success',
                        help='Endgame objective; deal mode always uses final points')
    parser.add_argument('--remaining', type=int, choices=range(1, 8), default=3)
    parser.add_argument('--partial-plays', type=int, choices=range(4), default=0)
    parser.add_argument('--policy', choices=tuple(VARIANTS), default='HHR')
    parser.add_argument('--output', choices=('position_value', 'all_action_values', 'maximum_contract'),
                        default='all_action_values', help='Deal: all_action_values or maximum_contract; endgame: '
                        'all_action_values or position_value')
    parser.add_argument('--move-ordering', choices=('none', 'heuristic', 'tt', 'combined'), default='none')
    parser.add_argument('--no-proof-cancellation', action='store_false', dest='proof_cancellation',
                        help='Deal maximum_contract ablation: classify every lead without bidder cancellation')
    parser.add_argument('--max-nodes', type=int, default=defaults.max_nodes,
                        help='Nodes per opening lead in deal mode, per position otherwise (default: 1048576)')
    parser.add_argument('--time-seconds', type=float, default=defaults.time_seconds,
                        help='Total seconds for deal analysis, per position otherwise (default: 1800)')
    parser.add_argument('--max-cache-entries', type=int, default=defaults.max_cache_entries)
    parser.add_argument('--samples', type=int, default=1,
                        help='Consecutive seeds; >1 prints a bounded benchmark summary')
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.mode == 'deal':
        if args.samples != 1:
            parser.error('deal mode analyzes one complete deal; --samples must be 1')
    elif args.output == 'maximum_contract':
        parser.error('maximum_contract output requires --mode deal')
    try:
        resolve_workers(args.workers)
        limits = SearchLimits(args.time_seconds, args.max_nodes, args.max_cache_entries)
    except ValueError as error:
        parser.error(str(error))
    seed = args.seed if args.seed is not None else randbits(64)
    print(f'Seed: {seed}')
    if args.mode == 'deal':
        deal = generate_schedule(1, seed, shaker=seed % 4)[0]
        hands = [[PLAY_ACTIONS[i] for i in hand] for hand in deal.hands]
        count = min(AUTO_WORKER_CAP, resolve_workers(args.workers, 252))
        output = 'maximum_contract' if args.output == 'maximum_contract' else 'all_action_values'
        print(f'Starting oracle analysis: {output}\nSearch trees: 252')
        print(f'Move ordering: {args.move_ordering}; proof cancellation: '
              f'{output == "maximum_contract" and args.proof_cancellation}')
        print(f'Workers: {count}\nMaximum nodes per tree: {limits.max_nodes}\n'
              f'Maximum total time: {limits.time_seconds} seconds\n'
              f'Aggregate cache entries: {limits.max_cache_entries}', flush=True)
        print(f'Requested oracle workers: {args.workers}; resolved oracle workers: {count}; '
              'independent search tasks: 252', flush=True)
        result = analyze_deal(hands, limits=limits, workers=args.workers, progress=print_progress,
                              move_ordering=args.move_ordering, output=output,
                              proof_cancellation=args.proof_cancellation)
        print_deal(hands, result)
        print(f'Requested output complete: {result.requested_output_complete}; '
              f'maximum contracts proven: {sum(b.contract_exact for b in result.bidders)}/4')
        print('Settled tasks:', ', '.join(f'{status}={count}' for status, count in result.task_status_counts))
        print(f'Search statistics: {result.statistics}')
        return
    print(f'Per-position limits: nodes={limits.max_nodes}, seconds={limits.time_seconds}; '
          f'aggregate cache entries={limits.max_cache_entries}')
    results = []
    task_counts = []
    undecided = 0
    for index in range(args.samples):
        state = generate_endgame(seed + index, args.remaining,
                                 policy=args.policy, partial_plays=args.partial_plays)
        team = state.contract.bidder % 2
        target = state.contract.bid.target_points
        undecided += (sum(state.points) < 42 if args.objective == 'final_points' else
                      state.points[team] < target and state.points[1 - team] <= 42 - target)
        solver = solve_points if args.objective == 'final_points' else solve
        tasks = len(state.legal_action_ids())
        task_counts.append(tasks)
        count = min(AUTO_WORKER_CAP, resolve_workers(args.workers, tasks))
        if args.samples == 1:
            print(f'Requested oracle workers: {args.workers}; resolved oracle workers: {count}; '
                  f'independent search tasks: {tasks}')
        result = solver(state, output=args.output, limits=limits, workers=args.workers,
                        move_ordering=args.move_ordering)
        results.append(result)
        if args.samples == 1:
            print(f'Search statistics: {result.statistics}')
            print_position(state, result, objective=args.objective)
    if args.samples > 1:
        counts = [min(AUTO_WORKER_CAP, resolve_workers(args.workers, n)) for n in task_counts]
        print(f'Requested oracle workers: {args.workers}; resolved oracle workers: '
              f'{min(counts)}–{max(counts)}; independent search tasks: {sum(task_counts)}')
        exact = sum(result.status == 'exact' for result in results)
        print(f'{args.samples} positions; {undecided} initially undecided; requested output: {args.output}')
        print(f'Exact completion: {exact}/{args.samples} ({exact / args.samples:.1%})')
        print(f'All action values complete: {sum(r.all_action_values_complete for r in results)}/{args.samples}')
        print('Search statistics:', sum((r.statistics for r in results), start=type(results[0].statistics)()))
        for label, values in (
                ('Nodes', [r.nodes for r in results]),
                ('Cache hits', [r.cache_hits for r in results]),
                ('Runtime seconds (solve only)', [r.elapsed_seconds for r in results])):
            print(f'{label}: mean={mean(values):.6f}, median={median(values):.6f}, max={max(values):.6f}')


if __name__ == '__main__':
    main()
