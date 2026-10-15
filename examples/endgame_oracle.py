"""Reachable open-hand analysis using ordinary policies and engine transitions."""
from __future__ import annotations

import argparse
from statistics import mean, median

from texas42.actions import PLAY_ACTIONS
from texas42.agents import VARIANTS, variant_factory
from texas42.arena import generate_schedule
from texas42.models import trump_name
from texas42.oracle import OracleResult, SearchLimits, solve
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


def print_position(state: HandState, result: OracleResult) -> None:
    contract = state.contract
    print(f'Contract: {contract.bid}, bidder seat {contract.bidder}, bidding team {contract.bidder % 2}')
    print(f'Trump: {trump_name(state.trump)}; scores: {state.points}')
    print(f'To act: seat {state.seat}, team {state.seat % 2}')
    for seat, hand in enumerate(state.hands):
        print(f'Seat {seat}: {format_hand(hand)}')
    print('Partial trick:', ' '.join(f'{seat}:{domino}' for seat, domino in state.trick) or '(empty)')
    print('Values: +1 bidding team can force success; -1 defenders can force failure')
    for row in result.action_values:
        value = f'{row.value:+d} exact' if row.exact else f'[{row.lower:+d}, {row.upper:+d}] unresolved'
        print(f'  {row.action_id:2d} {PLAY_ACTIONS[row.action_id]}: {value}; searched={row.searched}')
    print(f'Root: value={result.value}, bounds=[{result.lower:+d}, {result.upper:+d}], exact={result.exact}')
    print(f'Proven optimal action: {result.optimal_action}; complete optimal set: {result.optimal_actions}')
    print(f'Status: {result.status}; all action values complete: {result.all_action_values_complete}')
    print(f'Nodes: {result.nodes}; cache hits: {result.cache_hits}; runtime: {result.elapsed_seconds:.6f}s')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--remaining', type=int, choices=range(1, 8), default=3)
    parser.add_argument('--partial-plays', type=int, choices=range(4), default=0)
    parser.add_argument('--policy', choices=tuple(VARIANTS), default='HHR')
    parser.add_argument('--output', choices=('position_value', 'all_action_values'), default='all_action_values')
    parser.add_argument('--max-nodes', type=int)
    parser.add_argument('--time-seconds', type=float)
    parser.add_argument('--max-cache-entries', type=int, default=100_000)
    parser.add_argument('--samples', type=int, default=1,
                        help='Consecutive seeds; >1 prints a bounded benchmark summary')
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error('--samples must be positive')
    try:
        limits = SearchLimits(args.time_seconds, args.max_nodes, args.max_cache_entries)
    except ValueError as error:
        parser.error(str(error))
    results = []
    undecided = 0
    for index in range(args.samples):
        state = generate_endgame(args.seed + index, args.remaining,
                                 policy=args.policy, partial_plays=args.partial_plays)
        team = state.contract.bidder % 2
        target = state.contract.bid.target_points
        undecided += state.points[team] < target and state.points[1 - team] <= 42 - target
        result = solve(state, output=args.output, limits=limits)
        results.append(result)
        if args.samples == 1:
            print_position(state, result)
    if args.samples > 1:
        exact = sum(result.status == 'exact' for result in results)
        print(f'{args.samples} positions; {undecided} initially undecided; requested output: {args.output}')
        print(f'Exact completion: {exact}/{args.samples} ({exact / args.samples:.1%})')
        print(f'All action values complete: {sum(r.all_action_values_complete for r in results)}/{args.samples}')
        for label, values in (
                ('Nodes', [r.nodes for r in results]),
                ('Cache hits', [r.cache_hits for r in results]),
                ('Runtime seconds (solve only)', [r.elapsed_seconds for r in results])):
            print(f'{label}: mean={mean(values):.6f}, median={median(values):.6f}, max={max(values):.6f}')


if __name__ == '__main__':
    main()
