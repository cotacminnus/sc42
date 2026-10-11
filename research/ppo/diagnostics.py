"""Dependency-free reporting of public arena results."""
def evaluation_diagnostics(report):
    """Resolved-contract diagnostics and all four public auction-call frequencies."""
    from examples.heuristic_benchmark import breakdown
    details = breakdown(report)
    for policy in ('A', 'B'):
        roles = {role: dict(hands=0, wins=0, signed_marks_sum=0) for role in ('declarer', 'defender')}
        calls = bids = 0
        for game in report.games:
            seats = tuple(s for s, name in enumerate(game.seating) if name == policy)
            for hand in game.hands:
                calls += len(seats)
                bids += sum(hand.bid_counts[s] for s in seats)
                role = 'declarer' if hand.seating[hand.result.bidder] == policy else 'defender'
                won = any(s % 2 == hand.result.scoring_team for s in seats)
                roles[role]['hands'] += 1
                roles[role]['wins'] += int(won)
                roles[role]['signed_marks_sum'] += hand.result.marks_awarded * (1 if won else -1)
        for stats in roles.values():
            n = stats['hands']
            stats['win_rate'] = stats['wins'] / n if n else None
            stats['mean_signed_marks'] = stats['signed_marks_sum'] / n if n else None
        details[policy].update(roles=roles, bid_calls=bids, pass_calls=calls - bids,
                              bid_frequency=bids / calls if calls else None,
                              hands_before_resolution=1 if calls else None)
    return details
