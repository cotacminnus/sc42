"""Dependency-free terminal presentation shared by benchmark examples."""


def table(headers, rows):
    rows = [tuple(str(value) for value in row) for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in rows)])
              for i, header in enumerate(headers)]
    def line(row):
        return '  '.join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
    return '\n'.join([line(headers), line(tuple('-' * width for width in widths)),
                      *(line(row) for row in rows)])


def format_report(report, labels, elapsed):
    """Elapsed time covers evaluation plus bootstrap, excluding formatting/export."""
    def number(value, percent=False):
        if value is None:
            return 'N/A'
        return f'{value:.1%}' if percent else f'{value:.3f}'

    def interval(bounds, percent=False):
        if bounds is None:
            return 'N/A'
        return f'{number(bounds.low, percent)} to {number(bounds.high, percent)}'

    intervals = {item.policy: item for item in report.intervals}
    hands = sum(game.num_hands for game in report.games)
    decisions = sum(hand.decisions for game in report.games for hand in game.hands)
    speed = (f'{report.games_played / elapsed:.1f} games/s; '
             f'{decisions / elapsed:.1f} decisions/s') if elapsed > 0 else 'throughput N/A'
    lines = [
        'Evaluation benchmark',
        ' | '.join(f'{policy}: {name}' for policy, name in labels.items()),
        f'{len({game.group_id for game in report.games})} duplicate groups | '
        f'{report.games_played} games | {hands} hands | {decisions} decisions',
        f'Seed: {report.seed} | Bootstrap seed: {report.bootstrap_seed} | '
        f'Bootstrap samples: {report.bootstrap_samples}',
        f'Runtime (evaluation + bootstrap): {elapsed:.3f} s | {speed}',
        '',
        table(('Policy', 'Win rate', '95% CI', 'Marks/hand', '95% CI'), [
            (m.policy, number(m.win_rate, True), interval(intervals[m.policy].win_rate, True),
             number(m.mean_signed_marks), interval(intervals[m.policy].mean_signed_marks))
            for m in report.metrics]),
        '',
        table(('Policy', 'Contracts', 'Success rate', 'Mean value (marks)', 'Bid frequency'), [
            (m.policy, m.contracts_declared, number(m.contract_success_rate, True),
             number(m.mean_contract_value), number(m.bidding_frequency, True))
            for m in report.metrics]),
    ]
    if report.average_match_length is not None:
        lines.append(f'Average match length: {report.average_match_length:.3f} hands')
    lines.append('N/A = unavailable; 95% intervals resample complete duplicate groups.')
    return '\n'.join(lines)


def format_breakdown(details):
    lines = ['', 'Resolved contracts', table(
        ('Policy', 'Successful', 'Failed', 'Point bids', 'Mark bids'),
        [(policy, row['successful'], row['failed'], row['bid_types'].get('points', 0),
          row['bid_types'].get('marks', 0)) for policy, row in details.items()])]
    for policy, row in details.items():
        lines.extend(['', f'{policy}: outcomes by final bid'])
        bids = sorted(row['contract_values'], key=lambda key: (key.startswith('marks:'),
                                                               int(key.split(':')[1])))
        lines.append(table(('Bid', 'Successful', 'Failed'), [
            (f'{bid.split(":")[1]} {bid.split(":")[0]}',
             row['contract_values'][bid]['successful'], row['contract_values'][bid]['failed'])
            for bid in bids]) if bids else 'No contracts declared.')
        trumps = row['trumps']
        lines.append('Trump selections: ' + (', '.join(
            f'{trump}: {trumps[trump]}' for trump in sorted(trumps)) if trumps else 'none'))
    origins = []
    for policy, row in details.items():
        for origin, counts in row['thirty_point_bids'].items():
            total = counts['successful'] + counts['failed']
            rate = f"{counts['successful'] / total:.1%}" if total else 'N/A'
            origins.append((policy, origin.title(), counts['successful'], counts['failed'], rate))
    lines.extend(['', '30-point contract outcomes', table(
        ('Policy', 'Origin', 'Successful', 'Failed', 'Success rate'), origins)])
    def timing_row(policy, group, data):
        def mean(key):
            return f"{data[key]:.3f}" if data[key] is not None else 'N/A'
        return (policy, group, data['contracts'], mean('mean_decided_trick'),
                mean('mean_remaining_tricks'))

    timing_rows, distributions = [], []
    for policy, row in details.items():
        timing = row['decision_timing']
        groups = [('All', timing), *timing['by_outcome'].items(), *timing['by_origin'].items()]
        for group, data in groups:
            timing_rows.append(timing_row(policy, group.title(), data))
            distributions.append((policy, group.title(), *data['distribution'].values()))
    lines.extend(['', 'Contract decision timing', table(
        ('Policy', 'Group', 'Contracts', 'Mean decided trick', 'Mean remaining tricks'), timing_rows),
        '', 'Decision timing by completed trick', table(
            ('Policy', 'Group', *(f'Trick {i}' for i in range(1, 8))), distributions),
        'Timing means exclude unavailable records; all seven tricks are still played.'])
    origin_rows = []
    for policy, row in details.items():
        for bid, groups in row['contract_origins'].items():
            kind, value = bid.split(':')
            for origin, counts in groups.items():
                origin_rows.append((policy, f'{value} {kind}', origin.title(), counts['contracts'],
                    counts['successful'], counts['failed'], f"{counts['success_rate']:.1%}",
                    f"{counts['mean_signed_marks']:.3f}"))
    lines.extend(['', 'Contract outcomes by bid origin', table(
        ('Policy', 'Bid', 'Origin', 'Contracts', 'Success', 'Failed', 'Success rate', 'Mean signed marks'),
        origin_rows)])
    preference_rows = []
    for policy, row in details.items():
        for bid, groups in row.get('forced_bid_preferences', {}).items():
            kind, value = bid.split(':')
            for preference in ('would_pass', 'would_bid', 'unknown'):
                counts = groups.get(preference, {'successful': 0, 'failed': 0})
                total = sum(counts.values())
                rate = f"{counts['successful'] / total:.1%}" if total else 'N/A'
                preference_rows.append((policy, f'{value} {kind}', preference.replace('_', ' ').title(),
                                        total, counts['successful'], counts['failed'], rate))
    lines.extend(['', 'Policy preference when passing is unavailable', table(
        ('Policy', 'Bid', 'Preference', 'Contracts', 'Success', 'Failed', 'Success rate'),
        preference_rows),
        'Would pass: compelled to bid. Would bid: willing bidder in a forced auction slot.',
        'Unknown: policy does not report a preference; no extra random decision is sampled.'])
    return '\n'.join(lines)


def format_performance(performance):
    """Timing metadata is deliberately separate from deterministic outcome JSON."""
    detail = ""
    if "group_statistics_seconds" in performance:
        detail = (f"\nGroup statistics: {performance['group_statistics_seconds']:.3f} s | "
                  f"Bootstrap sampling/percentiles: {performance['bootstrap_sampling_seconds']:.3f} s | "
                  f"Result aggregation: {performance['aggregation_seconds']:.3f} s\n"
                  f"Simulation throughput: {performance['games_per_second']:.1f} games/s, "
                  f"{performance['decisions_per_second']:.1f} decisions/s; "
                  f"{performance['bootstrap_replicates_per_second']:.1f} bootstrap replicates/s")
    return (f"Requested threads: {performance['requested_workers']}\n"
            f"Resolved simulation workers: {performance['workers']}\n"
            f"Simulation: {performance['simulation_seconds']:.3f} s | "
            f"Bootstrap/aggregation: {performance['bootstrap_seconds']:.3f} s | "
            f"Total evaluation: {performance['total_seconds']:.3f} s" + detail)
