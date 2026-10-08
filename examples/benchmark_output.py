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
    return '\n'.join(lines)
