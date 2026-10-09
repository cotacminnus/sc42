"""Dependency-free cluster bootstrap of precomputed scalar contributions.

Counts are pooled within a replicate, never averaged across clusters. Callers
supply their own seeded RNG so domain-separated experiment streams stay intact.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroupStatistics:
    win_sum: float = 0.0
    win_count: int = 0
    mark_sum: float = 0.0
    mark_count: int = 0


def percentile_interval(values):
    """95% percentile endpoints with linear interpolation at (n - 1) * p."""
    if not values:
        return None
    values = sorted(values)
    def quantile(p):
        index = (len(values) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)
    return quantile(.025), quantile(.975)


def resample_groups(groups, rng, samples):
    """Return paired scalar samples for each column of group contributions.

    Each row is one independent cluster; each column is a reporting policy.
    A policy with an undefined replicate has unavailable intervals. Missing hand
    contributions make only its mark interval unavailable.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError('samples must be a positive integer')
    if len(groups) < 2:
        return None
    width = len(groups[0])
    if any(len(row) != width for row in groups):
        raise ValueError('Group statistics must have equal column counts')
    wins = [[] for _ in range(width)]
    marks = [[] for _ in range(width)]
    unavailable = [False] * width
    missing_marks = [False] * width
    choose = rng.choice
    for _ in range(samples):
        ws, wc, ms, mc = [0.0] * width, [0] * width, [0.0] * width, [0] * width
        for _ in groups:
            for i, contribution in enumerate(choose(groups)):
                ws[i] += contribution.win_sum
                wc[i] += contribution.win_count
                ms[i] += contribution.mark_sum
                mc[i] += contribution.mark_count
        for i in range(width):
            if wc[i] == 0:
                unavailable[i] = True
            else:
                wins[i].append(ws[i] / wc[i])
            if mc[i] == 0:
                missing_marks[i] = True
            else:
                marks[i].append(ms[i] / mc[i])
    return tuple((None if unavailable[i] else wins[i],
                  None if unavailable[i] or missing_marks[i] else marks[i])
                 for i in range(width))
