"""Statistical regression against the retained original object-list algorithm."""
from dataclasses import replace
import unittest
from texas42.arena import bootstrap, bootstrap_policies, compare_agents, group_statistics
from texas42.agents import variant_factory
from texas42.statistics import percentile_interval
from examples.bootstrap_benchmark import reference_bootstrap


class BootstrapTests(unittest.TestCase):
    def test_reference(self):
        for episode in ('hand', 'match'):
            report = compare_agents(variant_factory('RRR'), variant_factory('HHH'),
                                    num_deals=7, seed=42, episode=episode,
                                    target_marks=3, bootstrap_samples=1)
            for seed in (0, 17):
                for samples in (1, 13, 100):
                    expected = tuple(reference_bootstrap(report.games, p, seed=seed, samples=samples)
                                     for p in ('A', 'B'))
                    self.assertEqual(expected, bootstrap_policies(report.games, ('A', 'B'),
                                                                  seed=seed, samples=samples))
                    for p, interval in zip(('A', 'B'), expected):
                        self.assertEqual(interval, bootstrap(report.games, p, seed=seed, samples=samples))
            if episode == 'match':
                self.assertGreater(len(set(g.num_hands for g in report.games)), 1)
                groups = group_statistics(report.games, ('A', 'B'))
                self.assertEqual(sum(r[0].mark_count for r in groups),
                                 sum(g.num_hands for g in report.games))
            # Heterogeneous seats require averaging seats, including cross-team
            # policy occurrences; policies need not be complements.
            mixed = tuple(replace(g, seating=('A', 'B', 'B', 'A')) for g in report.games)
            self.assertEqual(tuple(reference_bootstrap(mixed, p, samples=50) for p in ('A', 'B')),
                             bootstrap_policies(mixed, ('A', 'B'), samples=50))

    def test_fractional_seat_averaging(self):
        games = compare_agents(variant_factory('HHH'), variant_factory('RRR'),
                               num_deals=5, seed=9, bootstrap_samples=1).games
        games = tuple(replace(g, seating=('A', 'A', 'A', 'B')) for g in games)
        expected = tuple(reference_bootstrap(games, p, seed=51, samples=77) for p in ('A', 'B'))
        actual = bootstrap_policies(games, ('A', 'B'), seed=51, samples=77)
        for old, new in zip(expected, actual):
            for metric in ('win_rate', 'mean_signed_marks'):
                for endpoint in ('low', 'high'):
                    self.assertLessEqual(abs(getattr(getattr(old, metric), endpoint) -
                                             getattr(getattr(new, metric), endpoint)), 1e-12)

    def test_empty_unavailable_and_percentiles(self):
        self.assertIsNone(bootstrap([], 'A').win_rate)
        games = compare_agents(variant_factory('HHH'), variant_factory('CHH'),
                               num_deals=2, bootstrap_samples=1).games
        self.assertIsNone(bootstrap(games[:2], 'A').win_rate)
        self.assertEqual(reference_bootstrap(games, 'absent'), bootstrap(games, 'absent'))
        empty_hands = tuple(replace(g, hands=()) for g in games)
        interval = bootstrap(empty_hands, 'A', samples=10)
        self.assertIsNotNone(interval.win_rate)
        self.assertIsNone(interval.mean_signed_marks)
        self.assertEqual(percentile_interval([0, 10]), (.25, 9.75))
        for invalid in (0, -1, True, 2.0):
            with self.assertRaises(ValueError):
                bootstrap(games, 'A', samples=invalid)
