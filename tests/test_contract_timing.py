"""Timing diagnostics observe completed tricks without changing transitions."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from texas42.agents import VARIANTS, variant_factory
from texas42.arena import contract_decided, compare_agents, generate_schedule, run_game
from texas42.episodes import EpisodeState
from texas42.models import Bid
from examples.heuristic_benchmark import breakdown, timing_summary, main as heuristic_main
from examples.policy_ablation_benchmark import main as ablation_main


class ContractTimingTests(unittest.TestCase):
    def test_explicit_threshold_boundaries(self):
        for bid, before_success, success, before_failure, failure in (
                (Bid(points=30), 29, 30, 12, 13),
                (Bid(points=37), 36, 37, 5, 6),
                (Bid(marks=1), 41, 42, 0, 1),
                (Bid(marks=5), 41, 42, 0, 1)):
            target = bid.target_points
            self.assertFalse(contract_decided(before_success, 0, target))
            self.assertTrue(contract_decided(success, 0, target))
            self.assertFalse(contract_decided(0, before_failure, target))
            self.assertTrue(contract_decided(0, failure, target))
        # An explicit cumulative sequence: first decided at three, never later.
        points = [(1, 0), (1, 12), (1, 13), (6, 13), (12, 13), (20, 13), (29, 13)]
        self.assertEqual(next(i for i, p in enumerate(points, 1) if contract_decided(*p, 30)), 3)

    def test_all_variants_match_completed_trick_snapshots(self):
        original = EpisodeState.apply
        saw_early = False
        for variant in VARIANTS:
            for episode in ('hand', 'match'):
                snapshots = []
                def observe(state, action):
                    before = len(state.hand.completed)
                    transition = original(state, action)
                    if len(state.hand.completed) > before:
                        snapshots.append((len(state.hand.completed), tuple(state.hand.points), state.hand.contract))
                    return transition
                factory = variant_factory(variant)
                schedule = generate_schedule(5, 42)
                # Establish identical execution with instrumentation wrapper absent.
                plain = run_game((factory,) * 4, schedule, episode=episode, target_marks=3)
                with patch.object(EpisodeState, 'apply', observe):
                    game = run_game((factory,) * 4, schedule, episode=episode, target_marks=3)
                self.assertEqual(game, plain)
                self.assertEqual(len(snapshots), 7 * game.num_hands)
                for index, hand in enumerate(game.hands):
                    history = snapshots[index * 7:(index + 1) * 7]
                    self.assertEqual([r[0] for r in history], list(range(1, 8)))
                    expected = None
                    for trick, points, contract in history:
                        team = contract.bidder % 2
                        target = contract.bid.target_points
                        if expected is None and (points[team] >= target or points[1-team] >= 43-target):
                            expected = trick
                    self.assertEqual(hand.contract_decided_at_trick, expected)
                    self.assertIn(expected, range(1, 8))
                    self.assertEqual(hand.decisions, 33)
                    self.assertEqual(sum(hand.result.team_points), 42)
                    _, points, contract = history[expected - 1]
                    self.assertEqual(points[contract.bidder % 2] >= contract.bid.target_points,
                                     hand.contract_success)
                    saw_early |= expected < 7
        self.assertTrue(saw_early)

    def test_reproducibility_aggregates_export_and_cli(self):
        kwargs = dict(num_deals=3, seed=42, bootstrap_samples=10)
        report = compare_agents(variant_factory('HHH'), variant_factory('HHR'), **kwargs)
        self.assertEqual(report, compare_agents(variant_factory('HHH'), variant_factory('HHR'), **kwargs))
        for left, right in zip(report.games[::2], report.games[1::2]):
            self.assertEqual(left.schedule_ids, right.schedule_ids)
        for policy, row in breakdown(report).items():
            timing = row['decision_timing']
            self.assertEqual(sum(timing['distribution'].values()), timing['contracts'])
            self.assertEqual(sum(v['contracts'] for v in timing['by_outcome'].values()), timing['contracts'])
            self.assertEqual(sum(v['contracts'] for v in timing['by_origin'].values()), timing['contracts'])
            if timing['contracts']:
                self.assertAlmostEqual(timing['mean_decided_trick'] + timing['mean_remaining_tricks'], 7)
        legacy = replace(report.games[0].hands[0], contract_decided_at_trick=None)
        self.assertIsNone(timing_summary([legacy])['mean_decided_trick'])
        self.assertEqual(timing_summary([legacy])['recorded'], 0)
        for main in (heuristic_main, ablation_main):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'report.json'
                with redirect_stdout(io.StringIO()) as out:
                    main(['--deals', '2', '--bootstrap-samples', '10', '--output', str(path)])
                data = json.loads(path.read_text())
                self.assertIn('Contract decision timing', out.getvalue())
                self.assertIn('Decision timing by completed trick', out.getvalue())
                for game in data['games']:
                    for hand in game['hands']:
                        self.assertIn(hand['contract_decided_at_trick'], range(1, 8))
                self.assertIn('decision_timing', data['breakdown']['A'])


if __name__ == '__main__':
    unittest.main()
