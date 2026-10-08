"""Contract-origin metadata must not infer origin from value or shaker seat."""
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import tempfile
import unittest

from texas42.actions import ACTION_REGISTRY
from texas42.arena import ArenaReport, run_game, generate_schedule
from texas42.models import Bid
from examples.heuristic_benchmark import breakdown, main as heuristic_main
from examples.policy_ablation_benchmark import main as ablation_main
from examples.benchmark_output import format_breakdown


class BidOriginTests(unittest.TestCase):
    def scenario(self, calls, mode='hand'):
        class Scripted:
            def reset(self, seat):
                self.seat = seat
            def act(self, obs):
                if obs.phase == 'bid':
                    return ACTION_REGISTRY.encode('bid', calls[len(obs.view.bid_history)])
                return obs.legal_action_ids[0]
        return run_game((lambda seat, seed: Scripted(),) * 4,
                        generate_schedule(5, 42), episode=mode, target_marks=3,
                        seating=('A', 'B', 'A', 'B'))

    def test_explicit_auction_origins_in_hands_and_matches(self):
        cases = [
            ((None, None, None, Bid(points=30)), True, 0),
            ((Bid(points=30), None, None, None), False, 1),
            ((None, None, None, Bid(points=37)), True, 0),
            ((Bid(points=30), None, None, Bid(points=31)), False, 0),
            ((Bid(points=30), Bid(points=31), None, None), False, 2),
        ]
        for mode in ('hand', 'match'):
            for calls, forced, offset in cases:
                with self.subTest(mode=mode, calls=calls):
                    game = self.scenario(calls, mode)
                    marks = [0, 0]
                    for hand in game.hands:
                        self.assertIs(hand.forced_bid, forced)
                        self.assertEqual(hand.result.bidder, (hand.shaker + offset) % 4)
                        self.assertEqual(hand.result.bid, next(b for b in reversed(calls) if b))
                        self.assertEqual(hand.decisions, 33)
                        self.assertEqual(sum(hand.result.team_points), 42)
                        marks[hand.result.scoring_team] += hand.result.marks_awarded
                        with self.assertRaises(FrozenInstanceError): hand.forced_bid = False
                    self.assertEqual(tuple(marks), game.final_marks)
                    if mode == 'match': self.assertGreaterEqual(marks[game.winner], 3)

    def test_breakdown_counts_only_final_thirties(self):
        games = tuple(self.scenario(calls) for calls in (
            (None, None, None, Bid(points=30)),
            (Bid(points=30), None, None, None),
            (None, None, None, Bid(points=37)),
            (Bid(points=30), Bid(points=31), None, None)))
        report = ArenaReport(42, 0, 10, games, (), ())
        details = breakdown(report)
        for policy, row in details.items():
            hands = [h for g in games for h in g.hands
                     if h.seating[h.result.bidder] == policy and h.result.bid.points == 30]
            self.assertEqual(sum(sum(c.values()) for c in row['thirty_point_bids'].values()), len(hands))
            for origin, counts in row['thirty_point_bids'].items():
                selected = [h for h in hands if h.forced_bid == (origin == 'forced')]
                self.assertEqual(counts['successful'], sum(h.contract_success for h in selected))
                self.assertEqual(counts['failed'], sum(not h.contract_success for h in selected))
                self.assertEqual(sum(counts.values()), len(selected))
        text = format_breakdown(details)
        self.assertIn('30-point contract outcomes', text)
        self.assertRegex(text, r'B\s+Forced\s+0\s+0\s+N/A')
        self.assertEqual([h['forced_bid'] for g in report.to_dict()['games'] for h in g['hands']],
                         [True, False, True, False])

    def test_both_clis_export_and_reproduce(self):
        for main, options in ((heuristic_main, ['--opponent', 'random']),
                              (ablation_main, ['--policy-a', 'HHH', '--policy-b', 'CHH'])):
            for mode in ('hand', 'match'):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'results.json'
                    args = options + ['--deals', '3', '--bootstrap-samples', '10',
                                      '--episode', mode, '--target-marks', '2', '--output', str(path)]
                    with redirect_stdout(io.StringIO()) as output:
                        first = main(args)
                        saved = path.read_text()
                        second = main(args)
                    self.assertEqual(first, second)
                    self.assertEqual(saved, path.read_text())
                    self.assertIn('30-point contract outcomes', output.getvalue())
                    data = json.loads(saved)
                    for game in data['games']:
                        for hand in game['hands']: self.assertIs(type(hand['forced_bid']), bool)
                    for policy, row in data['breakdown'].items():
                        total = sum(sum(c.values()) for c in row['thirty_point_bids'].values())
                        self.assertEqual(total, sum(row['contract_values'].get('points:30', {}).values()))


if __name__ == '__main__':
    unittest.main()
