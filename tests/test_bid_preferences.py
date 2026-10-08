"""Policy willingness is separate from the engine's no-pass auction rule."""
from dataclasses import replace
import unittest

from examples.benchmark_output import format_breakdown
from examples.heuristic_benchmark import breakdown
from texas42.actions import ACTION_REGISTRY
from texas42.agents import HeuristicAgent, RandomAgent
from texas42.agents.conservative_random import ConservativeRandomAgent
from texas42.agents.hybrid import HybridAgent
from texas42.arena import ArenaReport, Deal, generate_schedule, run_game
from texas42.models import Bid, Domino
from texas42.players import BidView
from texas42.selfplay import observation_from_view


def tiles(text):
    return tuple(Domino.make(int(a), int(b)) for a, b in text.split())


class BidPreferenceTests(unittest.TestCase):
    def test_same_forced_thirty_different_preferences(self):
        games = []
        for text, preference in (
            ('60 51 42 30 21 43 65', True),
            ('60 61 62 63 66 10 32', False),
            ('60 61 62 64 66 10 32', False),
        ):
            hand = tiles(text)
            ids = tuple(ACTION_REGISTRY.encode('play', d) for d in hand)
            rest = [i for i in range(28) if i not in ids]
            deal = Deal((ids, *(tuple(rest[i:i+7]) for i in (0, 7, 14))), 0)
            factories = (lambda seat, seed: HeuristicAgent(),) + (
                lambda seat, seed: ConservativeRandomAgent(seed),) * 3
            game = run_game(factories, (deal,), seating=('A', 'B', 'A', 'B'))
            result = game.hands[0]
            self.assertTrue(result.forced_bid)
            self.assertIs(result.would_pass_if_allowed, preference)
            self.assertEqual(result.result.bidder, 0)
            self.assertEqual(result.decisions, 33)
            self.assertEqual(sum(result.result.team_points), 42)
            if len(games) < 2:
                self.assertEqual(result.result.bid, Bid(points=30))
            else:
                self.assertEqual(result.result.bid, Bid(points=37))
            games.append(game)
        report = ArenaReport(0, 0, 10, tuple(games), (), ())
        details = breakdown(report)
        counts = details['A']['forced_bid_preferences']['points:30']
        self.assertEqual(sum(counts['would_pass'].values()), 1)
        self.assertEqual(sum(counts['would_bid'].values()), 1)
        output = format_breakdown(details)
        for label in ('Would pass', 'Would bid', 'Unknown', 'N/A'):
            self.assertIn(label.lower(), output.lower())
        self.assertIs(report.to_dict()['games'][0]['hands'][0]['would_pass_if_allowed'], True)

    def test_diagnostic_is_pure_and_hybrid_delegates(self):
        for text, preference in (('60 51 42 30 21 43 65', True),
                                 ('60 61 62 63 66 10 32', False)):
            obs = observation_from_view(BidView(0, 0, tiles(text), None, None, True,
                                       bid_history=((1, None), (2, None), (3, None))))
            for strategy in ('heuristic', 'conservative', 'random'):
                subject = HybridAgent(bidding=strategy, seed=42)
                control = HybridAgent(bidding=strategy, seed=42)
                subject.reset(0)
                control.reset(0)
                expected = preference if strategy == 'heuristic' else True if strategy == 'conservative' else None
                for _ in range(3):
                    self.assertIs(subject.would_pass_if_allowed(obs), expected)
                    self.assertEqual(subject.act(obs), control.act(obs))
                self.assertNotIn(ACTION_REGISTRY.encode('bid', None), obs.legal_action_ids)

    def test_unknown_and_match_reset(self):
        class Uninstrumented:
            def __init__(self, seed): self.delegate = ConservativeRandomAgent(seed)
            def reset(self, seat): self.delegate.reset(seat)
            def act(self, obs): return self.delegate.act(obs)
        schedule = generate_schedule(5, 42)
        known = run_game((lambda seat, seed: ConservativeRandomAgent(seed),) * 4,
                         schedule, episode='match', target_marks=3)
        unknown = run_game((lambda seat, seed: Uninstrumented(seed),) * 4,
                           schedule, episode='match', target_marks=3)
        self.assertTrue(all(h.would_pass_if_allowed is True for h in known.hands))
        self.assertTrue(all(h.would_pass_if_allowed is None for h in unknown.hands))
        self.assertEqual(replace(known, hands=tuple(replace(h, would_pass_if_allowed=None)
                                                   for h in known.hands)), unknown)
        self.assertFalse(hasattr(RandomAgent(), 'would_pass_if_allowed'))
