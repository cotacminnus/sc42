"""V1 policy regression cases use engine legality, not a second rule implementation."""
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from texas42 import Game, GameConfig, SelfPlayEnv
from texas42.actions import ACTION_REGISTRY
from texas42.agents import Agent, AgentPlayer, HeuristicAgent, RandomAgent
from texas42.arena import compare_agents, generate_schedule, run_game
from texas42.models import Bid, Domino
from texas42.players import BidView, TrumpView, PlayView
from texas42.rules import is_legal_bid, legal_plays, trick_winner
from texas42.selfplay import Observation, observation_from_view
from examples.heuristic_benchmark import heuristic_factory, random_factory, main, breakdown


def tiles(text):
    return tuple(Domino.make(int(a), int(b)) for a, b in text.split())


# All fixtures have seven unique dominoes; no competing suit dominates sixes.
HANDS = {
    'seven': '60 61 62 63 64 65 66',
    'six_count': '60 61 62 63 64 66 10',
    'five_count': '60 61 62 64 66 10 32',
    'six_plain': '60 61 62 63 65 66 10',
    'five_plain': '60 61 62 63 66 10 32',
    'four': '60 61 62 66 10 32 54',
    'weak': '60 51 42 30 21 43 65',
}


class HeuristicTests(unittest.TestCase):
    def choose(self, view):
        obs = observation_from_view(view)
        agent = HeuristicAgent()
        agent.reset(obs.seat)
        action = agent.act(obs)
        self.assertIs(type(action), int)
        self.assertIn(action, obs.legal_action_ids)
        self.assertTrue(obs.action_mask[action])
        self.assertLess(action, 55)
        return ACTION_REGISTRY[action].value

    def bid(self, name, current=None, bidder=None, forced=False):
        view = BidView(0, 0, tiles(HANDS[name]), current, bidder, forced,
                       bid_history=((1, None), (2, None), (3, None)) if forced else ())
        chosen = self.choose(view)
        if chosen is not None:
            self.assertTrue(is_legal_bid(chosen, current))
        return chosen

    def test_bidding_thresholds(self):
        expected = {'seven': Bid(marks=2), 'six_count': Bid(marks=2),
                    'five_count': Bid(points=37), 'six_plain': Bid(points=32),
                    'five_plain': Bid(points=30), 'four': None, 'weak': None}
        for name, bid in expected.items():
            with self.subTest(name=name):
                hand = tiles(HANDS[name])
                self.assertEqual(len(set(hand)), 7)
                if name != 'weak':
                    self.assertEqual(HeuristicAgent()._best_suit(hand), 6)
                self.assertEqual(self.bid(name), bid)
        self.assertEqual(self.bid('weak', forced=True), Bid(points=30))

    def test_raises_partner_and_mark_restrictions(self):
        self.assertEqual(self.bid('five_count', Bid(points=30), 1), Bid(points=31))
        self.assertIsNone(self.bid('five_count', Bid(points=38), 1))
        # V1 does not protect a partner's bid: it uses the same smallest raise.
        self.assertEqual(self.bid('five_count', Bid(points=30), 2), Bid(points=31))
        self.assertEqual(self.bid('seven', Bid(points=41), 1), Bid(marks=1))
        for marks in range(1, 5):
            self.assertEqual(self.bid('seven', Bid(marks=marks), 1), Bid(marks=marks + 1))
        self.assertIsNone(self.bid('seven', Bid(marks=5), 1))
        self.assertIsNone(self.bid('six_count', Bid(marks=2), 1))
        with self.assertRaises(ValueError):
            Bid(marks=37)

    def test_trump_selection_numbered_doubles_and_ties(self):
        for suit in range(7):
            hand = tuple(Domino.make(suit, other) for other in range(7))
            self.assertEqual(self.choose(TrumpView(0, hand, Bid(points=30))), suit)
        hand = tiles('00 11 22 33 44 55 66')
        self.assertEqual(HeuristicAgent()._best_suit(hand), 'doubles')
        # Doubles compete directly; even seven doubles never select Follow Me.
        self.assertEqual(self.choose(TrumpView(0, hand, Bid(points=30))), 'doubles')
        # Equal population, top rank, counters: enumeration order selects blanks.
        hand = tiles('60 61 20 21 30 31 54')
        self.assertEqual(self.choose(TrumpView(0, hand, Bid(points=30))), 0)
        # Equal population and top rank: doubles win on counter points.
        hand = tiles('66 55 44 60 61 32 10')
        self.assertEqual(self.choose(TrumpView(0, hand, Bid(points=30))), 'doubles')

    def play(self, hand_text, trick_text, trump, expected):
        hand = tiles(hand_text)
        # Actor is seat zero; the immediately preceding players are 1,2,3.
        dominoes = tiles(trick_text)
        trick = tuple(zip(range(4 - len(dominoes), 4), dominoes))
        legal = legal_plays(hand, dominoes[0] if dominoes else None, trump)
        view = PlayView(0, hand, legal, trick, (), 0, Bid(points=30), trump, (0, 0), (0, 0))
        chosen = self.choose(view)
        self.assertIn(chosen, legal)
        self.assertEqual(chosen, tiles(expected)[0])
        return trick, chosen

    def test_trick_priorities_and_rule_legality(self):
        cases = [
            # Lead strongest trump, then strongest nontrump when void.
            ('66 64 55', '', 6, '66'),
            ('55 43 21', '', 6, '55'),
            # Last to act: partner (seat2) wins; contribute a counter.
            ('64 60', '61 66 62', 6, '64'),
            # Partner's highest trump is safe even before the last seat plays.
            ('64 60', '66 62', 6, '64'),
            # Partner may be overtaken: conserve the counter.
            ('50 10', '63 60', 6, '10'),
            # Valuable opposing trick: smallest sufficient winning response.
            ('65 66 60', '64', 6, '65'),
            # Low-value opposing trick: discard cheap losing trump.
            ('65 66 60', '63', 6, '60'),
            # Last seat takes even a counter-free trick when it can.
            ('65 66 60', '61 62 63', 6, '65'),
            # Cannot win: cheapest legal domino.
            ('64 60', '66', 6, '60'),
            # Must follow fives, cannot use the otherwise strongest trump.
            ('51 66', '54', 6, '51'),
            # Void in fives: legally trump a valuable trick.
            ('60 21', '55', 6, '60'),
            # Follow Me: doubles lead their numbered suit and rank highest.
            ('55 51', '54', None, '51'),
            ('55 51', '50', None, '51'),
            ('55 51', '52 53 56', None, '55'),
            # Doubles trump: follow doubles even with a cheap non-double.
            ('66 22 10', '55', 'doubles', '66'),
            ('66 55 10', '', 'doubles', '66'),
        ]
        for hand, trick, trump, expected in cases:
            with self.subTest(hand=hand, trick=trick, trump=trump):
                plays, chosen = self.play(hand, trick, trump, expected)
                if expected in ('65', '60') and trick in ('64', '55'):
                    self.assertEqual(trick_winner(plays + ((0, chosen),), trump), 0)

    def test_doubles_dominate_trump_selection(self):
        hand = tiles("66 55 44 33 22 60 10")
        self.assertEqual(
            self.choose(TrumpView(0, hand, Bid(points=30))),
            "doubles",
        )

    def test_protocol_and_reset_errors(self):
        agent = HeuristicAgent()
        self.assertIsInstance(agent, Agent)
        obs = SelfPlayEnv(seed=0).reset()
        with self.assertRaises(RuntimeError): agent.act(obs)
        for seat in (-1, 4, True, 1.0):
            with self.assertRaises(ValueError): agent.reset(seat)
        agent.reset((obs.seat + 1) % 4)
        with self.assertRaises(RuntimeError): agent.act(obs)

    def test_full_arena_hands_matches_lifecycle_privacy(self):
        test = self
        class Checked:
            def __init__(self, delegate):
                self.delegate, self.resets, self.calls = delegate, [], 0
            def reset(self, seat):
                self.resets.append(seat)
                self.seat = seat
                self.delegate.reset(seat)
            def act(self, obs):
                test.assertIsInstance(obs, Observation)
                test.assertEqual(obs.seat, self.seat)
                def private(value):
                    test.assertNotIsInstance(value, (dict, list, set))
                    if is_dataclass(value):
                        test.assertTrue(type(value).__dataclass_params__.frozen)
                        for field in fields(value):
                            test.assertNotIn(field.name, ('hands', 'state', 'initial_deal', 'privileged'))
                            private(getattr(value, field.name))
                    elif isinstance(value, tuple):
                        for item in value: private(item)
                private(obs)
                self.calls += 1
                action = self.delegate.act(obs)
                test.assertIs(type(action), int)
                test.assertIn(action, obs.legal_action_ids)
                return action
        for seed in range(3):
            for episode in ('hand', 'match'):
                for factories in ((heuristic_factory,) * 4,
                                  (heuristic_factory, random_factory) * 2, (random_factory,) * 4):
                    made = []
                    def wrap(factory):
                        def create(seat, seed):
                            agent = Checked(factory(seat, seed))
                            made.append(agent)
                            return agent
                        return create
                    game = run_game(tuple(wrap(f) for f in factories), generate_schedule(5, seed),
                                    seed=seed, episode=episode, target_marks=3)
                    self.assertEqual(len({id(a) for a in made}), 4)
                    self.assertEqual([a.resets for a in made], [[s] for s in range(4)])
                    self.assertEqual(sum(a.calls for a in made), 33 * game.num_hands)
                    marks = [0, 0]
                    for hand in game.hands:
                        self.assertEqual(hand.decisions, 33)
                        self.assertEqual(sum(hand.result.team_points), 42)
                        marks[hand.result.scoring_team] += hand.result.marks_awarded
                    self.assertEqual(tuple(marks), game.final_marks)
                    if episode == 'match':
                        self.assertGreaterEqual(marks[game.winner], 3)
                        self.assertLess(marks[1 - game.winner], 3)
                        self.assertEqual(game.rewards, tuple(1 if s % 2 == game.winner else -1 for s in range(4)))
                    else:
                        self.assertEqual(game.winner, game.hands[0].result.scoring_team)

    def test_selfplay_and_game_adapter_equivalence(self):
        for episode in ('hand', 'match'):
            env = SelfPlayEnv(seed=19, episode=episode, target_marks=3)
            env.reset()
            test = self
            class Paired(HeuristicAgent):
                def __init__(self):
                    super().__init__()
                    self.resets = []
                def reset(self, seat):
                    super().reset(seat)
                    self.resets.append(seat)
                def act(self, obs):
                    test.assertEqual(obs, env.observe())
                    action = super().act(obs)
                    result = env.step(action)
                    if result.hand_result:
                        test.assertEqual(sum(result.hand_result.team_points), 42)
                    return action
            agents = [Paired() for _ in range(4)]
            game = Game([AgentPlayer(str(s), a) for s, a in enumerate(agents)],
                        GameConfig(seed=19, target_marks=3), write=lambda _: None)
            if episode == 'hand': game.play_hand()
            else: game.play_match()
            self.assertEqual(tuple(game.marks), env.match_marks)
            self.assertEqual([a.resets for a in agents], [[s] for s in range(4)])
            with self.assertRaises(RuntimeError): env.observe()

    def test_benchmark_reproducibility_swaps_export_and_factories(self):
        for opponent in (random_factory, heuristic_factory):
            kwargs = dict(num_deals=3, seed=42, bootstrap_seed=12, bootstrap_samples=20)
            report = compare_agents(heuristic_factory, opponent, **kwargs)
            self.assertEqual(report, compare_agents(heuristic_factory, opponent, **kwargs))
            for left, right in zip(report.games[::2], report.games[1::2]):
                self.assertEqual(left.schedule_ids, right.schedule_ids)
                self.assertEqual(left.group_id, right.group_id)
                self.assertEqual(left.seating, ('A', 'B', 'A', 'B'))
                self.assertEqual(right.seating, ('B', 'A', 'B', 'A'))
                if opponent is heuristic_factory:
                    self.assertEqual(left.hands[0].result, right.hands[0].result)
            for metric in report.metrics:
                expected = sum(g.seating[g.winner] == metric.policy for g in report.games) / len(report.games)
                self.assertEqual(metric.win_rate, expected)
            self.assertEqual(sum(v['successful'] + v['failed'] for v in breakdown(report).values()), 6)
        for factory in (heuristic_factory, random_factory):
            agents = [factory(0, 42) for _ in range(4)]
            self.assertEqual(len({id(a) for a in agents}), 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            with redirect_stdout(io.StringIO()):
                report = main(['--deals', '2', '--bootstrap-samples', '10', '--output', str(output)])
            data = json.loads(output.read_text())
            self.assertEqual(data['games_played'], 4)
            self.assertEqual(data['metrics'], list(report.to_dict()['metrics']))
            self.assertIn('breakdown', data)
            self.assertIn('intervals', data)
        subprocess.run([sys.executable, '-c', '''
import sys
from examples.heuristic_benchmark import main
main(['--opponent', 'heuristic', '--deals', '1', '--bootstrap-samples', '2'])
assert not {'numpy', 'torch', 'pyspiel', 'open_spiel'} & sys.modules.keys()
'''], check=True, capture_output=True)

    def test_match_benchmark_cli_reproducibility(self):
        for opponent in ('random', 'heuristic'):
            args = ['--opponent', opponent, '--episode', 'match', '--target-marks', '3',
                    '--deals', '2', '--seed', '7', '--bootstrap-seed', '8',
                    '--bootstrap-samples', '10']
            with redirect_stdout(io.StringIO()) as output:
                first = main(args)
                second = main(args)
            self.assertEqual(first, second)
            self.assertIn('Average match length:', output.getvalue())
            for left, right in zip(first.games[::2], first.games[1::2]):
                self.assertEqual(left.schedule_ids, right.schedule_ids)


if __name__ == '__main__':
    unittest.main()
