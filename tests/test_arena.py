"""Arena invariants, including direct-engine and independently calculated scores."""
from dataclasses import FrozenInstanceError, replace
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from texas42.actions import PLAY_ACTIONS
from texas42.agents import RandomAgent
from texas42.arena import (AgentExecutionError, Deal, bootstrap, compare_agents,
                           generate_schedule, policy_metrics, run_game)
from texas42.episodes import EpisodeState
from texas42.selfplay import observation_from_view


def random_factory(seat, seed):
    return RandomAgent(seed)


class FirstAgent:
    def reset(self, seat):
        self.seat = seat

    def act(self, obs):
        assert self.seat == obs.seat
        return obs.legal_action_ids[0]


def first_factory(seat, seed):
    return FirstAgent()


class ArenaTests(unittest.TestCase):
    def test_schedule_validation_and_immutability(self):
        deals = generate_schedule(8, 123, shaker=2)
        self.assertEqual(deals, generate_schedule(8, 123, shaker=2))
        self.assertNotEqual(deals, generate_schedule(8, 124, shaker=2))
        for i, deal in enumerate(deals):
            self.assertEqual(sorted(v for h in deal.hands for v in h), list(range(28)))
            self.assertEqual(deal.shaker, (i + 2) % 4)
            self.assertEqual(deal.deal_id, Deal(deal.hands, deal.shaker).deal_id)
        lists = [list(h) for h in deals[0].hands]
        copied = Deal(lists, 2)
        lists[0][0] = 100
        self.assertEqual(copied, deals[0])
        with self.assertRaises(FrozenInstanceError):
            copied.shaker = 1
        for hands, shaker in ((lists, 2), (deals[0].hands, True),
                               (deals[0].hands[:3], 0), ((tuple(range(7)),) * 4, 0)):
            with self.assertRaises(ValueError):
                Deal(hands, shaker)
        floats = [list(h) for h in deals[0].hands]
        floats[0][0] = float(floats[0][0])
        with self.assertRaises(ValueError):
            Deal(floats, 0)

    def test_hand_matches_direct_episode_and_no_input_mutation(self):
        schedule = generate_schedule(1, 45, shaker=3)
        original = repr(schedule)
        result = run_game((first_factory,) * 4, schedule)
        state = EpisodeState('hand', 7, 3)
        state.deal([[PLAY_ACTIONS[i] for i in h] for h in schedule[0].hands])
        count = 0
        while not state.terminated:
            obs = observation_from_view(state.hand.view())
            transition = state.apply(obs.legal_action_ids[0])
            count += 1
        self.assertEqual(result.hands[0].result, state.hand.result)
        self.assertEqual(result.final_marks, tuple(state.marks))
        self.assertEqual(result.rewards, transition.rewards)
        self.assertEqual(result.hands[0].decisions, count)
        self.assertEqual(count, 33)
        self.assertEqual(sum(result.hands[0].result.team_points), 42)
        self.assertEqual(result.hands[0].bid_counts[3], 1)  # forced shaker bid
        self.assertEqual(repr(schedule), original)

    def test_match_lifecycle_privacy_and_independent_memory(self):
        agents = []
        class Recorder(FirstAgent):
            def __init__(self):
                self.resets = []
                self.observations = []
            def reset(self, seat):
                super().reset(seat)
                self.resets.append(seat)
            def act(self, obs):
                self.observations.append((obs, repr(obs)))
                self.assert_private(obs)
                return super().act(obs)
            def assert_private(self, obs):
                assert not hasattr(obs, 'state') and not hasattr(obs.view, 'hands')
                assert not hasattr(obs, 'initial_deal')
                assert len(obs.view.hand) <= 7
                assert type(obs.view.hand) is tuple
        def factory(seat, seed):
            agent = Recorder()
            agents.append(agent)
            return agent
        schedule = generate_schedule(9, 91)
        game = run_game((factory,) * 4, schedule, episode='match', target_marks=5)
        self.assertGreaterEqual(game.num_hands, 5)
        self.assertLessEqual(game.num_hands, 9)
        self.assertEqual(len({id(a) for a in agents}), 4)
        for seat, agent in enumerate(agents):
            self.assertEqual(agent.resets, [seat])
            self.assertTrue(agent.observations)
            self.assertTrue(all(obs.seat == seat and repr(obs) == snapshot
                                for obs, snapshot in agent.observations))
            with self.assertRaises(FrozenInstanceError):
                agent.observations[0][0].seat = 9
        self.assertEqual([h.shaker for h in game.hands], [i % 4 for i in range(game.num_hands)])
        self.assertGreaterEqual(game.final_marks[game.winner], 5)
        self.assertLess(game.final_marks[1 - game.winner], 5)
        self.assertTrue(all(h.rewards == (0, 0, 0, 0) for h in game.hands[:-1]))
        self.assertEqual(game.rewards, tuple(1 if s % 2 == game.winner else -1 for s in range(4)))
        # Each hand is exactly reproducible with the same engine and persistent marks.
        direct = EpisodeState('match', 5, 0)
        for deal, hand in zip(schedule, game.hands):
            direct.deal([[PLAY_ACTIONS[i] for i in h] for h in deal.hands])
            while direct.hand.result is None:
                direct.apply(direct.hand.legal_action_ids()[0])
            self.assertEqual(direct.hand.result, hand.result)
        self.assertTrue(direct.terminated)
        self.assertEqual(tuple(direct.marks), game.final_marks)

    def test_invalid_actions_fail_before_transition_with_context(self):
        schedule = generate_schedule(1, 1)
        for invalid in (True, False, 28.0, '28', None, -1, 58, 55, 56, 57, 0, 46):
            class Bad(FirstAgent):
                def act(self, obs):
                    return invalid
            with patch.object(EpisodeState, 'apply') as apply:
                with self.assertRaisesRegex(AgentExecutionError, 'game=bad hand=0 seat=1 phase=bid') as error:
                    run_game((lambda seat, seed: Bad(),) * 4, schedule, game_id='bad')
                self.assertIsInstance(error.exception.__cause__, ValueError)
                apply.assert_not_called()
        class Explodes(FirstAgent):
            def act(self, obs):
                raise LookupError('broken policy')
        with self.assertRaises(AgentExecutionError) as error:
            run_game((lambda seat, seed: Explodes(),) * 4, schedule)
        self.assertIsInstance(error.exception.__cause__, LookupError)
        shared = FirstAgent()
        with self.assertRaisesRegex(AgentExecutionError, 'phase=factory'):
            run_game((lambda seat, seed: shared,) * 4, schedule)

    def test_reproducible_duplicates_and_policy_accounting(self):
        kwargs = dict(num_deals=8, seed=13, bootstrap_seed=14, bootstrap_samples=60)
        report = compare_agents(random_factory, first_factory, **kwargs)
        self.assertEqual(report, compare_agents(random_factory, first_factory, **kwargs))
        self.assertEqual(report.games_played, 16)
        for left, right in zip(report.games[::2], report.games[1::2]):
            self.assertEqual(left.group_id, right.group_id)
            self.assertEqual(left.schedule_ids, right.schedule_ids)
            self.assertEqual(left.agent_seeds, right.agent_seeds)
            self.assertEqual(left.hands[0].shaker, right.hands[0].shaker)
            self.assertEqual(left.seating, ('A', 'B', 'A', 'B'))
            self.assertEqual(right.seating, ('B', 'A', 'B', 'A'))
        a, b = report.metrics
        expected_wins = sum(g.seating[g.winner] == 'A' for g in report.games) / 16
        expected_marks = sum(h.result.marks_awarded *
                             (1 if g.seating[h.result.scoring_team] == 'A' else -1)
                             for g in report.games for h in g.hands) / 16
        self.assertEqual(a.win_rate, expected_wins)
        self.assertEqual(a.mean_signed_marks, expected_marks)
        self.assertAlmostEqual(a.win_rate + b.win_rate, 1)
        self.assertEqual(a.mean_signed_marks, -b.mean_signed_marks)
        self.assertIsNone(report.average_match_length)
        json.dumps(report.to_dict())
        self.assertIn('95% CI', report.summary())

    def test_swapped_matches_use_same_full_schedule_despite_length(self):
        report = compare_agents(random_factory, first_factory, num_deals=6, seed=2,
                                episode='match', target_marks=5, bootstrap_samples=30)
        different = False
        for left, right in zip(report.games[::2], report.games[1::2]):
            self.assertEqual(left.schedule_ids, right.schedule_ids)
            self.assertEqual(len(left.schedule_ids), 9)
            self.assertEqual(tuple(h.deal_id for h in left.hands), left.schedule_ids[:left.num_hands])
            self.assertEqual(tuple(h.deal_id for h in right.hands), right.schedule_ids[:right.num_hands])
            different |= left.num_hands != right.num_hands
        self.assertTrue(different)
        self.assertEqual(report.average_match_length, sum(g.num_hands for g in report.games) / 12)
        with self.assertRaises(ValueError):
            run_game((first_factory,) * 4, generate_schedule(1, 1), episode='match')
        schedule = list(generate_schedule(3, 1))
        schedule[1] = replace(schedule[1], shaker=3)
        with self.assertRaises(ValueError):
            run_game((first_factory,) * 4, schedule, episode='match', target_marks=2)

    def test_success_failure_multimarks_and_undefined_metrics(self):
        # Actual full games exercise all-mark stakes and failures, no mock scoring.
        class High(FirstAgent):
            def act(self, obs):
                return obs.legal_action_ids[-1] if obs.phase == 'bid' else super().act(obs)
        high = lambda seat, seed: High()
        games = [run_game((high,) * 4, generate_schedule(1, i)) for i in range(10)]
        self.assertTrue(all(g.hands[0].result.marks_awarded == 5 for g in games))
        self.assertTrue(any(not g.hands[0].contract_success for g in games))
        normal = [run_game((first_factory,) * 4, generate_schedule(1, i)) for i in range(30)]
        self.assertTrue(any(g.hands[0].contract_success for g in normal))
        never = policy_metrics(normal, 'seat1')
        self.assertEqual(never.contracts_declared, 0)
        self.assertIsNone(never.contract_success_rate)
        self.assertIsNone(never.mean_contract_value)
        self.assertEqual(never.bidding_frequency, 0)
        absent = policy_metrics([], 'A')
        self.assertIsNone(absent.win_rate)
        self.assertIsNone(absent.mean_signed_marks)
        declared = policy_metrics(games, 'seat0')
        self.assertEqual(declared.mean_contract_value, 5)
        self.assertEqual(declared.bidding_frequency, 1)

    def test_bootstrap_resamples_whole_groups(self):
        report = compare_agents(random_factory, first_factory, num_deals=4, bootstrap_samples=40)
        first = bootstrap(report.games, 'A', seed=9, samples=50)
        self.assertEqual(first, bootstrap(report.games, 'A', seed=9, samples=50))
        self.assertIsNone(bootstrap(report.games[:2], 'A').win_rate)
        self.assertIsNone(bootstrap([], 'A').win_rate)
        self.assertIsNone(bootstrap(report.games, 'missing', samples=5).win_rate)
        # Cached rows retain each complete two-game cluster. The repeated
        # statistic no longer calls policy_metrics on allocated game lists.
        from texas42.arena import group_statistics
        groups = group_statistics(report.games, ('A', 'B'))
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(row[0].win_count == row[1].win_count == 2 for row in groups))
        with patch('texas42.arena.policy_metrics', side_effect=AssertionError('recomputed')):
            bootstrap(report.games, 'A', samples=12)

    def test_explicit_seat_rotations_and_heterogeneous_partnerships(self):
        factories = (first_factory, random_factory, first_factory, random_factory)
        labels = ('one', 'two', 'three', 'four')
        schedule = generate_schedule(1, 8)
        for rotation in range(4):
            assigned = labels[rotation:] + labels[:rotation]
            fs = factories[rotation:] + factories[:rotation]
            game = run_game(fs, schedule, seating=assigned)
            self.assertEqual(game.seating, assigned)
            for seat, policy in enumerate(assigned):
                metrics = policy_metrics([game], policy)
                self.assertEqual(metrics.win_rate, float(seat % 2 == game.winner))
        # A policy on both sides is explicitly seat-averaged, never double-counted.
        game = run_game(factories, schedule, seating=('same',) * 4)
        self.assertEqual(policy_metrics([game], 'same').win_rate, .5)
        self.assertEqual(policy_metrics([game], 'same').mean_signed_marks, 0)

    def test_optional_dependencies_not_imported(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; import texas42.arena; '
                        'assert not {"numpy", "torch", "pyspiel", "open_spiel"}.intersection(sys.modules)'],
                       check=True)


if __name__ == '__main__':
    unittest.main()
