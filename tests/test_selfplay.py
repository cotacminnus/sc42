from __future__ import annotations

import random
import unittest
from dataclasses import FrozenInstanceError, fields

from texas42 import (
    BID_ACTIONS, PLAY_ACTIONS, TRUMP_ACTIONS, Bid, Game, GameConfig, SelfPlayEnv,
)
from texas42.players import BidView, PlayView, Player, TrumpView
from texas42.rules import is_legal_bid, legal_plays, trick_points, trick_winner


def finish_hand(env, observation, *, trump=8, bid_id=1):
    """Use a specified forced bid and trump, then deterministic legal play."""
    transitions = []
    while True:
        if observation.phase == "bid":
            action = bid_id if observation.view.forced else 0
        elif observation.phase == "trump":
            action = trump
        else:
            action = observation.legal_action_ids[0]
        transition = env.step(action)
        transitions.append(transition)
        if transition.hand_result is not None:
            return transitions
        observation = transition.observation


class SelfPlayTests(unittest.TestCase):
    def test_lifecycle_and_configuration(self):
        env = SelfPlayEnv()
        with self.assertRaises(RuntimeError):
            env.step(0)
        with self.assertRaises(RuntimeError):
            env.observe()
        for kwargs in ({"episode": "other"}, {"target_marks": 0}, {"target_marks": True}):
            with self.assertRaises(ValueError):
                SelfPlayEnv(**kwargs)
        for shaker in (-1, 4, True):
            with self.assertRaises(ValueError):
                env.reset(shaker=shaker)
        result = finish_hand(env, env.reset())[-1]
        self.assertTrue(result.terminated)
        self.assertIsNone(result.observation)
        with self.assertRaises(RuntimeError):
            env.step(0)
        with self.assertRaises(RuntimeError):
            env.observe()
        self.assertEqual(env.reset().phase, "bid")
        self.assertEqual(env.match_marks, (0, 0))

    def test_seeded_replay_and_stream_continuation(self):
        env = SelfPlayEnv(seed=21)
        first = env.reset()
        first_trace = finish_hand(env, first)
        second = env.reset()
        twin = SelfPlayEnv(seed=21)
        self.assertEqual(twin.reset(), first)
        finish_hand(twin, twin.observe())
        self.assertEqual(twin.reset(), second)
        self.assertNotEqual(first.view.hand, second.view.hand)
        self.assertEqual(env.reset(seed=21), first)
        self.assertEqual(finish_hand(env, env.observe()), first_trace)

    def test_auction_order_forced_pass_mask_and_history(self):
        env = SelfPlayEnv()
        obs = env.reset(seed=42, shaker=1)
        self.assertEqual(len(obs.action_mask), 18)
        for seat in (2, 3, 0):
            self.assertEqual(obs.seat, seat)
            self.assertIn(0, obs.legal_action_ids)
            obs = env.step(0).observation
        self.assertEqual(obs.seat, 1)
        self.assertTrue(obs.view.forced)
        self.assertNotIn(0, obs.legal_action_ids)
        self.assertEqual(obs.view.bid_history, ((2, None), (3, None), (0, None)))
        with self.assertRaises(ValueError):
            env.step(0)
        self.assertEqual(env.observe(), obs)
        obs = env.step(1).observation
        self.assertIsInstance(obs.view, TrumpView)
        self.assertEqual(obs.seat, 1)
        self.assertEqual(obs.view.bid, Bid(points=30))
        self.assertEqual(obs.view.bid_history[-1], (1, Bid(points=30)))
        obs = env.step(8).observation
        self.assertEqual(obs.phase, "play")
        self.assertEqual(obs.seat, 1)
        self.assertIsNone(obs.view.trump)

    def test_bid_masks_follow_raise_limits(self):
        env = SelfPlayEnv()
        obs = env.reset(seed=5)
        for chosen in (14, 15, 16, 17):  # 2, 3, 4, 5 marks
            expected = tuple(i for i, bid in enumerate(BID_ACTIONS)
                             if (not obs.view.forced if bid is None
                                 else is_legal_bid(bid, obs.view.current_bid)))
            self.assertEqual(obs.legal_action_ids, expected)
            obs = env.step(chosen).observation
        self.assertEqual(obs.view.bid, Bid(marks=5))

    def test_invalid_actions_are_atomic_in_every_phase(self):
        env = SelfPlayEnv()
        obs = env.reset(seed=9)
        for phase in ("bid", "trump", "play"):
            while obs.phase != phase:
                action = (1 if obs.view.forced else 0) if obs.phase == "bid" else 8
                obs = env.step(action).observation
            invalid = [-1, len(obs.action_mask), True, 1.0, None, "0"]
            invalid.extend(i for i, allowed in enumerate(obs.action_mask) if not allowed)
            for action in invalid:
                with self.assertRaises(ValueError):
                    env.step(action)
                self.assertEqual(obs, env.observe())
                self.assertEqual(env.match_marks, (0, 0))

    def test_observation_privacy_and_snapshot_immutability(self):
        env = SelfPlayEnv()
        old = env.reset(seed=3)
        hands = []
        obs = old
        for _ in range(4):
            hands.extend(obs.view.hand)
            self.assertEqual(len(obs.view.hand), 7)
            self.assertFalse(any(field.name == "hands" for field in fields(obs.view)))
            with self.assertRaises(FrozenInstanceError):
                obs.view.seat = 4
            obs = env.step(1 if obs.view.forced else 0).observation
        self.assertEqual(len(set(hands)), 28)
        self.assertEqual(old.view.bid_history, ())
        self.assertEqual(len(old.view.hand), 7)
        self.assertEqual(old.phase, "bid")

    def test_all_trumps_turns_scoring_and_masks(self):
        for trump_id in range(9):
            env = SelfPlayEnv()
            obs = env.reset(seed=trump_id)
            while obs.phase == "bid":
                obs = env.step(1 if obs.view.forced else 0).observation
            obs = env.step(trump_id).observation
            played = set()
            completed = []
            while True:
                view = obs.view
                self.assertEqual(view.trump, TRUMP_ACTIONS[trump_id])
                expected = legal_plays(view.hand, view.trick[0][1] if view.trick else None,
                                       view.trump)
                self.assertEqual({PLAY_ACTIONS[i] for i in obs.legal_action_ids}, set(expected))
                self.assertEqual(tuple(i for i, legal in enumerate(obs.action_mask) if legal),
                                 obs.legal_action_ids)
                action = obs.legal_action_ids[-1]
                domino = PLAY_ACTIONS[action]
                self.assertNotIn(domino, played)
                played.add(domino)
                trick = view.trick + ((obs.seat, domino),)
                transition = env.step(action)
                if len(trick) == 4:
                    completed.append(trick)
                    next_seat = trick_winner(trick, view.trump)
                else:
                    next_seat = (obs.seat + 1) % 4
                if transition.terminated:
                    result = transition.hand_result
                    break
                self.assertEqual(transition.rewards, (0, 0, 0, 0))
                obs = transition.observation
                self.assertEqual(obs.seat, next_seat)
                self.assertEqual(obs.view.completed_tricks, tuple(completed))
            totals = [0, 0]
            for trick in completed:
                totals[trick_winner(trick, view.trump) % 2] += trick_points(trick)
            self.assertEqual(result.team_points, tuple(totals))
            self.assertEqual(sum(totals), 42)
            self.assertEqual(len(played), 28)
            self.assertEqual(len(completed), 7)

    def test_hand_rewards_include_mark_stakes_for_both_partners(self):
        env = SelfPlayEnv()
        trace = finish_hand(env, env.reset(seed=12), bid_id=14)
        self.assertEqual(len(trace), 33)  # 4 bids + 1 trump + 28 plays
        self.assertTrue(all(step.rewards == (0, 0, 0, 0) for step in trace[:-1]))
        last = trace[-1]
        self.assertEqual(last.hand_result.marks_awarded, 2)
        for seat, reward in enumerate(last.rewards):
            self.assertEqual(reward, 2 if seat % 2 == last.hand_result.scoring_team else -2)
        self.assertEqual(sum(last.rewards), 0)

    def test_match_rotates_shaker_preserves_marks_and_pays_only_at_end(self):
        env = SelfPlayEnv(episode="match", target_marks=3)
        obs = env.reset(seed=42, shaker=2)
        marks = [0, 0]
        for hand in range(5):  # At most five one-mark hands to reach three.
            shaker = env.shaker
            trace = finish_hand(env, obs)
            last = trace[-1]
            marks[last.hand_result.scoring_team] += 1
            self.assertEqual(env.match_marks, tuple(marks))
            self.assertTrue(all(t.rewards == (0, 0, 0, 0) for t in trace[:-1]))
            if last.terminated:
                self.assertEqual(max(marks), 3)
                self.assertIsNone(last.observation)
                self.assertEqual(last.match_winner, marks.index(3))
                self.assertEqual(last.rewards, tuple(1 if s % 2 == last.match_winner else -1
                                                     for s in range(4)))
                break
            self.assertEqual(last.rewards, (0, 0, 0, 0))
            self.assertIsNone(last.match_winner)
            self.assertEqual(env.shaker, (shaker + 1) % 4)
            obs = last.observation
            self.assertEqual(obs.phase, "bid")
            self.assertEqual(obs.view.match_marks, tuple(marks))
            self.assertEqual(obs.view.target_marks, 3)
            self.assertEqual(obs.view.bid_history, ())
        else:
            self.fail("Match did not terminate")

    def test_game_and_selfplay_replay_identical_decisions(self):
        env = SelfPlayEnv()
        expected = []
        obs = env.reset(seed=123)
        rng = random.Random(42)
        while True:
            action_id = rng.choice(obs.legal_action_ids)
            actions = {"bid": BID_ACTIONS, "trump": TRUMP_ACTIONS, "play": PLAY_ACTIONS}
            expected.append((obs.view, actions[obs.phase][action_id]))
            transition = env.step(action_id)
            if transition.terminated:
                break
            obs = transition.observation
        decisions = iter(expected)
        case = self

        class ReplayPlayer(Player):
            def choose_bid(self, view):
                wanted, action = next(decisions)
                case.assertEqual(view, wanted)
                return action
            choose_trump = choose_bid
            choose_domino = choose_bid

        game = Game([ReplayPlayer(str(s)) for s in range(4)], GameConfig(seed=123),
                    write=lambda _: None)
        self.assertEqual(game.play_hand(), transition.hand_result)
        self.assertEqual(tuple(game.marks), env.match_marks)
        self.assertEqual(list(decisions), [])


if __name__ == "__main__":
    unittest.main()
