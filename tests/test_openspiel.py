"""Integration tests against the optional, real OpenSpiel package."""
import importlib.util
import pickle
import random
import subprocess
import sys
import unittest

_AVAILABLE = importlib.util.find_spec("pyspiel") is not None
if _AVAILABLE:
    import numpy as np
    import pyspiel
    from open_spiel.python import rl_environment
    from texas42 import NUM_ACTIONS, SelfPlayEnv, NIL_ACTION_ID, SPLASH_ACTION_ID, PLUNGE_ACTION_ID
    from texas42.openspiel import Texas42Game


@unittest.skipUnless(_AVAILABLE, "Install tx42[openspiel] to test the OpenSpiel adapter")
class OpenSpielTests(unittest.TestCase):
    def deal(self, state, ids=range(28)):
        for action in ids:
            self.assertTrue(state.is_chance_node())
            state.apply_action(action)
        self.assertFalse(state.is_chance_node())

    def test_metadata_and_configuration(self):
        game = pyspiel.load_game("python_texas42")
        self.assertEqual(game.num_players(), 4)
        self.assertEqual(game.num_distinct_actions(), NUM_ACTIONS)
        self.assertEqual(game.get_type().information, pyspiel.GameType.Information.IMPERFECT_INFORMATION)
        self.assertEqual(game.get_type().chance_mode, pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
        self.assertEqual(game.max_game_length(), 33)
        self.assertEqual(game.max_chance_nodes_in_history(), 29)
        self.assertEqual(game.min_utility(), -5)
        self.assertEqual(game.ruleset_id, "cadets_base_v1")
        self.assertEqual(game.get_parameters()["ruleset_id"], game.ruleset_id)
        for params in ({"episode": "other"}, {"target_marks": 0}, {"shaker": 4},
                       {"ruleset_id": "cadets_nil_v1"}):
            with self.assertRaises(ValueError):
                Texas42Game(params)

    def test_chance_distribution_and_no_replacement(self):
        state = Texas42Game().new_initial_state()
        self.assertEqual(state.chance_outcomes(), [(seat, .25) for seat in range(4)])
        state.apply_action(2)
        for domino_id in range(28):
            outcomes = state.chance_outcomes()
            self.assertEqual(len(outcomes), 28 - domino_id)
            self.assertAlmostEqual(sum(prob for _, prob in outcomes), 1)
            self.assertEqual([action for action, _ in outcomes], list(range(domino_id, 28)))
            state.apply_action(domino_id)
        self.assertEqual(state.current_player(), 3)
        self.assertEqual(state.session.hand.shaker, 2)
        self.assertEqual([len(hand) for hand in state.session.hand.hands], [7] * 4)
        self.assertEqual(state.chance_outcomes(), [])

    def test_hand_and_match_replay_native_engine(self):
        for mode in ("hand", "match"):
            for seed in range(10):
                env = SelfPlayEnv(episode=mode, target_marks=3, seed=seed)
                obs = env.reset()
                game = Texas42Game({"episode": mode, "target_marks": 3, "shaker": env.shaker})
                state = game.new_initial_state()
                rng = random.Random(seed)
                actions = 0
                while True:
                    if state.is_chance_node():
                        deck = [domino.id for hand in env._session.hand.hands for domino in hand]
                        self.deal(state, deck)
                    self.assertEqual(state.current_player(), obs.seat)
                    self.assertEqual(state.legal_actions(),
                                     list(obs.legal_action_ids))
                    self.assertEqual(state.legal_actions_mask(), list(map(int, obs.action_mask)))
                    before = str(state), state.history()
                    for reserved in (NIL_ACTION_ID, SPLASH_ACTION_ID, PLUNGE_ACTION_ID):
                        with self.assertRaises(ValueError):
                            state.apply_action(reserved)
                        self.assertEqual((str(state), state.history()), before)
                    for seat in range(4):
                        if seat != obs.seat:
                            self.assertEqual(state.legal_actions(seat), [])
                    action = rng.choice(obs.legal_action_ids)
                    state.apply_action(action)
                    transition = env.step(action)
                    actions += 1
                    self.assertEqual(state.is_terminal(), transition.terminated)
                    self.assertEqual(tuple(state.session.marks), env.match_marks)
                    if transition.hand_result:
                        self.assertEqual(state.session.hand.result, transition.hand_result)
                    if transition.terminated:
                        self.assertEqual(state.returns(), list(transition.rewards))
                        self.assertEqual(state.current_player(), pyspiel.PlayerId.TERMINAL)
                        for seat in range(4):
                            self.assertEqual(state.legal_actions(seat), [])
                        break
                    self.assertEqual(state.returns(), [0] * 4)
                    obs = transition.observation
                self.assertLessEqual(actions, game.max_game_length())

    def test_hidden_hands_never_enter_player_observations(self):
        game = Texas42Game({"shaker": 0})
        first, second = game.new_initial_state(), game.new_initial_state()
        self.deal(first)
        self.deal(second, list(range(7)) + list(range(14, 21)) + list(range(7, 14)) + list(range(21, 28)))
        for method in ("information_state_string", "observation_string",
                       "information_state_tensor", "observation_tensor"):
            self.assertEqual(getattr(first, method)(0), getattr(second, method)(0))
            self.assertNotEqual(getattr(first, method)(1), getattr(second, method)(1))
        # Privacy also holds while dealing opponents' cards.
        first, second = game.new_initial_state(), game.new_initial_state()
        for i in range(7):
            first.apply_action(i)
            second.apply_action(i)
        first.apply_action(7)
        second.apply_action(14)
        self.assertEqual(first.information_state_tensor(0), second.information_state_tensor(0))

    def test_perfect_recall_retains_private_deal_order(self):
        game = Texas42Game({"shaker": 0})
        first, second = game.new_initial_state(), game.new_initial_state()
        self.deal(first)
        self.deal(second, [1, 0] + list(range(2, 28)))
        self.assertEqual(first.observation_tensor(0), second.observation_tensor(0))
        self.assertNotEqual(first.information_state_tensor(0), second.information_state_tensor(0))
        self.assertNotEqual(first.information_state_string(0), second.information_state_string(0))

    def test_observer_privacy_options_and_fixed_shapes(self):
        game = Texas42Game({"episode": "match", "target_marks": 2, "shaker": 0})
        state = game.new_initial_state()
        self.deal(state)
        for recall in (False, True):
            for privacy in (pyspiel.PrivateInfoType.NONE, pyspiel.PrivateInfoType.SINGLE_PLAYER,
                            pyspiel.PrivateInfoType.ALL_PLAYERS):
                observer = game.make_py_observer(pyspiel.IIGObservationType(
                    perfect_recall=recall, public_info=False, private_info=privacy))
                observer.set_from(state, 1)
                self.assertEqual(observer.tensor.dtype, np.float32)
                self.assertNotIn("history", observer.dict)
                if privacy == pyspiel.PrivateInfoType.NONE:
                    self.assertNotIn("hand", observer.dict)
                else:
                    self.assertEqual(observer.dict["hand"].sum(),
                                     28 if privacy == pyspiel.PrivateInfoType.ALL_PLAYERS else 7)
        obs_size = game.observation_tensor_size()
        info_size = game.information_state_tensor_size()
        observer = game.make_py_observer(pyspiel.IIGObservationType(perfect_recall=True))
        original_deal = list(state.deals[0])
        while not state.is_terminal():
            self.assertEqual(len(state.observation_tensor(0)), obs_size)
            self.assertEqual(len(state.information_state_tensor(0)), info_size)
            observer.set_from(state, 0)
            for turn, domino in enumerate(original_deal[:7]):
                self.assertEqual(observer.dict["private_deals"][0, 0, turn, domino], 1)
            state.apply_action(state.legal_actions()[0])
        self.assertEqual(len(state.information_state_tensor(0)), info_size)

    def test_clone_child_serialization_and_pickle(self):
        game = Texas42Game({"shaker": 0})
        state = game.new_initial_state()
        self.deal(state)
        original = str(state)
        child = state.child(state.legal_actions()[0])
        self.assertEqual(str(state), original)
        self.assertNotEqual(str(child), original)
        self.assertNotEqual(child.history(), state.history())
        for restored in (state.clone(), pickle.loads(pickle.dumps(state)),
                         pyspiel.deserialize_game_and_state(
                             pyspiel.serialize_game_and_state(game, state))[1]):
            self.assertEqual(str(restored), str(state))
            self.assertEqual(restored.history(), state.history())
            restored.apply_action(restored.legal_actions()[0])
            self.assertEqual(str(state), original)

    def test_invalid_action_is_atomic(self):
        state = Texas42Game({"shaker": 0}).new_initial_state()
        state.apply_action(0)
        before = str(state), state.history()
        with self.assertRaises(ValueError):
            state.apply_action(0)
        self.assertEqual((str(state), state.history()), before)
        self.deal(state, range(1, 28))
        before = str(state), state.history()
        with self.assertRaises(ValueError):
            state.apply_action(0)  # Play ID in the bidding phase.
        self.assertEqual((str(state), state.history()), before)

    def test_openspiel_simulation_and_rl_environment(self):
        for mode in ("hand", "match"):
            game = Texas42Game({"episode": mode, "target_marks": 3})
            pyspiel.random_sim_test(game, 25, True, False)
            env = rl_environment.Environment(game)
            timestep = env.reset()
            while not timestep.last():
                player = timestep.observations["current_player"]
                legal = timestep.observations["legal_actions"][player]
                timestep = env.step([legal[0]])
            self.assertEqual(timestep.rewards[0], timestep.rewards[2])
            self.assertEqual(timestep.rewards[1], -timestep.rewards[0])

    def test_core_does_not_import_openspiel(self):
        subprocess.run([sys.executable, "-c",
                        "import sys, texas42; assert 'pyspiel' not in sys.modules; "
                        "assert 'numpy' not in sys.modules"], check=True)


if __name__ == "__main__":
    unittest.main()
