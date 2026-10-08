from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass
import random
import subprocess
import sys
import unittest

from texas42 import ACTION_REGISTRY, Game, GameConfig, SelfPlayEnv
from texas42.agents import Agent, AgentPlayer, PrivilegedAgent, PrivilegedState, RandomAgent
from texas42.players import Player
from texas42.selfplay import Observation, observation_from_view


class RecordingAgent:
    def __init__(self, action=None):
        self.action = action
        self.resets = []
        self.observations = []

    def reset(self, seat):
        self.seat = seat
        self.resets.append(seat)
        self.memory = []

    def act(self, obs):
        assert obs.seat == self.seat
        self.observations.append(obs)
        self.memory.append(obs.view.hand)
        return self.action if self.action is not None else obs.legal_action_ids[0]


def choose(adapter, view, phase):
    return {'bid': adapter.choose_bid, 'trump': adapter.choose_trump,
            'play': adapter.choose_domino}[phase](view)


class AgentTests(unittest.TestCase):
    def test_protocols_and_privileged_snapshot(self):
        self.assertIsInstance(RandomAgent(), Agent)
        self.assertIsInstance(RecordingAgent(), Agent)
        class OracleStub:
            def reset(self, seat): pass
            def act_privileged(self, obs, state): return obs.legal_action_ids[0]
        self.assertIsInstance(OracleStub(), PrivilegedAgent)
        self.assertNotIsInstance(OracleStub(), Agent)
        hands = [[0], [1], [2], [3]]
        snapshot = PrivilegedState(hands)
        hands[0].append(4)
        self.assertEqual(snapshot.hands, ((0,), (1,), (2,), (3,)))
        with self.assertRaises(FrozenInstanceError):
            snapshot.hands = ()
        with self.assertRaises(ValueError):
            PrivilegedState(((0,), (0,), (), ()))

    def test_random_all_phases_seed_replay_and_reset_stream(self):
        for seed in range(8):
            env = SelfPlayEnv(seed=seed)
            obs = env.reset()
            agents = [RandomAgent(seed + seat) for seat in range(4)]
            twins = [RandomAgent(seed + seat) for seat in range(4)]
            for seat in range(4):
                agents[seat].reset(seat)
                twins[seat].reset(seat)
            phases = set()
            while obs:
                action = agents[obs.seat].act(obs)
                self.assertIs(type(action), int)
                self.assertIn(action, obs.legal_action_ids)
                self.assertEqual(action, twins[obs.seat].act(obs))
                phases.add(obs.phase)
                obs = env.step(action).observation
            self.assertEqual(phases, {'bid', 'trump', 'play'})
        obs = SelfPlayEnv(seed=4).reset()
        agent = RandomAgent(9)
        reference = random.Random(9)
        seen = set()
        for _ in range(500):
            agent.reset(obs.seat)
            action = agent.act(obs)
            self.assertEqual(action, reference.choice(obs.legal_action_ids))
            seen.add(action)
        self.assertEqual(seen, set(obs.legal_action_ids))
        with self.assertRaises(RuntimeError):
            RandomAgent().act(obs)
        agent.reset((obs.seat + 1) % 4)
        with self.assertRaises(RuntimeError):
            agent.act(obs)

    def test_adapter_observations_roundtrips_and_no_state_mutation(self):
        env = SelfPlayEnv(seed=42)
        obs = env.reset()
        while obs:
            state = env._session.hand
            view = state.view()
            before = deepcopy(state.__dict__)
            agent = RecordingAgent()
            adapter = AgentPlayer('test', agent)
            self.assertIsInstance(adapter, Player)
            adapter.reset(obs.seat)
            for action in obs.legal_action_ids:
                agent.action = action
                value = choose(adapter, view, obs.phase)
                self.assertEqual(ACTION_REGISTRY.encode(obs.phase, value), action)
                self.assertEqual(agent.observations[-1], obs)
            self.assertEqual(observation_from_view(view), obs)
            self.assertEqual(state.__dict__, before)
            obs = env.step(obs.legal_action_ids[0]).observation

    def test_forced_bid_invalid_ids_and_wrong_phase_are_rejected(self):
        env = SelfPlayEnv(seed=0)
        obs = env.reset()
        for _ in range(3):
            obs = env.step(28).observation
        self.assertTrue(obs.view.forced)
        self.assertNotIn(28, obs.legal_action_ids)
        while obs:
            state = env._session.hand
            before = deepcopy(state.__dict__)
            invalid = [True, False, float(obs.legal_action_ids[0]), -1, 58, 55, 56, 57,
                       '29', None]
            invalid += [i for i in range(58) if i not in obs.legal_action_ids]
            for action in invalid:
                agent = RecordingAgent()
                # Override instead of RecordingAgent.action: None is also invalid.
                agent.act = lambda observation, result=action: result
                adapter = AgentPlayer('bad', agent)
                adapter.reset(obs.seat)
                with self.subTest(phase=obs.phase, action=action), self.assertRaises(ValueError):
                    choose(adapter, state.view(), obs.phase)
                self.assertEqual(state.__dict__, before)
            obs = env.step(obs.legal_action_ids[0]).observation

    def test_game_matches_native_and_resets_once_per_match(self):
        native = SelfPlayEnv(episode='match', target_marks=3, seed=42)
        native.reset()
        class PairedAgent(RecordingAgent):
            def act(agent, obs):
                self.assertEqual(obs, native.observe())
                action = super(PairedAgent, agent).act(obs)
                native.step(action)
                return action
        agents = [PairedAgent() for _ in range(4)]
        game = Game([AgentPlayer(str(i), agent) for i, agent in enumerate(agents)],
                    GameConfig(seed=42, target_marks=3), write=lambda _: None)
        game.play_match()
        self.assertEqual(tuple(game.marks), native.match_marks)
        self.assertGreater(game._hand_number, 1)
        for seat, agent in enumerate(agents):
            self.assertEqual(agent.resets, [seat])
            self.assertTrue(all(obs.seat == seat for obs in agent.observations))
        agents[0].memory.clear()
        self.assertTrue(all(agent.memory for agent in agents[1:]))

    def test_standalone_hands_reset_and_random_agents_complete_match(self):
        agents = [RecordingAgent() for _ in range(4)]
        game = Game([AgentPlayer(str(i), a) for i, a in enumerate(agents)],
                    GameConfig(seed=0), write=lambda _: None)
        game.play_hand()
        game.play_hand()
        for seat, agent in enumerate(agents):
            self.assertEqual(agent.resets, [seat, seat])
        random_game = Game([AgentPlayer(str(i), RandomAgent(i)) for i in range(4)],
                           GameConfig(seed=0, target_marks=3), write=lambda _: None)
        self.assertIn(random_game.play_match(), (0, 1))

    def test_shared_agent_rejected_and_match_failure_cleans_lifecycle(self):
        shared = RecordingAgent()
        game = Game([AgentPlayer(str(i), shared) for i in range(4)], write=lambda _: None)
        with self.assertRaisesRegex(ValueError, 'separate agent'):
            game.play_match()
        agents = [RecordingAgent(55) for _ in range(4)]
        game = Game([AgentPlayer(str(i), a) for i, a in enumerate(agents)], write=lambda _: None)
        with self.assertRaises(ValueError):
            game.play_match()
        self.assertFalse(game._in_match)
        for agent in agents:
            agent.action = None
        game.play_hand()
        for seat, agent in enumerate(agents):
            self.assertEqual(agent.resets, [seat, seat])

    def test_observation_contains_only_immutable_policy_inputs(self):
        env = SelfPlayEnv(seed=13)
        obs = env.reset()
        agent = RecordingAgent()
        adapter = AgentPlayer('private', agent)
        adapter.reset(obs.seat)
        adapter.choose_bid(env._session.hand.view())
        received = agent.observations[-1]
        self.assertIsInstance(received, Observation)
        self.assertEqual(set(received.view.hand), {d.id for d in env._session.hand.hands[obs.seat]})
        def check(value):
            self.assertNotIsInstance(value, (dict, list, set))
            if is_dataclass(value):
                self.assertTrue(type(value).__dataclass_params__.frozen)
                for field in fields(value):
                    self.assertNotIn(field.name, ('hands', 'state', 'env', 'initial_deal', 'privileged'))
                    check(getattr(value, field.name))
            elif isinstance(value, tuple):
                for item in value: check(item)
        check(received)
        with self.assertRaises(FrozenInstanceError):
            received.view.hand = ()

    def test_no_mandatory_ml_or_openspiel_imports(self):
        script = '''
import sys
import texas42.agents
assert not {'numpy', 'torch', 'pyspiel', 'open_spiel'} & sys.modules.keys()
'''
        subprocess.run([sys.executable, '-c', script], check=True)


if __name__ == '__main__':
    unittest.main()
