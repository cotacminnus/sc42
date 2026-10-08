"""Offline input compatibility and separation of policy inputs from labels."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import random
import unittest

from texas42 import (
    Contract, ContractDeclaration, ContractKind, Phase,
    PublicEvent, PublicHistory, SelfPlayEnv, TrainingCollector,
    TrainingBidView, TrainingTrumpView, TrainingPlayView,
    encode, loads_observation, observation_from_dict, observation_from_record,
    public_history, training_metadata, training_view_from_dict,
)
from texas42.models import double_six_set
from texas42.state import HandState
from texas42.training import training_view


def wire(value):
    return json.loads(json.dumps(value))


def rotate(view, offset):
    seat = lambda s: (s + offset) % 4
    teams = lambda pair: pair[::-1] if offset % 2 else pair
    pairs = lambda items: tuple((seat(s), a) for s, a in items)
    fields = dict(seat=seat(view.seat), match_marks=teams(view.match_marks),
                  bid_history=pairs(view.bid_history),
                  contract=replace(view.contract, bidder=seat(view.contract.bidder))
                  if view.contract else None)
    if isinstance(view, TrainingBidView):
        fields.update(shaker=seat(view.shaker),
                      current_bidder=seat(view.current_bidder)
                      if view.current_bidder is not None else None)
    elif isinstance(view, TrainingPlayView):
        fields.update(bidder=seat(view.bidder), team_points=teams(view.team_points),
                      trick=pairs(view.trick),
                      completed_tricks=tuple(pairs(t) for t in view.completed_tricks))
    return replace(view, **fields)


class TrainingDataTests(unittest.TestCase):
    def test_json_roundtrip_every_phase_trump_and_bid_type(self):
        phases, trump_values, bid_types = set(), set(), set()
        for seed in range(18):
            records = []
            env = SelfPlayEnv(seed=seed, collector=TrainingCollector(records.append))
            rng = random.Random(seed)
            obs = env.reset()
            while obs:
                original = obs
                if obs.phase == Phase.TRUMP:
                    action = 46 + seed % 9
                elif obs.phase == Phase.BID:
                    action = obs.legal_action_ids[-1 if seed % 2 else 0]
                else:
                    action = rng.choice(obs.legal_action_ids)
                obs = env.step(action).observation
                record = wire(records[-1])
                before = deepcopy(record)
                loaded = loads_observation(json.dumps(record))
                self.assertEqual(loaded, original)
                self.assertIs(type(loaded.view), type(original.view))
                self.assertIsInstance(loaded.phase, Phase)
                self.assertEqual(encode(loaded), encode(original))
                self.assertEqual(len(encode(loaded)), 387)
                self.assertEqual(record, before)
                self.assertEqual(PublicHistory.from_dict(record['public_history']),
                                 public_history(original.view))
                self.assertNotIn('privileged', record)
                phases.add(loaded.phase)
                if loaded.view.contract:
                    self.assertIsInstance(loaded.view.contract, Contract)
                    self.assertIsInstance(loaded.view.contract.kind, ContractKind)
                    bid_types.add(loaded.view.contract.bid.points is None)
                if isinstance(loaded.view, TrainingPlayView):
                    trump_values.add(loaded.view.trump)
        self.assertEqual(phases, set(Phase))
        self.assertEqual(trump_values, set(range(7)) | {'doubles', None})
        self.assertEqual(bid_types, {True, False})

    def test_declaration_metadata_roundtrip_does_not_enable_actions(self):
        obs = SelfPlayEnv(seed=0).reset()
        declaration = ContractDeclaration('nil')
        view = replace(obs.view, bid_history=((0, declaration),), current_bid=declaration)
        restored = training_view_from_dict('bid', wire(asdict(view)))
        self.assertEqual(restored, view)
        self.assertIsInstance(restored.current_bid, ContractDeclaration)
        self.assertFalse(any(obs.action_mask[55:]))

    def test_rotated_observations_have_identical_features_and_relative_history(self):
        env = SelfPlayEnv(seed=17, episode='match', target_marks=3)
        obs = env.reset()
        while obs:
            history = public_history(obs.view)
            for offset in range(4):
                rotated = replace(obs, seat=(obs.seat + offset) % 4,
                                  view=rotate(obs.view, offset))
                loaded = observation_from_dict(wire(asdict(rotated)))
                self.assertEqual(encode(loaded), encode(obs))
                self.assertEqual(public_history(loaded.view).relative_events(loaded.seat),
                                 history.relative_events(obs.seat))
            obs = env.step(obs.legal_action_ids[0]).observation

    def test_privileged_match_collection_and_public_replay(self):
        records = []
        env = SelfPlayEnv(episode='match', target_marks=3, seed=42,
                          collector=TrainingCollector(records.append, include_initial_deal=True))
        obs = env.reset()
        replay = None
        events = []
        hand_count = 0
        initial = None
        while obs:
            if replay is None or replay.result is not None:
                hand_count += 1
                initial = tuple(tuple(sorted(d.id for d in h)) for h in env._session.hand.hands)
                replay = HandState([[double_six_set()[i] for i in hand] for hand in initial],
                                   env.shaker, env.match_marks, env.target_marks)
                events = []
            expected_obs = obs
            action = obs.legal_action_ids[0]
            obs = env.step(action).observation
            record = records[-1]
            self.assertEqual(record['hand'], hand_count)
            self.assertEqual(record['privileged']['initial_deal'], initial)
            self.assertEqual(sorted(i for h in initial for i in h), list(range(28)))
            history = PublicHistory.from_dict(wire(record['public_history']))
            self.assertEqual(history.events, tuple(events))
            self.assertEqual(history, public_history(training_view(replay.view())))
            self.assertEqual(encode(observation_from_record(wire(record))), encode(expected_obs))
            self.assertEqual(set(replay.hands[expected_obs.seat]),
                             {double_six_set()[i] for i in expected_obs.view.hand})
            events.append(PublicEvent(expected_obs.phase, expected_obs.seat, action))
            replay.apply(action)
            if record['hand_result']:
                self.assertEqual(len(events), 33)
                self.assertEqual(asdict(replay.result), record['hand_result'])
        self.assertGreater(hand_count, 1)
        self.assertTrue(records[-1]['terminated'])
        # A reset starts a new episode and captures a new deal, never stale labels.
        first_deal = records[0]['privileged']['initial_deal']
        obs = env.reset(seed=99)
        env.step(obs.legal_action_ids[0])
        self.assertEqual((records[-1]['episode'], records[-1]['hand']), (2, 1))
        self.assertNotEqual(records[-1]['privileged']['initial_deal'], first_deal)

    def test_hidden_hands_do_not_change_policy_inputs(self):
        records = []
        private = SelfPlayEnv(seed=7, collector=TrainingCollector(records.append, include_initial_deal=True))
        ordinary = SelfPlayEnv(seed=7)
        obs = private.reset()
        self.assertEqual(obs, ordinary.reset())
        for _ in range(12):
            original = private.observe()
            state = private._session.hand
            hidden = [s for s in range(4) if s != original.seat and state.hands[s]]
            a, b = hidden[:2]
            state.hands[a][0], state.hands[b][0] = state.hands[b][0], state.hands[a][0]
            self.assertEqual(private.observe(), original)
            self.assertEqual(encode(private.observe()), encode(original))
            self.assertEqual(public_history(private.observe().view), public_history(original.view))
            state.hands[a][0], state.hands[b][0] = state.hands[b][0], state.hands[a][0]
            action = original.legal_action_ids[0]
            result = private.step(action)
            self.assertEqual(result, ordinary.step(action))
            obs_dict = records[-1]['observation']
            self.assertNotIn('privileged', obs_dict)
            self.assertNotIn('initial_deal', obs_dict['view'])
            self.assertFalse(hasattr(original, 'initial_deal'))
            # Changing detached labels cannot alter existing/future observations.
            records[-1]['privileged']['initial_deal'] = ()
            self.assertEqual(private.observe(), ordinary.observe())
        count = len(records)
        with self.assertRaises(ValueError):
            private.step(55)
        self.assertEqual(len(records), count)

    def test_incompatible_versions_and_invalid_masks_rejected(self):
        records = []
        env = SelfPlayEnv(seed=1, collector=TrainingCollector(records.append))
        obs = env.reset()
        env.step(obs.legal_action_ids[0])
        record = wire(records[0])
        old_schema5 = deepcopy(record)
        del old_schema5['public_history'], old_schema5['encoder_layout_version']
        self.assertEqual(observation_from_record(old_schema5), obs)
        for key in ('schema_version', 'feature_schema_version', 'encoder_layout_version',
                    'action_space_version'):
            bad = deepcopy(record)
            bad[key] = 999
            with self.subTest(key=key), self.assertRaises(ValueError):
                observation_from_record(bad)
        for key in ('feature_schema_version', 'action_space_version', 'num_actions',
                    'encoder_layout_version'):
            bad = deepcopy(record)
            bad['observation'][key] = 999
            with self.subTest(observation_key=key), self.assertRaises(ValueError):
                observation_from_record(bad)
        for path in ((), ('observation',)):
            bad = deepcopy(record)
            target = bad if not path else bad[path[0]]
            target['ruleset_id'] = 'unknown'
            with self.assertRaises(ValueError):
                observation_from_record(bad)
        bad = deepcopy(record)
        bad['public_history']['version'] = 999
        with self.assertRaises(ValueError):
            observation_from_record(bad)
        bad = deepcopy(record)
        bad['privileged'] = {'version': 999, 'initial_deal': []}
        with self.assertRaises(ValueError):
            observation_from_record(bad)
        bad = deepcopy(record)
        bad['observation']['action_mask'][55] = True
        with self.assertRaises(ValueError):
            observation_from_record(bad)
        bad = deepcopy(record)
        bad['observation']['phase'] = 'play'
        with self.assertRaises(ValueError):
            observation_from_record(bad)
        self.assertEqual(training_metadata()['public_history_version'], 1)

    def test_collector_requires_explicit_privileged_opt_in(self):
        with self.assertRaises(ValueError):
            TrainingCollector(lambda _: None, include_initial_deal='false')
        with self.assertRaises(TypeError):
            TrainingCollector(None)


if __name__ == '__main__':
    unittest.main()
