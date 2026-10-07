"""Persistence and extension guarantees for the shared player vocabulary."""
import unittest

from texas42 import (
    ACTIONS, ACTION_REGISTRY, BID_ACTIONS, PLAY_ACTIONS, TRUMP_ACTIONS,
    NIL_ACTION_ID, SPLASH_ACTION_ID, PLUNGE_ACTION_ID, NUM_ACTIONS,
    ActionSpec, ContractAction, SelfPlayEnv, legacy_action_id, upgrade_selfplay_record,
)


class ActionTests(unittest.TestCase):
    def test_exact_ids_and_round_trip_decoding(self):
        for phase, values, offset in (("play", PLAY_ACTIONS, 0), ("bid", BID_ACTIONS, 28),
                                      ("trump", TRUMP_ACTIONS, 46)):
            for local, value in enumerate(values):
                with self.subTest(phase=phase, local=local):
                    action_id = offset + local
                    self.assertEqual(ACTIONS[action_id].phase, phase)
                    self.assertEqual(ACTIONS[action_id].value, value)
                    self.assertEqual(ACTION_REGISTRY.encode(phase, value), action_id)
                    self.assertEqual(legacy_action_id(phase, local), action_id)
        # Identical decoded None values still identify distinct decisions.
        self.assertEqual(ACTION_REGISTRY.encode("bid", None), 28)
        self.assertEqual(ACTION_REGISTRY.encode("trump", None), 54)
        for value in (True, 1.0):
            with self.assertRaises(ValueError):
                ACTION_REGISTRY.encode("trump", value)

    def test_appending_preserves_ids_and_mask_prefix(self):
        extension = ActionSpec(NUM_ACTIONS, "contract.future", "bid", ContractAction("future"), False)
        extended = ACTION_REGISTRY.appended(extension)
        self.assertEqual(len(extended), NUM_ACTIONS + 1)
        self.assertEqual(extended.entries[:NUM_ACTIONS], ACTION_REGISTRY.entries)
        self.assertEqual(extended.mask((28, 29))[:-1], ACTION_REGISTRY.mask((28, 29)))
        self.assertFalse(extended.mask((28, 29))[-1])
        self.assertEqual(len(ACTION_REGISTRY), NUM_ACTIONS)
        self.assertEqual(extended.manifest()[:NUM_ACTIONS], ACTION_REGISTRY.manifest())
        for entry in (ActionSpec(NUM_ACTIONS - 1, "bad.id", "bid", None),
                      ActionSpec(NUM_ACTIONS, "bid.pass", "bid", None),
                      ActionSpec(NUM_ACTIONS, "duplicate.pass", "bid", None)):
            with self.assertRaises(ValueError):
                ACTION_REGISTRY.appended(entry)

    def test_legacy_migration_rejects_inconsistent_labels(self):
        record = {"schema_version": 2, "action_id": 0, "observation": {
            "phase": "bid", "legal_action_ids": [0], "action_mask": [True] + [False] * 17,
        }}
        upgraded = upgrade_selfplay_record(record)
        self.assertEqual(upgraded["action_id"], 28)
        self.assertEqual(upgraded["observation"]["legal_action_ids"], [28])
        for bad in ({**record, "feature_schema_version": 999},
                    {**record, "schema_version": 3}, {**record, "action_id": 1},
                    {**record, "observation": {**record["observation"], "action_mask": [True]}},
                    {**record, "observation": {**record["observation"], "legal_action_ids": [1]}}):
            with self.assertRaises(ValueError):
                upgrade_selfplay_record(bad)

    def test_future_contracts_never_become_legal_without_rules(self):
        reserved = (NIL_ACTION_ID, SPLASH_ACTION_ID, PLUNGE_ACTION_ID)
        self.assertEqual(reserved, (55, 56, 57))
        env = SelfPlayEnv(seed=42)
        observation = env.reset()
        phases = set()
        while True:
            phases.add(observation.phase)
            self.assertEqual(len(observation.action_mask), NUM_ACTIONS)
            self.assertEqual(observation.num_actions, NUM_ACTIONS)
            for action_id in reserved:
                self.assertFalse(observation.action_mask[action_id])
                self.assertNotIn(action_id, observation.legal_action_ids)
                with self.assertRaises(ValueError):
                    env.step(action_id)
                self.assertEqual(env.observe(), observation)
            result = env.step(observation.legal_action_ids[0])
            if result.terminated:
                break
            observation = result.observation
        self.assertEqual(phases, {"bid", "trump", "play"})

    def test_wrong_phase_actions_are_rejected_atomically(self):
        env = SelfPlayEnv(seed=1)
        observation = env.reset()
        for action_id in (0, 27, 46, 54, NUM_ACTIONS, -1, True, 1.0):
            with self.assertRaises(ValueError):
                env.step(action_id)
            self.assertEqual(env.observe(), observation)
