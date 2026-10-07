"""Neural layout and rules provenance must not depend on action eligibility."""
import json
import unittest
from dataclasses import replace

from texas42 import (
    ACTION_REGISTRY, Contract, ContractKind, GameConfig, Phase, RULESET_ID,
    SelfPlayEnv, Bid, enabled_contracts, training_metadata,
)


class CompatibilityTests(unittest.TestCase):
    def test_stable_phase_heads_and_enablement(self):
        expected = {"bid": list(range(28, 46)) + [55, 56, 57],
                    "trump": list(range(46, 55)), "play": list(range(28))}
        for phase, ids in expected.items():
            self.assertEqual([entry.action_id for entry in ACTION_REGISTRY.phase_slots(phase)], ids)
            for slot, action_id in enumerate(ids):
                self.assertEqual(ACTION_REGISTRY.global_id(phase, slot), action_id)
                self.assertEqual(ACTION_REGISTRY.phase_slot(phase, action_id), slot)
        self.assertEqual(len(ACTION_REGISTRY.implemented_for_phase("bid")), 18)
        self.assertEqual(ACTION_REGISTRY.phase_mask("bid", (28, 29))[-3:], (False,) * 3)
        # A hypothetical eligibility change preserves checkpoint row identities;
        # this isolated registry does not enable any rule in the actual engine.
        future = replace(ACTION_REGISTRY, entries=tuple(
            replace(entry, implemented=True) if entry.action_id == 55 else entry
            for entry in ACTION_REGISTRY.entries))
        self.assertEqual(future.global_id("bid", 18), 55)
        self.assertEqual(len(future.phase_slots("bid")), 21)
        self.assertTrue(future.phase_mask("bid", (55,))[18])
        self.assertFalse(ACTION_REGISTRY[55].implemented)
        for phase, slot in (("bid", 21), ("bid", -1), ("bid", True), ("trump", 9)):
            with self.assertRaises(ValueError):
                ACTION_REGISTRY.global_id(phase, slot)
        with self.assertRaises(ValueError):
            ACTION_REGISTRY.phase_slot("bid", 0)
        with self.assertRaises(ValueError):
            ACTION_REGISTRY.phase_mask("trump", (28,))

    def test_contract_resolves_after_auction_and_survives_terminal(self):
        env = SelfPlayEnv(seed=42)
        observation = env.reset(shaker=0)
        self.assertIsNone(observation.view.contract)
        for _ in range(3):
            observation = env.step(28).observation
            self.assertIsNone(observation.view.contract)
        observation = env.step(29).observation
        contract = Contract(ContractKind.NORMAL, 0, Bid(points=30))
        self.assertEqual(observation.view.contract, contract)
        self.assertEqual(observation.phase, Phase.TRUMP)
        while True:
            result = env.step(observation.legal_action_ids[0])
            if result.terminated:
                self.assertEqual(result.hand_result.contract, contract)
                break
            observation = result.observation
            self.assertEqual(observation.view.contract, contract)
        self.assertEqual(enabled_contracts(), (ContractKind.NORMAL,))
        for action_id, kind in ((55, ContractKind.NIL), (56, ContractKind.SPLASH),
                                (57, ContractKind.PLUNGE)):
            self.assertEqual(ACTION_REGISTRY[action_id].value.kind, kind)
        with self.assertRaises(ValueError):
            Contract(ContractKind.NORMAL, 0)

    def test_ruleset_provenance_and_checkpoint_metadata(self):
        env = SelfPlayEnv(ruleset_id=RULESET_ID)
        self.assertEqual(env.reset().ruleset_id, RULESET_ID)
        metadata = training_metadata()
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        self.assertEqual(metadata["ruleset_id"], RULESET_ID)
        self.assertEqual(len(metadata["phase_slots"]["bid"]), 21)
        for factory in (SelfPlayEnv, GameConfig, training_metadata):
            with self.assertRaises(ValueError):
                factory(ruleset_id="cadets_nil_v1")
