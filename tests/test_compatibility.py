"""Neural layout and rules provenance must not depend on action eligibility."""
import json
import unittest
from dataclasses import asdict, replace

from texas42 import (
    ACTION_REGISTRY, Contract, ContractKind, GameConfig, Phase, RULESET_ID,
    AuctionCall, ContractAction, ContractDeclaration, FEATURE_SCHEMA_VERSION, HandResult,
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

    def test_declaration_views_and_contract_authoritative_results(self):
        from texas42.players import BidView, TrumpView, PlayView
        from texas42.training import training_view
        declaration: AuctionCall = ContractDeclaration("nil")
        self.assertIs(ContractAction, ContractDeclaration)
        self.assertEqual(ACTION_REGISTRY[55].value, declaration)
        history = ((0, declaration), (1, None))
        view = BidView(2, 3, (), declaration, 0, False, bid_history=history)
        self.assertEqual(training_view(view).current_bid, declaration)
        self.assertEqual(training_view(view).bid_history, history)
        contract = Contract(ContractKind.NIL, 0)
        trump = TrumpView(0, (), None, bid_history=history, contract=contract)
        play = PlayView(0, (), (), (), (), 0, None, None, (0, 0), (0, 0),
                        bid_history=history, contract=contract)
        for view in (trump, play):
            encoded = training_view(view)
            self.assertEqual(encoded.contract, contract)
            self.assertIsNone(encoded.bid)
        result = HandResult(contract, None, (0, 0), 1, 1)
        self.assertIsNone(result.bid)
        self.assertEqual(result.bidder, 0)
        self.assertEqual(asdict(result)["contract"]["kind"], "nil")
        normal = replace(result, contract=Contract(ContractKind.NORMAL, 2, Bid(points=30)))
        self.assertEqual(normal.bid, Bid(points=30))
        self.assertEqual(normal.bidder, 2)
        with self.assertRaises(ValueError):
            replace(result, contract=None)

    def test_feature_version_metadata(self):
        self.assertEqual(training_metadata()["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(SelfPlayEnv().reset().feature_schema_version, FEATURE_SCHEMA_VERSION)
