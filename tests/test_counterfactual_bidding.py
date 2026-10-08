"""Counterfactual interventions use real arena hands and exact policy replays."""
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass, replace
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from examples.counterfactual_bidding import main, format_results
from examples.heuristic_benchmark import breakdown
from texas42.actions import ACTION_REGISTRY
from texas42.agents import RandomAgent, variant_factory
from texas42.arena import ArenaReport, AgentExecutionError, generate_schedule, run_game
from texas42.experiments.counterfactual_bidding import (
    PASS, bootstrap_delta, compare_bidding, evaluate_hand, signed_marks, summarize)
from texas42.models import Bid


class PassAgent:
    def reset(self, seat): self.seat = seat
    def act(self, obs):
        if obs.phase == 'bid' and PASS in obs.legal_action_ids: return PASS
        return obs.legal_action_ids[0]


class MinimumBidAgent(PassAgent):
    def act(self, obs):
        if obs.phase == 'bid':
            bids = [i for i in obs.legal_action_ids if isinstance(ACTION_REGISTRY[i].value, Bid)]
            if bids: return min(bids, key=lambda i: ACTION_REGISTRY[i].value.strength)
        return super().act(obs)


def passes(seat, seed): return PassAgent()
def minimum(seat, seed): return MinimumBidAgent()


class CounterfactualTests(unittest.TestCase):
    def evaluate(self, factories, seating=('B', 'A', 'B', 'A')):
        return evaluate_hand(factories, generate_schedule(1, 19)[0], seating=seating,
                             target_policy='A', seed=15)

    def test_first_opportunity_exact_intervention_and_partner_overbid(self):
        record = self.evaluate((passes, minimum, passes, minimum))
        self.assertEqual(record.eligible_decisions, 2)
        self.assertEqual(record.skipped_eligible_decisions, 1)
        row = record.intervention
        self.assertEqual(row.auction_position, 1)
        self.assertEqual(row.acting_seat, 1)
        self.assertFalse(row.forced)
        self.assertEqual(row.original_bid, Bid(points=30))
        self.assertEqual(row.bid.action_ids[0], ACTION_REGISTRY.encode('bid', Bid(points=30)))
        self.assertEqual(row.passed.action_ids[0], PASS)
        # The target's partner independently changes its subsequent bid.
        self.assertEqual(row.bid.hand.result.bidder, 3)
        self.assertEqual(row.bid.hand.result.bid, Bid(points=31))
        self.assertEqual(row.passed.hand.result.bidder, 3)
        self.assertEqual(row.passed.hand.result.bid, Bid(points=30))
        self.assertEqual(row.bid.hand.deal_id, row.passed.hand.deal_id)
        self.assertEqual(row.bid.hand.shaker, row.passed.hand.shaker)
        self.assertTrue(row.bid_replay_verified)

    def test_nonempty_prefix_and_opponent_continuation(self):
        # Seats 1 passes, 2 bids, 3 (opponent) raises; both replays share prefix.
        record = self.evaluate((passes, passes, minimum, minimum), ('B', 'B', 'A', 'B'))
        row = record.intervention
        self.assertEqual(row.auction_position, 2)
        self.assertEqual(row.bid.action_ids[:1], row.passed.action_ids[:1])
        self.assertEqual(row.bid.hand.result.bidder, 3)
        self.assertEqual(row.passed.hand.result.bidder, 3)
        self.assertEqual(row.passed.hand.result.bid, Bid(points=30))
        # A non-bidding continuation can instead move declaration to the shaker.
        row = self.evaluate((passes, minimum, passes, passes)).intervention
        self.assertEqual(row.bid.hand.result.bidder, 1)
        self.assertEqual(row.passed.hand.result.bidder, 0)
        self.assertTrue(row.passed.hand.forced_bid)
        self.assertEqual(row.bid.signed_marks, signed_marks(row.bid.hand, 1))
        self.assertEqual(row.passed.signed_marks, signed_marks(row.passed.hand, 1))
        self.assertEqual(row.delta_marks, row.bid.signed_marks-row.passed.signed_marks)
        for branch in (row.bid, row.passed):
            self.assertEqual(branch.acting_team_won, branch.hand.result.scoring_team == 1)
            self.assertEqual(branch.hand.decisions, 33)
            self.assertEqual(len(branch.action_ids), 33)
            self.assertEqual(sorted(branch.action_ids[5:]), list(range(28)))
            self.assertEqual(sum(branch.hand.result.team_points), 42)
            self.assertIn(branch.hand.contract_decided_at_trick, range(1, 8))
            self.assertEqual(branch.hand.result.contract.kind, 'normal')

    def test_forced_bids_never_selected_including_above_thirty(self):
        class HighForced(PassAgent):
            def act(self, obs):
                if obs.phase == 'bid' and obs.view.forced:
                    return ACTION_REGISTRY.encode('bid', Bid(points=37))
                return super().act(obs)
        record = self.evaluate((lambda seat, seed: HighForced(),) * 4, ('A',) * 4)
        self.assertEqual(record.eligible_decisions, 0)
        self.assertIsNone(record.intervention)
        report = compare_bidding('CHH', 'CRR', target_policy='CHH', num_deals=2, bootstrap_samples=10)
        data = report.to_dict()
        self.assertEqual(data['ineligible_hands'], 4)
        self.assertEqual(data['overall']['samples'], 0)
        self.assertIsNone(data['overall']['mean_delta_marks'])
        self.assertIn('N/A', format_results(data, 0))

    def test_stochastic_replay_privacy_and_fresh_agents(self):
        made = []
        test = self
        class Checked(RandomAgent):
            def __init__(self, seed):
                super().__init__(seed)
                self.resets = []
                self.traces = []
            def reset(self, seat):
                super().reset(seat)
                self.resets.append(seat)
            def act(self, obs):
                def private(value):
                    if is_dataclass(value):
                        test.assertTrue(type(value).__dataclass_params__.frozen)
                        for field in fields(value):
                            test.assertNotIn(field.name, ('hands', 'state', 'initial_deal', 'privileged'))
                            private(getattr(value, field.name))
                    elif isinstance(value, tuple):
                        for v in value: private(v)
                private(obs)
                action = super().act(obs)
                self.traces.append((obs, action))
                return action
        def factory(seat, seed):
            agent = Checked(seed)
            made.append(agent)
            return agent
        result = self.evaluate((factory,) * 4, ('A',) * 4)
        self.assertIsNotNone(result.intervention)
        self.assertEqual(len(made), 12)
        self.assertEqual(len({id(a) for a in made}), 12)
        for i, agent in enumerate(made): self.assertEqual(agent.resets, [i % 4])
        for original, replay in zip(made[:4], made[4:8]):
            self.assertEqual(original.traces, replay.traces)
        # All policies are still called for every decision, including the override.
        self.assertEqual(sum(len(a.traces) for a in made), 99)
        self.assertEqual(result, self.evaluate((factory,) * 4, ('A',) * 4))

    def test_bad_factories_and_replay_drift_are_rejected(self):
        shared = MinimumBidAgent()
        with self.assertRaises(AgentExecutionError):
            self.evaluate((lambda seat, seed: shared,) * 4)
        made = []
        def drifting(seat, seed):
            agent = MinimumBidAgent() if len(made) < 4 else PassAgent()
            made.append(agent)
            return agent
        with self.assertRaises(AgentExecutionError):
            self.evaluate((drifting,) * 4)
        class Illegal(PassAgent):
            def act(self, obs): return True
        with self.assertRaises(AgentExecutionError):
            self.evaluate((lambda seat, seed: Illegal(),) * 4)

    def test_statistics_perspective_and_group_bootstrap(self):
        row = self.evaluate((passes, minimum, passes, passes)).intervention
        positive = replace(row, group_id='one', delta_marks=2,
                           bid=replace(row.bid, signed_marks=1), passed=replace(row.passed, signed_marks=-1))
        negative = replace(row, group_id='two', delta_marks=-2,
                           bid=replace(row.bid, signed_marks=-1), passed=replace(row.passed, signed_marks=1))
        equal = replace(row, group_id='two', delta_marks=0,
                        bid=replace(row.bid, signed_marks=1), passed=replace(row.passed, signed_marks=1))
        stats = summarize([positive, negative, equal], seed=1, samples=20)
        self.assertEqual(stats['mean_delta_marks'], 0)
        self.assertEqual(stats['median_delta_marks'], 0)
        self.assertEqual(stats['bid_better'] + stats['pass_better'] + stats['equal'], 3)
        self.assertAlmostEqual(sum(stats[k] for k in ('fraction_bid_better', 'fraction_pass_better', 'fraction_equal')), 1)
        self.assertEqual(stats, summarize([positive, negative, equal], seed=1, samples=20))
        self.assertIsNone(bootstrap_delta([positive], samples=10))
        self.assertIsNone(bootstrap_delta([positive, replace(negative, group_id='one')], samples=10))
        groups_seen = []
        class ClusterSampler:
            def choice(self, groups):
                groups_seen.append(groups)
                return groups[1]  # Always choose the complete two-observation group.
        with patch('texas42.experiments.counterfactual_bidding.random.Random', return_value=ClusterSampler()):
            ci = bootstrap_delta([positive, negative, equal], samples=5)
        self.assertEqual(ci, {'low': -1, 'high': -1})
        self.assertTrue(all(g == ([2], [-2, 0]) for g in groups_seen))

    def test_duplicate_schedules_and_original_arena_equivalence(self):
        from texas42.arena import _seed
        report = compare_bidding('RRR', 'CHH', target_policy='RRR', num_deals=3, seed=7, bootstrap_samples=10)
        self.assertEqual(report, compare_bidding('RRR', 'CHH', target_policy='RRR', num_deals=3, seed=7, bootstrap_samples=10))
        for index, (left, right) in enumerate(zip(report.hands[::2], report.hands[1::2])):
            self.assertEqual(left.deal_id, right.deal_id)
            self.assertEqual(left.group_id, right.group_id)
            for hand in (left, right):
                row = hand.intervention
                if row is None: continue
                schedule = generate_schedule(1, _seed(7, 'schedule', index), shaker=index % 4)
                direct = run_game(tuple(variant_factory(p) for p in hand.seating), schedule,
                    seating=hand.seating, seed=_seed(7, 'episode', index), game_id=hand.game_id, group_id=hand.group_id)
                self.assertEqual(row.bid.hand, direct.hands[0])
                self.assertEqual(row.agent_seeds, direct.agent_seeds)
        with self.assertRaises(ValueError): compare_bidding(episode='match')
        with self.assertRaises(ValueError): compare_bidding(target_policy='RRR')

    def test_resolved_origin_breakdown_including_high_forced(self):
        class High(PassAgent):
            def act(self, obs):
                if obs.phase == 'bid' and obs.view.forced:
                    return ACTION_REGISTRY.encode('bid', Bid(marks=2))
                return super().act(obs)
        game = run_game((lambda s, seed: High(),) * 4, generate_schedule(1, 7), seating=('A','B','A','B'))
        data = breakdown(ArenaReport(7, 0, 10, (game,), (), ()))
        counts = data['A']['contract_origins']['marks:2']['forced']
        self.assertEqual(counts['contracts'], 1)
        self.assertEqual(counts['successful']+counts['failed'], 1)
        self.assertEqual(counts['mean_signed_marks'], signed_marks(game.hands[0], 0))

    def test_cli_json_determinism_and_no_hidden_deals(self):
        subprocess.run([sys.executable, '-c',
            'import sys; import texas42.experiments.counterfactual_bidding; '
            'assert not {"numpy", "torch", "pyspiel", "open_spiel"}.intersection(sys.modules)'], check=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            args = ['--policy-a','RRR','--target-policy','RRR','--deals','2',
                    '--bootstrap-samples','10','--output',str(path)]
            with redirect_stdout(io.StringIO()) as output:
                first = main(args)
                saved = path.read_text()
                self.assertEqual(main(args), first)
            self.assertEqual(path.read_text(), saved)
            data = json.loads(saved)
            self.assertEqual(data['total_hands'], 4)
            self.assertIn('Counterfactual bidding outcomes', output.getvalue())
            self.assertNotIn('initial_deal', saved)
            for hand in data['hands']:
                row = hand['intervention']
                if row:
                    self.assertEqual(len(row['features']['hand']), 7)
                    self.assertIn('by_hand_strength', data)
                    self.assertTrue(row['bid_replay_verified'])


if __name__ == '__main__':
    unittest.main()
