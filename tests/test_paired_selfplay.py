"""Paired-partner PPO: seat routing, separate updates, checkpoints, evaluation, CLI.

Tiny hidden_size=8 fixtures in the style of test_ppo_multiphase. These check
mechanics and reproducibility, not whether paired partners learn better.
"""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import contextlib
    from dataclasses import asdict, replace
    import io
    import json
    import shutil
    import tempfile
    from pathlib import Path
    import torch
    from research.ppo.trainer import Config, Trainer
    from research.ppo.checkpoint import save as save_legacy, load_model, read, load_frozen
    from research.ppo.composition import (CompositeAgentFactory, PairedAgentFactory, PhasePolicySpec,
                                          CompositePolicySpec, partner_index)
    from research.ppo.evaluation import evaluate_policies
    from research.ppo.opponents import PolicySpec, CheckpointIdentity
    from research.ppo.selfplay_checkpoint import save, save_pair, pair_paths, resume_pair, config_from_dict
    from research.ppo.selfplay_rollout import collect_phase_hand
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer, PairedSelfPlayTrainer
    from texas42.agents import VARIANTS


def parameters(model):
    return [p.detach().clone() for p in model.parameters()]


def same(first, second):
    return all(torch.equal(a, b) for a, b in zip(first, second))


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class PairedSelfPlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu'))
        self.addCleanup(legacy.close)
        self.anchor_path = self.path / 'anchor.pt'
        save_legacy(self.anchor_path, legacy)
        self.initial, _ = load_model(self.anchor_path)
        self.anchor = PolicySpec.checkpoint('anchor', self.anchor_path)
        self.policies = tuple(PolicySpec(n) for n in VARIANTS) + (self.anchor,)

    def config(self, **changes):
        c = SelfPlayConfig(total_hands=4, hidden_size=8, device='cpu', seed=42, learn_phases=('bid', 'play'),
            rollout_decisions=8, ppo_epochs=2, minibatch_size=4, anchor=self.anchor,
            opponent_policies=self.policies, opponent_probabilities=(('HHH', '1'),),
            eval_deals=2, bootstrap_samples=5, forgetting_hands=1, partner_mode='paired')
        return replace(c, **changes)

    def trainer(self, **changes):
        trainer = PairedSelfPlayTrainer(self.config(**changes), self.initial)
        self.addCleanup(trainer.close)
        return trainer

    def test_configuration_rules(self):
        with self.assertRaises(ValueError):
            self.config(partner_mode='both')
        with self.assertRaisesRegex(ValueError, 'self-snapshots'):
            self.config(opponent_probabilities=(('HHH', '1'), ('self-snapshots', '1')))
        self.config(partner_mode='shared', opponent_probabilities=(('self-snapshots', '1'),))
        with self.assertRaises(ValueError):
            PairedSelfPlayTrainer(self.config(partner_mode='shared'), self.initial)
        # Checkpoints written before pairing existed have no partner_mode key.
        legacy = asdict(self.config(partner_mode='shared'))
        legacy.pop('partner_mode')
        self.assertEqual(config_from_dict(legacy).partner_mode, 'shared')

    def test_integration_with_scratch_self_current_and_critic(self):
        # Combinations introduced by main: from-scratch initialization and the
        # self-current opponent work in paired mode; the centralized critic is rejected.
        from research.ppo.multiphase import LearnerPhaseSpec, PHASES
        specs = tuple(LearnerPhaseSpec('heuristic', initialization='none') if p == 'trump'
                      else LearnerPhaseSpec('neural-train', None, 'random') for p in PHASES)
        config = SelfPlayConfig(total_hands=4, hidden_size=8, device='cpu', seed=42, from_scratch=True,
            learner_phases=specs, learn_phases=('bid', 'play'), rollout_decisions=8, ppo_epochs=1,
            minibatch_size=4, anchor=None, opponent_policies=tuple(PolicySpec(n) for n in VARIANTS),
            opponent_probabilities=(('self-current', '1'),), evaluation_names=('HHH',),
            primary_opponent='HHH', eval_deals=2, bootstrap_samples=5, forgetting_hands=1,
            partner_mode='paired')
        t = PairedSelfPlayTrainer(config)
        self.addCleanup(t.close)
        self.assertTrue(same(parameters(t.model), parameters(t.partner.model)))  # Same seeded init.
        # With equal weights, paired routing (including the self-current opponent
        # pair) reproduces the shared collection exactly.
        pool = t.opponent_pool()
        for episode in range(2):
            shared = collect_phase_hand(t.model, 42, episode, ('bid', 'play'), pool)
            paired = collect_phase_hand((t.model, t.partner.model), 42, episode, ('bid', 'play'), pool)
            self.assertEqual(shared['paths'], paired['paths'])
            self.assertEqual(shared['opponent'], 'self-current')
        buffers, _ = t.collect()
        t.update(buffers)
        self.assertEqual({label for _, label, _ in t.opponent_assignments}, {'self-current'})
        with self.assertRaisesRegex(ValueError, 'centralized critic'):
            replace(config, critic_mode='centralized')

    def test_identical_start_and_seat_routing(self):
        t = self.trainer()
        self.assertTrue(same(parameters(t.model), parameters(t.partner.model)))
        # Make partner 2 distinguishable, then check each record's stored
        # log-probability came from its own seat's network.
        with torch.no_grad():
            for p in t.partner.model.parameters():
                p.add_(0.25)
        (first, second), _ = t.collect()
        self.assertTrue(len(first) and len(second))
        self.assertEqual({partner_index(r.seat) for r in first.records}, {0})
        self.assertEqual({partner_index(r.seat) for r in second.records}, {1})
        for buffer, model in ((first, t.model), (second, t.partner.model)):
            data = buffer.tensors()
            with torch.no_grad():
                logp, _, _ = model.evaluate(data['features'], data['masks'], data['actions'])
            self.assertTrue(torch.allclose(logp, data['old_log_probs'], atol=1e-5))
        self.assertEqual((t.samples, t.partner.samples), (len(first), len(second)))
        self.assertEqual(t.partner.hands, t.hands)
        self.assertEqual(sum(sum(c.values()) for c in t.partner_phase_counts), len(first) + len(second))

    def test_identical_partners_reproduce_shared_collection(self):
        # With equal weights, paired routing must not change any record: the
        # sampling stream and opponents are shared exactly as in shared mode.
        t = self.trainer()
        pool = t.opponent_pool()
        for episode in range(3):
            shared = collect_phase_hand(t.model, 42, episode, ('bid', 'play'), pool)
            paired = collect_phase_hand((t.model, t.partner.model), 42, episode, ('bid', 'play'), pool)
            self.assertEqual(shared['paths'], paired['paths'])

    def test_each_partner_updates_only_from_its_own_samples(self):
        t = self.trainer()
        start = parameters(t.model)
        buffers, _ = t.collect()
        metrics = t.update(buffers)
        self.assertEqual(set(metrics), {'partner_1', 'partner_2'})
        self.assertFalse(same(start, parameters(t.model)))
        self.assertFalse(same(start, parameters(t.partner.model)))
        self.assertFalse(same(parameters(t.model), parameters(t.partner.model)))  # Now specialized.
        self.assertEqual((t.updates, t.partner.updates), (1, 1))
        self.assertIsNot(t.optimizer, t.partner.optimizer)
        with self.assertRaises(ValueError):
            t.publish_snapshot(self.path / 'snapshots')

    def test_worker_count_invariance(self):
        results = []
        for threads in (1, 2):
            t = self.trainer(threads=threads)
            buffers, reward = t.collect()
            t.update(buffers)
            results.append(([b.records for b in buffers], reward,
                            parameters(t.model), parameters(t.partner.model)))
        (records_1, reward_1, first_1, second_1), (records_2, reward_2, first_2, second_2) = results
        self.assertEqual((records_1, reward_1), (records_2, reward_2))
        self.assertTrue(same(first_1, first_2) and same(second_1, second_2))

    def test_pair_checkpoints_resume_exactly_and_load_alone(self):
        t = self.trainer()
        t.update(t.collect()[0])
        first, second = save_pair(self.path / 'checkpoints', 'latest.pt', t)
        self.assertEqual(pair_paths(second), (first, second))
        resumed = self.trainer()
        resume_pair(second, resumed)
        for a, b in ((t, resumed), (t.partner, resumed.partner)):
            self.assertTrue(same(parameters(a.model), parameters(b.model)))
            self.assertEqual((a.hands, a.samples, a.updates), (b.hands, b.samples, b.updates))
            self.assertEqual(a.optimizer.state_dict()['state'].keys(), b.optimizer.state_dict()['state'].keys())
        # Continuing both copies gives identical partners (exact resume).
        for trainer in (t, resumed):
            trainer.update(trainer.collect()[0])
        self.assertTrue(same(parameters(t.partner.model), parameters(resumed.partner.model)))
        # Each partner file is an ordinary checkpoint usable on its own.
        model, metadata = load_frozen(CheckpointIdentity.from_path(second))
        self.assertEqual(metadata['pairing']['partner'], 2)
        spec = CompositePolicySpec('P2', PhasePolicySpec('neural', CheckpointIdentity.from_path(second)),
                                   PhasePolicySpec('heuristic'),
                                   PhasePolicySpec('neural', CheckpointIdentity.from_path(second)))
        CompositeAgentFactory(spec).prepare()
        # Mismatched or unpaired files are rejected.
        other = self.trainer(seed=7)
        other.update(other.collect()[0])
        mixed = save_pair(self.path / 'other', 'latest.pt', other)
        shutil.copy(first, mixed[0])
        with self.assertRaises(ValueError):
            resume_pair(mixed[1], self.trainer())
        shared = SelfPlayTrainer(self.config(partner_mode='shared'), self.initial)
        self.addCleanup(shared.close)
        save(self.path / 'shared.pt', shared)
        with self.assertRaisesRegex(ValueError, 'not a paired'):
            pair_paths(self.path / 'shared.pt')

    def test_seat_aware_factories(self):
        calls = []
        factory = PairedAgentFactory(lambda seat, seed: calls.append(('first', seat)),
                                     lambda seat, seed: calls.append(('second', seat)))
        for seat in range(4):
            factory(seat, 0)
        self.assertEqual(calls, [('first', 0), ('first', 1), ('second', 2), ('second', 3)])
        # A pair of identical networks validates exactly like the shared model.
        t = self.trainer()
        alone = evaluate_policies(t.model, (PolicySpec('HHH'),), deals=2, bootstrap_samples=5, seed=3)
        paired = evaluate_policies(t.model, (PolicySpec('HHH'),), deals=2, bootstrap_samples=5, seed=3,
                                   partner_model=t.partner.model)
        self.assertEqual(alone['HHH'].games, paired['HHH'].games)

    def test_two_spec_team_evaluation_is_worker_invariant(self):
        from research.ppo.composition import evaluate_composed
        import pickle
        first = CompositePolicySpec('A', PhasePolicySpec('conservative'))
        partner = CompositePolicySpec('A partner', trump=PhasePolicySpec('random'),
                                      playing=PhasePolicySpec('random'))
        team = PairedAgentFactory(CompositeAgentFactory(first), CompositeAgentFactory(partner))
        self.assertEqual(pickle.loads(pickle.dumps(team)), team)  # Portable to spawn workers.
        reports = [evaluate_composed(first, CompositePolicySpec('B'), a_partner=partner, num_deals=4,
                                     seed=5, bootstrap_samples=5, threads=threads) for threads in (1, 2)]
        self.assertEqual(reports[0].games, reports[1].games)
        mirrored = evaluate_composed(first, CompositePolicySpec('B'), num_deals=4, seed=5, bootstrap_samples=5)
        self.assertNotEqual(mirrored.games, reports[0].games)  # The partner spec is actually used.

    def test_cli_training_resume_and_pair_evaluation(self):
        from examples.train_selfplay_ppo import main as train
        from examples.evaluate_agent_vs_agent import main as evaluate
        common = ['--rollout-decisions', '8', '--ppo-epochs', '1', '--minibatch-size', '4',
                  '--eval-deals', '2', '--bootstrap-samples', '5', '--forgetting-hands', '1',
                  '--eval-opponents', 'HHH,anchor', '--opponent-pool', 'HHH:1']
        run = self.path / 'run'
        with contextlib.redirect_stdout(io.StringIO()):
            trained = train(['--init-play-checkpoint', str(self.anchor_path), '--learn-phases', 'bid,play',
                             '--paired-partners', '--total-hands', '2', '--output-dir', str(run),
                             '--log-dir', str(self.path / 'log1')] + common)
        self.assertIsInstance(trained, PairedSelfPlayTrainer)
        checkpoints = run / 'checkpoints'
        for partner in ('partner-1', 'partner-2'):
            self.assertTrue((checkpoints / partner / 'latest.pt').exists())
        row = json.loads((run / 'progress.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['training_samples'], sum(p['samples'] for p in row['partners'].values()))
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = train(['--resume', str(checkpoints / 'partner-1/latest.pt'), '--total-hands', '4',
                             '--output-dir', str(self.path / 'resumed'), '--log-dir', str(self.path / 'log2')])
        self.assertEqual((resumed.hands, resumed.partner.hands, resumed.config.partner_mode), (4, 4, 'paired'))
        output = self.path / 'pair.json'
        with contextlib.redirect_stdout(io.StringIO()):
            # Partner 2 plays heuristic, but its neural bid inherits the first seat's checkpoint.
            evaluate(['--agent-a-bid', 'neural', '--agent-a-play', 'neural',
                      '--agent-a-checkpoint', str(checkpoints / 'partner-1/latest.pt'),
                      '--agent-a-partner-play', 'heuristic', '--deals', '1', '--bootstrap-samples', '5',
                      '--output', str(output), '--log-dir', str(self.path / 'log3')])
        composition = json.loads(output.read_text())['composition']
        partner = composition['partners']['A']
        self.assertEqual((partner['bidding']['strategy'], partner['playing']['strategy']), ('neural', 'heuristic'))
        self.assertTrue(composition['partner_comparisons']['A']['neural_parameters_identical']['bid'])
        self.assertIsNone(composition['partners']['B'])
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate(['--deals', '1', '--bootstrap-samples', '5', '--output', str(output),
                      '--log-dir', str(self.path / 'log4')])
        self.assertNotIn('partners', json.loads(output.read_text())['composition'])


if __name__ == '__main__':
    unittest.main()
