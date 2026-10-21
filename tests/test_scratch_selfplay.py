"""From-scratch multi-phase PPO, current-policy self-play and anchor-free lineages."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import contextlib
    import hashlib
    import io
    import json
    import tempfile
    from dataclasses import replace
    from pathlib import Path
    from unittest.mock import patch
    import torch
    from examples.train_selfplay_ppo import main as train
    from research.ppo.agent import PPOAgent
    from research.ppo.checkpoint import load_model, read, save as save_legacy
    from research.ppo.multiphase import LearnerPhaseSpec, PHASES
    from research.ppo.opponents import HistoricalPool, PolicySpec, policy_registry
    from research.ppo.rollout import snapshot
    from research.ppo.selfplay_checkpoint import resume, save
    from research.ppo.selfplay_rollout import collect_phase_hand
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
    from research.ppo.trainer import Config, Trainer
    from texas42.agents import VARIANTS
    from texas42.selfplay import SelfPlayEnv


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class ScratchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.variants = tuple(PolicySpec(n) for n in VARIANTS)

    def specs(self, *states):
        return tuple(LearnerPhaseSpec('neural-train', None, 'random') if s == 'neural-train'
                     else LearnerPhaseSpec('heuristic', initialization='none') for s in states)

    def config(self, states=('neural-train',) * 3, pool=(('self-current', '1'),), **changes):
        specs = self.specs(*states)
        c = SelfPlayConfig(total_hands=8, hidden_size=8, device='cpu', seed=42, from_scratch=True,
                           learner_phases=specs, learn_phases=tuple(p for p, s in zip(PHASES, states) if s == 'neural-train'),
                           rollout_decisions=32, ppo_epochs=2, minibatch_size=8, anchor=None,
                           opponent_policies=self.variants, opponent_probabilities=pool,
                           evaluation_names=('HHH', 'CHH'), primary_opponent='HHH', eval_deals=2,
                           bootstrap_samples=5, forgetting_hands=1, snapshot_every_hands=4, snapshot_pool_size=2)
        return replace(c, **changes)

    def trainer(self, config):
        trainer = SelfPlayTrainer(config)
        self.addCleanup(trainer.close)
        return trainer

    def cli(self, name, *extra, total='8'):
        args = ['--from-scratch', '--total-hands', total, '--rollout-decisions', '32', '--hidden-size', '8',
                '--minibatch-size', '16', '--ppo-epochs', '1', '--eval-deals', '2', '--bootstrap-samples', '5',
                '--forgetting-hands', '1', '--device', 'cpu', '--seed', '42', *extra,
                '--output-dir', str(self.path / name), '--log-dir', str(self.path / (name + '-logs'))]
        return quiet(train, args)

    # Initialization -------------------------------------------------------
    def test_random_initialization_for_phase_subsets(self):
        for states in (('neural-train',) * 3, ('heuristic', 'heuristic', 'neural-train'),
                       ('neural-train', 'heuristic', 'heuristic'), ('heuristic', 'neural-train', 'heuristic'),
                       ('neural-train', 'heuristic', 'neural-train')):
            with self.subTest(states=states):
                t = self.trainer(self.config(states))
                neural = tuple(p for p, s in zip(PHASES, states) if s != 'heuristic')
                self.assertEqual(t.model.neural_phases, neural)
                self.assertEqual(t.model.trained_phases, ())
                self.assertEqual(t.learner_phases, self.specs(*states))
                self.assertEqual((t.hands, t.updates, t.optimizer_steps), (0, 0, 0))
                self.assertFalse(t.optimizer.state)
                for phase in PHASES:
                    trainable = all(p.requires_grad for p in t.model.networks[phase].parameters())
                    self.assertEqual(trainable, phase in neural)

    def test_seeded_initialization_and_no_pretrained_reuse(self):
        weights = lambda t: [v.clone() for v in t.model.state_dict().values()]
        a, b = weights(self.trainer(self.config())), weights(self.trainer(self.config()))
        c = weights(self.trainer(self.config(seed=43)))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(all(torch.equal(x, y) for x, y in zip(a, c)))
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu', seed=42))
        anchor_path = self.path / 'anchor.pt'
        save_legacy(anchor_path, legacy)
        anchor, _ = load_model(anchor_path)
        scratch = self.trainer(self.config())
        for phase in PHASES:  # No phase copies the pretrained trunk or heads.
            self.assertFalse(torch.equal(scratch.model.networks[phase].trunk[0].weight, anchor.trunk[0].weight))
        with self.assertRaises(ValueError):
            SelfPlayTrainer(self.config(), anchor)

    def test_invalid_scratch_configurations(self):
        with self.assertRaises(ValueError):
            LearnerPhaseSpec('neural-frozen', None, 'random')
        with self.assertRaises(ValueError):
            self.config(learner_phases=(LearnerPhaseSpec('neural-train', None, 'representation'),
                                        *self.specs('heuristic', 'heuristic')))
        with self.assertRaises(ValueError):
            replace(self.config(), from_scratch=False, opponent_probabilities=(('HHH', '1'),))
        for flags in (['--learner-play', 'neural-frozen'], ['--learner-play-source', 'anchor'],
                      ['--learner-bid', 'neural-train', '--learn-phases', 'play'],
                      ['--learner-bid', 'heuristic', '--learner-trump', 'heuristic', '--learner-play', 'heuristic'],
                      ['--init-checkpoint', str(self.path / 'missing.pt')],
                      ['--eval-opponents', 'HHH,self-current'], ['--selection-opponent', 'anchor'],
                      ['--opponent-pool', 'anchor:1'], ['--opponent-pool', 'HHH:1,HHH:1']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.cli('invalid-' + str(len(flags)) + flags[-1].replace(',', '-').replace('/', '_'), *flags)

    def test_cli_phase_defaults(self):
        t = self.cli('all', '--opponent-pool', 'HHH:1')
        self.assertEqual(t.config.learn_phases, PHASES)
        self.assertEqual(t.config.evaluation_names, ('HHH', 'CHH'))
        self.assertEqual(t.config.primary_opponent, 'HHH')
        t = self.cli('play', '--learner-play', 'neural-train')
        self.assertEqual([s.state for s in t.config.learner_phases], ['heuristic', 'heuristic', 'neural-train'])
        t = self.cli('learn', '--learn-phases', 'bid,play')
        self.assertEqual([s.initialization for s in t.config.learner_phases], ['random', 'none', 'random'])

    # Self-play ------------------------------------------------------------
    def test_self_current_first_rollout_frozen_snapshot_and_refresh(self):
        t = self.trainer(self.config())
        acted = []
        original = PPOAgent.act

        def act(agent, obs):
            action = original(agent, obs)
            acted.append((agent.seat, obs.legal_action_ids, action, agent.stochastic_phases))
            return action
        before = hashlib.sha256(snapshot(t.model)).hexdigest()
        with patch.object(PPOAgent, 'act', act):
            buffer, _ = t.collect()
        first = {row[2] for row in t.opponent_assignments}
        self.assertEqual({row[1] for row in t.opponent_assignments}, {'self-current'})
        self.assertEqual(first, {before})  # One immutable snapshot for the whole rollout.
        self.assertTrue(acted)
        for seat, legal, action, stochastic in acted:
            self.assertIn(action, legal)
            self.assertEqual(stochastic, PHASES)  # Trainable phases are sampled, not greedy.
        teams = {episode: (42 + episode) % 2 for episode, _, _ in t.opponent_assignments}
        self.assertTrue(all(r.seat % 2 == teams[r.episode_id] for r in buffer.records))
        for r in buffer.records:
            if r.done:
                partner = [x for x in buffer.records if x.done and x.episode_id == r.episode_id]
                self.assertEqual({x.reward for x in partner}, {r.reward})
                self.assertNotEqual(r.reward, 0)
        self.assertEqual(t.phase_counts['bid'], 2 * len(teams))
        self.assertEqual(t.phase_counts['play'], 14 * len(teams))
        frozen = [p.clone() for p in t.model.parameters()]
        t.update(buffer)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(frozen, t.model.parameters())))
        t.collect()
        after = {row[2] for row in t.opponent_assignments}
        self.assertEqual(after, {hashlib.sha256(snapshot(t.model)).hexdigest()})
        self.assertNotEqual(after, first)

    def test_stochastic_current_policy_sampling(self):
        t = self.trainer(self.config())
        obs = SelfPlayEnv(seed=1).reset()
        actions = set()
        for seed in range(30):
            agent = PPOAgent(t.model, seed=seed, stochastic_phases=PHASES)
            agent.reset(obs.seat)
            actions.add(agent.act(obs))
        greedy = set()
        for seed in range(5):
            agent = PPOAgent(t.model, seed=seed)
            agent.reset(obs.seat)
            greedy.add(agent.act(obs))
        self.assertGreater(len(actions), 1)
        self.assertEqual(len(greedy), 1)

    def test_mixtures_and_snapshot_fallbacks(self):
        pool = HistoricalPool((('HHH', '3'), ('CHH', '2'), ('self-current', '5')), self.variants, None,
                              snapshot_fallback='current')
        self.assertEqual(pool.probabilities, (('CHH', '1/5'), ('HHH', '3/10'), ('self-current', '1/2')))
        chosen = [pool.choose(42, e) for e in range(200)]
        self.assertEqual({name for name, _ in chosen}, {'HHH', 'CHH', 'self-current'})
        self.assertTrue(all((spec is None) == (name == 'self-current') for name, spec in chosen))
        self.assertEqual(chosen, [pool.choose(42, e) for e in range(200)])
        history = HistoricalPool((('self-snapshots', '1'),), self.variants, None, snapshot_fallback='current')
        self.assertEqual(history.choose(1, 0), ('self-snapshots', None))
        published = PolicySpec('HHR')  # Any frozen spec stands in for a published snapshot here.
        self.assertEqual(replace(history, snapshots=(published,)).choose(1, 0), ('self-snapshots', published))
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu'))
        save_legacy(self.path / 'anchor.pt', legacy)
        anchor = PolicySpec.checkpoint('anchor', self.path / 'anchor.pt')
        old = HistoricalPool((('self-snapshots', '1'),), (*self.variants, anchor), anchor)
        self.assertEqual(old.choose(1, 0), ('self-snapshots', anchor))
        with self.assertRaisesRegex(ValueError, 'no original play-only anchor'):
            HistoricalPool((('anchor', '1'),), self.variants, None, snapshot_fallback='current')
        with self.assertRaises(ValueError):
            HistoricalPool((('HHH', '1'),), self.variants, None)  # anchor fallback without anchor
        self.assertNotIn('anchor', policy_registry(None))
        for names in (('HHH', 'self-current'), ('HHH', 'self-snapshots')):
            with self.assertRaisesRegex(ValueError, 'changes during training'):
                self.config(evaluation_names=names)
        with self.assertRaisesRegex(ValueError, 'no original play-only anchor'):
            self.config(evaluation_names=('HHH', 'anchor'))

    def test_self_snapshots_without_anchor(self):
        t = self.trainer(self.config(pool=(('self-snapshots', '7'), ('HHH', '3'))))
        buffer, _ = t.collect()
        current = hashlib.sha256(snapshot(t.model)).hexdigest()
        self.assertTrue({row[2] for row in t.opponent_assignments if row[1] == 'self-snapshots'} <= {current})
        t.update(buffer)
        spec = t.publish_snapshot(self.path / 'snapshots')
        t.collect()
        self.assertTrue({row[2] for row in t.opponent_assignments if row[1] == 'self-snapshots'} <= {spec.sha256})

    # Training -------------------------------------------------------------
    def test_updates_critics_estimators_and_trained_phases(self):
        for changes in (dict(), dict(critic_mode='centralized'), dict(return_estimator='monte-carlo'),
                        dict(critic_mode='centralized', return_estimator='monte-carlo')):
            with self.subTest(**changes):
                t = self.trainer(self.config(('neural-train', 'heuristic', 'neural-train'), **changes))
                buffer, _ = t.collect()
                self.assertEqual(len(buffer), 16 * (t.hands))
                self.assertEqual(t.model.trained_phases, ())
                metrics = t.update(buffer)
                self.assertIn('play_policy_loss', metrics)
                self.assertEqual(t.model.trained_phases, ('bid', 'play'))
                self.assertEqual(t.critic is not None, changes.get('critic_mode') == 'centralized')
        with self.assertRaises(ValueError):  # Scratch never needs oracle labels, but C still validates.
            self.config(critic_mode='centralized', oracle_aux_coef=.5)

    def test_worker_invariance(self):
        results = []
        for threads in (1, 2):
            t = self.trainer(self.config(threads=threads, pool=(('HHH', '1'), ('self-current', '1'))))
            t.update(t.collect()[0])
            t.update(t.collect()[0])
            results.append(([v.clone() for v in t.model.state_dict().values()], t.opponent_assignments))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(results[0][0], results[1][0])))
        self.assertEqual(results[0][1], results[1][1])

    # Checkpoints and evaluation -------------------------------------------
    def test_split_run_resume_reproducibility(self):
        config = self.config(critic_mode='centralized', pool=(('self-snapshots', '1'), ('HHH', '1')))
        continuous = self.trainer(config)
        continuous.update(continuous.collect()[0])
        continuous.publish_snapshot(self.path / 'snapshots')
        save(self.path / 'split.pt', continuous)
        continuous.update(continuous.collect()[0])
        resumed = self.trainer(config)
        resume(self.path / 'split.pt', resumed)
        self.assertEqual(resumed.snapshots, continuous.snapshots)
        resumed.update(resumed.collect()[0])
        for a, b in ((continuous.model, resumed.model), (continuous.critic, resumed.critic)):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))
        self.assertEqual(continuous.opponent_assignments, resumed.opponent_assignments)
        self.assertEqual((continuous.hands, continuous.optimizer_steps), (resumed.hands, resumed.optimizer_steps))

    def test_cli_examples_checkpoints_resume_new_stage_and_evaluation(self):
        from examples.evaluate_agent_vs_agent import main as versus
        from examples.evaluate_selfplay_ppo import main as evaluate
        examples = {
            'A': ['--learner-bid', 'neural-train', '--learner-trump', 'neural-train', '--learner-play', 'neural-train',
                  '--opponent-pool', 'HHH:1', '--eval-opponents', 'HHH,CHH', '--selection-opponent', 'HHH'],
            'B': ['--learner-bid', 'neural-train', '--learner-trump', 'neural-train', '--learner-play', 'neural-train',
                  '--opponent-pool', 'self-current:1', '--eval-opponents', 'HHH,CHH', '--selection-opponent', 'HHH'],
            'C': ['--learner-bid', 'neural-train', '--learner-trump', 'neural-train', '--learner-play', 'neural-train',
                  '--critic-mode', 'centralized', '--return-estimator', 'gae',
                  '--opponent-pool', 'HHH:0.3,CHH:0.2,self-current:0.5', '--eval-opponents', 'HHH,CHH',
                  '--selection-opponent', 'HHH'],
            'D': ['--learner-bid', 'neural-train', '--learner-trump', 'neural-train', '--learner-play', 'neural-train',
                  '--opponent-pool', 'self-snapshots:0.7,HHH:0.3', '--snapshot-every-hands', '4',
                  '--snapshot-pool-size', '8', '--eval-opponents', 'HHH,CHH', '--selection-opponent', 'HHH'],
            'E': ['--learner-bid', 'heuristic', '--learner-trump', 'heuristic', '--learner-play', 'neural-train',
                  '--opponent-pool', 'HHH:1', '--eval-opponents', 'HHH', '--selection-opponent', 'HHH'],
        }
        for name, flags in examples.items():
            with self.subTest(example=name):
                t = self.cli(name, *flags, '--threads', '2')
                self.assertGreater(t.updates, 0)
                for checkpoint in ('best', 'latest'):
                    model, data = load_model(self.path / name / 'checkpoints' / f'{checkpoint}.pt')
                    m = data['metadata']
                    self.assertEqual((m['initialization'], m['initialization_seed']), ('random', 42))
                    self.assertIsNone(m['initial_checkpoint_path'])
                    self.assertIsNone(m['initial_checkpoint_sha256'])
                    self.assertIsNone(m['original_play_sha256'])
                    self.assertIsNone(data['config']['anchor'])
                    self.assertTrue(all(s['initialization'] in ('random', 'none') and s['source'] is None
                                        for s in m['learner_phases']))
                progress = [json.loads(l) for l in (self.path / name / 'progress.jsonl').read_text().splitlines()]
                self.assertTrue(all(row['playing_anchor_kl'] is None for row in progress))
        snapshots = list((self.path / 'D' / 'snapshots').glob('policy-*.pt'))
        self.assertTrue(snapshots)
        for path in snapshots:  # Published only with trained-phase metadata.
            model, data = load_model(path)
            self.assertEqual(set(data['metadata']['trained_phases']), set(model.neural_phases))
        latest = self.path / 'B' / 'checkpoints' / 'latest.pt'
        resumed = quiet(train, ['--resume', str(latest), '--total-hands', '16', '--output-dir', str(self.path / 'B'),
                                '--log-dir', str(self.path / 'B-logs')])
        self.assertEqual(resumed.hands, 16)
        self.assertTrue(resumed.config.from_scratch)
        self.assertIsNone(resumed.config.anchor)
        # A new stage from the scratch checkpoint needs no play-only anchor.
        stage = quiet(train, ['--init-checkpoint', str(latest), '--learner-bid', 'neural-train',
                              '--learner-trump', 'neural-frozen', '--learner-play', 'neural-train',
                              '--opponent-pool', 'HHH:0.5,self-current:0.5', '--total-hands', '4',
                              '--rollout-decisions', '16', '--minibatch-size', '16', '--eval-deals', '2',
                              '--bootstrap-samples', '5', '--forgetting-hands', '1', '--device', 'cpu',
                              '--output-dir', str(self.path / 'stage'), '--log-dir', str(self.path / 'stage-logs')])
        self.assertFalse(stage.config.from_scratch)
        self.assertIsNone(stage.config.anchor)
        self.assertEqual(stage.config.snapshot_fallback, 'current')
        self.assertEqual(read(self.path / 'stage/checkpoints/latest.pt')['metadata']['initialization'], 'checkpoint')
        best = str(self.path / 'A' / 'checkpoints' / 'best.pt')
        reports = quiet(evaluate, ['--checkpoint', best, '--deals', '2', '--output', str(self.path / 'e.json'),
                                   '--log-dir', str(self.path / 'e-logs')])
        self.assertEqual(set(reports), {'HHH', 'CHH', 'HHR'})
        self.assertIsNone(json.loads((self.path / 'e.json').read_text())['anchor_id'])
        for flags in (['--opponents', 'HHH,anchor'], ['--anchor-checkpoint', best]):
            with self.assertRaises(SystemExit):
                quiet(evaluate, ['--checkpoint', best, '--deals', '2', *flags, '--log-dir', str(self.path / 'x')])
        report = quiet(versus, ['--agent-a-checkpoint', best, '--agent-a-bid', 'neural', '--agent-a-trump', 'neural',
                                '--agent-a-play', 'neural', '--agent-b-bid', 'heuristic', '--agent-b-trump', 'heuristic',
                                '--agent-b-play', 'heuristic', '--deals', '4', '--threads', '2',
                                '--log-dir', str(self.path / 'v-logs')])
        self.assertEqual(len(report.games), 8)


if __name__ == '__main__':
    unittest.main()
