"""Tiny deterministic phase, frozen-play, migration and historical-pool fixtures."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    from dataclasses import replace, asdict
    import tempfile
    from pathlib import Path
    import random
    import numpy as np
    import torch
    from unittest.mock import patch
    from research.ppo.trainer import Config, Trainer
    from research.ppo.checkpoint import save as save_legacy, load_model, read
    from research.ppo.selfplay_checkpoint import save, resume
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
    from research.ppo.multiphase import PhaseActorCritic, LearnerPhaseSpec, resolve_learner_phases, PHASES
    from research.ppo.opponents import PolicySpec, HistoricalPool, historical_pool, CheckpointIdentity, policy_registry
    from research.ppo.selfplay_evaluation import evaluate_phases, frozen_model, evaluation_diagnostics
    from research.ppo.imitation import clone_bidding, teacher_data, playing_kl
    from research.ppo.agent import PPOAgent
    from research.ppo.buffer import gae
    from research.ppo.seeds import stream_seed
    from texas42.agents import VARIANTS, Agent, HeuristicAgent
    from texas42.selfplay import SelfPlayEnv
    from texas42.encoding import encode


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class MultiPhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu'))
        self.anchor_path = self.path / 'anchor.pt'
        save_legacy(self.anchor_path, legacy)
        self.initial, _ = load_model(self.anchor_path)
        self.anchor = PolicySpec.checkpoint('anchor', self.anchor_path)
        self.policies = tuple(PolicySpec(n) for n in VARIANTS) + (self.anchor,)

    def config(self, **changes):
        c = SelfPlayConfig(total_hands=4, hidden_size=8, device='cpu', seed=42,
            rollout_decisions=4, ppo_epochs=2, minibatch_size=4, anchor=self.anchor,
            opponent_policies=self.policies, eval_deals=2, bootstrap_samples=5,
            forgetting_hands=1, snapshot_every_hands=2, snapshot_pool_size=2)
        return replace(c, **changes)

    def trainer(self, **changes):
        trainer = SelfPlayTrainer(self.config(**changes), self.initial)
        self.addCleanup(trainer.close)
        return trainer

    def phase_source(self, name='joint', trained=('bid', 'trump', 'play')):
        trainer = self.trainer(learn_phases=tuple(trained))
        # Explicit checkpoint metadata fixture for phase-loading tests; no strength claim.
        trainer.model.trained_phases = tuple(trained)
        path = self.path / f'{name}.pt'
        save(path, trainer)
        return path, CheckpointIdentity.from_path(path)

    def test_independent_phase_states_and_optimizer_membership(self):
        from itertools import product
        path, identity = self.phase_source()
        initial, _ = load_model(path)
        registry = policy_registry(self.anchor)
        for states in product(('heuristic', 'neural-frozen', 'neural-train'), repeat=3):
            if 'neural-train' not in states:
                with self.assertRaises(ValueError):
                    resolve_learner_phases(states, (None,) * 3, registry, identity)
                continue
            with self.subTest(states=states):
                specs = resolve_learner_phases(states, (None,) * 3, registry, identity)
                learn = tuple(p for p, state in zip(PHASES, states) if state == 'neural-train')
                t = SelfPlayTrainer(self.config(learner_phases=specs, learn_phases=learn,
                    total_hands=2, rollout_decisions=100, max_rollout_hands=2), initial)
                self.addCleanup(t.close)
                ids = {id(p) for group in t.optimizer.param_groups for p in group['params']}
                self.assertEqual(ids, {id(p) for phase in learn for p in t.model.networks[phase].parameters()})
                before = {p: {k:v.clone() for k,v in n.state_dict().items()} for p,n in t.model.networks.items()}
                b, _ = t.collect()
                self.assertEqual(t.hands, 2)
                self.assertTrue(all(PHASES[r.observation_features[:3].index(1.)] in learn for r in b.records))
                self.assertTrue(all(r.legal_action_mask[r.action_id] and r.action_id < 55 for r in b.records))
                if b:
                    metrics = t.update(b)
                    self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                for phase in PHASES:
                    if phase not in learn:
                        self.assertTrue(all(torch.equal(v, t.model.networks[phase].state_dict()[k])
                                            for k,v in before[phase].items()))

    def test_sources_freezing_new_initialization_and_architecture(self):
        path, identity = self.phase_source(trained=('bid','play'))
        registry = policy_registry(self.anchor, (f'archive={path}',))
        initial, _ = load_model(path)
        states = ('neural-frozen', 'neural-train', 'neural-frozen')
        specs = resolve_learner_phases(states, ('archive', 'new', 'anchor'), registry, identity)
        self.assertEqual(specs[1].initialization, 'representation')
        t = SelfPlayTrainer(self.config(learn_phases=('trump',), learner_phases=specs), initial)
        self.addCleanup(t.close)
        for phase, source in (('bid', initial.networks['bid']), ('play', self.initial)):
            self.assertTrue(all(torch.equal(v, source.state_dict()[k]) for k,v in t.model.networks[phase].state_dict().items()))
        self.assertNotIn('trump', t.model.trained_phases)
        self.assertTrue(all(torch.equal(v, initial.networks['play'].trunk.state_dict()[k])
                            for k,v in t.model.networks['trump'].trunk.state_dict().items()))
        for phase in ('bid', 'trump'):
            for state in ('neural-frozen', 'neural-train'):
                choices = ['heuristic', 'heuristic', 'neural-train']; choices[PHASES.index(phase)] = state
                sources = [None, None, 'anchor']; sources[PHASES.index(phase)] = 'anchor'
                with self.assertRaisesRegex(ValueError, 'no trained neural'):
                    resolve_learner_phases(choices, sources, registry, identity)
        with self.assertRaisesRegex(ValueError, 'no trained neural'):
            resolve_learner_phases(('neural-train','neural-frozen','neural-frozen'),
                                  (None,None,None), registry, identity)
        with self.assertRaisesRegex(ValueError, 'heuristic'):
            resolve_learner_phases(('heuristic','heuristic','neural-train'),
                                  ('archive',None,'anchor'), registry, identity)
        bad = replace(specs[0], source=replace(identity, sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'changed'):
            SelfPlayTrainer(self.config(learn_phases=('trump',), learner_phases=(bad,*specs[1:])), initial)

    def test_trump_only_empty_paths_terminal_rewards_and_budget(self):
        from research.ppo.selfplay_rollout import collect_phase_hand
        t = self.trainer(learn_phases=('trump',), total_hands=8, rollout_decisions=50, max_rollout_hands=3)
        transitions = []
        original = SelfPlayEnv.step
        def step(env, action):
            result = original(env, action)
            if result.terminated: transitions.append(result)
            return result
        rows = []
        with patch.object(SelfPlayEnv, 'step', step):
            for episode in range(12):
                rows.append(collect_phase_hand(t.model, 42, episode, ('trump',), t.opponent_pool()))
        self.assertTrue(any(not any(r['paths']) for r in rows))
        self.assertTrue(any(any(r['paths']) for r in rows))
        for row, transition in zip(rows, transitions):
            team = (row['episode']+42)%2
            records = [r for path in row['paths'] for r in path]
            self.assertEqual(len(records), int(row['bidder']%2 == team))
            self.assertEqual(row['decisions'], 33)
            self.assertEqual(sum(row['points']), 42)
            if records:
                r, = records
                self.assertEqual(r.seat, row['bidder'])
                self.assertTrue(r.done)
                self.assertEqual(r.reward, transition.rewards[r.seat])
                self.assertEqual(r.observation_features[:3], (0.,1.,0.))
        b, _ = t.collect()
        self.assertEqual(t.hands, 3)  # Explicit safeguard, independent of samples.
        self.assertLessEqual(len(b), 3)
        if b:
            before = {k:v.clone() for k,v in t.model.networks['trump'].state_dict().items()}
            t.update(b)
            self.assertTrue(any(not torch.equal(v,t.model.networks['trump'].state_dict()[k]) for k,v in before.items()))
        # A real hand whose opponent declares produces no fictitious terminal record.
        empty = next(r for r in rows if not any(r['paths']))
        t.hands = empty['episode']
        t.config = replace(t.config, total_hands=t.hands+1)
        b, _ = t.collect()
        self.assertEqual(len(b), 0)
        with self.assertRaisesRegex(ValueError, 'skip'):
            t.update(b)

    def test_trump_worker_invariance_resume_and_frozen_parameters(self):
        path, identity = self.phase_source(trained=('bid','play'))
        initial, _ = load_model(path)
        specs = resolve_learner_phases(('neural-frozen','neural-train','neural-frozen'),
                                      (None,None,None), policy_registry(self.anchor), identity)
        c = self.config(learner_phases=specs, learn_phases=('trump',), total_hands=16,
                        max_rollout_hands=8, rollout_decisions=4)
        records = metrics = weights = None
        for workers in (1,2,4,0):
            t = SelfPlayTrainer(replace(c, threads=workers), initial); self.addCleanup(t.close)
            if workers in (1,2,4): self.assertEqual(t.workers,workers)
            b, _ = t.collect()
            self.assertGreater(len(b),0)
            current = (b.records.copy(), b.advantages.copy(), t.opponent_assignments.copy(), t.hands)
            m = t.update(b)
            if records is not None:
                self.assertEqual(current,records); self.assertEqual(m,metrics)
                self.assertTrue(all(torch.equal(v,t.model.state_dict()[k]) for k,v in weights.items()))
            records, metrics = current,m
            weights = {k:v.clone() for k,v in t.model.state_dict().items()}
            for phase in ('bid','play'):
                self.assertTrue(all(torch.equal(v,initial.networks[phase].state_dict()[k])
                                    for k,v in t.model.networks[phase].state_dict().items()))
            t.close()
        saved = self.path/'trump.pt'; save(saved,t)
        new_initial, data = load_model(saved)
        self.assertEqual(data['metadata']['learn_phases'], ('trump',))
        self.assertEqual(data['metadata']['learner_phases'][1]['initialization'], 'representation')
        t.config = replace(t.config, total_hands=24)
        b,_=t.collect(); expected=b.records.copy(); m=t.update(b)
        r=SelfPlayTrainer(replace(c,total_hands=24,threads=2),new_initial); self.addCleanup(r.close)
        resume(saved,r)
        b,_=r.collect(); self.assertEqual(b.records,expected); self.assertEqual(r.update(b),m)
        self.assertTrue(all(torch.equal(v,r.model.state_dict()[k]) for k,v in t.model.state_dict().items()))
        changed = replace(specs[2], source=CheckpointIdentity(self.anchor.path,self.anchor.sha256))
        wrong = SelfPlayTrainer(replace(c,learner_phases=(*specs[:2],changed)),new_initial)
        self.addCleanup(wrong.close)
        with self.assertRaisesRegex(ValueError,'mismatch'): resume(saved,wrong)

    def test_composed_opponents_and_snapshot_phase_sources(self):
        import json
        path, identity = self.phase_source()
        definition = self.path/'opponent.json'
        definition.write_text(json.dumps(dict(bid=dict(strategy='neural',source='archive'),
            trump=dict(strategy='heuristic'),play=dict(strategy='neural',source='anchor'))))
        registry=policy_registry(self.anchor,(f'archive={path}',),compositions=(f'mix={definition}',))
        spec=registry['mix']; factory=spec.factory(); factory.prepare()
        a,b=factory(0,1),factory(2,2)
        self.assertIs(a.delegates['play'].network,b.delegates['play'].network)
        self.assertIsNot(a.delegates['play'],b.delegates['play'])
        self.assertTrue(all(not p.requires_grad for p in a.delegates['bid'].network.parameters()))
        t=self.trainer(opponent_policies=tuple(registry.values()),opponent_probabilities=(('mix','1'),))
        buffer,_=t.collect(); t.update(buffer)
        snapshot=t.publish_snapshot(self.path/'snapshots')
        registry=policy_registry(self.anchor,(f'archive={snapshot.path}',),compositions=(f'mix={definition}',))
        registry['mix'].factory().prepare()
        saved=self.path/'composed.pt'; save(saved,t)
        model,data=load_model(saved)
        from research.ppo.selfplay_checkpoint import config_from_dict
        c=config_from_dict(data['config'])
        r=SelfPlayTrainer(c,model); self.addCleanup(r.close); resume(saved,r)
        self.assertEqual(r.config.opponent_policies,t.config.opponent_policies)
        self.assertEqual(r.snapshots,t.snapshots)
        for threads in (1,2):
            report=evaluate_phases(t.model,(spec,),deals=2,bootstrap_samples=5,threads=threads)
            if threads==1: expected=report
            else: self.assertEqual(report,expected)

    def test_generalized_cli_trump_resume_compositions_and_conflicts(self):
        import contextlib
        import io
        import json
        from examples.train_selfplay_ppo import main
        path, identity = self.phase_source(trained=('bid','play'))
        definition = self.path/'pretrump.json'
        definition.write_text(json.dumps(dict(bid=dict(strategy='neural',source='pretrump'),
            trump=dict(strategy='heuristic'),play=dict(strategy='neural',source='pretrump'))))
        directory=self.path/'trump-cli'
        args=['--init-checkpoint',str(path),'--learner-bid','neural-frozen',
            '--learner-trump','neural-train','--learner-play','neural-frozen',
            '--opponent-checkpoint',f'pretrump={path}','--opponent-composition',f'composed={definition}',
            '--opponent-pool','composed:0.7,HHH:0.3','--eval-opponents','composed,HHH,anchor',
            '--selection-opponent','composed','--total-hands','8','--rollout-decisions','2',
            '--max-rollout-hands','4','--ppo-epochs','1','--minibatch-size','4','--eval-deals','2',
            '--bootstrap-samples','5','--forgetting-hands','1','--device','cpu',
            '--output-dir',str(directory),'--log-dir',str(self.path/'trump-logs')]
        with contextlib.redirect_stdout(io.StringIO()): t=main(args)
        self.assertGreater(t.updates,0)
        self.assertEqual(t.config.learn_phases,('trump',))
        latest=directory/'checkpoints/latest.pt'
        for phase in ('bid','play'):
            source,_=load_model(path)
            self.assertTrue(all(torch.equal(v,source.networks[phase].state_dict()[k])
                                for k,v in t.model.networks[phase].state_dict().items()))
        progress=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()]
        self.assertEqual(set(progress[-1]['evaluations']),{'composed','HHH','anchor'})
        self.assertTrue(all(all(row['frozen_phases_verified'].values()) for row in progress))
        self.assertEqual(read(directory/'checkpoints/best.pt')['metadata']['checkpoint_selection_opponent'],'composed')
        with contextlib.redirect_stdout(io.StringIO()):
            r=main(['--resume',str(latest),'--total-hands','12','--threads','2',
                '--learner-trump-source','init','--output-dir',str(self.path/'trump-resume'),
                '--log-dir',str(self.path/'trump-resume-logs')])
        self.assertEqual(r.hands,12)
        self.assertEqual(r.config.learner_phases,t.config.learner_phases)
        for flags in (['--learn-phases','bid'], ['--play-policy','trainable'],
                      ['--trump-policy','heuristic'], ['--learner-bid-source','anchor']):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ['--output-dir',str(self.path/'invalid'),*flags])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--resume',str(self.anchor_path)])

    def test_zero_sample_cli_skips_optimizer_and_saves_resumable_checkpoint(self):
        import contextlib
        import io
        from examples.train_selfplay_ppo import main
        from research.ppo.selfplay_rollout import collect_phase_hand
        # Find a deterministic first hand declared by the opponent with heuristic bidding.
        seed = next(seed for seed in range(32) if not any(collect_phase_hand(
            self.initial,seed,0,('trump',),HistoricalPool((('HHH','1'),),self.policies,self.anchor))['paths']))
        args=['--init-play-checkpoint',str(self.anchor_path),'--learner-bid','heuristic',
            '--learner-trump','neural-train','--learner-play','heuristic',
            '--opponent-pool','HHH:1','--total-hands','1','--seed',str(seed),
            '--eval-deals','1','--bootstrap-samples','2','--forgetting-hands','1',
            '--output-dir',str(self.path/'empty'),'--log-dir',str(self.path/'empty-logs')]
        with contextlib.redirect_stdout(io.StringIO()), patch.object(SelfPlayTrainer,'update',
                side_effect=AssertionError('Empty buffer must not reach PPO update')):
            t=main(args)
        self.assertEqual(t.samples,0); self.assertEqual(t.updates,0); self.assertEqual(t.hands,1)
        model,data=load_model(self.path/'empty/checkpoints/latest.pt')
        self.assertEqual(model.trained_phases,())
        self.assertFalse(data['optimizer']['state'])
        from research.ppo.selfplay_checkpoint import config_from_dict
        resumed=SelfPlayTrainer(replace(config_from_dict(data['config']),total_hands=2),model)
        self.addCleanup(resumed.close); resume(self.path/'empty/checkpoints/latest.pt',resumed)
        self.assertEqual(resumed.hands,1)

    def test_exact_transfer_phase_routing_and_frozen_play(self):
        t = self.trainer()
        env = SelfPlayEnv(seed=1); obs = env.reset()
        x = torch.tensor([encode(obs)])
        bid, _ = t.model(x)
        self.assertTrue(torch.equal(bid, t.model.networks['bid'](x)[0]))
        x[:, :3] = torch.tensor([0., 0., 1.])
        self.assertTrue(torch.equal(t.model(x)[0], self.initial(x)[0]))
        self.assertTrue(torch.equal(t.model(x)[1], self.initial(x)[1]))
        before = {k: v.clone() for k, v in t.model.networks['play'].state_dict().items()}
        bid_before = {k: v.clone() for k, v in t.model.networks['bid'].state_dict().items()}
        for _ in range(2):
            buffer, _ = t.collect()
            self.assertEqual(len(buffer), 4)
            self.assertTrue(all(r.done and r.action_id >= 28 for r in buffer.records))
            metrics = t.update(buffer)
            self.assertIn('bid_entropy', metrics)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(v, bid_before[k]) for k, v in t.model.networks['bid'].state_dict().items()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in t.model.networks['play'].state_dict().items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in t.model.networks['play'].parameters()))
        data = teacher_data(8, 1, anchor=self.initial)
        self.assertEqual(playing_kl(t.model, self.initial, data), 0.)

    def test_bid_legality_forced_pass_and_heuristic_trump_privacy(self):
        t = self.trainer()
        agent = PPOAgent(t.model)
        self.assertIsInstance(agent, Agent)
        for forced in (False, True):
            env = SelfPlayEnv(seed=3); obs = env.reset()
            if forced:
                for _ in range(3):
                    obs = env.step(28).observation
                self.assertTrue(obs.view.forced)
                self.assertNotIn(28, obs.legal_action_ids)
            agent.reset(obs.seat)
            action = agent.act(obs)
            self.assertIs(type(action), int)
            self.assertIn(action, obs.legal_action_ids)
            self.assertLess(action, 55)
        env = SelfPlayEnv(seed=5); obs = env.reset(); decisions = 0
        agents = [PPOAgent(t.model) for _ in range(4)]
        for s, a in enumerate(agents): a.reset(s)
        self.assertEqual(len({id(a) for a in agents}), 4)
        while True:
            self.assertFalse(hasattr(obs, 'state')); self.assertFalse(hasattr(obs.view, 'hands'))
            a = agents[obs.seat].act(obs)
            if obs.phase == 'trump':
                h = HeuristicAgent(); h.reset(obs.seat)
                self.assertEqual(a, h.act(obs))
            result = env.step(a); decisions += 1
            if result.terminated: break
            obs = result.observation
        self.assertEqual(decisions, 33)
        self.assertEqual(sum(result.hand_result.team_points), 42)

    def test_joint_trajectories_rewards_and_gae(self):
        t = self.trainer(learn_phases=('bid', 'play'), rollout_decisions=32)
        transitions = []
        original = SelfPlayEnv.step
        def step(env, action):
            result = original(env, action)
            if result.terminated: transitions.append(result)
            return result
        with patch.object(SelfPlayEnv, 'step', step): buffer, _ = t.collect()
        self.assertEqual(len(buffer), 32)
        self.assertEqual(t.phase_counts, {'bid': 4, 'play': 28})
        for episode, result in enumerate(transitions):
            team = (42 + episode) % 2
            for seat in (team, team + 2):
                rows = [r for r in buffer.records if (r.episode_id, r.seat) == (episode, seat)]
                self.assertEqual(len(rows), 8)
                self.assertEqual(rows[0].observation_features[:3], (1., 0., 0.))
                self.assertEqual([r.reward for r in rows], [0.] * 7 + [float(result.rewards[seat])])
                self.assertEqual([r.done for r in rows], [False] * 7 + [True])
                self.assertTrue(all(r.legal_action_mask[r.action_id] for r in rows))
        before = {p: [v.clone() for v in t.model.networks[p].parameters()] for p in ('bid', 'play')}
        metrics = t.update(buffer)
        for p in before:
            self.assertIn(p + '_policy_loss', metrics)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before[p], t.model.networks[p].parameters())))
        a, returns = gae([0., 2.], [.5, 1.], 1., .95)
        self.assertEqual(a, [1.45, 1.]); self.assertEqual(returns, [1.95, 2.])
        self.assertEqual(gae([3.], [.25]), ([2.75], [3.]))

    def test_optional_trump_variable_counts_and_budget(self):
        t = self.trainer(total_hands=3, learn_phases=('bid', 'trump', 'play'), rollout_decisions=200)
        b, _ = t.collect()
        self.assertEqual(t.hands, 3)
        self.assertEqual(t.phase_counts['bid'], 6)
        self.assertEqual(t.phase_counts['play'], 42)
        self.assertTrue(48 <= len(b) <= 51)
        self.assertTrue(all(r.action_id < 55 for r in b.records))
        t.update(b)
        with self.assertRaises(ValueError): t.collect()

    def test_all_variants_and_anchor_fixed_frozen_opponents(self):
        for name in (*VARIANTS, 'anchor'):
            t = self.trainer(total_hands=1, rollout_decisions=2, opponent_probabilities=((name, '1'),))
            b, _ = t.collect()
            self.assertEqual(len(b), 2)
            self.assertEqual(t.opponent_assignments[0][1], name)
        model = frozen_model(self.anchor)
        self.assertIs(model, frozen_model(self.anchor))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        bad = replace(self.anchor, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'changed'): frozen_model(bad)

    def test_pool_validation_and_determinism(self):
        pool = historical_pool('HHH:7,anchor:2,self-snapshots:1', self.policies, self.anchor)
        ratio = historical_pool('self-snapshots:0.1,anchor:0.2,HHH:0.7', self.policies, self.anchor)
        self.assertEqual(pool.probabilities, ratio.probabilities)
        state = random.getstate()
        self.assertEqual([pool.choose(1, i) for i in range(100)], [ratio.choose(1, i) for i in range(100)])
        self.assertEqual(state, random.getstate())
        fallback = HistoricalPool((('self-snapshots', '1'),), self.policies, self.anchor)
        self.assertEqual(fallback.choose(1, 0), ('self-snapshots', self.anchor))
        for bad in ('', 'HHH:0', 'HHH:nan', 'HHH:1,HHH:2', 'unknown:1', 'HHH:-1'):
            with self.assertRaises(ValueError): historical_pool(bad, self.policies, self.anchor)
        # Trump-only is now a valid zero-to-one-sample rollout configuration.
        self.assertEqual(self.config(learn_phases=('trump',)).learn_phases, ('trump',))
        for bad in ({'learn_phases': ()}, {'snapshot_pool_size': 0}, {'primary_opponent': 'unknown'}):
            with self.assertRaises(ValueError): self.config(**bad)

    def test_shared_checkpoint_capture_restores_every_rng_stream(self):
        from research.ppo.checkpoint import resume as resume_legacy
        def draws(trainer):
            return (random.random(), np.random.random(), torch.rand(3),
                    torch.rand(3, generator=trainer.sampling),
                    torch.rand(3, generator=trainer.shuffle))
        for version in (1, 2):
            config = Config(total_hands=4, hidden_size=8, device='cpu', rollout_play_steps=14)
            original = Trainer(config) if version == 1 else self.trainer()
            self.addCleanup(original.close)
            buffer, _ = original.collect(); original.update(buffer)
            path = self.path / f'version-{version}.pt'
            saver, restore = (save_legacy, resume_legacy) if version == 1 else (save, resume)
            # Saving must not deserialize a temporary legacy file or consume RNG.
            with patch('torch.load', side_effect=AssertionError('Unexpected checkpoint reload')):
                saver(path, original)
            expected = draws(original)
            continued = Trainer(config) if version == 1 else self.trainer()
            self.addCleanup(continued.close)
            restore(path, continued)
            actual = draws(continued)
            self.assertEqual(expected[:2], actual[:2])
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected[2:], actual[2:])))
            self.assertEqual(read(path)['metadata']['checkpoint_version'], version)
            self.assertFalse(path.with_name(path.name + '.capture').exists())

    def test_named_opponent_registry_preserves_anchor_and_archives(self):
        from research.ppo.opponents import policy_registry
        registry = policy_registry(self.anchor, (f'archive={self.anchor_path}',))
        self.assertEqual(tuple(registry), (*VARIANTS, 'anchor', 'archive'))
        self.assertIs(registry['anchor'], self.anchor)
        self.assertEqual(registry['archive'].sha256, self.anchor.sha256)
        for entries in ((f'HHH={self.anchor_path}',), (f'anchor={self.anchor_path}',),
                        (f'x={self.anchor_path}', f'x={self.anchor_path}')):
            with self.assertRaisesRegex(ValueError, 'name'):
                policy_registry(self.anchor, entries)
        # The historical pool reserves this label; standalone archive evaluation
        # historically permits it as an explicit checkpoint name.
        self.assertIn('self-snapshots', policy_registry(self.anchor, (f'self-snapshots={self.anchor_path}',)))
        with self.assertRaises(ValueError):
            policy_registry(self.anchor, (f'self-snapshots={self.anchor_path}',), reserved=('self-snapshots',))

    def test_worker_invariance_and_optimizer(self):
        c = self.config(learn_phases=('bid', 'play'), rollout_decisions=64,
            opponent_probabilities=(('HHH', '1'), ('anchor', '1'), ('self-snapshots', '1')))
        records = metrics = state = None
        for workers in (1, 2, 4, 0):
            t = SelfPlayTrainer(replace(c, threads=workers), self.initial)
            self.addCleanup(t.close)
            before = {k: v.clone() for k, v in t.model.state_dict().items()}
            b, _ = t.collect()
            self.assertEqual({r.episode_id for r in b.records}, set(range(4)))
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in t.model.state_dict().items()))
            current = (b.records.copy(), b.advantages.copy(), t.opponent_assignments.copy())
            outcome = t.update(b)
            current_state = {k: v.clone() for k, v in t.model.state_dict().items()}
            if records is not None:
                self.assertEqual(current, records); self.assertEqual(outcome, metrics)
                self.assertTrue(all(torch.equal(v, state[k]) for k, v in current_state.items()))
            records, metrics, state = current, outcome, current_state
            t.close(); self.assertIsNone(t.pool)

    def test_checkpoint_roster_exact_resume_and_new_stage(self):
        c = self.config(learn_phases=('bid', 'play'), rollout_decisions=32,
                        opponent_probabilities=(('self-snapshots', '1'),))
        t = SelfPlayTrainer(c, self.initial); self.addCleanup(t.close)
        b, _ = t.collect(); t.update(b)
        snapshot = t.publish_snapshot(self.path / 'snapshots')
        path = self.path / 'latest.pt'; save(path, t)
        initial, data = load_model(path)
        self.assertEqual(data['metadata']['original_play_sha256'], self.anchor.sha256)
        self.assertEqual(data['metadata']['checkpoint_version'], 2)
        b, _ = t.collect(); expected = (b.records.copy(), t.opponent_assignments.copy())
        metrics = t.update(b)
        r = SelfPlayTrainer(replace(c, threads=2), initial); self.addCleanup(r.close)
        resume(path, r)
        self.assertEqual(r.snapshots, [snapshot])
        b, _ = r.collect(); self.assertEqual((b.records, r.opponent_assignments), expected)
        self.assertEqual(metrics, r.update(b))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(t.model.parameters(), r.model.parameters())))
        for changes in ({'learn_phases': ('bid',)}, {'learning_rate': .01}, {'opponent_probabilities': (('HHH', '1'),)}):
            other = SelfPlayTrainer(replace(c, **changes), initial)
            with self.assertRaisesRegex(ValueError, 'mismatch'): resume(path, other)
        warm = SelfPlayTrainer(replace(c, learn_phases=('bid',)), initial)
        self.assertEqual(warm.hands, 0); self.assertEqual(len(warm.optimizer.state), 0)
        self.assertEqual(warm.snapshots, [])
        self.assertTrue(all(not p.requires_grad for p in warm.model.networks['play'].parameters()))
        # Inference reconstructs the saved neural routing without training metadata guesswork.
        opponent = PolicySpec.checkpoint('historical', path)
        self.assertEqual(frozen_model(opponent).neural_phases, ('bid', 'play'))
        for key in ('ruleset_id', 'encoder_layout_version', 'architecture'):
            bad = dict(data, metadata=dict(data['metadata'], **{key: 'bad'}))
            invalid = self.path / 'invalid.pt'; torch.save(bad, invalid)
            with self.assertRaises(ValueError): load_model(invalid)

    def test_behavior_cloning_separate_seed_frozen_play(self):
        t = self.trainer()
        play = {k: v.clone() for k, v in t.model.networks['play'].state_dict().items()}
        sampling, shuffle = t.sampling.get_state(), t.shuffle.get_state()
        data = teacher_data(42, 16)
        self.assertTrue(data['forced'].any()); self.assertTrue((~data['forced']).any())
        self.assertTrue(all(bool(m[a]) for m, a in zip(data['masks'], data['actions'])))
        metrics = clone_bidding(t, 16)
        self.assertEqual(metrics['samples'], 64)
        self.assertIsNotNone(metrics['forced_accuracy']); self.assertIsNotNone(metrics['voluntary_accuracy'])
        self.assertTrue(all(torch.equal(v, play[k]) for k, v in t.model.networks['play'].state_dict().items()))
        self.assertTrue(torch.equal(sampling, t.sampling.get_state()))
        self.assertTrue(torch.equal(shuffle, t.shuffle.get_state()))
        self.assertFalse(t.optimizer.state)

    def test_arena_equivalence_privacy_and_training_rng(self):
        t = self.trainer(); policies = tuple(PolicySpec(p) for p in VARIANTS) + (self.anchor,)
        python, numpy, cpu, shuffle = random.getstate(), np.random.get_state(), torch.get_rng_state(), t.shuffle.get_state()
        sequential = evaluate_phases(t.model, policies, deals=2, bootstrap_samples=5, seed=7)
        parallel = evaluate_phases(t.model, (self.anchor,), deals=2, bootstrap_samples=5, seed=7, threads=2)
        self.assertEqual(sequential['anchor'], parallel['anchor'])
        self.assertEqual(random.getstate(), python)
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(cpu, torch.get_rng_state())); self.assertTrue(torch.equal(shuffle, t.shuffle.get_state()))
        for report in sequential.values():
            self.assertEqual(report.games[0].seating, ('A', 'B', 'A', 'B'))
            self.assertEqual(report.games[1].seating, ('B', 'A', 'B', 'A'))
            self.assertEqual(report.games[0].schedule_ids, report.games[1].schedule_ids)
            self.assertTrue(all(h.decisions == 33 and sum(h.result.team_points) == 42 for g in report.games for h in g.hands))
            details = evaluation_diagnostics(report)
            self.assertEqual(details['A']['bid_calls'] + details['A']['pass_calls'], 8)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(t.model.networks['play'].parameters(), frozen_model(self.anchor).parameters())))

    def test_bounded_immutable_snapshots_and_worker_failure(self):
        t = self.trainer(total_hands=6, learn_phases=('bid', 'play'), rollout_decisions=32)
        published = []
        for _ in range(3):
            b, _ = t.collect(); t.update(b)
            spec = t.publish_snapshot(self.path / 'pool')
            published.append(spec)
        self.assertEqual(t.snapshots, published[-2:])
        self.assertTrue(Path(published[0].path).exists())  # Archives retain old resume dependencies.
        for spec in published:
            self.assertEqual(PolicySpec.checkpoint(spec.name, spec.path), spec)
        with self.assertRaisesRegex(ValueError, 'Immutable'): t.publish_snapshot(self.path / 'pool')
        bad = replace(self.anchor, name='bad', sha256='0' * 64)
        failing = self.trainer(threads=2, opponent_policies=(*self.policies, bad),
                              opponent_probabilities=(('bad', '1'),))
        with self.assertRaisesRegex(RuntimeError, 'episode='):
            failing.collect()
        self.assertIsNone(failing.pool)

    def test_cli_primary_selection_and_stage_warm_start(self):
        import contextlib
        import io
        import json
        from examples.train_selfplay_ppo import main
        directory = self.path / 'cli'
        from research.ppo.selfplay_evaluation import evaluate_phases as original_evaluate
        def controlled_selection(*args, **kwargs):
            # Explicit synthetic selection fixture: non-primary must not win best.pt.
            reports = original_evaluate(*args, **kwargs)
            return {name: replace(report, metrics=(replace(report.metrics[0],
                mean_signed_marks=-.25 if name == 'anchor' else .75), report.metrics[1]))
                for name, report in reports.items()}
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'examples.train_selfplay_ppo.evaluate_phases', side_effect=controlled_selection):
            trained = main(['--init-play-checkpoint', str(self.anchor_path), '--total-hands', '2',
                '--rollout-decisions', '4', '--ppo-epochs', '1', '--minibatch-size', '4',
                '--eval-deals', '2', '--bootstrap-samples', '5', '--forgetting-hands', '1',
                '--eval-opponents', 'HHH,anchor', '--selection-opponent', 'anchor',
                '--output-dir', str(directory), '--log-dir', str(self.path / 'logs')])
        progress = [json.loads(line) for line in (directory / 'progress.jsonl').read_text().splitlines()]
        score = progress[-1]['evaluations']['anchor']['metrics'][0]['mean_signed_marks']
        best = read(directory / 'checkpoints/best.pt')
        self.assertEqual(best['progress']['best_score'], score)
        self.assertEqual(score, -.25)
        self.assertEqual(best['metadata']['checkpoint_selection_opponent'], 'anchor')
        self.assertTrue((self.path / 'logs/training.csv').exists())
        initial, _ = load_model(directory / 'checkpoints/best.pt')
        stage_b = SelfPlayTrainer(self.config(learn_phases=('bid', 'play')), initial)
        self.assertEqual(stage_b.hands, 0); self.assertFalse(stage_b.optimizer.state)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(initial.parameters(), stage_b.model.parameters())))
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = main(['--resume', str(directory / 'checkpoints/latest.pt'), '--total-hands', '4',
                '--threads', '2', '--output-dir', str(self.path / 'resumed'), '--log-dir', str(self.path / 'resume-log')])
        self.assertEqual(resumed.hands, 4)
        self.assertEqual(resumed.config.learn_phases, ('bid',))

    def test_historical_rounds_remain_worker_invariant(self):
        c = self.config(learn_phases=('bid', 'play'), rollout_decisions=32,
                        opponent_probabilities=(('self-snapshots', '1'),))
        expected = weights = None
        for workers in (1, 2):
            t = SelfPlayTrainer(replace(c, threads=workers), self.initial)
            self.addCleanup(t.close)
            b, _ = t.collect(); t.update(b)
            t.publish_snapshot(self.path / f'historical-{workers}')
            b, _ = t.collect()
            observations = (b.records.copy(), b.advantages.copy())
            metrics = t.update(b)
            if expected is not None:
                self.assertEqual((observations, metrics), expected)
                self.assertTrue(all(torch.equal(v, weights[k]) for k, v in t.model.state_dict().items()))
            expected = observations, metrics
            weights = {k: v.clone() for k, v in t.model.state_dict().items()}
            t.close()
