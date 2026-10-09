"""Opponent pools, exact continuation, labelled validation, and spawn collection."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from research.ppo.opponents import (OpponentPool, parse_opponents, hand_agents,
                                   select_primary, evaluation_opponents)
from research.ppo.seeds import stream_seed
from texas42.agents import VARIANTS, HeuristicAgent, variant_factory


class OpponentConfigurationTests(unittest.TestCase):
    def test_canonical_and_fixed(self):
        expected = parse_opponents('HHH:7,CHH:2,HHR:1')
        self.assertEqual(expected, parse_opponents('HHR:0.1,CHH:0.2,HHH:0.7'))
        self.assertEqual(expected, OpponentPool(json.loads(json.dumps(expected))).probabilities)
        self.assertEqual(OpponentPool().probabilities, (('HHH', '1'),))
        for name in VARIANTS:
            with patch('research.ppo.opponents.random.Random', side_effect=AssertionError('sampled fixed opponent')):
                self.assertEqual(OpponentPool(((name, 2),)).choose(42, 0), name)
        for episode in range(12):
            a = hand_agents('HHH', 42, episode, episode % 2)
            self.assertTrue(all(isinstance(agent, HeuristicAgent) for agent in a))
            self.assertEqual(len({id(agent) for agent in a}), 4)

    def test_invalid_and_primary(self):
        for text in ('', ' ', 'HHH', 'HHH:', ':1', 'UNKNOWN:1', 'HHH:1,HHH:2',
                     'HHH:0', 'HHH:-1', 'HHH:nan', 'HHH:inf', 'HHH:1,'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_opponents(text)
        self.assertEqual(select_primary(('CHH',)), 'CHH')
        self.assertEqual(select_primary(('HHR', 'HHH')), 'HHH')
        for names, primary in ((('CHH', 'HHR'), None), (('CHH',), 'HHH')):
            with self.assertRaises(ValueError): select_primary(names, primary)
        with self.assertRaises(ValueError): evaluation_opponents(('HHH', 'HHH'))

    def test_dedicated_selection_stream(self):
        pool = OpponentPool(parse_opponents('HHH:1,CHH:1,HHR:1'))
        state = random.getstate()
        names = [pool.choose(42, i) for i in range(100)]
        self.assertEqual(state, random.getstate())
        self.assertEqual(names, [pool.choose(42, i) for i in range(100)])
        self.assertEqual(set(names), {'HHH', 'CHH', 'HHR'})


TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import numpy as np
    import torch
    from research.ppo.trainer import Trainer, Config
    from research.ppo.rollout import collect_hand
    from research.ppo.checkpoint import save, resume, load_model, read
    from research.ppo.evaluation import evaluate_many
    from texas42.selfplay import SelfPlayEnv
    from examples.train_ppo import main


@unittest.skipUnless(TORCH, 'optional rl dependencies unavailable')
class OpponentTrainingTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(Config(total_hands=16, rollout_play_steps=112, hidden_size=16,
                              ppo_epochs=1, minibatch_size=32, device='cpu', sampling_scheme='episode-v1'), **kwargs)

    def test_each_variant_and_learning_policy(self):
        original_step = SelfPlayEnv.step
        for name in VARIANTS:
            with Trainer(self.config(total_hands=2, rollout_play_steps=28, train_opponents=((name, 1),))) as trainer:
                episode, decisions, completions = [0], [], []
                expected = {}
                for i in range(2):
                    for seat in range(4):
                        agent = variant_factory(name)(seat, stream_seed(42, 'training-opponent-agent-v1', i * 4 + seat))
                        agent.reset(seat); expected[i, seat] = agent
                def step(env, action):
                    obs = env.observe()
                    i = episode[0]
                    learning = obs.seat % 2 == (i + 42) % 2
                    self.assertFalse(hasattr(obs.view, 'hands'))
                    self.assertIn(action, obs.legal_action_ids)
                    if learning and obs.phase != 'play':
                        h = HeuristicAgent(); h.reset(obs.seat)
                        self.assertEqual(action, h.act(obs))
                    elif not learning:
                        self.assertEqual(action, expected[i, obs.seat].act(obs))
                    decisions.append(obs)
                    result = original_step(env, action)
                    if result.terminated:
                        completions.append(result); episode[0] += 1
                    return result
                before = [p.clone() for p in trainer.model.parameters()]
                with patch.object(SelfPlayEnv, 'step', step):
                    buffer, reward = trainer.collect()
                self.assertEqual(len(buffer), 28)
                self.assertEqual(len(decisions), 66)
                self.assertEqual(trainer.opponent_assignments, [(0, name), (1, name)])
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, trainer.model.parameters())))
                for i, result in enumerate(completions):
                    self.assertEqual(sum(result.hand_result.team_points), 42)
                    for seat in range(4):
                        path = [r for r in buffer.records if (r.episode_id, r.seat) == (i, seat)]
                        if seat % 2 != i % 2:
                            self.assertFalse(path); continue
                        self.assertEqual(len(path), 7)
                        self.assertEqual([r.reward for r in path], [0.] * 6 + [result.rewards[seat]])
                        self.assertEqual([r.done for r in path], [False] * 6 + [True])
                        self.assertTrue(all(r.legal_action_mask[r.action_id] for r in path))
                self.assertEqual(trainer.opponent_metrics[name]['mean_signed_reward'], reward)

    def test_default_path_and_same_deals(self):
        original_reset = SelfPlayEnv.reset
        deals = []
        def reset(env, **kwargs):
            obs = original_reset(env, **kwargs)
            deals.append(tuple(tuple(hand) for hand in env._session.hand.hands))
            return obs
        with Trainer(self.config(total_hands=1)) as trainer:
            rng = torch.Generator().manual_seed(7)
            with patch.object(SelfPlayEnv, 'reset', reset):
                baseline = collect_hand(trainer.model, rng, 42, 0)
            # Old construction plus the new default produces the same draws and records.
            with patch('research.ppo.rollout.hand_agents', side_effect=lambda *args: [self.heuristic(s) for s in range(4)]):
                expected = collect_hand(trainer.model, torch.Generator().manual_seed(7), 42, 0)
            self.assertEqual(baseline, expected)
            with patch.object(SelfPlayEnv, 'reset', reset):
                collect_hand(trainer.model, torch.Generator().manual_seed(7), 42, 0, 'RRR')
            self.assertEqual(deals[0], deals[1])

    def heuristic(self, seat):
        agent = HeuristicAgent(); agent.reset(seat); return agent

    def test_mixture_parallel_and_optimizer(self):
        reference = None
        config = self.config(train_opponents=parse_opponents('HHH:7,CHH:2,HHR:1'))
        for workers in (1, 2, 4, 8, 0):
            with Trainer(replace(config, threads=workers)) as trainer:
                buffer, reward = trainer.collect()
                self.assertEqual(len(buffer), 112)
                self.assertEqual(sorted({r.episode_id for r in buffer.records}), list(range(8)))
                self.assertEqual(trainer.opponent_assignments,
                                 [(i, trainer.opponents.choose(42, i)) for i in range(8)])
                counts = trainer.opponent_metrics
                self.assertEqual(sum(s['hands'] for s in counts.values()), 8)
                self.assertEqual(sum(s['reward_sum'] for s in counts.values()) / 8, reward)
                records, advantages = buffer.records.copy(), buffer.advantages.copy()
                metrics = trainer.update(buffer)
                weights = [p.clone() for p in trainer.model.parameters()]
                actual = records, advantages, trainer.opponent_assignments, counts, metrics
                if reference is None:
                    reference, expected_weights = actual, weights
                else:
                    self.assertEqual(reference, actual)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(weights, expected_weights)))
            self.assertIsNone(trainer.pool)

    def test_resume_migration_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            c = self.config(train_opponents=parse_opponents('HHH:1,CHH:1'))
            with Trainer(replace(c, threads=2)) as trainer:
                b, _ = trainer.collect(); trainer.update(b); save(path, trainer)
                expected, _ = trainer.collect(); expected_records = expected.records.copy(); expected_metrics = trainer.update(expected)
                expected_weights = [p.clone() for p in trainer.model.parameters()]
            with Trainer(replace(c, threads=4)) as other:
                resume(path, other)
                actual, _ = other.collect()
                self.assertEqual(actual.records, expected_records)
                self.assertEqual(other.update(actual), expected_metrics)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected_weights, other.model.parameters())))
            with Trainer(self.config()) as other:
                with self.assertRaisesRegex(ValueError, 'train_opponents'):
                    resume(path, other)
                model, data = load_model(path)
                other.model.load_state_dict(model.state_dict())  # Intentional warm start, fresh optimizer/counters.
                self.assertEqual(other.hands, 0)
            with Trainer(self.config()) as trainer:
                b, _ = trainer.collect(); trainer.update(b); save(path, trainer)
                old = torch.load(path, weights_only=True)
                for key in ('train_opponents', 'eval_opponents', 'selection_opponent'):
                    old['config'].pop(key)
                for key in ('training_opponents', 'opponent_selection_scheme', 'checkpoint_selection_opponent'):
                    old['metadata'].pop(key)
                torch.save(old, path)
                expected, _ = trainer.collect(); records = expected.records.copy()
            with Trainer(self.config()) as other:
                resume(path, other)
                actual, _ = other.collect()
                self.assertEqual(records, actual.records)
            with Trainer(self.config(eval_opponents=('CHH',), selection_opponent='CHH')) as other:
                with self.assertRaisesRegex(ValueError, 'eval_opponents'):
                    resume(path, other)

    def test_multieval_rng_and_primary_export(self):
        with Trainer(self.config(total_hands=2)) as trainer:
            before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            sampling = trainer.sampling.get_state().clone()
            reports = evaluate_many(trainer.model, opponents=('HHH', 'CHH', 'HHR'), deals=2, bootstrap_samples=5)
            self.assertEqual(tuple(reports), ('HHH', 'CHH', 'HHR'))
            self.assertEqual([g.schedule_ids for g in reports['HHH'].games], [g.schedule_ids for g in reports['CHH'].games])
            self.assertEqual(random.getstate(), before[0])
            np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
            self.assertTrue(torch.equal(trainer.sampling.get_state(), sampling))
            parallel = evaluate_many(trainer.model, opponents=('HHH', 'CHH', 'HHR'), deals=2, bootstrap_samples=5, threads=2)
            self.assertEqual({k: v.to_dict() for k,v in reports.items()}, {k: v.to_dict() for k,v in parallel.items()})
        def distinct_selection_fixture(model, **kwargs):
            outcomes = evaluate_many(model, **kwargs)
            # Deliberately distinguish the primary score from an opponent with
            # a higher score, so using the wrong best-checkpoint criterion fails.
            return {name: replace(report, metrics=(replace(report.metrics[0],
                    mean_signed_marks=-0.25 if name == 'CHH' else 0.75), report.metrics[1]))
                    for name, report in outcomes.items()}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch('examples.train_ppo.evaluate_many', distinct_selection_fixture):
            trainer = main(['--total-hands','2','--rollout-play-steps','28','--hidden-size','16',
                            '--ppo-epochs','1','--minibatch-size','14','--eval-deals','2',
                            '--bootstrap-samples','5','--train-opponents','HHH:1,CHH:1',
                            '--eval-opponents','HHH,CHH,HHR','--selection-opponent','CHH',
                            '--output-dir', directory, '--log-dir', str(Path(directory)/'logs')])
            progress = json.loads((Path(directory)/'progress.jsonl').read_text().splitlines()[-1])
            self.assertEqual(progress['evaluation'], progress['evaluations']['CHH'])
            self.assertEqual(trainer.best_score, -0.25)
            self.assertEqual(trainer.best_score, progress['evaluations']['CHH']['metrics'][0]['mean_signed_marks'])
            self.assertEqual(read(Path(directory)/'checkpoints/best.pt')['config']['selection_opponent'], 'CHH')
            self.assertIn('eval_HHR_win_rate', (Path(directory)/'logs/training.csv').read_text())

    def test_standalone_multi_and_single_export(self):
        import csv
        from examples.evaluate_ppo import main as evaluate_cli
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            with Trainer(self.config()) as trainer:
                save(directory / 'model.pt', trainer)
            args = ['--checkpoint', str(directory / 'model.pt'), '--deals', '2',
                    '--bootstrap-samples', '5', '--log-dir', str(directory / 'logs')]
            multi = evaluate_cli(args + ['--opponents', 'HHH,CHH,HHR', '--output', str(directory / 'multi.json')])
            self.assertEqual(tuple(multi), ('HHH', 'CHH', 'HHR'))
            self.assertEqual(set(json.loads((directory / 'multi.json').read_text())['evaluations']), set(multi))
            with (directory / 'logs/multi.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 12)
            self.assertEqual({r['opponent'] for r in rows}, set(multi))
            self.assertTrue(all(r['policy_b'] == r['opponent'] for r in rows))
            single = evaluate_cli(args + ['--opponent', 'HHH', '--output', str(directory / 'single.json')])
            data = json.loads((directory / 'single.json').read_text())
            self.assertNotIn('evaluations', data)
            self.assertEqual(data['games_played'], single.games_played)

    def test_cli_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            for args in (['--train-opponent','HHH','--train-opponents','CHH:1'],
                         ['--eval-opponent','HHH','--eval-opponents','CHH']):
                with self.assertRaises(SystemExit):
                    main(args + ['--log-dir', directory])
