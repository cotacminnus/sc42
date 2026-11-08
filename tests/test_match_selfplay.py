"""Match-win training objective: complete-match rollouts, returns, selection, resume and pipelines."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import contextlib
    import io
    import json
    import subprocess
    import tempfile
    from dataclasses import replace
    from pathlib import Path
    from unittest import mock
    import torch
    from examples import run_pipeline as rp
    from examples.train_selfplay_ppo import main as train
    from research.ppo.buffer import Experience, RolloutBuffer
    from research.ppo.checkpoint import load_model, read
    from research.ppo.multiphase import LearnerPhaseSpec, PhaseActorCritic, PHASES
    from research.ppo.opponents import HistoricalPool, PolicySpec
    from research.ppo.selfplay_checkpoint import resume, save
    from research.ppo.selfplay_rollout import collect_phase_hand, collect_phase_match
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer, PairedSelfPlayTrainer
    from texas42.agents import VARIANTS
    from texas42.selfplay import SelfPlayEnv

REPOSITORY = Path(__file__).resolve().parents[1] if TORCH else None


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


class MatchEnvironmentTests(unittest.TestCase):
    """The reused engine: zero rewards at intermediate hands, +1/-1 at match end."""

    def test_intermediate_zero_rewards_terminal_signed_unit_and_shaker_rotation(self):
        import random
        from texas42.selfplay import SelfPlayEnv as Env
        for seed in range(5):
            env = Env(episode='match', target_marks=7, seed=seed)
            obs, rng, marks, shakers = env.reset(), random.Random(seed), [0, 0], [env.shaker]
            while True:
                step = env.step(rng.choice(obs.legal_action_ids))
                if step.hand_result is not None:
                    r = step.hand_result
                    marks[r.scoring_team] += r.marks_awarded
                    self.assertEqual(env.match_marks, tuple(marks))
                    if not step.terminated:
                        self.assertEqual(step.rewards, (0, 0, 0, 0))
                        shakers.append(env.shaker)
                elif not step.terminated:
                    self.assertEqual(step.rewards, (0, 0, 0, 0))
                if step.terminated:
                    break
                obs = step.observation
            self.assertGreaterEqual(max(marks), 7)
            winner = step.match_winner
            self.assertEqual(step.rewards, tuple(1 if s % 2 == winner else -1 for s in range(4)))
            self.assertEqual(shakers, [(shakers[0] + i) % 4 for i in range(len(shakers))])


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class MatchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.variants = tuple(PolicySpec(n) for n in VARIANTS)

    def config(self, states=('neural-train',) * 3, pool=(('HHH', '1'), ('self-current', '1')), **changes):
        specs = tuple(LearnerPhaseSpec('neural-train', None, 'random') if s == 'neural-train'
                      else LearnerPhaseSpec('heuristic', initialization='none') for s in states)
        c = SelfPlayConfig(total_hands=24, hidden_size=8, device='cpu', seed=42, from_scratch=True,
                           learner_phases=specs,
                           learn_phases=tuple(p for p, s in zip(PHASES, states) if s == 'neural-train'),
                           rollout_decisions=96, ppo_epochs=2, minibatch_size=16, anchor=None,
                           opponent_policies=self.variants, opponent_probabilities=pool,
                           evaluation_names=('HHH', 'CHH'), primary_opponent='HHH', eval_deals=2,
                           bootstrap_samples=5, forgetting_hands=1, reward_objective='match-win')
        return replace(c, **changes)

    def trainer(self, config, cls=None):
        trainer = (cls or SelfPlayTrainer)(config)
        self.addCleanup(trainer.close)
        return trainer

    def pool(self, *names):
        return HistoricalPool(tuple((n, '1') for n in names), self.variants, None, snapshot_fallback='current')

    def model(self, neural=PHASES):
        torch.manual_seed(0)
        return PhaseActorCritic(8, neural).eval()

    def cli(self, name, *extra, total='24'):
        args = ['--from-scratch', '--reward-objective', 'match-win', '--total-hands', total,
                '--rollout-decisions', '96', '--hidden-size', '8', '--minibatch-size', '16', '--ppo-epochs', '1',
                '--eval-deals', '2', '--eval-every-hands', '8', '--bootstrap-samples', '5',
                '--forgetting-hands', '1', '--device', 'cpu', '--seed', '42',
                '--opponent-pool', 'HHH:0.5,self-current:0.5', '--eval-opponents', 'HHH,CHH',
                '--selection-opponent', 'HHH', *extra,
                '--output-dir', str(self.path / name), '--log-dir', str(self.path / (name + '-logs'))]
        return quiet(train, args)

    def assert_match_paths(self, row):
        """Seat-separated trajectories: zero rewards except one terminal record per nonempty seat path."""
        terminal = row['reward']
        self.assertIn(terminal, (1, -1))
        seats = set()
        for path in row['paths']:
            if not path:
                continue
            seat = path[0].seat
            seats.add(seat)
            self.assertTrue(all(r.seat == seat and r.episode_id == row['episode'] for r in path))
            self.assertTrue(all(not r.done and r.reward == 0.0 for r in path[:-1]))
            self.assertTrue(path[-1].done)
            self.assertEqual(path[-1].reward, float(terminal))  # Both partners share the team outcome.
        self.assertLessEqual(len({s % 2 for s in seats}), 1)  # Never mixes partnerships.


class ConfigTests(MatchTestCase):
    def test_defaults_and_validation(self):
        default = SelfPlayConfig(anchor=None, opponent_policies=self.variants, opponent_probabilities=(('HHH', '1'),),
                                 evaluation_names=('HHH',), primary_opponent='HHH')
        self.assertEqual((default.reward_objective, default.target_marks), ('hand-marks', 7))
        self.assertFalse(default.match_objective)
        for changes, message in ((dict(reward_objective='marks'), 'reward_objective'),
                                 (dict(target_marks=0), 'target_marks'),
                                 (dict(target_marks=7.0), 'target_marks'),
                                 (dict(search_fraction=0.5), 'Search inside training'),
                                 (dict(reward_objective='hand-marks', target_marks=5), 'match-win')):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.config(**changes)
        with self.assertRaisesRegex(ValueError, 'Oracle auxiliary'):
            self.config(critic_mode='centralized', oracle_aux_coef=0.1, oracle_dataset='x', oracle_dataset_sha256='y')
        self.assertEqual(self.config(target_marks=3).target_marks, 3)

    def test_cli_rejects_incompatible_combinations(self):
        for flags in (['--search-fraction', '0.1'], ['--target-marks', '0'], ['--reward-objective', 'win']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.cli('bad', *flags)
        with self.assertRaises(SystemExit):  # A target only applies to complete matches.
            quiet(train, ['--from-scratch', '--target-marks', '5', '--output-dir', str(self.path / 'x'),
                          '--log-dir', str(self.path / 'x-logs')])


class RolloutTests(MatchTestCase):
    def test_complete_match_trajectories(self):
        model = self.model()
        for match in range(4):
            with self.subTest(match=match):
                row = collect_phase_match(model, 42, match, PHASES, self.pool('HHH', 'self-current'), target_marks=7)
                self.assert_match_paths(row)
                self.assertGreaterEqual(row['hands'], 2)  # A hand awards at most five marks.
                self.assertLessEqual(row['hands'], 13)
                self.assertEqual(row['decisions'], 33 * row['hands'])
                learner, opponent = row['final_marks']
                self.assertEqual(sum(m for m in row['hand_marks'] if m > 0), learner)
                self.assertEqual(-sum(m for m in row['hand_marks'] if m < 0), opponent)
                self.assertEqual(row['reward'], 1 if learner >= 7 else -1)
                self.assertTrue(max(learner, opponent) >= 7 > min(learner, opponent))
                self.assertEqual(row['shakers'], tuple((row['shakers'][0] + i) % 4 for i in range(row['hands'])))
                self.assertEqual(len(row['bids']), row['hands'])
                for path in row['paths']:  # One bid and seven plays per hand, plus declared trumps.
                    self.assertGreaterEqual(len(path), 8 * row['hands'])
                    self.assertLessEqual(len(path), 9 * row['hands'])
        again = collect_phase_match(model, 42, 1, PHASES, self.pool('HHH', 'self-current'), target_marks=7)
        first = collect_phase_match(model, 42, 1, PHASES, self.pool('HHH', 'self-current'), target_marks=7)
        self.assertEqual(again['paths'], first['paths'])

    def test_hand_collector_unchanged_by_match_streams(self):
        model = self.model()
        hand = collect_phase_hand(model, 42, 0, PHASES, self.pool('HHH', 'self-current'))
        self.assertEqual(len(hand['bids']), 1)
        for path in hand['paths']:
            self.assertTrue(path[-1].done)
            self.assertEqual(path[-1].reward, float(hand['reward']))
            self.assertTrue(all(r.reward == 0 and not r.done for r in path[:-1]))

    def test_frozen_phases_and_short_target(self):
        model = self.model()
        row = collect_phase_match(model, 7, 0, ('play',), self.pool('HHH'), target_marks=2)
        self.assert_match_paths(row)
        for path in row['paths']:
            self.assertEqual(len(path), 7 * row['hands'])
            self.assertTrue(all(r.observation_features[2] == 1.0 for r in path))

    def test_trump_only_outcome_reaches_earlier_hand_and_empty_paths(self):
        model = self.model(('trump',))  # Heuristic bidding: the learner team does not always declare.
        found_earlier = found_empty = False
        for match in range(40):
            row = collect_phase_match(model, 3, match, ('trump',), self.pool('HHH'), target_marks=3)
            self.assert_match_paths(row)
            declared_last = row['bids'][-1][0]
            for path in row['paths']:
                if not path:
                    found_empty = True
                elif not declared_last:
                    # No learner trump decision in the final hand: the terminal outcome is
                    # attached to the seat's last record from an earlier hand.
                    found_earlier = True
                    self.assertEqual(path[-1].reward, float(row['reward']))
            if found_earlier and found_empty:
                break
        self.assertTrue(found_earlier and found_empty)


class ReturnTests(MatchTestCase):
    def test_returns_span_hand_boundaries(self):
        def record(i, reward=0.0, done=False, value=0.0):
            mask = (True,) + (False,) * 57
            return Experience((0.0,) * 3, mask, 0, 0.0, value, reward, done, 0, 5)
        # Three hands' worth of decisions; only the match end pays.
        values = [0.1, -0.2, 0.3, 0.0, 0.5]
        path = [record(i, value=v) for i, v in enumerate(values[:-1])] + [record(4, 1.0, True, values[-1])]
        mc = RolloutBuffer()
        mc.add_trajectory(path, 1.0, 0.95, estimator='monte-carlo')
        self.assertEqual(mc.returns, [1.0] * 5)
        self.assertEqual(mc.advantages, [1.0 - v for v in values])
        g = RolloutBuffer()
        g.add_trajectory(path, 1.0, 0.9, estimator='gae')
        expected, carry, following = [0.0] * 5, 0.0, 0.0
        for i in reversed(range(5)):
            delta = (1.0 if i == 4 else 0.0) + following - values[i]
            carry = delta + 0.9 * carry
            expected[i], following = carry, values[i]
        for a, b in zip(g.advantages, expected):
            self.assertAlmostEqual(a, b)
        discounted = RolloutBuffer()
        discounted.add_trajectory(path, 0.5, 0.95, estimator='monte-carlo')
        self.assertEqual(discounted.returns, [0.0625, 0.125, 0.25, 0.5, 1.0])
        with self.assertRaises(ValueError):  # An intermediate hand is never terminal.
            RolloutBuffer().add_trajectory([record(0, 0.0, True)] + path[1:], 1.0, 0.95)
        policy_only = RolloutBuffer()
        policy_only.add_trajectory(path, 1.0, 0.95, estimator='monte-carlo', baseline=False)
        self.assertEqual(policy_only.advantages, [1.0] * 5)

    def test_collected_match_returns_equal_outcome(self):
        row = collect_phase_match(self.model(), 42, 0, PHASES, self.pool('HHH'), target_marks=7)
        buffer = RolloutBuffer()
        for path in row['paths']:
            buffer.add_trajectory(path, 1.0, 0.95, estimator='monte-carlo')
        self.assertEqual(set(buffer.returns), {float(row['reward'])})


class TrainerTests(MatchTestCase):
    def test_collection_counters_and_budget(self):
        t = self.trainer(self.config(return_estimator='monte-carlo'))
        buffer, reward = t.collect()
        m = t.match_metrics
        self.assertEqual((t.matches, t.hands), (m['matches'], m['hands']))
        self.assertGreater(t.hands, t.matches)  # Multi-hand episodes.
        self.assertEqual(reward, m['mean_terminal_reward'])
        self.assertEqual(len(buffer), sum(t.phase_counts.values()))
        self.assertEqual(sum(r.done for r in buffer.records), len({(r.episode_id, r.seat) for r in buffer.records}))
        self.assertEqual({r.episode_id for r in buffer.records}, set(range(t.matches)))
        self.assertTrue(set(buffer.returns) <= {1.0, -1.0})  # gamma 1: every return is the match outcome.
        self.assertEqual(sum(sum(v.values()) for v in t.bid_metrics.values()), t.hands)
        self.assertEqual(sum(s['matches'] for s in t.opponent_metrics.values()), t.matches)
        t.update(buffer)
        while t.hands < t.config.total_hands:
            before = t.matches
            t.update(t.collect()[0])
            self.assertGreater(t.matches, before)
        self.assertLessEqual(t.hands, t.config.total_hands + 2 * t.config.target_marks - 2)
        with self.assertRaises(ValueError):
            t.collect()

    def test_critic_modes_paired_and_trump_only(self):
        for name, config, cls in (
                ('policy-only', self.config(critic_mode='none', return_estimator='monte-carlo'), None),
                ('centralized', self.config(critic_mode='centralized'), None),
                ('paired', self.config(partner_mode='paired', total_hands=12), PairedSelfPlayTrainer),
                ('trump-only', self.config(('heuristic', 'neural-train', 'heuristic'), rollout_decisions=4,
                                           max_rollout_hands=16, total_hands=16), None)):
            with self.subTest(name=name):
                t = self.trainer(config, cls)
                buffer, _ = t.collect()
                samples = sum(len(b) for b in buffer) if cls else len(buffer)
                if samples:
                    t.update(buffer)
                self.assertGreater(t.matches, 0)
                if cls:
                    self.assertEqual((t.partner.hands, t.partner.matches), (t.hands, t.matches))

    def test_worker_invariance(self):
        results = []
        for threads in (1, 2):
            t = self.trainer(self.config(threads=threads, critic_mode='centralized'))
            t.update(t.collect()[0])
            t.update(t.collect()[0])
            results.append(([v.clone() for v in t.model.state_dict().values()], t.opponent_assignments,
                            t.hands, t.matches, t.match_metrics))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(results[0][0], results[1][0])))
        self.assertEqual(results[0][1:], results[1][1:])

    def test_exact_resume_matches_uninterrupted(self):
        config = self.config(pool=(('self-snapshots', '1'), ('HHH', '1')), snapshot_every_hands=4)
        continuous = self.trainer(config)
        continuous.update(continuous.collect()[0])
        continuous.publish_snapshot(self.path / 'snapshots')
        save(self.path / 'split.pt', continuous)
        continuous.update(continuous.collect()[0])
        resumed = self.trainer(config)
        resume(self.path / 'split.pt', resumed)
        self.assertEqual(resumed.matches, read(self.path / 'split.pt')['progress']['matches'])
        resumed.update(resumed.collect()[0])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(continuous.model.state_dict().values(),
                                                              resumed.model.state_dict().values())))
        self.assertEqual(continuous.opponent_assignments, resumed.opponent_assignments)
        self.assertEqual((continuous.hands, continuous.matches, continuous.optimizer_steps),
                         (resumed.hands, resumed.matches, resumed.optimizer_steps))
        for changes in (dict(reward_objective='hand-marks', target_marks=7), dict(target_marks=5)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'mismatch'):
                resume(self.path / 'split.pt', self.trainer(replace(config, **changes)))


class CommandTests(MatchTestCase):
    def test_cli_training_selection_resume_new_stages_and_evaluation(self):
        from examples.evaluate_selfplay_ppo import main as evaluate
        t = self.cli('match')
        self.assertGreater(t.updates, 0)
        self.assertGreaterEqual(t.hands, 24)
        rows = [json.loads(l) for l in (self.path / 'match/progress.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['matches'], t.matches)
        self.assertTrue(all(r['reward_objective'] == 'match-win' and r['target_marks'] == 7 for r in rows))
        evaluated = [r for r in rows if 'evaluations' in r]
        self.assertGreater(len(evaluated), 1)
        for r in evaluated:
            self.assertEqual(r['selection_metric'], 'match_win_rate')
            games = r['evaluations']['HHH']['games']
            self.assertTrue(all(g['episode'] == 'match' and max(g['final_marks']) >= 7 for g in games))
            self.assertEqual(r['selection_score'], r['evaluation']['metrics'][0]['win_rate'])
        # best.pt: first strict maximum of the match win rate (deterministic ties keep the earliest).
        best_score = max(r['selection_score'] for r in evaluated)
        best_row = next(r for r in evaluated if r['selection_score'] == best_score)
        self.assertEqual(t.best_score, best_score)
        best = read(self.path / 'match/checkpoints/best.pt')
        self.assertEqual(best['progress']['hands'], best_row['hands'])
        self.assertEqual(best['metadata']['checkpoint_selection_metric'], 'match_win_rate')
        self.assertEqual((best['metadata']['reward_objective'], best['metadata']['target_marks']), ('match-win', 7))
        self.assertEqual(best['metadata']['value_initialization'], 'random')
        latest = self.path / 'match/checkpoints/latest.pt'
        load_model(latest)  # Loadable as an ordinary version-2 checkpoint.

        # Exact resume restores the objective; changing it or the target is rejected.
        for flags in (['--reward-objective', 'hand-marks'], ['--target-marks', '5']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                quiet(train, ['--resume', str(latest), '--total-hands', '40', *flags,
                              '--output-dir', str(self.path / 'match'), '--log-dir', str(self.path / 'r-logs')])
        resumed = quiet(train, ['--resume', str(latest), '--total-hands', '40', '--output-dir',
                                str(self.path / 'match'), '--log-dir', str(self.path / 'r-logs')])
        self.assertEqual((resumed.config.reward_objective, resumed.config.target_marks), ('match-win', 7))
        self.assertGreater(resumed.matches, t.matches)

        def stage(name, source, objective):
            return quiet(train, ['--init-checkpoint', str(source), '--learner-bid', 'neural-train',
                                 '--learner-trump', 'neural-frozen', '--learner-play', 'neural-train',
                                 '--reward-objective', objective, '--opponent-pool', 'HHH:1',
                                 '--eval-opponents', 'HHH', '--selection-opponent', 'HHH', '--total-hands', '4',
                                 '--rollout-decisions', '16', '--minibatch-size', '16', '--eval-deals', '2',
                                 '--bootstrap-samples', '5', '--forgetting-hands', '1', '--device', 'cpu',
                                 '--output-dir', str(self.path / name), '--log-dir', str(self.path / (name + '-l'))])
        source, _ = load_model(latest)
        # match-win -> match-win inherits the value heads; -> hand-marks resets them.
        same = stage('same', latest, 'match-win')
        self.assertEqual(same.value_initialization, 'inherited')
        hand = stage('hand', latest, 'hand-marks')
        self.assertEqual((hand.value_initialization, hand.initial_objective), ('reset', 'match-win'))
        data = read(self.path / 'hand/checkpoints/latest.pt')
        self.assertEqual((data['metadata']['value_initialization'], data['metadata']['checkpoint_selection_metric']),
                         ('reset', 'mean_signed_marks'))
        # A hand-marks checkpoint initializing match-win keeps actor weights and resets value heads.
        with mock.patch('examples.train_selfplay_ppo.SelfPlayTrainer.update', autospec=True,
                                 side_effect=lambda self, b: {}):
            back = stage('back', self.path / 'hand/checkpoints/latest.pt', 'match-win')
        initial, _ = load_model(self.path / 'hand/checkpoints/latest.pt')
        self.assertEqual(back.value_initialization, 'reset')
        for phase in ('bid', 'play'):
            a, b = back.model.networks[phase], initial.networks[phase]
            self.assertTrue(all(torch.equal(x, y.to(x.device)) for x, y in zip(
                (*a.trunk.parameters(), *a.policy.parameters()), (*b.trunk.parameters(), *b.policy.parameters()))))
            self.assertFalse(torch.equal(a.value.weight, b.value.weight))
        frozen_a, frozen_b = back.model.networks['trump'], initial.networks['trump']
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(frozen_a.parameters(), frozen_b.parameters())))

        result = self.path / 'eval.json'
        reports = quiet(evaluate, ['--checkpoint', str(self.path / 'match/checkpoints/best.pt'), '--opponents', 'HHH',
                                   '--episode', 'match', '--target-marks', '7', '--deals', '2',
                                   '--bootstrap-samples', '5', '--output', str(result),
                                   '--log-dir', str(self.path / 'e-logs')])
        report = reports['HHH']
        self.assertEqual(report.games_played, 4)
        self.assertTrue(all(g.episode == 'match' and max(g.final_marks) >= 7 for g in report.games))
        self.assertIsNotNone(report.average_match_length)
        self.assertEqual(json.loads(result.read_text())['episode'], 'match')

    def test_old_checkpoints_default_to_hand_marks(self):
        t = self.trainer(replace(self.config(reward_objective='hand-marks'), total_hands=4))
        t.update(t.collect()[0])
        save(self.path / 'old.pt', t)
        data = torch.load(self.path / 'old.pt', weights_only=True)
        for key in ('reward_objective', 'target_marks', 'checkpoint_selection_metric', 'initial_reward_objective',
                    'value_initialization'):
            data['metadata'].pop(key)
        for key in ('reward_objective', 'target_marks'):
            data['config'].pop(key)
        data['progress'].pop('matches')
        torch.save(data, self.path / 'old.pt')
        loaded = read(self.path / 'old.pt')
        self.assertNotIn('reward_objective', loaded['config'])
        resumed = self.trainer(replace(self.config(reward_objective='hand-marks'), total_hands=8))
        resume(self.path / 'old.pt', resumed)
        self.assertEqual((resumed.matches, resumed.config.reward_objective), (0, 'hand-marks'))


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class PipelineTests(MatchTestCase):
    CONFIG = None

    def setUp(self):
        super().setUp()
        self.CONFIG = REPOSITORY / 'configs/pipelines/match_win_selfplay.json'

    def test_example_configuration_parses_and_plans(self):
        pipeline = rp.load_config(self.CONFIG)
        rows = rp.plan(pipeline)  # Every step passes its command's own parser.
        argv = {step.id: list(arguments) for step, arguments, _, _ in rows}
        for sid in ('match-scratch', 'match-finetune'):
            self.assertIn('--reward-objective', argv[sid])
            self.assertEqual(argv[sid][argv[sid].index('--reward-objective') + 1], 'match-win')
            self.assertEqual(argv[sid][argv[sid].index('--target-marks') + 1], '7')
        finetune = argv['match-finetune']
        self.assertTrue(finetune[finetune.index('--init-checkpoint') + 1].endswith('match-scratch/checkpoints/best.pt'))
        self.assertIn('match', argv['head-to-head-matches'])
        self.assertEqual(quiet(rp.main, ['--config', str(self.CONFIG), '--dry-run']), 0)

    def test_parser_rejects_invalid_objective(self):
        config = json.loads(self.CONFIG.read_text())
        config['steps'][0]['args']['reward-objective'] = 'match-wins'
        config['output_root'] = str(self.path / 'bad')
        with self.assertRaises(rp.PipelineError):
            quiet(rp.plan, rp.parse_config(config))

    def test_smoke_pipeline(self):
        config = json.loads(self.CONFIG.read_text())
        config['output_root'] = str(self.path / 'out')
        tiny = {'hidden-size': 8, 'eval-deals': 2, 'bootstrap-samples': 5, 'forgetting-hands': 1, 'minibatch-size': 32,
                'ppo-epochs': 1, 'device': 'cpu', 'total-hands': 16, 'rollout-decisions': 64, 'threads': 1,
                'eval-threads': 1, 'eval-every-hands': 8, 'checkpoint-every-hands': 8, 'snapshot-every-hands': 8}
        for step in config['steps']:
            args = step['args']
            if step['command'] == 'train_selfplay_ppo':
                args.update({k: v for k, v in tiny.items() if k != 'hidden-size' or 'from-scratch' in args})
            else:
                args.update(deals=2, threads=1, **{'bootstrap-samples': 5})
        logs = []

        def captured(command, cwd):
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            logs.append(result.stdout[-2000:] + result.stderr[-2000:])
            return result
        code, output = quiet_result(rp.run, rp.parse_config(config), runner=captured)
        self.assertEqual(code, 0, output + '\n'.join(logs))
        out = self.path / 'out'
        for sid in ('match-scratch', 'match-finetune'):
            data = read(out / sid / 'checkpoints/best.pt')
            self.assertEqual(data['metadata']['reward_objective'], 'match-win')
            self.assertEqual(data['metadata']['checkpoint_selection_metric'], 'match_win_rate')
            self.assertEqual(set(data['metadata']['trained_phases']), set(PHASES))
            rows = [json.loads(l) for l in (out / sid / 'progress.jsonl').read_text().splitlines()]
            self.assertGreater(rows[-1]['updates'], 0)
            self.assertLessEqual(rows[-1]['hands'], 16 + 12)
        self.assertEqual(read(out / 'match-finetune/checkpoints/best.pt')['metadata']['value_initialization'],
                         'inherited')
        matches = json.loads((out / 'held-out-matches/result.json').read_text())
        self.assertEqual(matches['episode'], 'match')
        self.assertTrue(all(g['episode'] == 'match' for r in matches['evaluations'].values() for g in r['games']))
        hands = json.loads((out / 'held-out-hands/result.json').read_text())
        self.assertNotIn('episode', hands)  # The legacy hand envelope is unchanged.
        self.assertTrue(all(g['episode'] == 'hand' for r in hands['evaluations'].values() for g in r['games']))
        versus = json.loads((out / 'head-to-head-matches/result.json').read_text())
        self.assertTrue(all(g['episode'] == 'match' and g['target_marks'] == 7 for g in versus['games']))


def quiet_result(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return function(*args, **kwargs), buffer.getvalue()


if __name__ == '__main__':
    unittest.main()
