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

    def model(self, neural=None):
        # PHASES is imported only with the rl extra; a default of PHASES would be
        # evaluated at class definition and break import even when skipped.
        torch.manual_seed(0)
        return PhaseActorCritic(8, PHASES if neural is None else neural).eval()

    def cli(self, name, *extra, total='24'):
        args = ['--from-scratch', '--reward-objective', 'match-win', '--total-hands', total,
                '--rollout-decisions', '96', '--hidden-size', '8', '--minibatch-size', '16', '--ppo-epochs', '1',
                *(() if '--eval-match-pairs' in extra else ('--eval-deals', '2')),
                '--eval-every-hands', '8', '--bootstrap-samples', '5',
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
        for key in ('reward_objective', 'target_marks', 'total_matches'):
            data['config'].pop(key)
        for key in ('budget_unit', 'budget_target'):
            data['metadata'].pop(key)
        data['progress'].pop('matches')
        torch.save(data, self.path / 'old.pt')
        loaded = read(self.path / 'old.pt')
        self.assertNotIn('reward_objective', loaded['config'])
        resumed = self.trainer(replace(self.config(reward_objective='hand-marks'), total_hands=8))
        resume(self.path / 'old.pt', resumed)
        self.assertEqual((resumed.matches, resumed.config.reward_objective), (0, 'hand-marks'))


class MatchBudgetTests(MatchTestCase):
    """total_matches: exactly N complete matches; total_hands ignored."""

    def run_to_budget(self, trainer):
        rollouts = []
        while not trainer.budget_exhausted:
            before = (trainer.matches, trainer.hands)
            buffer, _ = trainer.collect()
            samples = sum(len(b) for b in buffer) if isinstance(buffer, tuple) else len(buffer)
            rollouts.append(dict(samples=samples, matches=trainer.matches - before[0],
                                 hands=trainer.hands - before[1], metrics=trainer.match_metrics,
                                 records=[] if isinstance(buffer, tuple) else list(buffer.records)))
            if samples:
                trainer.update(buffer)
        return rollouts

    def test_validation(self):
        self.assertIsNone(self.config().total_matches)
        self.assertEqual((self.config().budget_unit, self.config().budget_target), ('hands', 24))
        c = self.config(total_matches=5)
        self.assertEqual((c.budget_unit, c.budget_target), ('matches', 5))
        for value in (0, -1, 2.0, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'total_matches'):
                self.config(total_matches=value)
        with self.assertRaisesRegex(ValueError, 'requires reward_objective match-win'):
            self.config(reward_objective='hand-marks', total_matches=5)

    def test_exactly_n_matches_overriding_a_small_hand_budget(self):
        t = self.trainer(self.config(total_matches=6, total_hands=1, rollout_decisions=300))
        rollouts = self.run_to_budget(t)
        self.assertEqual(t.matches, 6)  # Exactly N, never more.
        self.assertGreater(t.hands, t.config.total_hands)  # total_hands was not a stopping condition.
        self.assertEqual(t.hands, sum(r['hands'] for r in rollouts))
        self.assertEqual(t.hands, sum(r['metrics']['hands'] for r in rollouts))
        self.assertGreater(len(rollouts), 1)
        self.assertTrue(all(r['samples'] >= 300 for r in rollouts[:-1]))
        self.assertLess(rollouts[-1]['samples'], 300)  # The final rollout is cut by the match budget.
        seen = []
        for r in rollouts:  # Complete matches only: one terminal record per nonempty seat trajectory.
            episodes = sorted({x.episode_id for x in r['records']})
            seen += episodes
            for episode in episodes:
                for seat in {x.seat for x in r['records'] if x.episode_id == episode}:
                    path = [x for x in r['records'] if (x.episode_id, x.seat) == (episode, seat)]
                    self.assertEqual([x.done for x in path], [False] * (len(path) - 1) + [True])
        self.assertEqual(seen, list(range(6)))  # Contiguous indices; no extra match was scheduled.
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            t.collect()

    def test_trump_only_and_paired_match_budgets(self):
        trump = self.trainer(self.config(('heuristic', 'neural-train', 'heuristic'), total_matches=4,
                                         rollout_decisions=2, max_rollout_hands=4))
        rollouts = self.run_to_budget(trump)
        self.assertEqual(trump.matches, 4)
        self.assertEqual(sum(r['matches'] for r in rollouts), 4)
        paired = self.trainer(self.config(partner_mode='paired', total_matches=3, total_hands=1),
                              PairedSelfPlayTrainer)
        self.run_to_budget(paired)
        self.assertEqual((paired.matches, paired.partner.matches), (3, 3))

    def test_worker_invariance(self):
        results = []
        for threads in (1, 2):
            t = self.trainer(self.config(threads=threads, total_matches=5, rollout_decisions=200))
            self.run_to_budget(t)
            results.append(([v.clone() for v in t.model.state_dict().values()], t.opponent_assignments,
                            t.hands, t.matches, t.bidding_metrics))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(results[0][0], results[1][0])))
        self.assertEqual(results[0][1:], results[1][1:])

    def test_exact_resume_extension_and_rejections(self):
        config = self.config(total_matches=5, rollout_decisions=200)
        continuous = self.trainer(config)
        continuous.update(continuous.collect()[0])
        save(self.path / 'split.pt', continuous)
        self.run_to_budget(continuous)
        resumed = self.trainer(config)
        resume(self.path / 'split.pt', resumed)
        self.run_to_budget(resumed)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(continuous.model.state_dict().values(),
                                                              resumed.model.state_dict().values())))
        self.assertEqual((continuous.hands, continuous.matches, continuous.optimizer_steps, continuous.samples),
                         (resumed.hands, resumed.matches, resumed.optimizer_steps, resumed.samples))
        done = read(self.path / 'split.pt')['progress']['matches']
        extended = self.trainer(replace(config, total_matches=8))
        resume(self.path / 'split.pt', extended)
        self.run_to_budget(extended)
        self.assertEqual(extended.matches, 8)
        for changes, message in ((dict(total_matches=done - 1), 'precedes'),
                                 (dict(total_matches=None), 'switch'),
                                 (dict(target_marks=5), 'mismatch'),
                                 (dict(reward_objective='hand-marks', total_matches=None), 'switch')):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                resume(self.path / 'split.pt', self.trainer(replace(config, **changes)))


class BiddingDiagnosticsTests(MatchTestCase):
    def test_per_decision_records_and_score_conditioning(self):
        from research.ppo.selfplay_rollout import CALL_CATEGORIES
        forced_seen = False
        for match in range(8):
            row = collect_phase_match(self.model(('trump', 'play')), 5, match, ('play',), self.pool('HHH'),
                                      target_marks=7)
            calls = row['bid_decisions']
            self.assertEqual(len(calls), 4 * row['hands'])  # Every seat calls once per hand.
            learner = opponent = 0
            for hand, marks in enumerate(row['hand_marks']):
                auction = calls[4 * hand:4 * hand + 4]
                self.assertIsNone(auction[0]['standing'])
                self.assertEqual(sum(c['learner'] for c in auction), 2)
                learner_declared, category = row['bids'][hand]
                for c in auction:
                    self.assertIn(c['call'], c['alternatives'])
                    self.assertTrue(set(c['alternatives']) <= set(CALL_CATEGORIES))
                    self.assertEqual(c['pass_legal'], 'pass' in c['alternatives'])
                    self.assertEqual(c['auction_won'], c['learner'] == learner_declared)
                    own, other = (learner, opponent) if c['learner'] else (opponent, learner)
                    self.assertEqual((c['own_marks'], c['opponent_marks']), (own, other))
                    self.assertEqual(c['score_state'], 'leading' if own > other else
                                     'trailing' if own < other else 'tied')
                    if c['forced']:
                        forced_seen = True
                        self.assertFalse(c['pass_legal'])
                        self.assertNotEqual(c['call'], 'pass')
                winners = [c for c in auction if c['call'] != 'pass']
                self.assertEqual(winners[-1]['call'], category)  # The last bid is the contract.
                learner += max(marks, 0)
                opponent += max(-marks, 0)
        self.assertTrue(forced_seen)

    def test_aggregates(self):
        from research.ppo.selfplay_trainer import bidding_summary
        rows = [collect_phase_match(self.model(), 42, m, PHASES, self.pool('HHH'), target_marks=7)
                for m in range(4)]
        summary = bidding_summary(rows)
        for team in ('learner', 'opponent'):
            s = summary[team]
            calls = [c for r in rows for c in r['bid_decisions'] if c['learner'] == (team == 'learner')]
            self.assertEqual(s['decisions'], len(calls))
            self.assertEqual(sum(s['calls'].values()), len(calls))
            self.assertEqual(sum(v['decisions'] for v in s['by_score_state'].values()), len(calls))
            self.assertEqual(sum(v['decisions'] for v in s['by_match_score'].values()), len(calls))
            self.assertEqual(s['forced_calls'], sum(c['forced'] for c in calls))
            declared = [c for r in rows for c in r['contracts'] if c[0] == (team == 'learner')]
            self.assertEqual(s['contracts']['total'], len(declared))
            self.assertEqual(sum(v['made'] for v in s['contracts']['by_category'].values()), sum(c[2] for c in declared))
            for category, attempt in s['bid_attempts'].items():
                self.assertEqual(attempt['calls'], s['calls'][category])
        hand = collect_phase_hand(self.model(), 42, 0, PHASES, self.pool('HHH'))
        self.assertEqual(len(hand['bid_decisions']), 4)
        self.assertTrue(all(c['score_state'] == 'tied' for c in hand['bid_decisions']))


class EvaluationUnitTests(MatchTestCase):
    def parse(self, module, *argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return module.build_parser().parse_args(list(argv))

    def test_match_pair_aliases(self):
        from examples import evaluate_agent_vs_agent, evaluate_selfplay_ppo
        for module, required in ((evaluate_selfplay_ppo, ['--checkpoint', 'x.pt']), (evaluate_agent_vs_agent, [])):
            with self.subTest(module=module.__name__):
                self.assertEqual(self.parse(module, *required).deals, 2048)
                self.assertEqual(self.parse(module, *required, '--deals', '9').deals, 9)
                self.assertEqual(self.parse(module, *required, '--episode', 'match', '--deals', '9').deals, 9)
                self.assertEqual(self.parse(module, *required, '--episode', 'match', '--match-pairs', '7').deals, 7)
                for argv in (['--episode', 'match', '--match-pairs', '7', '--deals', '7'], ['--match-pairs', '7']):
                    with self.subTest(argv=argv), self.assertRaises(SystemExit):
                        self.parse(module, *required, *argv)
        from examples import train_selfplay_ppo
        self.assertIsNone(self.parse(train_selfplay_ppo).eval_deals)  # Resolved to 128 in main().
        with self.assertRaises(SystemExit):
            self.parse(train_selfplay_ppo, '--eval-match-pairs', '4', '--eval-deals', '4')


class MatchBudgetCommandTests(MatchTestCase):
    def test_cli_total_matches_progress_resume_and_rejections(self):
        t = self.cli('budget', '--total-matches', '3', '--eval-match-pairs', '2', total='1')
        self.assertEqual(t.matches, 3)
        self.assertEqual(t.config.eval_deals, 2)
        rows = [json.loads(l) for l in (self.path / 'budget/progress.jsonl').read_text().splitlines()]
        last = rows[-1]
        for key in ('reward_objective', 'budget_unit', 'budget_target', 'matches', 'hands', 'training_samples',
                    'updates', 'rollout_matches', 'rollout_hands', 'mean_hands_per_match', 'match_win_rate',
                    'mean_terminal_reward', 'mean_signed_marks_per_hand', 'bidding', 'rollout_bids'):
            self.assertIn(key, last)
        self.assertEqual((last['budget_unit'], last['budget_target'], last['matches']), ('matches', 3, 3))
        self.assertEqual(sum(r['rollout_matches'] for r in rows), 3)
        self.assertEqual(sum(r['rollout_hands'] for r in rows), last['hands'])
        self.assertTrue(all(isinstance(r['bidding']['learner']['by_score_state'], dict) for r in rows))
        evaluated = [r for r in rows if 'evaluation_sample_units' in r]
        self.assertEqual(evaluated[-1]['evaluation_sample_units']['HHH']['complete_matches'], 4)
        latest = str(self.path / 'budget/checkpoints/latest.pt')

        def again(*flags):
            return quiet(train, ['--resume', latest, *flags, '--output-dir', str(self.path / 'budget'),
                                 '--log-dir', str(self.path / 'budget-r')])
        for flags in (['--total-matches', '2'], ['--reward-objective', 'hand-marks'], ['--target-marks', '5']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                again(*flags)
        extended = again('--total-matches', '5')
        self.assertEqual((extended.matches, extended.config.total_matches), (5, 5))
        self.assertEqual(extended.config.reward_objective, 'match-win')
        # A hand-budget match run cannot switch to a match budget on exact resume, only in a new stage.
        self.cli('hands', total='8')
        with self.assertRaises(SystemExit):
            quiet(train, ['--resume', str(self.path / 'hands/checkpoints/latest.pt'), '--total-matches', '9',
                          '--output-dir', str(self.path / 'hands'), '--log-dir', str(self.path / 'hands-r')])
        stage = quiet(train, ['--init-checkpoint', str(self.path / 'hands/checkpoints/latest.pt'),
                              '--learner-bid', 'neural-train', '--learner-trump', 'neural-train',
                              '--learner-play', 'neural-train', '--reward-objective', 'match-win',
                              '--total-matches', '2', '--opponent-pool', 'HHH:1', '--eval-opponents', 'HHH',
                              '--selection-opponent', 'HHH', '--eval-match-pairs', '2', '--rollout-decisions', '32',
                              '--minibatch-size', '16', '--bootstrap-samples', '5', '--forgetting-hands', '1',
                              '--device', 'cpu', '--output-dir', str(self.path / 'stage'),
                              '--log-dir', str(self.path / 'stage-l')])
        self.assertEqual(stage.matches, 2)

    def test_cli_rejections(self):
        base = ['--from-scratch', '--device', 'cpu', '--log-dir', str(self.path / 'l')]
        for flags in (['--total-matches', '3'],  # hand-marks
                      ['--eval-match-pairs', '2'],  # hand-marks
                      ['--reward-objective', 'match-win', '--total-matches', '0'],
                      ['--reward-objective', 'match-win', '--eval-match-pairs', '2', '--eval-deals', '2']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                quiet(train, [*base, *flags, '--output-dir', str(self.path / 'x')])


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

    def shrink(self, config, matches=3):
        """Smoke-scale copy of an example pipeline; every unit-specific key is kept in its own unit."""
        config['output_root'] = str(self.path / config['name'])
        tiny = {'bootstrap-samples': 5, 'forgetting-hands': 1, 'minibatch-size': 32, 'ppo-epochs': 1, 'device': 'cpu',
                'rollout-decisions': 64, 'threads': 1, 'eval-threads': 1, 'eval-every-hands': 8,
                'checkpoint-every-hands': 8, 'snapshot-every-hands': 8}
        for step in config['steps']:
            args = step['args']
            if step['command'] == 'train_selfplay_ppo':
                args.update(tiny)
                if 'from-scratch' in args:
                    args['hidden-size'] = 8
                for key, value in (('total-matches', matches), ('total-hands', 16), ('eval-match-pairs', 2),
                                   ('eval-deals', 2)):
                    if key in args:
                        args[key] = value
            else:
                args.update(threads=1, **{'bootstrap-samples': 5})
                args['match-pairs' if 'match-pairs' in args else 'deals'] = 2
        return config

    def run_pipeline(self, config):
        logs = []

        def captured(command, cwd):
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            logs.append(result.stdout[-2000:] + result.stderr[-2000:])
            return result
        code, output = quiet_result(rp.run, rp.parse_config(config), runner=captured)
        self.assertEqual(code, 0, output + '\n'.join(logs))
        return Path(config['output_root'])

    def test_smoke_pipeline(self):
        out = self.run_pipeline(self.shrink(json.loads(self.CONFIG.read_text())))
        for sid in ('match-scratch', 'match-finetune'):
            data = read(out / sid / 'checkpoints/best.pt')
            self.assertEqual(data['metadata']['reward_objective'], 'match-win')
            self.assertEqual(data['metadata']['checkpoint_selection_metric'], 'match_win_rate')
            self.assertEqual(data['metadata']['budget_unit'], 'matches')
            self.assertEqual(set(data['metadata']['trained_phases']), set(PHASES))
            rows = [json.loads(l) for l in (out / sid / 'progress.jsonl').read_text().splitlines()]
            self.assertGreater(rows[-1]['updates'], 0)
            self.assertEqual((rows[-1]['matches'], rows[-1]['budget_unit'], rows[-1]['budget_target']),
                             (3, 'matches', 3))
            self.assertEqual(read(out / sid / 'checkpoints/latest.pt')['progress']['matches'], 3)
        self.assertEqual(read(out / 'match-finetune/checkpoints/best.pt')['metadata']['value_initialization'],
                         'inherited')
        matches = json.loads((out / 'held-out-matches/result.json').read_text())
        self.assertEqual((matches['episode'], matches['match_pairs']), ('match', 2))
        self.assertTrue(all(u == dict(u, duplicate_match_groups=2, complete_matches=4)
                            for u in matches['sample_units'].values()))
        self.assertTrue(all(g['episode'] == 'match' for r in matches['evaluations'].values() for g in r['games']))
        hands = json.loads((out / 'held-out-hands/result.json').read_text())
        self.assertNotIn('episode', hands)  # The legacy hand envelope is unchanged.
        self.assertTrue(all(g['episode'] == 'hand' for r in hands['evaluations'].values() for g in r['games']))
        versus = json.loads((out / 'head-to-head-matches/result.json').read_text())
        self.assertTrue(all(g['episode'] == 'match' and g['target_marks'] == 7 for g in versus['games']))
        self.assertEqual(versus['sample_units']['complete_matches'], 4)

    def test_comparison_pipeline_design_and_smoke(self):
        config = json.loads((REPOSITORY / 'configs/pipelines/compare_hand_vs_match.json').read_text())
        hand, match = (next(s for s in config['steps'] if s['id'] == i)['args'] for i in ('hand-marks', 'match-win'))
        # The branches differ only in the objective and its validation unit; equal simulated-hand budgets.
        unit_keys = {'reward-objective', 'target-marks', 'eval-deals', 'eval-match-pairs'}
        self.assertEqual({k: v for k, v in hand.items() if k not in unit_keys},
                         {k: v for k, v in match.items() if k not in unit_keys})
        self.assertNotIn('total-matches', match)
        self.assertEqual(hand['eval-deals'], match['eval-match-pairs'])
        self.assertEqual(len(rp.plan(rp.parse_config(config))), len(config['steps']))
        out = self.run_pipeline(self.shrink(config))
        a, b = (read(out / sid / 'checkpoints/latest.pt') for sid in ('hand-marks', 'match-win'))
        self.assertEqual((a['metadata']['reward_objective'], b['metadata']['reward_objective']),
                         ('hand-marks', 'match-win'))
        self.assertEqual(a['progress']['hands'], 16)
        self.assertTrue(16 <= b['progress']['hands'] <= 16 + 12)
        primary = [json.loads((out / sid / 'result.json').read_text()) for sid in ('hand-marks-matches',
                                                                                  'match-win-matches')]
        self.assertEqual(*[[g['schedule_ids'] for r in p['evaluations'].values() for g in r['games']]
                           for p in primary])  # Identical held-out match schedules.

    def test_interrupted_match_budget_step_resumes_with_its_budget(self):
        step = rp.parse_config(dict(schema_version=1, name='x', output_root=str(self.path / 'r'), steps=[dict(
            id='s', command='train_selfplay_ppo', args={'from-scratch': True, 'reward-objective': 'match-win',
                                                        'total-matches': 9, 'threads': 2})])).steps[0]
        argv = rp.resume_argv(step, step.args)
        self.assertEqual(argv[argv.index('--total-matches') + 1], '9')
        self.assertNotIn('--reward-objective', argv)  # Restored from the checkpoint by the command itself.


def quiet_result(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return function(*args, **kwargs), buffer.getvalue()


if __name__ == '__main__':
    unittest.main()
