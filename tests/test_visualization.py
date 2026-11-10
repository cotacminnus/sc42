"""Read-only experiment visualization: parsing, normalization, selection, outputs (no training is run)."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from research.analysis import FAMILIES, RESEARCH_PRESETS

VIZ = all(importlib.util.find_spec(name) for name in ('numpy', 'pandas', 'matplotlib', 'plotly'))
SHA_A, SHA_B, SHA_C = 'a' * 64, 'b' * 64, 'c' * 64

if VIZ:
    import numpy as np
    from examples import visualize_experiments as cli
    from research.analysis import loader
    from research.analysis.metrics import compare_experiments, partner_matrices, partnership_gap, partnership_relation
    from research.analysis.normalize import build_dataset, category_of
    from research.analysis.plots import Options, build_figures


def game(index, episode='hand', hands=1, group=None):
    hand = dict(deal_id=f'd{index}', seating=['A', 'B', 'A', 'B'], shaker=0,
                result=dict(contract=dict(kind='normal', bidder=0, bid=dict(points=30, marks=None)), bidder=0, bid=dict(points=30, marks=None),
                            trump=0, team_points=[30, 12], scoring_team=0, marks_awarded=1),
                decisions=33, bid_counts=[1, 0, 0, 0], rewards=[1, -1, 1, -1], forced_bid=False, contract_decided_at_trick=5, would_pass_if_allowed=None,
                contract_success=True)
    return dict(game_id=f'{index}:A', group_id=group or f'{index // 2}:g', episode=episode, target_marks=7, seating=['A', 'B', 'A', 'B'], agent_seeds=[1, 2, 3, 4],
                schedule_ids=['s'], hands=[dict(hand, deal_id=f'd{index}-{i}') for i in range(hands)], winner=0, final_marks=[7, 0], rewards=[1, -1, 1, -1], num_hands=hands)


def report(win=0.6, marks=0.2, episode='hand', games=4, hands=1, seed=11, low=None):
    a = dict(policy='A', games_played=games, win_rate=win, mean_signed_marks=marks, contracts_declared=3, contract_success_rate=0.5, mean_contract_value=1.0, bidding_frequency=0.4)
    b = dict(a, policy='B', win_rate=1 - win, mean_signed_marks=-marks)
    ci = dict(low=(win - 0.1) if low is None else low, high=win + 0.1)
    interval = dict(win_rate=ci, mean_signed_marks=dict(low=marks - 0.1, high=marks + 0.1))
    return dict(seed=seed, bootstrap_seed=0, bootstrap_samples=200, games=[game(i, episode, hands) for i in range(games)], metrics=[a, b],
                intervals=[dict(interval, policy='A'), dict(interval, policy='B')], games_played=games, average_match_length=float(hands), contract_success_rate=0.5)


BREAKDOWN = {'A': dict(successful=2, failed=1, contract_origins={'points:30': dict(forced=dict(contracts=1, successful=0, failed=1), voluntary=dict(contracts=2, successful=2, failed=0))},
                       contract_values={'points:30': dict(successful=2, failed=1)},
                       decision_timing=dict(contracts=3, recorded=2, distribution={'3': 1, '5': 1}),
                       roles=dict(declarer=dict(hands=3, wins=2, signed_marks_sum=1, win_rate=0.6, mean_signed_marks=0.3), defender=dict(hands=1, wins=0, signed_marks_sum=-1, win_rate=0.0, mean_signed_marks=-1.0)))}


def bidding(state_decisions=(0, 4, 0)):
    leading, tied, trailing = state_decisions
    return dict(decisions=leading + tied + trailing, calls={'pass': 2, '30': 1, '31-41': 1, '42': 0, '2+ marks': 0}, voluntary_pass_rate=2 / 3, forced_calls=1,
                by_score_state=dict(leading=dict(decisions=leading, bid_rate=0.5 if leading else None, high_bid_rate=0.0 if leading else None),
                                    tied=dict(decisions=tied, bid_rate=0.5 if tied else None, high_bid_rate=0.0 if tied else None),
                                    trailing=dict(decisions=trailing, bid_rate=0.5 if trailing else None, high_bid_rate=0.0 if trailing else None)),
                by_match_score={'0-0': dict(decisions=tied, bid_rate=0.5, high_bid_rate=0.0)} if tied and not leading else {'0-0': dict(decisions=2, bid_rate=0.5, high_bid_rate=0.0), '1-0': dict(decisions=2, bid_rate=1.0, high_bid_rate=0.0)},
                bid_attempts={c: dict(calls=1, auction_won=1, overbid_rate=0.0) for c in ('30', '31-41')} | {c: dict(calls=0, auction_won=0, overbid_rate=None) for c in ('42', '2+ marks')},
                contracts=dict(total=2, by_category={'30': dict(contracts=1, made=1, success_rate=1.0), '31-41': dict(contracts=1, made=0, success_rate=0.0),
                                                     '42': dict(contracts=0, made=0, success_rate=None), '2+ marks': dict(contracts=0, made=0, success_rate=None)}))


def training_row(update, *, critic='local', objective='hand-marks', evaluate=False, runtime=None, **extra):
    value = None if critic == 'none' else 0.5
    metrics = dict(policy_loss=-0.01, entropy=0.9, approx_kl=0.01, clip_fraction=0.1, gradient_norm=0.5, value_loss=value, bid_entropy=1.0, trump_entropy=0.8, play_entropy=0.7,
                   bid_approx_kl=0.01, trump_approx_kl=0.01, play_approx_kl=0.01, bid_value_loss=value)
    diagnostics = dict(samples=10, return_mean=0.2, return_std=0.5, advantage_mean=0.0, advantage_std=0.4, value_mean=None if critic == 'none' else 0.1)
    row = dict(hands=update * 100, updates=update, optimizer_steps=update * 4, training_samples=update * 500, mean_reward=0.1 * update, phase_samples=dict(bid=10, trump=10, play=80), metrics=metrics,
               value_diagnostics=dict(explained_variance=None if critic == 'none' else 0.4, bid=diagnostics, trump=diagnostics, play=diagnostics),
               training_by_opponent=dict(HHH=dict(hands=50, reward_sum=5.0, mean_signed_reward=0.1), **{'self-current': dict(hands=50, reward_sum=-5.0, mean_signed_reward=-0.1)}),
               train_opponents=[['HHH', '1/2'], ['self-current', '1/2']],
               opponent_assignments=[[update, 'HHH', None], [update + 1, 'self-current', 'd' * 64]],
               collection_seconds=1.0, optimization_seconds=2.0, evaluation_seconds=3.0 if evaluate else 0.0, checkpoint_seconds=0.1, runtime_seconds=runtime if runtime is not None else update * 10.0,
               collection_hands_per_second=100.0, collection_decisions_per_second=1000.0, rollout_performance=dict(worker_peak_rss_bytes=2**28), critic_mode=critic, return_estimator='monte-carlo' if critic == 'none' else 'gae',
               reward_objective=objective, rollout_bids=dict(learner={'30': 1, '31-41': 1, '42': 0, '2+ marks': 0}, opponent={'30': 1, '31-41': 0, '42': 0, '2+ marks': 0}),
               bidding=dict(learner=bidding(), opponent=bidding()))
    if objective == 'match-win':
        row.update(matches=update * 10, rollout_matches=10, rollout_hands=100, mean_hands_per_match=10.0, match_win_rate=0.5, hand_win_rate=0.5, mean_signed_marks_per_hand=0.1, mean_terminal_reward=0.1,
                   bidding=dict(learner=bidding((2, 2, 2)), opponent=bidding((2, 2, 2))))
        row['training_by_opponent']['HHH'].update(matches=5)
        row['training_by_opponent']['self-current'].update(matches=5)
    if evaluate:
        episode = 'match' if objective == 'match-win' else 'hand'
        row['evaluations'] = dict(HHH=report(0.5 + update / 100, episode=episode, hands=10 if episode == 'match' else 1), CHH=report(0.4, episode=episode, hands=10 if episode == 'match' else 1))
        row['evaluation_identities'] = dict(HHH=None, CHH=None)
    return {**row, **extra}


def write_run(directory, rows, config=None):
    directory.mkdir(parents=True)
    (directory / 'progress.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (directory / 'config.json').write_text(json.dumps({**dict(seed=42, total_hands=500, hidden_size=256, eval_every_hands=200, eval_deals=2, learn_phases=['bid', 'trump', 'play'], train_opponents=[['HHH', '1']]), **(config or {})}), encoding='utf-8')
    checkpoints = directory / 'checkpoints'
    checkpoints.mkdir()
    for name in ('best', 'latest'):
        (checkpoints / f'{name}.pt').write_bytes(name.encode())
    return directory


def simple_run(directory, **kwargs):
    return write_run(directory, [training_row(u, evaluate=u % 2 == 0, **kwargs) for u in range(1, 6)], dict(critic_mode=kwargs.get('critic', 'local')))


def legacy_rows():
    return [dict(hands=u * 100, updates=u, mean_signed_reward=0.1, policy_loss=-0.1, value_loss=0.5, entropy=0.8, approx_kl=0.01, clip_fraction=0.1, gradient_norm=0.5, runtime_seconds=u * 5.0, train_opponents=[['HHH', '1']],
                 training_by_opponent=dict(HHH=dict(hands=100, reward_sum=10, mean_signed_reward=0.1)), **({'evaluation': report()} if u == 2 else {})) for u in range(1, 4)]


def agent(strategy='neural', path='runs/x/checkpoints/best.pt', sha=SHA_A):
    phase = dict(strategy=strategy, checkpoint=dict(path=path, sha256=sha) if strategy == 'neural' else None)
    return dict(name='A', bidding=phase, trump=phase, playing=phase)


def composed(a, b, *, a_partner=None, b_partner=None, win=0.6, marks=0.2, episode='hand', identical=False, hands=1):
    data = report(win, marks, episode, hands=hands)
    data['composition'] = dict(agents=dict(A=a, B=b), comparisons=dict(phase_policies_identical=dict(bid=identical, trump=identical, play=identical)), neural_inference='greedy')
    if a_partner or b_partner:
        data['composition']['partners'] = dict(A=a_partner, B=b_partner)
    data['breakdown'] = BREAKDOWN
    return data


def multi_opponent(episode='hand'):
    return dict(checkpoint_id=SHA_A, anchor_id=None, opponent_identities=dict(HHH=None), evaluations=dict(HHH=report(0.7, episode=episode), CHH=report(0.8, episode=episode)), breakdowns=dict(HHH=BREAKDOWN, CHH=BREAKDOWN))


def search_report(rank=True):
    beliefs = dict(uniform=dict(p_true=0.3, log_loss=1.2, ess=None), **{'heuristic-likelihood': dict(p_true=0.4, log_loss=1.1, ess=8.0), 'model-likelihood': dict(p_true=0.5, log_loss=1.0, ess=6.0)})
    summary = dict(positions=2, beliefs={k: dict(p_true=v['p_true'], log_loss=v['log_loss'], mean_ess=v['ess']) for k, v in beliefs.items()},
                   values={m: dict(bias=0.1, rmse=0.9, correlation=0.4, agreement=0.5, mean_seconds=0.2, max_seconds=0.3) for m in ('heuristic-rollouts', 'model-rollouts')})
    if rank:
        summary['ranking'] = dict(noise=dict(candidate='a', reference='b', concordance=0.9, top1=0.8, spearman=0.7, spearman_positions=2, regret=0.1, regret_units='points'),
                                  truncation=dict(candidate='c', reference='d', concordance=0.6, top1=0.5, spearman=0.4, spearman_positions=2, regret=0.2, regret_units='marks'))
    rows = [dict(game=g, seat=0, action_id=1, tricks_completed=g, realized_marks=1, beliefs=beliefs, values={}) for g in (1, 2)]
    return dict(model=dict(path='m.pt', sha256=SHA_A), table='model', games=2, positions_per_game=1, samples=8, seed=3, smoothing=0.01, summary=summary, rows=rows, rank_check=rank)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def run_cli(*arguments):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        try:
            code = cli.main([str(a) for a in arguments])
        except SystemExit as stop:
            code = stop.code
    return code, out.getvalue(), err.getvalue()


def tree_digest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob('*')) if p.is_file()}


def write_pipeline(root):
    write_json(root / 'pipeline_state.json', dict(name='p', steps=[
        dict(id='train', command='train_selfplay_ppo', kind='training', state='completed', locations=dict(directory='/nonexistent/train'), dependencies=[]),
        dict(id='broken', command='train_selfplay_ppo', kind='training', state='failed'),
        dict(id='labels', command='oracle_labels', kind='dataset', state='completed'),
        dict(id='evaluate', command='evaluate_selfplay_ppo', kind='evaluation', state='completed', dependencies=[dict(reference='train.best', path='x', sha256=SHA_A)])]))
    simple_run(root / 'train')
    write_json(root / 'evaluate' / 'result.json', multi_opponent())
    return root


@unittest.skipUnless(VIZ, 'requires the viz extra')
class VizCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def dataset(self, *runs, evaluations=(), searches=()):
        training = [loader.load_training(path.name, path) for path in runs]
        reports = [loader.load_evaluation(path.stem, path) for path in evaluations]
        found = [loader.load_search_accuracy(path.stem, path) for path in searches]
        return build_dataset(training, reports, found)

    def figures(self, data, *families, **options):
        figures, omissions = build_figures(data, families, Options(**options))
        return {f.id: f for f in figures}, {o.id: o for o in omissions}


class LoaderTests(VizCase):
    def test_strip_games_matches_a_full_parse(self):
        row = dict(evaluations=dict(HHH=report(episode='match', hands=3), CHH=report()), evaluation=report(games=2))
        for text in (json.dumps(row), json.dumps(row, indent=2), json.dumps(row, separators=(',', ':'))):
            fast = json.loads(loader.strip_games(text))
            self.assertEqual(fast, loader._strip_parsed(json.loads(text)))
        summary = fast['evaluations']['HHH']['_games']
        self.assertEqual((summary['games'], summary['groups'], summary['hands'], summary['episode']), (4, 2, 12, 'match'))

    def test_training_jsonl_keeps_counts_and_never_keeps_games(self):
        run = simple_run(self.root / 'run')
        rows = [row for _, row in loader.load_training('run', run).rows()]
        self.assertEqual(len(rows), 5)
        self.assertNotIn('games', rows[1]['evaluations']['HHH'])
        self.assertEqual(rows[1]['evaluations']['HHH']['_games']['groups'], 2)

    def test_missing_paths_and_wrong_files_are_input_errors(self):
        with self.assertRaises(loader.InputError):
            loader.load_training('x', self.root / 'absent')
        with self.assertRaises(loader.InputError):
            loader.load_evaluation('x', self.root / 'absent.json')
        with self.assertRaises(loader.InputError):
            loader.load_evaluation('x', write_json(self.root / 'list.json', [1, 2]))
        with self.assertRaises(loader.InputError):
            loader.load_search_accuracy('x', write_json(self.root / 'not_search.json', dict(summary={})))

    def test_pipeline_discovery_uses_completed_steps_only_and_survives_moves(self):
        contents = loader.discover_pipeline(write_pipeline(self.root / 'pipe'))
        self.assertEqual([s.name for s in contents.training], ['train'])
        self.assertEqual([s.name for s in contents.evaluations], ['evaluate'])
        self.assertEqual(contents.evaluations[0].references, ['train.best'])
        self.assertTrue(any('broken' in note and 'failed' in note for note in contents.skipped))
        self.assertTrue(any('labels' in note for note in contents.skipped))

    def test_report_without_breakdown_falls_back_to_hands(self):
        source = loader.load_evaluation('x', write_json(self.root / 'old.json', report(games=4)))
        data = build_dataset([], [source], [])
        self.assertEqual(int(data.contracts['contracts'].sum()), 4)
        self.assertEqual(set(data.contracts['category']), {'30'})

    def test_metrics_and_hand_csvs_load(self):
        directory = self.root / 'logs'
        directory.mkdir()
        (directory / 'r-metrics.csv').write_text('policy,games_played,win_rate,mean_signed_marks,contracts_declared,contract_success_rate,win_rate_low,win_rate_high,mean_signed_marks_low,mean_signed_marks_high,experiment_seed,bootstrap_seed,bootstrap_samples\n'
                                                 'A,4,0.6,0.2,3,0.5,0.5,0.7,0.1,0.3,11,0,200\n', encoding='utf-8')
        (directory / 'r.csv').write_text('group_id,game_id,declaring_policy,bid_points,bid_marks,success,forced_bid,contract_decided_at_trick\n'
                                         'g0,0:A,A,30,,True,False,5\ng0,0:B,A,31,,False,True,\n', encoding='utf-8')
        data = build_dataset([], [loader.load_evaluation('r', directory / 'r-metrics.csv')], [])
        win = data.evaluation[data.evaluation['metric'] == 'hand_win_rate'].iloc[0]
        self.assertEqual((win['value'], win['ci_low'], win['groups']), (0.6, 0.5, 1))
        self.assertEqual(int(data.contracts['contracts'].sum()), 2)
        self.assertEqual(int(data.timing['recorded'].iloc[0]), 1)

    def test_flattened_training_csv_is_only_a_fallback(self):
        directory = self.root / 'run'
        (directory / 'logs').mkdir(parents=True)
        (directory / 'logs' / 'training.csv').write_text('hands,updates,mean_reward,ppo_entropy,eval_HHH_win_rate,eval_HHH_win_rate_low,eval_HHH_win_rate_high\n100,1,0.1,0.9,0.6,0.5,0.7\n', encoding='utf-8')
        data = build_dataset([loader.load_training('run', directory)], [], [])
        self.assertAlmostEqual(data.training['run']['ppo.entropy'].iloc[0], 0.9)
        self.assertEqual(data.evaluation['episode'].iloc[0], 'unknown')
        self.assertTrue(any('flattened' in w for w in data.warnings))


class NormalizationTests(VizCase):
    def test_hand_and_match_win_rates_never_share_a_metric(self):
        hand = simple_run(self.root / 'hand')
        match = simple_run(self.root / 'match', objective='match-win')
        data = self.dataset(hand, match)
        by_run = data.evaluation[data.evaluation['policy'] == 'A'].groupby('experiment')['metric'].agg(set)
        self.assertIn('hand_win_rate', by_run['hand'])
        self.assertNotIn('match_win_rate', by_run['hand'])
        self.assertIn('match_win_rate', by_run['match'])
        self.assertNotIn('hand_win_rate', by_run['match'])
        self.assertEqual(data.meta('match')['reward_unit'], 'terminal reward per match (+1/-1)')
        self.assertFalse(data.training['hand'].get('matches', np.array([np.nan])).notna().any() if 'matches' in data.training['hand'] else False)
        self.assertEqual(data.training['match']['matches'].iloc[-1], 50)

    def test_critic_free_values_stay_absent_even_if_a_log_wrote_zero(self):
        rows = [training_row(u, critic='none') for u in range(1, 3)]
        for row in rows:
            row['metrics']['value_loss'] = 0.0
            row['value_diagnostics']['bid']['value_mean'] = 0.0
        data = self.dataset(write_run(self.root / 'none', rows))
        frame = data.training['none']
        self.assertNotIn('ppo.value_loss', frame)
        self.assertNotIn('value.explained_variance', frame)
        self.assertNotIn('value.bid.value_mean', frame)
        self.assertIn('value.bid.advantage_std', frame)
        self.assertEqual(data.meta('none')['return_estimator'], 'monte-carlo')

    def test_local_and_centralized_records_label_their_critic(self):
        data = self.dataset(simple_run(self.root / 'local'), simple_run(self.root / 'central', critic='centralized'))
        figures, _ = self.figures(data, 'critic')
        labels = [s.label for s in figures['critic_value_loss'].panels[0].series]
        self.assertTrue(any('critic: local; returns: gae' in label for label in labels))
        self.assertTrue(any('critic: centralized; returns: gae' in label for label in labels))

    def test_critic_targets_of_different_objectives_get_separate_figures(self):
        data = self.dataset(simple_run(self.root / 'hand'), simple_run(self.root / 'match', objective='match-win'))
        figures, _ = self.figures(data, 'critic')
        self.assertNotIn('critic_value_loss', figures)
        self.assertEqual([s.key for s in figures['critic_value_loss_match_win'].panels[0].series], ['match'])
        self.assertEqual([s.key for s in figures['critic_value_loss_hand_marks'].panels[0].series], ['hand'])
        self.assertIn('terminal reward', figures['critic_value_vs_return_match_win'].panels[0].ylabel)
        self.assertIn('marks', figures['critic_value_vs_return_hand_marks'].panels[0].ylabel)

    def test_legacy_v1_logs_load_with_flat_metrics_and_inferred_settings(self):
        data = self.dataset(write_run(self.root / 'legacy', legacy_rows()))
        meta = data.meta('legacy')
        self.assertEqual((meta['reward_objective'], meta['critic_mode'], meta['return_estimator']), ('hand-marks', 'local', 'gae'))
        self.assertIn('inferred', meta['critic_mode_source'])
        self.assertAlmostEqual(data.training['legacy']['ppo.entropy'].iloc[0], 0.8)
        self.assertEqual(data.training['legacy']['train_reward'].iloc[0], 0.1)
        self.assertEqual(set(data.evaluation['opponent']), {'unknown'})

    def test_missing_optional_fields_are_nan_not_zero(self):
        rows = [dict(hands=100, updates=1, mean_reward=0.1), dict(hands=200, updates=2, mean_reward=0.2, metrics=dict(entropy=0.5))]
        frame = self.dataset(write_run(self.root / 'sparse', rows)).training['sparse']
        self.assertTrue(np.isnan(frame['ppo.entropy'].iloc[0]))
        self.assertEqual(frame['ppo.entropy'].iloc[1], 0.5)
        self.assertNotIn('ppo.policy_loss', frame)

    def test_resume_replay_is_dropped_and_runtime_continues(self):
        rows = [training_row(1, runtime=10.0), training_row(2, runtime=20.0), training_row(3, runtime=30.0), dict(training_row(2, runtime=1.0)), training_row(3, runtime=2.0), training_row(4, runtime=3.0)]
        data = self.dataset(write_run(self.root / 'resumed', rows))
        frame = data.training['resumed']
        self.assertEqual(frame['hands'].tolist(), [100, 200, 300, 400])
        self.assertEqual(frame['source_line'].tolist(), [1, 4, 5, 6])
        self.assertEqual(frame['runtime_cumulative'].tolist(), [10.0, 11.0, 12.0, 13.0])
        self.assertTrue(any('replayed' in w for w in data.warnings))
        self.assertTrue(any('restarted' in w for w in data.warnings))

    def test_recorded_intervals_are_read_not_recomputed(self):
        data = self.dataset(evaluations=[write_json(self.root / 'multi.json', multi_opponent())])
        row = data.evaluation.query("metric == 'hand_win_rate' and policy == 'A' and opponent == 'HHH'").iloc[0]
        self.assertAlmostEqual(row['ci_low'], 0.6)
        self.assertAlmostEqual(row['ci_high'], 0.8)
        self.assertEqual((row['games'], row['groups'], row['sample_unit']), (4, 2, 'hands'))

    def test_match_sample_units_prefer_recorded_groups(self):
        data = dict(multi_opponent('match'), sample_units=dict(HHH=dict(duplicate_match_groups=9, complete_matches=18, hands=180, average_match_length=10.0)))
        frame = self.dataset(evaluations=[write_json(self.root / 'm.json', data)]).evaluation
        row = frame.query("metric == 'match_win_rate' and opponent == 'HHH' and policy == 'A'").iloc[0]
        self.assertEqual((row['groups'], row['sample_hands'], row['sample_unit']), (9, 180, 'matches'))

    def test_contract_categories_match_rollout_buckets(self):
        self.assertEqual([category_of(k) for k in ('points:30', 'points:33', 'points:42', 'marks:1', 'marks:2', 'nello')], ['30', '31-41', '42', '42', '2+ marks', 'other'])

    def test_checkpoint_names_are_linked_to_runs(self):
        run = simple_run(self.root / 'train')
        digest = loader.file_sha256(run / 'checkpoints' / 'best.pt')
        data = self.dataset(run, evaluations=[write_json(self.root / 'e.json', dict(multi_opponent(), checkpoint_id=digest))])
        row = data.evaluation[data.evaluation['source'] == 'report'].iloc[0]
        self.assertEqual(row['checkpoint_name'], 'train:best.pt')
        self.assertEqual(data.meta('train')['checkpoints']['best'], digest)


class FigureTests(VizCase):
    def test_every_family_runs_on_an_empty_selection_without_crashing(self):
        data = self.dataset(write_run(self.root / 'tiny', [dict(hands=1, updates=1, mean_reward=0.0)]))
        figures, omissions = build_figures(data, list(FAMILIES), Options())
        self.assertTrue(omissions)
        for omission in omissions:
            self.assertTrue(omission.reason and omission.required, omission.id)

    def test_learning_curves_separate_opponents_and_use_recorded_points(self):
        figures, _ = self.figures(self.dataset(simple_run(self.root / 'run')), 'learning')
        self.assertIn('learning_hand_win_rate_vs_hhh', figures)
        self.assertIn('learning_hand_win_rate_vs_chh', figures)
        series = figures['learning_hand_win_rate_vs_hhh'].panels[0].series[0]
        self.assertEqual(series.x, [200.0, 400.0])
        self.assertAlmostEqual(series.low[0], 0.42)

    def test_missing_evaluations_break_the_line_instead_of_interpolating(self):
        rows = [training_row(u, evaluate=u in (2, 4, 6, 12)) for u in range(1, 13)]
        figures, _ = self.figures(self.dataset(write_run(self.root / 'gap', rows)), 'learning')
        x = figures['learning_hand_win_rate_vs_hhh'].panels[0].series[0].x
        self.assertTrue(any(v != v for v in x), x)

    def test_multi_seed_aggregate_is_labelled_and_individual_runs_remain(self):
        for name, seed in (('s1', 1), ('s2', 2)):
            run = simple_run(self.root / name)
            config = json.loads((run / 'config.json').read_text())
            (run / 'config.json').write_text(json.dumps(dict(config, seed=seed)))
        data = self.dataset(self.root / 's1', self.root / 's2')
        figures, _ = self.figures(data, 'learning', aggregate_seeds=True)
        labels = [s.label for s in figures['learning_hand_win_rate_vs_hhh'].panels[0].series]
        self.assertIn('s1', labels)
        self.assertIn('s2', labels)
        self.assertTrue(any(label.startswith('mean of 2 seeds') for label in labels))
        self.assertIn('between-seed', figures['learning_hand_win_rate_vs_hhh'].caption)

    def test_training_reward_units_get_separate_figures(self):
        data = self.dataset(simple_run(self.root / 'hand'), simple_run(self.root / 'match', objective='match-win'))
        figures, _ = self.figures(data, 'learning')
        self.assertEqual(len(figures['learning_training_reward_signed_marks_per_hand'].panels[0].series), 1)
        self.assertEqual(len(figures['learning_training_reward_terminal_reward_per_match_1_1'].panels[0].series), 1)

    def test_critic_free_is_omitted_or_listed_never_plotted_as_zero(self):
        figures, omissions = self.figures(self.dataset(simple_run(self.root / 'none', critic='none')), 'critic')
        self.assertIn('critic_value_loss', omissions)
        self.assertIn('critic-free', omissions['critic_value_loss'].reason)
        figures, _ = self.figures(self.dataset(simple_run(self.root / 'none2', critic='none'), simple_run(self.root / 'local')), 'critic')
        self.assertIn('critic-free', figures['critic_explained_variance'].unavailable['none2'])
        self.assertEqual(len(figures['critic_explained_variance'].panels[0].series), 1)

    def test_nested_bidding_pools_counts_and_keeps_undefined_categories_nan(self):
        figures, _ = self.figures(self.dataset(simple_run(self.root / 'run')), 'bidding')
        panel = figures['bidding_contract_success'].panels[0]
        self.assertEqual(panel.categories[:2], ['30', '31-41'])
        self.assertEqual(panel.series[0].y[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(panel.series[0].y[2]))
        self.assertEqual(panel.series[0].n[0], '5/5')
        calls = figures['bidding_call_distribution'].panels[1].series[0]
        self.assertAlmostEqual(calls.y[0], 0.5)
        self.assertEqual({p.title for p in figures['bidding_call_distribution'].panels}, {'Learner partnership', 'Opponent partnership'})
        self.assertEqual(figures['bidding_voluntary_forced'].panels[0].series[0].n[1], '10/15')

    def test_score_state_figures_cover_both_objectives_without_inventing_states(self):
        data = self.dataset(simple_run(self.root / 'hand'), simple_run(self.root / 'match', objective='match-win'))
        figures, omissions = self.figures(data, 'match-bidding')
        panel = figures['match_bidding_by_score_state'].panels[0]
        hand = next(s for s in panel.series if s.label == 'hand')
        self.assertTrue(np.isnan(hand.y[0]))
        self.assertEqual(hand.y[1], 0.5)
        self.assertIn('match_bidding_exact_score_hand', omissions)
        self.assertIn('match_bidding_exact_score_match', figures)
        self.assertIn('match_bidding_objective_comparison', figures)
        self.assertIn('match_bidding_heldout_match_win_rate_vs_hhh', figures)

    def test_opponent_exposure_keeps_identities_and_realized_share(self):
        figures, _ = self.figures(self.dataset(simple_run(self.root / 'run')), 'opponents')
        realized = next(s for s in figures['opponents_configured_vs_realized_run'].panels[0].series if s.label == 'realized')
        self.assertEqual(realized.y, [0.5, 0.5])
        identity = figures['opponents_identities'].panels[0]
        self.assertEqual(identity.series[0].y, [0, 1])
        self.assertIn('not one immutable policy', figures['opponents_identities'].caption)
        self.assertAlmostEqual(next(iter(figures['opponents_training_reward'].panels[0].series)).y[0], 0.1)

    def test_evaluation_contract_and_timing_figures(self):
        figures, omissions = self.figures(self.dataset(evaluations=[write_json(self.root / 'multi.json', multi_opponent())]), 'evaluation', 'contracts')
        self.assertIn('evaluation_hand_win_rate', figures)
        self.assertIn('evaluation_hand_win_rate_by_opponent', figures)
        self.assertIn('evaluation_match_length', omissions)
        self.assertEqual(figures['evaluation_hand_win_rate'].panels[0].series[0].n[0], 'n=2 groups')
        timing = figures['contracts_decision_timing']
        self.assertEqual(timing.panels[0].categories, ['3', '5'])
        self.assertEqual(timing.panels[0].series[0].y, [0.5, 0.5])
        self.assertIn('recorded for 2 of 3 contracts', timing.caption)

    def test_round_robin_handles_missing_cells_duplicates_and_episode_separation(self):
        heuristic = agent('heuristic')
        a, a_copy, b = agent(path='r1/checkpoints/best.pt'), agent(path='copy/checkpoints/latest.pt'), agent(path='r2/checkpoints/best.pt', sha=SHA_B)
        paths = [write_json(self.root / name, data) for name, data in (('ab.json', composed(a, b, win=0.6, marks=0.3)), ('ah.json', composed(a_copy, heuristic, marks=0.5)), ('match.json', composed(a, heuristic, marks=0.9, win=0.8, episode='match', hands=10)))]
        data = self.dataset(evaluations=paths)
        figures, _ = self.figures(data, 'round-robin')
        marks = figures['round_robin_hand_signed_marks']
        labels = marks.panels[0].row_labels
        self.assertEqual(len(labels), 3)
        matrix = marks.panels[0].matrix
        ai, bi, hi = labels.index('NNN copy:latest'), labels.index('NNN r2:best'), labels.index('HHH')
        self.assertAlmostEqual(matrix[ai, bi], 0.3)
        self.assertAlmostEqual(matrix[bi, ai], -0.3)
        self.assertAlmostEqual(matrix[ai, hi], 0.5)
        self.assertTrue(np.isnan(matrix[bi, hi]))
        self.assertIn('counted once', marks.caption)
        self.assertEqual(figures['round_robin_match_win_rate'].panels[0].matrix.shape, (2, 2))
        self.assertAlmostEqual(np.nanmax(figures['round_robin_match_signed_marks'].panels[0].matrix), 0.9)

    def test_round_robin_needs_two_reports_and_explains(self):
        _, omissions = self.figures(self.dataset(evaluations=[write_json(self.root / 'one.json', composed(agent(), agent('heuristic')))]), 'round-robin')
        self.assertIn('at least two', omissions['round_robin'].reason)

    def test_partner_relations_only_trust_recorded_paths(self):
        one = lambda path, sha: [dict(path=path, sha256=sha)]
        p1, p2 = one('runs/p/checkpoints/partner-1/best.pt', SHA_A), one('runs/p/checkpoints/partner-2/best.pt', SHA_B)
        self.assertEqual(partnership_relation(p1, p2), 'original')
        self.assertEqual(partnership_relation(p1, one('runs/q/checkpoints/partner-2/best.pt', SHA_C)), 'cross')
        self.assertEqual(partnership_relation(p1, None), 'cross')
        self.assertEqual(partnership_relation(one('runs/s/checkpoints/best.pt', SHA_A), None), 'original')
        self.assertEqual(partnership_relation(p1, one('runs/p/checkpoints/partner-2/latest.pt', SHA_B)), 'unknown')
        self.assertEqual(partnership_relation(one('runs/s/checkpoints/best.pt', SHA_A), []), 'unknown')
        self.assertEqual(partnership_relation([], None), 'unknown')

    def test_partner_matrix_marks_originals_and_reports_gap(self):
        hhh = agent('heuristic')
        p1, p2 = agent(path='runs/p/checkpoints/partner-1/best.pt', sha=SHA_A), agent(path='runs/p/checkpoints/partner-2/best.pt', sha=SHA_B)
        q2 = agent(path='runs/q/checkpoints/partner-2/best.pt', sha=SHA_C)
        paths = [write_json(self.root / 'orig.json', composed(p1, hhh, a_partner=p2, marks=0.5)), write_json(self.root / 'cross.json', composed(p1, hhh, a_partner=q2, marks=0.2))]
        data = self.dataset(evaluations=paths)
        results = partner_matrices(data)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(partnership_gap(results[0]['rows'], 'mean_signed_marks_per_hand')['gap'], 0.3)
        figures, _ = self.figures(data, 'partners')
        figure = next(iter(figures.values()))
        self.assertEqual(sorted(figure.panels[0].outlines.values()), ['cross', 'original'])
        self.assertIn('+0.300', figure.caption)
        self.assertEqual(set(figure.table['relation']), {'original', 'cross'})

    def test_partner_figure_explains_required_runs(self):
        _, omissions = self.figures(self.dataset(evaluations=[write_json(self.root / 'one.json', composed(agent(), agent('heuristic')))]), 'partners')
        self.assertIn('--agent-a-partner-checkpoint', omissions['partners_compatibility'].reason)

    def test_belief_and_search_reports(self):
        data = self.dataset(searches=[write_json(self.root / 'search.json', search_report())])
        figures, omissions = self.figures(data, 'belief', 'search')
        ess = figures['belief_effective_sample_size'].panels[0]
        self.assertEqual(ess.categories, ['heuristic-likelihood', 'model-likelihood'])
        self.assertIn('belief_by_tricks_completed', figures)
        self.assertIn('not information exchange', figures['belief_true_owner_probability'].caption)
        self.assertEqual(len(figures['search_ranking_regret'].panels), 2)
        self.assertIn('search_accuracy_vs_cost', figures)
        self.assertIn('search_training_overhead', omissions)

    def test_search_without_rank_check_is_omitted_with_a_reason(self):
        data = self.dataset(searches=[write_json(self.root / 'plain.json', search_report(rank=False))])
        _, omissions = self.figures(data, 'search')
        self.assertIn('--rank-check', omissions['search_top1_agreement'].reason + omissions['search_top1_agreement'].required)

    def test_search_inside_training_and_resources(self):
        row = lambda u: training_row(u, rollout_performance=dict(search_seconds=4.0, search_wall_seconds=1.0, searched_decisions=8), metrics=dict(training_row(u)['metrics'], search_loss=0.2, search_disagreement=0.3, search_abs_advantage=0.1))
        figures, _ = self.figures(self.dataset(write_run(self.root / 'run', [row(u) for u in (1, 2)])), 'search', 'resources')
        labels = [s.label for s in figures['search_training_overhead'].panels[0].series]
        self.assertTrue(any('worker' in label for label in labels) and any('wall' in label for label in labels))
        self.assertEqual(len(figures['search_training_trajectories'].panels), 3)
        self.assertIn('resources_search_worker_vs_wall', figures)

    def test_paired_runs_plot_each_partner(self):
        partners = dict(partner_1=dict(metrics=dict(entropy=0.5), phase_samples=dict(play=10)), partner_2=dict(metrics=dict(entropy=0.6), phase_samples=dict(play=20)))
        rows = [dict(hands=u, updates=u, mean_reward=0.1, metrics=dict(partner_1={}, partner_2={}), partners=partners) for u in (1, 2)]
        figures, _ = self.figures(self.dataset(write_run(self.root / 'paired', rows)), 'ppo')
        self.assertEqual(sorted(s.label for s in figures['ppo_entropy'].panels[0].series), ['paired [partner 1]', 'paired [partner 2]'])

    def test_comparison_disclosures(self):
        a = simple_run(self.root / 'a')
        rows = [training_row(u, objective='match-win', evaluate=u == 2) for u in range(1, 9)]
        b = write_run(self.root / 'b', rows, dict(hidden_size=64, eval_every_hands=999, learn_phases=['play'], eval_opponents=['CHH']))
        data = self.dataset(a, b)
        table, warnings = compare_experiments(data)
        text = ' '.join(warnings)
        for needle in ('reward objective', 'training budget', 'trained phases', 'evaluation schedule', 'network architecture', 'evaluation opponents'):
            self.assertIn(needle, text)
        self.assertTrue(table.set_index('setting').loc['Reward objective', 'differs'])
        self.assertIn('equal simulated-hand', text)
        same = self.dataset(simple_run(self.root / 'c'), simple_run(self.root / 'd'))
        self.assertEqual(compare_experiments(same)[1], [])


class CliTests(VizCase):
    def run_output(self, *arguments):
        code, out, err = run_cli(*arguments)
        self.assertEqual(code, 0, err + out)
        return out

    def test_list_plots_names_every_family_and_preset(self):
        out = self.run_output('--list-plots')
        for name in list(FAMILIES) + list(RESEARCH_PRESETS):
            self.assertIn(name, out)

    def test_validation_errors(self):
        run = simple_run(self.root / 'run')
        output = self.root / 'output'
        cases = {'Unsupported format': ('--run', f'r={run}', '--plots', 'ppo', '--formats', 'jpeg', '--output-dir', output),
                 'Unknown plot selection': ('--run', f'r={run}', '--plots', 'nope', '--output-dir', output),
                 'does not exist': ('--run', f'r={self.root / "absent"}', '--plots', 'ppo', '--output-dir', output),
                 'incompatible': ('--run', f'r={run}', '--plots', 'ppo', '--x-axis', 'matches', '--output-dir', output),
                 'NAME=DIR': ('--run', str(run), '--plots', 'ppo', '--output-dir', output),
                 'No inputs': ('--plots', 'ppo', '--output-dir', output),
                 'unique': ('--run', f'r={run}', '--run', f'r={run}', '--plots', 'ppo', '--output-dir', output),
                 'inside the input directory': ('--run', f'r={run}', '--plots', 'ppo', '--output-dir', run / 'viz'),
                 'required': ('--run', f'r={run}', '--plots', 'ppo')}
        for message, arguments in cases.items():
            code, out, err = run_cli(*arguments)
            self.assertEqual(code, 2, message)
            self.assertIn(message, err)
        self.assertFalse(output.exists())

    def test_matches_axis_works_for_match_runs(self):
        out = self.run_output('--run', f'r={simple_run(self.root / "run", objective="match-win")}', '--plots', 'learning', '--x-axis', 'matches', '--dry-run')
        self.assertIn('learning_match_win_rate_vs_hhh', out)

    def test_dry_run_writes_nothing_and_reports_availability(self):
        run = simple_run(self.root / 'run', critic='none')
        output = self.root / 'output'
        before = tree_digest(self.root)
        out = self.run_output('--run', f'r={run}', '--plots', 'critic,partners', '--formats', 'png', '--output-dir', output, '--dry-run')
        self.assertEqual(tree_digest(self.root), before)
        self.assertFalse(output.exists())
        self.assertIn('OMITTED critic_value_loss', out)
        self.assertIn('Dry run', out)

    def test_presets_select_families_and_plots_override_or_extend(self):
        run = str(simple_run(self.root / 'run'))
        arguments = ('--run', f'r={run}', '--dry-run')
        self.assertIn('Figure families: learning, ppo, critic, evaluation, resources', self.run_output(*arguments, '--rq', 'rq1'))
        self.assertIn('Figure families: learning, ppo, critic, evaluation, resources, bidding', self.run_output(*arguments, '--rq', 'rq1', '--plots', '+bidding'))
        self.assertIn('resources, bidding, contracts\n', self.run_output(*arguments, '--rq', 'rq1', '--plots', '+bidding,contracts'))
        self.assertIn('Figure families: bidding\n', self.run_output(*arguments, '--rq', 'rq1', '--plots', 'bidding'))
        self.assertIn('do not claim', self.run_output(*arguments, '--rq', 'rq2'))

    def test_metrics_filter_selects_figures(self):
        out = self.run_output('--run', f'r={simple_run(self.root / "run")}', '--plots', 'ppo', '--metrics', 'entropy', '--dry-run')
        self.assertIn('ppo_entropy', out)
        self.assertNotIn('ppo_gradient_norm', out)

    def test_formats_and_paths_match_the_selection(self):
        output = self.root / 'output'
        self.run_output('--run', f'r={simple_run(self.root / "run")}', '--plots', 'ppo', '--metrics', 'ppo_entropy', '--formats', 'svg', '--output-dir', output)
        names = sorted(p.name for p in (output / 'figures').iterdir())
        self.assertEqual(names, ['ppo_entropy.svg'])
        self.assertFalse((output / 'index.html').exists())
        manifest = json.loads((output / 'visualization_manifest.json').read_text())
        self.assertEqual(manifest['figures'][0]['files'], ['figures/ppo_entropy.svg'])
        self.assertIn('tables/normalized_training.csv', manifest['files'])
        for name in ('normalized_training.csv', 'experiment_comparison.csv'):
            self.assertTrue((output / 'tables' / name).is_file())

    def test_rerun_removes_only_previously_generated_files(self):
        run = simple_run(self.root / 'run')
        output = self.root / 'output'
        self.run_output('--run', f'r={run}', '--plots', 'ppo', '--metrics', 'ppo_entropy', '--formats', 'png', '--output-dir', output)
        (output / 'notes.txt').write_text('keep')
        self.run_output('--run', f'r={run}', '--plots', 'ppo', '--metrics', 'ppo_approx_kl', '--formats', 'png', '--output-dir', output)
        self.assertEqual(sorted(p.name for p in (output / 'figures').iterdir()), ['ppo_approx_kl.png'])
        self.assertTrue((output / 'notes.txt').exists())

    def test_inputs_are_never_modified(self):
        pipeline = write_pipeline(self.root / 'inputs' / 'pipe')
        report_path = write_json(self.root / 'inputs' / 'search.json', search_report())
        before = tree_digest(self.root / 'inputs')
        self.run_output('--pipeline', pipeline, '--search-accuracy', report_path, '--plots', 'all', '--formats', 'png,html', '--output-dir', self.root / 'output')
        self.assertEqual(tree_digest(self.root / 'inputs'), before)

    def test_config_reproduces_figures_and_cli_overrides_are_predictable(self):
        run = simple_run(self.root / 'run')
        other = simple_run(self.root / 'other')
        config = write_json(self.root / 'config.json', dict(schema_version=1, name='saved', runs=dict(r=str(run)), plots=['ppo'], metrics=['ppo_entropy'], formats=['svg'], x_axis='hands', output_dir=str(self.root / 'one')))
        self.run_output('--config', config)
        self.run_output('--config', config, '--output-dir', self.root / 'two')
        first, second = self.root / 'one' / 'figures' / 'ppo_entropy.svg', self.root / 'two' / 'figures' / 'ppo_entropy.svg'
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual((self.root / 'one' / 'tables' / 'normalized_training.csv').read_bytes(), (self.root / 'two' / 'tables' / 'normalized_training.csv').read_bytes())
        out = self.run_output('--config', config, '--run', f'added={other}', '--plots', '+resources', '--dry-run')
        self.assertIn('Loading 2 training run(s)', out)
        self.assertIn('Figure families: ppo, resources', out)
        self.assertIn('Figure families: bidding', self.run_output('--config', config, '--plots', 'bidding', '--dry-run'))
        self.assertIn('Figure families: learning, ppo, critic', self.run_output('--config', config, '--rq', 'rq1', '--dry-run'))
        self.assertIn('Formats: pdf', self.run_output('--config', config, '--formats', 'pdf', '--dry-run'))

    def test_config_validation(self):
        for name, data, message in (('version', dict(schema_version=2), 'schema_version'), ('keys', dict(schema_version=1, bogus=True), 'unknown configuration keys'), ('rq', dict(schema_version=1, rq='rq9'), 'Unknown research-question')):
            code, _, err = run_cli('--config', write_json(self.root / f'{name}.json', data))
            self.assertEqual(code, 2)
            self.assertIn(message, err)

    def test_shipped_configurations_are_valid(self):
        directory = Path(__file__).resolve().parents[1] / 'configs' / 'visualization'
        self.assertEqual({p.name for p in directory.glob('*.json')}, {'template.json', 'critic_comparison.json', 'research_overview.json'})
        parser = cli.build_parser()
        for path in directory.glob('*.json'):
            config = cli._read_config(path, parser)
            self.assertTrue(cli.resolve_families(parser.parse_args([]), config, parser))
            self.assertTrue(cli.resolve_formats(parser.parse_args([]), config, parser))


class SmokeTests(VizCase):
    """One real render per required output type, through the CLI."""

    def test_learning_png(self):
        output = self.root / 'output'
        run_code = run_cli('--run', f'r={simple_run(self.root / "run")}', '--plots', 'learning', '--metrics', 'learning_hand_win_rate_vs_hhh', '--formats', 'png', '--output-dir', output)
        self.assertEqual(run_code[0], 0, run_code[2])
        self.assertTrue((output / 'figures' / 'learning_hand_win_rate_vs_hhh.png').read_bytes().startswith(b'\x89PNG'))

    def test_ppo_svg_is_reproducible(self):
        run = simple_run(self.root / 'run')
        outputs = [self.root / 'one', self.root / 'two']
        for output in outputs:
            self.assertEqual(run_cli('--run', f'r={run}', '--plots', 'ppo', '--metrics', 'ppo_entropy', '--formats', 'svg', '--output-dir', output)[0], 0)
        text = (outputs[0] / 'figures' / 'ppo_entropy.svg').read_text()
        self.assertIn('<svg', text)
        self.assertEqual(text, (outputs[1] / 'figures' / 'ppo_entropy.svg').read_text())

    def test_evaluation_pdf(self):
        output = self.root / 'output'
        path = write_json(self.root / 'multi.json', multi_opponent())
        self.assertEqual(run_cli('--evaluation', f'e={path}', '--plots', 'evaluation', '--metrics', 'evaluation_hand_win_rate', '--formats', 'pdf', '--output-dir', output)[0], 0)
        self.assertTrue((output / 'figures' / 'evaluation_hand_win_rate.pdf').read_bytes().startswith(b'%PDF'))

    def test_interactive_html_needs_no_network(self):
        output = self.root / 'output'
        self.assertEqual(run_cli('--run', f'r={simple_run(self.root / "run")}', '--plots', 'ppo', '--metrics', 'ppo_entropy', '--formats', 'html', '--output-dir', output)[0], 0)
        text = (output / 'figures' / 'ppo_entropy.html').read_text()
        self.assertIn('src="plotly.min.js"', text)
        self.assertNotIn('cdn.', text)
        self.assertTrue((output / 'figures' / 'plotly.min.js').is_file())

    def test_research_report(self):
        output = self.root / 'output'
        pipeline = write_pipeline(self.root / 'pipe')
        path = write_json(self.root / 'search.json', search_report())
        code = run_cli('--pipeline', pipeline, '--search-accuracy', f'accuracy={path}', '--plots', 'all', '--formats', 'html,png', '--output-dir', output)
        self.assertEqual(code[0], 0, code[2])
        page = (output / 'index.html').read_text()
        for needle in ('<nav>', 'href="#learning"', 'href="#belief"', 'Omitted figures', 'partners_compatibility', 'hand-marks', 'figures/ppo_entropy.png', 'iframe', 'Checkpoint identity', 'Explained variance'):
            self.assertIn(needle, page)
        self.assertNotIn('http://', page)
        self.assertNotIn('https://', page)
        manifest = json.loads((output / 'visualization_manifest.json').read_text())
        self.assertTrue(manifest['omissions'])
        self.assertTrue(all(o['reason'] and o['required'] for o in manifest['omissions']))
        self.assertTrue(all((output / name).is_file() for name in manifest['files']))

    def test_report_can_be_forced_for_static_formats(self):
        output = self.root / 'output'
        self.assertEqual(run_cli('--run', f'r={simple_run(self.root / "run")}', '--plots', 'ppo', '--metrics', 'ppo_entropy', '--formats', 'png', '--report', '--output-dir', output)[0], 0)
        self.assertIn('<img src="figures/ppo_entropy.png"', (output / 'index.html').read_text())

    def test_round_robin_heatmap(self):
        output = self.root / 'output'
        a, b, h = agent(path='r1/checkpoints/best.pt'), agent(path='r2/checkpoints/best.pt', sha=SHA_B), agent('heuristic')
        first = write_json(self.root / 'ab.json', composed(a, b))
        second = write_json(self.root / 'ah.json', composed(a, h))
        code = run_cli('--evaluation', f'ab={first}', '--evaluation', f'ah={second}', '--plots', 'round-robin', '--formats', 'png,html', '--annotate', 'ci', '--output-dir', output)
        self.assertEqual(code[0], 0, code[2])
        self.assertTrue((output / 'figures' / 'round_robin_hand_signed_marks.png').read_bytes().startswith(b'\x89PNG'))
        self.assertTrue((output / 'tables' / 'round_robin_hand_signed_marks.csv').is_file())


if __name__ == '__main__':
    unittest.main()
