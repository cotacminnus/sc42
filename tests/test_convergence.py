"""Convergence diagnostics: synthetic drift/cycle cases, record classification and the report CLI."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import contextlib
    import io
    import json
    import math
    import tempfile
    from pathlib import Path
    import torch
    from examples import convergence_report as cr
    from examples import run_pipeline as rp
    from research.ppo import convergence as cv


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def tabular(rows):
    """[states, 3] masked log-probabilities from rows of rock/paper/scissors probabilities."""
    return torch.tensor(rows, dtype=torch.float32).log()


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class MetricTests(unittest.TestCase):
    """Pure functions on synthetic rock-paper-scissors sequences."""
    masks = None

    def setUp(self):
        self.masks = torch.ones((1, 3), dtype=torch.bool)

    def test_fixed_opponent_best_response_converges_without_recurrence(self):
        # Against "always rock", the learner moves monotonically toward paper.
        policies = [tabular([[(1 - q) / 2, q, (1 - q) / 2]]) for q in (0.4, 0.7, 0.85, 0.925, 0.9625)]
        result = cv.drift(policies, self.masks)
        steps = result['consecutive_kl']
        self.assertTrue(all(a > b for a, b in zip(steps, steps[1:])), steps)
        self.assertEqual(result['recurrent'], [])
        self.assertEqual(result['kl_to_final'][-1], 0.0)
        self.assertEqual(result['consecutive_greedy_change'], [0.0] * 4)

    def test_self_play_chase_is_recurrent(self):
        rock, paper, scissors = [[0.8, 0.1, 0.1]], [[0.1, 0.8, 0.1]], [[0.1, 0.1, 0.8]]
        result = cv.drift([tabular(p) for p in (rock, paper, scissors, rock, paper, scissors)], self.masks)
        self.assertEqual([(r['index'], r['nearest_earlier']) for r in result['recurrent']], [(3, 0), (4, 1), (5, 2)])
        self.assertAlmostEqual(min(result['consecutive_kl']), max(result['consecutive_kl']), places=5)  # No decay.
        self.assertEqual(result['consecutive_greedy_change'], [1.0] * 5)

    def test_stationary_wandering_is_not_recurrence(self):
        # Steps far below the floor: which near-identical earlier policy is nearest means nothing.
        base = [1 / 3, 1 / 3, 1 / 3]
        wobble = [[[base[0] + d, base[1] - d, base[2]]] for d in (0.0, 0.004, 0.001, 0.003)]
        self.assertEqual(cv.drift([tabular(p) for p in wobble], self.masks)['recurrent'], [])

    def test_drift_uses_legal_moves_only(self):
        masks = torch.tensor([[True, True, False]])
        old = torch.tensor([[0.5, 0.5, 0.0]]).log()
        new = torch.tensor([[0.25, 0.75, 0.0]]).log()
        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        self.assertAlmostEqual(cv.drift([old, new], masks)['consecutive_kl'][0], expected, places=6)

    def test_crossplay_cycle_and_transitive_progress(self):
        significant = (0.3, 0.1, 0.5)
        cyclic = {('P', 'R'): significant, ('S', 'P'): significant, ('R', 'S'): significant}
        summary = cv.crossplay_summary(cyclic, ['R', 'P', 'S'])
        self.assertEqual(len(summary['cycles']), 1)
        self.assertEqual(summary['transitive_progress'], dict(later_better=2, later_worse=1))
        transitive = {('b', 'a'): significant, ('c', 'a'): significant, ('c', 'b'): significant, ('c', 'HHH'): (0.1, -0.1, 0.3)}
        summary = cv.crossplay_summary(transitive, ['a', 'b', 'c'], panel=('HHH',), hands=dict(a=0, b=1, c=2))
        self.assertEqual(summary['cycles'], [])
        self.assertEqual(summary['transitive_progress'], dict(later_better=3))
        self.assertEqual([r['opponent'] for r in summary['final_vs_earlier']], ['a', 'b'])
        self.assertEqual(summary['panel']['HHH']['points'], [dict(policy='c', hands=2, marks=0.1, low=-0.1, high=0.3)])
        # The mirror of a recorded comparison is implied (duplicate play is zero-sum).
        mirrored = cv.crossplay_summary({('a', 'b'): (-0.3, -0.5, -0.1)}, ['a', 'b'])
        self.assertEqual(mirrored['final_vs_earlier'], [dict(opponent='a', marks=0.3, low=0.1, high=0.5)])

    def test_plateau(self):
        points = [dict(hands=h, marks=m, low=m - 0.05, high=m + 0.05)
                  for h, m in ((1, -0.4), (2, -0.1), (3, 0.1), (4, 0.12), (5, 0.11))]
        result = cv.plateau(points)
        self.assertEqual((result['plateau_hands'], result['best_hands'], result['final_marks']), (3, 4, 0.11))
        wide = [dict(p, low=p['marks'] - 1, high=p['marks'] + 1) for p in points]
        self.assertEqual(cv.plateau(wide)['plateau_hands'], 1)  # Wide intervals: an early, uninformative plateau.

    def test_generalization_gap(self):
        gap = cv.generalization_gap(dict(HHH=0.2, CHH=0.1, anchor=0.0), ['HHH'])
        self.assertEqual((gap['held_out'], gap['held_out_marks']), (['CHH', 'anchor'], 0.05))
        self.assertAlmostEqual(gap['gap'], 0.15)
        self.assertIsNone(cv.generalization_gap(dict(HHH=0.2), ['HHH']))
        self.assertIsNone(cv.generalization_gap(dict(HHH=0.2), ['self-snapshots'])['gap'])

    def test_opponents_are_classified_by_sha_not_name(self):
        anchor, snapshot = 'a' * 64, 'b' * 64
        identities = {anchor: ('anchor', 'anchor'), snapshot: ('snapshot', 'policy-000000004')}
        rows = [dict(opponent_assignments=[[0, 'self-snapshots', anchor], [1, 'self-snapshots', anchor],
                                           [2, 'HHH', None]]),
                dict(opponent_assignments=[[4, 'self-snapshots', snapshot], [5, 'self-current', 'c' * 64],
                                           [6, 'self-snapshots', 'd' * 64], [7, 'renamed-pool', 'e' * 64]])]
        exposure = cv.opponent_exposure(rows, identities, window_hands=4)
        self.assertEqual(exposure['by_class'], dict(heuristic=1 / 7, anchor=2 / 7, snapshot=1 / 7,
                                                    current=2 / 7, unknown=1 / 7))
        relabelled = {(r['name'], r['cls']): r for r in exposure['relabelled']}
        # Drawn as self-snapshots but played the anchor (the pre-snapshot fallback).
        self.assertEqual((relabelled['self-snapshots', 'anchor']['hands'],
                          relabelled['self-snapshots', 'anchor']['last']), (2, 1))
        # Drawn as self-snapshots but played the learner's own weights (a 'current' fallback).
        self.assertEqual(relabelled['self-snapshots', 'current']['hands'], 1)
        self.assertIn(('renamed-pool', 'unknown'), relabelled)  # Unknown names are reported, not fatal.
        self.assertEqual(len(relabelled), 3)
        self.assertEqual([w['start'] for w in exposure['windows']], [0, 4])
        # Version-1 rows record names only.
        legacy = cv.opponent_exposure([dict(training_by_opponent=dict(HHH=dict(hands=3))),
                                       dict(training_by_opponent=dict(HHH=dict(hands=1)))], {})
        self.assertEqual((legacy['hands'], legacy['identities_recorded'], legacy['by_class']), (4, False, dict(heuristic=1.0)))

    def test_weights_digest_ignores_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {'w': torch.arange(6, dtype=torch.float32).reshape(2, 3), 'b': torch.zeros(2)}
            paths = [Path(directory) / f'{i}.pt' for i in range(3)]
            torch.save(dict(model=state, metadata=dict(note='snapshot')), paths[0])
            torch.save(dict(model=state, metadata=dict(note='best'), progress=dict(hands=4)), paths[1])
            torch.save(dict(model={**state, 'b': torch.ones(2)}, metadata=dict(note='snapshot')), paths[2])
            digests = [cv.weights_digest(p) for p in paths]
            self.assertNotEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertEqual(digests[0], digests[1])
            self.assertNotEqual(digests[0], digests[2])

    def test_probe_is_deterministic_neutral_and_by_phase(self):
        first, second = cv.probe_positions(7, 2), cv.probe_positions(7, 2)
        for phase in ('bid', 'trump', 'play'):
            self.assertTrue(torch.equal(first[phase]['features'], second[phase]['features']))
            self.assertTrue(bool((first[phase]['masks'].sum(-1) >= 2).all()))
            self.assertTrue(bool((first[phase]['features'][:, :3].argmax(-1)
                                  == ('bid', 'trump', 'play').index(phase)).all()))
        self.assertFalse(torch.equal(first['play']['features'], cv.probe_positions(8, 2)['play']['features']))


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class LineageTests(unittest.TestCase):
    """Tiny version-1, shared self-snapshot and paired runs, through the report CLI."""

    @classmethod
    def setUpClass(cls):
        from examples.train_ppo import main as train_legacy
        from examples.train_selfplay_ppo import main as train
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        quiet(train_legacy, ['--total-hands', '4', '--rollout-play-steps', '28', '--hidden-size', '8',
                             '--minibatch-size', '28', '--eval-deals', '2', '--bootstrap-samples', '5',
                             '--eval-every-hands', '2', '--device', 'cpu',
                             '--output-dir', str(cls.path / 'base'), '--log-dir', str(cls.path / 'base-logs')])
        cls.anchor = cls.path / 'base' / 'checkpoints' / 'latest.pt'
        common = ['--init-play-checkpoint', str(cls.anchor), '--learn-phases', 'bid,play', '--hidden-size', '8',
                  '--rollout-decisions', '8', '--ppo-epochs', '1', '--minibatch-size', '4', '--eval-deals', '2',
                  '--bootstrap-samples', '5', '--forgetting-hands', '1', '--eval-every-hands', '2',
                  '--device', 'cpu', '--seed', '42']
        quiet(train, [*common, '--opponent-pool', 'self-snapshots:1', '--snapshot-every-hands', '2',
                      '--total-hands', '6', '--eval-opponents', 'HHH,CHH', '--selection-opponent', 'HHH',
                      '--output-dir', str(cls.path / 'shared'), '--log-dir', str(cls.path / 'shared-logs')])
        quiet(train, [*common, '--opponent-pool', 'HHH:1', '--paired-partners', '--total-hands', '2',
                      '--eval-opponents', 'HHH,anchor', '--output-dir', str(cls.path / 'paired'),
                      '--log-dir', str(cls.path / 'paired-logs')])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def report(self, *extra):
        output = self.path / 'report.json'
        quiet(cr.main, ['--run', f'base={self.path / "base"}', '--run', f'shared={self.path / "shared"}',
                        '--run', f'paired={self.path / "paired"}', '--probe-hands', '2', '--output', str(output),
                        '--log-dir', str(self.path / 'report-logs'), *extra])
        return json.loads(output.read_text(encoding='utf-8'))

    def test_lineages(self):
        base, shared, paired = (cv.lineage(self.path / n) for n in ('base', 'shared', 'paired'))
        self.assertEqual((base['version'], base['paired']), (1, False))
        self.assertEqual([p['label'] for p in base['sequences']['policy']][-1], 'latest')
        self.assertEqual(shared['sequences']['policy'][0]['role'], 'initial')
        self.assertTrue(any(p['role'] == 'snapshot' for p in shared['sequences']['policy']))
        self.assertTrue(paired['paired'])
        self.assertEqual(set(paired['sequences']), {'partner-1', 'partner-2'})
        self.assertTrue(any('paired run' in w for w in paired['warnings']))
        self.assertTrue(any('version-1 progress' in w for w in base['warnings']))
        latest = next(p for p in shared['policies'] if p['role'] == 'latest')
        last_snapshot = max((p for p in shared['policies'] if p['role'] == 'snapshot'), key=lambda p: p['hands'])
        self.assertEqual(latest['weights'], last_snapshot['weights'])  # Published at the final update.
        self.assertNotEqual(latest['sha256'], last_snapshot['sha256'])
        self.assertTrue(any('same weights under different file SHAs' in w for w in shared['warnings']))

    def test_report_matches_the_records(self):
        report = self.report()
        shared = report['runs']['shared']
        self.assertIn('play', shared['drift'])
        self.assertIn('bid', shared['drift'])
        self.assertEqual(set(report['runs']['paired']['partner_divergence']), {'bid', 'play'})
        self.assertIn('partner-1 play', report['runs']['paired']['drift'])
        # Every self-snapshots hand that logged the anchor's SHA is reported as such,
        # whichever fallback produced it.
        rows = [json.loads(l) for l in (self.path / 'shared' / 'progress.jsonl').read_text().splitlines()]
        anchor = rows[-1]['anchor_sha256']
        logged = sum(1 for r in rows for _, name, sha in r['opponent_assignments']
                     if name == 'self-snapshots' and sha == anchor)
        relabelled = sum(r['hands'] for r in shared['exposure']['relabelled'] if r['cls'] == 'anchor')
        self.assertEqual(relabelled, logged)
        self.assertEqual(shared['exposure']['hands'], 6)
        self.assertEqual(set(shared['strength']), {'HHH', 'CHH'})
        self.assertEqual(report['runs']['base']['exposure']['identities_recorded'], False)

    def test_crossplay_pipeline_and_results(self):
        config_path, root = self.path / 'crossplay.json', self.path / 'crossplay-out'
        flags = ['--panel', 'HHH', '--deals', '2', '--crossplay-policies', '2', '--pipeline-name', 'convergence-test',
                 '--pipeline-root', str(root)]
        self.report(*flags, '--crossplay-pipeline', str(config_path))
        pipeline = rp.load_config(config_path)
        rp.plan(pipeline)  # Every step validates with evaluate_agent_vs_agent's parser.
        kinds = {s.id.split('-')[0] for s in pipeline.steps}
        self.assertTrue({'base', 'shared', 'paired', 'final', 'mixed'} <= kinds)
        self.assertTrue(any('agent-a-partner-checkpoint' in s.args for s in pipeline.steps if s.id.startswith('paired-')))
        # Fake each step's result: agent A wins by +0.5 with a significant interval.
        for step in pipeline.steps:
            output = rp.output_paths(step)['output']
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(dict(
                metrics=[dict(policy='A', mean_signed_marks=0.5, games_played=4)],
                intervals=[dict(policy='A', mean_signed_marks=dict(low=0.25, high=0.75))])))
        report = self.report(*flags, '--crossplay-results', str(config_path))
        crossplay = report['crossplay']
        self.assertEqual(crossplay['missing_steps'], [])
        self.assertEqual(crossplay['runs']['shared']['transitive_progress'], dict(later_better=1))
        self.assertEqual(crossplay['runs']['shared']['cycles'], [])
        self.assertTrue(all(row['compatibility'] == 0.0 for row in crossplay['partners']))
        # A pipeline generated for other settings is refused rather than misread.
        with self.assertRaises(SystemExit):
            quiet(cr.main, ['--run', f'base={self.path / "base"}', '--probe-hands', '1', *flags, '--panel', 'CHH',
                            '--crossplay-results', str(config_path), '--log-dir', str(self.path / 'mismatch-logs')])


if __name__ == '__main__':
    unittest.main()
