"""JSON experiment pipelines: parsing, arguments, references, execution, recovery, integration."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from examples import run_pipeline as rp

TORCH = importlib.util.find_spec('torch') is not None
CONFIGS = Path(__file__).resolve().parents[1] / 'configs' / 'pipelines'


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        result = function(*args, **kwargs)
    return result, out.getvalue()


class FakeRunner:
    """Records commands and writes the outputs a successful command would leave."""

    def __init__(self, fail=None, code=3):
        self.calls, self.fail, self.code = [], fail, code

    def __call__(self, command, cwd):
        assert command[:2] == [sys.executable, '-m'] and cwd == rp.REPOSITORY
        self.calls.append(command)
        module = command[2].split('.')[-1]
        if self.fail == module or self.fail == len(self.calls):
            return SimpleNamespace(returncode=self.code)
        option = lambda name: command[command.index(name) + 1]
        if rp.COMMANDS[module] == 'training':
            directory = Path(option('--output-dir'))
            (directory / 'checkpoints').mkdir(parents=True, exist_ok=True)
            for name in ('best', 'latest'):
                (directory / 'checkpoints' / f'{name}.pt').write_bytes(f'{directory}:{name}'.encode())
            (directory / 'progress.jsonl').write_text('{"hands": 1}\n')
        else:
            Path(option('--output')).parent.mkdir(parents=True, exist_ok=True)
            Path(option('--output')).write_text('{}')
        return SimpleNamespace(returncode=0)


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def write(self, config, name='pipeline.json'):
        path = self.path / name
        path.write_text(json.dumps(config) if not isinstance(config, str) else config)
        return path

    def config(self, steps=None, **changes):
        steps = steps if steps is not None else [
            dict(id='play', command='train_selfplay_ppo',
                 args={'from-scratch': True, 'learner-play': 'neural-train', 'total-hands': 4}),
            dict(id='bidding', command='train_selfplay_ppo',
                 args={'init-checkpoint': '${play.best}', 'learner-bid': 'neural-train',
                       'learner-bid-source': 'new', 'learner-play': 'neural-frozen'}),
            dict(id='evaluation', command='evaluate_selfplay_ppo', args={'checkpoint': '${bidding.latest}'}),
        ]
        return dict(dict(schema_version=1, name='test', output_root=str(self.path / 'out'), steps=steps), **changes)


class ParsingTests(PipelineTestCase):
    def test_valid_and_example_configurations(self):
        pipeline = rp.load_config(self.write(self.config()))
        self.assertEqual([s.id for s in pipeline.steps], ['play', 'bidding', 'evaluation'])
        self.assertEqual(pipeline.output_root, self.path / 'out')
        for path in sorted(CONFIGS.glob('*.json')):
            with self.subTest(path=path.name):
                pipeline = rp.load_config(path)
                rows = rp.plan(pipeline)  # Each command's own argparse accepts every step.
                self.assertEqual(len(rows), len(pipeline.steps))
        self.assertEqual({p.name for p in CONFIGS.glob('*.json')},
                         {'template.json', 'progressive_training.json', 'scratch_selfplay.json'})

    def test_rejections(self):
        base = self.config()
        cases = {
            'malformed': '{"schema_version": 1,',
            'version': dict(base, schema_version=2),
            'version-type': dict(base, schema_version='1'),
            'missing': {k: v for k, v in base.items() if k != 'output_root'},
            'unknown-top': dict(base, extra=1),
            'empty': dict(base, steps=[]),
            'duplicate': dict(base, steps=[base['steps'][0], base['steps'][0]]),
            'command': dict(base, steps=[dict(id='x', command='rm -rf /')]),
            'shell': dict(base, steps=[dict(id='x', command='examples.train_ppo')]),
            'step-field': dict(base, steps=[dict(id='x', command='train_ppo', shell=True)]),
            'step-id': dict(base, steps=[dict(id='../x', command='train_ppo')]),
            'missing-id': dict(base, steps=[dict(command='train_ppo')]),
        }
        for name, config in cases.items():
            with self.subTest(name=name), self.assertRaises(rp.PipelineError):
                rp.load_config(self.write(config, name + '.json'))
        with self.assertRaises(rp.PipelineError):
            rp.load_config(self.path / 'missing.json')


class ArgumentTests(PipelineTestCase):
    def step(self, args, command='train_selfplay_ppo'):
        return rp.Step(0, 's', command, args)

    def test_conversion(self):
        argv = rp.to_argv(self.step({'from-scratch': True, 'resume': False, 'total-hands': 262144,
                                     'learning-rate': rp.Number('0.0001'), 'opponent-pool': 'HHH:0.3,self-current:0.7',
                                     'opponent-checkpoint': ['a=x.pt', 'b=y.pt'], 'seed': 42}))
        self.assertEqual(argv, ['--from-scratch', '--total-hands', '262144', '--learning-rate', '0.0001',
                                '--opponent-pool', 'HHH:0.3,self-current:0.7', '--opponent-checkpoint', 'a=x.pt',
                                '--opponent-checkpoint', 'b=y.pt', '--seed', '42'])
        text = '{"schema_version":1,"name":"n","output_root":"o","steps":[{"id":"s","command":"train_ppo",' \
               '"args":{"learning-rate":0.00030,"clip-epsilon":1e-1}}]}'
        pipeline = rp.load_config(self.write(text))
        self.assertEqual(rp.to_argv(pipeline.steps[0]), ['--learning-rate', '0.00030', '--clip-epsilon', '1e-1'])
        for args in ({'seed': None}, {'seed': {'a': 1}}, {'x': [True]}, {'x': [[1]]}, {'x': []},
                     {'--seed': 1}, {'Seed': 1}, {'seed': [None]}):
            with self.subTest(args=args), self.assertRaises(rp.PipelineError):
                rp.to_argv(self.step(args))

    def test_authoritative_parser_validation(self):
        for args in ({'no-such-flag': 1}, {'total-hands': 'many'}, {'critic-mode': 'global'},
                     {'from-scratch': 'yes'}, {'from-scratch': True, 'init-checkpoint': 'x.pt'}):
            config = self.config([dict(id='s', command='train_selfplay_ppo', args=args)])
            with self.subTest(args=args), self.assertRaisesRegex(rp.PipelineError, 'rejects its arguments'):
                rp.plan(rp.parse_config(config))


class ReferenceTests(PipelineTestCase):
    def test_references_and_outputs(self):
        pipeline = rp.parse_config(self.config())
        play, bidding, evaluation = pipeline.steps
        args, dependencies = rp.resolve_args(bidding, pipeline, require=False)
        best = self.path / 'out/play/checkpoints/best.pt'
        self.assertEqual(args['init-checkpoint'], str(best))
        self.assertEqual(dependencies, [dict(reference='play.best', path=str(best), sha256=None)])
        argv = rp.command_argv(evaluation, rp.resolve_args(evaluation, pipeline, require=False)[0])
        self.assertIn(str(self.path / 'out/bidding/checkpoints/latest.pt'), argv)
        self.assertEqual(argv[-4:], ['--output', str(self.path / 'out/evaluation/result.json'),
                                     '--log-dir', str(self.path / 'out/evaluation/logs')])
        with self.assertRaisesRegex(rp.PipelineError, 'does not exist'):
            rp.resolve_args(bidding, pipeline)
        best.parent.mkdir(parents=True)
        (best.parent / 'latest.pt').write_bytes(b'latest')
        with self.assertRaisesRegex(rp.PipelineError, 'only latest.pt exists'):
            rp.resolve_args(bidding, pipeline)
        best.write_bytes(b'best')
        _, dependencies = rp.resolve_args(bidding, pipeline)
        self.assertEqual(dependencies[0]['sha256'], rp.sha256_file(best))
        relative = rp.parse_config(self.config(output_root='runs/pipelines/x'))
        self.assertEqual(relative.output_root, rp.REPOSITORY / 'runs/pipelines/x')

    def test_invalid_references_and_collisions(self):
        train = lambda sid, **args: dict(id=sid, command='train_selfplay_ppo', args=args)
        evaluate = lambda sid, **args: dict(id=sid, command='evaluate_selfplay_ppo', args=args)
        cases = [
            [train('a', **{'init-checkpoint': '${b.best}'}), train('b')],        # forward
            [train('a', **{'init-checkpoint': '${a.best}'})],                    # self
            [train('a', **{'init-checkpoint': '${zzz.best}'})],                  # unknown
            [train('a'), train('b', **{'init-checkpoint': '${a.final}'})],       # artifact
            [train('a'), train('b', **{'init-checkpoint': '${a.best'})],         # malformed
            [train('a'), train('b', **{'init-checkpoint': '${eval("x")}'})],     # expression
            [evaluate('e'), train('b', **{'init-checkpoint': '${e.best}'})],     # report as checkpoint
            [train('a'), evaluate('e', checkpoint='${a.output}')],               # checkpoint as report
            [train('a'), train('b', **{'output-dir': str(self.path / 'out/a')})],
            [train('a', **{'output-dir': str(self.path / 'shared')}),
             train('b', **{'output-dir': str(self.path / 'shared/inner')})],
            [evaluate('e', output=str(self.path / 'out/pipeline_state.json'))],
        ]
        for steps in cases:
            with self.subTest(steps=steps), self.assertRaises(rp.PipelineError):
                rp.parse_config(self.config(steps))


class ExecutionTests(PipelineTestCase):
    def run_pipeline(self, config, runner, resume=False):
        return quiet(rp.run, rp.parse_config(config), resume=resume, runner=runner)

    def manifest(self):
        return json.loads((self.path / 'out' / rp.STATE_FILE).read_text())

    def test_order_progress_manifest_and_dry_run(self):
        config = self.config()
        (code, output) = quiet(rp.main, ['--config', str(self.write(config)), '--dry-run'])
        self.assertEqual(code, 0)
        self.assertFalse((self.path / 'out').exists())  # Dry run creates nothing.
        self.assertIn('depends on ${play.best}', output)
        runner = FakeRunner()
        code, output = self.run_pipeline(config, runner)
        self.assertEqual(code, 0)
        self.assertEqual([c[2] for c in runner.calls], ['examples.train_selfplay_ppo'] * 2 + ['examples.evaluate_selfplay_ppo'])
        self.assertIn('[1/3] play       TRAINING', output)
        self.assertIn('[3/3] evaluation COMPLETED', output)
        self.assertIn('Pipeline completed successfully.', output)
        manifest = self.manifest()
        self.assertEqual([s['state'] for s in manifest['steps']], ['completed'] * 3)
        self.assertEqual(manifest['steps'][1]['dependencies'][0]['reference'], 'play.best')
        self.assertEqual(manifest['steps'][1]['argv'], runner.calls[1])
        self.assertIn('sha', manifest['git'])
        self.assertEqual(manifest['config'], json.loads(json.dumps(config)))
        with self.assertRaisesRegex(rp.PipelineError, 'use --resume'):
            self.run_pipeline(config, FakeRunner())

    def test_stop_on_failure_exit_status_and_resume_without_repetition(self):
        config = self.config()
        code, output = self.run_pipeline(config, FakeRunner(fail='evaluate_selfplay_ppo', code=7))
        self.assertEqual(code, 7)
        self.assertIn('FAILED (exit status 7)', output)
        self.assertEqual([s['state'] for s in self.manifest()['steps']], ['completed', 'completed', 'failed'])
        failing = FakeRunner(fail=2, code=5)
        other = self.config(output_root=str(self.path / 'other'))
        code, _ = quiet(rp.run, rp.parse_config(other), runner=failing)
        self.assertEqual((code, len(failing.calls)), (5, 2))  # Nothing after the failed step runs.
        retry = FakeRunner()
        code, output = self.run_pipeline(config, retry, resume=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(retry.calls), 1)  # Completed steps are not repeated.
        self.assertIn('SKIPPED', output)
        self.assertEqual(self.manifest()['steps'][2]['attempts'][-1]['mode'], 'retry')
        code, _ = self.run_pipeline(config, FakeRunner(), resume=True)
        self.assertEqual(code, 0)

    def test_resume_detects_changes(self):
        config = self.config()
        self.run_pipeline(config, FakeRunner())
        changed = json.loads(json.dumps(config))
        changed['steps'][0]['args']['total-hands'] = 5
        with self.assertRaisesRegex(rp.PipelineError, 'configuration changed'):
            self.run_pipeline(changed, FakeRunner(), resume=True)
        latest = self.path / 'out/bidding/checkpoints/latest.pt'
        latest.write_bytes(b'tampered')
        with self.assertRaisesRegex(rp.PipelineError, 'modified'):
            self.run_pipeline(config, FakeRunner(), resume=True)
        latest.unlink()
        with self.assertRaisesRegex(rp.PipelineError, 'missing'):
            self.run_pipeline(config, FakeRunner(), resume=True)
        with self.assertRaisesRegex(rp.PipelineError, 'No manifest'):
            self.run_pipeline(self.config(output_root=str(self.path / 'new')), FakeRunner(), resume=True)

    def test_missing_outputs_and_partial_training_without_checkpoint(self):
        def silent(command, cwd):  # Exit 0 without writing anything.
            return SimpleNamespace(returncode=0)
        code, _ = self.run_pipeline(self.config(), silent)
        self.assertEqual(code, 1)
        self.assertIn('produced no latest.pt', self.manifest()['steps'][0]['attempts'][0]['error'])

        def crash(command, cwd):  # Progress written, no checkpoint yet.
            directory = Path(command[command.index('--output-dir') + 1])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'progress.jsonl').write_text('{"hands": 1}\n')
            return SimpleNamespace(returncode=1)
        config = self.config(output_root=str(self.path / 'crash'))
        quiet(rp.run, rp.parse_config(config), runner=crash)
        never = FakeRunner()
        with self.assertRaisesRegex(rp.PipelineError, 'no latest.pt to resume'):
            quiet(rp.run, rp.parse_config(config), resume=True, runner=never)
        self.assertEqual(never.calls, [])  # Never silently restarts from random weights.

    def test_external_inputs_are_verified_and_recorded(self):
        missing = self.config([dict(id='e', command='evaluate_selfplay_ppo',
                                    args={'checkpoint': str(self.path / 'nope.pt')})])
        with self.assertRaisesRegex(rp.PipelineError, 'does not exist'):
            self.run_pipeline(missing, FakeRunner())
        external = self.path / 'given.pt'
        external.write_bytes(b'x')
        present = self.config([dict(id='e', command='evaluate_selfplay_ppo', args={'checkpoint': str(external)})],
                              output_root=str(self.path / 'given'))
        quiet(rp.run, rp.parse_config(present), runner=FakeRunner())
        record = json.loads((self.path / 'given' / rp.STATE_FILE).read_text())['steps'][0]
        self.assertEqual(record['inputs'], [dict(argument='checkpoint', path=str(external),
                                                 sha256=rp.sha256_file(external))])


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class IntegrationTests(PipelineTestCase):
    """Real child processes with tiny budgets; mechanics only, not policy strength."""

    TINY = {'hidden-size': 8, 'eval-deals': 2, 'bootstrap-samples': 5, 'forgetting-hands': 1,
            'minibatch-size': 32, 'ppo-epochs': 1, 'device': 'cpu'}

    def train(self, sid, **args):
        tiny = dict(self.TINY)
        if 'init-checkpoint' in args:
            tiny.pop('hidden-size')
        return dict(id=sid, command='train_selfplay_ppo', args={**tiny, 'total-hands': 8,
                                                                'rollout-decisions': 64, **args})

    def execute(self, config, **kwargs):
        logs = []

        def captured(command, cwd):  # Keep child output out of the test report unless it fails.
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            logs.append(result.stdout[-2000:] + result.stderr[-2000:])
            return result
        code, output = quiet(rp.run, rp.parse_config(config), **{'runner': captured, **kwargs})
        self.assertEqual(code, 0, output + '\n'.join(logs))
        return json.loads((Path(config['output_root']) / rp.STATE_FILE).read_text())

    def test_progressive_curriculum_and_self_current(self):
        hhh = {'opponent-pool': 'HHH:1', 'eval-opponents': 'HHH,CHH', 'selection-opponent': 'HHH'}
        steps = [
            self.train('play', **{'from-scratch': True, 'learner-bid': 'heuristic', 'learner-trump': 'heuristic',
                                  'learner-play': 'neural-train', **hhh}),
            self.train('bidding', **{'init-checkpoint': '${play.best}', 'learner-bid': 'neural-train',
                                     'learner-bid-source': 'new', 'learner-trump': 'heuristic',
                                     'learner-play': 'neural-frozen', **hhh}),
            self.train('trump', **{'init-checkpoint': '${bidding.best}', 'learner-bid': 'neural-frozen',
                                   'learner-trump': 'neural-train', 'learner-trump-source': 'new',
                                   'learner-play': 'neural-frozen', 'rollout-decisions': 8,
                                   'max-rollout-hands': 32, 'total-hands': 64, **hhh}),
            self.train('joint', **{'init-checkpoint': '${trump.best}', 'learner-bid': 'neural-train',
                                   'learner-trump': 'neural-train', 'learner-play': 'neural-train',
                                   'critic-mode': 'centralized', 'opponent-pool': 'HHH:0.3,self-snapshots:0.7',
                                   'eval-opponents': 'HHH,CHH', 'selection-opponent': 'HHH',
                                   'snapshot-every-hands': 4}),
            dict(id='evaluation', command='evaluate_agent_vs_agent', args={
                'agent-a-checkpoint': '${joint.best}', 'agent-a-bid': 'neural', 'agent-a-trump': 'neural',
                'agent-a-play': 'neural', 'agent-b-checkpoint': '${trump.best}', 'agent-b-bid': 'neural',
                'agent-b-trump': 'neural', 'agent-b-play': 'neural', 'deals': 4, 'bootstrap-samples': 10}),
            self.train('scratch-selfplay', **{'from-scratch': True, 'opponent-pool': 'HHH:0.3,self-current:0.7',
                                              'eval-opponents': 'HHH,CHH', 'selection-opponent': 'HHH'}),
        ]
        manifest = self.execute(self.config(steps))
        self.assertEqual([s['state'] for s in manifest['steps']], ['completed'] * 6)
        from research.ppo.checkpoint import load_model
        expected = {'play': (('play',), ('play',)), 'bidding': (('bid', 'play'), ('bid',)),
                    'trump': (('bid', 'trump', 'play'), ('trump',)), 'joint': (('bid', 'trump', 'play'),) * 2}
        for sid, (neural, learn) in expected.items():
            model, data = load_model(self.path / 'out' / sid / 'checkpoints/best.pt')
            self.assertEqual((model.neural_phases, tuple(data['metadata']['learn_phases'])), (neural, learn))
            self.assertEqual(set(data['metadata']['trained_phases']), set(neural))
            self.assertIsNone(data['metadata']['original_play_sha256'])
        result = json.loads((self.path / 'out/evaluation/result.json').read_text())
        self.assertIn('games', json.dumps(result))
        progress = (self.path / 'out/scratch-selfplay/progress.jsonl').read_text()
        self.assertIn('self-current', progress)

    def test_interrupted_training_resumes_exactly(self):
        def config(root):
            return self.config([
                self.train('stage', **{'from-scratch': True, 'total-hands': 32, 'rollout-decisions': 64,
                                       'checkpoint-every-hands': 8, 'opponent-pool': 'HHH:0.5,self-snapshots:0.5',
                                       'snapshot-every-hands': 8, 'eval-opponents': 'HHH,CHH',
                                       'selection-opponent': 'HHH'}),
                dict(id='evaluation', command='evaluate_selfplay_ppo',
                     args={'checkpoint': '${stage.latest}', 'opponents': 'HHH', 'deals': 2}),
            ], output_root=str(self.path / root))
        reference = self.execute(config('reference'))

        def interrupted(command, cwd):  # Train 16 of 32 hands, then fake extra history and crash.
            command = list(command)
            if '--total-hands' in command:
                command[command.index('--total-hands') + 1] = '16'
                result = subprocess.run(command, cwd=cwd, capture_output=True)
                directory = Path(command[command.index('--output-dir') + 1])
                with (directory / 'progress.jsonl').open('a') as sink:
                    sink.write(json.dumps({'hands': 24, 'stale': True}) + '\n')
                (directory / 'snapshots' / 'policy-000000024.pt').write_bytes(b'stale')
                return SimpleNamespace(returncode=1 if result.returncode == 0 else result.returncode)
            raise AssertionError('Evaluation must not run after a failure')
        crashed = config('crashed')
        code, _ = quiet(rp.run, rp.parse_config(crashed), runner=interrupted)
        self.assertEqual(code, 1)
        manifest = self.execute(crashed, resume=True)
        attempt = manifest['steps'][0]['attempts'][-1]
        self.assertEqual(attempt['mode'], 'resume')
        self.assertIn('--resume', attempt['argv'])
        self.assertEqual(attempt['reconciliation']['checkpoint_hands'], 16)
        self.assertEqual(attempt['reconciliation']['discarded_progress_rows'], 1)
        self.assertEqual(len(attempt['reconciliation']['moved_snapshots']), 1)
        rows = [json.loads(l) for l in (self.path / 'crashed/stage/progress.jsonl').read_text().splitlines()]
        self.assertEqual([r['hands'] for r in rows], [r['hands'] for r in map(json.loads,
                         (self.path / 'reference/stage/progress.jsonl').read_text().splitlines())])
        import torch
        a = torch.load(self.path / 'crashed/stage/checkpoints/latest.pt', weights_only=True)['model']
        b = torch.load(self.path / 'reference/stage/checkpoints/latest.pt', weights_only=True)['model']
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))  # Split run equals uninterrupted run.
        self.assertEqual(manifest['steps'][1]['state'], 'completed')
        self.assertEqual(reference['steps'][1]['state'], 'completed')

    def test_legacy_train_ppo_oracle_dataset_and_version_1_chain(self):
        tiny = {'hidden-size': 8, 'eval-deals': 2, 'bootstrap-samples': 5, 'device': 'cpu', 'minibatch-size': 32}
        steps = [
            dict(id='legacy', command='train_ppo', args={**tiny, 'total-hands': 8, 'rollout-play-steps': 56,
                                                         'checkpoint-every-hands': 4}),
            dict(id='legacy-eval', command='evaluate_ppo', args={'checkpoint': '${legacy.best}', 'deals': 2}),
            dict(id='modern', command='train_selfplay_ppo', args={
                'init-play-checkpoint': '${legacy.best}', 'learner-bid': 'neural-train', 'learner-bid-source': 'new',
                'learner-play': 'neural-frozen', 'learner-play-source': 'anchor', 'total-hands': 4,
                'rollout-decisions': 8, 'eval-deals': 2, 'bootstrap-samples': 5, 'forgetting-hands': 1,
                'device': 'cpu'}),
            dict(id='modern-eval', command='evaluate_selfplay_ppo',
                 args={'checkpoint': '${modern.best}', 'deals': 2}),
            dict(id='labels', command='oracle_labels', args={'deals': 6, 'max-nodes': 5000, 'time-seconds': 2}),
            self.train('critic', **{'from-scratch': True, 'critic-mode': 'centralized', 'oracle-aux-coef': 0.5,
                                    'oracle-dataset': '${labels.output}', 'opponent-pool': 'HHH:1'}),
        ]
        manifest = self.execute(self.config(steps))
        self.assertEqual([s['state'] for s in manifest['steps']], ['completed'] * 6)
        self.assertEqual(manifest['steps'][5]['dependencies'][0]['reference'], 'labels.output')
        progress = (self.path / 'out/critic/progress.jsonl').read_text()
        self.assertIn('oracle_aux_loss', progress)

    def test_interrupted_legacy_training_repeats_its_configuration(self):
        steps = [dict(id='legacy', command='train_ppo', args={
            'hidden-size': 8, 'eval-deals': 2, 'bootstrap-samples': 5, 'device': 'cpu', 'minibatch-size': 32,
            'total-hands': 16, 'rollout-play-steps': 56, 'checkpoint-every-hands': 4})]

        def interrupted(command, cwd):
            command = list(command)
            command[command.index('--total-hands') + 1] = '8'
            subprocess.run(command, cwd=cwd, capture_output=True, check=True)
            return SimpleNamespace(returncode=1)
        config = self.config(steps)
        quiet(rp.run, rp.parse_config(config), runner=interrupted)
        manifest = self.execute(config, resume=True)
        argv = manifest['steps'][0]['attempts'][-1]['argv']
        self.assertIn('--resume', argv)
        self.assertIn('--hidden-size', argv)  # Legacy resume repeats the saved configuration.
        rows = [json.loads(l) for l in (self.path / 'out/legacy/progress.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['hands'], 16)


if __name__ == '__main__':
    unittest.main()
