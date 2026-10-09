"""Portable process orchestration and unchanged duplicate-deal outcomes."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from texas42.parallel import resolve_workers, available_cpus, OrderedPool, AUTO_WORKER_CAP
from texas42.agents import variant_factory
from texas42.arena import compare_agents
from texas42.experiments.counterfactual_bidding import compare_bidding


def square(value): return value * value, os.getpid()
def fail(value): raise ValueError(f'bad task {value}')


class WorkerTests(unittest.TestCase):
    def test_resolution_and_missing_cpu_information(self):
        with patch('texas42.parallel.available_cpus', return_value=12):
            self.assertEqual(resolve_workers(0), AUTO_WORKER_CAP)
            self.assertEqual(resolve_workers(1), 1)
            self.assertEqual(resolve_workers(20), 12)
            self.assertEqual(resolve_workers(4, 2), 2)
            self.assertEqual(resolve_workers(0, 0), 1)
        for value in (-1, True, 1.5, '2', None):
            with self.assertRaises(ValueError): resolve_workers(value)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError): resolve_workers(0, value)
        with patch('os.cpu_count', return_value=None), patch('os.sched_getaffinity', side_effect=OSError, create=True), patch('pathlib.Path.read_text', side_effect=OSError):
            self.assertEqual(available_cpus(), 1)
        self.assertGreaterEqual(resolve_workers(0), 1)

    def test_quota_and_no_ml_imports(self):
        def read(path, *args, **kwargs):
            if str(path) == '/sys/fs/cgroup/cpu.max':
                return '250000 100000'
            raise OSError('unavailable')
        with patch('os.cpu_count', return_value=8), patch('os.sched_getaffinity', return_value=set(range(8)), create=True), patch('pathlib.Path.read_text', read):
            self.assertEqual(available_cpus(), 2)
            self.assertEqual(resolve_workers(0), 2)
        subprocess.run([sys.executable, '-c',
                        'import texas42.parallel, texas42.arena, sys; '
                        'assert not {"torch", "numpy", "pyspiel"} & sys.modules.keys()'], check=True)

    def test_spawn_order_failure_and_shutdown(self):
        with OrderedPool(2) as pool:
            results = pool.map(square, range(12))
            self.assertEqual([n for n, _ in results], [n*n for n in range(12)])
            self.assertTrue(all(pid != os.getpid() for _, pid in results))
            self.assertEqual(pool.map(square, [4])[0][0], 16)
        self.assertIsNone(pool.executor)
        with OrderedPool(2) as pool:
            with self.assertRaisesRegex(RuntimeError, 'task 0 failed.*bad task'):
                pool.map(fail, [7, 8, 9])
        self.assertIsNone(pool.executor)

    def test_unpicklable_factory_explanation(self):
        with patch('texas42.parallel.available_cpus', return_value=2):
            with self.assertRaisesRegex(ValueError, 'top-level'):
                compare_agents(lambda s, r: None, variant_factory('HHH'), num_deals=2, threads=2)


class ParallelArenaTests(unittest.TestCase):
    def test_equal_reports_all_worker_counts_and_policies(self):
        for policies in (('HHH', 'CHH'), ('RRR', 'CRR')):
            factories = [variant_factory(p) for p in policies]
            reference = compare_agents(*factories, num_deals=8, bootstrap_samples=10)
            for workers in (2, 4, 0):
                actual = compare_agents(*factories, num_deals=8, bootstrap_samples=10, threads=workers)
                self.assertEqual(reference.to_dict(), actual.to_dict())
                for a, b in zip(actual.games[::2], actual.games[1::2]):
                    self.assertEqual(a.group_id, b.group_id)
                    self.assertEqual(a.schedule_ids, b.schedule_ids)

    def test_match_and_counterfactual(self):
        kwargs = dict(num_deals=4, episode='match', target_marks=3, bootstrap_samples=10)
        policies = [variant_factory(p) for p in ('HHH', 'RRR')]
        self.assertEqual(compare_agents(*policies, **kwargs), compare_agents(*policies, threads=2, **kwargs))
        self.assertEqual(compare_bidding('RRR', 'CHH', target_policy='RRR', num_deals=4, bootstrap_samples=10).to_dict(),
                         compare_bidding('RRR', 'CHH', target_policy='RRR', num_deals=4, bootstrap_samples=10, threads=2).to_dict())
