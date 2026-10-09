"""Bounded, ordered spawn workers for independent simulation tasks (no ML imports)."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
import pickle

AUTO_WORKER_CAP = 8
_THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def available_cpus():
    """Best-effort affinity, Linux physical-core and cgroup quota intersection."""
    limits = []
    try:
        affinity = set(os.sched_getaffinity(0))
        limits.append(len(affinity))
    except (AttributeError, OSError):
        affinity = None
    try:
        count = os.cpu_count()
    except OSError:
        count = None
    if count:
        limits.append(count)
    if affinity:
        try:
            cores = {(Path(f'/sys/devices/system/cpu/cpu{i}/topology/physical_package_id').read_text().strip(),
                      Path(f'/sys/devices/system/cpu/cpu{i}/topology/core_id').read_text().strip())
                     for i in affinity}
            limits.append(len(cores))
        except (OSError, ValueError):
            pass
    # Inspect the process cgroup and its ancestors as quotas can be inherited.
    paths = {Path('/sys/fs/cgroup')}
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            if not controllers:
                current = Path('/sys/fs/cgroup') / relative.lstrip('/')
                while current.is_relative_to('/sys/fs/cgroup'):
                    paths.add(current)
                    current = current.parent
    except (OSError, ValueError):
        pass
    for path in paths:
        try:
            quota, period = (path / 'cpu.max').read_text().split()
            if quota != 'max':
                limits.append(max(1, int(quota) // int(period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    try:
        root = Path('/sys/fs/cgroup/cpu')
        quota = int((root / 'cpu.cfs_quota_us').read_text())
        period = int((root / 'cpu.cfs_period_us').read_text())
        if quota > 0 and period > 0:
            limits.append(max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, min(limits)) if limits else 1


def resolve_workers(requested: int, tasks: int | None = None) -> int:
    if type(requested) is not int or requested < 0:
        raise ValueError('threads must be a nonnegative integer (0=auto, 1=sequential)')
    if tasks is not None and (type(tasks) is not int or tasks < 0):
        raise ValueError('tasks must be a nonnegative integer or None')
    if requested == 1:
        return 1
    cap = available_cpus()
    workers = min(cap, AUTO_WORKER_CAP if requested == 0 else requested)
    return max(1, min(workers, tasks) if tasks is not None else workers)


def require_picklable(value):
    try:
        pickle.dumps(value)
    except Exception as exc:
        raise ValueError('Parallel execution requires picklable top-level factories/functions; '
                         'use variant_factory or a top-level callable class, or threads=1') from exc


@contextmanager
def _worker_environment():
    old = {key: os.environ.get(key) for key in _THREAD_ENV}
    os.environ.update({key: '1' for key in _THREAD_ENV})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _initialize(initializer, initargs):
    os.environ.update({key: '1' for key in _THREAD_ENV})
    if initializer:
        initializer(*initargs)


class OrderedPool:
    """Reusable process pool with <=2*workers outstanding tasks and ordered results.

    Main-process orchestration only. Close explicitly or use a context manager.
    Failure cancels queued work and waits for bounded running tasks; partial
    results are never returned. Temporary spawn environment changes assume one
    orchestration thread. No nested pools are allowed.
    """
    def __init__(self, workers, *, initializer=None, initargs=()):
        if type(workers) is not int or workers < 1:
            raise ValueError('Resolved workers must be a positive integer')
        self.workers = workers
        self.executor = None
        if workers > 1:
            if mp.current_process().name != 'MainProcess':
                raise RuntimeError('Nested simulation pools are unsupported')
            require_picklable((initializer, initargs))
            self.executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                initializer=_initialize, initargs=(initializer, initargs))
        elif initializer:
            initializer(*initargs)

    def map(self, function, tasks):
        if self.executor is None:
            return [function(task) for task in tasks]
        require_picklable(function)
        iterator, pending, results = iter(enumerate(tasks)), deque(), []
        def submit():
            try:
                index, task = next(iterator)
            except StopIteration:
                return False
            require_picklable(task)
            with _worker_environment():
                pending.append((index, self.executor.submit(function, task)))
            return True
        try:
            for _ in range(2 * self.workers):
                if not submit(): break
            while pending:
                index, future = pending.popleft()
                try:
                    results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f'Simulation task {index} failed: {exc}') from exc
                submit()
            return results
        except BaseException:
            for _, future in pending:
                future.cancel()
            self.close()
            raise

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def execution_summary(requested, tasks=None):
    return f'Requested threads: {requested}\nResolved simulation workers: {resolve_workers(requested, tasks)}'
