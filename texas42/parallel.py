"""Bounded spawn workers with ordered or completion-driven simulation results."""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from time import monotonic

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
    """Reusable spawn pool with bounded ordered or completion-driven collection.

    map keeps <=2*workers tasks outstanding and returns ordered results.
    iter_completed keeps <=workers outstanding and supports a shared deadline.

    Main-process orchestration only. Close explicitly or use a context manager.
    Failure cancels queued work and waits for bounded running tasks. map never
    returns a partial list; iter_completed may have yielded results before an
    error propagates. Temporary spawn environment changes assume one
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

    def iter_completed(self, function, tasks, *, deadline=None, skip=None):
        """Yield (index, value, status) as bounded tasks settle.

        Status is completed, cancelled, or not_started; the latter two have no
        value. At most workers futures are in flight. An absolute monotonic
        deadline stops submission and cancels queued work. Running functions
        must cooperate with that deadline; they are collected before shutdown.
        Optional skip(index) returns a settlement status for unnecessary work.
        Completed futures are always collected; queued futures are cancelled
        when possible. Running functions must implement cooperative stopping.
        """
        iterator = iter(enumerate(tasks))
        pending = {}
        expired = False
        submitted_all = False
        if self.executor is not None:
            require_picklable(function)
        try:
            while True:
                expired = expired or (deadline is not None and monotonic() >= deadline)
                if expired:
                    for future, index in list(pending.items()):
                        if future.cancel():
                            del pending[future]
                            yield index, None, 'cancelled'
                    for index, _ in iterator:
                        yield index, None, 'not_started'
                else:
                    if skip is not None:
                        for future, index in list(pending.items()):
                            reason = None if future.done() else skip(index)
                            if reason is not None and future.cancel():
                                del pending[future]
                                yield index, None, reason
                    while not submitted_all and len(pending) < self.workers:
                        if deadline is not None and monotonic() >= deadline:
                            break
                        try:
                            index, task = next(iterator)
                        except StopIteration:
                            submitted_all = True
                            break
                        reason = skip(index) if skip is not None else None
                        if reason is not None:
                            yield index, None, reason
                            continue
                        if self.executor is None:
                            yield index, function(task), 'completed'
                            break
                        require_picklable(task)
                        with _worker_environment():
                            pending[self.executor.submit(function, task)] = index
                if pending:
                    timeout = None if expired or deadline is None else max(0, deadline - monotonic())
                    done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=pending.get):
                        index = pending.pop(future)
                        try:
                            value = future.result()
                        except Exception as exc:
                            raise RuntimeError(f'Simulation task {index} failed: {exc}') from exc
                        yield index, value, 'completed'
                elif expired or submitted_all:
                    break
        except BaseException:
            for future in pending:
                future.cancel()
            self.close()
            raise

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def execution_summary(requested, tasks=None):
    return f'Requested threads: {requested}\nResolved simulation workers: {resolve_workers(requested, tasks)}'
