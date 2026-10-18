"""The SIGTERM handler must turn SIGTERM into SystemExit only where it was installed.

A forked child inherits Python signal handlers. RVsearch and multi-ensemble
MCMC fork ``multiprocessing`` workers from inside an API worker, and
``Pool.terminate()`` stops them with SIGTERM, expecting them to die. With
the inherited handler they raised SystemExit instead, and where CPython
drops the exception (an at-fork hook, a finalizer) the child carried on.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time

import pytest


# See test_job_recovery.py: every radvel.api import sits inside a function so
# the base `test` matrix, which installs no extras, can still collect this.
pytestmark = pytest.mark.api


@pytest.fixture
def handler_installed():
    from radvel.api.drivers_adapter import _install_sigterm_handler

    previous = signal.getsignal(signal.SIGTERM)
    _install_sigterm_handler()
    yield
    signal.signal(signal.SIGTERM, previous)


def _sleep(seconds):
    """Sleep in short steps, so a SIGTERM cannot slip past it.

    A signal that lands just before a long sleep() blocks is recorded but
    does not interrupt it, so one sleep(60) can ignore a SIGTERM for a
    minute. Waking every 50 ms lets the interpreter run the handler.
    """
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(0.05)


def _ready_then_sleep(ready, seconds):
    ready.set()
    _sleep(seconds)


def test_the_installing_process_gets_systemexit(handler_installed):
    with pytest.raises(SystemExit) as excinfo:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)  # the handler raises at the next bytecode boundary
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_a_forked_child_dies_of_sigterm(handler_installed):
    ctx = mp.get_context("fork")
    ready = ctx.Event()
    child = ctx.Process(target=_ready_then_sleep, args=(ready, 60))
    child.start()
    try:
        # CPython discards a signal that lands before the child has finished
        # forking, so wait until the child is running before sending it.
        assert ready.wait(10), "child never started"
        os.kill(child.pid, signal.SIGTERM)
        child.join(timeout=10)
        # -15 is death by the signal; 143 would be the inherited handler's
        # SystemExit(128 + 15) unwinding the child instead.
        assert child.exitcode == -signal.SIGTERM
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


def test_sigterm_inside_the_fork_window_still_kills_the_child(handler_installed):
    """Exact timing: an at-fork hook sends the SIGTERM while the child is forking.

    The inherited handler used to raise SystemExit inside the hook, where
    CPython prints and drops it, and the child carried on.
    """
    armed = [False]

    def sigterm_self_when_armed():
        if armed[0]:
            os.kill(os.getpid(), signal.SIGTERM)

    # A hook cannot be unregistered, so it only acts while armed.
    os.register_at_fork(after_in_child=sigterm_self_when_armed)
    child = mp.get_context("fork").Process(target=_sleep, args=(60,))
    armed[0] = True
    try:
        child.start()
    finally:
        armed[0] = False
    try:
        child.join(timeout=10)
        assert child.exitcode == -signal.SIGTERM
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


def _install_then_fork_and_report(conn):
    """Stand-in for an API job worker: install the handler, fork a child."""
    from radvel.api.drivers_adapter import _install_sigterm_handler

    _install_sigterm_handler()
    ctx = mp.get_context("fork")
    ready = ctx.Event()
    grandchild = ctx.Process(target=_ready_then_sleep, args=(ready, 60))
    grandchild.start()
    try:
        assert ready.wait(10), "grandchild never started"
        grandchild.terminate()
        grandchild.join(timeout=10)
        conn.send(grandchild.exitcode)
    finally:
        if grandchild.is_alive():
            grandchild.kill()
            grandchild.join(timeout=5)
    _sleep(60)  # wait here for the SIGTERM from the test


def test_handler_installed_in_a_forked_worker():
    """The production shape: the API process imports, a forked worker installs.

    The installing process is then neither the one that imported the module
    nor the pytest process. The worker must still get SystemExit (exit 143),
    and the child it forks must die of SIGTERM (-15).
    """
    # Import here, in the parent, so the module is loaded before the fork as
    # it is in the API process. A pid captured at import time would then be
    # the parent's, not the worker's.
    import radvel.api.drivers_adapter  # noqa: F401

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=_install_then_fork_and_report, args=(child_conn,))
    worker.start()
    child_conn.close()
    try:
        assert parent_conn.poll(30), "worker never reported"
        assert parent_conn.recv() == -signal.SIGTERM, "grandchild did not die of SIGTERM"
        os.kill(worker.pid, signal.SIGTERM)
        worker.join(timeout=10)
        assert worker.exitcode == 128 + signal.SIGTERM
    finally:
        parent_conn.close()
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)


def _mark_then_sleep(args):
    path, seconds = args
    open(path, "w").close()
    _sleep(seconds)


def _stop_pool(pool):
    pool.terminate()
    pool.join()


def test_pool_terminate_kills_busy_workers(handler_installed, tmp_path):
    """The RVsearch case: an API worker forks a multiprocessing.Pool.

    Both pool workers are mid-task, so neither can leave through the pool's
    own shutdown sentinel; only SIGTERM can stop them.
    """
    pool = mp.get_context("fork").Pool(2)
    workers = list(pool._pool)
    marks = [str(tmp_path / "task{}".format(i)) for i in range(2)]
    stopper = None
    try:
        pool.map_async(_mark_then_sleep, [(m, 60) for m in marks], chunksize=1)
        deadline = time.monotonic() + 10
        while not all(os.path.exists(m) for m in marks):
            assert time.monotonic() < deadline, "pool tasks never started"
            time.sleep(0.02)
        # A worker that swallows SIGTERM would hang terminate()/join() for
        # good, so give them a deadline and fail instead.
        stopper = threading.Thread(target=_stop_pool, args=(pool,), daemon=True)
        stopper.start()
        stopper.join(timeout=30)
        assert not stopper.is_alive(), "Pool.terminate() hung on a worker"
        assert [w.exitcode for w in workers] == [-signal.SIGTERM] * 2
    finally:
        for w in workers:
            if w.is_alive():
                w.kill()
        if stopper is not None:
            stopper.join(timeout=5)
