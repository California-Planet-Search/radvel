"""The SIGTERM handler must turn SIGTERM into SystemExit only where it was installed.

A forked child inherits Python signal handlers. RVsearch and multi-ensemble
MCMC fork ``multiprocessing`` workers from inside an API worker, and
``Pool.terminate()`` stops them with SIGTERM, expecting them to die on the
spot. With the inherited handler they raised SystemExit instead and unwound
through whatever they were doing, queue locks included.
"""

from __future__ import annotations

import os
import signal
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
    time.sleep(seconds)


def test_the_installing_process_gets_systemexit(handler_installed):
    with pytest.raises(SystemExit) as excinfo:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)  # the handler raises at the next bytecode boundary
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_a_forked_child_dies_of_sigterm(handler_installed):
    import multiprocessing as mp

    child = mp.get_context("fork").Process(target=_sleep, args=(60,))
    child.start()
    try:
        os.kill(child.pid, signal.SIGTERM)
        child.join(timeout=10)
        # -15 is death by the signal; 143 would be the inherited handler's
        # SystemExit(128 + 15) unwinding the child instead.
        assert child.exitcode == -signal.SIGTERM
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


def _mark_then_sleep(args):
    path, seconds = args
    open(path, "w").close()
    time.sleep(seconds)


def test_pool_terminate_kills_busy_workers(handler_installed, tmp_path):
    """The RVsearch case: an API worker forks a multiprocessing.Pool.

    Both pool workers are mid-task, so neither can leave through the pool's
    own shutdown sentinel; only SIGTERM can stop them.
    """
    import multiprocessing as mp

    pool = mp.get_context("fork").Pool(2)
    workers = list(pool._pool)
    marks = [str(tmp_path / "task{}".format(i)) for i in range(2)]
    try:
        pool.map_async(_mark_then_sleep, [(m, 60) for m in marks], chunksize=1)
        deadline = time.monotonic() + 10
        while not all(os.path.exists(m) for m in marks):
            assert time.monotonic() < deadline, "pool tasks never started"
            time.sleep(0.02)
        pool.terminate()
        pool.join()
        assert [w.exitcode for w in workers] == [-signal.SIGTERM] * 2
    finally:
        for w in workers:
            if w.is_alive():
                w.kill()
