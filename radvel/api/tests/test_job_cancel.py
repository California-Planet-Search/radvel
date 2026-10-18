"""A cancelled job must actually stop, and must not stop the jobs after it.

On 2026-10-17 an RVsearch job was cancelled after 24 h. The row said
``cancelled``, but the worker had swallowed the SIGTERM (it landed inside a
``multiprocessing`` finalizer, which prints "Exception ignored" and carries
on), so the search ran for almost four more hours, wrote all its outputs, and
left one child process stuck on a dead queue lock. ``cancel()`` sent one
signal and never looked again.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest


# See test_job_recovery.py: every radvel.api import sits inside a function so
# the base `test` matrix, which installs no extras, can still collect this.
pytestmark = pytest.mark.api


def _write_pid(path, pid):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(pid))
    os.replace(tmp, path)


def _runs_until_cancelled(run_id, params_json):
    """A well-behaved worker: honours SIGTERM like the MCMC worker does."""
    import json

    from radvel.api.drivers_adapter import _install_sigterm_handler

    _install_sigterm_handler()
    _write_pid(json.loads(params_json)["pidfile"], os.getpid())
    time.sleep(120)
    return {"ok": True}


def _swallows_sigterm(run_id, params_json):
    """The 2026-10-17 worker: SIGTERM becomes SystemExit, and something eats it.

    It also starts a child process, as RVsearch and multi-ensemble MCMC do.
    """
    import json

    from radvel.api.drivers_adapter import _install_sigterm_handler

    params = json.loads(params_json)
    _install_sigterm_handler()
    child = subprocess.Popen(["sleep", "300"])
    _write_pid(params["childfile"], child.pid)
    _write_pid(params["pidfile"], os.getpid())
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            time.sleep(0.05)
        except SystemExit:
            pass
    return {"ok": True}


def _leaves_a_marker(run_id, params_json):
    import json

    Path(json.loads(params_json)["marker"]).write_text("ran")
    return {"ok": True}


def _succeeds(run_id, params_json):
    return {"ok": True}


@pytest.fixture
def runner(settings_env, monkeypatch):
    monkeypatch.setenv("RADVEL_API_WORKERS", "1")
    monkeypatch.setenv("RADVEL_API_CANCEL_GRACE_S", "1")
    from radvel.api.config import get_settings
    from radvel.api.jobs import JobRegistry, JobRunner

    get_settings.cache_clear()
    job_runner = JobRunner(JobRegistry(settings=get_settings()))
    yield job_runner
    job_runner.shutdown(wait=False)


def _wait_state(runner, job_id, states, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = runner.registry.get(job_id)
        if row.state in states:
            return row
        time.sleep(0.05)
    raise AssertionError(
        "job {} still {!r} after {}s".format(
            job_id, runner.registry.get(job_id).state, timeout)
    )


def _wait_terminal(runner, job_id, timeout=30.0):
    return _wait_state(runner, job_id, {"succeeded", "failed", "cancelled"}, timeout)


def _read_pid(path, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return int(Path(path).read_text())
        time.sleep(0.05)
    raise AssertionError("{} never appeared".format(path))


def _gone(pid):
    """True once ``pid`` has exited. A zombie has exited; it only awaits a reap."""
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _wait_gone(pid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _gone(pid):
            return True
        time.sleep(0.05)
    return False


def _kill_quietly(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def test_a_job_submitted_after_a_cancel_still_runs(runner, tmp_path):
    """A cancel that works must not take the pool's dispatcher down with it.

    The cancelled worker's ``SystemExit`` comes back as the future's
    exception, and the done-callback runs on the executor's manager thread.
    """
    pidfile = str(tmp_path / "worker.pid")
    job = runner.submit("run-cancelled", "mcmc", {"pidfile": pidfile},
                        worker=_runs_until_cancelled)
    _read_pid(pidfile)
    _wait_state(runner, job.job_id, {"running"})

    runner.cancel(job.job_id)
    assert runner.registry.get(job.job_id).state == "cancelled"
    runner._futures[job.job_id].exception(timeout=10)  # the worker stopped

    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)

    assert _wait_terminal(runner, after.job_id).state == "succeeded"
    time.sleep(1.5)  # past the 1 s grace: a worker that stopped is not killed
    assert runner.registry.get(job.job_id).error == "cancelled by client"


def test_a_worker_that_swallows_sigterm_is_killed_with_its_children(runner, tmp_path):
    pidfile, childfile = str(tmp_path / "worker.pid"), str(tmp_path / "child.pid")
    job = runner.submit("run-ghost", "mcmc",
                        {"pidfile": pidfile, "childfile": childfile},
                        worker=_swallows_sigterm)
    worker_pid, child_pid = _read_pid(pidfile), _read_pid(childfile)
    _wait_state(runner, job.job_id, {"running"})
    try:
        row = runner.cancel(job.job_id)
        assert row.state == "cancelled"

        # Grace is 1 s in this test; the worker ignores SIGTERM, so it must
        # be killed shortly after, and its child with it.
        assert _wait_gone(worker_pid, timeout=15), "worker survived the cancel"
        assert _wait_gone(child_pid, timeout=5), "worker's child survived the cancel"
    finally:
        _kill_quietly(child_pid)
        _kill_quietly(worker_pid)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = runner.registry.get(job.job_id)
        if "killed" in (row.error or ""):
            break
        time.sleep(0.05)
    assert row.state == "cancelled"
    assert "killed" in (row.error or ""), row.error
    assert "1 child" in row.error, row.error

    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)
    assert _wait_terminal(runner, after.job_id).state == "succeeded"


def test_a_job_cancelled_while_it_waits_for_a_worker_never_runs(runner, tmp_path):
    """With one worker, the second job waits inside the executor's call queue.

    ``Future.cancel()`` cannot stop it there (the executor already marked it
    running), so only the row can tell the worker not to start it.
    """
    pidfile = str(tmp_path / "worker.pid")
    marker = tmp_path / "ran"
    first = runner.submit("run-first", "mcmc", {"pidfile": pidfile},
                          worker=_runs_until_cancelled)
    _read_pid(pidfile)
    waiting = runner.submit("run-waiting", "mcmc", {"marker": str(marker)},
                            worker=_leaves_a_marker)
    assert runner.registry.get(waiting.job_id).state == "queued"
    deadline = time.monotonic() + 10
    while not runner._futures[waiting.job_id].running():
        assert time.monotonic() < deadline, "job never reached the call queue"
        time.sleep(0.02)

    runner.cancel(waiting.job_id)
    runner.cancel(first.job_id)
    runner._futures[waiting.job_id].exception(timeout=15)  # dispatched and done

    assert not marker.exists(), "a cancelled job ran anyway"
    assert runner.registry.get(waiting.job_id).state == "cancelled"
