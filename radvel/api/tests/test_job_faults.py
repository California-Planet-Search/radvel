"""One failed call in the runner must not strand a job.

A locked database or a process out of file descriptors is most likely at the
moment a runaway job is being killed. Each test here injects one such failure
and checks the job still ends, its row still says so, and its run can be
resubmitted.
"""

from __future__ import annotations

import errno
import os
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from radvel.api.tests.test_job_cancel import (
    _kill_quietly, _read_pid, _succeeds, _swallows_sigterm, _wait_error,
    _wait_gone, _wait_state, _wait_terminal,
)
from radvel.api.tests.test_job_processes import _leaves_a_sleeper_behind


# See test_job_recovery.py: every radvel.api import sits inside a function so
# the base `test` matrix, which installs no extras, can still collect this.
pytestmark = pytest.mark.api


class _FailOnce:
    """Call ``target``, but raise ``exc`` the first time ``when()`` holds."""

    def __init__(self, target, when, exc):
        self.target, self.when, self.exc = target, when, exc
        self.raised = 0

    def __call__(self, *args, **kwargs):
        if not self.raised and self.when():
            self.raised += 1
            raise self.exc
        return self.target(*args, **kwargs)


def _on_the_monitor_thread():
    return threading.current_thread().name == "radvel-job-monitor"


def _called_from(name):
    # Frame 0 is when(), frame 1 is _FailOnce.__call__, frame 2 its caller.
    return lambda: sys._getframe(2).f_code.co_name == name


def _exits_without_a_result(run_id, params_json):
    os._exit(3)


def _wait_leads_group(runner, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner._running.get(job_id)
        if job is not None and job.leads_group:
            return
        time.sleep(0.05)
    raise AssertionError("job {} never led its process group".format(job_id))


def test_a_row_the_database_would_not_fail_is_failed_on_a_retry(make_runner):
    """A dead job's row left ``running`` blocks its run until a restart."""
    runner = make_runner(workers=1)
    fault = _FailOnce(runner.registry.mark_finished, _on_the_monitor_thread,
                      sqlite3.OperationalError("database is locked"))
    runner.registry.mark_finished = fault

    job = runner.submit("run-1", "mcmc", {}, worker=_exits_without_a_result)

    row = _wait_terminal(runner, job.job_id, timeout=15)
    assert fault.raised == 1
    assert row.state == "failed"
    assert "exited with code 3" in (row.error or ""), row.error
    assert not runner._running and not runner._futures
    again = runner.submit("run-1", "mcmc", {}, worker=_succeeds)  # 409 if still running
    assert _wait_terminal(runner, again.job_id).state == "succeeded"


def test_a_job_that_cannot_start_is_failed_even_if_the_database_refuses_once(make_runner):
    runner = make_runner(workers=1)

    class _Unstartable:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise OSError(errno.EAGAIN, "fork: Resource temporarily unavailable")

    real_context = runner._context
    runner._context = types.SimpleNamespace(Process=_Unstartable)
    fault = _FailOnce(runner.registry.mark_finished, _on_the_monitor_thread,
                      sqlite3.OperationalError("database is locked"))
    runner.registry.mark_finished = fault

    job = runner.submit("run-1", "mcmc", {}, worker=_succeeds)

    row = _wait_terminal(runner, job.job_id, timeout=15)
    assert fault.raised == 1
    assert row.state == "failed"
    assert "could not start the job process" in (row.error or ""), row.error
    assert not runner._running and not runner._futures
    runner._context = real_context
    again = runner.submit("run-1", "mcmc", {}, worker=_succeeds)
    assert _wait_terminal(runner, again.job_id).state == "succeeded"


def test_group_members_says_when_it_could_not_look(monkeypatch, tmp_path):
    """"Could not look" must not read as "the group is empty"."""
    from radvel.api import jobs

    own_group = os.getpgid(0)
    assert os.getpid() in jobs._group_members(own_group)

    if Path("/proc/self/stat").exists():
        def no_fds(self):
            raise OSError(errno.EMFILE, "Too many open files")

        with monkeypatch.context() as m:
            m.setattr(jobs.Path, "iterdir", no_fds)
            assert jobs._group_members(own_group) is None

    def no_ps(*args, **kwargs):
        raise OSError(errno.EAGAIN, "fork: Resource temporarily unavailable")

    real_path = jobs.Path
    monkeypatch.setattr(jobs, "Path", lambda p: real_path(tmp_path / "no-proc"))
    monkeypatch.setattr(jobs.subprocess, "run", no_ps)
    assert jobs._group_members(own_group) is None


def test_leftovers_are_killed_even_when_the_group_cannot_be_listed(
        make_runner, monkeypatch, tmp_path):
    from radvel.api import jobs

    real = jobs._group_members
    looked = []

    def cannot_look(pgid):
        if sys._getframe(1).f_code.co_name == "_clear_group":
            looked.append(pgid)
            return None
        return real(pgid)

    monkeypatch.setattr(jobs, "_group_members", cannot_look)
    runner = make_runner(workers=1)
    orphanfile = str(tmp_path / "orphan.pid")

    job = runner.submit("run-1", "mcmc", {"orphanfile": orphanfile},
                        worker=_leaves_a_sleeper_behind)

    row = _wait_terminal(runner, job.job_id)
    sleeper = _read_pid(orphanfile)
    try:
        assert row.state == "succeeded", row.error
        assert _wait_gone(sleeper, timeout=10), "the process it left behind survived"
        assert looked
    finally:
        _kill_quietly(sleeper)


def _start_a_stubborn_job(runner, tmp_path):
    pidfile, childfile = str(tmp_path / "worker.pid"), str(tmp_path / "child.pid")
    job = runner.submit("run-1", "mcmc",
                        {"pidfile": pidfile, "childfile": childfile},
                        worker=_swallows_sigterm)
    pids = _read_pid(pidfile), _read_pid(childfile)
    _wait_state(runner, job.job_id, {"running"})
    _wait_leads_group(runner, job.job_id)
    return job, pids


def test_a_cancel_still_kills_after_one_failed_kill(make_runner, monkeypatch, tmp_path):
    """The job must not be marked killed before the signal is sent."""
    from radvel.api import jobs

    runner = make_runner(workers=1)
    job, (worker_pid, child_pid) = _start_a_stubborn_job(runner, tmp_path)
    fault = _FailOnce(jobs._kill_group, _called_from("_kill"),
                      OSError(errno.EMFILE, "Too many open files"))
    monkeypatch.setattr(jobs, "_kill_group", fault)
    real = jobs._group_members
    monkeypatch.setattr(
        jobs, "_group_members",
        lambda pgid: None if sys._getframe(1).f_code.co_name == "_kill" else real(pgid))

    try:
        runner.cancel(job.job_id)
        assert _wait_gone(worker_pid, timeout=15), "the worker survived a failed kill"
        assert _wait_gone(child_pid, timeout=5), "its child survived a failed kill"
    finally:
        _kill_quietly(child_pid)
        _kill_quietly(worker_pid)
    assert fault.raised == 1
    row = _wait_error(runner, job.job_id, "killed")
    assert row.state == "cancelled"
    assert "the rest of its process group" in (row.error or ""), row.error


def test_shutdown_returns_after_one_failed_kill(make_runner, monkeypatch, tmp_path):
    from radvel.api import jobs

    runner = make_runner(workers=1)
    job, (worker_pid, child_pid) = _start_a_stubborn_job(runner, tmp_path)
    fault = _FailOnce(jobs._kill_group, _called_from("_kill"),
                      OSError(errno.EMFILE, "Too many open files"))
    monkeypatch.setattr(jobs, "_kill_group", fault)

    stopper = threading.Thread(target=runner.shutdown, daemon=True)
    stopper.start()
    stopper.join(20)
    try:
        assert not stopper.is_alive(), "shutdown() did not return"
        assert _wait_gone(worker_pid, timeout=5)
        assert _wait_gone(child_pid, timeout=5)
    finally:
        _kill_quietly(child_pid)
        _kill_quietly(worker_pid)
    assert fault.raised == 1
    assert runner.registry.get(job.job_id).state == "failed"
