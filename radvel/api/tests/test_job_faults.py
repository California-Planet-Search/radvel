"""One failed call in the runner must not strand a job.

A locked database or a process out of file descriptors is most likely at the
moment a runaway job is being killed. Each test here injects such a failure
and checks the job still ends, its row still says so, and its run can be
resubmitted.
"""

from __future__ import annotations

import errno
import logging
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

_ALWAYS = 10 ** 9


class _Fault:
    """Call ``target``, but raise ``exc`` the first ``times`` times ``when()`` holds."""

    def __init__(self, target, when, exc, times=1):
        self.target, self.when, self.exc, self.times = target, when, exc, times
        self.raised = 0

    def __call__(self, *args, **kwargs):
        if self.raised < self.times and self.when():
            self.raised += 1
            raise self.exc
        return self.target(*args, **kwargs)


def _on_the_monitor_thread():
    return threading.current_thread().name == "radvel-job-monitor"


def _called_from(name):
    # Frame 0 is when(), frame 1 is _Fault.__call__, frame 2 its caller.
    return lambda: sys._getframe(2).f_code.co_name == name


def _exits_without_a_result(run_id, params_json):
    os._exit(3)


def _wait_until(condition, what, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("{} after {}s".format(what, timeout))


def _wait_leads_group(runner, job_id, timeout=30.0):
    def leads():
        job = runner._running.get(job_id)
        return job is not None and job.leads_group
    _wait_until(leads, "job {} never led its process group".format(job_id), timeout)


def test_a_row_the_database_would_not_fail_is_failed_on_a_retry(make_runner):
    """A dead job's row left ``running`` blocks its run until a restart."""
    runner = make_runner(workers=1)
    fault = _Fault(runner.registry.mark_finished, _on_the_monitor_thread,
                   sqlite3.OperationalError("database is locked"))
    runner.registry.mark_finished = fault

    job = runner.submit("run-1", "mcmc", {}, worker=_exits_without_a_result)

    row = _wait_terminal(runner, job.job_id)
    assert fault.raised == 1
    assert row.state == "failed"
    assert "exited with code 3" in (row.error or ""), row.error
    assert not runner._running and not runner._futures
    again = runner.submit("run-1", "mcmc", {}, worker=_succeeds)  # 409 if still running
    assert _wait_terminal(runner, again.job_id).state == "succeeded"


def test_a_row_is_retried_until_the_database_takes_it(make_runner):
    """The first retry is immediate; later ones need the monitor to keep waking."""
    runner = make_runner(workers=1)
    fault = _Fault(runner.registry.mark_finished, _on_the_monitor_thread,
                   sqlite3.OperationalError("database is locked"), times=3)
    runner.registry.mark_finished = fault

    job = runner.submit("run-1", "mcmc", {}, worker=_exits_without_a_result)

    row = _wait_terminal(runner, job.job_id)
    assert fault.raised == 3
    assert row.state == "failed"
    assert not runner._repairs


def test_shutdown_fails_a_row_the_monitor_could_not(make_runner):
    """With the monitor stopped, nothing else would retry it before a restart."""
    runner = make_runner(workers=1)
    fault = _Fault(runner.registry.mark_finished, _on_the_monitor_thread,
                   sqlite3.OperationalError("database is locked"), times=_ALWAYS)
    runner.registry.mark_finished = fault

    job = runner.submit("run-1", "mcmc", {}, worker=_exits_without_a_result)
    _wait_until(lambda: job.job_id in runner._repairs and not runner._running,
                "the dead job's row was never queued for a retry")
    assert runner.registry.get(job.job_id).state == "running"

    runner.shutdown()

    row = runner.registry.get(job.job_id)
    assert row.state == "failed"
    assert "exited with code 3" in (row.error or ""), row.error
    assert not runner._repairs


def test_a_row_that_keeps_failing_is_warned_about(make_runner, caplog):
    """After the first traceback a retry fails quietly, so say so now and then."""
    from radvel.api import jobs

    runner = make_runner(workers=1)
    runner.shutdown()  # the monitor thread owns _repairs while it runs
    runner.registry.mark_finished = _Fault(
        runner.registry.mark_finished, lambda: True,
        sqlite3.OperationalError("database is locked"), times=_ALWAYS)
    runner._repairs["j-stuck"] = "exited with code 3"

    with caplog.at_level(logging.WARNING, logger="radvel.api.jobs"):
        for _ in range(2 * jobs._REPAIR_WARN_EVERY):
            runner._retry_repairs(now=True)

    warned = [r.getMessage() for r in caplog.records
              if "still not failed" in r.getMessage()]
    assert len(warned) == 2, warned
    assert "j-stuck" in warned[-1]


def test_a_job_that_cannot_start_is_failed_even_if_the_database_refuses_once(make_runner):
    runner = make_runner(workers=1)

    class _Unstartable:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise OSError(errno.EAGAIN, "fork: Resource temporarily unavailable")

    real_context = runner._context
    runner._context = types.SimpleNamespace(Process=_Unstartable)
    fault = _Fault(runner.registry.mark_finished, _on_the_monitor_thread,
                   sqlite3.OperationalError("database is locked"))
    runner.registry.mark_finished = fault

    job = runner.submit("run-1", "mcmc", {}, worker=_succeeds)

    row = _wait_terminal(runner, job.job_id)
    assert fault.raised == 1
    assert row.state == "failed"
    assert "could not start the job process" in (row.error or ""), row.error
    assert not runner._running and not runner._futures
    runner._context = real_context
    again = runner.submit("run-1", "mcmc", {}, worker=_succeeds)
    assert _wait_terminal(runner, again.job_id).state == "succeeded"


def _fake_proc(monkeypatch, root, entries):
    """Point ``_group_members`` at a /proc of our own: name -> stat bytes."""
    from radvel.api import jobs

    (root / "self").mkdir(parents=True)
    (root / "self" / "stat").write_bytes(b"1 (self) S 0 1 1")
    for name, stat in entries.items():
        (root / name).mkdir()
        if stat is not None:
            (root / name / "stat").write_bytes(stat)
    real_path = jobs.Path
    monkeypatch.setattr(jobs, "Path", lambda p: real_path(root))


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

    real_path, real_run = jobs.Path, jobs.subprocess.run
    with monkeypatch.context() as m:
        m.setattr(jobs, "Path", lambda p: real_path(tmp_path / "no-proc"))

        def no_ps(*args, **kwargs):
            raise OSError(errno.EAGAIN, "fork: Resource temporarily unavailable")

        m.setattr(jobs.subprocess, "run", no_ps)
        assert jobs._group_members(own_group) is None

        def ps_fails(args, **kwargs):  # prints nothing, exits 1
            return real_run([sys.executable, "-c", "raise SystemExit(1)"], **kwargs)

        m.setattr(jobs.subprocess, "run", ps_fails)
        assert jobs._group_members(own_group) is None


def test_group_members_reads_any_command_name(monkeypatch, tmp_path):
    """The kernel cuts a name at 15 bytes, which can split a UTF-8 character."""
    from radvel.api import jobs

    _fake_proc(monkeypatch, tmp_path / "proc", {
        "100": b"100 (a\xe6\xb5\x8b\xe8\xaf) S 1 4242 4242 0",
        "101": b"101 (odd) name)) R 1 4242 4242 0",
        "102": b"102 (zombie) Z 1 4242 4242 0",
        "103": b"103 (elsewhere) S 1 7 7 0",
        "104": None,  # exited while we looked
    })
    assert sorted(jobs._group_members(4242)) == [100, 101]


@pytest.mark.parametrize("stat", [b"105 (cut short", b"105 (x) S 1 notanumber 0"],
                         ids=["no-close-paren", "bad-pgrp"])
def test_group_members_cannot_look_past_a_stat_line_it_cannot_read(
        monkeypatch, tmp_path, stat):
    from radvel.api import jobs

    _fake_proc(monkeypatch, tmp_path / "proc", {
        "100": b"100 (job) S 1 4242 4242 0", "105": stat})
    assert jobs._group_members(4242) is None


def test_group_members_cannot_look_past_a_stat_file_it_cannot_open(monkeypatch, tmp_path):
    """Only a process that went away may be skipped; EMFILE is not that."""
    from radvel.api import jobs

    _fake_proc(monkeypatch, tmp_path / "proc", {"100": b"100 (job) S 1 4242 4242 0"})
    real_read = Path.read_bytes

    def emfile_for_105(self):
        if self.parent.name == "105":
            raise OSError(errno.EMFILE, "Too many open files")
        return real_read(self)

    (tmp_path / "proc" / "105").mkdir()
    monkeypatch.setattr(Path, "read_bytes", emfile_for_105)
    assert jobs._group_members(4242) is None


@pytest.mark.parametrize("fails", ["returns-none", "raises"])
def test_leftovers_are_killed_even_when_the_group_cannot_be_listed(
        make_runner, monkeypatch, tmp_path, fails):
    from radvel.api import jobs

    real = jobs._group_members
    looked = []

    def cannot_look(pgid):
        if sys._getframe(1).f_code.co_name == "_clear_group":
            looked.append(pgid)
            if fails == "raises":
                raise ValueError("unreadable /proc entry")
            return None
        return real(pgid)

    monkeypatch.setattr(jobs, "_group_members", cannot_look)
    # A long grace, so only the clear-up after the job can kill the sleeper.
    runner = make_runner(workers=1, cancel_grace_s=60)
    orphanfile = str(tmp_path / "orphan.pid")

    job = runner.submit("run-1", "mcmc", {"orphanfile": orphanfile},
                        worker=_leaves_a_sleeper_behind)

    row = _wait_terminal(runner, job.job_id)
    sleeper = _read_pid(orphanfile)
    try:
        assert row.state == "succeeded", row.error
        _wait_until(lambda: job.job_id not in runner._running,
                    "the job never left the runner")
        assert looked
        assert _wait_gone(sleeper, timeout=10), "the process it left behind survived"
    finally:
        _kill_quietly(sleeper)


def test_a_dead_job_is_failed_even_if_clearing_its_group_fails(make_runner, monkeypatch):
    from radvel.api import jobs

    fault = _Fault(jobs._kill_group, _called_from("_clear_group"),
                   OSError(errno.EIO, "injected"), times=_ALWAYS)
    monkeypatch.setattr(jobs, "_kill_group", fault)
    real = jobs._group_members
    monkeypatch.setattr(
        jobs, "_group_members",
        lambda pgid: None if sys._getframe(1).f_code.co_name == "_clear_group" else real(pgid))
    runner = make_runner(workers=1)

    job = runner.submit("run-1", "mcmc", {}, worker=_exits_without_a_result)

    row = _wait_terminal(runner, job.job_id)
    assert fault.raised >= 1
    assert row.state == "failed"
    assert "exited with code 3" in (row.error or ""), row.error
    _wait_until(lambda: not runner._running and not runner._futures,
                "the dead job kept its slot")
    again = runner.submit("run-1", "mcmc", {}, worker=_succeeds)  # 409 if still running
    assert _wait_terminal(runner, again.job_id).state == "succeeded"


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
    fault = _Fault(jobs._kill_group, _called_from("_kill"),
                   OSError(errno.EMFILE, "Too many open files"))
    monkeypatch.setattr(jobs, "_kill_group", fault)
    real = jobs._group_members
    monkeypatch.setattr(
        jobs, "_group_members",
        lambda pgid: None if sys._getframe(1).f_code.co_name == "_kill" else real(pgid))

    try:
        runner.cancel(job.job_id)
        assert _wait_gone(worker_pid, timeout=30), "the worker survived a failed kill"
        assert _wait_gone(child_pid, timeout=5), "its child survived a failed kill"
    finally:
        _kill_quietly(child_pid)
        _kill_quietly(worker_pid)
    assert fault.raised == 1
    row = _wait_error(runner, job.job_id, "killed")
    assert row.state == "cancelled"
    assert "the rest of its process group" in (row.error or ""), row.error


def test_a_cancel_kills_even_if_listing_the_group_always_fails(
        make_runner, monkeypatch, tmp_path):
    """An unreadable /proc entry of some other process must not stop the kill."""
    from radvel.api import jobs

    runner = make_runner(workers=1)
    job, (worker_pid, child_pid) = _start_a_stubborn_job(runner, tmp_path)
    fault = _Fault(jobs._group_members, _called_from("_kill"),
                   ValueError("unreadable /proc entry"), times=_ALWAYS)
    monkeypatch.setattr(jobs, "_group_members", fault)

    try:
        runner.cancel(job.job_id)
        assert _wait_gone(worker_pid, timeout=30), "the worker survived"
        assert _wait_gone(child_pid, timeout=5), "its child survived"
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
    fault = _Fault(jobs._kill_group, _called_from("_kill"),
                   OSError(errno.EMFILE, "Too many open files"))
    monkeypatch.setattr(jobs, "_kill_group", fault)

    stopper = threading.Thread(target=runner.shutdown, daemon=True)
    stopper.start()
    stopper.join(30)
    try:
        assert not stopper.is_alive(), "shutdown() did not return"
        assert _wait_gone(worker_pid, timeout=5)
        assert _wait_gone(child_pid, timeout=5)
    finally:
        _kill_quietly(child_pid)
        _kill_quietly(worker_pid)
    assert fault.raised == 1
    assert runner.registry.get(job.job_id).state == "failed"
