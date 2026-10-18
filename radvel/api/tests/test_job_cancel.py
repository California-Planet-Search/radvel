"""A cancelled job must actually stop, and must not stop the jobs after it.

On 2026-10-17 an RVsearch job was cancelled after 24 h. The row said
``cancelled``, but the worker had swallowed the SIGTERM (it landed inside a
``multiprocessing`` finalizer, which prints "Exception ignored" and carries
on), so the search ran for almost four more hours, wrote all its outputs, and
left one child process stuck on a dead queue lock. ``cancel()`` sent one
signal and never looked again.

Each job runs in its own process, leading its own process group, so a cancel
that has to kill reaches everything the job started and nothing else.
"""

from __future__ import annotations

import os
import shlex
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


def _orphan_a_sleeper(pidfile):
    """Start ``sleep 300`` through a shell that exits at once.

    The sleeper is reparented away from the job, so walking the job's
    children cannot find it; it is still in the job's process group.
    """
    tmp = shlex.quote(pidfile + ".tmp")
    subprocess.run(["sh", "-c", "sleep 300 >/dev/null 2>&1 & echo $! > {} && mv {} {}".format(
        tmp, tmp, shlex.quote(pidfile))], check=True)


def _swallows_sigterm(run_id, params_json):
    """The 2026-10-17 worker: SIGTERM becomes SystemExit, and something eats it.

    It also starts a child process, as RVsearch and multi-ensemble MCMC do,
    and orphans one, as a crashed pool can.
    """
    import json

    from radvel.api.drivers_adapter import _install_sigterm_handler

    params = json.loads(params_json)
    _install_sigterm_handler()
    child = subprocess.Popen(["sleep", "300"])
    _write_pid(params["childfile"], child.pid)
    if "orphanfile" in params:
        _orphan_a_sleeper(params["orphanfile"])
    _write_pid(params["pidfile"], os.getpid())
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            time.sleep(0.05)
        except SystemExit:
            pass
    return {"ok": True}


def _waits_for_release(run_id, params_json):
    """Run until the test creates the release file."""
    import json

    params = json.loads(params_json)
    _write_pid(params["pidfile"], os.getpid())
    deadline = time.monotonic() + 120
    while not os.path.exists(params["release"]):
        if time.monotonic() > deadline:
            raise TimeoutError("never released")
        time.sleep(0.05)
    return {"ok": True}


def _leaves_a_marker(run_id, params_json):
    import json

    Path(json.loads(params_json)["marker"]).write_text("ran")
    return {"ok": True}


def _succeeds(run_id, params_json):
    return {"ok": True}


@pytest.fixture
def runner(make_runner):
    return make_runner(workers=1)


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
    """A cancel that works must not stop the runner from starting later jobs."""
    pidfile = str(tmp_path / "worker.pid")
    job = runner.submit("run-cancelled", "mcmc", {"pidfile": pidfile},
                        worker=_runs_until_cancelled)
    _read_pid(pidfile)
    _wait_state(runner, job.job_id, {"running"})
    future = runner._futures[job.job_id]

    runner.cancel(job.job_id)
    assert runner.registry.get(job.job_id).state == "cancelled"
    future.result(timeout=10)  # the worker stopped

    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)

    assert _wait_terminal(runner, after.job_id).state == "succeeded"
    time.sleep(1.5)  # past the 1 s grace: a worker that stopped is not killed
    assert runner.registry.get(job.job_id).error == "cancelled by client"


def test_a_worker_that_swallows_sigterm_is_killed_with_its_children(runner, tmp_path):
    pidfile, childfile = str(tmp_path / "worker.pid"), str(tmp_path / "child.pid")
    orphanfile = str(tmp_path / "orphan.pid")
    job = runner.submit("run-ghost", "mcmc",
                        {"pidfile": pidfile, "childfile": childfile,
                         "orphanfile": orphanfile},
                        worker=_swallows_sigterm)
    worker_pid, child_pid = _read_pid(pidfile), _read_pid(childfile)
    orphan_pid = _read_pid(orphanfile)
    _wait_state(runner, job.job_id, {"running"})
    try:
        row = runner.cancel(job.job_id)
        assert row.state == "cancelled"

        # Grace is 1 s in this test; the worker ignores SIGTERM, so it must
        # be killed shortly after, with its child and the sleeper it orphaned.
        assert _wait_gone(worker_pid, timeout=15), "worker survived the cancel"
        assert _wait_gone(child_pid, timeout=5), "worker's child survived the cancel"
        assert _wait_gone(orphan_pid, timeout=5), "the orphaned sleeper survived"
    finally:
        for pid in (orphan_pid, child_pid, worker_pid):
            _kill_quietly(pid)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = runner.registry.get(job.job_id)
        if "killed" in (row.error or ""):
            break
        time.sleep(0.05)
    assert row.state == "cancelled"
    assert "killed" in (row.error or ""), row.error
    assert "2 other process" in row.error, row.error

    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)
    assert _wait_terminal(runner, after.job_id).state == "succeeded"


def test_a_job_cancelled_while_it_waits_for_a_worker_never_runs(runner, tmp_path):
    """With one worker slot, the second job waits until the first is done."""
    first_pidfile = str(tmp_path / "first.pid")
    marker = tmp_path / "ran"
    first = runner.submit("run-first", "mcmc", {"pidfile": first_pidfile},
                          worker=_runs_until_cancelled)
    first_future = runner._futures[first.job_id]
    _read_pid(first_pidfile)
    waiting = runner.submit("run-waiting", "mcmc", {"marker": str(marker)},
                            worker=_leaves_a_marker)
    assert runner.registry.get(waiting.job_id).state == "queued"

    runner.cancel(waiting.job_id)
    runner.cancel(first.job_id)
    first_future.result(timeout=15)  # first job's process is gone
    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)
    assert _wait_terminal(runner, after.job_id).state == "succeeded"

    assert not marker.exists(), "a cancelled job ran anyway"
    assert runner.registry.get(waiting.job_id).state == "cancelled"


def test_killing_one_job_leaves_the_job_beside_it_running(make_runner, tmp_path):
    """With a shared worker pool, killing one worker broke the pool.

    Every other job in it failed: on a service running 8 jobs at once, one
    stuck cancel would have taken the other 7 down with it.
    """
    runner = make_runner(workers=2)
    stuck_pidfile, childfile = str(tmp_path / "stuck.pid"), str(tmp_path / "child.pid")
    beside_pidfile, release = str(tmp_path / "beside.pid"), str(tmp_path / "release")
    stuck = runner.submit("run-stuck", "mcmc",
                          {"pidfile": stuck_pidfile, "childfile": childfile},
                          worker=_swallows_sigterm)
    beside = runner.submit("run-beside", "mcmc",
                           {"pidfile": beside_pidfile, "release": release},
                           worker=_waits_for_release)
    stuck_pid, child_pid = _read_pid(stuck_pidfile), _read_pid(childfile)
    beside_pid = _read_pid(beside_pidfile)
    _wait_state(runner, beside.job_id, {"running"})
    try:
        runner.cancel(stuck.job_id)
        assert _wait_gone(stuck_pid, timeout=15), "the stuck job survived the cancel"
        assert _wait_gone(child_pid, timeout=5)
        time.sleep(0.5)  # time for any collateral kill to land

        assert not _gone(beside_pid), "killing one job killed the job beside it"
        assert runner.registry.get(beside.job_id).state == "running"
        Path(release).touch()
        assert _wait_terminal(runner, beside.job_id).state == "succeeded"
    finally:
        for pid in (child_pid, stuck_pid, beside_pid):
            _kill_quietly(pid)


def _wait_error(runner, job_id, text, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = runner.registry.get(job_id)
        if text in (row.error or ""):
            return row
        time.sleep(0.05)
    return row


def test_a_cancel_sent_to_another_api_process_still_stops_the_job(make_runner, tmp_path):
    """``radvel serve --workers 2`` runs two API processes on one jobs database.

    A ``DELETE /jobs/{id}`` can land on either. The one that did not start
    the job has no handle on its process, so the one that did has to notice
    the cancelled row and stop it.
    """
    owner, other = make_runner(workers=1), make_runner(workers=1)
    pidfile, childfile = str(tmp_path / "worker.pid"), str(tmp_path / "child.pid")
    job = owner.submit("run-elsewhere", "mcmc",
                       {"pidfile": pidfile, "childfile": childfile},
                       worker=_swallows_sigterm)
    worker_pid, child_pid = _read_pid(pidfile), _read_pid(childfile)
    _wait_state(owner, job.job_id, {"running"})
    try:
        assert other.cancel(job.job_id).state == "cancelled"

        assert _wait_gone(worker_pid, timeout=15), \
            "the job ran on after a cancel sent to the other API process"
        assert _wait_gone(child_pid, timeout=5), "the job's child survived the cancel"
    finally:
        for pid in (child_pid, worker_pid):
            _kill_quietly(pid)

    row = _wait_error(owner, job.job_id, "killed")
    assert row.state == "cancelled"
    assert "killed" in (row.error or ""), row.error


def test_a_job_whose_row_was_cancelled_before_it_started_never_runs(settings_env, tmp_path):
    """Only the row says the job was cancelled: another API process did it.

    The job process must check the row when it claims it, and not run.
    """
    import json
    import multiprocessing

    from radvel.api.config import get_settings
    from radvel.api.jobs import JobRegistry, _worker_entrypoint

    registry = JobRegistry(settings=get_settings())
    marker = tmp_path / "ran"
    params = {"marker": str(marker)}
    row = registry.submit("run-pre-cancelled", "mcmc", params)
    registry.mark_finished(row.job_id, state="cancelled", error="cancelled by client")

    process = multiprocessing.get_context("spawn").Process(
        target=_worker_entrypoint,
        args=(_leaves_a_marker, row.job_id, row.run_id, json.dumps(params),
              str(registry.db_path), os.getpid()))
    process.start()
    process.join(60)
    try:
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()

    assert not marker.exists(), "a cancelled job ran anyway"
    row = registry.get(row.job_id)
    assert (row.state, row.error) == ("cancelled", "cancelled by client")


def test_a_kill_reaches_every_process_in_the_job_group(make_runner, tmp_path):
    """The kill itself must reach the job's children, not the clean-up after it.

    When the job ends, the runner also kills whatever is left in its group,
    and that hides a kill that only reached the job process.
    """
    from concurrent.futures import Future

    from radvel.api.jobs import _Job

    runner = make_runner(workers=1)
    childfile = str(tmp_path / "child.pid")
    tmp = shlex.quote(childfile + ".tmp")
    leader = subprocess.Popen(
        ["sh", "-c", "sleep 300 & echo $! > {} && mv {} {}; wait".format(
            tmp, tmp, shlex.quote(childfile))],
        start_new_session=True)
    child_pid = None
    try:
        child_pid = _read_pid(childfile)
        job = _Job(job_id="j-kill", args=(), future=Future(), process=leader,
                   leads_group=True)

        assert runner._kill(job) == 1

        assert _wait_gone(child_pid, timeout=5), "the kill missed the job's child"
    finally:
        leader.kill()
        leader.wait()
        if child_pid is not None:
            _kill_quietly(child_pid)
