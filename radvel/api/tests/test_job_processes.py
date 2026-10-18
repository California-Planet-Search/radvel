"""Each job runs in a process of its own, and no job outlives the service.

The 2026-10-17 RVsearch job left a child process behind when it finished,
stuck on a dead pool's lock, and nothing ever cleared it. A job's processes
share its process group, so the runner can clear whatever is left when the
job ends, and a job whose API process dies stops itself instead of running
on as a ghost.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from radvel.api.tests.test_job_cancel import (
    _kill_quietly, _orphan_a_sleeper, _read_pid, _wait_gone, _wait_state,
    _wait_terminal, _waits_for_release, _write_pid,
)


# See test_job_recovery.py: every radvel.api import sits inside a function so
# the base `test` matrix, which installs no extras, can still collect this.
pytestmark = pytest.mark.api


def _starts_its_own_pool(run_id, params_json):
    """What RVsearch does: a multiprocessing pool inside the job."""
    import json
    import multiprocessing as mp

    with mp.get_context("fork").Pool(2) as pool:
        values = pool.map(abs, [-1, -2, -3])
    Path(json.loads(params_json)["result"]).write_text(repr(values))
    return {"ok": True}


def _leaves_a_sleeper_behind(run_id, params_json):
    import json

    _orphan_a_sleeper(json.loads(params_json)["orphanfile"])
    return {"ok": True}


def _runs_with_a_child(run_id, params_json):
    import json

    params = json.loads(params_json)
    child = subprocess.Popen(["sleep", "300"])
    _write_pid(params["childfile"], child.pid)
    _write_pid(params["pidfile"], os.getpid())
    time.sleep(120)
    return {"ok": True}


def test_a_job_can_start_its_own_multiprocessing_pool(make_runner, tmp_path):
    """Daemonic job processes could not: they may not have children."""
    runner = make_runner(workers=1)
    result = tmp_path / "result"
    job = runner.submit("run-pool", "mcmc", {"result": str(result)},
                        worker=_starts_its_own_pool)

    row = _wait_terminal(runner, job.job_id, timeout=60)

    assert row.state == "succeeded", row.error
    assert result.read_text() == "[1, 2, 3]"


def test_processes_a_job_leaves_behind_are_killed_when_it_ends(make_runner, tmp_path):
    runner = make_runner(workers=1)
    orphanfile = str(tmp_path / "orphan.pid")
    job = runner.submit("run-leaves", "mcmc", {"orphanfile": orphanfile},
                        worker=_leaves_a_sleeper_behind)
    orphan_pid = _read_pid(orphanfile)
    try:
        assert _wait_terminal(runner, job.job_id).state == "succeeded"
        assert _wait_gone(orphan_pid, timeout=10), "the leftover sleeper survived its job"
    finally:
        _kill_quietly(orphan_pid)


def test_no_more_than_workers_jobs_run_at_once(make_runner, tmp_path):
    runner = make_runner(workers=2)
    jobs, pidfiles, releases = [], [], []
    for i in range(3):
        pidfiles.append(str(tmp_path / "job{}.pid".format(i)))
        releases.append(tmp_path / "release{}".format(i))
        jobs.append(runner.submit(
            "run-{}".format(i), "mcmc",
            {"pidfile": pidfiles[i], "release": str(releases[i])},
            worker=_waits_for_release))
    try:
        _wait_state(runner, jobs[0].job_id, {"running"})
        _wait_state(runner, jobs[1].job_id, {"running"})
        time.sleep(1.0)
        assert runner.registry.get(jobs[2].job_id).state == "queued"

        releases[0].touch()
        assert _wait_terminal(runner, jobs[0].job_id).state == "succeeded"
        _wait_state(runner, jobs[2].job_id, {"running"})

        releases[1].touch()
        releases[2].touch()
        assert _wait_terminal(runner, jobs[1].job_id).state == "succeeded"
        assert _wait_terminal(runner, jobs[2].job_id).state == "succeeded"
    finally:
        for release in releases:
            release.touch()


_FAKE_API = """
import sys, time
from radvel.api.config import get_settings
from radvel.api.jobs import JobRegistry, JobRunner
from radvel.api.tests.test_job_cancel import _succeeds
from radvel.api.tests.test_job_processes import _runs_with_a_child

runner = JobRunner(JobRegistry(settings=get_settings()))
running = runner.submit("run-running", "mcmc",
                        {"pidfile": sys.argv[1], "childfile": sys.argv[2]},
                        worker=_runs_with_a_child)
queued = runner.submit("run-queued", "mcmc", {}, worker=_succeeds)
print(running.job_id, queued.job_id, flush=True)
time.sleep(300)
"""


def test_jobs_stop_when_the_api_process_dies_and_startup_fails_the_rest(
        settings_env, monkeypatch, tmp_path):
    """The restart path: the API process dies with a job running and one queued.

    The running job must not carry on as a ghost; it stops itself and its
    child. The queued one has no process at all, so the next startup fails it.
    """
    monkeypatch.setenv("RADVEL_API_WORKERS", "1")
    pidfile, childfile = str(tmp_path / "job.pid"), str(tmp_path / "child.pid")
    api = subprocess.Popen([sys.executable, "-c", _FAKE_API, pidfile, childfile],
                           stdout=subprocess.PIPE, text=True)
    job_pid = child_pid = None
    try:
        running_id, queued_id = api.stdout.readline().split()
        job_pid, child_pid = _read_pid(pidfile, timeout=60), _read_pid(childfile)

        api.kill()
        api.wait(timeout=10)

        assert _wait_gone(job_pid, timeout=15), "the job outlived its API process"
        assert _wait_gone(child_pid, timeout=5), "the job's child outlived the API process"
    finally:
        if api.poll() is None:
            api.kill()
        for pid in (child_pid, job_pid):
            if pid is not None:
                _kill_quietly(pid)

    from radvel.api.config import get_settings
    from radvel.api.jobs import JobRegistry

    get_settings.cache_clear()
    registry = JobRegistry(settings=get_settings())
    row = registry.get(running_id)
    assert row.state == "failed"
    assert "API process" in (row.error or ""), row.error

    assert registry.reconcile_orphaned() == 1  # the startup half
    row = registry.get(queued_id)
    assert row.state == "failed"
    assert "never started" in (row.error or ""), row.error
