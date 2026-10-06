"""The worker pool must outlive the jobs it runs.

On 2026-09-30 one MCMC job failed in production with a NaN in the driver.
Every MCMC submitted after it -- for six days, for every user -- answered
``500 Internal Server Error`` with ``BrokenProcessPool: A child process
terminated abruptly, the process pool is not usable anymore``. No process had
terminated. The worker re-raised an ``AdapterError``; that exception could not
be rebuilt in the parent, and ``concurrent.futures`` answers an unreadable
result by declaring the whole pool broken, permanently.

A ``ProcessPoolExecutor`` never recovers from that state, so a job runner that
keeps one for the life of the process has to replace it.
"""

from __future__ import annotations

import os
import pickle
import time

import pytest


# See test_job_recovery.py: every radvel.api import sits inside a function so
# the base `test` matrix, which installs no extras, can still collect this.
pytestmark = pytest.mark.api


class _NeedsTwoArguments(Exception):
    """An exception pickle cannot rebuild: args holds one value, __init__ wants two."""

    def __init__(self, first, second):
        super().__init__(first)
        self.second = second


def _succeeds(run_id, params_json):
    return {"ok": True}


def _raises_adapter_error(run_id, params_json):
    from radvel.api.drivers_adapter import AdapterError

    raise AdapterError(
        error_type="ValueError",
        message="cannot convert float NaN to integer",
        status_code=500,
        traceback_id="abc123def456",
    )


def _raises_unpicklable(run_id, params_json):
    raise _NeedsTwoArguments("first", "second")


def _dies_after_a_moment(run_id, params_json):
    time.sleep(1.0)
    os._exit(1)


@pytest.fixture
def runner(settings_env, monkeypatch):
    monkeypatch.setenv("RADVEL_API_WORKERS", "1")
    from radvel.api.config import get_settings
    from radvel.api.jobs import JobRegistry, JobRunner

    get_settings.cache_clear()
    job_runner = JobRunner(JobRegistry(settings=get_settings()))
    yield job_runner
    job_runner.shutdown(wait=False)


def _wait_terminal(runner, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = runner.registry.get(job_id)
        if row.state in {"succeeded", "failed", "cancelled"}:
            return row
        time.sleep(0.05)
    raise AssertionError(
        "job {} still {!r} after {}s".format(job_id, runner.registry.get(job_id).state, timeout)
    )


def test_adapter_error_survives_pickling():
    """The exception every failed pipeline step raises crosses the process boundary."""
    from radvel.api.drivers_adapter import AdapterError

    original = AdapterError(
        error_type="ValueError",
        message="cannot convert float NaN to integer",
        status_code=500,
        traceback_id="abc123def456",
    )

    rebuilt = pickle.loads(pickle.dumps(original))

    assert rebuilt == original


def test_a_job_that_fails_with_adapter_error_leaves_the_pool_usable(runner):
    """The production regression: one failed MCMC must not end all later ones."""
    pool = runner.pool
    failed = runner.submit("run-fails", "mcmc", {}, worker=_raises_adapter_error)
    assert _wait_terminal(runner, failed.job_id).state == "failed"

    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)

    assert _wait_terminal(runner, after.job_id).state == "succeeded"
    # Replacing a broken pool would also pass the line above, and would kill
    # every other fit running in it. A failed job must not break it at all.
    assert runner.pool is pool


def test_a_job_that_fails_with_an_unpicklable_exception_leaves_the_pool_usable(runner):
    """Drivers raise whatever they raise; the pool cannot depend on its shape."""
    pool = runner.pool
    failed = runner.submit("run-fails", "mcmc", {}, worker=_raises_unpicklable)
    assert _wait_terminal(runner, failed.job_id).state == "failed"

    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)

    assert _wait_terminal(runner, after.job_id).state == "succeeded"
    # Replacing a broken pool would also pass the line above, and would kill
    # every other fit running in it. A failed job must not break it at all.
    assert runner.pool is pool


def test_a_job_submitted_after_a_worker_process_dies_still_runs(runner):
    """A killed worker (OOM, a stray signal) really does break the pool."""
    died = runner.submit("run-dies", "mcmc", {}, worker=_dies_after_a_moment)
    assert _wait_terminal(runner, died.job_id).state == "failed"

    after = runner.submit("run-after", "mcmc", {}, worker=_succeeds)

    assert _wait_terminal(runner, after.job_id).state == "succeeded"


def test_a_job_waiting_in_the_pool_when_it_breaks_is_failed_not_left_queued(runner):
    """With one worker the second job is still waiting when the first dies."""
    died = runner.submit("run-dies", "mcmc", {}, worker=_dies_after_a_moment)
    waiting = runner.submit("run-waiting", "mcmc", {}, worker=_succeeds)
    _wait_terminal(runner, died.job_id)

    row = _wait_terminal(runner, waiting.job_id)

    assert row.state == "failed"
    assert "resubmit" in (row.error or "").lower()
