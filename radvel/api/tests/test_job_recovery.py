"""Startup recovery for jobs a previous process left mid-flight.

Two states strand when the API process dies, and only one of them used to
be repaired. The `queued` case is the one that caused a two-week silent
outage in production: three MCMC jobs sat at `queued` from 2026-09-10,
survived a full container recreate untouched, and every client blocked for
its entire read timeout and reported what looked like a slow fit.
"""

from __future__ import annotations

import os

from radvel.api.config import get_settings
from radvel.api.jobs import JobRegistry


def _registry(settings_env) -> JobRegistry:
    return JobRegistry(settings=get_settings())


def test_a_queued_job_from_a_dead_process_is_failed_not_left_queued(settings_env):
    """The regression: a row inserted but never dispatched.

    `JobRunner.submit` inserts the row and *then* hands the work to the
    ProcessPoolExecutor. If the process dies in between -- or the pool dies
    -- the row keeps `state='queued'` with no pid. It is not `running`, so
    the old reconcile skipped it, and the executor only knows jobs submitted
    during its own process's life, so nothing would ever pick it up again.
    """
    reg = _registry(settings_env)
    row = reg.submit("run-stranded", "mcmc", {"nsteps": 10})
    assert reg.get(row.job_id).state == "queued"

    repaired = reg.reconcile_orphaned()

    assert repaired >= 1
    recovered = reg.get(row.job_id)
    assert recovered.state == "failed", (
        "a queued job whose process died must reach a terminal state; "
        "leaving it queued is invisible to every client"
    )
    assert "resubmit" in (recovered.error or "").lower()


def test_the_failure_says_why_so_a_client_can_act_on_it(settings_env):
    """A terminal state is only useful if it explains itself."""
    reg = _registry(settings_env)
    row = reg.submit("run-explains", "mcmc", {"nsteps": 10})
    reg.reconcile_orphaned()
    error = (reg.get(row.job_id).error or "").lower()
    assert "queued" in error and "never started" in error


def test_a_running_job_whose_pid_is_gone_is_still_failed(settings_env):
    """The pre-existing behaviour must survive the change."""
    reg = _registry(settings_env)
    row = reg.submit("run-was-running", "mcmc", {"nsteps": 10})
    # A pid that cannot be alive, on this host, is the "worker died" shape.
    reg.mark_running(row.job_id, pid=2 ** 22, host=os.uname().nodename)
    assert reg.get(row.job_id).state == "running"

    reg.reconcile_orphaned()

    recovered = reg.get(row.job_id)
    assert recovered.state == "failed"
    assert "process gone" in (recovered.error or "")


def test_reconcile_leaves_terminal_rows_alone(settings_env):
    """Idempotent: a second startup must not rewrite finished history."""
    reg = _registry(settings_env)
    row = reg.submit("run-finished", "mcmc", {"nsteps": 10})
    reg.mark_finished(row.job_id, state="succeeded")

    assert reg.reconcile_orphaned() == 0
    assert reg.get(row.job_id).state == "succeeded"

    # And running it twice over a stranded row does not double-count.
    stranded = reg.submit("run-twice", "mcmc", {"nsteps": 10})
    assert reg.reconcile_orphaned() == 1
    assert reg.reconcile_orphaned() == 0
    assert reg.get(stranded.job_id).state == "failed"
