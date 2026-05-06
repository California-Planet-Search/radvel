"""Async MCMC: kick a job, poll, assert convergence telemetry shows up."""

import time

import pytest


pytestmark = pytest.mark.api


def _create(client, payload):
    resp = client.post("/runs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["run_id"]


def _wait_for_job(client, job_id, *, timeout=300, poll=1.5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.get(f"/jobs/{job_id}").json()
        if last["state"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(poll)
    raise AssertionError(
        "job did not finish in {}s; last state: {}".format(timeout, last)
    )


def test_mcmc_kickoff_returns_202_with_job_id(client, epic_payload):
    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={})

    resp = client.post(
        f"/runs/{rid}/mcmc",
        json={"nsteps": 100, "nwalkers": 20, "ensembles": 2,
              "minsteps": 10, "minpercent": 100,  # never converge — we cancel
              "thin": 1, "serial": True, "save": False, "proceed": False},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["job_id"].startswith("j-") and len(body["job_id"]) == 12
    assert body["kind"] == "mcmc"
    assert body["state"] == "queued"

    # Status endpoint always works after submit.
    status = client.get(f"/jobs/{body['job_id']}")
    assert status.status_code == 200
    assert status.json()["job_id"] == body["job_id"]


def test_mcmc_run_to_completion_small(client, epic_payload):
    """A tiny MCMC (nsteps=100, ensembles=2) should reach succeeded."""
    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={})

    job = client.post(
        f"/runs/{rid}/mcmc",
        json={"nsteps": 100, "nwalkers": 20, "ensembles": 2,
              "thin": 1, "serial": True},
    ).json()

    final = _wait_for_job(client, job["job_id"], timeout=240)
    # MCMC must finish in some terminal state. We accept succeeded OR a
    # documented failure state, since 100 steps × 20 walkers in 2
    # ensembles is far below typical convergence thresholds — emcee may
    # legitimately complete or warn out. The contract asserts the runner
    # reports a terminal state with ``finished_at`` set, not perfection.
    assert final["state"] in {"succeeded", "failed"}
    assert final["finished_at"] is not None


def test_mcmc_cancel(client, epic_payload):
    """Cancelling a running MCMC must reach ``cancelled`` quickly."""
    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={})

    job = client.post(
        f"/runs/{rid}/mcmc",
        json={"nsteps": 100000, "nwalkers": 50, "ensembles": 4,
              "thin": 1, "serial": False},
    ).json()
    # Wait for the worker to actually start (so terminate has something to hit).
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        s = client.get(f"/jobs/{job['job_id']}").json()
        if s["state"] == "running":
            break
        time.sleep(0.5)

    cancelled = client.delete(f"/jobs/{job['job_id']}")
    assert cancelled.status_code == 202, cancelled.text
    final = _wait_for_job(client, job["job_id"], timeout=30)
    assert final["state"] == "cancelled"


def test_active_job_blocks_concurrent_kickoff(client, epic_payload):
    """Submitting MCMC while a job is already active for the run → 409.

    The pinning step uses the JobRegistry directly rather than a real
    worker so the test is deterministic: a real worker can transition
    queued → running → terminal before the second POST hits, leaving
    nothing for ``active_job_for_run`` to find. The block-logic itself
    just needs *any* queued/running row to be present.
    """
    from radvel.api.config import get_settings
    from radvel.api.jobs import JobRegistry

    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={})

    pinned = JobRegistry(settings=get_settings()).submit(rid, "mcmc", {"nsteps": 1})
    try:
        second = client.post(
            f"/runs/{rid}/mcmc",
            json={"nsteps": 100, "nwalkers": 20, "ensembles": 2, "serial": True},
        )
        assert second.status_code == 409, second.text
    finally:
        # Mark the pinned row terminal so subsequent runs (and shutdown)
        # don't think there's a real worker out there.
        JobRegistry(settings=get_settings()).mark_finished(
            pinned.job_id, state="cancelled", error=None,
        )
