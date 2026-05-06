"""POST/GET/LIST/DELETE for runs."""

import pytest


pytestmark = pytest.mark.api


def test_create_run_then_get_then_list(client, epic_payload):
    create = client.post("/runs", json=epic_payload)
    assert create.status_code == 201, create.text
    run_id = create.json()["run_id"]
    assert run_id.startswith("r-") and len(run_id) == 12

    get = client.get(f"/runs/{run_id}")
    assert get.status_code == 200
    assert get.json()["starname"] == "epic203771098"
    assert get.json()["nplanets"] == 2

    listed = client.get("/runs")
    assert listed.status_code == 200
    assert any(r["run_id"] == run_id for r in listed.json())


def test_unknown_run_id_returns_404(client):
    assert client.get("/runs/r-not-real").status_code == 404
    # path traversal also 404
    assert client.get("/runs/..%2F..%2Fetc").status_code == 404


def test_invalid_fitting_basis_rejected(client, epic_payload):
    epic_payload["fitting_basis"] = "totally bogus"
    resp = client.post("/runs", json=epic_payload)
    assert resp.status_code == 422
    assert "fitting_basis" in resp.text


def test_delete_is_idempotent(client, epic_payload):
    rid = client.post("/runs", json=epic_payload).json()["run_id"]
    assert client.delete(f"/runs/{rid}").status_code == 204
    assert client.delete(f"/runs/{rid}").status_code == 204
    assert client.get(f"/runs/{rid}").status_code == 404


def test_get_run_reports_active_job(client, epic_payload):
    """active_job should reflect a queued/running mcmc/ns job, not always None.

    Goes directly through JobRegistry rather than through the async
    runner so the test is deterministic — submitting a real worker
    creates a queued→running→terminal race window that's unobservable
    on fast machines and noisy on CI.
    """
    from radvel.api.config import get_settings
    from radvel.api.jobs import JobRegistry

    rid = client.post("/runs", json=epic_payload).json()["run_id"]
    client.post(f"/runs/{rid}/fit", json={})

    # No active job initially.
    assert client.get(f"/runs/{rid}").json()["active_job"] is None

    # Insert a queued mcmc row directly. The wiring under test is the
    # /runs/{id} → JobRegistry.active_job_for_run hand-off.
    registry = JobRegistry(settings=get_settings())
    row = registry.submit(rid, "mcmc", {"nsteps": 1})

    info = client.get(f"/runs/{rid}").json()
    assert info["active_job"] is not None
    assert info["active_job"]["job_id"] == row.job_id
    assert info["active_job"]["kind"] == "mcmc"
    assert info["active_job"]["state"] == "queued"

    # Once it's marked terminal, active_job clears.
    registry.mark_finished(row.job_id, state="cancelled", error=None)
    assert client.get(f"/runs/{rid}").json()["active_job"] is None
