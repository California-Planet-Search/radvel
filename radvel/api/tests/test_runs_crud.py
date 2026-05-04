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
