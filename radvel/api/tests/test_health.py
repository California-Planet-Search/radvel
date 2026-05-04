"""/healthz and /version smoke tests."""

import pytest


pytestmark = pytest.mark.api


def test_healthz_returns_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["version"]
    assert isinstance(body["kepler_c"], bool)


def test_version(client):
    import radvel

    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["radvel"] == radvel.__version__
    assert body["api"] == radvel.__version__
    assert body["python"]
    assert body["platform"]
