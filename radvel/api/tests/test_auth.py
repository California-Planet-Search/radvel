"""Tests for RADVEL_API_AUTH_KEY authentication middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

_KEY = "test-secret-key"


@pytest.fixture
def authed_client(settings_env, monkeypatch):
    """TestClient with RADVEL_API_AUTH_KEY configured."""
    monkeypatch.setenv("RADVEL_API_AUTH_KEY", _KEY)
    from radvel.api.config import get_settings
    get_settings.cache_clear()
    from radvel.api.main import create_app
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_no_key_configured_allows_all(client):
    """Without RADVEL_API_AUTH_KEY set, requests pass through freely."""
    resp = client.get("/healthz")
    assert resp.status_code == 200

    resp = client.get("/runs")
    assert resp.status_code == 200


def test_correct_key_allows_request(authed_client):
    resp = authed_client.get("/runs", headers={"X-API-Key": _KEY})
    assert resp.status_code == 200


def test_missing_key_returns_401(authed_client):
    resp = authed_client.get("/runs")
    assert resp.status_code == 401
    assert "detail" in resp.json()


def test_wrong_key_returns_401(authed_client):
    resp = authed_client.get("/runs", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_healthz_exempt_from_auth(authed_client):
    """Health check endpoints must work without a key for monitoring."""
    resp = authed_client.get("/healthz")
    assert resp.status_code == 200

    resp = authed_client.get("/version")
    assert resp.status_code == 200
