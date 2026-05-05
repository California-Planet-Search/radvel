"""Tests for the optional /ui static frontend (M4.5)."""

import pytest


pytestmark = pytest.mark.api


def test_ui_index_served_when_enabled(client):
    """Default settings have RADVEL_API_ENABLE_UI=true → /ui returns the SPA shell."""
    resp = client.get("/ui")
    assert resp.status_code == 200, resp.text
    assert "<title>RadVel</title>" in resp.text
    assert "/ui/app.js" in resp.text
    assert resp.headers.get("content-security-policy", "").startswith("default-src 'self'")


def test_ui_assets_served(client):
    js = client.get("/ui/app.js")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith(("text/javascript", "application/javascript"))
    assert "viewRunsList" in js.text  # smoke check the bundle is real

    css = client.get("/ui/styles.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")

    fav = client.get("/ui/favicon.svg")
    assert fav.status_code == 200
    assert fav.headers["content-type"].startswith("image/svg")


def test_ui_path_traversal_blocked(client):
    # FastAPI normalises ``..`` before routing; the resulting path doesn't
    # match our /ui handler so we get 404 either way. The router's
    # ``relative_to(base)`` check is the second line of defense.
    resp = client.get("/ui/../../etc/passwd")
    assert resp.status_code == 404


def test_ui_unknown_asset_404(client):
    resp = client.get("/ui/does-not-exist.js")
    assert resp.status_code == 404


def test_ui_disabled_returns_404(tmp_path, monkeypatch):
    """Setting RADVEL_API_ENABLE_UI=false unwires both /ui routes."""
    monkeypatch.setenv("RADVEL_API_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RADVEL_API_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("RADVEL_API_ENABLE_UI", "false")
    from radvel.api.config import get_settings
    get_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from radvel.api.main import create_app
        with TestClient(create_app()) as test_client:
            assert test_client.get("/ui").status_code == 404
            assert test_client.get("/ui/app.js").status_code == 404
            # Other endpoints still work
            assert test_client.get("/healthz").status_code == 200
    finally:
        get_settings.cache_clear()
