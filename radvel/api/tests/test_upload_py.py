"""Tests for the gated POST /runs/upload-py endpoint."""

import os
from pathlib import Path

import pytest

import radvel


pytestmark = pytest.mark.api


def _example_setup_bytes() -> bytes:
    src = Path(radvel.__file__).parent.parent / "example_planets" / "epic203771098.py"
    return src.read_bytes()


def test_upload_py_disabled_by_default(client):
    """RADVEL_API_ALLOW_PY_UPLOAD defaults to false → endpoint returns 415."""
    resp = client.post(
        "/runs/upload-py",
        files={"setup_file": ("epic203771098.py", _example_setup_bytes(), "text/x-python")},
    )
    assert resp.status_code == 415, resp.text


def test_upload_py_creates_run_when_enabled(tmp_path, monkeypatch):
    """With the flag on, uploading the canonical example creates a run."""
    monkeypatch.setenv("RADVEL_API_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RADVEL_API_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("RADVEL_API_ALLOW_PY_UPLOAD", "true")
    from radvel.api.config import get_settings
    get_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from radvel.api.main import create_app
        with TestClient(create_app()) as test_client:
            # Health echoes the flag back so the UI can decide what to render.
            assert test_client.get("/healthz").json()["allow_py_upload"] is True

            resp = test_client.post(
                "/runs/upload-py",
                files={"setup_file": (
                    "epic203771098.py",
                    _example_setup_bytes(),
                    "text/x-python",
                )},
            )
            assert resp.status_code == 201, resp.text
            body = resp.json()
            assert body["run_id"].startswith("run-") and len(body["run_id"]) == 14

            # Run is fully usable: GET /runs/{id} works, fit kicks off normally.
            r = test_client.get(f"/runs/{body['run_id']}")
            assert r.status_code == 200
            r_body = r.json()
            assert r_body["starname"] == "epic203771098"
            assert r_body["nplanets"] == 2

            fit = test_client.post(f"/runs/{body['run_id']}/fit", json={})
            assert fit.status_code == 200, fit.text
    finally:
        get_settings.cache_clear()


def test_upload_py_rejects_non_py_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("RADVEL_API_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RADVEL_API_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("RADVEL_API_ALLOW_PY_UPLOAD", "true")
    from radvel.api.config import get_settings
    get_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from radvel.api.main import create_app
        with TestClient(create_app()) as test_client:
            resp = test_client.post(
                "/runs/upload-py",
                files={"setup_file": ("setup.json", b"{}", "application/json")},
            )
            assert resp.status_code == 400
    finally:
        get_settings.cache_clear()


def test_upload_py_surfaces_import_failure(tmp_path, monkeypatch):
    """A .py that can't import cleans up the run dir and returns 400."""
    monkeypatch.setenv("RADVEL_API_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RADVEL_API_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("RADVEL_API_ALLOW_PY_UPLOAD", "true")
    from radvel.api.config import get_settings
    get_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from radvel.api.main import create_app
        with TestClient(create_app()) as test_client:
            resp = test_client.post(
                "/runs/upload-py",
                files={"setup_file": ("broken.py", b"this is not python\n!@#$",
                                       "text/x-python")},
            )
            assert resp.status_code == 400
            # No half-written run dir left behind.
            runs = list((tmp_path / "runs").iterdir()) if (tmp_path / "runs").is_dir() else []
            assert runs == []
    finally:
        get_settings.cache_clear()
