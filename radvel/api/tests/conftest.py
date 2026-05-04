"""Shared fixtures for the API integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import radvel


@pytest.fixture
def settings_env(tmp_path: Path, monkeypatch):
    """Point the API settings at a fresh tmp directory and clear the cache."""
    runs_dir = tmp_path / "runs"
    db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("RADVEL_API_RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("RADVEL_API_DB_PATH", str(db_path))
    monkeypatch.setenv("RADVEL_API_ALLOW_PY_UPLOAD", "false")

    from radvel.api.config import get_settings
    get_settings.cache_clear()
    yield {"runs_dir": runs_dir, "db_path": db_path}
    get_settings.cache_clear()


@pytest.fixture
def client(settings_env):
    from fastapi.testclient import TestClient

    from radvel.api.main import create_app
    return TestClient(create_app())


@pytest.fixture
def epic_payload() -> dict:
    """The canonical epic203771098 setup as a JSON payload."""
    raw = pd.read_csv(os.path.join(radvel.DATADIR, "epic203771098.csv"))
    rows = [
        {"time": float(t), "mnvel": float(v), "errvel": float(e), "tel": "j"}
        for t, v, e in zip(raw.t, raw.vel, raw.errvel)
    ]
    return {
        "starname": "epic203771098",
        "nplanets": 2,
        "instnames": ["j"],
        "fitting_basis": "per tc secosw sesinw k",
        "bjd0": 2454833.0,
        "planet_letters": {"1": "b", "2": "c"},
        "params": {
            "per1": {"value": 20.885258, "vary": False},
            "tc1": {"value": 2072.79438, "vary": False},
            "secosw1": {"value": 0.0, "vary": False},
            "sesinw1": {"value": 0.1, "vary": False},
            "k1": {"value": 10.0},
            "per2": {"value": 42.363011, "vary": False},
            "tc2": {"value": 2082.62516, "vary": False},
            "secosw2": {"value": 0.0, "vary": False},
            "sesinw2": {"value": 0.1, "vary": False},
            "k2": {"value": 10.0},
            "dvdt": {"value": 0.0},
            "curv": {"value": 0.0},
            "gamma_j": {"value": 1.0, "vary": False, "linear": True},
            "jit_j": {"value": 2.6},
        },
        "data": {"kind": "inline", "rows": rows},
        "priors": [
            {"type": "eccentricity", "num_planets": 2},
            {"type": "positivek", "num_planets": 2},
            {"type": "jeffreys", "param": "k1", "minval": 1e-2, "maxval": 1e3},
            {"type": "jeffreys", "param": "k2", "minval": 1e-2, "maxval": 1e3},
            {"type": "hardbounds", "param": "jit_j", "minval": 0.0, "maxval": 15.0},
            {"type": "gaussian", "param": "dvdt", "mu": 0.0, "sigma": 1.0},
            {"type": "gaussian", "param": "curv", "mu": 0.0, "sigma": 1e-1},
        ],
        "stellar": {"mstar": 1.12, "mstar_err": 0.05},
    }
