"""Extension packages (e.g. rvsearch) can add job kinds to the service."""

import json
import time

import pytest


# Needs the [api] extra; runs in the `api-test` CI job. radvel.api imports
# stay inside functions so the base `test` matrix (no extras) can still
# collect this module -- see test_job_recovery.py.
pytestmark = pytest.mark.api


def _echo_worker(run_id: str, params_json: str) -> dict:
    """Module-level so the ProcessPoolExecutor child can import it."""
    from radvel.api import extensions

    record = extensions.worker_setup(run_id)
    params = json.loads(params_json)
    extensions.ProgressWriter(extensions.progress_path(record, "echo")).write(
        {"stage": "done", "pcomplete": 100.0, "echo": params["value"]})
    return {"ok": True}


def _router():
    # No `from __future__ import annotations` in this module: FastAPI must see
    # the real Request class, which is only imported here.
    from fastapi import APIRouter, Request

    router = APIRouter()

    @router.post("/runs/{run_id}/echo", status_code=202)
    def start_echo(run_id: str, request: Request):
        row = request.app.state.job_runner.submit(
            run_id, "echo", {"value": 42}, worker=_echo_worker)
        return {"job_id": row.job_id}

    return router


@pytest.fixture
def echo_kind():
    from radvel.api import extensions
    from radvel.api.jobs import JOB_KINDS

    extensions.register_job_kind("echo")
    yield
    JOB_KINDS.discard("echo")


def test_unregistered_kind_is_rejected(settings_env):
    from radvel.api.jobs import JobRegistry

    with pytest.raises(ValueError, match="unknown job kind"):
        JobRegistry().submit("run-x", "echo", {})


def test_register_job_kind_validates_name():
    from radvel.api import extensions

    with pytest.raises(ValueError):
        extensions.register_job_kind("bad kind; drop")


def test_progress_filename_keeps_mcmc_name_for_builtins():
    from radvel.api import extensions

    assert extensions.progress_filename("mcmc") == "mcmc_progress.json"
    assert extensions.progress_filename("ns") == "mcmc_progress.json"
    assert extensions.progress_filename("search") == "search_progress.json"


def test_extension_job_runs_and_reports_its_progress(settings_env, epic_payload, echo_kind):
    from fastapi.testclient import TestClient
    from radvel.api.main import create_app

    app = create_app()
    app.include_router(_router())
    with TestClient(app) as client:
        run_id = client.post("/runs", json=epic_payload).json()["run_id"]
        kick = client.post("/runs/{}/echo".format(run_id))
        assert kick.status_code == 202, kick.text
        job_id = kick.json()["job_id"]

        job = {"state": "unpolled"}
        deadline = time.time() + 60
        while time.time() < deadline:
            job = client.get("/jobs/{}".format(job_id)).json()
            if job["state"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.2)

    assert job["state"] == "succeeded", job.get("error")
    assert job["kind"] == "echo"
    assert job["progress"]["echo"] == 42
    assert job["progress"]["stage"] == "done"
