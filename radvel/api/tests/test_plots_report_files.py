"""Tests for plots, report, and files endpoints (M4)."""

import os
import shutil
import time

import pytest


pytestmark = pytest.mark.api


def _create(client, payload):
    resp = client.post("/runs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["run_id"]


def _wait_for_job(client, job_id, *, timeout=240, poll=1.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.get(f"/jobs/{job_id}").json()
        if last["state"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(poll)
    raise AssertionError("job did not finish in {}s; last: {}".format(timeout, last))


def _fit_then_mcmc(client, payload):
    rid = _create(client, payload)
    fit = client.post(f"/runs/{rid}/fit", json={})
    assert fit.status_code == 200, fit.text
    job = client.post(
        f"/runs/{rid}/mcmc",
        json={"nsteps": 100, "nwalkers": 20, "ensembles": 2,
              "thin": 1, "serial": True},
    ).json()
    final = _wait_for_job(client, job["job_id"], timeout=240)
    assert final["state"] in {"succeeded", "failed"}
    return rid


def test_plots_rv_after_fit(client, epic_payload):
    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={})
    resp = client.post(f"/runs/{rid}/plots", json={"types": ["rv"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert any(f["type"] == "rv" for f in body["files"])
    rv_url = next(f["url"] for f in body["files"] if f["type"] == "rv")
    dl = client.get(rv_url)
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("application/pdf")
    assert len(dl.content) > 1000


def test_corner_plot_requires_mcmc(client, epic_payload):
    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={})
    resp = client.post(f"/runs/{rid}/plots", json={"types": ["corner"]})
    # Driver asserts MCMC has run; surfaced as 409 step_prerequisite_unmet.
    assert resp.status_code == 409, resp.text


def test_files_listing_and_download(client, epic_payload):
    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={})
    resp = client.get(f"/runs/{rid}/files")
    assert resp.status_code == 200
    listing = resp.json()
    names = {f["name"] for f in listing}
    # The fit step writes a status file and a posterior pickle.
    assert any(n.endswith("_radvel.stat") for n in names)
    assert any(n.endswith("_post_obj.pkl") for n in names)


def test_files_path_traversal_blocked(client, epic_payload):
    rid = _create(client, epic_payload)
    resp = client.get(f"/runs/{rid}/files/../../etc/passwd")
    # FastAPI normalises ../ in the path before our handler — the resulting
    # URL is /etc/passwd which doesn't match the route → 404.
    assert resp.status_code == 404


def test_files_unknown_run_returns_404(client):
    resp = client.get("/runs/r-doesnotexist/files")
    assert resp.status_code == 404


@pytest.mark.skipif(shutil.which("pdflatex") is None,
                    reason="pdflatex not on PATH")
def test_report_after_full_pipeline(client, epic_payload):
    rid = _fit_then_mcmc(client, epic_payload)
    # corner plot is required for a meaningful report
    client.post(f"/runs/{rid}/plots", json={"types": ["rv", "corner"]})
    resp = client.post(f"/runs/{rid}/report", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["file"].endswith("_results.pdf")
    dl = client.get(body["url"])
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("application/pdf")
    assert len(dl.content) > 1000
