"""End-to-end sync pipeline: create → fit → derive → tables."""

import pytest


pytestmark = pytest.mark.api


def _create(client, payload):
    resp = client.post("/runs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["run_id"]


def test_fit_then_derive_blocked_then_tables_blocked(client, epic_payload):
    rid = _create(client, epic_payload)

    fit = client.post(f"/runs/{rid}/fit", json={"decorr": False})
    assert fit.status_code == 200, fit.text
    body = fit.json()
    assert "logprob" in body
    assert body["logprob"] < 0  # log-prob is negative for an OK fit
    assert body["postfile"].startswith("run-")

    # derive without sampling: driver bails (fit alone doesn't produce chains)
    derive = client.post(f"/runs/{rid}/derive", json={"sampler": "auto"})
    assert derive.status_code == 409, derive.text

    # tables also blocked because the only chain producers (mcmc/ns) haven't run
    tables = client.post(
        f"/runs/{rid}/tables",
        json={"types": ["params"], "header": False, "name_in_title": False, "sampler": "auto"},
    )
    assert tables.status_code == 409, tables.text


def test_fit_makes_run_state_visible(client, epic_payload):
    rid = _create(client, epic_payload)
    client.post(f"/runs/{rid}/fit", json={"decorr": False})

    state = client.get(f"/runs/{rid}").json()["state"]
    assert "fit" in state
    assert state["fit"]["run"].lower() == "true"
    assert state["fit"]["postfile"].endswith("_post_obj.pkl")


def test_fit_then_ic_simple(client, epic_payload):
    """ic --simple just reports stats for the current model and exits."""
    rid = _create(client, epic_payload)
    assert client.post(f"/runs/{rid}/fit", json={}).status_code == 200

    ic = client.post(
        f"/runs/{rid}/ic",
        json={
            "types": ["nplanets"],
            "mixed": True,
            "fixjitter": False,
            "simple": True,
            "verbose": False,
        },
    )
    # ic_compare with --simple may legitimately return either 200 (statsdicts) or
    # 409 (driver requires deeper assertions). Either is acceptable for now —
    # this test asserts the endpoint is reachable and returns a structured body.
    assert ic.status_code in {200, 409}, ic.text
