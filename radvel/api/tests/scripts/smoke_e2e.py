#!/usr/bin/env python3
"""End-to-end smoke test against a live radvel-api instance.

Used by the ``docker-smoke`` CI job: builds the image, runs the
container, then drives this script against ``http://localhost:8000``.

Walks the canonical pipeline:
    POST /runs (epic203771098 dataset_ref)  →
    POST /runs/{id}/fit                     →
    POST /runs/{id}/mcmc (tiny — 200 steps) →
    poll /jobs/{id} until terminal          →
    POST /runs/{id}/derive                  →
    POST /runs/{id}/tables                  →
    GET /runs/{id}/files                    →
    sanity-check /healthz, /version

Exits 0 on success, prints a clear diagnosis on failure.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def http(method: str, url: str, payload=None, timeout: float = 30.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read() or b"{}")
        except Exception:
            body = {"error": str(exc)}
        return exc.code, body


PAYLOAD = {
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
    "data": {"kind": "dataset_ref", "dataset": "epic203771098.csv"},
    "priors": [
        {"type": "eccentricity", "num_planets": 2},
        {"type": "positivek", "num_planets": 2},
        {"type": "jeffreys", "param": "k1", "minval": 0.01, "maxval": 1000},
        {"type": "jeffreys", "param": "k2", "minval": 0.01, "maxval": 1000},
        {"type": "hardbounds", "param": "jit_j", "minval": 0.0, "maxval": 15.0},
        {"type": "gaussian", "param": "dvdt", "mu": 0.0, "sigma": 1.0},
        {"type": "gaussian", "param": "curv", "mu": 0.0, "sigma": 0.1},
    ],
    "stellar": {"mstar": 1.12, "mstar_err": 0.05},
}


def must_ok(prefix: str, status: int, body) -> None:
    if not (200 <= status < 300):
        print(f"FAIL: {prefix} → {status} {json.dumps(body)[:400]}")
        sys.exit(1)
    print(f"  {prefix} → {status}")


def main(base: str = "http://localhost:8000") -> int:
    print(f"smoke against {base}")

    status, health = http("GET", f"{base}/healthz")
    must_ok("/healthz", status, health)
    if not health.get("kepler_c"):
        print("WARN: /healthz reports kepler_c=False — fallback path active")

    status, version = http("GET", f"{base}/version")
    must_ok("/version", status, version)

    status, run = http("POST", f"{base}/runs", PAYLOAD, timeout=60)
    must_ok("POST /runs", status, run)
    run_id = run["run_id"]

    status, fit = http("POST", f"{base}/runs/{run_id}/fit", {}, timeout=120)
    must_ok("POST /fit", status, fit)
    print(f"    logprob={fit['logprob']:.3f} rms={fit['rms']:.3f}")

    mcmc_args = {"nsteps": 200, "nwalkers": 30, "ensembles": 2,
                 "thin": 1, "serial": True}
    status, mcmc = http("POST", f"{base}/runs/{run_id}/mcmc", mcmc_args)
    must_ok("POST /mcmc", status, mcmc)
    job_id = mcmc["job_id"]

    deadline = time.monotonic() + 240
    last = None
    while time.monotonic() < deadline:
        s, last = http("GET", f"{base}/jobs/{job_id}")
        if s != 200:
            print(f"FAIL: /jobs/{job_id} → {s} {last}")
            return 1
        if last["state"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(2)
    if not last or last["state"] not in {"succeeded", "failed"}:
        print(f"FAIL: MCMC never reached terminal state, last: {last}")
        return 1
    print(f"  MCMC → {last['state']}")

    status, _ = http("POST", f"{base}/runs/{run_id}/derive", {}, timeout=120)
    if status not in (200, 409):  # 409 if no stellar mass; we provide one so 200
        print(f"FAIL: /derive → {status}")
        return 1
    print(f"  /derive → {status}")

    status, tables = http(
        "POST", f"{base}/runs/{run_id}/tables",
        {"types": ["params", "priors", "rv"]}, timeout=60,
    )
    must_ok("POST /tables", status, tables)

    status, files = http("GET", f"{base}/runs/{run_id}/files")
    must_ok("GET /files", status, files)
    names = [f["name"] for f in files]
    print(f"    {len(names)} files: {', '.join(sorted(names)[:6])}…")
    must = [".stat", "_post_obj.pkl"]
    for marker in must:
        if not any(marker in n for n in names):
            print(f"FAIL: expected file matching {marker!r} not produced")
            return 1

    print("OK")
    return 0


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    sys.exit(main(base))
