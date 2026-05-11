Run RadVel as a service
=======================

.. versionadded:: 1.6

RadVel 1.6 ships an HTTP service that wraps the entire CLI workflow
(``fit → mcmc/ns → derive → ic → tables → plots → report``) behind a
JSON API plus an optional browser UI. It exists for the cases where the
Python REPL or CLI is awkward — web frontends, language-agnostic
clients, batch pipelines, multi-tenant deployments — and runs as a
single self-contained Docker image.

When to use the service
-----------------------

Stick with the CLI for one-off interactive analyses on your laptop.
Reach for the service when:

- You want to fit RVs from a frontend that isn't Python (JS, Go, R…).
- You need a long-running MCMC to survive a notebook restart.
- Multiple users (or a CI pipeline) need to drive the same pipeline.
- You want a containerised, reproducible deployment.

Two-line quickstart
-------------------

.. code-block:: console

   $ docker run --rm -p 8000:8000 \
       -v rv-data:/data \
       ghcr.io/california-planet-search/radvel-api:1.6
   $ open http://localhost:8000/ui

Visit ``/docs`` for live Swagger UI, ``/redoc`` for ReDoc.

.. note::

   The image runs as a non-root operator (``uid 10001``). The example
   above uses a named docker volume (``rv-data``) which the daemon
   creates with the right ownership. If you prefer a host bind-mount
   (``-v "$PWD/.runs:/data"``) so the run outputs land on the host
   filesystem, make the directory writable by uid 10001 first — either
   ``chmod 0777 ./.runs`` or run with ``--user $(id -u):$(id -g)``.
   Otherwise the lifespan crashes on the first write with
   ``PermissionError``.

JSON setup file
---------------

The service replaces the Python setup module with a JSON document. Every
attribute the legacy ``epic203771098.py`` setup exposes has a JSON
equivalent. A pared-down example::

   {
     "starname": "epic203771098",
     "nplanets": 2,
     "instnames": ["j"],
     "fitting_basis": "per tc secosw sesinw k",
     "bjd0": 2454833.0,
     "planet_letters": {"1": "b", "2": "c"},
     "params": {
       "per1": {"value": 20.885258, "vary": false},
       "tc1":  {"value": 2072.79438, "vary": false},
       "secosw1": {"value": 0.0, "vary": false},
       "sesinw1": {"value": 0.1, "vary": false},
       "k1":   {"value": 10.0},
       "per2": {"value": 42.363011, "vary": false},
       "tc2":  {"value": 2082.62516, "vary": false},
       "secosw2": {"value": 0.0, "vary": false},
       "sesinw2": {"value": 0.1, "vary": false},
       "k2":   {"value": 10.0},
       "dvdt": {"value": 0.0},
       "curv": {"value": 0.0},
       "gamma_j": {"value": 1.0, "vary": false, "linear": true},
       "jit_j":   {"value": 2.6}
     },
     "data": {"kind": "dataset_ref", "dataset": "epic203771098.csv"},
     "priors": [
       {"type": "eccentricity", "num_planets": 2},
       {"type": "positivek",    "num_planets": 2},
       {"type": "jeffreys",     "param": "k1", "minval": 0.01, "maxval": 1000},
       {"type": "jeffreys",     "param": "k2", "minval": 0.01, "maxval": 1000},
       {"type": "hardbounds",   "param": "jit_j", "minval": 0.0, "maxval": 15.0},
       {"type": "gaussian",     "param": "dvdt", "mu": 0.0, "sigma": 1.0},
       {"type": "gaussian",     "param": "curv", "mu": 0.0, "sigma": 0.1}
     ],
     "stellar": {"mstar": 1.12, "mstar_err": 0.05}
   }

The four ``data`` shapes are:

- ``{"kind": "inline", "rows": [{"time": ..., "mnvel": ..., "errvel": ..., "tel": ...}]}``
- ``{"kind": "csv_base64", "csv_base64": "...", "separator": ","}``
- ``{"kind": "server_path", "path": "/some/csv"}`` *(only allowed if the
  path lives under ``RADVEL_DATA_ALLOWLIST``)*
- ``{"kind": "dataset_ref", "dataset": "epic203771098.csv"}`` *(any
  fixture in* ``radvel.DATADIR`` *)*

Pydantic v2 validates every payload and returns ``422`` with a
field-by-field error report on bad input. Hit ``/docs`` for the full
machine-readable schema.

End-to-end pipeline (``curl``)
------------------------------

.. code-block:: console

   # 1. Create the run
   $ RID=$(curl -s -X POST http://localhost:8000/runs \
                -H 'content-type: application/json' \
                -d @epic.json | jq -r .run_id)

   # 2. MAP fit (synchronous, ~1 s)
   $ curl -s -X POST "http://localhost:8000/runs/$RID/fit" -d '{}' \
          -H 'content-type: application/json' | jq

   # 3. MCMC (asynchronous — returns 202 with a job_id)
   $ JID=$(curl -s -X POST "http://localhost:8000/runs/$RID/mcmc" \
                -H 'content-type: application/json' \
                -d '{"nsteps":1000, "nwalkers":40, "ensembles":4}' | jq -r .job_id)

   # 4. Poll the job until terminal
   $ while :; do
       state=$(curl -s "http://localhost:8000/jobs/$JID" | jq -r .state)
       echo "$state"; [[ $state =~ succeeded|failed|cancelled ]] && break
       sleep 2
     done

   # 5. Derive, tables, report
   $ curl -s -X POST "http://localhost:8000/runs/$RID/derive" -d '{}' \
          -H 'content-type: application/json' | jq
   $ curl -s -X POST "http://localhost:8000/runs/$RID/tables" \
          -H 'content-type: application/json' \
          -d '{"types":["params","priors","rv"]}' | jq
   $ curl -s -X POST "http://localhost:8000/runs/$RID/report" -d '{}' \
          -H 'content-type: application/json' | jq

   # 6. Download the PDF
   $ curl -OJ "http://localhost:8000/runs/$RID/files/${RID}_results.pdf"

The same flow in Python with ``httpx`` ships in
:doc:`tutorials/api_quickstart`.

Async job semantics
-------------------

``mcmc`` and ``ns`` run as background jobs because they take minutes to
hours. State machine::

    queued → running → succeeded | failed | cancelled

Job state lives in SQLite at ``${RADVEL_API_DB_PATH}`` (default
``/data/jobs.db``) so the container can restart without losing
records. ``GET /jobs/{id}`` exposes live MCMC convergence telemetry
(:doc:`autocorrintro` covers the convergence quantities). ``DELETE
/jobs/{id}`` issues SIGTERM to the worker; the job reaches
``cancelled`` within a few seconds.

If the host crashes mid-job, startup runs a reconciliation pass that
marks any stranded ``running`` rows as ``failed: process gone (likely
container restart)`` so polling clients see a clean terminal state.

Environment variables
---------------------

==============================  ==============================  ==============
Variable                        Default                          Meaning
==============================  ==============================  ==============
``RADVEL_API_RUNS_DIR``         ``/data/runs``                   Per-run output dir.
``RADVEL_API_DB_PATH``          ``/data/jobs.db``                SQLite jobs table.
``RADVEL_API_HOST``             ``0.0.0.0``                      uvicorn bind host.
``RADVEL_API_PORT``             ``8000``                         uvicorn port.
``RADVEL_API_WORKERS``          ``1``                            ProcessPool size for MCMC/NS.
``RADVEL_API_ALLOW_PY_UPLOAD``  ``false``                        Permit ``POST /runs/upload-py`` and ``userdefined`` priors. **Off by default** because it executes user-supplied Python.
``RADVEL_API_ENABLE_UI``        ``true``                         Mount ``/ui``. Set to ``false`` for headless deployments.
``RADVEL_DATADIR``              install-relative                 Where ``dataset_ref`` looks for fixtures.
``MPLBACKEND``                  ``Agg`` *(set by package init)*  Matplotlib backend.
==============================  ==============================  ==============

Security notes
--------------

RadVel 1.6 ships **no first-party authentication**. The service trusts
every reachable client. For shared deployments:

- Put a reverse proxy in front (nginx or Caddy) and terminate auth
  there. Both `basic auth`_ and mTLS work fine.
- Keep ``RADVEL_API_ALLOW_PY_UPLOAD=false``. The ``.py`` setup-file
  upload endpoint and the ``userdefined`` prior both ``exec`` user
  Python and are **only** appropriate for trusted local users.
- Set ``RADVEL_API_ENABLE_UI=false`` on headless / multi-tenant nodes.
- Mount ``/data`` on a local volume; the SQLite jobs table is the only
  shared state.

.. _`basic auth`: https://caddyserver.com/docs/caddyfile/directives/basic_auth

Multi-arch image
----------------

Releases publish ``ghcr.io/california-planet-search/radvel-api`` for
both ``linux/amd64`` and ``linux/arm64``. Docker picks the right
architecture automatically; force it with ``--platform`` if you need
to.

Known limitations (1.6)
-----------------------

- No built-in auth (mitigation: reverse proxy).
- Single-host job runner — no clustering.
- Pickle files are tied to the radvel version that wrote them; the
  service refuses to read pickles from a different version with a
  ``409 version_mismatch``.
- `RADVEL_API_WORKERS=N` × ``ensembles`` should not exceed the host
  CPU count.
