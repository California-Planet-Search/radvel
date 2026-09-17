Changelog
=========

1.6.5 (2026-09-17)
------------------

- **A queued job is no longer stranded when the API process dies.**
  ``JobRegistry.reconcile_orphaned()`` repaired rows left in ``running``
  whose worker pid was gone, but never looked at ``queued``.
  ``JobRunner.submit()`` inserts the row and *then* hands the work to the
  ``ProcessPoolExecutor``: if the process dies in between, the row keeps
  ``state='queued'`` with no pid, so it is not ``running`` and reconcile
  skipped it — while the executor only ever knows jobs submitted during
  its own process's life, so nothing would dispatch it again. Three MCMC
  jobs sat queued for six days behind this gap and a container recreate
  did not rescue them; every client meanwhile blocked for its full read
  timeout, which is indistinguishable from a slow fit. Such rows now
  reach ``failed`` with an error telling the caller to resubmit.
  Failing rather than re-dispatching is deliberate: an unattended restart
  that silently starts N queued MCMC fits is a thundering herd on a
  shared host, and the caller has the context to decide whether the fit
  is still wanted.
- **The container health check can pass again.** It ran ``python -c
  "import radvel._kepler"`` as a second clause, starting a fresh
  interpreter and importing a compiled extension inside a 5-second
  budget on a host that may be running several MCMC fits. ``/healthz``
  already imports that extension in-process and reports it as
  ``kepler_c``, so the clause re-answered a question the endpoint had
  just answered. The check timed out permanently — a failing streak over
  2200 while the service was healthy — and a container that is always
  unhealthy carries no signal, which is why the queue outage above went
  unnoticed for a fortnight. The check now greps the payload the
  endpoint already returns: same assertion, no interpreter start-up.
  ``curl --fail`` alone would not do, because a missing extension
  returns 200 with ``status: "degraded"``.
- **The service user has the home directory it is told it has.**
  ``useradd --no-create-home`` left ``$HOME=/home/radvel`` pointing at
  nothing, so matplotlib logged ``mkdir -p failed for path
  /home/radvel/.config/matplotlib`` on every start and fell back to
  ``/tmp``. Functionally harmless, but it is the first thing an outage
  investigation finds and it cost real time before being ruled out.
- **Optional API key authentication for the HTTP service.** Setting
  ``RADVEL_API_AUTH_KEY`` makes every request other than ``/healthz``
  and ``/version`` require a matching ``X-API-Key`` header. Leaving it
  unset keeps the middleware a no-op, so deployments that rely on
  network-level controls — a localhost-only bind, or a host firewall —
  need no configuration change. Keys are compared with
  ``secrets.compare_digest``, so a wrong key takes the same time to
  reject however many leading bytes happen to be correct. The two health
  paths stay exempt so monitoring keeps working against an authenticated
  service.

1.6.4 (2026-09-10)
------------------

- Re-cut release. The v1.6.3 tag exists on GitHub but never reached PyPI,
  failing the same way v1.6.2 did. The ``numpy<2.3`` pin added in 1.6.3
  was placed in ``CIBW_BEFORE_BUILD``, which installs into cibuildwheel's
  outer environment. ``pip wheel`` then builds under PEP 517 isolation: it
  creates a fresh environment and reinstalls the ``[build-system]
  requires`` list from ``pyproject.toml`` from scratch, so the pin was
  never present in the environment that actually performed the build and
  an unpinned NumPy 2.5.3 was source-built again.
- The pin now lives in ``[build-system] requires`` in ``pyproject.toml``,
  which is the list build isolation reads. It carries a
  ``python_version < '3.14'`` marker so that a flat pin does not make
  radvel unbuildable from source on Python versions that ``numpy<2.3``
  does not support; ``requires-python`` has no upper bound.

1.6.3 (2026-09-10)
------------------

- Re-cut release. The v1.6.2 tag exists on GitHub but never reached
  PyPI: the ``build-wheels (ubuntu-latest)`` job failed, so the
  ``publish`` job was skipped. NumPy 2.5.3 stopped shipping
  glibc-2.17 wheels for CPython 3.12 and 3.13 (it now ships
  ``manylinux_2_27``/``manylinux_2_28`` only), and the Linux wheels
  are built in the ``manylinux2014`` image, which is glibc 2.17. pip
  could not use those wheels, fell back to the NumPy sdist, and the
  source build failed with ``NumPy requires GCC >= 10.3``.
- The build-time NumPy is now pinned to ``numpy<2.3``, which still
  publishes ``manylinux_2_17`` wheels across the whole build matrix
  (2.2.6 for CPython 3.11 through 3.13, 2.0.2 for 3.9). Building an
  extension against an older NumPy 2.x is safe at runtime: the
  NumPy 2 C ABI is forward compatible, so the wheels keep working
  with newer NumPy at import time. Pinning here rather than moving
  the builder to ``manylinux_2_28`` keeps publishing Linux wheels
  usable on older cluster operating systems such as CentOS 7.

1.6.2 (2026-09-10)
------------------

- Docker image no longer ships the build-stage wheels. They were
  copied into their own layer with ``COPY``, so the ``rm -rf /wheels``
  that followed could only write a whiteout on top of a layer that
  was already committed, and every pull carried them. The wheels are
  now bind-mounted from the builder stage with
  ``RUN --mount=type=bind``, which is never committed to a layer, so
  there is nothing left to remove.
- Docs build pins ``mistune==3.3.3``.

1.6.1 (2026-05-14)
------------------

- Re-cut release. The v1.6.0 release tag exists on GitHub and GHCR
  (``ghcr.io/california-planet-search/radvel-api:1.6.0``) but never
  reached PyPI because the ``build-wheels (macos-13)`` job sat queued
  for 19+ hours on GitHub's effectively-unavailable Intel macOS
  runners. v1.6.1 drops ``macos-13`` from the matrix and
  cross-compiles x86_64 macOS wheels from the ``macos-14`` ARM
  runner via cibuildwheel's ``CIBW_ARCHS_MACOS=arm64 x86_64`` — same
  wheel coverage, no dependence on Intel runners.

1.6.0 (2026-05-11)
------------------

**HTTP API + Docker service.** RadVel now ships an FastAPI-based HTTP
service that wraps the entire CLI workflow behind a JSON API plus an
optional browser UI. See :doc:`api_service` and :doc:`ui_guide`.

- New synchronous endpoints: ``POST /runs`` (create from JSON setup),
  ``POST /runs/{id}/{fit,derive,ic,tables,plots,report}``,
  ``GET /runs/{id}`` (parsed ``.stat``),
  ``GET /runs/{id}/files[/{name}]`` (download).
- New async endpoints with SQLite-persisted job state and SIGTERM
  cancellation: ``POST /runs/{id}/{mcmc,ns}`` returns ``202 {job_id}``;
  ``GET /jobs/{id}`` exposes live MCMC convergence telemetry from
  :any:`radvel.mcmc.statevars`; ``DELETE /jobs/{id}`` cancels.
- New optional UI at ``/ui`` (vanilla HTML + Tailwind CDN, no build
  step). Toggle with ``RADVEL_API_ENABLE_UI``.
- ``radvel serve`` CLI subcommand boots the service via uvicorn.
- Multi-stage Dockerfile bundles TeX Live so ``/report`` works out of
  the box. Multi-arch (amd64 + arm64) image published to
  ``ghcr.io/california-planet-search/radvel-api`` on every release.

**Library**

- :func:`radvel.utils.initialize_posterior_from_dict` builds a
  ``Posterior`` from a JSON-shaped dict; verified bit-for-bit
  consistent with the legacy file path.
- :func:`radvel.mcmc.mcmc` accepts an additive ``progress_callback``
  kwarg invoked at the end of each convergence check.
- The Cython Kepler extension build is unchanged; the existing
  NumPy fallback still runs when ``radvel._kepler`` is unavailable.

1.5.7 (2025-XX)
---------------

- Fix Coveralls upload on Dependabot PRs (#418).
- Build platform-specific wheels with cibuildwheel (#413).
