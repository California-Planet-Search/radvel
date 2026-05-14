Changelog
=========

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
