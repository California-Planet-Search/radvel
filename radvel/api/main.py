"""FastAPI application factory for the RadVel HTTP service.

Importing this module requires the ``[api]`` extra (FastAPI + uvicorn +
pydantic v2 + pydantic-settings). For local development::

    pip install -e '.[api]'
    radvel serve --host 0.0.0.0 --port 8000

In production the same app runs under uvicorn inside the radvel-api
Docker image (see ``Dockerfile``).
"""

from __future__ import annotations

import contextlib
import logging
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import radvel as _radvel
from radvel.api.config import get_settings
from radvel.api.jobs import JobRegistry, JobRunner
from radvel.api.routers import files, health, jobs, pipeline, runs, ui


log = logging.getLogger("radvel.api")

# Paths that bypass API key auth so health monitoring always works.
_AUTH_EXEMPT = frozenset({"/healthz", "/version"})


class _APIKeyMiddleware(BaseHTTPMiddleware):
    """Require ``X-API-Key: <key>`` on every non-exempt request when
    ``RADVEL_API_KEY`` is configured.  When the env var is unset the
    middleware is a no-op, leaving network-level controls (e.g.
    localhost-only binding) as the sole access gate.
    """

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.auth_key or request.url.path in _AUTH_EXEMPT:
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(provided, settings.auth_key):
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )
        return await call_next(request)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the C extension and ensure the runs/db dirs exist."""
    settings = get_settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import radvel._kepler  # noqa: F401
        log.info("radvel._kepler C extension loaded")
    except Exception as exc:  # pragma: no cover — exercised only when the .so is missing
        log.warning("radvel._kepler not available, falling back to NumPy: %s", exc)

    # Repair stale 'running' rows from a prior process that didn't shut down
    # cleanly, then start the executor for new jobs.
    job_registry = JobRegistry(settings=settings)
    repaired = job_registry.reconcile_orphaned()
    if repaired:
        log.warning("reconciled %d orphaned job(s) at startup", repaired)
    app.state.job_runner = JobRunner(job_registry)
    try:
        yield
    finally:
        app.state.job_runner.shutdown(wait=False)


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Wired so all endpoints are discoverable at ``/docs`` (Swagger UI) and
    ``/redoc``. Long-running job endpoints land in M3; the static UI
    lands in M4.5.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="RadVel HTTP API",
        version=_radvel.__version__,
        description=(
            "HTTP interface to the RadVel Keplerian-orbit pipeline. "
            "All synchronous endpoints follow the CLI workflow "
            "fit → derive → ic → tables. Long-running mcmc/ns "
            "endpoints arrive in v1.6 milestone M3."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(_APIKeyMiddleware)

    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(pipeline.router)
    app.include_router(jobs.router)
    app.include_router(files.router)
    ui.register(app)

    return app


app = create_app()
