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

from fastapi import FastAPI

from radvel.api.config import get_settings
from radvel.api.jobs import JobRegistry, JobRunner
from radvel.api.routers import files, health, jobs, pipeline, runs


log = logging.getLogger("radvel.api")


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
        version="1.6.0",
        description=(
            "HTTP interface to the RadVel Keplerian-orbit pipeline. "
            "All synchronous endpoints follow the CLI workflow "
            "fit → derive → ic → tables. Long-running mcmc/ns "
            "endpoints arrive in v1.6 milestone M3."
        ),
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(pipeline.router)
    app.include_router(jobs.router)
    app.include_router(files.router)

    return app


app = create_app()
