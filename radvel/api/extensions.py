"""Public hooks for packages that add job kinds to the RadVel HTTP service.

An extension (e.g. ``rvsearch``) builds the RadVel app, registers its job
kinds, and mounts its own router::

    from radvel.api.main import create_app
    from radvel.api import extensions

    extensions.register_job_kind("search")
    app = create_app()
    app.include_router(my_router)

Its endpoints submit through the shared runner
(``request.app.state.job_runner.submit(run_id, "search", params, worker=fn)``)
so they get the same SQLite job rows, one-active-job-per-run guard,
cancellation, and ``GET /jobs/{id}`` polling as MCMC. ``fn`` must be a
module-level function ``(run_id, params_json) -> dict`` (it runs in a
child process). Inside it, use :func:`worker_setup` and
:func:`capture_output`, and write progress snapshots to
:func:`progress_path` so ``GET /jobs/{id}`` returns them.
"""

from __future__ import annotations

from pathlib import Path

from radvel.api.drivers_adapter import AdapterError
from radvel.api.drivers_adapter import _capture as capture_output
from radvel.api.drivers_adapter import _install_sigterm_handler
from radvel.api.drivers_adapter import _resolve_record
from radvel.api.jobs import JobActiveError, JobRunner, progress_filename, register_job_kind  # noqa: F401
from radvel.api.progress import ProgressWriter
from radvel.api.runs import RunNotFound, RunRecord, RunRegistry, is_valid_run_id

__all__ = [
    "AdapterError",
    "JobActiveError",
    "JobRunner",
    "ProgressWriter",
    "RunNotFound",
    "RunRecord",
    "RunRegistry",
    "capture_output",
    "is_valid_run_id",
    "progress_filename",
    "progress_path",
    "register_job_kind",
    "worker_setup",
]


def worker_setup(run_id: str) -> RunRecord:
    """Call first inside a job worker: makes SIGTERM cancellable, returns the run."""
    _install_sigterm_handler()
    return _resolve_record(run_id)


def progress_path(record: RunRecord, kind: str) -> Path:
    """Where a worker of ``kind`` writes progress for ``GET /jobs/{id}``."""
    return record.outputdir / progress_filename(kind)
