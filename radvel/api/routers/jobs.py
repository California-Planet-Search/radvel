"""Async job endpoints — kick MCMC/NS, poll progress, cancel."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from radvel.api import schemas
from radvel.api.config import get_settings
from radvel.api.drivers_adapter import _run_mcmc_worker, _run_ns_worker
from radvel.api.jobs import (
    JobActiveError,
    JobNotFound,
    JobRegistry,
    JobRunner,
)
from radvel.api.progress import ProgressWriter
from radvel.api.runs import RunNotFound, RunRegistry, is_valid_run_id


router = APIRouter(tags=["jobs"])


def _runner(request: Request) -> JobRunner:
    runner: Optional[JobRunner] = getattr(request.app.state, "job_runner", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="job runner not initialised")
    return runner


def _registry() -> JobRegistry:
    return JobRegistry(settings=get_settings())


def _run_registry() -> RunRegistry:
    return RunRegistry(settings=get_settings())


def _resolve_run(run_id: str, run_registry: RunRegistry):
    if not is_valid_run_id(run_id):
        raise HTTPException(status_code=404, detail="unknown run_id")
    try:
        return run_registry.get(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail="unknown run_id")


def _job_to_summary(job) -> schemas.JobSummary:
    return schemas.JobSummary(
        job_id=job.job_id,
        run_id=job.run_id,
        kind=job.kind,
        state=job.state,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _job_to_status(job, progress: dict) -> schemas.JobStatus:
    return schemas.JobStatus(
        job_id=job.job_id,
        run_id=job.run_id,
        kind=job.kind,
        state=job.state,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress=schemas.JobProgress(**progress),
        error=job.error,
    )


@router.post(
    "/runs/{run_id}/mcmc",
    response_model=schemas.JobKickoffResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_mcmc(
    run_id: str,
    body: schemas.MCMCRequest,
    runner: JobRunner = Depends(_runner),
    run_registry: RunRegistry = Depends(_run_registry),
) -> schemas.JobKickoffResponse:
    _resolve_run(run_id, run_registry)
    try:
        row = runner.submit(run_id, "mcmc", body.model_dump(mode="json"),
                             worker=_run_mcmc_worker)
    except JobActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return schemas.JobKickoffResponse(
        job_id=row.job_id, run_id=row.run_id, kind=row.kind, state=row.state,
    )


@router.post(
    "/runs/{run_id}/ns",
    response_model=schemas.JobKickoffResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_ns(
    run_id: str,
    body: schemas.NSRequest,
    runner: JobRunner = Depends(_runner),
    run_registry: RunRegistry = Depends(_run_registry),
) -> schemas.JobKickoffResponse:
    _resolve_run(run_id, run_registry)
    try:
        row = runner.submit(run_id, "ns", body.model_dump(mode="json"),
                             worker=_run_ns_worker)
    except JobActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return schemas.JobKickoffResponse(
        job_id=row.job_id, run_id=row.run_id, kind=row.kind, state=row.state,
    )


@router.get("/jobs", response_model=List[schemas.JobStatus])
def list_jobs(
    run_id: Optional[str] = None,
    state: Optional[str] = None,
    registry: JobRegistry = Depends(_registry),
    run_registry: RunRegistry = Depends(_run_registry),
) -> List[schemas.JobStatus]:
    rows = registry.list(run_id=run_id, state=state)
    out: List[schemas.JobStatus] = []
    for row in rows:
        progress = _read_progress(row, run_registry)
        out.append(_job_to_status(row, progress))
    return out


@router.get("/jobs/{job_id}", response_model=schemas.JobStatus)
def get_job(
    job_id: str,
    registry: JobRegistry = Depends(_registry),
    run_registry: RunRegistry = Depends(_run_registry),
) -> schemas.JobStatus:
    try:
        row = registry.get(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="unknown job_id")
    progress = _read_progress(row, run_registry)
    return _job_to_status(row, progress)


@router.delete("/jobs/{job_id}", response_model=schemas.JobStatus,
               status_code=status.HTTP_202_ACCEPTED)
def cancel_job(
    job_id: str,
    runner: JobRunner = Depends(_runner),
    run_registry: RunRegistry = Depends(_run_registry),
) -> schemas.JobStatus:
    try:
        row = runner.cancel(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="unknown job_id")
    progress = _read_progress(row, run_registry)
    return _job_to_status(row, progress)


def _read_progress(row, run_registry: RunRegistry) -> dict:
    """Read the on-disk progress file for the run, if any."""
    try:
        record = run_registry.get(row.run_id)
    except RunNotFound:
        return {}
    progress_path = record.outputdir / "mcmc_progress.json"
    return ProgressWriter(progress_path).read()
