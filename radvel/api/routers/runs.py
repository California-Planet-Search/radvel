"""Run CRUD endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

import radvel
from radvel.api import schemas
from radvel.api.config import Settings, get_settings
from radvel.api.jobs import JobRegistry
from radvel.api.runs import RunNotFound, RunRegistry, is_valid_run_id


router = APIRouter(prefix="/runs", tags=["runs"])


def _registry() -> RunRegistry:
    return RunRegistry(settings=get_settings())


@router.post(
    "",
    response_model=schemas.RunCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_run(
    payload: schemas.RunCreateRequest,
    registry: RunRegistry = Depends(_registry),
) -> schemas.RunCreateResponse:
    """Materialise a new run directory from a typed setup payload."""
    record = registry.create(
        payload=payload.model_dump(mode="json"),
        starname=payload.starname,
        fitting_basis=payload.fitting_basis,
        nplanets=payload.nplanets,
    )
    return schemas.RunCreateResponse(
        run_id=record.run_id,
        outputdir=str(record.outputdir),
        created_at=record.created_at,
    )


@router.get("", response_model=List[schemas.RunSummary])
def list_runs(
    registry: RunRegistry = Depends(_registry),
) -> List[schemas.RunSummary]:
    return [
        schemas.RunSummary(
            run_id=r.run_id,
            starname=r.starname,
            created_at=r.created_at,
            fitting_basis=r.fitting_basis,
            nplanets=r.nplanets,
        )
        for r in registry.iter_runs()
    ]


@router.get("/{run_id}", response_model=schemas.RunStatus)
def get_run(
    run_id: str,
    registry: RunRegistry = Depends(_registry),
) -> schemas.RunStatus:
    if not is_valid_run_id(run_id):
        raise HTTPException(status_code=404, detail="unknown run_id")
    try:
        record = registry.get(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail="unknown run_id")
    job_row = JobRegistry(settings=get_settings()).active_job_for_run(run_id)
    active_job = (
        schemas.JobSummary(
            job_id=job_row.job_id,
            run_id=job_row.run_id,
            kind=job_row.kind,
            state=job_row.state,
            submitted_at=job_row.submitted_at,
            started_at=job_row.started_at,
            finished_at=job_row.finished_at,
        ) if job_row is not None else None
    )
    return schemas.RunStatus(
        run_id=record.run_id,
        starname=record.starname,
        fitting_basis=record.fitting_basis,
        nplanets=record.nplanets,
        created_at=record.created_at,
        radvel_version=record.radvel_version,
        state=registry.stat_dict(record),
        active_job=active_job,
    )


@router.post(
    "/upload-py",
    response_model=schemas.RunCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_py_setup(
    setup_file: UploadFile = File(..., description="A radvel setup .py file"),
    registry: RunRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> schemas.RunCreateResponse:
    """Create a run by uploading an existing ``radvel`` setup ``.py``.

    Gated by ``RADVEL_API_ALLOW_PY_UPLOAD`` because it executes arbitrary
    Python from the request body. Off by default.
    """
    if not settings.allow_py_upload:
        raise HTTPException(
            status_code=415,
            detail="setup-file upload disabled (set RADVEL_API_ALLOW_PY_UPLOAD=true)",
        )
    name = (setup_file.filename or "").lower()
    if not name.endswith(".py"):
        raise HTTPException(
            status_code=400, detail="filename must end with .py"
        )
    body = await setup_file.read(settings.max_upload_bytes + 1)
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="upload exceeds {} bytes".format(settings.max_upload_bytes),
        )
    try:
        record = registry.create_from_py(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return schemas.RunCreateResponse(
        run_id=record.run_id,
        outputdir=str(record.outputdir),
        created_at=record.created_at,
    )


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(
    run_id: str,
    keep_files: bool = False,
    registry: RunRegistry = Depends(_registry),
):
    if not is_valid_run_id(run_id):
        # Idempotent: deleting an unknown id is a no-op.
        return
    registry.delete(run_id, keep_files=keep_files)
