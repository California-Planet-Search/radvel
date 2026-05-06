"""File listing and download endpoints for a run's output directory."""

from __future__ import annotations

import datetime as _dt
import mimetypes
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from radvel.api import schemas
from radvel.api.config import get_settings
from radvel.api.runs import RunNotFound, RunRegistry, is_valid_run_id


router = APIRouter(prefix="/runs/{run_id}/files", tags=["files"])


def _registry() -> RunRegistry:
    return RunRegistry(settings=get_settings())


def _resolve(run_id: str, registry: RunRegistry):
    if not is_valid_run_id(run_id):
        raise HTTPException(status_code=404, detail="unknown run_id")
    try:
        return registry.get(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail="unknown run_id")


def _safe_path(outputdir: Path, filename: str) -> Path:
    """Resolve `filename` strictly inside `outputdir`. Defends against `..`."""
    base = outputdir.resolve()
    try:
        target = (base / filename).resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="file not found")
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=404, detail="file not found")
    return target


@router.get("", response_model=List[schemas.FileEntry])
def list_files(
    run_id: str,
    registry: RunRegistry = Depends(_registry),
) -> List[schemas.FileEntry]:
    record = _resolve(run_id, registry)
    entries: List[schemas.FileEntry] = []
    for path in sorted(record.outputdir.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        ctype, _ = mimetypes.guess_type(path.name)
        entries.append(schemas.FileEntry(
            name=path.name,
            size=stat.st_size,
            mtime=_dt.datetime.fromtimestamp(stat.st_mtime, _dt.timezone.utc),
            content_type=ctype,
        ))
    return entries


@router.get("/{filename}")
def download_file(
    run_id: str,
    filename: str,
    registry: RunRegistry = Depends(_registry),
) -> FileResponse:
    record = _resolve(run_id, registry)
    target = _safe_path(record.outputdir, filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    ctype, _ = mimetypes.guess_type(target.name)
    return FileResponse(
        path=target,
        media_type=ctype or "application/octet-stream",
        filename=target.name,
    )
