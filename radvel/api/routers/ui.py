"""Optional static operator UI mounted at ``/ui``.

Disabled when ``RADVEL_API_ENABLE_UI=false`` — the routes simply aren't
registered, so both ``/ui`` and ``/ui/{path}`` return 404. This router
is the only thing that imports the static files; nothing else depends
on them.
"""

from __future__ import annotations

import importlib.resources as _ir
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from radvel.api.config import get_settings


router = APIRouter(tags=["ui"])

_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.tailwindcss.com 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
    "img-src 'self' data:; "
    "connect-src 'self'"
)


def _static_dir() -> Path:
    """Resolve the on-disk path of ``radvel/api/static`` for this install."""
    return Path(_ir.files("radvel.api").joinpath("static"))  # type: ignore[arg-type]


def register(app) -> None:
    """Attach UI routes if the kill-switch flag is on."""
    settings = get_settings()
    if not settings.enable_ui:
        return
    app.include_router(router)


@router.get("/ui", include_in_schema=False)
def ui_index() -> Response:
    path = _static_dir() / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="UI not bundled")
    return FileResponse(
        path=path,
        media_type="text/html",
        headers={"Content-Security-Policy": _CSP},
    )


@router.get("/ui/{filename:path}", include_in_schema=False)
def ui_asset(filename: str) -> Response:
    base = _static_dir().resolve()
    try:
        target = (base / filename).resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="not found")
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=404, detail="not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    ctype, _ = mimetypes.guess_type(target.name)
    return FileResponse(
        path=target,
        media_type=ctype or "application/octet-stream",
        headers={"Content-Security-Policy": _CSP},
    )
