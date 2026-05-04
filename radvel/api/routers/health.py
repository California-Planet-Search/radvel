"""Health and version endpoints."""

from __future__ import annotations

import platform
import sys

from fastapi import APIRouter

import radvel
from radvel.api import schemas


router = APIRouter()


@router.get("/healthz", response_model=schemas.HealthResponse, tags=["meta"])
def healthz() -> schemas.HealthResponse:
    """Liveness probe.

    ``kepler_c`` reports whether the Cython Kepler-solver extension is
    importable. The pure-NumPy fallback still works correctly, but a
    ``False`` here is a sign that the build artefacts are out of sync
    with the installed package.
    """
    try:
        import radvel._kepler  # noqa: F401
        kepler_c = True
    except Exception:
        kepler_c = False
    return schemas.HealthResponse(
        status="ok" if kepler_c else "degraded",
        kepler_c=kepler_c,
        version=radvel.__version__,
    )


@router.get("/version", response_model=schemas.VersionResponse, tags=["meta"])
def version() -> schemas.VersionResponse:
    return schemas.VersionResponse(
        radvel=radvel.__version__,
        api=radvel.__version__,
        python=sys.version.split()[0],
        platform=platform.platform(terse=True),
    )
