"""HTTP API package for RadVel.

This package wraps the existing :mod:`radvel.driver` workflow behind a
FastAPI service. Importing the full package requires the optional
``api`` extra (``pip install radvel[api]``) which pulls in FastAPI,
uvicorn, pydantic-settings and friends. The :mod:`radvel.api.schemas`
submodule only needs pydantic.

The API is introduced in RadVel 1.6 (see ``docs/api_service.rst``) and is
intentionally kept off the default import path so library users who never
touch HTTP do not pay for the optional dependency.
"""

import os as _os

# The API is a headless service — pin matplotlib to a non-interactive
# backend before any plotting code runs a draw. The macOS native backend
# (default in interactive Python on darwin) crashes when invoked from a
# request thread because it tries to talk to a window server. ``radvel``
# imports ``radvel.plot`` (and thereby ``matplotlib``) at top-level, so by
# the time we get here matplotlib is usually already imported with the
# wrong backend; ``matplotlib.use(..., force=True)`` switches it
# regardless, as long as no figure exists yet.
_os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib as _mpl
    _mpl.use("Agg", force=True)
except Exception:  # pragma: no cover — matplotlib import errors handled at draw time
    pass
