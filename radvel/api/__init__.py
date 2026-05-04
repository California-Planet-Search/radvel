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
