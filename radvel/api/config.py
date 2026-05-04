"""Settings for the RadVel HTTP API.

All knobs are read from environment variables prefixed ``RADVEL_API_``
(plus a couple of legacy ``RADVEL_`` names that already exist in the
codebase). Defaults are set so that ``docker run -p 8000:8000 ...
ghcr.io/.../radvel-api:1.6`` works without any explicit configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for :mod:`radvel.api.main`."""

    model_config = SettingsConfigDict(
        env_prefix="RADVEL_API_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    runs_dir: Path = Field(default=Path("/data/runs"))
    db_path: Path = Field(default=Path("/data/jobs.db"))
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    request_timeout_s: int = 1800
    max_upload_bytes: int = 50_000_000
    allow_py_upload: bool = False
    enable_ui: bool = True
    data_allowlist: List[Path] = Field(default_factory=list)
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-global settings instance.

    Cached so all routers and adapters share a single, immutable view of
    configuration. Tests that need to override settings should call
    ``get_settings.cache_clear()`` after mutating env vars.
    """
    return Settings()
