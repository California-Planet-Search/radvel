"""Progress telemetry helper.

The MCMC worker calls a :class:`ProgressWriter` after every convergence
check; ``GET /jobs/{id}`` reads the JSON file written here. The pattern
keeps the SQLite database tiny (one row per job, no per-tick history)
while still letting clients poll live convergence stats.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class ProgressWriter:
    """Atomically write a JSON snapshot of MCMC convergence stats to disk.

    Writes are atomic: each call lands the bytes into a temp file in the
    same directory and ``os.replace``-s into place. Readers therefore see
    either the previous snapshot or the new one, never a partial file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, snapshot: dict) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=self.path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(snapshot, f)
            os.replace(tmp_path, self.path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def read(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
