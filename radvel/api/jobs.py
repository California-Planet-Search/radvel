"""SQLite-backed job registry + ProcessPoolExecutor for long-running ops.

A *job* is one execution of a long-running pipeline step (``mcmc`` or
``ns``) for a specific run. The state machine is:

::

    queued → running → succeeded | failed | cancelled

State is persisted in ``${RADVEL_API_DB_PATH:-/data/jobs.db}`` so the
container can restart without losing job records. Live convergence
telemetry is *not* in the database; it lives in
``runs/{run_id}/mcmc_progress.json`` written by
:class:`radvel.api.progress.ProgressWriter`.

The runner uses a ``concurrent.futures.ProcessPoolExecutor`` (size from
``RADVEL_API_WORKERS``, default 1). Each MCMC job already spawns
``ensembles`` subprocesses internally; total cores ≈ ``workers ×
ensembles``. Process isolation gives clean cancellation via
``terminate()`` and side-steps the ``radvel.mcmc.statevars`` singleton.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import signal
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import Settings, get_settings


_SLUG_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    state         TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    submitted_at  TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT,
    pid           INTEGER,
    host          TEXT
);
CREATE INDEX IF NOT EXISTS jobs_run_id_idx ON jobs(run_id);
CREATE INDEX IF NOT EXISTS jobs_state_idx  ON jobs(state);
"""


def make_job_id() -> str:
    suffix = "".join(secrets.choice(_SLUG_ALPHABET) for _ in range(10))
    return "j-" + suffix


@dataclass
class JobRow:
    job_id: str
    run_id: str
    kind: str
    state: str
    params: Dict[str, Any]
    submitted_at: _dt.datetime
    started_at: Optional[_dt.datetime] = None
    finished_at: Optional[_dt.datetime] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    host: Optional[str] = None


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(value: Optional[_dt.datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _from_iso(value: Optional[str]) -> Optional[_dt.datetime]:
    return _dt.datetime.fromisoformat(value) if value else None


def _row_to_job(row: sqlite3.Row) -> JobRow:
    return JobRow(
        job_id=row["job_id"],
        run_id=row["run_id"],
        kind=row["kind"],
        state=row["state"],
        params=json.loads(row["params_json"]),
        submitted_at=_from_iso(row["submitted_at"]),
        started_at=_from_iso(row["started_at"]),
        finished_at=_from_iso(row["finished_at"]),
        error=row["error"],
        pid=row["pid"],
        host=row["host"],
    )


class JobNotFound(LookupError):
    """Raised when a caller references an unknown ``job_id``."""


class JobRegistry:
    """Thread-safe access to the SQLite jobs table.

    The registry is intentionally synchronous and process-local.
    Long-running work happens in a separate process owned by
    :class:`JobRunner`; this object only handles the metadata.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.db_path = self.settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def submit(self, run_id: str, kind: str, params: Dict[str, Any]) -> JobRow:
        if kind not in {"mcmc", "ns"}:
            raise ValueError("kind must be 'mcmc' or 'ns'; got {!r}".format(kind))
        if self.active_job_for_run(run_id) is not None:
            raise JobActiveError("a job is already active for run {!r}".format(run_id))
        job_id = make_job_id()
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs(job_id, run_id, kind, state, params_json, submitted_at) "
                "VALUES (?, ?, ?, 'queued', ?, ?)",
                (job_id, run_id, kind, json.dumps(params), _iso(now)),
            )
        return JobRow(
            job_id=job_id,
            run_id=run_id,
            kind=kind,
            state="queued",
            params=params,
            submitted_at=now,
        )

    def get(self, job_id: str) -> JobRow:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return _row_to_job(row)

    def list(self, *, run_id: Optional[str] = None,
             state: Optional[str] = None) -> List[JobRow]:
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state)
        sql = "SELECT * FROM jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY submitted_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_job(r) for r in rows]

    def active_job_for_run(self, run_id: str) -> Optional[JobRow]:
        for state in ("running", "queued"):
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE run_id = ? AND state = ? "
                    "ORDER BY submitted_at DESC LIMIT 1",
                    (run_id, state),
                ).fetchone()
            if row is not None:
                return _row_to_job(row)
        return None

    def mark_running(self, job_id: str, pid: int, host: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET state='running', pid=?, host=?, started_at=? "
                "WHERE job_id=?",
                (pid, host, _iso(_now()), job_id),
            )

    def mark_finished(self, job_id: str, *, state: str,
                      error: Optional[str] = None) -> None:
        if state not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("invalid terminal state: {}".format(state))
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET state=?, finished_at=?, error=? "
                "WHERE job_id=?",
                (state, _iso(_now()), error, job_id),
            )

    def reconcile_orphaned(self) -> int:
        """At startup, mark `running` rows whose pid is gone as `failed`.

        Survives container restarts: a job whose worker process died while
        the host was offline is repaired with a clear error so clients
        polling ``GET /jobs/{id}`` see the correct terminal state instead
        of forever-running.
        """
        repaired = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, pid, host FROM jobs WHERE state='running'"
            ).fetchall()
        host = os.uname().nodename if hasattr(os, "uname") else ""
        for row in rows:
            pid = row["pid"]
            same_host = (row["host"] or "") == host
            alive = same_host and pid and _pid_alive(pid)
            if not alive:
                self.mark_finished(
                    row["job_id"], state="failed",
                    error="process gone (likely container restart)"
                )
                repaired += 1
        return repaired


class JobActiveError(RuntimeError):
    """Raised when a run already has an active job (409)."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


# ---- runner --------------------------------------------------------------


# Type for worker callables — all workers receive (run_id, params_json).
Worker = Callable[[str, str], Dict[str, Any]]


class JobRunner:
    """Submit jobs into a ProcessPoolExecutor and update the registry.

    A single instance is created at FastAPI startup (see
    :func:`radvel.api.main.lifespan`). Jobs survive the executor going
    away because their state persists in SQLite.
    """

    def __init__(self, registry: JobRegistry) -> None:
        self.registry = registry
        self.pool = ProcessPoolExecutor(max_workers=registry.settings.workers)
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def shutdown(self, wait: bool = False) -> None:
        """SIGTERM running workers, then close the pool.

        Without the SIGTERM step, ``ProcessPoolExecutor.shutdown(wait=False)``
        cannot interrupt a worker that is mid-MCMC — the test (or container)
        process then hangs at exit waiting on the non-daemon child.
        """
        host = os.uname().nodename if hasattr(os, "uname") else ""
        for row in self.registry.list(state="running"):
            if row.pid and (row.host or "") == host:
                try:
                    os.kill(row.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        # Belt-and-braces: terminate any pool subprocess still alive so the
        # parent (test runner / uvicorn) can exit promptly. Capture process
        # handles BEFORE shutdown — afterwards ``_processes`` is set to None.
        procs = list((getattr(self.pool, "_processes", None) or {}).values())
        self.pool.shutdown(wait=wait, cancel_futures=True)
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
            except Exception:
                pass

    def submit(self, run_id: str, kind: str, params: Dict[str, Any],
               worker: Worker) -> JobRow:
        """Insert a queued row, then enqueue the worker.

        Returns the freshly-created row immediately; the executor will
        flip the state to `running` (and later `succeeded`/`failed`) as
        the worker runs.
        """
        row = self.registry.submit(run_id, kind, params)
        future = self.pool.submit(_worker_entrypoint, worker, row.job_id,
                                  run_id, json.dumps(params),
                                  str(self.registry.db_path))
        future.add_done_callback(lambda f, jid=row.job_id: self._on_done(jid, f))
        with self._lock:
            self._futures[row.job_id] = future
        return row

    def cancel(self, job_id: str) -> JobRow:
        """Issue SIGTERM to the running worker. Idempotent."""
        try:
            current = self.registry.get(job_id)
        except JobNotFound:
            raise
        if current.state in {"succeeded", "failed", "cancelled"}:
            return current
        with self._lock:
            future = self._futures.get(job_id)
        if current.pid:
            try:
                os.kill(current.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if future is not None:
            future.cancel()
        self.registry.mark_finished(job_id, state="cancelled",
                                    error="cancelled by client")
        return self.registry.get(job_id)

    def _on_done(self, job_id: str, future: Future) -> None:
        try:
            future.result()
        except Exception as exc:
            # Worker exits with state already persisted; only log here.
            try:
                row = self.registry.get(job_id)
                if row.state == "running":
                    self.registry.mark_finished(
                        job_id, state="failed", error=repr(exc)
                    )
            except JobNotFound:
                pass


def _worker_entrypoint(worker: Worker, job_id: str, run_id: str,
                       params_json: str, db_path: str) -> Dict[str, Any]:
    """Subprocess entry. Updates the SQLite row and runs the worker."""
    # Reopen the registry from inside the subprocess; sqlite handles are
    # not safely shareable across forked/spawned children.
    settings = get_settings()
    settings = settings.model_copy(update={"db_path": Path(db_path)})
    sub_registry = JobRegistry(settings=settings)
    host = os.uname().nodename if hasattr(os, "uname") else ""
    sub_registry.mark_running(job_id, pid=os.getpid(), host=host)
    try:
        result = worker(run_id, params_json)
    except SystemExit:
        sub_registry.mark_finished(job_id, state="cancelled",
                                   error="SIGTERM received")
        raise
    except Exception as exc:
        import traceback as _tb
        sub_registry.mark_finished(
            job_id, state="failed",
            error=("{}: {}\n{}".format(type(exc).__name__, exc,
                                       _tb.format_exc()))[:8000]
        )
        raise
    sub_registry.mark_finished(job_id, state="succeeded")
    return result
