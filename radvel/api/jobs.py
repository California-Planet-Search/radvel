"""SQLite-backed job registry + ProcessPoolExecutor for long-running ops.

A *job* is one execution of a long-running pipeline step (``mcmc``,
``ns``, or a kind registered by an extension package) for a specific run. The state machine is:

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
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor, Future
from concurrent.futures import wait as wait_futures
from concurrent.futures.process import BrokenProcessPool
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


# Job kinds the registry accepts. Extension packages (e.g. rvsearch) add
# their own via :func:`register_job_kind` before submitting.
JOB_KINDS = {"mcmc", "ns"}


def register_job_kind(kind: str) -> None:
    """Allow ``kind`` in :meth:`JobRegistry.submit`. Idempotent."""
    if not kind or not kind.replace("_", "").isalnum():
        raise ValueError("job kind must be alphanumeric/underscore; got {!r}".format(kind))
    JOB_KINDS.add(kind)


def progress_filename(kind: str) -> str:
    """Name of the per-run progress snapshot that ``GET /jobs/{id}`` reads.

    ``mcmc`` and ``ns`` share ``mcmc_progress.json`` (historical name);
    registered kinds get ``<kind>_progress.json``.
    """
    return "mcmc_progress.json" if kind in ("mcmc", "ns") else "{}_progress.json".format(kind)


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
        if kind not in JOB_KINDS:
            raise ValueError("unknown job kind {!r}; known: {}".format(
                kind, sorted(JOB_KINDS)))
        job_id = make_job_id()
        now = _now()
        # The active-job check and the INSERT live in the same locked
        # block so two concurrent /mcmc submissions can't both pass the
        # pre-check and enqueue duplicate queued rows for one run.
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT job_id FROM jobs WHERE run_id = ? "
                "AND state IN ('queued', 'running') LIMIT 1",
                (run_id,),
            ).fetchone()
            if existing is not None:
                raise JobActiveError(
                    "a job is already active for run {!r}".format(run_id)
                )
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

    def mark_running(self, job_id: str, pid: int, host: str) -> bool:
        """Claim a queued row for a worker. False means do not run it.

        Only a ``queued`` row can start. A job cancelled while it waited
        inside the executor's call queue is already ``cancelled`` by the
        time a worker reaches it, and ``Future.cancel()`` could not stop it
        there, because the executor marks those futures running.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET state='running', pid=?, host=?, started_at=? "
                "WHERE job_id=? AND state='queued'",
                (pid, host, _iso(_now()), job_id),
            )
            return cur.rowcount == 1

    def amend_error(self, job_id: str, *, state: str, error: str) -> None:
        """Replace the error text of a row that is still in ``state``."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET error=? WHERE job_id=? AND state=?",
                (error, job_id, state),
            )

    def mark_finished(self, job_id: str, *, state: str,
                      error: Optional[str] = None) -> None:
        if state not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("invalid terminal state: {}".format(state))
        # Guard the terminal transition: only move from a non-terminal
        # state. Without this, a worker that completes its happy-path
        # right as the parent issues `cancel()` can overwrite the
        # cancelled row with `succeeded`.
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET state=?, finished_at=?, error=? "
                "WHERE job_id=? AND state IN ('queued', 'running')",
                (state, _iso(_now()), error, job_id),
            )

    def reconcile_orphaned(self) -> int:
        """At startup, fail rows a previous process left mid-flight.

        Two states strand, not one, and the second is the one that hurt:

        ``running`` — the worker process died while the host was offline.
        Repaired with a clear error so clients polling ``GET /jobs/{id}``
        see a terminal state instead of forever-running.

        ``queued`` — the row was inserted by :meth:`JobRunner.submit` but
        the process died before (or instead of) the executor picking it
        up. It has no pid, so it is not ``running``, and the in-memory
        ``ProcessPoolExecutor`` only ever knows jobs submitted during its
        own process's life — so nothing will ever dispatch it. Left
        unhandled these sit at ``queued`` forever while every client
        blocks on a long read and reports a timeout, which is
        indistinguishable from a slow fit.

        That is not hypothetical: three MCMC jobs sat queued from
        2026-09-10 to 2026-09-16 behind exactly this gap, and a container
        restart did not rescue them because this method only looked at
        ``running``.

        Failing them (rather than re-dispatching) is deliberate: an
        unattended restart that silently starts N queued MCMC fits is a
        thundering herd on a shared box, and the caller has the context
        to decide whether the fit is still wanted.
        """
        repaired = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, pid, host FROM jobs WHERE state='running'"
            ).fetchall()
            stranded = conn.execute(
                "SELECT job_id FROM jobs WHERE state='queued'"
            ).fetchall()
        for row in stranded:
            self.mark_finished(
                row["job_id"], state="failed",
                error="queued but never started (the service restarted "
                      "before it was dispatched); resubmit if still wanted",
            )
            repaired += 1
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


def _sigkill(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _parent_pids() -> Dict[int, int]:
    """Map pid to parent pid for every visible process."""
    table: Dict[int, int] = {}
    proc = Path("/proc")
    if (proc / "self" / "stat").exists():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue  # exited while we looked
            # The command name may hold spaces or ')', so split after the
            # last ')': the fields there are state, then ppid.
            table[int(entry.name)] = int(stat.rsplit(")", 1)[1].split()[1])
        return table
    try:  # no /proc (macOS)
        out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return table
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            table[int(fields[0])] = int(fields[1])
    return table


def _process_tree(root: int) -> List[int]:
    """``root`` followed by all its descendants.

    Taken before anything is killed: once ``root`` dies its children are
    reparented and can no longer be traced back to it.
    """
    children: Dict[int, List[int]] = {}
    for pid, ppid in _parent_pids().items():
        children.setdefault(ppid, []).append(pid)
    tree, frontier = [root], [root]
    while frontier:
        frontier = [c for p in frontier for c in children.get(p, ())
                    if c not in tree]
        tree.extend(frontier)
    return tree


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
        self._closed = False

    def shutdown(self, wait: bool = False) -> None:
        """SIGTERM running workers, then close the pool.

        Without the SIGTERM step, ``ProcessPoolExecutor.shutdown(wait=False)``
        cannot interrupt a worker that is mid-MCMC — the test (or container)
        process then hangs at exit waiting on the non-daemon child.
        """
        self._closed = True
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
        args = (_worker_entrypoint, worker, row.job_id, run_id,
                json.dumps(params), str(self.registry.db_path))
        try:
            future = self.pool.submit(*args)
        except BrokenProcessPool:
            # A ProcessPoolExecutor never recovers once a worker has died
            # (OOM kill, a cancel's SIGTERM): every later submit raises
            # this, for the life of the process. Replace it and carry on.
            self._replace_pool()
            future = self.pool.submit(*args)
        future.add_done_callback(lambda f, jid=row.job_id: self._on_done(jid, f))
        with self._lock:
            self._futures[row.job_id] = future
        return row

    def _replace_pool(self) -> None:
        with self._lock:
            broken = self.pool
            self.pool = ProcessPoolExecutor(
                max_workers=self.registry.settings.workers
            )
        broken.shutdown(wait=False, cancel_futures=True)

    def cancel(self, job_id: str) -> JobRow:
        """Cancel a job and make sure its worker stops. Idempotent.

        The row turns ``cancelled`` at once, which is what clients poll
        for. A running worker gets SIGTERM. If it is still on this job
        ``cancel_grace_s`` later, it and every process under it are
        SIGKILLed: on 2026-10-17 an RVsearch worker swallowed its SIGTERM
        (it landed inside a multiprocessing finalizer) and ran for almost
        four more hours under a row that said ``cancelled``.
        """
        current = self.registry.get(job_id)
        if current.state in {"succeeded", "failed", "cancelled"}:
            return current
        # Mark the row before reading the pid. A worker that has not
        # claimed the row yet will now find it cancelled and skip the job;
        # one that already claimed it wrote its pid, which the re-read
        # below sees. Either way no worker runs it unsignalled.
        self.registry.mark_finished(job_id, state="cancelled",
                                    error="cancelled by client")
        row = self.registry.get(job_id)
        if row.state != "cancelled":
            return row  # finished on its own just before the cancel
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.cancel()  # only helps while it is still in the pool's backlog
        # A done future means the pool worker has moved on, possibly to
        # another job, so its pid is no longer this job's to signal.
        if row.pid and (future is None or not future.done()):
            try:
                os.kill(row.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                if future is not None:
                    threading.Thread(
                        target=self._kill_if_still_running,
                        args=(job_id, row.pid, future),
                        name="radvel-cancel-{}".format(job_id),
                        daemon=True,
                    ).start()
        return row

    def _kill_if_still_running(self, job_id: str, pid: int,
                               future: Future) -> None:
        """After the grace period, SIGKILL a worker that ignored SIGTERM.

        The future is the test of "still on this job", not the pid: pool
        workers outlive their jobs, so a live pid proves nothing, but the
        future stays pending until the worker hands this job back.

        Killing a worker breaks its pool. A fresh pool goes in first, so
        new submits never land in the broken one. Any other job still in
        the old pool (running, with ``workers > 1``, or queued behind the
        stuck one) fails with a "pool broke" error, as on any worker death.
        """
        grace = self.registry.settings.cancel_grace_s
        wait_futures([future], timeout=grace)
        if future.done() or self._closed:
            return
        tree = _process_tree(pid)
        with self._lock:
            old: Optional[ProcessPoolExecutor] = self.pool
            if pid in (getattr(old, "_processes", None) or {}):
                self.pool = ProcessPoolExecutor(
                    max_workers=self.registry.settings.workers
                )
            else:
                old = None  # that pool was already replaced
        killed = [p for p in tree if _sigkill(p)]
        if old is not None:
            old.shutdown(wait=False, cancel_futures=True)
        self.registry.amend_error(
            job_id, state="cancelled",
            error="cancelled by client; the worker (pid {}) was still on the "
                  "job {:g}s after SIGTERM, so it was killed with SIGKILL "
                  "along with {} child process(es)".format(
                      pid, grace, max(len(killed) - 1, 0)),
        )

    def _on_done(self, job_id: str, future: Future) -> None:
        # BaseException, not Exception: a cancelled worker sends back the
        # SystemExit its SIGTERM handler raised. This callback runs on the
        # executor's manager thread, and Future._invoke_callbacks only
        # catches Exception, so a SystemExit escaping here ends that
        # thread. Nothing dispatches after that: every later job stays
        # queued for the life of the process.
        try:
            future.result()
        except BaseException as exc:
            # Worker exits with state already persisted; only log here.
            try:
                row = self.registry.get(job_id)
                if row.state == "running":
                    self.registry.mark_finished(
                        job_id, state="failed", error=repr(exc)
                    )
                elif row.state == "queued":
                    # Still waiting in the pool when the pool broke: no
                    # worker will ever take it.
                    self.registry.mark_finished(
                        job_id, state="failed",
                        error="queued but never started (the worker pool "
                              "broke before it was dispatched); resubmit "
                              "if still wanted",
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
    if not sub_registry.mark_running(job_id, pid=os.getpid(), host=host):
        return {}  # cancelled (or failed) before a worker reached it
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
        # Not a bare `raise`: the parent must unpickle whatever leaves this
        # process, and an exception it cannot rebuild breaks the whole pool
        # for every later job. The row above already holds the real error.
        raise RuntimeError(
            "{}: {}".format(type(exc).__name__, exc)
        ) from None
    sub_registry.mark_finished(job_id, state="succeeded")
    return result
