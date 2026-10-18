"""SQLite-backed job registry, and a runner that gives each job its own process.

A *job* is one execution of a long-running pipeline step (``mcmc``,
``ns``, or a kind registered by an extension package) for a specific run. The state machine is:

::

    queued → running → succeeded | failed | cancelled

State is persisted in ``${RADVEL_API_DB_PATH:-/data/jobs.db}`` so the
container can restart without losing job records. Live convergence
telemetry is *not* in the database; it lives in
``runs/{run_id}/mcmc_progress.json`` written by
:class:`radvel.api.progress.ProgressWriter`.

The runner starts one process per job and runs at most
``RADVEL_API_WORKERS`` (default 1) at a time; the rest wait, first in
first out. Each MCMC job already spawns ``ensembles`` subprocesses
internally; total cores ≈ ``workers × ensembles``. Process isolation
side-steps the ``radvel.mcmc.statevars`` singleton.

A process per job, rather than a shared pool, is what makes a cancel
safe. Each job process leads its own process group, so a cancel that
has to SIGKILL reaches everything the job started and nothing else. In
a ``ProcessPoolExecutor`` that kill broke the pool, and every other job
in it failed. Job processes start with ``spawn``, because the API
process has threads and a forked copy of a threaded process can inherit
a lock nobody will release. They are not daemonic, so a job can start
its own ``multiprocessing`` pool, as RVsearch does.

A job process also watches its parent. If the API process dies, the job
marks its row failed and kills its process group, so it does not run on
as a ghost; :meth:`JobRegistry.reconcile_orphaned` fails the rows that
were still queued when the service next starts.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import multiprocessing
import os
import secrets
import signal
import sqlite3
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .config import Settings, get_settings


logger = logging.getLogger(__name__)

# How often the runner checks its job processes, and a job process its parent.
_POLL_S = 0.1
_PARENT_POLL_S = 0.5
# How often the runner reads a running job's row, for changes made by
# another process sharing the database.
_ROW_POLL_S = 1.0
# The most a service shutdown waits between SIGTERM and SIGKILL.
_SHUTDOWN_GRACE_S = 5.0

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

        Only a ``queued`` row can start. A job cancelled while its process
        was starting is already ``cancelled`` by the time the process gets
        here, and must not run.
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
        the process died before (or instead of) the runner starting it.
        It has no pid, so it is not ``running``, and the runner's
        in-memory queue only ever knows jobs submitted during its own
        process's life, so nothing will ever dispatch it. Left
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


def _signal(pid: int, signum: int) -> None:
    try:
        os.kill(pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _pgid(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _group_members(pgid: int) -> Optional[List[int]]:
    """Live processes in process group ``pgid``, or None if it could not look.

    Zombies are left out. None is not an empty group: a caller must not
    leave a group alone because it could not see into it.
    """
    members: List[int] = []
    proc = Path("/proc")
    if (proc / "self" / "stat").exists():
        try:
            entries = list(proc.iterdir())
        except OSError:  # out of file descriptors, say
            return None
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue  # exited while we looked
            # The command name may hold spaces or ')', so split after the
            # last ')': the fields there are state, ppid, then pgrp.
            fields = stat.rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid:
                members.append(int(entry.name))
        return members
    try:  # no /proc (macOS)
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "pgid=", "-o", "stat="],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        fields = line.split()
        if (len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit()
                and int(fields[1]) == pgid and not fields[2].startswith("Z")):
            members.append(int(fields[0]))
    return members


def _exit_status(process: Any) -> Tuple[bool, Optional[int]]:
    """Whether a job process has exited, and its exit code, without reaping it.

    ``waitid`` with ``WNOWAIT`` leaves an exited process a zombie, which
    keeps its process group id reserved while the runner clears the group.
    Not ``process.sentinel``: that pipe only reports an exit once every
    process holding it has closed it, and the job's forked children inherit
    it.
    """
    waitid = getattr(os, "waitid", None)
    if waitid is None:  # macOS before Python 3.13
        if process.is_alive():
            return False, None
        return True, process.exitcode
    try:
        info = waitid(os.P_PID, process.pid,
                      os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        # multiprocessing reaped it already: Process.start() reaps any
        # finished sibling.
        return True, process.exitcode
    if info is None:
        return False, None
    if info.si_code == os.CLD_EXITED:
        return True, info.si_status
    return True, -info.si_status


def _group_text(others: Optional[int]) -> str:
    if others is None:  # the group could not be listed
        return "the rest of its process group"
    return "{} other process(es) in its process group".format(others)


def _describe_exit(code: Optional[int]) -> str:
    if code is None:
        how = "exited"
    elif code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = "signal {}".format(-code)
        how = "was killed by {}".format(name)
        if -code == signal.SIGKILL:
            how += " (the out-of-memory killer sends SIGKILL)"
    else:
        how = "exited with code {}".format(code)
    return ("the job process {} before it recorded a result; see the "
            "service log".format(how))


# ---- runner --------------------------------------------------------------


# Type for worker callables — all workers receive (run_id, params_json).
Worker = Callable[[str, str], Dict[str, Any]]


@dataclass
class _Job:
    job_id: str
    args: Tuple[Any, ...]
    future: Future
    process: Any = None
    leads_group: bool = False      # seen leading its own process group
    cancel_requested: bool = False
    kill_at: Optional[float] = None  # time.monotonic() to SIGKILL at
    killed: bool = False
    row_checked_at: float = float("-inf")  # time.monotonic() of last read
    recorded_result: bool = False  # its row says it finished, yet it runs


class JobRunner:
    """Run each job in its own process, at most ``workers`` at a time.

    A single instance is created at FastAPI startup (see
    :func:`radvel.api.main.lifespan`). Jobs survive the runner going
    away because their state persists in SQLite.

    One monitor thread owns the job processes: it starts, signals and
    reaps all of them. Two threads waiting on one child race, and the
    loser never learns that it exited.
    """

    def __init__(self, registry: JobRegistry) -> None:
        self.registry = registry
        self._context = multiprocessing.get_context("spawn")
        self._futures: Dict[str, Future] = {}
        self._queue: Deque[_Job] = deque()
        self._running: Dict[str, _Job] = {}
        # Rows to fail that the database refused: job id -> error text.
        # Monitor thread only (or shutdown, once that thread has stopped).
        self._repairs: Dict[str, str] = {}
        self._repairs_tried_at = float("-inf")  # time.monotonic()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._closed = False
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="radvel-job-monitor", daemon=True)
        self._monitor.start()

    def shutdown(self, wait: bool = False) -> None:
        """Stop every job process, then return. Idempotent.

        Running jobs are marked ``failed`` and get SIGTERM, then SIGKILL
        for their whole process group if they are still running a few
        seconds later; otherwise the service could not exit while a fit
        ran. Jobs still waiting never start, and stay ``queued`` until
        the next startup fails them (see
        :meth:`JobRegistry.reconcile_orphaned`). ``wait`` is kept for
        compatibility: shutdown always waits for the job processes to go.
        """
        with self._wake:
            if self._closed:
                return
            self._closed = True
            waiting = list(self._queue)
            self._queue.clear()
            self._wake.notify_all()
        self._monitor.join()
        for job in waiting:
            job.future.cancel()
        running = list(self._running.values())
        for job in running:
            # Fail the row before the signal. The job turns SIGTERM into a
            # cancel of its own, and a client that sees "cancelled" will
            # not resubmit a fit the service dropped.
            try:
                self.registry.mark_finished(
                    job.job_id, state="failed",
                    error="the API service shut down while this job was "
                          "running; resubmit if still wanted")
            except Exception:
                logger.exception("could not fail job %s at shutdown",
                                 job.job_id)
            _signal(job.process.pid, signal.SIGTERM)
        deadline = time.monotonic() + min(
            self.registry.settings.cancel_grace_s, _SHUTDOWN_GRACE_S)
        while running:
            for job in list(running):
                try:
                    self._note_group(job)
                    exited, code = _exit_status(job.process)
                    if exited:
                        running.remove(job)
                        self._finish(job, code)
                    elif not job.killed and time.monotonic() >= deadline:
                        self._kill(job)
                except Exception:
                    logger.exception("stopping job %s at shutdown failed",
                                     job.job_id)
            if running:
                time.sleep(_POLL_S)

    def submit(self, run_id: str, kind: str, params: Dict[str, Any],
               worker: Worker) -> JobRow:
        """Insert a queued row, then queue the job for a process.

        Returns the freshly-created row immediately; the job process flips
        it to ``running`` (and later ``succeeded``/``failed``) as it runs.
        """
        row = self.registry.submit(run_id, kind, params)
        job = _Job(job_id=row.job_id, future=Future(),
                   args=(worker, row.job_id, run_id, json.dumps(params),
                         str(self.registry.db_path), os.getpid()))
        with self._wake:
            closed = self._closed
            if not closed:
                self._futures[row.job_id] = job.future
                self._queue.append(job)
                self._wake.notify_all()
        if closed:
            self.registry.mark_finished(
                row.job_id, state="failed",
                error="the API service was shutting down; resubmit if still wanted")
            raise RuntimeError("the job runner is shut down")
        return row

    def cancel(self, job_id: str) -> JobRow:
        """Cancel a job and make sure its process stops. Idempotent.

        The row turns ``cancelled`` at once, which is what clients poll
        for. A job still waiting never starts. A running job's process
        gets SIGTERM, and if it has not exited ``cancel_grace_s`` later,
        its whole process group gets SIGKILL: on 2026-10-17 an RVsearch
        worker swallowed its SIGTERM (it landed inside a multiprocessing
        finalizer) and ran for almost four more hours under a row that
        said ``cancelled``.

        A job running under another API process sharing the database is
        only marked here; the runner that started it sees the row and
        stops it.
        """
        current = self.registry.get(job_id)
        if current.state in {"succeeded", "failed", "cancelled"}:
            return current
        # Mark the row first. A job process that has not claimed it yet
        # will find it cancelled and not run the job.
        self.registry.mark_finished(job_id, state="cancelled",
                                    error="cancelled by client")
        row = self.registry.get(job_id)
        if row.state != "cancelled":
            return row  # finished on its own just before the cancel
        with self._wake:
            job = self._running.get(job_id)
            if job is not None:
                job.cancel_requested = True
                self._wake.notify_all()
            else:
                waiting = [j for j in self._queue if j.job_id == job_id]
                for queued in waiting:
                    self._queue.remove(queued)
                    queued.future.cancel()
                    self._futures.pop(job_id, None)
        return row

    # The methods below run on the monitor thread, or in shutdown() once
    # that thread has stopped.

    def _monitor_loop(self) -> None:
        while True:
            try:
                with self._wake:
                    if self._closed:
                        return
                    self._start_waiting_jobs()
                    running = list(self._running.values())
                for job in running:
                    try:
                        self._check(job)
                    except Exception:
                        # One job's failure must not stop the others
                        # being checked.
                        logger.exception("checking job %s failed", job.job_id)
                self._retry_repairs()
                with self._wake:
                    if self._closed:
                        return
                    if not self._has_work():
                        busy = self._running or self._repairs
                        self._wake.wait(_POLL_S if busy else None)
            except Exception:
                # A dead monitor thread would leave every job queued for
                # the life of the process.
                logger.exception("job monitor iteration failed")
                time.sleep(_POLL_S)

    def _has_work(self) -> bool:
        if self._queue and len(self._running) < self.registry.settings.workers:
            return True
        return any(j.cancel_requested and j.kill_at is None
                   for j in self._running.values())

    def _start_waiting_jobs(self) -> None:
        """Start queued jobs while there are free slots. Holds the lock."""
        while self._queue and len(self._running) < self.registry.settings.workers:
            job = self._queue.popleft()
            if not job.future.set_running_or_notify_cancel():
                continue  # cancelled while it waited
            # Not daemonic: a daemonic process may not have children, and
            # RVsearch starts a multiprocessing pool inside its job.
            process = self._context.Process(
                target=_worker_entrypoint, args=job.args,
                name="radvel-job-" + job.job_id, daemon=False)
            try:
                process.start()
            except Exception as exc:
                self._fail_row(job.job_id,
                               "could not start the job process: {!r}".format(exc))
                job.future.set_exception(exc)
                self._futures.pop(job.job_id, None)
                continue
            job.process = process
            self._running[job.job_id] = job

    def _note_group(self, job: _Job) -> None:
        if not job.leads_group:
            job.leads_group = _pgid(job.process.pid) == job.process.pid

    def _check(self, job: _Job) -> None:
        self._note_group(job)
        exited, code = _exit_status(job.process)
        if exited:
            self._finish(job, code)
            return
        if not job.cancel_requested and job.kill_at is None:
            self._check_row(job)
        if job.cancel_requested and job.kill_at is None:
            _signal(job.process.pid, signal.SIGTERM)
            job.kill_at = time.monotonic() + self.registry.settings.cancel_grace_s
        elif (job.kill_at is not None and not job.killed
              and time.monotonic() >= job.kill_at):
            others = self._kill(job)
            if job.recorded_result:
                logger.warning(
                    "job %s recorded its result, but its process (pid %d) "
                    "was still running %gs later; killed it along with %s",
                    job.job_id, job.process.pid,
                    self.registry.settings.cancel_grace_s, _group_text(others))
                return
            self.registry.amend_error(
                job.job_id, state="cancelled",
                error="cancelled by client; the job process (pid {}) was "
                      "still running {:g}s after SIGTERM, so it was killed "
                      "with SIGKILL along with {}".format(
                          job.process.pid,
                          self.registry.settings.cancel_grace_s,
                          _group_text(others)))

    def _check_row(self, job: _Job) -> None:
        """Act on a change another process made to a live job's row.

        Under ``radvel serve --workers N`` a cancel can reach an API
        process that did not start the job, and all it can do is mark the
        row; the runner that owns the process stops it here. A row with a
        result means the job is done, so its process gets
        ``cancel_grace_s`` to exit and is then killed. One stuck at exit,
        waiting on a thread or a pool, would otherwise hold its slot for
        good.
        """
        now = time.monotonic()
        if now - job.row_checked_at < _ROW_POLL_S:
            return
        job.row_checked_at = now
        try:
            state = self.registry.get(job.job_id).state
        except JobNotFound:
            return
        if state == "cancelled":
            job.cancel_requested = True
        elif state in ("succeeded", "failed"):
            job.recorded_result = True
            job.kill_at = now + self.registry.settings.cancel_grace_s

    def _kill(self, job: _Job) -> Optional[int]:
        """SIGKILL a job's process group.

        Returns how many other processes the group held, or None if they
        could not be listed. ``killed`` is set only once the signal is
        sent, so a kill that fails is tried again on the next pass.
        """
        pid = job.process.pid
        if not job.leads_group:  # it never got as far as leading a group
            _signal(pid, signal.SIGKILL)
            job.killed = True
            return 0
        members = _group_members(pid)
        _kill_group(pid)
        job.killed = True
        if members is None:
            return None
        return len([p for p in members if p != pid])

    def _finish(self, job: _Job, code: Optional[int]) -> None:
        """Clear up after an exited job process, then reap it.

        One step failing does not skip the next. The slot is freed and the
        future resolved whatever happens, and a row the database would not
        fail is retried (see :meth:`_fail_row`).
        """
        try:
            try:
                self._clear_group(job)
            except Exception:
                logger.exception("clearing up after job %s failed", job.job_id)
            job.process.join()
            if code is None:
                code = job.process.exitcode
            # A row still queued or running means the process ended without
            # recording a result: killed (OOM), os._exit, or a crash before
            # it got going. mark_finished leaves any other row alone.
            self._fail_row(job.job_id, _describe_exit(code))
        finally:
            with self._wake:
                self._running.pop(job.job_id, None)
                self._futures.pop(job.job_id, None)
                self._wake.notify_all()
            if not job.future.done():
                job.future.set_result(code)

    def _clear_group(self, job: _Job) -> None:
        """Kill what the job started that outlived it.

        Like the child left on 2026-10-17, stuck on a dead pool's lock.
        This does not need to know the job led its group: while any
        process is in group ``pid``, that id cannot be handed to a new
        process, and an empty group has nothing to kill. The leader may
        already be reaped here, since ``Process.start()`` reaps finished
        siblings, so ``leads_group`` can be wrong.
        """
        pid = job.process.pid
        leftover = _group_members(pid)
        if leftover is None:
            # Could not look, so kill it anyway: killing an empty group
            # only fails, and _kill_group ignores that.
            _kill_group(pid)
            logger.warning("could not list what job %s left running; killed "
                           "its process group anyway", job.job_id)
        elif leftover:
            _kill_group(pid)
            logger.warning("job %s left %d process(es) running; killed them",
                           job.job_id, len(leftover))

    def _fail_row(self, job_id: str, error: str) -> None:
        """Fail a job's row if it is still queued or running.

        If the database refuses (locked past its timeout, say), the row is
        kept and retried by :meth:`_retry_repairs`. Left ``running``, it
        would make every resubmit of its run fail as already active. A row
        still waiting when the service stops is failed by the next startup
        (:meth:`JobRegistry.reconcile_orphaned`).
        """
        try:
            self.registry.mark_finished(job_id, state="failed", error=error)
        except Exception:
            logger.exception("could not fail job %s; will retry", job_id)
            self._repairs[job_id] = error

    def _retry_repairs(self) -> None:
        """Retry the rows :meth:`_fail_row` could not fail, once a second."""
        now = time.monotonic()
        if not self._repairs or now - self._repairs_tried_at < _ROW_POLL_S:
            return
        self._repairs_tried_at = now
        for job_id, error in list(self._repairs.items()):
            try:
                self.registry.mark_finished(job_id, state="failed", error=error)
            except Exception:
                continue  # logged when it first failed
            del self._repairs[job_id]
            logger.info("failed job %s after an earlier database error", job_id)


def _watch_parent(parent_pid: int, registry: JobRegistry, job_id: str) -> None:
    """Stop this job if the API process that started it goes away.

    Otherwise the job runs on with nothing tracking it, under a row that
    says ``running``. It compares against the parent's pid, not against
    1: under a container init or a subreaper an orphan's new parent is
    not pid 1. Not ``PR_SET_PDEATHSIG``: that fires when the parent's
    starting *thread* exits, and the job's own children do not inherit it.
    """
    def watch() -> None:
        while os.getppid() == parent_pid:
            time.sleep(_PARENT_POLL_S)
        try:
            registry.mark_finished(
                job_id, state="failed",
                error="the API process that ran this job exited; "
                      "resubmit if still wanted")
        finally:
            if _pgid(0) == os.getpid():
                os.killpg(0, signal.SIGKILL)  # this process and its children
            os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=watch, name="radvel-parent-watch",
                     daemon=True).start()


def _worker_entrypoint(worker: Worker, job_id: str, run_id: str,
                       params_json: str, db_path: str, parent_pid: int) -> None:
    """Job process entry. Updates the SQLite row and runs the worker."""
    # Lead a process group before anything starts children, so one signal
    # to the group reaches everything this job starts.
    try:
        os.setpgid(0, 0)
    except OSError:
        pass  # still stoppable on its own; the runner checks before killpg
    settings = get_settings()
    settings = settings.model_copy(update={"db_path": Path(db_path)})
    # A spawned process starts with logging unconfigured.
    logging.basicConfig(level=settings.log_level)
    sub_registry = JobRegistry(settings=settings)
    _watch_parent(parent_pid, sub_registry, job_id)
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if not sub_registry.mark_running(job_id, pid=os.getpid(), host=host):
        return  # cancelled (or failed) before the job started
    try:
        worker(run_id, params_json)
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
