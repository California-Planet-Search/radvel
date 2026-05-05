"""Adapters wrapping :mod:`radvel.driver` for the HTTP API.

Two flavours:

1. **Sync** — used directly by the FastAPI request handler, runs in the
   web process, returns a typed result. Covers ``fit``, ``derive``,
   ``ic_compare``, ``tables``.
2. **Async workers** — invoked by :class:`radvel.api.jobs.JobRunner` in
   a child process. Cover ``mcmc`` and ``ns``. They wire a
   :class:`radvel.api.progress.ProgressWriter` into the driver call so
   ``GET /jobs/{id}`` can stream live convergence stats.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import signal
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import radvel
from radvel import driver
from radvel.driver import load_status

from .progress import ProgressWriter
from .runs import RunNotFound, RunRecord, RunRegistry


@dataclass
class AdapterError(Exception):
    """Wrapped error that the API layer turns into a structured 4xx/5xx."""

    error_type: str
    message: str
    status_code: int = 500
    traceback_id: Optional[str] = None


def _build_args(record: RunRecord, **extra: Any) -> argparse.Namespace:
    """Build an argparse namespace mirroring the CLI subcommand."""
    base = dict(
        setupfn=str(record.setup_py),
        outputdir=str(record.outputdir),
        decorr=False,
    )
    base.update(extra)
    return argparse.Namespace(**base)


@contextlib.contextmanager
def _capture(record: RunRecord):
    """Capture stdout/stderr and translate driver exceptions to AdapterError."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    cwd = os.getcwd()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            yield buf_out, buf_err
    except AssertionError as exc:
        # The driver uses assertions for "step prerequisite not met"
        # (e.g. "Must perform max-likelihood fit before plotting").
        raise AdapterError(
            error_type="step_prerequisite_unmet",
            message=str(exc) or "step prerequisite assertion failed",
            status_code=409,
        ) from exc
    except Exception as exc:
        traceback_id = uuid.uuid4().hex[:12]
        # Persist the full traceback so operators can correlate from logs.
        log_path = record.outputdir / "errors.log"
        with open(log_path, "a") as f:
            f.write("=== {} ===\n".format(traceback_id))
            f.write("stdout:\n{}\n".format(buf_out.getvalue()))
            f.write("stderr:\n{}\n".format(buf_err.getvalue()))
            traceback.print_exc(file=f)
            f.write("\n")
        raise AdapterError(
            error_type=type(exc).__name__,
            message=str(exc),
            status_code=500,
            traceback_id=traceback_id,
        ) from exc
    finally:
        os.chdir(cwd)


# ---- fit -----------------------------------------------------------------


def run_fit(record: RunRecord, *, decorr: bool = False) -> Dict[str, Any]:
    args = _build_args(record, decorr=decorr)
    with _capture(record):
        driver.fit(args)
    postfile = record.outputdir / "{}_post_obj.pkl".format(record.run_id)
    post = radvel.posterior.load(str(postfile))

    params = {k: float(post.params[k].value) for k in post.params.keys()}
    residuals = post.likelihood.residuals()
    rms = float(np.sqrt(np.mean(np.asarray(residuals) ** 2)))
    return {
        "logprob": float(post.logprob()),
        "rms": rms,
        "params": params,
        "postfile": postfile.name,
    }


# ---- derive --------------------------------------------------------------


def run_derive(record: RunRecord, *, sampler: str = "auto") -> Dict[str, Any]:
    args = _build_args(record, sampler=sampler)
    with _capture(record):
        driver.derive(args)
    status = load_status(str(record.stat_file))
    if not status.has_section("derive") or not status.getboolean("derive", "run"):
        raise AdapterError(
            error_type="derive_skipped",
            message="derive completed without producing output (likely missing stellar mass in setup)",
            status_code=409,
        )
    quantfile = status.get("derive", "quantfile")
    chainfile = status.get("derive", "chainfile")
    chains = pd.read_csv(chainfile)
    return {
        "columns": list(chains.columns),
        "quantfile": os.path.basename(quantfile),
    }


# ---- ic_compare ----------------------------------------------------------


def run_ic_compare(record: RunRecord, *, types: List[str], mixed: bool = True,
                   fixjitter: bool = False, simple: bool = False,
                   verbose: bool = False) -> Dict[str, Any]:
    args = _build_args(
        record,
        type=list(types),
        mixed=bool(mixed),
        unmixed=not mixed,
        fixjitter=fixjitter,
        simple=simple,
        verbose=verbose,
    )
    with _capture(record):
        driver.ic_compare(args)
    status = load_status(str(record.stat_file))
    raw = status.get("ic_compare", "ic")
    # The driver stores the ic_compare result as repr(OrderedDict([...])); eval
    # needs both OrderedDict and (under NumPy 2.x) `np` in scope because newer
    # numpy reprs scalars as `np.float64(...)`. The string is produced by
    # driver.ic_compare itself, never user input.
    from collections import OrderedDict
    statsdicts = eval(  # noqa: S307
        raw,
        {"OrderedDict": OrderedDict, "np": np, "numpy": np},
    )
    return {"statsdicts": list(statsdicts)}


# ---- tables --------------------------------------------------------------


def run_tables(record: RunRecord, *, types: List[str], header: bool = False,
               name_in_title: bool = False, sampler: str = "auto") -> Dict[str, Any]:
    args = _build_args(
        record,
        type=list(types),
        header=header,
        name_in_title=name_in_title,
        sampler=sampler,
    )
    with _capture(record):
        driver.tables(args)
    latex: Dict[str, str] = {}
    files: Dict[str, str] = {}
    for tabtype in types:
        path = record.outputdir / "{}_{}.tex".format(record.run_id, tabtype)
        if path.is_file():
            latex[tabtype] = path.read_text()
            files[tabtype] = path.name
    return {"latex": latex, "files": files}


# ---- async workers (run in a child process via JobRunner) ----------------


def _resolve_record(run_id: str) -> RunRecord:
    """Re-acquire a RunRecord inside the worker process."""
    registry = RunRegistry()
    return registry.get(run_id)


def _install_sigterm_handler() -> None:
    """Translate SIGTERM into SystemExit so JobRunner.cancel can interrupt."""
    def _handler(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _handler)


def _run_mcmc_worker(run_id: str, params_json: str) -> Dict[str, Any]:
    """ProcessPool worker for ``POST /runs/{id}/mcmc``.

    Receives only JSON-serialisable arguments because it runs in a fresh
    child process. Writes outputs identically to ``radvel.driver.mcmc``
    so subsequent ``derive``/``tables``/``report`` calls keep working.
    """
    _install_sigterm_handler()
    record = _resolve_record(run_id)
    params = json.loads(params_json)

    progress_path = record.outputdir / "mcmc_progress.json"
    writer = ProgressWriter(progress_path)
    writer.write({"state": "starting"})

    args = _build_args(
        record,
        nsteps=int(params.get("nsteps", 10000)),
        nwalkers=int(params.get("nwalkers", 50)),
        ensembles=int(params.get("ensembles", 8)),
        minAfactor=float(params.get("minAfactor", 40.0)),
        maxArchange=float(params.get("maxArchange", 0.03)),
        burnAfactor=float(params.get("burnAfactor", 25.0)),
        burnGR=float(params.get("burnGR", 1.03)),
        maxGR=float(params.get("maxGR", 1.01)),
        minTz=int(params.get("minTz", 1000)),
        minsteps=int(params.get("minsteps", 0)),
        minpercent=float(params.get("minpercent", 5.0)),
        thin=int(params.get("thin", 1)),
        serial=bool(params.get("serial", False)),
        save=bool(params.get("save", False)),
        proceed=bool(params.get("proceed", False)),
        headless=True,  # never animate from inside a worker process
        progress_callback=writer.write,
    )
    with _capture(record):
        driver.mcmc(args)

    status = load_status(str(record.stat_file))
    summary = {
        "logprob": None,
        "chainfile": None,
        "postfile": None,
    }
    if status.has_section("mcmc"):
        summary["chainfile"] = os.path.basename(status.get("mcmc", "chainfile"))
        summary["postfile"] = os.path.basename(status.get("mcmc", "postfile"))
        try:
            post = radvel.posterior.load(status.get("mcmc", "postfile"))
            summary["logprob"] = float(post.logprob())
        except Exception:
            pass
    return summary


def _run_ns_worker(run_id: str, params_json: str) -> Dict[str, Any]:
    """ProcessPool worker for ``POST /runs/{id}/ns``."""
    _install_sigterm_handler()
    record = _resolve_record(run_id)
    params = json.loads(params_json)

    args = _build_args(
        record,
        sampler=params.get("sampler", "ultranest"),
        sampler_kwargs=_kwargs_to_str(params.get("sampler_kwargs") or {}),
        run_kwargs=_kwargs_to_str(params.get("run_kwargs") or {}),
        proceed=bool(params.get("proceed", False)),
        overwrite=bool(params.get("overwrite", False)),
    )
    with _capture(record):
        driver.nested_sampling(args)

    status = load_status(str(record.stat_file))
    summary: Dict[str, Any] = {
        "chainfile": None,
        "postfile": None,
    }
    if status.has_section("ns"):
        summary["chainfile"] = os.path.basename(status.get("ns", "chainfile"))
        summary["postfile"] = os.path.basename(status.get("ns", "postfile"))
    return summary


def _kwargs_to_str(kwargs: Dict[str, Any]) -> Optional[str]:
    """Convert a dict into the space-separated `key=value` form the CLI uses.

    The driver's NS path only accepts the string form (because the CLI
    parses it that way). ``None`` means "no kwargs".
    """
    if not kwargs:
        return None
    return " ".join("{}={}".format(k, v) for k, v in kwargs.items())
