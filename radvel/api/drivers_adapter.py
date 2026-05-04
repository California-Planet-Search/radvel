"""Sync adapters wrapping :mod:`radvel.driver` for the HTTP API.

The driver functions accept an ``argparse.Namespace`` and write outputs
relative to ``args.outputdir``. These adapters build that namespace from
typed request models, capture stdout, and translate exceptions into HTTP
errors. Long-running operations (``mcmc``, ``nested_sampling``) ship in
M3 via the async job runner; this module covers the synchronous
operations only.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import radvel
from radvel import driver
from radvel.driver import load_status

from .runs import RunRecord


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
