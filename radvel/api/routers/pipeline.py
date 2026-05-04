"""Synchronous pipeline endpoints (fit, derive, ic, tables).

Long-running operations (mcmc, ns) live in :mod:`radvel.api.routers.jobs`
once M3 lands. Plotting and reports land in M4.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from radvel.api import schemas
from radvel.api.config import get_settings
from radvel.api.drivers_adapter import (
    AdapterError,
    run_derive,
    run_fit,
    run_ic_compare,
    run_tables,
)
from radvel.api.runs import RunNotFound, RunRegistry, is_valid_run_id


router = APIRouter(prefix="/runs/{run_id}", tags=["pipeline"])


def _registry() -> RunRegistry:
    return RunRegistry(settings=get_settings())


def _resolve(run_id: str, registry: RunRegistry):
    if not is_valid_run_id(run_id):
        raise HTTPException(status_code=404, detail="unknown run_id")
    try:
        return registry.get(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail="unknown run_id")


def _surface(err: AdapterError) -> HTTPException:
    return HTTPException(
        status_code=err.status_code,
        detail={
            "error_type": err.error_type,
            "message": err.message,
            "traceback_id": err.traceback_id,
        },
    )


@router.post("/fit", response_model=schemas.FitResult, status_code=status.HTTP_200_OK)
def fit_endpoint(
    run_id: str,
    body: schemas.FitRequest = schemas.FitRequest(),
    registry: RunRegistry = Depends(_registry),
) -> schemas.FitResult:
    record = _resolve(run_id, registry)
    try:
        result = run_fit(record, decorr=body.decorr)
    except AdapterError as err:
        raise _surface(err)
    return schemas.FitResult(**result)


@router.post("/derive", response_model=schemas.DeriveResult)
def derive_endpoint(
    run_id: str,
    body: schemas.DeriveRequest = schemas.DeriveRequest(),
    registry: RunRegistry = Depends(_registry),
) -> schemas.DeriveResult:
    record = _resolve(run_id, registry)
    try:
        result = run_derive(record, sampler=body.sampler)
    except AdapterError as err:
        raise _surface(err)
    return schemas.DeriveResult(**result)


@router.post("/ic", response_model=schemas.ICCompareResult)
def ic_endpoint(
    run_id: str,
    body: schemas.ICCompareRequest,
    registry: RunRegistry = Depends(_registry),
) -> schemas.ICCompareResult:
    record = _resolve(run_id, registry)
    try:
        result = run_ic_compare(
            record,
            types=body.types,
            mixed=body.mixed,
            fixjitter=body.fixjitter,
            simple=body.simple,
            verbose=body.verbose,
        )
    except AdapterError as err:
        raise _surface(err)
    return schemas.ICCompareResult(**result)


@router.post("/tables", response_model=schemas.TablesResult)
def tables_endpoint(
    run_id: str,
    body: schemas.TablesRequest,
    registry: RunRegistry = Depends(_registry),
) -> schemas.TablesResult:
    record = _resolve(run_id, registry)
    try:
        result = run_tables(
            record,
            types=body.types,
            header=body.header,
            name_in_title=body.name_in_title,
            sampler=body.sampler,
        )
    except AdapterError as err:
        raise _surface(err)
    return schemas.TablesResult(**result)
