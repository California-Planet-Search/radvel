"""Pydantic v2 schemas for the RadVel HTTP API.

These models mirror the namespace of a Python setup file (see
:func:`radvel.utils.initialize_posterior_from_dict`) plus the request and
response shapes for every endpoint exposed in :mod:`radvel.api.main`.

Importing this module requires pydantic v2; it is included in the
``[api]`` extra. The module is **not** imported by ``radvel/__init__.py``
so library users without the API extra installed are unaffected.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from radvel import basis as _basis


JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled")
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
JobKind = Literal["mcmc", "ns"]


class ParameterIn(BaseModel):
    """One orbital / instrument parameter (matches :class:`radvel.model.Parameter`)."""

    model_config = ConfigDict(extra="forbid")

    value: float
    vary: bool = True
    mcmcscale: Optional[float] = None
    linear: bool = False


# ---- Data ----------------------------------------------------------------


class _DataInBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataRow(BaseModel):
    """A single radial-velocity epoch."""

    model_config = ConfigDict(extra="allow")

    time: float
    mnvel: float
    errvel: float
    tel: str


class DataInline(_DataInBase):
    """Pass RV epochs inline as a list of rows."""

    kind: Literal["inline"] = "inline"
    rows: List[DataRow]


class DataCsvBase64(_DataInBase):
    """Pass an entire CSV file (with header) base64-encoded."""

    kind: Literal["csv_base64"] = "csv_base64"
    csv_base64: str
    separator: str = ","


class DataServerPath(_DataInBase):
    """Reference a CSV/text file already on the server.

    The server enforces the file lives inside ``RADVEL_DATA_ALLOWLIST``;
    requests pointing outside the allowlist receive ``403``.
    """

    kind: Literal["server_path"] = "server_path"
    path: str


class DataDatasetRef(_DataInBase):
    """Reference one of the example datasets shipped with the radvel package."""

    kind: Literal["dataset_ref"] = "dataset_ref"
    dataset: str


DataIn = Annotated[
    Union[DataInline, DataCsvBase64, DataServerPath, DataDatasetRef],
    Field(discriminator="kind"),
]


# ---- Priors --------------------------------------------------------------


class _PriorBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GaussianPrior(_PriorBase):
    type: Literal["gaussian"] = "gaussian"
    param: str
    mu: float
    sigma: float


class JeffreysPrior(_PriorBase):
    type: Literal["jeffreys"] = "jeffreys"
    param: str
    minval: float
    maxval: float


class ModifiedJeffreysPrior(_PriorBase):
    type: Literal["modifiedjeffreys"] = "modifiedjeffreys"
    param: str
    minval: float
    maxval: float
    kneeval: float


class HardBoundsPrior(_PriorBase):
    type: Literal["hardbounds"] = "hardbounds"
    param: str
    minval: float
    maxval: float


class EccentricityPriorIn(_PriorBase):
    type: Literal["eccentricity"] = "eccentricity"
    num_planets: Union[int, List[int]]
    upperlims: Union[float, List[float]] = 0.99


class PositiveKPriorIn(_PriorBase):
    type: Literal["positivek"] = "positivek"
    num_planets: int


class SecondaryEclipsePriorIn(_PriorBase):
    type: Literal["secondaryeclipse"] = "secondaryeclipse"
    planet_num: int
    ts: float
    ts_err: float


class InformativeBaselinePriorIn(_PriorBase):
    type: Literal["informative_baseline"] = "informative_baseline"
    param: str
    baseline: float
    duration: float = 0.0


PriorIn = Annotated[
    Union[
        GaussianPrior,
        JeffreysPrior,
        ModifiedJeffreysPrior,
        HardBoundsPrior,
        EccentricityPriorIn,
        PositiveKPriorIn,
        SecondaryEclipsePriorIn,
        InformativeBaselinePriorIn,
    ],
    Field(discriminator="type"),
]


# ---- Optional sub-objects ------------------------------------------------


class StellarIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mstar: float
    mstar_err: float


# ---- Top-level request ---------------------------------------------------


class RunCreateRequest(BaseModel):
    """JSON body for ``POST /runs`` mirroring an example_planets/<x>.py file."""

    model_config = ConfigDict(extra="forbid")

    starname: str
    nplanets: int = Field(ge=1, le=20)
    instnames: List[str] = Field(min_length=1)
    fitting_basis: str
    bjd0: float = 0.0
    time_base: Optional[float] = None
    planet_letters: Optional[Dict[str, str]] = None

    params: Dict[str, ParameterIn]
    anybasis_params: Optional[Dict[str, ParameterIn]] = None
    any_basis: Optional[str] = None

    data: DataIn
    priors: List[PriorIn] = Field(default_factory=list)

    stellar: Optional[StellarIn] = None
    planet: Optional[Dict[str, float]] = None
    decorr_vars: Optional[List[str]] = None
    hnames: Optional[Dict[str, List[str]]] = None
    kernel_name: Optional[Dict[str, Literal["QuasiPer", "Celerite"]]] = None

    @field_validator("fitting_basis")
    @classmethod
    def _check_fitting_basis(cls, v: str) -> str:
        if v not in _basis.BASIS_NAMES:
            raise ValueError(
                "fitting_basis must be one of {}".format(sorted(_basis.BASIS_NAMES))
            )
        return v

    @field_validator("any_basis")
    @classmethod
    def _check_any_basis(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _basis.BASIS_NAMES:
            raise ValueError(
                "any_basis must be one of {}".format(sorted(_basis.BASIS_NAMES))
            )
        return v


# ---- Responses -----------------------------------------------------------


class RunCreateResponse(BaseModel):
    run_id: str
    outputdir: str
    created_at: datetime


class RunSummary(BaseModel):
    run_id: str
    starname: str
    created_at: datetime
    fitting_basis: str
    nplanets: int


class RunStatus(BaseModel):
    """Full state for a single run, parsed from the on-disk ``.stat`` file."""

    run_id: str
    starname: str
    fitting_basis: str
    nplanets: int
    created_at: datetime
    radvel_version: str
    state: Dict[str, Any]
    active_job: Optional["JobSummary"] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    kepler_c: bool
    version: str
    allow_py_upload: bool = False
    enable_ui: bool = True


class VersionResponse(BaseModel):
    radvel: str
    api: str
    python: str
    platform: str


class FitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decorr: bool = False


class FitResult(BaseModel):
    logprob: float
    rms: float
    chi2_reduced: Optional[float] = None
    params: Dict[str, float]
    postfile: str


class MCMCRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nsteps: int = Field(default=10000, ge=1)
    nwalkers: int = Field(default=50, ge=4)
    ensembles: int = Field(default=8, ge=1)
    minAfactor: float = 40.0
    maxArchange: float = 0.03
    maxGR: float = 1.01
    burnGR: float = 1.03
    minTz: int = 1000
    minsteps: int = 0
    minpercent: float = 5.0
    thin: int = 1
    serial: bool = False
    save: bool = False
    proceed: bool = False


class NSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sampler: Literal["ultranest", "dynesty", "PyMultiNest"] = "ultranest"
    sampler_kwargs: Dict[str, Any] = Field(default_factory=dict)
    run_kwargs: Dict[str, Any] = Field(default_factory=dict)
    proceed: bool = False
    overwrite: bool = False


class JobProgress(BaseModel):
    """Live progress snapshot — fields mirror :data:`radvel.mcmc.statevars`."""

    model_config = ConfigDict(extra="allow")

    nsteps_complete: Optional[int] = None
    totsteps: Optional[int] = None
    pcomplete: Optional[float] = None
    rate: Optional[float] = None
    ar: Optional[float] = None
    minafactor: Optional[float] = None
    maxarchange: Optional[float] = None
    mintz: Optional[float] = None
    maxgr: Optional[float] = None
    ismixed: Optional[bool] = None
    burn_complete: Optional[bool] = None
    nburn: Optional[int] = None
    iteration: Optional[int] = None
    live_points: Optional[int] = None
    logz: Optional[float] = None
    dlogz: Optional[float] = None


class JobSummary(BaseModel):
    job_id: str
    run_id: str
    kind: JobKind
    state: JobState
    submitted_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class JobStatus(JobSummary):
    progress: JobProgress = Field(default_factory=JobProgress)
    error: Optional[str] = None


class JobKickoffResponse(BaseModel):
    job_id: str
    run_id: str
    kind: JobKind
    state: JobState = "queued"


class DeriveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sampler: Literal["auto", "mcmc", "ns"] = "auto"


class DeriveResult(BaseModel):
    columns: List[str]
    quantfile: str


class ICCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    types: List[Literal["nplanets", "e", "trend", "jit", "gp"]]
    mixed: bool = True
    fixjitter: bool = False
    simple: bool = False
    verbose: bool = False


class ICCompareResult(BaseModel):
    statsdicts: List[Dict[str, Any]]


class TablesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    types: List[Literal["params", "priors", "rv", "ic_compare", "derived", "crit"]]
    header: bool = False
    name_in_title: bool = False
    sampler: Literal["auto", "mcmc", "ns"] = "auto"


class TablesResult(BaseModel):
    latex: Dict[str, str]
    files: Dict[str, str]


class PlotsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    types: List[Literal["rv", "auto", "corner", "trend", "derived"]]
    plotkw: Dict[str, Any] = Field(default_factory=dict)
    gp: bool = False
    sampler: Literal["auto", "mcmc", "ns"] = "auto"


class PlotFile(BaseModel):
    type: str
    path: str
    url: str


class PlotsResult(BaseModel):
    files: List[PlotFile]


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comptype: Literal["ic"] = "ic"
    latex_compiler: str = "pdflatex"
    sampler: Literal["auto", "mcmc", "ns"] = "auto"


class ReportResult(BaseModel):
    file: str
    url: str
    stdout: str = ""


class FileEntry(BaseModel):
    name: str
    size: int
    mtime: datetime
    content_type: Optional[str] = None


class ErrorPayload(BaseModel):
    """Standard error envelope used for all 4xx/5xx responses."""

    error_type: str
    message: str
    traceback_id: Optional[str] = None


# Resolve the forward reference on RunStatus.active_job
RunStatus.model_rebuild()
