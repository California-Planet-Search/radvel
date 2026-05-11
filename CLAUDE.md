# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RadVel is an open-source Python package for modeling Keplerian orbits in radial velocity (RV) time series to detect and characterize exoplanets. It performs MAP fitting and MCMC posterior sampling with real-time convergence tests. Reference paper: [Fulton et al. (2018, PASP 130, 044504)](http://adsabs.harvard.edu/abs/2018PASP..130d4504F).

## Build & Development Commands

```bash
# Development install (build C extension + install in editable mode)
python setup.py build_ext -i && pip install -e .

# Run full test suite
pytest radvel -v --tb=short

# Run a single test function
pytest radvel/tests/test_api.py::test_k2_24 -v

# Run tests with coverage
pytest radvel --cov=radvel --cov-report=term-missing -v

# Build C Kepler solver extension only
python setup.py build_ext -i
```

The C extension (`radvel/_kepler`) is critical for performance. If it fails to build, RadVel falls back to a pure NumPy Kepler solver. Verify it loaded with: `python -c "import radvel._kepler; print('OK')"`.

On macOS, install `pytables` and `h5py` via conda before pip installing to avoid compilation issues.

## Architecture

### Core Pipeline (CLI: `radvel` command)

The CLI workflow is sequential: **fit → mcmc/ns → derive → ic_compare → tables → plot → report**

- [cli.py](radvel/cli.py) — Entry point, argument parsing, dispatches to driver functions
- [driver.py](radvel/driver.py) — High-level orchestration; each CLI subcommand maps to a driver function
- State is persisted between steps via pickle files (`_post_obj.pkl`) and a `.stat` status file

### Data Model

- [model.py](radvel/model.py) — `Parameter` (value/vary/mcmcscale/linear), `Parameters` (OrderedDict of Parameter objects), `RVModel` (forward model computing RV from parameters), `Vector` (dict↔array conversion)
- [basis.py](radvel/basis.py) — `Basis` class converts between orbital parameterizations. The **synthesis basis** (`per tp e w k`) is the internal representation used by the Kepler solver. The standard **fitting basis** is `per tc secosw sesinw k` which imposes flat priors on all orbital elements, avoids biasing K>0, and speeds MCMC convergence. Priors are assumed uniform in the fitting basis, so basis choice imposes implicit priors on orbital elements. New bases are added by updating `BASIS_NAMES` and the `to_synth`/`from_synth` methods. See `BASIS_NAMES` for the 10 supported bases.

### Kepler Solver

- [kepler.py](radvel/kepler.py) — `rv_drive()` computes RV signal from orbital elements `[per, tp, e, w, K]`. Solves Kepler's equation (M = E - e sin E) using the iterative method of Danby (1988) with third-order corrections. Uses C extension (`src/_kepler.pyx`) when available, else pure NumPy. Has a fast path for circular orbits (e=0). Eccentricity is clamped to [0, 0.99].
- The RV equation is `v_r = K[cos(nu + w) + e*cos(w)]` where the z-axis points *away* from the observer (positive RV = redshift). This is the standard observational convention (see paper Section 2.1).

### Statistical Framework

- [likelihood.py](radvel/likelihood.py) — `RVLikelihood` computes ln(L) assuming Gaussian noise: `-0.5 * sum((model - data)^2 / (err^2 + jit^2)) - sum(ln(sqrt(2*pi*(err^2 + jit^2))))`. The jitter (sigma_i) is per-instrument white noise added in quadrature. `CompositeLikelihood` sums log-likelihoods across instruments. `GPLikelihood` adds Gaussian Process correlated noise. `extra_params` holds non-Keplerian params (gamma, jitter) needed for likelihood but not the RV model itself.
- [posterior.py](radvel/posterior.py) — `Posterior` wraps Likelihood + list of Prior objects. `logprob = ln(L) + sum(ln(P_k))` for each prior P_k. If no priors defined, all are assumed uniform.
- [prior.py](radvel/prior.py) — Prior types: `Gaussian`, `Jeffreys`, `HardBounds`, `EccentricityPrior`, `PositiveKPrior`, `SecondaryEclipsePrior`. Priors are callable objects returning a single log-probability value.
- [gp.py](radvel/gp.py) — GP kernels: squared exponential, periodic, quasiperiodic, celerite-backed kernels. GP parameters prefixed `gp_` (e.g., `gp_amp`, `gp_per`, `gp_explength`, `gp_perlength`)

### Sampling

- [fitting.py](radvel/fitting.py) — MAP estimation via Powell's method in `scipy.optimize.minimize`
- [mcmc.py](radvel/mcmc.py) — MCMC via `emcee>=3` Affine Invariant sampler. Default: 8 independent ensembles of 50 walkers each, run in parallel on separate CPUs. Burn-in phase runs until G-R < 1.03, then chains restart. Production convergence requires all four criteria met for 5 consecutive checks: Gelman-Rubin < 1.01 (`maxGR`), independent samples Tz > 1000 (`minTz`), autocorrelation factor N/tau >= 40 (`minAfactor`), autocorrelation relative change < 0.03 (`maxArchange`). G-R and Tz are computed across ensembles (not within). Default MCMC step sizes: 10% of parameter value (0.001% for period).
- [nested_sampling.py](radvel/nested_sampling.py) — Nested sampling via `ultranest` (default), `dynesty`, or `PyMultiNest`

### Output

- [plot/](radvel/plot/) — `orbit_plots.py` (RV multipanel), `mcmc_plots.py` (corner, trend, autocorrelation)
- [report.py](radvel/report.py) — PDF report generation via Jinja2 templates in [templates/](radvel/templates/). Requires `pdflatex` on PATH.

### RV Model Equation

The total RV model for multiple planets is: `V_r = sum(v_r_k) + gamma + gamma_dot*(t - t0) + gamma_ddot*(t - t0)^2` where `t0` (`time_base`) is an arbitrary reference epoch near the midpoint of the time baseline, and the sum is over all planets.

## Setup File Convention

Planet configurations live in `example_planets/` as Python scripts defining: `starname`, `nplanets`, `instnames`, `fitting_basis`, `params` (Parameters object), `data` (DataFrame with columns: `time`, `mnvel`, `errvel`, `tel`), `priors` (list), and optionally `stellar` dict (with `mstar`, `mstar_err` for deriving physical parameters). `bjd0` defines the reference epoch subtracted from timestamps. `planet_letters` maps planet indices to letters for display. These are loaded via `radvel.utils.initialize_posterior()`.

## Key Implementation Details

- NumPy multithreading is disabled at import (`MKL_NUM_THREADS=1`, etc.) to avoid conflicts with emcee's multiprocessing
- `RADVEL_DATADIR` environment variable overrides example data path lookup
- Parameter names follow the pattern `{param}{planet_num}` for orbital params (e.g., `per1`, `tc2`) and `{param}_{instrument}` for instrument params (e.g., `gamma_k`, `jit_j`)
- Parameters with `vary=False` and `linear=True` (gamma params) are solved analytically
- Python 3.8, 3.9, 3.11, 3.12 supported. CI runs on GitHub Actions.
- For low-SNR detections, MCMC can produce bimodal posteriors for K (jumping between +q and -q solutions). This also affects Tc and/or w. Use caution interpreting posteriors and explore multiple basis sets.

## API & UI module layout (v1.6+)

`radvel/api/` is an optional FastAPI service that wraps the CLI workflow over HTTP. Disabled by default at import time — only loaded when the `[api]` extra is installed (`pip install -e ".[api,dev-api]"`). Layout:

- `main.py` — FastAPI app factory, lifespan installs the `JobRunner` on `app.state.job_runner`.
- `config.py` — pydantic-settings reading `RADVEL_API_*` env vars.
- `schemas.py` — Pydantic v2 request/response models (mirrors the legacy Python setup file).
- `runs.py` — RunRegistry / RunRecord; per-run dir holds `<run_id>.json` + a tiny `<run_id>.py` shim that loads the JSON so legacy `radvel.driver` code keeps working.
- `jobs.py` — SQLite-backed JobRegistry + ProcessPoolExecutor JobRunner (mcmc + ns).
- `drivers_adapter.py` — sync adapters around `radvel.driver.*` plus async workers for mcmc/ns. Translates driver `AssertionError` (step prereq) → 409, other exceptions → 500 with traceback IDs.
- `progress.py` — atomic-write JSON callback target for live MCMC telemetry.
- `routers/{health,runs,pipeline,jobs,files,ui}.py` — endpoint definitions.
- `static/{index.html,app.js,styles.css,favicon.svg}` — the optional `/ui` SPA (Tailwind via CDN, no build step).

Common ops:

```bash
# Run service locally with hot reload
radvel serve --reload --port 8000          # forbidden in prod

# API integration tests (slow, needs [api] extra)
pytest radvel -m api -v --tb=short

# Smoke against a running container (the docker-smoke CI job runs this)
python radvel/api/tests/scripts/smoke_e2e.py http://localhost:8000

# Multi-stage image
docker compose up --build
```

### Parity guarantee for `initialize_posterior_from_dict`

`radvel.utils.initialize_posterior_from_dict` builds a Posterior from a dict-shaped setup; `_finalise_posterior` is shared with the legacy `initialize_posterior` so JSON ↔ Python paths must produce identical `post.logprob()` (`<= 1e-12`). `radvel/tests/test_dict_init_parity.py` enforces this — break it and the default suite fails.

### `radvel.mcmc.statevars` is a module-level singleton

Never run MCMC in-process inside the API server — the singleton would collide between concurrent requests. Hence the ProcessPoolExecutor in `radvel/api/jobs.py`. Each MCMC job is a fresh child process; cancellation is `os.kill(pid, SIGTERM)` translated to `SystemExit` by a handler installed in the worker.

### Refreshing UI screenshots

Maintainer-only task. The screenshots in `docs/_static/screenshots/` are regenerated by `scripts/regen_ui_screenshots.py` (Playwright). Run on every UI-affecting PR and commit the resulting PNGs. Not run on RTD or default CI.

## Contributing

- Fork the repo, submit PRs into the `next-release` branch. Maintainers merge into `master` for tagged releases.
- Follow PEP 8. Use Napoleon-compatible docstrings (Google Python Style Guide).
- Include unit tests for new code (test suite in `radvel/tests/`).
