# syntax=docker/dockerfile:1
# RadVel HTTP API — single fat image bundling TeX Live so /report works.
#
# Build:
#   docker build -t radvel-api:dev .
#
# Run:
#   docker run --rm -p 8000:8000 -v $PWD/.runs:/data radvel-api:dev
#
# Multi-arch release builds happen in .github/workflows/docker.yml on
# `release: published` with linux/amd64 + linux/arm64.

# ---- builder ------------------------------------------------------------
# Compiles the Cython _kepler extension and the wheels for runtime deps.
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install --no-install-recommends -y \
        gcc g++ git pkg-config libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements.txt ./

# Build wheels for runtime deps once so the runtime stage can install
# without a compiler.
RUN pip install --upgrade pip wheel setuptools \
 && pip wheel --wheel-dir=/wheels \
        Cython pybind11 numpy hatchling \
 && pip wheel --wheel-dir=/wheels -r requirements.txt celerite \
 && pip wheel --wheel-dir=/wheels \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.7" \
        "pydantic-settings>=2.4" python-multipart httpx aiosqlite

# Copy the source last so dependency wheels above stay cached across edits.
COPY . /src

# Build the radvel wheel itself (this compiles _kepler).
RUN pip wheel --no-deps --wheel-dir=/wheels .


# ---- runtime ------------------------------------------------------------
# Same Python base, plus runtime libs and TeX Live for /report.
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    RADVEL_DATADIR=/usr/local/share/radvel/example_data \
    RADVEL_API_RUNS_DIR=/data/runs \
    RADVEL_API_DB_PATH=/data/jobs.db \
    RADVEL_API_HOST=0.0.0.0 \
    RADVEL_API_PORT=8000 \
    RADVEL_API_WORKERS=1 \
    RADVEL_API_ALLOW_PY_UPLOAD=false \
    RADVEL_API_ENABLE_UI=true

# libhdf5-dev pulls in the runtime libhdf5-* package as a dependency
# regardless of which SOVERSION the active Debian release ships
# (Bookworm = 103, Trixie = 310, …). Worth ~5 MB to avoid a per-release
# package-name dance. We install it here even though h5py never compiles
# in the runtime stage — it just dlopens the SO at import time.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libhdf5-dev \
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        texlive-publishers \
        texlive-plain-generic \
        lmodern \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root operator. /data is the only writable mount.
# `--no-create-home` with HOME still set to /home/radvel left the service
# user without a writable home, so anything that caches there failed on
# first use -- matplotlib logged `mkdir -p failed for path
# /home/radvel/.config/matplotlib` on every start and fell back to /tmp.
# Harmless in itself, but it is noise in the logs that sent an outage
# investigation down a false path, so give the user the home it is told
# it has.
RUN useradd --system --uid 10001 --gid root --no-create-home radvel \
 && mkdir -p /data /usr/local/share/radvel /home/radvel \
 && chown -R radvel:root /data /usr/local/share/radvel /home/radvel

# The wheels are bind-mounted from the builder stage rather than COPYed in. A
# COPY commits them to their own layer, and the `rm -rf` that used to follow
# could only write a whiteout on top of that layer -- it cannot remove a layer
# that is already committed, so the wheels shipped in every pull. A bind mount
# is never committed to a layer, so there is nothing left to remove.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-index --find-links=/wheels \
        radvel \
        celerite \
        fastapi "uvicorn[standard]" pydantic pydantic-settings \
        python-multipart httpx aiosqlite

# Bundle the example datasets so air-gapped installs still have inputs.
COPY --chown=radvel:root example_data /usr/local/share/radvel/example_data

USER radvel
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8000

# `/healthz` already imports `radvel._kepler` in-process and reports the
# result as `kepler_c`, so the old second clause (`python -c "import
# radvel._kepler"`) re-answered a question the endpoint had just answered --
# at the cost of starting a fresh interpreter and importing a compiled
# extension inside the 5s budget, on a box that may be running several MCMC
# fits. That is what pushed the check past its timeout: the recorded output
# showed a valid `{"status": "ok", "kepler_c": true, ...}` AND a failure, with
# a failing streak in the thousands. A permanently-unhealthy container is
# worse than none, because nobody can tell "always been like that" from "just
# broke" -- it is why a two-week outage went unnoticed.
#
# Grepping the payload keeps the same assertion (the extension is importable)
# with no interpreter start-up: `status` is "degraded", not an error code,
# when it is missing, so `curl --fail` alone would not catch it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/healthz \
        | grep -q '"kepler_c":true' \
        || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "radvel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
