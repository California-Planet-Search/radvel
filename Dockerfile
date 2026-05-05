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
FROM python:3.12-slim AS builder

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
FROM python:3.12-slim AS runtime

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

RUN apt-get update && apt-get install --no-install-recommends -y \
        libhdf5-103-1 \
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        lmodern \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root operator. /data is the only writable mount.
RUN useradd --system --uid 10001 --gid root --no-create-home radvel \
 && mkdir -p /data /usr/local/share/radvel \
 && chown -R radvel:root /data /usr/local/share/radvel

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels \
        radvel \
        celerite \
        fastapi "uvicorn[standard]" pydantic pydantic-settings \
        python-multipart httpx aiosqlite \
 && rm -rf /wheels

# Bundle the example datasets so air-gapped installs still have inputs.
COPY --chown=radvel:root example_data /usr/local/share/radvel/example_data

USER radvel
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/healthz \
        && python -c "import radvel._kepler" \
        || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "radvel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
