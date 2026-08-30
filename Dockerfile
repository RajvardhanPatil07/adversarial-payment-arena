# Adversarial Payment Arena -- single-container deployment.
#
# The frontend is exported to static assets and served by FastAPI so the UI,
# REST endpoints and WebSocket share one origin in deployment.

# --------------------------------------------------------------------------- #
# Stage 1: build the UI to static assets
# --------------------------------------------------------------------------- #
FROM node:22-slim AS ui

WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
ENV STATIC_EXPORT=1
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# --------------------------------------------------------------------------- #
# Stage 2: compile the Rust/PyO3 transaction-feature + graph-risk hot paths
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS rust-core

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cargo \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "maturin>=1.9.4,<2"

WORKDIR /build
COPY backend/rust_core/ ./backend/rust_core/
COPY backend/rust_graph_core/ ./backend/rust_graph_core/
RUN mkdir -p /wheels \
 && maturin build \
      --release \
      --manifest-path backend/rust_core/Cargo.toml \
      --interpreter python3 \
      --out /wheels \
 && maturin build \
      --release \
      --manifest-path backend/rust_graph_core/Cargo.toml \
      --interpreter python3 \
      --out /wheels

# --------------------------------------------------------------------------- #
# Stage 3: python runtime that serves API + WS + the exported UI
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS runtime

# libgomp1 is required by xgboost's OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt ./backend/requirements.txt
COPY --from=rust-core /wheels/ /tmp/rust-wheels/
RUN pip install --no-cache-dir -r backend/requirements.txt \
 && pip install --no-cache-dir /tmp/rust-wheels/*.whl \
 && rm -rf /tmp/rust-wheels

COPY backend/ ./backend/
COPY artifacts/ ./artifacts/
COPY docs/ ./docs/
COPY Makefile ./

COPY --from=ui /ui/out ./frontend_dist

RUN useradd --create-home --uid 10001 arena && chown -R arena:arena /app
USER arena

ENV PORT=8000 \
    SERVE_STATIC_DIR=/app/frontend_dist
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
