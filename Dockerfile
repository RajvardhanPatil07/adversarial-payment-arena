# Adversarial Payment Arena -- single-container deployment.
#
# WHY ONE CONTAINER AND NOT TWO
# -----------------------------
# The obvious split is "frontend on Vercel, backend on a VM". We deliberately
# do not do that, for three reasons that are all about the live demo:
#
#   1. The arena is a WebSocket application. Vercel's serverless functions do
#      not hold long-lived socket connections, so the backend has to live
#      somewhere else regardless -- the split buys nothing.
#   2. Two origins means CORS. A misconfigured ALLOWED_ORIGINS turns the demo
#      into a blank panel with a console-only error. Same-origin removes the
#      entire failure class.
#   3. An https:// page cannot open a ws:// socket (mixed content, blocked with
#      no fallback). Serving the UI from the same origin as the socket means the
#      scheme is always correct by construction.
#
# So: Next.js is exported to static assets at build time, and FastAPI serves
# those assets alongside /api and /ws. One URL, one process, no cross-origin
# surface. A judge opens one link and everything works.

# --------------------------------------------------------------------------- #
# Stage 1: build the UI to static assets
# --------------------------------------------------------------------------- #
FROM node:22-slim AS ui

WORKDIR /ui

# Copy manifests first so dependency installation is cached independently of
# source edits -- a source-only change does not re-resolve 638 packages.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./

# STATIC_EXPORT flips next.config.ts into `output: "export"`. It is a build-time
# switch rather than a committed default so that `npm run dev` keeps full
# server-side capability (including the /api/analyst streaming route) on a
# laptop, while the container ships a static bundle.
ENV STATIC_EXPORT=1
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# --------------------------------------------------------------------------- #
# Stage 2: python runtime that serves API + WS + the exported UI
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS runtime

# libgomp1 is required by xgboost's OpenMP runtime. Without it the import
# succeeds at build time and fails at first inference -- a failure that would
# only appear under live demo load.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
# The artifact set is part of the deliverable: the Evidence page reads these
# files, so a deployed prototype without them would show 404s where the
# reproducible numbers should be.
COPY artifacts/ ./artifacts/
COPY docs/ ./docs/
COPY Makefile ./

# Exported UI lands where main.py's static mount expects it.
COPY --from=ui /ui/out ./frontend_dist

# Non-root: the container needs no write access to anything it ships.
RUN useradd --create-home --uid 10001 arena && chown -R arena:arena /app
USER arena

ENV PORT=8000 \
    SERVE_STATIC_DIR=/app/frontend_dist
EXPOSE 8000

# Hitting the real readiness endpoint, not just the TCP port: the models are
# loaded during lifespan startup, so a port that accepts connections does not
# yet mean the defense stack can score a transaction.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
