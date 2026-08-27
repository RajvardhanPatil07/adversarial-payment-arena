"""
Additive HTTP routers.

Routers here are mounted onto the existing FastAPI app in `main.py` and are
read-only: they serve artifacts produced by `backend/experiments/`, they do not
recompute metrics per request.
"""
