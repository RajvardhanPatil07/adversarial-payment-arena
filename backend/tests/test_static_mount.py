"""
Deployment wiring tests for the single-origin container.

The deployed prototype serves the UI, the REST API and the WebSocket from one
origin (see Dockerfile for why). That is implemented by mounting an exported
Next.js bundle at "/" -- a catch-all. A catch-all mount registered before the
API routes silently shadows them: every /api call would return the UI's HTML
with a 200, and the dashboard would render empty panels with no console error.

That failure only appears in the container, never in `next dev`, so it would
surface for the first time in front of a judge. These tests pin the ordering and
the fallback behaviour so it cannot regress.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_main(monkeypatch, static_dir: str | None):
    """Import main.py fresh, since the mount happens at module import time."""
    if static_dir is None:
        monkeypatch.delenv("SERVE_STATIC_DIR", raising=False)
    else:
        monkeypatch.setenv("SERVE_STATIC_DIR", static_dir)
    sys.modules.pop("main", None)
    return importlib.import_module("main")


@pytest.fixture
def exported_ui(tmp_path: Path) -> Path:
    """A minimal stand-in for `next build` output."""
    dist = tmp_path / "frontend_dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>arena-ui</body></html>")
    sub = dist / "evidence"
    sub.mkdir()
    (sub / "index.html").write_text("<html><body>evidence-page</body></html>")
    return dist


def test_no_mount_without_env_var(monkeypatch):
    """Local dev and CI must be untouched: no static mount, no behaviour change."""
    main = _load_main(monkeypatch, None)
    assert not any(
        isinstance(r, Mount) and r.name == "ui" for r in main.app.routes
    ), "UI must not be mounted unless SERVE_STATIC_DIR is set"


def test_missing_directory_degrades_instead_of_crashing(monkeypatch, tmp_path: Path):
    """A bad path must not crash-loop the container.

    If the mount raised, the whole app would die and a judge would lose the API
    too. Losing the UI while keeping /api and /ws is strictly better.
    """
    main = _load_main(monkeypatch, str(tmp_path / "does_not_exist"))
    assert not any(isinstance(r, Mount) and r.name == "ui" for r in main.app.routes)
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200


def test_ui_mount_is_registered_last(monkeypatch, exported_ui: Path):
    """The catch-all must be the final route or it shadows the API."""
    main = _load_main(monkeypatch, str(exported_ui))
    last = main.app.routes[-1]
    assert isinstance(last, Mount) and last.name == "ui", (
        "the '/' UI mount must be the last registered route; anything added "
        "after it is unreachable, and anything it precedes is shadowed"
    )


def test_api_routes_win_over_the_catch_all(monkeypatch, exported_ui: Path):
    """The regression that matters: /api must return JSON, not index.html."""
    main = _load_main(monkeypatch, str(exported_ui))
    with TestClient(main.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert "application/json" in health.headers["content-type"]

        attacks = client.get("/api/attacks")
        assert attacks.status_code == 200
        assert "attacks" in attacks.json()


def test_ui_is_served_at_root_and_nested_routes(monkeypatch, exported_ui: Path):
    """html=True must resolve directory-style export output."""
    main = _load_main(monkeypatch, str(exported_ui))
    with TestClient(main.app) as client:
        root = client.get("/")
        assert root.status_code == 200 and "arena-ui" in root.text

        # trailingSlash export layout: /evidence/ -> evidence/index.html
        nested = client.get("/evidence/")
        assert nested.status_code == 200 and "evidence-page" in nested.text


def test_websocket_still_reachable_behind_the_mount(monkeypatch, exported_ui: Path):
    """The socket is the whole application; a shadowed /ws is fatal."""
    main = _load_main(monkeypatch, str(exported_ui))
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"


@pytest.fixture(autouse=True)
def _restore_main_module():
    """Leave sys.modules as we found it so test ordering cannot leak state."""
    yield
    sys.modules.pop("main", None)
