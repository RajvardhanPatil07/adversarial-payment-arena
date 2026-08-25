"""
Step 5 exit criteria — live server + WebSocket campaign test.

Boots uvicorn as a subprocess (offline attacker forced by stripping
OPENROUTER_API_KEY), then:
  * REST smoke: /api/health, /api/merchants, /api/load_attack, /api/graph/snapshot
  * WS: run a 10-transaction attack_1 campaign and assert the mandated event
    types arrive: agent_thought, payload_generated, plausibility_check,
    defense_decision — plus cost_update and graph_update.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(port: int, timeout_s: float = 40.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.25)
    return False


@pytest.fixture(scope="module")
def server_port():
    port = _free_port()
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)  # force deterministic offline fraudster
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(BACKEND_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        assert _wait_ready(port), "uvicorn did not become ready"
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------- #
# REST contract
# --------------------------------------------------------------------------- #


def test_rest_endpoints(server_port: int):
    base = f"http://127.0.0.1:{server_port}"

    health = httpx.get(f"{base}/api/health").json()
    assert health["status"] == "ok"

    merchants = httpx.get(f"{base}/api/merchants").json()
    assert len(merchants) == 20
    assert {"merchant_id", "mcc", "country", "is_online"} <= set(merchants[0])

    loaded = httpx.post(f"{base}/api/load_attack", json={"filename": "attack_1"}).json()
    assert loaded["spec"]["spec_id"] == "ATTACK_1_MFA_RESET_VOICE_CLONE"
    assert loaded["spec"]["economic_model"]["acquisition_cost_usd"] == 50

    snap = httpx.get(f"{base}/api/graph/snapshot").json()
    assert set(snap) == {"nodes", "edges"}
    assert all({"id", "type"} == set(n) for n in snap["nodes"])

    # path traversal is rejected at the door
    assert httpx.post(f"{base}/api/load_attack",
                      json={"filename": "../secrets"}).status_code == 400


# --------------------------------------------------------------------------- #
# Mandated WebSocket campaign flow
# --------------------------------------------------------------------------- #


async def _ws_campaign_flow(port: int) -> tuple[dict, int]:
    import websockets

    seen_types: list[str] = []
    defense_decisions = 0
    summary = None

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        await ws.send(json.dumps({
            "type": "start_campaign",
            "attack_file": "attack_1",
            "campaign_size": 10,
            "sleep_s": 0,          # tests don't pay the rate-limit tax
        }))
        while summary is None:
            evt = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            seen_types.append(evt["type"])
            if evt["type"] == "defense_decision":
                defense_decisions += 1
                assert evt["decision"] in {"APPROVE", "STEP_UP", "DECLINE", "MANUAL_REVIEW"}
                assert isinstance(evt["reasons"], list)
            if evt["type"] == "campaign_summary":
                summary = evt
    return {
        "types": set(seen_types),
        "defense_decisions": defense_decisions,
        "summary": summary,
    }, defense_decisions


def test_ws_ten_txn_campaign(server_port: int):
    result, defenses = asyncio.run(_ws_campaign_flow(server_port))

    mandated = {"agent_thought", "payload_generated", "plausibility_check", "defense_decision"}
    missing = mandated - result["types"]
    assert not missing, f"missing mandated events: {missing}"
    assert defenses >= 10, f"expected >=10 defense decisions, got {defenses}"
    assert result["summary"]["data"]["accepted"] >= 3   # offline stub clears retry on slot 1
    assert "cost_update" in result["types"]
    assert "graph_update" in result["types"]
