"""Live FastAPI/WebSocket integration tests."""

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
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(port: int, timeout_s: float = 50.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = httpx.get(
                f"http://127.0.0.1:{port}/api/health", timeout=2.0
            )
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.25)
    return False


@pytest.fixture(scope="module")
def server_port():
    port = _free_port()
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(BACKEND_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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


def test_rest_endpoints(server_port: int):
    base = f"http://127.0.0.1:{server_port}"

    health = httpx.get(f"{base}/api/health").json()
    assert health["status"] == "ok"
    assert health["feature_backend"] in {"python", "rust"}
    assert health["graph_backend"] in {"python", "rust"}
    assert set(health["graph_state_sizes"]) == {"devices", "ips", "merchants"}
    assert "threat_miner" in health
    assert health["challenger"]["controls_live_authorizations"] is False

    merchants = httpx.get(f"{base}/api/merchants").json()
    assert len(merchants) == 20
    assert {"merchant_id", "mcc", "country", "is_online"} <= set(merchants[0])

    loaded = httpx.post(
        f"{base}/api/load_attack", json={"filename": "attack_1"}
    ).json()
    assert loaded["spec"]["spec_id"] == "ATTACK_1_MFA_RESET_VOICE_CLONE"
    assert loaded["spec"]["economic_model"]["acquisition_cost_usd"] == 50

    snapshot = httpx.get(f"{base}/api/graph/snapshot").json()
    assert set(snapshot) == {"nodes", "edges"}
    assert all({"id", "type"} == set(node) for node in snapshot["nodes"])

    threats = httpx.get(f"{base}/api/threats").json()
    assert set(threats) == {"diagnostics", "threats"}
    assert isinstance(threats["threats"], list)

    challenger = httpx.get(f"{base}/api/challenger").json()
    assert challenger["controls_live_authorizations"] is False
    assert challenger["name"].startswith("random_forest")

    containment = httpx.get(f"{base}/api/containment/last").json()
    assert "containment" in containment

    assert (
        httpx.post(f"{base}/api/load_attack", json={"filename": "../secrets"}).status_code
        == 400
    )


async def _ws_campaign_flow(port: int) -> tuple[dict, int]:
    import websockets

    seen_types: list[str] = []
    defense_decisions = 0
    defender_feedback = 0
    containment_summary = None
    summary = None

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        await ws.send(json.dumps({
            "type": "start_campaign",
            "attack_file": "attack_1",
            "campaign_size": 10,
            "sleep_s": 0,
            "feedback_mode": "gray",
        }))
        while containment_summary is None:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
            seen_types.append(event["type"])
            if event["type"] == "defense_decision":
                defense_decisions += 1
                assert event["decision"] in {
                    "APPROVE",
                    "STEP_UP",
                    "DECLINE",
                    "MANUAL_REVIEW",
                }
                assert isinstance(event["reasons"], list)
            if event["type"] == "defender_feedback":
                defender_feedback += 1
                assert event["mode"] == "gray"
                assert event["decision"] in {
                    "APPROVE",
                    "STEP_UP",
                    "DECLINE",
                    "MANUAL_REVIEW",
                }
                assert isinstance(event["signal_families"], list)
                assert event["signal_families"]
            if event["type"] == "campaign_summary":
                summary = event
                assert "containment" in event["data"]
            if event["type"] == "containment_summary":
                containment_summary = event["data"]

    return {
        "types": set(seen_types),
        "defense_decisions": defense_decisions,
        "defender_feedback": defender_feedback,
        "summary": summary,
        "containment": containment_summary,
    }, defense_decisions


def test_ws_ten_txn_campaign(server_port: int):
    result, defenses = asyncio.run(_ws_campaign_flow(server_port))

    mandated = {
        "agent_thought",
        "payload_generated",
        "plausibility_check",
        "defense_decision",
        "defender_feedback",
        "containment_summary",
    }
    missing = mandated - result["types"]
    assert not missing, f"missing mandated events: {missing}"
    assert defenses >= 10, f"expected >=10 defense decisions, got {defenses}"
    assert result["defender_feedback"] == defenses
    assert result["summary"]["data"]["accepted"] >= 3
    assert result["containment"]["transactions_scored"] == defenses
    assert 0.0 <= result["containment"]["escape_rate"] <= 1.0
    assert "cost_update" in result["types"]
    assert "graph_update" in result["types"]

    base = f"http://127.0.0.1:{server_port}"
    last = httpx.get(f"{base}/api/containment/last").json()["containment"]
    assert last["transactions_scored"] == defenses


async def _ws_validation_flow(port: int) -> list[dict]:
    import websockets

    replies: list[dict] = []
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        bad_messages = [
            [],
            {"type": "not_a_campaign", "attack_file": "attack_1"},
            {
                "type": "start_campaign",
                "attack_file": "attack_1",
                "campaign_size": "not-an-int",
            },
            {
                "type": "start_campaign",
                "attack_file": "attack_1",
                "sleep_s": "nan",
            },
            {
                "type": "start_campaign",
                "attack_file": "attack_1",
                "feedback_mode": "oracle",
            },
        ]
        for message in bad_messages:
            await ws.send(json.dumps(message))
            replies.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))
    return replies


def test_ws_rejects_malformed_and_unbounded_controls(server_port: int):
    replies = asyncio.run(_ws_validation_flow(server_port))
    assert len(replies) == 5
    assert all(reply["type"] == "error" for reply in replies)
    assert "JSON objects" in replies[0]["data"]
    assert "send" in replies[1]["data"]
    assert "invalid campaign controls" in replies[2]["data"]
    assert "invalid campaign controls" in replies[3]["data"]
    assert "feedback_mode" in replies[4]["data"]
