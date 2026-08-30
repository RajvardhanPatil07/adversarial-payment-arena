"""FastAPI control plane for the Adversarial Payment Arena.

REST exposes health, registry, attack-spec, graph and evidence endpoints. The
WebSocket endpoint streams campaign events, decisions, cost updates and
incremental graph changes. Ambient legitimate traffic shares the same in-memory
issuer/decision state.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faker import Faker
from pydantic import BaseModel, Field

from agents.attacker import RATE_LIMIT_SLEEP_S, AttackerAgent
from api.evidence import router as evidence_router
from data.legit_generator import build_legit_payload
from defense.decision import DecisionEngine
from defense.novelty import NoveltyDetector
from defense.realtime import DEFAULT_MODEL_PATH, VelocityScorer
from environment.payment_stack import PaymentEnvironment
from schemas.attack import load_attack_spec
from schemas.payment import PaymentMessage

BACKEND_ROOT = Path(__file__).resolve().parent
SPECS_DIR = BACKEND_ROOT / "attack_specs"

AMBIENT_INTERVAL_S = 2.0
MAX_CAMPAIGN_SIZE = 200
MAX_SLEEP_S = 3.0

_SAFE_SPEC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
NODE_TYPE = {"C": "customer", "D": "device", "I": "ip", "M": "merchant"}

DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "https://adversarial-payment-arena.vercel.app",
]


class ArenaStack:
    """One simulated issuer world, defense engine and running cost ledger."""

    def __init__(self) -> None:
        self.env = PaymentEnvironment(n_customers=1000, seed=42)
        self.engine = DecisionEngine(
            environment=self.env,
            scorer=VelocityScorer(DEFAULT_MODEL_PATH),
            novelty=NoveltyDetector(),
        )
        self.costs = DecisionEngine._new_cost_totals()
        self.campaign_lock = asyncio.Lock()
        self._pending_graph_edges: set[tuple[str, str]] = set()

    def apply_cost(self, record: dict, truth: str) -> None:
        self.engine.apply_to_running_totals(self.costs, record, truth)

    def cost_summary(self) -> dict:
        return self.engine.summarize_totals(self.costs)

    def queue_graph_edges(self, pairs) -> None:
        for a, b in pairs:
            self._pending_graph_edges.add((min(a, b), max(a, b)))

    def drain_graph_edges(self) -> list[tuple[str, str]]:
        fresh = sorted(self._pending_graph_edges)
        self._pending_graph_edges.clear()
        return fresh


def _node_obj(graph, node_id: str) -> dict:
    del graph  # retained in the signature for compatibility with existing callers
    return {
        "id": node_id,
        "type": NODE_TYPE.get(node_id.split(":", 1)[0], "unknown"),
    }


def _edge_objs(nx_graph, pairs) -> list[dict]:
    output = []
    for a, b in pairs:
        weight = nx_graph[a][b]["weight"] if nx_graph.has_edge(a, b) else 1
        output.append({"source": a, "target": b, "weight": weight})
    return output


def resolve_attack_path(name: str) -> Path:
    """Resolve a spec name strictly inside ``attack_specs/``."""
    name = name.strip().removesuffix(".yaml")
    if not _SAFE_SPEC_NAME.match(name):
        raise HTTPException(status_code=400, detail=f"unsafe spec name: {name!r}")

    exact = SPECS_DIR / f"{name}.yaml"
    if exact.exists():
        return exact

    matches = [path for path in SPECS_DIR.glob("*.yaml") if path.stem.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    raise HTTPException(status_code=404, detail=f"no unique attack spec for {name!r}")


async def pump_campaign(
    stack: ArenaStack,
    agent: AttackerAgent,
    campaign_size: int,
    emit,
) -> None:
    """Drive one campaign and merge decision/cost/graph events into its stream."""
    try:
        async for event in agent.run_campaign(campaign_size=campaign_size):
            await emit(event)
            if event["type"] != "payload_generated":
                continue

            payload_wire: dict = event["data"]
            record = stack.engine.decide(PaymentMessage.model_validate(payload_wire))
            truth = "attack" if payload_wire.get("stolen_resource") else "legit"
            stack.apply_cost(record, truth)
            stack.queue_graph_edges(record.get("graph_new_edges", ()))

            await emit({
                "type": "defense_decision",
                "txn_index": event.get("txn_index"),
                "decision": record["decision"],
                "reasons": record["reasons"],
                "scores": record["scores"],
                "amount": record["amount"],
            })
            await emit({"type": "cost_update", **stack.cost_summary()})

            fresh = stack.drain_graph_edges()
            if fresh:
                graph = stack.engine.graph.g
                endpoint_ids = sorted({node for pair in fresh for node in pair})
                await emit({
                    "type": "graph_update",
                    "nodes": [_node_obj(graph, node) for node in endpoint_ids],
                    "edges": _edge_objs(graph, fresh),
                })
    except WebSocketDisconnect:
        raise
    except Exception as exc:  # surface engine/provider failures to the client
        import traceback

        await emit({
            "type": "error",
            "data": f"Campaign error: {exc}\n{traceback.format_exc()[-600:]}",
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = ArenaStack()
    app.state.stack = stack
    print(
        f"[arena] models loaded: xgb={stack.engine.scorer.model_source} "
        f"iforest={stack.engine.novelty.model_source} "
        f"features={stack.engine.scorer.extractor.backend}"
    )
    ambient = asyncio.create_task(_ambient_legit_drip(stack))
    try:
        yield
    finally:
        ambient.cancel()
        with suppress(asyncio.CancelledError):
            await ambient


app = FastAPI(title="Adversarial Payment Arena", version="0.7.0", lifespan=lifespan)


def _origins() -> list[str]:
    extra = os.getenv("ALLOWED_ORIGINS", "")
    return DEFAULT_ORIGINS + [item.strip() for item in extra.split(",") if item.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(evidence_router)


async def _ambient_legit_drip(stack: ArenaStack) -> None:
    rng = random.Random(2026)
    fake = Faker()
    Faker.seed(2026)

    while True:
        await asyncio.sleep(AMBIENT_INTERVAL_S)
        try:
            message = build_legit_payload(stack.env, rng, fake)
            result = stack.env.ingest(message)
            if result["accepted"]:
                record = stack.engine.decide(message)
                stack.apply_cost(record, "legit")
                stack.queue_graph_edges(record.get("graph_new_edges", ()))
        except Exception as exc:
            print(f"[ambient] skipped txn: {exc!r}")


class LoadAttackRequest(BaseModel):
    filename: str = Field(description="e.g. 'attack_1' or full yaml filename")


@app.get("/api/health")
async def health():
    stack: ArenaStack = app.state.stack
    return {
        "status": "ok",
        "models": {
            "xgb": stack.engine.scorer.model_source,
            "iforest": stack.engine.novelty.model_source,
        },
        "feature_backend": stack.engine.scorer.extractor.backend,
        "events_seen": len(stack.env.event_stream),
        "costs": stack.cost_summary(),
    }


@app.get("/api/merchants")
async def merchants():
    stack: ArenaStack = app.state.stack
    return [merchant.model_dump() for merchant in stack.env.merchant_registry.values()]


@app.get("/api/attacks")
async def attacks():
    return {"attacks": sorted(path.stem for path in SPECS_DIR.glob("*.yaml"))}


@app.post("/api/load_attack")
async def load_attack(req: LoadAttackRequest):
    path = resolve_attack_path(req.filename)
    spec = load_attack_spec(path)
    return {"filename": path.name, "spec": spec.model_dump(mode="json")}


@app.get("/api/graph/snapshot")
async def graph_snapshot():
    stack: ArenaStack = app.state.stack
    graph = stack.engine.graph.g
    nodes = [_node_obj(graph, node) for node in graph.nodes()]
    edges = [
        {"source": a, "target": b, "weight": data.get("weight", 1)}
        for a, b, data in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


async def _send_ws_usage_error(ws: WebSocket, detail: str | None = None) -> None:
    await ws.send_json({
        "type": "error",
        "data": detail
        or 'send {"type":"start_campaign","attack_file":"...","campaign_size":N}',
    })


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    stack: ArenaStack = app.state.stack

    async def emit(event: dict) -> None:
        await ws.send_json(event)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _send_ws_usage_error(ws, "messages must be JSON")
                continue

            if not isinstance(message, dict):
                await _send_ws_usage_error(ws, "messages must be JSON objects")
                continue

            if message.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                continue

            # Both fields are required. The old `and` condition accidentally
            # accepted arbitrary message types whenever attack_file existed.
            if message.get("type") != "start_campaign" or "attack_file" not in message:
                await _send_ws_usage_error(ws)
                continue

            try:
                path = resolve_attack_path(str(message.get("attack_file", "")))
                spec = load_attack_spec(path)
            except HTTPException as exc:
                await _send_ws_usage_error(ws, str(exc.detail))
                continue

            try:
                raw_size = message.get("campaign_size", 20)
                raw_sleep = message.get("sleep_s", RATE_LIMIT_SLEEP_S)
                if isinstance(raw_size, bool) or isinstance(raw_sleep, bool):
                    raise ValueError("boolean values are not valid numeric controls")
                size = int(raw_size)
                sleep_s = float(raw_sleep)
                if not math.isfinite(sleep_s) or sleep_s < 0:
                    raise ValueError("sleep_s must be a finite non-negative number")
            except (TypeError, ValueError, OverflowError) as exc:
                await _send_ws_usage_error(ws, f"invalid campaign controls: {exc}")
                continue

            size = max(1, min(size, MAX_CAMPAIGN_SIZE))
            sleep_s = min(sleep_s, MAX_SLEEP_S)

            if stack.campaign_lock.locked():
                await _send_ws_usage_error(ws, "another campaign is running")
                continue

            async with stack.campaign_lock:
                agent = AttackerAgent(spec, stack.env, sleep_between_calls_s=sleep_s)
                await ws.send_json({
                    "type": "campaign_accepted",
                    "spec": spec.spec_id,
                    "size": size,
                    "attack_file": path.name,
                })
                await pump_campaign(stack, agent, size, emit)
                await ws.send_json({"type": "cost_update", **stack.cost_summary()})
    except WebSocketDisconnect:
        return


def _mount_static_ui(application: FastAPI) -> None:
    configured = os.getenv("SERVE_STATIC_DIR", "").strip()
    if not configured:
        return

    dist = Path(configured)
    if not dist.is_dir():
        print(f"[arena] SERVE_STATIC_DIR={configured!r} is not a directory -- UI not mounted")
        return

    from fastapi.staticfiles import StaticFiles

    application.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")
    print(f"[arena] serving static UI from {dist}")


_mount_static_ui(app)
