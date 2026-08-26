"""
FastAPI backend — the arena's control tower.

REST:
  GET  /api/health          model status + uptime probe
  GET  /api/merchants       merchant registry
  GET  /api/attacks         available attack YAMLs
  POST /api/load_attack     load + validate one AttackSpec
  GET  /api/graph/snapshot  full entity graph {nodes, edges}

WebSocket /ws — the real-time fight. Client sends
  {"type": "start_campaign", "attack_file": "attack_1", "campaign_size": 50}
and receives, in order:
  * every AttackerAgent event (agent_thought / payload_generated /
    plausibility_check / system_feedback / campaign_summary),
  * a defense_decision event for each accepted payload (DecisionEngine),
  * cost_update events (running FP/FN/TP counters),
  * graph_update events when new entity edges form (incremental: new edges
    plus their endpoint nodes — keeps the xyflow canvas cheap to patch).

An ambient legit-traffic task drips cardholder transactions through the same
engine so the cost matrix has honest false positives to price and the SOC
dashboard shows mixed traffic, not just attacks.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.attacker import RATE_LIMIT_SLEEP_S, AttackerAgent
from data.legit_generator import build_legit_payload
from defense.decision import APPROVE, DECLINE, DecisionEngine
from defense.novelty import NoveltyDetector
from defense.realtime import DEFAULT_MODEL_PATH, VelocityScorer
from environment.payment_stack import PaymentEnvironment
from faker import Faker
from schemas.attack import load_attack_spec
from schemas.payment import PaymentMessage

BACKEND_ROOT = Path(__file__).resolve().parent
SPECS_DIR = BACKEND_ROOT / "attack_specs"

AMBIENT_INTERVAL_S = 2.0          # legit drip cadence
MAX_CAMPAIGN_SIZE = 200           # sanity clamp per WS session

_SAFE_SPEC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

NODE_TYPE = {"C": "customer", "D": "device", "I": "ip", "M": "merchant"}

DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "https://adversarial-payment-arena.vercel.app",
]


# --------------------------------------------------------------------------- #
# Shared stack (single world; WS sessions and ambient drip all mutate it)
# --------------------------------------------------------------------------- #


class ArenaStack:
    """One simulated issuer world + defense engine + running cost ledger."""

    def __init__(self) -> None:
        self.env = PaymentEnvironment(n_customers=1000, seed=42)
        self.engine = DecisionEngine(
            environment=self.env,
            scorer=VelocityScorer(DEFAULT_MODEL_PATH),   # loads saved model if built
            novelty=NoveltyDetector(),                   # loads saved iForest if built
        )
        self.costs = DecisionEngine._new_cost_totals()
        self.campaign_lock = asyncio.Lock()
        self._known_edges: set[tuple[str, str]] = self._all_edge_pairs()

    def apply_cost(self, record: dict, truth: str) -> None:
        self.engine.apply_to_running_totals(self.costs, record, truth)

    def cost_summary(self) -> dict:
        return self.engine.summarize_totals(self.costs)

    # ---- graph helpers ---------------------------------------------------- #

    def _all_edge_pairs(self) -> set[tuple[str, str]]:
        return {
            (min(a, b), max(a, b))
            for a, b in self.engine.graph.g.edges()
        }

    def graph_diff_and_sync(self) -> list[tuple[str, str]]:
        """Edges formed since last call (undirected canonical pairs)."""
        current = self._all_edge_pairs()
        fresh = current - self._known_edges
        self._known_edges = current
        return sorted(fresh)


def _node_obj(graph, node_id: str) -> dict:
    return {"id": node_id, "type": NODE_TYPE.get(node_id.split(":", 1)[0], "unknown")}


def _edge_objs(nx_graph, pairs) -> list[dict]:
    out = []
    for a, b in pairs:
        w = nx_graph[a][b]["weight"] if nx_graph.has_edge(a, b) else 1
        out.append({"source": a, "target": b, "weight": w})
    return out


def resolve_attack_path(name: str) -> Path:
    """Resolve 'attack_1', 'attack_1_mfa_reset_voice_clone' or a full yaml
    filename INSIDE attack_specs/. Path traversal dies here."""
    name = name.strip().removesuffix(".yaml")
    if not _SAFE_SPEC_NAME.match(name):
        raise HTTPException(status_code=400, detail=f"unsafe spec name: {name!r}")
    for cand in (SPECS_DIR / f"{name}.yaml",):
        if cand.exists():
            return cand
    matches = [p for p in SPECS_DIR.glob("*.yaml") if p.stem.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    raise HTTPException(status_code=404, detail=f"no unique attack spec for {name!r}")


# --------------------------------------------------------------------------- #
# Campaign pipeline shared by WS handler and the CLI harness
# --------------------------------------------------------------------------- #


async def pump_campaign(
    stack: ArenaStack,
    agent: AttackerAgent,
    campaign_size: int,
    emit,
) -> None:
    """
    Drive one attacker campaign through the defense stack, emitting a merged
    event stream via `emit` (an async callable taking a dict).

    Event ordering guarantee per accepted payload:
      payload_generated -> defense_decision -> cost_update [-> graph_update]
    """
    try:
        async for evt in agent.run_campaign(campaign_size=campaign_size):
            await emit(evt)

            if evt["type"] != "payload_generated":
                continue

            payload_wire: dict = evt["data"]
            record = stack.engine.decide(PaymentMessage.model_validate(payload_wire))
            truth = "attack" if payload_wire.get("stolen_resource") else "legit"
            stack.apply_cost(record, truth)

            await emit({
                "type": "defense_decision",
                "txn_index": evt.get("txn_index"),
                "decision": record["decision"],
                "reasons": record["reasons"],
                "scores": record["scores"],
                "amount": record["amount"],
            })
            await emit({"type": "cost_update", **stack.cost_summary()})

            fresh = stack.graph_diff_and_sync()
            if fresh:
                graph = stack.engine.graph.g
                endpoint_ids = sorted({n for pair in fresh for n in pair})
                await emit({
                    "type": "graph_update",
                    "nodes": [_node_obj(stack.engine.graph.g, n) for n in endpoint_ids],
                    "edges": _edge_objs(stack.engine.graph.g, fresh),
                })
    except Exception as exc:  # noqa: BLE001 — any LLM or engine error must surface to the SOC, not kill the socket
        import traceback

        await emit({"type": "error", "data": f"Campaign error: {exc}\n{traceback.format_exc()[-600:]}"})


# --------------------------------------------------------------------------- #
# App wiring
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = ArenaStack()
    app.state.stack = stack
    print(f"[arena] models loaded: xgb={stack.engine.scorer.model_source} "
          f"iforest={stack.engine.novelty.model_source}")
    ambient = asyncio.create_task(_ambient_legit_drip(stack))
    yield
    ambient.cancel()


app = FastAPI(title="Adversarial Payment Arena", version="0.5.0", lifespan=lifespan)


def _origins() -> list[str]:
    """localhost:3000 for dev + the Vercel deploy, plus anything ops adds via
    the ALLOWED_ORIGINS env var (comma-separated)."""
    extra = os.getenv("ALLOWED_ORIGINS", "")
    return DEFAULT_ORIGINS + [o.strip() for o in extra.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _ambient_legit_drip(stack: ArenaStack) -> None:
    """Background cardholders: keeps the world alive between campaigns and
    gives the cost matrix real false positives to price."""
    rng = random.Random(2026)
    fake = Faker()
    Faker.seed(2026)
    while True:
        await asyncio.sleep(AMBIENT_INTERVAL_S)
        try:
            msg = build_legit_payload(stack.env, rng, fake)
            result = stack.env.ingest(msg)
            if result["accepted"]:
                record = stack.engine.decide(msg)
                stack.apply_cost(record, "legit")
        except Exception as exc:  # never let the drip kill the server
            print(f"[ambient] skipped txn: {exc!r}")


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


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
        "events_seen": len(stack.env.event_stream),
        "costs": stack.cost_summary(),
    }


@app.get("/api/merchants")
async def merchants():
    stack: ArenaStack = app.state.stack
    return [m.model_dump() for m in stack.env.merchant_registry.values()]


@app.get("/api/attacks")
async def attacks():
    return {"attacks": sorted(p.stem for p in SPECS_DIR.glob("*.yaml"))}


@app.post("/api/load_attack")
async def load_attack(req: LoadAttackRequest):
    path = resolve_attack_path(req.filename)
    spec = load_attack_spec(path)
    return {"filename": path.name, "spec": spec.model_dump(mode="json")}


@app.get("/api/graph/snapshot")
async def graph_snapshot():
    stack: ArenaStack = app.state.stack
    g = stack.engine.graph.g
    nodes = [_node_obj(g, n) for n in g.nodes()]
    edges = [
        {"source": a, "target": b, "weight": d.get("weight", 1)}
        for a, b, d in g.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    stack: ArenaStack = app.state.stack

    async def emit(evt: dict) -> None:
        await ws.send_json(evt)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "data": "messages must be JSON"})
                continue

            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if msg.get("type") != "start_campaign" and "attack_file" not in msg:
                await ws.send_json({
                    "type": "error",
                    "data": 'send {"type":"start_campaign","attack_file":"...","campaign_size":N}',
                })
                continue

            try:
                path = resolve_attack_path(str(msg.get("attack_file", "")))
                spec = load_attack_spec(path)
            except HTTPException as exc:
                await ws.send_json({"type": "error", "data": exc.detail})
                continue

            size = max(1, min(int(msg.get("campaign_size", 20)), MAX_CAMPAIGN_SIZE))
            sleep_s = float(msg.get("sleep_s", RATE_LIMIT_SLEEP_S))

            if stack.campaign_lock.locked():
                await ws.send_json({"type": "error", "data": "another campaign is running"})
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
                # final ledger sync so the client ends on the true totals
                await ws.send_json({"type": "cost_update", **stack.cost_summary()})
    except WebSocketDisconnect:
        return
