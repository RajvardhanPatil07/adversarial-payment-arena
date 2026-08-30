"""FastAPI control plane for the Adversarial Payment Arena.

REST exposes health, registry, attack-spec, graph, emerging-threat, shadow-model
and evidence endpoints. The WebSocket endpoint streams campaign events,
decisions, defender feedback, threat fingerprints, shadow comparisons, cost
updates and incremental graph changes.
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
from defense.challenger import ShadowChallenger
from defense.decision import DecisionEngine
from defense.novelty import NoveltyDetector
from defense.realtime import DEFAULT_MODEL_PATH, VelocityScorer
from defense.threat_miner import ThreatMiner
from environment.payment_stack import PaymentEnvironment
from evidence.containment import CampaignContainment
from schemas.attack import load_attack_spec
from schemas.payment import PaymentMessage

BACKEND_ROOT = Path(__file__).resolve().parent
SPECS_DIR = BACKEND_ROOT / "attack_specs"

AMBIENT_INTERVAL_S = 2.0
MAX_CAMPAIGN_SIZE = 200
MAX_SLEEP_S = 3.0
FEEDBACK_MODES = {"black", "gray", "white"}

_SAFE_SPEC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
NODE_TYPE = {"C": "customer", "D": "device", "I": "ip", "M": "merchant"}

DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "https://adversarial-payment-arena.vercel.app",
]


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class ArenaStack:
    """One simulated issuer world, defense engine and running evidence state."""

    def __init__(self) -> None:
        self.env = PaymentEnvironment(
            n_customers=1000,
            seed=42,
            event_stream_maxlen=50_000,
            gate_rejects_maxlen=10_000,
        )
        self.engine = DecisionEngine(
            environment=self.env,
            scorer=VelocityScorer(DEFAULT_MODEL_PATH),
            novelty=NoveltyDetector(),
        )
        self.threat_miner = ThreatMiner()
        self.challenger = ShadowChallenger()
        self.challenger_enabled = _env_enabled("ENABLE_SHADOW_CHALLENGER", True)
        if not self.challenger_enabled:
            self.challenger.training_metrics = {"status": "disabled"}
        self.last_containment: dict | None = None
        self.costs = DecisionEngine._new_cost_totals()
        self.campaign_lock = asyncio.Lock()
        self._pending_graph_edges: set[tuple[str, str]] = set()

    def apply_cost(self, record: dict, truth: str) -> None:
        self.engine.apply_to_running_totals(self.costs, record, truth)

    def cost_summary(self) -> dict:
        return self.engine.summarize_totals(self.costs)

    def observe_threat(self, record: dict) -> dict | None:
        return self.threat_miner.observe(record)

    def compare_shadow(self, record: dict, truth: str) -> dict | None:
        if not self.challenger_enabled:
            return None
        return self.challenger.observe(record, truth)

    def queue_graph_edges(self, pairs) -> None:
        for a, b in pairs:
            self._pending_graph_edges.add((min(a, b), max(a, b)))

    def drain_graph_edges(self) -> list[tuple[str, str]]:
        fresh = sorted(self._pending_graph_edges)
        self._pending_graph_edges.clear()
        return fresh


def _node_obj(graph, node_id: str) -> dict:
    del graph
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


def _coarse_signal_families(record: dict) -> list[str]:
    families: list[str] = []
    reasons = record.get("reasons", ())
    if any(str(reason).startswith("ring_detected") for reason in reasons):
        families.append("shared-infrastructure topology")
    if any(str(reason).startswith("velocity>") for reason in reasons):
        families.append("behavioural velocity")
    if any("novelty" in str(reason) for reason in reasons):
        families.append("out-of-distribution behaviour")
    return families or ["no dominant defense signal"]


def _defender_feedback(record: dict, txn_index: int | None, mode: str) -> dict:
    """Build one feedback turn consumed before the next adversarial move."""
    decision = str(record["decision"])
    families = _coarse_signal_families(record)
    prefix = f"Defender outcome for transaction {txn_index or '?'}: {decision}."

    if mode == "black":
        prompt = (
            f"{prefix} Treat only that outcome as observable. Adapt the next "
            "synthetic move if useful while staying inside the attack specification."
        )
    elif mode == "white":
        prompt = (
            f"{prefix} Lab-only white-box telemetry: reasons={record.get('reasons', [])}; "
            f"scores={record.get('scores', {})}. Adapt the next synthetic move while "
            "remaining schema-valid, economically plausible and inside the declared attack spec."
        )
    else:
        prompt = (
            f"{prefix} Observable signal family: {', '.join(families)}. Adapt the next "
            "synthetic move if useful, but do not leave the declared attack specification."
        )

    return {
        "mode": mode,
        "decision": decision,
        "signal_families": families,
        "prompt": prompt,
    }


async def pump_campaign(
    stack: ArenaStack,
    agent: AttackerAgent,
    campaign_size: int,
    emit,
    feedback_mode: str = "gray",
) -> None:
    """Drive one adaptive campaign and attach discovery/containment evidence."""
    containment = CampaignContainment(agent.spec.spec_id)
    try:
        async for original_event in agent.run_campaign(campaign_size=campaign_size):
            event = original_event
            if event["type"] == "campaign_summary":
                summary = containment.summary()
                stack.last_containment = summary
                event = {
                    **event,
                    "data": {**event["data"], "containment": summary},
                }
                await emit(event)
                await emit({"type": "containment_summary", "data": summary})
                continue

            await emit(event)
            if event["type"] != "payload_generated":
                continue

            payload_wire: dict = event["data"]
            record = stack.engine.decide(PaymentMessage.model_validate(payload_wire))
            truth = "attack" if payload_wire.get("stolen_resource") else "legit"
            txn_index = event.get("txn_index")
            stack.apply_cost(record, truth)
            stack.queue_graph_edges(record.get("graph_new_edges", ()))
            if truth != "legit":
                containment.observe(record, txn_index)

            await emit({
                "type": "defense_decision",
                "txn_index": txn_index,
                "decision": record["decision"],
                "reasons": record["reasons"],
                "scores": record["scores"],
                "amount": record["amount"],
            })

            fingerprint = stack.observe_threat(record)
            if fingerprint is not None:
                if truth != "legit":
                    containment.mark_emerging_threat(fingerprint, txn_index)
                await emit({
                    "type": "emerging_threat",
                    "txn_index": txn_index,
                    "data": fingerprint,
                })

            comparison = stack.compare_shadow(record, truth)
            if comparison is not None:
                await emit({
                    "type": "shadow_comparison",
                    "txn_index": txn_index,
                    "data": comparison,
                })

            feedback = _defender_feedback(record, txn_index, feedback_mode)
            agent._user_say(feedback["prompt"])
            await emit({
                "type": "defender_feedback",
                "txn_index": txn_index,
                "mode": feedback["mode"],
                "decision": feedback["decision"],
                "signal_families": feedback["signal_families"],
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
    except Exception as exc:
        import traceback

        await emit({
            "type": "error",
            "data": f"Campaign error: {exc}\n{traceback.format_exc()[-600:]}",
        })


async def _train_shadow_challenger(stack: ArenaStack) -> None:
    if not stack.challenger_enabled:
        return
    stack.challenger.training_metrics = {"status": "training", "name": stack.challenger.name}
    try:
        metrics = await asyncio.to_thread(stack.challenger.train_default)
        print(f"[arena] shadow challenger ready: {metrics}")
    except Exception as exc:
        stack.challenger.ready = False
        stack.challenger.training_metrics = {
            "status": "error",
            "name": stack.challenger.name,
            "error": repr(exc),
        }
        print(f"[arena] shadow challenger training failed: {exc!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = ArenaStack()
    app.state.stack = stack
    print(
        f"[arena] models loaded: xgb={stack.engine.scorer.model_source} "
        f"iforest={stack.engine.novelty.model_source} "
        f"features={stack.engine.scorer.extractor.backend} "
        f"graph={stack.engine.graph.backend}"
    )
    ambient = asyncio.create_task(_ambient_legit_drip(stack))
    challenger_task = asyncio.create_task(_train_shadow_challenger(stack))
    try:
        yield
    finally:
        ambient.cancel()
        with suppress(asyncio.CancelledError):
            await ambient
        if not challenger_task.done():
            challenger_task.cancel()
        with suppress(asyncio.CancelledError):
            await challenger_task


app = FastAPI(title="Adversarial Payment Arena", version="1.0.0", lifespan=lifespan)


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
                stack.observe_threat(record)
                stack.compare_shadow(record, "legit")
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
        "graph_backend": stack.engine.graph.backend,
        "graph_state_sizes": stack.engine.graph.risk_state_sizes(),
        "events_seen": stack.env.events_seen_total,
        "threat_miner": stack.threat_miner.diagnostics(),
        "challenger": stack.challenger.snapshot(),
        "costs": stack.cost_summary(),
    }


@app.get("/api/merchants")
async def merchants():
    stack: ArenaStack = app.state.stack
    return [merchant.model_dump() for merchant in stack.env.merchant_registry.values()]


@app.get("/api/attacks")
async def attacks():
    return {"attacks": sorted(path.stem for path in SPECS_DIR.glob("*.yaml"))}


@app.get("/api/threats")
async def threats(include_candidates: bool = False):
    stack: ArenaStack = app.state.stack
    return {
        "diagnostics": stack.threat_miner.diagnostics(),
        "threats": stack.threat_miner.snapshot(include_candidates=include_candidates),
    }


@app.get("/api/challenger")
async def challenger_status():
    stack: ArenaStack = app.state.stack
    return stack.challenger.snapshot()


@app.get("/api/containment/last")
async def last_containment():
    stack: ArenaStack = app.state.stack
    return {"containment": stack.last_containment}


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
                feedback_mode = str(message.get("feedback_mode", "gray")).strip().lower()
                if feedback_mode not in FEEDBACK_MODES:
                    raise ValueError(
                        f"feedback_mode must be one of {sorted(FEEDBACK_MODES)}"
                    )
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
                    "feedback_mode": feedback_mode,
                    "challenger_shadow_only": True,
                })
                await pump_campaign(
                    stack,
                    agent,
                    size,
                    emit,
                    feedback_mode=feedback_mode,
                )
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
