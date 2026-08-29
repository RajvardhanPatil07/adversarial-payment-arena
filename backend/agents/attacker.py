"""
LLM red-team attacker agent — the "Generate" phase of the closed loop.

The LLM (an OpenRouter model via an OpenAI-compatible client — a free
reasoning model by default, overridable via OPENROUTER_MODEL; stealth/ox-alpha
when that slug is served) does NOT classify anything: it *plays the fraudster*. It plans a campaign (PLANNER
phase), then emits one PaymentMessage payload per move via OpenAI-compatible
STRUCTURED OUTPUTS (json_schema derived from the PaymentMessage Pydantic
model). Anything that fails local Pydantic validation is treated as a
malformed move and coached back — free-text JSON never reaches the stack.

Loop contract (per transaction slot):
  generate -> local schema validation -> PlausibilityGate via env.ingest()
    rejected? -> feed reason code back as a user message, retry (<= RETRY_LIMIT)
    accepted? -> yield agent_thought / payload_generated / plausibility_check

Rate limiting: spec calls for time.sleep(2) between LLM calls. We honor the
2s spacing with asyncio.sleep so we never block the event loop that will
later stream these events to the dashboard over WebSockets.

Offline mode: with no OPENROUTER_API_KEY (or an injected client) the agent
falls back to a deterministic heuristic fraudster that speaks the exact same
protocol. This keeps tests/CI green without network and gives judges a
no-key demo path. Live traffic always prefers OpenRouter.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections import Counter
from typing import AsyncIterator, Iterator, Optional

from pydantic import ValidationError

from environment.payment_stack import PaymentEnvironment, Merchant
from schemas.attack import AttackSpec
from schemas.payment import (
    RESOURCE_COST_TABLE_USD,
    PaymentMessage,
    PosEntryMode,
    ThreeDSStatus,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# The project brief names stealth/ox-alpha. That slug is not always served
# to every account on OpenRouter; the runtime override is OPENROUTER_MODEL.
# A reasoning-class free model keeps the "LLM-as-adversary" experience
# intact when stealth endpoints rotate away.
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

# Spec: "Agent retries up to 3 times" on rejection.
RETRY_LIMIT = 3
# Spec: 2s spacing between LLM calls (rate limits).
RATE_LIMIT_SLEEP_S = 2.0
# Rolling conversational window beyond system+planner (keeps latency sane).
_WINDOW = 14


class MalformedMoveError(Exception):
    """LLM output failed local Pydantic validation — retryable."""


class LLMProviderError(Exception):
    """OpenRouter / OpenAI provider error (bad model, auth, rate-limit) — not retryable by gate coaching."""


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_TEMPLATE = """You are a fraudster. Your goal: {attack_name}. \
Preconditions: {preconditions}. Economic model: cost ${cost}, expected payoff \
${payoff}, breakeven at {breakeven} transactions. You have a tool \
`submit_payment` that accepts a PaymentMessage JSON payload. Generate payloads \
that will pass the plausibility gate and evade detection. Think step by step \
about your campaign."""

_PLANNER_INSTRUCTION = """Before your first move, emit your campaign strategy:
which customer profiles you will target and why, your amount band per move,
your timing/velocity cadence, and how you will keep each payload coherent
with the merchant registry. Be concrete. One paragraph."""

_KNOWN_UNIVERSE_TEMPLATE = """
OPERATIONAL BRIEF (recon):
{merchant_table}
Customer id space: CUST_0000..CUST_{max_cust:04d} (4-digit zero-padded).
Device ids look like DEV_ plus 10 hex chars; victims' bound devices are best
but a "new" device is not automatically fatal.
Known issuer screens: {evasion_notes}
Every move must be returned through structured output: {{"reasoning": str,
"payload": <PaymentMessage>}}. Wire field for 3DS is exactly "3ds_status".
"""


def render_system_prompt(spec: AttackSpec, env: PaymentEnvironment) -> str:
    """Fill the mandated template from the AttackSpec, then append recon."""
    base = SYSTEM_PROMPT_TEMPLATE.format(
        attack_name=spec.attack_name,
        preconditions=spec.preconditions,
        cost=spec.economic_model.acquisition_cost_usd,
        payoff=spec.economic_model.expected_payoff_usd,
        breakeven=spec.economic_model.breakeven_txns,
    )
    rows = "\n".join(
        f"- {m.merchant_id}: name={m.name!r} mcc={m.mcc} geo={m.country} "
        f"vertical={m.category} online={m.is_online}"
        for m in env.merchant_registry.values()
    )
    return (
        base
        + _KNOWN_UNIVERSE_TEMPLATE.format(
            merchant_table=rows,
            max_cust=len(env.customers) - 1,
            evasion_notes=spec.evasion_notes or "(none published)",
        )
    )


# --------------------------------------------------------------------------- #
# Structured output schema (derived from the Pydantic contract)
# --------------------------------------------------------------------------- #


def build_move_schema() -> dict:
    """OpenAI-compatible response_format forcing {reasoning, payload} where
    `payload` IS the PaymentMessage JSON Schema (wire aliases, strict).

    Pydantic omits defaulted fields from `required`, but JSON-schema strict
    mode requires every property to be required — patch that in explicitly,
    hoisting $defs so nested enum $refs resolve at the wrapper root.
    """
    payload_schema = PaymentMessage.model_json_schema(by_alias=True)
    defs = payload_schema.pop("$defs", {})
    payload_schema["required"] = sorted(payload_schema["properties"].keys())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "fraudster_move",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "payload": payload_schema,
                },
                "required": ["reasoning", "payload"],
                "additionalProperties": False,
                "$defs": defs,
            },
        },
    }


def strip_code_fences(text: str) -> str:
    """Some OpenRouter models wrap JSON in ```json fences despite the schema."""
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    return fence.group(1) if fence else text


# --------------------------------------------------------------------------- #
# Offline deterministic fallback (same protocol as the live client)
# --------------------------------------------------------------------------- #


class _OpenRouterClientAdapter:
    """Uniform .create(**kwargs) surface over the OpenAI SDK's nested path,
    so the agent (and the offline shim) share one call convention."""

    def __init__(self, sdk_client: object) -> None:
        self._sdk = sdk_client

    def create(self, **kwargs: object) -> object:
        return self._sdk.chat.completions.create(**kwargs)


class _OfflineHeuristicAttacker:
    """
    Deterministic stand-in for the LLM attacker used in CI/no-key demos.

    Speaks the same request/response protocol as chat.completions.create
    (messages in, JSON string out) and behaves like a competent fraudster:
    coherent-by-construction payloads inside the spec's envelope, one
    deliberate first-shot mistake per campaign so the rejection-feedback
    retry path is exercised end-to-end.
    """

    def __init__(self, spec: AttackSpec, env: PaymentEnvironment, seed: int = 99) -> None:
        self.spec = spec
        self.env = env
        self.rng = random.Random(seed)
        self.calls = 0
        # A mule-ring campaign must actually COLLUDE, or the graph layer has
        # nothing to catch. For ATTACK_2 we pin a fixed set of >=3 mule
        # customers onto ONE shared device + ONE egress IP, drawn round-robin
        # below, so >=3 distinct customers land on the same directly-linked
        # infra node and the ring rule (graph.py:96,102) fires deterministically
        # in a no-key demo. Every other spec keeps fresh-identity-per-payload.
        self._ring: dict | None = None
        self._ring_i = 0
        if spec.spec_id == "ATTACK_2_SYNTHETIC_MULE_RING":
            pool = sorted(self.env.customers.keys())
            self._ring = {
                "customers": self.rng.sample(pool, min(4, len(pool))),
                "device": f"DEV_{self.rng.getrandbits(40):010x}",
                "ip": f"{self.rng.randint(1, 223)}.{self.rng.randint(0, 255)}."
                      f"{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}",
            }

    # -- protocol shim -----------------------------------------------------
    # Call sequence is deterministic: call 1 -> planner, call 2 -> first move
    # WITH a deliberate MCC fault, calls >= 3 -> clean moves. This exercises
    # the rejection-feedback retry loop on every offline run.

    def create(self, model: str, messages: list[dict], response_format: dict | None = None, **_: object) -> object:
        self.calls += 1
        if self.calls == 1:
            content = self._plan_text()
        elif self.calls == 2:
            move = self._payload()
            move["payload"]["mcc"] = 7994  # deliberate incoherence vs registry
            content = json.dumps(move)
        else:
            content = json.dumps(self._payload())
        return _FakeResponse(content)

    # -- behavior ----------------------------------------------------------

    def _plan_text(self) -> str:
        cons = self.spec.constraints
        return (
            f"Target recently-active customers in verticals {cons.target_verticals}; "
            f"monetize each voice-reset session within minutes while the MFA reset "
            f"is still warm. Amounts ${cons.min_amount_usd:.0f}-${cons.max_amount_usd:.0f}: "
            f"above the ${self.spec.economic_model.acquisition_cost_usd:.0f} tooling floor, below "
            f"manual-review tickets. Rail {cons.pos_entry_modes[0].value} with 3DS="
            f"{cons.preferred_three_ds[0].value} first; pace ~1 txn/victim/hour."
        )

    def _pick_merchant(self) -> Merchant:
        cands = [
            m for m in self.env.merchant_registry.values()
            if m.category in self.spec.constraints.target_verticals
        ] or list(self.env.merchant_registry.values())
        return self.rng.choice(cands)

    def _payload(self) -> dict:
        cons = self.spec.constraints
        customer_id = f"CUST_{self.rng.randrange(len(self.env.customers)):04d}"
        merchant = self._pick_merchant()
        mode = cons.pos_entry_modes[0]
        tds = cons.preferred_three_ds[0]
        if mode == PosEntryMode.CONTACTLESS:
            tds = ThreeDSStatus.N  # physics: tap never runs 3DS
        lo, hi = cons.amount_band
        floor = (
            0.0 if cons.stolen_resource is None
            else RESOURCE_COST_TABLE_USD[cons.stolen_resource]
        )
        amount = round(min(max(self.rng.uniform(lo, hi), floor + 5), hi), 2)
        device = f"DEV_{self.rng.getrandbits(40):010x}"
        ip_address = (
            f"{self.rng.randint(1,223)}.{self.rng.randint(0,255)}."
            f"{self.rng.randint(0,255)}.{self.rng.randint(1,254)}"
        )
        if self._ring is not None:
            # Mule ring: override the fresh identity with a fixed mule drawn
            # round-robin over the shared device + egress IP, so >=3 distinct
            # customers collude on one infra node and graph.py's ring rule
            # fires. The RNG draws above are kept intact so non-ring specs stay
            # byte-for-byte identical.
            customer_id = self._ring["customers"][self._ring_i % len(self._ring["customers"])]
            self._ring_i += 1
            device = self._ring["device"]
            ip_address = self._ring["ip"]
        reasoning = (
            f"Victim session is fresh post-MFA-reset; monetizing via {mode.value} at "
            f"{merchant.name} (mcc {merchant.mcc}, geo {merchant.country}) for ${amount}. "
            f"3DS={tds.value} avoids challenge friction; amount clears the "
            f"{cons.stolen_resource.value if cons.stolen_resource else 'zero'}-cost floor."
        )
        return {
            "reasoning": reasoning,
            "payload": {
                "transaction_id": f"{self.rng.getrandbits(64):016X}",
                "customer_id": customer_id,
                "merchant_id": merchant.merchant_id,
                "mcc": merchant.mcc,
                "amount": amount,
                "currency": "USD",
                "pos_entry_mode": mode.value,
                "3ds_status": tds.value,
                "ip_address": ip_address,
                "ip_country": merchant.country,
                "device_id": device,
                "stolen_resource": cons.stolen_resource.value if cons.stolen_resource else None,
            },
        }


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #


class AttackerAgent:
    """
    Drives one fraud campaign against the simulated stack.

    Usage (live):
        agent = AttackerAgent(spec, env)
        async for event in agent.run_campaign(campaign_size=50): ...

    Events yielded (dashboard protocol, v1):
      campaign_start      {spec_id, size}
      agent_thought       {role: PLANNER|OPERATOR, data}
      payload_generated   {data: payload_wire, txn_index}
      plausibility_check  {data: {accepted, reason, risk_flags}, txn_index}
      system_feedback     {data: coach-back message, txn_index}   (on rejects)
      campaign_summary    {stats...}
    """

    RETRY_LIMIT = RETRY_LIMIT
    RATE_LIMIT_SLEEP_S = RATE_LIMIT_SLEEP_S

    def __init__(
        self,
        spec: AttackSpec,
        environment: PaymentEnvironment,
        client: Optional[object] = None,
        model: Optional[str] = None,
        sleep_between_calls_s: float = RATE_LIMIT_SLEEP_S,
        seed: int = 99,
    ) -> None:
        self.spec = spec
        self.env = environment
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.sleep_s = sleep_between_calls_s
        self.client = client if client is not None else self._make_openrouter_client()
        if self.client is None:  # no key -> deterministic offline fraudster
            self.client = _OfflineHeuristicAttacker(spec, environment, seed=seed)
        self._move_schema = build_move_schema()

        self.messages: list[dict] = [
            {"role": "system", "content": render_system_prompt(spec, environment)}
        ]
        self.stats: dict = {
            "llm_calls": 0,
            "txn_slots": 0,
            "accepted": 0,
            "attempts": 0,
            "gate_rejects": Counter(),
            "malformed": 0,
            "gross_value_usd": 0.0,
        }

    # ------------------------------------------------------------------ #
    # Client plumbing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_openrouter_client() -> Optional[object]:
        """OpenAI SDK pointed at OpenRouter; None when unkeyed (offline mode)."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # lazy: offline environments need no dep
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai (or unset OPENROUTER_API_KEY)") from exc
        return _OpenRouterClientAdapter(OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        ))

    async def _complete(self, response_format: Optional[dict]) -> str:
        """One LLM turn. Sync SDK call pushed off-loop via asyncio.to_thread."""
        kwargs: dict = {"model": self.model, "messages": list(self._window())}
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = await asyncio.to_thread(self.client.create, **kwargs)
        except Exception as exc:  # noqa: BLE001 — provider surface is OpenAI SDK (NotFoundError, RateLimitError, etc.)
            raise LLMProviderError(f"LLM provider error for model {self.model!r}: {exc}") from exc
        self.stats["llm_calls"] += 1
        msg = resp.choices[0].message
        # Reasoning models (e.g., Nemotron) may put the JSON in `reasoning`
        # when `content` is empty. Prefer content, fall back to reasoning.
        content = (getattr(msg, "content", None) or getattr(msg, "reasoning", None) or "")
        await asyncio.sleep(self.sleep_s)  # rate-limit spacing (async time.sleep)
        return content

    def _window(self) -> Iterator[dict]:
        yield self.messages[0]  # system prompt never ages out
        yield from self.messages[-_WINDOW:]

    def _user_say(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def _assistant_say(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    def _parse_move(self, raw: str) -> tuple[str, PaymentMessage]:
        """Structured output -> (reasoning, PaymentMessage). Local Pydantic
        validation is the real gatekeeper; provider leniency can't bypass it."""
        try:
            cleaned = strip_code_fences(raw)
            data = json.loads(cleaned)
            # Some providers wrap the JSON in extra text — try to extract the
            # outermost object containing "payload" if the first parse lacks it.
            if "payload" not in data:
                # Attempt regex extraction of a JSON object with a payload key
                m = re.search(r"\{[^{}]*\"payload\"\s*:.*\}", cleaned, flags=re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
            reasoning = str(data.get("reasoning", ""))
            payload = PaymentMessage.model_validate(data["payload"])
            return reasoning, payload
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
            self.stats["malformed"] += 1
            # Include a preview of the raw output for debugging retries
            preview = (raw or "")[:400].replace("\n", " ")
            raise MalformedMoveError(f"structured-output violation: {exc} | raw preview: {preview!r}") from exc

    @staticmethod
    def _coach_back(reason: str) -> str:
        return (
            f"Your last payload was rejected by the plausibility gate: {reason}. "
            "Fix the failure and resubmit."
        ) if reason != "malformed_payload" else (
            "Your last move was not valid structured output for PaymentMessage. "
            'Return exactly {"reasoning": str, "payload": {...}} with wire alias '
            '"3ds_status". Resubmit.'
        )

    # ------------------------------------------------------------------ #
    # Campaign loop
    # ------------------------------------------------------------------ #

    async def run_campaign(self, campaign_size: int = 50) -> AsyncIterator[dict]:
        yield {"type": "campaign_start", "data": {"spec_id": self.spec.spec_id, "size": campaign_size}}

        # ---- PLANNER phase: strategy before any payload ------------------
        plan_instruction = _PLANNER_INSTRUCTION
        self._user_say(plan_instruction)
        try:
            plan_text = await self._complete(response_format=None)
        except LLMProviderError as exc:
            yield {"type": "error", "data": str(exc)}
            # Graceful end: campaign summary reflects zero progress so the
            # dashboard can surface the fix ("set OPENROUTER_MODEL to a valid slug").
            yield {"type": "campaign_summary", "data": {
                "spec_id": self.spec.spec_id,
                "txn_slots": 0,
                "accepted": 0,
                "accept_rate": 0.0,
                "attempts": 0,
                "gate_rejects": {},
                "malformed": 0,
                "llm_calls": self.stats["llm_calls"],
                "gross_value_usd": 0.0,
                "net_vs_tooling_usd": round(0.0 - self.spec.economic_model.acquisition_cost_usd, 2),
                "error": str(exc),
            }}
            return
        self._assistant_say(plan_text)
        yield {"type": "agent_thought", "role": "PLANNER", "data": plan_text}

        # ---- OPERATOR phase: one adversarial loop per txn slot -----------
        for idx in range(1, campaign_size + 1):
            self.stats["txn_slots"] += 1
            attempts = 0
            while attempts <= self.RETRY_LIMIT:  # initial + up to 3 retries
                attempts += 1
                self.stats["attempts"] += 1
                try:
                    raw = await self._complete(response_format=self._move_schema)
                    reasoning, payload = self._parse_move(raw)
                except LLMProviderError as exc:
                    yield {"type": "error", "data": str(exc), "txn_index": idx}
                    yield {"type": "txn_abandoned", "txn_index": idx, "data": {"reason": "llm_provider_error"}}
                    # Persistent config/rate-limit errors won't heal on retry within this slot
                    continue
                except MalformedMoveError as exc:
                    self._assistant_say(raw)
                    self._user_say(self._coach_back("malformed_payload"))
                    yield {
                        "type": "system_feedback", "txn_index": idx,
                        "data": self._coach_back("malformed_payload") + f" [{exc}]",
                    }
                    continue

                result = self.env.ingest(payload)

                if result["accepted"]:
                    self.stats["accepted"] += 1
                    self.stats["gross_value_usd"] += payload.amount
                    self._assistant_say(raw)
                    yield {"type": "agent_thought", "role": "OPERATOR", "data": reasoning, "txn_index": idx}
                    yield {"type": "payload_generated", "data": payload.to_wire(), "txn_index": idx}
                    yield {
                        "type": "plausibility_check", "txn_index": idx,
                        "data": {"accepted": True, "reason": result["reason"],
                                 "risk_flags": result["internal_event"]["risk_flags"],
                                 "attempt": attempts},
                    }
                    break

                # ---- rejected: coach the adversary, stay in the loop ------
                self.stats["gate_rejects"][result["reason"]] += 1
                self._assistant_say(raw)
                self._user_say(self._coach_back(result["reason"]))
                yield {
                    "type": "plausibility_check", "txn_index": idx,
                    "data": {"accepted": False, "reason": result["reason"], "attempt": attempts},
                }
                yield {"type": "system_feedback", "txn_index": idx,
                       "data": self._coach_back(result["reason"])}
            else:
                # exhausted retries: slot conceded (real campaigns have losses)
                yield {"type": "txn_abandoned", "txn_index": idx,
                       "data": {"reason": "retry_limit_exhausted"}}

        # ---- summary ------------------------------------------------------
        yield {"type": "campaign_summary", "data": {
            "spec_id": self.spec.spec_id,
            "txn_slots": self.stats["txn_slots"],
            "accepted": self.stats["accepted"],
            "accept_rate": round(self.stats["accepted"] / max(1, self.stats["txn_slots"]), 3),
            "attempts": self.stats["attempts"],
            "gate_rejects": dict(self.stats["gate_rejects"]),
            "malformed": self.stats["malformed"],
            "llm_calls": self.stats["llm_calls"],
            "gross_value_usd": round(self.stats["gross_value_usd"], 2),
            "net_vs_tooling_usd": round(
                self.stats["gross_value_usd"] - self.spec.economic_model.acquisition_cost_usd, 2
            ),
        }}


__all__ = ["AttackerAgent", "MalformedMoveError", "LLMProviderError", "build_move_schema", "render_system_prompt"]
