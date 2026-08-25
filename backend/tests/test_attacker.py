"""
Attacker agent tests — Step 3 exit criteria.

Mandated: load attack_1 YAML, run a 5-transaction mini-campaign offline,
assert >=3 payloads pass the plausibility gate. Plus contract checks on the
spec loader, prompt rendering, structured-output schema, and event protocol.
A live OpenRouter smoke test exists but is opt-in via RUN_LIVE=1 + API key.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from agents.attacker import (  # noqa: E402
    AttackerAgent,
    build_move_schema,
    render_system_prompt,
)
from environment.payment_stack import PaymentEnvironment  # noqa: E402
from schemas.attack import AttackSpec, load_attack_spec  # noqa: E402

SPEC_PATH = BACKEND_ROOT / "attack_specs" / "attack_1_mfa_reset_voice_clone.yaml"


@pytest.fixture(scope="module")
def spec() -> AttackSpec:
    return load_attack_spec(SPEC_PATH)


@pytest.fixture()
def env() -> PaymentEnvironment:
    return PaymentEnvironment(n_customers=1000, seed=42)


# --------------------------------------------------------------------------- #
# Taxonomy contract
# --------------------------------------------------------------------------- #


def test_attack_spec_loads_and_validates(spec: AttackSpec):
    assert spec.spec_id == "ATTACK_1_MFA_RESET_VOICE_CLONE"
    assert spec.economic_model.acquisition_cost_usd == 50      # cloned voice
    assert spec.economic_model.breakeven_txns == 3
    assert spec.constraints.stolen_resource is not None
    assert "CNP" in [m.value for m in spec.constraints.pos_entry_modes]


def test_system_prompt_filled_from_spec(spec: AttackSpec, env: PaymentEnvironment):
    prompt = render_system_prompt(spec, env)
    for needle in (
        spec.attack_name,
        "$50",            # cost interpolated into the mandated template
        "breakeven at 3", # breakeven interpolated
        "MERCH_ELEC_BESTBUYX",  # recon table present
        '"3ds_status"',   # wire alias surfaced to the model
    ):
        assert needle in prompt, f"missing {needle!r} in system prompt"


def test_move_schema_is_strict_and_aliases_wire_field():
    schema = build_move_schema()["json_schema"]["schema"]
    assert schema["properties"]["payload"]["additionalProperties"] is False
    wire_required = schema["properties"]["payload"]["required"]
    assert "3ds_status" in wire_required          # alias, not three_ds_status
    assert set(wire_required) == set(schema["properties"]["payload"]["properties"].keys())
    assert "$defs" in schema                       # hoisted for nested enum refs


# --------------------------------------------------------------------------- #
# Mandated: 5-txn mini-campaign against attack_1
# --------------------------------------------------------------------------- #


def _drain(agent: AttackerAgent, campaign_size: int) -> list[dict]:
    async def _run() -> list[dict]:
        return [evt async for evt in agent.run_campaign(campaign_size=campaign_size)]

    return asyncio.run(_run())


def test_mini_campaign_5_txns_at_least_3_pass(spec: AttackSpec, env: PaymentEnvironment):
    agent = AttackerAgent(
        spec,
        env,
        client=_offline_stub(spec, env),
        sleep_between_calls_s=0.0,   # tests don't pay the rate-limit tax
    )
    events = _drain(agent, campaign_size=5)

    checks = [e for e in events if e["type"] == "plausibility_check"]
    accepted_idx = {e["txn_index"] for e in checks if e["data"]["accepted"]}
    assert len(accepted_idx) >= 3, f"only {len(accepted_idx)}/5 passed: {checks}"

    # planner phase happened exactly once, before any payload
    planners = [i for i, e in enumerate(events) if e["type"] == "agent_thought" and e["role"] == "PLANNER"]
    operators = [i for i, e in enumerate(events) if e["type"] == "agent_thought" and e["role"] == "OPERATOR"]
    assert len(planners) == 1
    assert operators and planners[0] < operators[0]

    # per-accepted-txn event order: thought -> payload_generated -> ACCEPTED
    # plausibility_check. Rejected attempts may precede (retry loop), so the
    # final check occurrence anchors the ordering.
    for idx in accepted_idx:
        kinds = [e["type"] for e in events if e.get("txn_index") == idx]
        assert kinds.index("agent_thought") < kinds.index("payload_generated")
        assert len(kinds) - 1 - kinds[::-1].index("plausibility_check") > kinds.index("payload_generated")

    # retry loop was exercised: first move of the offline stub is a fault
    rejected = [e for e in checks if not e["data"]["accepted"]]
    assert rejected and rejected[0]["data"]["reason"] == "metadata_incoherent"

    summary = events[-1]
    assert summary["type"] == "campaign_summary"
    assert summary["data"]["accepted"] == 5


def test_campaign_state_reaches_environment(spec: AttackSpec):
    env = PaymentEnvironment(n_customers=1000, seed=7)
    agent = AttackerAgent(spec, env, client=_offline_stub(spec, env), sleep_between_calls_s=0.0)
    _drain(agent, campaign_size=4)
    # every accepted payload must be queryable in some customer ledger
    total_history = sum(len(env.get_customer_history(c)) for c in env.customers)
    assert total_history >= 4


# --------------------------------------------------------------------------- #
# Live smoke test — opt-in only (needs OPENROUTER_API_KEY + RUN_LIVE=1)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (os.getenv("OPENROUTER_API_KEY") and os.getenv("RUN_LIVE") == "1"),
    reason="live LLM test is opt-in: OPENROUTER_API_KEY + RUN_LIVE=1",
)
def test_live_openrouter_smoke(spec: AttackSpec, env: PaymentEnvironment):
    agent = AttackerAgent(spec, env, sleep_between_calls_s=1.0)
    assert not isinstance(agent.client, type(_offline_stub(spec, env))), "expected live client"
    events = _drain(agent, campaign_size=2)
    assert any(e["type"] == "campaign_summary" for e in events)


def _offline_stub(spec: AttackSpec, env: PaymentEnvironment):
    """Explicit deterministic client so tests never depend on host API keys."""
    from agents.attacker import _OfflineHeuristicAttacker
    return _OfflineHeuristicAttacker(spec, env, seed=99)
