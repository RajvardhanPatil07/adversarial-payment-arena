"""No-key attacker feedback, vector-policy and portfolio-strategy tests."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

from agents.campaign_strategy import CampaignStrategist, StrategyAction
from agents.feedback_adapter import FeedbackAwareOfflineAdapter
from agents.fraud_portfolio import PORTFOLIO, validate_portfolio
from agents.portfolio_runner import pump_portfolio_campaign
from agents.synthetic_vector_policy import SyntheticVectorPolicy
from environment.payment_stack import PaymentEnvironment
from schemas.attack import load_attack_spec
from schemas.payment import PaymentMessage

SPECS = Path(__file__).resolve().parents[1] / "attack_specs"


class _Base:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            constraints=SimpleNamespace(amount_band=(100.0, 200.0))
        )

    def create(self, **kwargs):
        del kwargs
        move = {
            "reasoning": "baseline move",
            "payload": {"amount": 150.0},
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(move)))]
        )


def _move(response) -> dict:
    return json.loads(response.choices[0].message.content)


def test_offline_adapter_reduces_ticket_after_friction():
    adapter = FeedbackAwareOfflineAdapter(_Base())
    response = adapter.create(
        response_format={"type": "json_schema"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Defender outcome for transaction 2: STEP_UP. "
                    "Observable signal family: behavioural velocity. Adapt the next synthetic move."
                ),
            }
        ],
    )
    move = _move(response)
    assert move["payload"]["amount"] == 135.0
    assert "behavioural velocity" in move["reasoning"]
    assert "reduced ticket size" in move["reasoning"]


def test_offline_adapter_reinforces_approved_strategy_within_band():
    adapter = FeedbackAwareOfflineAdapter(_Base())
    response = adapter.create(
        response_format={"type": "json_schema"},
        messages=[
            {
                "role": "user",
                "content": "Defender outcome for transaction 2: APPROVE. Adapt the next synthetic move.",
            }
        ],
    )
    move = _move(response)
    assert move["payload"]["amount"] == 157.5
    assert "reinforced" in move["reasoning"]


def test_offline_adapter_does_not_mutate_without_feedback_for_minimal_double():
    adapter = FeedbackAwareOfflineAdapter(_Base())
    response = adapter.create(
        response_format={"type": "json_schema"},
        messages=[{"role": "user", "content": "Generate a payment."}],
    )
    move = _move(response)
    assert move["payload"]["amount"] == 150.0
    assert move["reasoning"] == "baseline move"


def _base_policy_move(spec, env: PaymentEnvironment, index: int) -> dict:
    customer_id = sorted(env.customers)[index % len(env.customers)]
    customer = env.customers[customer_id]
    allowed = set(spec.constraints.target_verticals)
    merchants = [
        merchant
        for merchant in env.merchant_registry.values()
        if not allowed or merchant.category in allowed
    ]
    merchant = merchants[index % len(merchants)]
    resource = spec.constraints.stolen_resource
    return {
        "reasoning": "baseline",
        "payload": {
            "transaction_id": f"POLICY_{index:08d}",
            "customer_id": customer_id,
            "merchant_id": merchant.merchant_id,
            "mcc": merchant.mcc,
            "amount": spec.constraints.min_amount_usd,
            "currency": "USD",
            "pos_entry_mode": spec.constraints.pos_entry_modes[0].value,
            "3ds_status": spec.constraints.preferred_three_ds[0].value,
            "ip_address": f"203.0.113.{index % 253 + 1}",
            "ip_country": merchant.country,
            "device_id": customer.devices[0],
            "stolen_resource": resource.value if resource is not None else None,
        },
    }


def _policy_moves(spec_id: str, count: int = 4):
    env = PaymentEnvironment(n_customers=40, seed=991)
    spec = load_attack_spec(SPECS / PORTFOLIO[spec_id].attack_file)
    policy = SyntheticVectorPolicy(spec, env, random.Random(992))
    rows = []
    for index in range(count):
        move = _base_policy_move(spec, env, index)
        policy.apply(move)
        message = PaymentMessage.model_validate(move["payload"])
        gate = env.ingest(message)
        assert gate["accepted"], (spec_id, index, gate["reason"])
        rows.append(move["payload"])
    return env, rows


def test_portfolio_maps_all_14_specs_and_has_valid_edges():
    assert len(PORTFOLIO) == 14
    assert validate_portfolio() == []
    for profile in PORTFOLIO.values():
        assert profile.transitions
        assert profile.primitives
        assert profile.mutation_axes


def test_every_vector_policy_generates_gate_valid_synthetic_traffic():
    for spec_id in PORTFOLIO:
        _policy_moves(spec_id, count=2)


def test_vector_fingerprints_are_behaviorally_distinct():
    _, ring = _policy_moves("ATTACK_2_SYNTHETIC_MULE_RING", count=4)
    assert len({row["customer_id"] for row in ring}) >= 3
    assert len({row["device_id"] for row in ring}) == 1

    _, merchant = _policy_moves("ATTACK_3_PROMPT_INJECTED_MERCHANT", count=5)
    assert len({row["customer_id"] for row in merchant}) >= 4
    assert len({row["merchant_id"] for row in merchant}) == 1

    env, agentic = _policy_moves("ATTACK_11_AGENTIC_SCOPE_EXPANSION", count=6)
    assert len({row["customer_id"] for row in agentic}) == 1
    assert len({row["device_id"] for row in agentic}) == 1
    assert agentic[0]["device_id"] in env.customers[agentic[0]["customer_id"]].devices
    assert [row["amount"] for row in agentic] == sorted(row["amount"] for row in agentic)

    _, boundary = _policy_moves("ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE", count=7)
    probe_mean = sum(row["amount"] for row in boundary[:4]) / 4
    exploit_mean = sum(row["amount"] for row in boundary[4:]) / 3
    assert exploit_mean > probe_mean * 1.5


def test_campaign_strategist_pivots_only_after_repeated_friction():
    strategist = CampaignStrategist("ATTACK_4_CNP_HIGH_VELOCITY", max_vectors=4)
    strategist.observe(
        "ATTACK_4_CNP_HIGH_VELOCITY",
        {"decision": "STEP_UP", "amount": 100.0},
        signal_families=["behavioural velocity"],
    )
    assert strategist.decide().action is StrategyAction.MUTATE
    for _ in range(2):
        strategist.observe(
            "ATTACK_4_CNP_HIGH_VELOCITY",
            {"decision": "DECLINE", "amount": 100.0},
            signal_families=["behavioural velocity"],
        )
    decision = strategist.decide()
    assert decision.action is StrategyAction.PIVOT
    assert decision.next_spec_id in PORTFOLIO["ATTACK_4_CNP_HIGH_VELOCITY"].transitions


def test_portfolio_runner_uses_existing_single_vector_pump_and_pivots():
    async def scenario():
        env = PaymentEnvironment(n_customers=30, seed=883)
        stack = SimpleNamespace(env=env)
        initial = load_attack_spec(
            SPECS / PORTFOLIO["ATTACK_4_CNP_HIGH_VELOCITY"].attack_file
        )
        events = []

        async def emit(event: dict) -> None:
            events.append(event)

        async def fake_single_pump(stack, agent, campaign_size, emit, feedback_mode="gray"):
            del stack, feedback_mode
            for index in range(1, campaign_size + 1):
                await emit({
                    "type": "defense_decision",
                    "txn_index": index,
                    "decision": "DECLINE",
                    "reasons": ["velocity>0.1"],
                    "amount": 100.0,
                })
            await emit({"type": "campaign_summary", "data": {"spec_id": agent.spec.spec_id}})

        result = await pump_portfolio_campaign(
            stack,
            initial,
            6,
            emit,
            single_pump=fake_single_pump,
            specs_dir=SPECS,
            segment_size=3,
            max_vectors=3,
        )
        return events, result

    events, result = asyncio.run(scenario())
    starts = [event for event in events if event["type"] == "vector_segment_start"]
    transitions = [event for event in events if event["type"] == "strategy_transition"]
    assert len(starts) >= 2
    assert starts[0]["data"]["spec_id"] != starts[1]["data"]["spec_id"]
    assert transitions[0]["data"]["action"] == "pivot"
    assert result["strategy"]["unique_vectors"] >= 2
