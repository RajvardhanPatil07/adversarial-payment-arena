"""Defense stack tests and regression coverage for the live decision path."""

import random
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from data.corpus_builder import build_corpus  # noqa: E402
from defense.decision import (  # noqa: E402
    APPROVE,
    DECLINE,
    MANUAL_REVIEW,
    STEP_UP,
    DecisionEngine,
)
from defense.graph import EntityGraph  # noqa: E402
from environment.payment_stack import PaymentEnvironment  # noqa: E402

TRAIN_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 230,
    "ATTACK_2_SYNTHETIC_MULE_RING": 230,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 240,
}
EVAL_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 34,
    "ATTACK_2_SYNTHETIC_MULE_RING": 33,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 33,
}


@pytest.fixture(scope="module")
def trained_engine():
    corpus = build_corpus(n_legit=3000, attack_counts=TRAIN_COUNTS, seed=123)
    engine = DecisionEngine(environment=corpus["env"])
    metrics = engine.train(corpus["rows"])
    return engine, metrics


@pytest.fixture(scope="module")
def eval_run(trained_engine):
    engine, _ = trained_engine
    ev = build_corpus(n_legit=1000, attack_counts=EVAL_COUNTS, seed=777)
    engine_eval = DecisionEngine(
        environment=ev["env"], scorer=engine.scorer, novelty=engine.novelty
    )

    records, truths = [], []
    for row in sorted(ev["rows"], key=lambda item: item["payload"]["timestamp"]):
        records.append(engine_eval.decide(row["payload"]))
        truths.append("legit" if row["label"] == 0 else "attack")
    return records, truths


def test_fpr_under_5_percent(eval_run):
    records, truths = eval_run
    legit = [(record, truth) for record, truth in zip(records, truths) if truth == "legit"]
    flagged = sum(1 for record, _ in legit if record["decision"] != APPROVE)
    fpr = flagged / len(legit)
    print(f"\nFPR = {flagged}/{len(legit)} = {fpr:.2%}")
    assert fpr < 0.05, f"FPR budget blown: {fpr:.2%}"


def test_tpr_over_80_percent(eval_run):
    records, truths = eval_run
    attacks = [(record, truth) for record, truth in zip(records, truths) if truth == "attack"]
    caught = sum(1 for record, _ in attacks if record["decision"] != APPROVE)
    tpr = caught / len(attacks)
    declined = sum(1 for record, _ in attacks if record["decision"] == DECLINE)
    print(f"\nTPR = {caught}/{len(attacks)} = {tpr:.2%} (declined={declined})")
    assert tpr > 0.80, f"catch rate too low: {tpr:.2%}"


def test_cost_matrix_shape_on_eval_run(eval_run, trained_engine):
    engine, _ = trained_engine
    records, truths = eval_run
    cost = engine.compute_cost_matrix(records, truths)
    assert set(cost) >= {"fp_cost_bps", "fn_loss", "tp_saved", "net_savings"}
    assert cost["net_savings"] > 0
    assert cost["fp_cost_bps"] < 15.0 * 0.05 + 1e-6


def _wire(cid: str, dev: str, ip: str = "9.9.9.9", merchant: str = "MERCH_ELEC_BESTBUYX") -> dict:
    return {
        "customer_id": cid,
        "device_id": dev,
        "ip_address": ip,
        "merchant_id": merchant,
    }


def test_graph_flags_third_customer_on_shared_device_immediately():
    graph = EntityGraph()
    graph.observe(_wire("CUST_0001", "DEV_AAA", ip="1.1.1.1"))
    graph.observe(_wire("CUST_0002", "DEV_AAA", ip="2.2.2.2"))

    # check() runs before observe(), but the current transaction is evaluated
    # prospectively. The third customer should therefore be caught on entry,
    # not only on a later returning transaction.
    result = graph.check(_wire("CUST_0003", "DEV_AAA", ip="3.3.3.3"))
    assert result["ring_detected"]
    assert result["component_customers"] == 3
    assert any(item["linked_customers"] == 3 for item in result["shared_infra"])
    assert result["risk_score"] > 0.5


def test_graph_merchant_hub_cannot_inflate_ring_identity():
    graph = EntityGraph()
    merchant = "MERCH_GROC_SAFewayX"

    # Many unrelated customers share a popular merchant but unique infra.
    for index in range(50):
        graph.observe(
            _wire(
                f"CUST_{index:04d}",
                f"DEV_UNIQUE_{index:04d}",
                ip=f"10.0.0.{index + 1}",
                merchant=merchant,
            )
        )

    # Two existing customers additionally share one device; the third current
    # customer completes that direct-infrastructure ring.
    graph.observe(_wire("CUST_0001", "DEV_RING", ip="11.0.0.1", merchant=merchant))
    graph.observe(_wire("CUST_0002", "DEV_RING", ip="11.0.0.2", merchant=merchant))
    result = graph.check(_wire("CUST_0050", "DEV_RING", ip="11.0.0.3", merchant=merchant))

    assert result["ring_detected"]
    assert result["component_customers"] == 3
    assert result["risk_score"] < 1.0


def test_graph_observe_returns_only_new_edge_deltas():
    graph = EntityGraph()
    first = graph.observe(_wire("CUST_0001", "DEV_AAA"))
    second = graph.observe(_wire("CUST_0001", "DEV_AAA"))

    assert len(first) == 3
    assert second == []
    assert graph.g["C:CUST_0001"]["D:DEV_AAA"]["weight"] == 2


def test_graph_no_ring_for_unique_devices():
    graph = EntityGraph()
    for index in range(50):
        graph.observe({
            "customer_id": f"CUST_{index:04d}",
            "device_id": f"DEV_{index:010d}",
            "ip_address": f"10.0.{index // 256}.{index % 256}",
            "merchant_id": "MERCH_GROC_SAFewayX",
        })
    result = graph.check({
        "customer_id": "CUST_0051",
        "device_id": "DEV_new",
        "ip_address": "10.0.1.1",
        "merchant_id": "MERCH_GROC_SAFewayX",
    })
    assert not result["ring_detected"]
    assert result["risk_score"] < 0.2


def test_ladder_ring_beats_velocity(trained_engine):
    engine, _ = trained_engine
    engine.scorer.score_from_features = lambda feats: 0.99
    engine.graph.check = lambda payload: {
        "ring_detected": True,
        "ring_id": "RING_X",
        "risk_score": 0.9,
        "shared_infra": [],
        "component_customers": 4,
    }
    env = PaymentEnvironment(seed=5)
    from data.legit_generator import build_legit_payload
    from faker import Faker

    message = build_legit_payload(env, random.Random(1), Faker())
    record = engine.decide(message)
    assert record["decision"] == DECLINE
    assert record["reasons"][0].startswith("ring_detected")


def test_manual_review_is_reachable(trained_engine):
    engine, _ = trained_engine
    engine.scorer.score_from_features = lambda feats: 0.10
    engine.novelty.detect = lambda payload, features: {
        "is_anomaly": True,
        "anomaly_score": 0.7,
    }
    engine.graph.check = lambda payload: {
        "ring_detected": False,
        "ring_id": None,
        "risk_score": 0.0,
        "shared_infra": [],
        "component_customers": 1,
    }
    env = PaymentEnvironment(seed=6)
    from data.legit_generator import build_legit_payload
    from faker import Faker

    record = engine.decide(build_legit_payload(env, random.Random(2), Faker()))
    assert record["decision"] == MANUAL_REVIEW


def test_cost_matrix_arithmetic():
    engine = DecisionEngine()
    records = [
        {"decision": APPROVE, "amount": 100.0},
        {"decision": STEP_UP, "amount": 200.0},
        {"decision": DECLINE, "amount": 500.0},
        {"decision": APPROVE, "amount": 1000.0},
    ]
    truths = ["legit", "legit", "attack", "attack"]
    cost = engine.compute_cost_matrix(records, truths)
    assert cost["counts"] == {
        "false_positives": 1,
        "false_negatives": 1,
        "true_positives_declined": 1,
    }
    assert cost["fp_cost_usd"] == pytest.approx(0.30)
    assert cost["fp_cost_bps"] == pytest.approx(0.30 / 300.0 * 10_000, rel=1e-3)
    assert cost["fn_loss"] == pytest.approx(1000.0)
    assert cost["tp_saved"] == pytest.approx(500.0)
    assert cost["net_savings"] == pytest.approx(500.0 - 1000.0 - 0.30)


def test_novelty_trained_on_legit_only(trained_engine):
    _, metrics = trained_engine
    assert metrics["iforest"]["train_rows_legit"] == 3000
