"""
Defense stack tests — Step 4 exit criteria.

Mandated: 1000 legit + 100 attack transactions through the full
DecisionEngine with FPR < 5% and TPR > 80%. Plus unit contracts: ring
detection topology, ladder ordering (ring beats velocity), MANUAL_REVIEW
reachability, cost-matrix arithmetic, and novelty training on legit-only.
"""

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
    """Fresh-seed eval corpus replayed through the trained engine."""
    engine, _ = trained_engine
    ev = build_corpus(n_legit=1000, attack_counts=EVAL_COUNTS, seed=777)
    engine_eval = DecisionEngine(environment=ev["env"], scorer=engine.scorer, novelty=engine.novelty)

    records, truths = [], []
    for r in sorted(ev["rows"], key=lambda r: r["payload"]["timestamp"]):
        records.append(engine_eval.decide(r["payload"]))
        truths.append("legit" if r["label"] == 0 else "attack")
    return records, truths


# --------------------------------------------------------------------------- #
# Mandated end-to-end metrics
# --------------------------------------------------------------------------- #


def test_fpr_under_5_percent(eval_run):
    records, truths = eval_run
    legit = [(r, t) for r, t in zip(records, truths) if t == "legit"]
    flagged = sum(1 for r, _ in legit if r["decision"] != APPROVE)
    fpr = flagged / len(legit)
    print(f"\nFPR = {flagged}/{len(legit)} = {fpr:.2%}")
    assert fpr < 0.05, f"FPR budget blown: {fpr:.2%}"


def test_tpr_over_80_percent(eval_run):
    records, truths = eval_run
    attacks = [(r, t) for r, t in zip(records, truths) if t == "attack"]
    caught = sum(1 for r, _ in attacks if r["decision"] != APPROVE)
    tpr = caught / len(attacks)
    declined = sum(1 for r, _ in attacks if r["decision"] == DECLINE)
    print(f"\nTPR = {caught}/{len(attacks)} = {tpr:.2%} (declined={declined})")
    assert tpr > 0.80, f"catch rate too low: {tpr:.2%}"


def test_cost_matrix_shape_on_eval_run(eval_run, trained_engine):
    engine, _ = trained_engine
    records, truths = eval_run
    cost = engine.compute_cost_matrix(records, truths)
    assert set(cost) >= {"fp_cost_bps", "fn_loss", "tp_saved", "net_savings"}
    # a profitable stack saves more than it loses + friction
    assert cost["net_savings"] > 0
    assert cost["fp_cost_bps"] < 15.0 * 0.05 + 1e-6  # bps consistent w/ FPR<5%


# --------------------------------------------------------------------------- #
# Layer unit contracts
# --------------------------------------------------------------------------- #


def test_graph_flags_mule_ring_shared_device():
    g = EntityGraph()
    rng = random.Random(3)

    def wire(cid, dev, ip="9.9.9.9"):
        return {
            "customer_id": cid, "device_id": dev, "ip_address": ip,
            "merchant_id": "MERCH_ELEC_BESTBUYX",
        }

    g.observe(wire("CUST_0001", "DEV_AAA"))
    g.observe(wire("CUST_0002", "DEV_AAA"))
    res = g.check(wire("CUST_0003", "DEV_AAA"))  # check BEFORE observing 3rd
    assert not res["ring_detected"]              # device links only 2 customers

    g.observe(wire("CUST_0003", "DEV_AAA"))
    # a RETURNING ring member trips the screen: the checked customer must be
    # directly linked to the shared infra (merchant-mediated co-residence in
    # a component never implicates anyone — see EntityGraph.check).
    res = g.check(wire("CUST_0001", "DEV_AAA"))
    assert res["ring_detected"]
    assert res["component_customers"] >= 3
    assert any(s["linked_customers"] >= 3 for s in res["shared_infra"])
    assert res["risk_score"] > 0.5


def test_graph_no_ring_for_unique_devices():
    g = EntityGraph()
    for i in range(50):
        g.observe({
            "customer_id": f"CUST_{i:04d}",
            "device_id": f"DEV_{i:010d}",
            "ip_address": f"10.0.{i // 256}.{i % 256}",
            "merchant_id": "MERCH_GROC_SAFewayX",
        })
    res = g.check({"customer_id": "CUST_0051", "device_id": "DEV_new",
                   "ip_address": "10.0.1.1", "merchant_id": "MERCH_GROC_SAFewayX"})
    assert not res["ring_detected"]
    assert res["risk_score"] < 0.2


def test_ladder_ring_beats_velocity(trained_engine):
    engine, _ = trained_engine
    engine.scorer.score_from_features = lambda feats: 0.99          # would DECLINE anyway
    engine.graph.check = lambda payload: {"ring_detected": True, "ring_id": "RING_X",
                                          "risk_score": 0.9, "shared_infra": [],
                                          "component_customers": 4}
    env = PaymentEnvironment(seed=5)
    from data.legit_generator import build_legit_payload
    from faker import Faker
    msg = build_legit_payload(env, random.Random(1), Faker())
    rec = engine.decide(msg)
    assert rec["decision"] == DECLINE
    assert rec["reasons"][0].startswith("ring_detected")


def test_manual_review_is_reachable(trained_engine):
    """Guards the documented deviation: anomaly + LOW velocity -> review queue."""
    engine, _ = trained_engine
    engine.scorer.score_from_features = lambda feats: 0.10
    engine.novelty.detect = lambda payload, features: {"is_anomaly": True, "anomaly_score": 0.7}
    engine.graph.check = lambda payload: {"ring_detected": False, "ring_id": None,
                                          "risk_score": 0.0, "shared_infra": [],
                                          "component_customers": 1}
    env = PaymentEnvironment(seed=6)
    from data.legit_generator import build_legit_payload
    from faker import Faker
    rec = engine.decide(build_legit_payload(env, random.Random(2), Faker()))
    assert rec["decision"] == MANUAL_REVIEW


def test_cost_matrix_arithmetic():
    engine = DecisionEngine()
    records = [
        {"decision": APPROVE, "amount": 100.0},   # TP (legit approved) — ignored
        {"decision": STEP_UP, "amount": 200.0},   # FP: 200 * 15bps = $0.30
        {"decision": DECLINE, "amount": 500.0},   # TP saved: $500
        {"decision": APPROVE, "amount": 1000.0},  # FN loss: $1000
    ]
    truths = ["legit", "legit", "attack", "attack"]
    cost = engine.compute_cost_matrix(records, truths)
    assert cost["counts"] == {"false_positives": 1, "false_negatives": 1, "true_positives_declined": 1}
    assert cost["fp_cost_usd"] == pytest.approx(0.30)
    # legit volume = 300 -> bps of total volume
    assert cost["fp_cost_bps"] == pytest.approx(0.30 / 300.0 * 10_000, rel=1e-3)
    assert cost["fn_loss"] == pytest.approx(1000.0)
    assert cost["tp_saved"] == pytest.approx(500.0)
    assert cost["net_savings"] == pytest.approx(500.0 - 1000.0 - 0.30)


def test_novelty_trained_on_legit_only(trained_engine):
    _, metrics = trained_engine
    corpus_legit = 3000
    assert metrics["iforest"]["train_rows_legit"] == corpus_legit
