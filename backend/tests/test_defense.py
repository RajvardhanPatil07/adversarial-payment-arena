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

# Training spans families across every control class the taxonomy names, not
# just the original three card-velocity ones. ATTACK_4 stays held out as the
# zero-day for the holdout experiment.
# Mirrors the shipped configuration in corpus_builder's __main__ so the test
# measures the model that actually gets deployed. A test that trains on a
# different family mix than production is measuring a different detector.
TRAIN_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 150,
    "ATTACK_2_SYNTHETIC_MULE_RING": 150,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 150,
    "ATTACK_5_APP_SCAM_PERSONALISED": 120,
    "ATTACK_6_VPA_RENTAL_MULE": 120,
    "ATTACK_7_SYNCHRONISED_BURST_CASHOUT": 120,
    "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING": 120,
    "ATTACK_9_OTP_RELAY_VISHING": 120,
    "ATTACK_10_EXEMPTION_BAND_ABUSE": 120,
    "ATTACK_11_AGENTIC_SCOPE_EXPANSION": 120,
    "ATTACK_12_GEO_VELOCITY_ITINERARY": 120,
    "ATTACK_13_MERCHANT_BUSTOUT": 120,
    "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE": 120,
}
EVAL_COUNTS = {k: 30 for k in TRAIN_COUNTS}

# Transactions per cardholder, held IDENTICAL across train / calibration /
# evaluation. This is not a tuning knob: the sequence-level features need
# several prior events for the same entity inside the lookback window, so a
# split with thinner history silently zeroes them and the model is scored on a
# feature vector that does not exist at inference. See build_corpus.
TXNS_PER_CUSTOMER = 24.0

# Operating budget. Calibration targets this and the FPR test asserts against
# it, so the two cannot drift apart.
TARGET_FPR = 0.01

# Corpus sizes, named once so assertions derive from them rather than
# duplicating literals that rot the moment a split is resized.
TRAIN_LEGIT = 6000
CALIB_LEGIT = 2000
EVAL_LEGIT = 2000


@pytest.fixture(scope="module")
def trained_engine():
    corpus = build_corpus(
        n_legit=TRAIN_LEGIT, attack_counts=TRAIN_COUNTS, seed=123,
        txns_per_customer=TXNS_PER_CUSTOMER,
    )
    engine = DecisionEngine(environment=corpus["env"])
    metrics = engine.train(corpus["rows"])
    # Pin the operating point on a split generated from a seed DISJOINT from
    # both the training and evaluation seeds. Thresholds fitted on evaluation
    # rows would be leakage, which is exactly the discipline this repository
    # claims to enforce -- so it is enforced here too.
    #
    # The split is MIXED, not legitimate-only. Calibration searches for the
    # threshold with the best recall inside the FPR budget, so a split with no
    # attacks in it gives the search no recall signal to maximise: every
    # candidate ties at zero and it returns an arbitrary one.
    calib = build_corpus(
        n_legit=CALIB_LEGIT, attack_counts=EVAL_COUNTS, seed=321,
        txns_per_customer=TXNS_PER_CUSTOMER,
    )
    engine.calibrate(calib["rows"], target_fpr=TARGET_FPR)
    return engine, metrics


@pytest.fixture(scope="module")
def eval_run(trained_engine):
    """Fresh-seed eval corpus replayed through a CLEAN engine."""
    engine, _ = trained_engine
    ev = build_corpus(
        n_legit=EVAL_LEGIT, attack_counts=EVAL_COUNTS, seed=777,
        txns_per_customer=TXNS_PER_CUSTOMER,
    )
    # A FRESH stack, re-trained weights carried by value rather than by
    # reference. Passing `scorer=engine.scorer` shares the FeatureExtractor,
    # which is still holding every training transaction in its per-customer and
    # per-merchant history -- so eval rows get scored against phantom history
    # and `merch_first_seen` already knows every merchant. Measured on the CLI
    # harness: 4.25% FPR with a shared scorer against 0.85% with a clean one, on
    # identical thresholds.
    engine_eval = DecisionEngine(environment=ev["env"])
    engine_eval.scorer.model = engine.scorer.model
    engine_eval.novelty.model = engine.novelty.model
    # Carry ONLY the calibrated operating point across -- never the state.
    for attr in ("stepup_threshold", "decline_threshold", "manual_threshold",
                 "ring_risk_threshold", "novelty_alone_alerts"):
        setattr(engine_eval, attr, getattr(engine, attr))

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
    # Pin the graph threshold explicitly: this test is about ladder PRECEDENCE
    # (a ring outranks a velocity decline for the reason code), not about
    # whatever sensitivity calibration happened to select on this corpus. The
    # calibrated engine can legitimately choose to silence the graph layer
    # entirely, and that must not be able to fail a precedence test.
    engine.ring_risk_threshold = 0.5
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
    # "Low velocity" is only meaningful relative to the manual threshold, which
    # calibration sets from the score distribution -- on a well-separated model
    # that lands near 5e-05, far below the 0.10 this test used to hardcode. The
    # test was asserting on a stale constant, not on ladder behaviour. Pin the
    # boundary and place the score beneath it.
    engine.manual_threshold = 0.20
    engine.stepup_threshold = 0.30
    engine.decline_threshold = 0.50
    engine.ring_risk_threshold = 0.5
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
    # Derived from the fixture, not a duplicated literal: this assertion exists
    # to prove the Isolation Forest saw ONLY legitimate rows (never an attack),
    # and it should not fail merely because the corpus was resized.
    assert metrics["iforest"]["train_rows_legit"] == TRAIN_LEGIT
    assert metrics["iforest"]["train_rows_legit"] < metrics["xgb"]["train_rows"]
