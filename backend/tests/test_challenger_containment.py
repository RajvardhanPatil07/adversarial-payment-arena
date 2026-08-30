"""Shadow model and campaign-containment regressions."""

from datetime import datetime, timedelta, timezone

from defense.challenger import ShadowChallenger
from defense.realtime import FEATURE_NAMES
from evidence.containment import CampaignContainment


def _features(index: int, attack: bool) -> dict:
    base = {
        "cust_txn_count_10m": 1 + (index % 2),
        "cust_amount_sum_10m": 50.0 + index,
        "amount_over_mean30": 1.0,
        "cust_mcc_distinct_1h": 1,
        "device_age_hours": 48.0,
        "dev_txn_count_10m": 1,
        "merch_txn_count_10m": 2,
        "merch_distinct_custs_10m": 2,
        "device_known": 1,
        "pos_entry_code": 0,
        "tds_code": 0,
        # Sequence-level features added by the 14-family calibrated corpus.
        "dev_distinct_custs_1h": 1,
        "iat_regularity": 0.25,
        "amount_escalation": 0.0,
        "amount_band_tightness": 0.35,
        "round_amount_frac": 0.25,
        "merch_youth": 0.05,
        "low_value_probe_ratio": 0.0,
    }
    if attack:
        base.update(
            cust_txn_count_10m=7 + index % 4,
            cust_amount_sum_10m=1800.0 + 10 * index,
            amount_over_mean30=4.5,
            device_age_hours=0.01,
            dev_txn_count_10m=9,
            merch_txn_count_10m=14,
            merch_distinct_custs_10m=9,
            device_known=0,
            tds_code=2,
            dev_distinct_custs_1h=5,
            iat_regularity=0.92,
            amount_escalation=0.42,
            amount_band_tightness=0.88,
            round_amount_frac=0.02,
            merch_youth=0.95,
            low_value_probe_ratio=0.84,
        )
    assert set(base) == set(FEATURE_NAMES)
    return base


def _row(index: int, attack: bool) -> dict:
    base = datetime(2026, 8, 30, 6, 0, tzinfo=timezone.utc)
    return {
        "label": int(attack),
        "features": _features(index, attack),
        "payload": {
            "transaction_id": f"T{index}",
            "customer_id": f"C{index:03d}",
            "device_id": "SHARED" if attack and index % 3 == 0 else f"D{index:03d}",
            "ip_address": "10.1.1.1" if attack else f"10.0.0.{index+1}",
            "merchant_id": "MULE" if attack else f"M{index % 5}",
            "timestamp": (base + timedelta(seconds=index * 10)).isoformat(),
        },
    }


def test_shadow_challenger_trains_and_never_controls_live_decision():
    rows = [_row(index, attack=False) for index in range(60)]
    rows += [_row(100 + index, attack=True) for index in range(30)]
    challenger = ShadowChallenger()
    metrics = challenger.fit_rows(rows)
    assert metrics["status"] == "ready"
    assert challenger.ready

    record = {
        "decision": "STEP_UP",
        "features": _features(999, True),
        "scores": {"ring_risk": 0.2, "ring_detected": False, "velocity": 0.7},
    }
    comparison = challenger.observe(record, "attack")
    assert comparison is not None
    assert comparison["challenger"]["shadow_only"] is True
    snapshot = challenger.snapshot()
    assert snapshot["controls_live_authorizations"] is False
    assert snapshot["compared"] == 1
    assert snapshot["simulator_truth_metrics"]["attack_rows"] == 1


def test_containment_tracks_exposure_before_controls_fire():
    tracker = CampaignContainment("ATTACK_X")
    tracker.observe({"decision": "APPROVE", "amount": 100.0}, 1)
    tracker.observe({"decision": "APPROVE", "amount": 200.0}, 2)
    tracker.observe({"decision": "STEP_UP", "amount": 300.0}, 3)
    tracker.mark_emerging_threat({"threat_id": "EMERGENT_0001"}, 3)
    tracker.observe({"decision": "DECLINE", "amount": 400.0}, 4)

    summary = tracker.summary()
    assert summary["approved_escapes"] == 2
    assert summary["escape_rate"] == 0.5
    assert summary["transactions_to_first_friction"] == 3
    assert summary["transactions_to_first_decline"] == 4
    assert summary["transactions_to_emerging_threat"] == 3
    assert summary["fraud_value_before_first_friction"] == 300.0
    assert summary["fraud_value_before_first_decline"] == 600.0
