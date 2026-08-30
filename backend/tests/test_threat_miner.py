"""Emerging-threat miner regression tests."""

from defense.threat_miner import ThreatMiner


def _record(index: int, *, attack: bool = True, decision: str = "STEP_UP") -> dict:
    return {
        "decision": decision,
        "label_hint": attack,
        "features": {
            "cust_txn_count_10m": 3,
            "cust_amount_sum_10m": 750.0,
            "amount_over_mean30": 3.5,
            "cust_mcc_distinct_1h": 2,
            "device_age_hours": 0.02,
            "dev_txn_count_10m": 7,
            "merch_txn_count_10m": 12,
            "merch_distinct_custs_10m": 8,
            "device_known": 0,
            "pos_entry_code": 0,
            "tds_code": 2,
        },
        "scores": {
            "velocity": 0.72,
            "novelty_anomaly": 0.25,
            "is_anomaly": True,
            "ring_risk": 0.16,
            "ring_detected": False,
            "ring_id": None,
        },
        "payload": {
            "transaction_id": f"T{index}",
            "merchant_id": "MERCH_X",
            "pos_entry_mode": "ECOM",
            "3ds_status": "N",
            "timestamp": f"2026-08-30T06:{index:02d}:00+00:00",
        },
    }


def test_repeated_suspicious_pattern_becomes_emerging_threat():
    miner = ThreatMiner(min_cluster_size=3)
    emitted = []
    for index in range(1, 6):
        fingerprint = miner.observe(_record(index))
        if fingerprint:
            emitted.append(fingerprint)

    assert emitted
    first = emitted[0]
    assert first["status"] == "emerging"
    assert first["transactions"] == 3
    assert first["threat_id"].startswith("EMERGENT_")
    assert first["novelty_confidence"] > 0.5
    assert first["top_merchants"][0]["value"] == "MERCH_X"
    assert miner.snapshot()[0]["transactions"] == 5


def test_clustering_does_not_require_simulator_truth():
    miner = ThreatMiner(min_cluster_size=3)
    for index in range(1, 4):
        row = _record(index, attack=False)
        row.pop("label_hint")
        miner.observe(row)

    fingerprint = miner.snapshot()[0]
    assert fingerprint["status"] == "emerging"
    assert fingerprint["simulator_attack_fraction"] == 0.0


def test_approved_normal_rows_are_not_clustered():
    miner = ThreatMiner()
    row = _record(1, attack=False, decision="APPROVE")
    row["scores"].update(
        velocity=0.12,
        novelty_anomaly=-0.1,
        is_anomaly=False,
        ring_risk=0.0,
        ring_detected=False,
    )
    assert miner.observe(row) is None
    assert miner.diagnostics()["events_clustered"] == 0
    assert miner.snapshot() == []
