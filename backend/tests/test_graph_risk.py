"""Evidence-weighted graph-risk regressions."""

from defense.graph import EntityGraph


def _wire(customer: str, device: str, ip: str, merchant: str, ts: int) -> dict:
    return {
        "customer_id": customer,
        "device_id": device,
        "ip_address": ip,
        "merchant_id": merchant,
        "timestamp": f"2026-08-30T06:{ts:02d}:00+00:00",
    }


def test_shared_ip_without_other_evidence_is_not_ring():
    graph = EntityGraph()
    for index in range(4):
        row = _wire(f"C{index}", f"D{index}", "10.0.0.1", f"M{index}", index)
        assert not graph.check(row)["ring_detected"]
        graph.observe(row)

    result = graph.check(_wire("C4", "D4", "10.0.0.1", "M4", 4))
    assert not result["ring_detected"]
    assert result["evidence"]["ip_degree_10m"] == 5
    assert result["evidence"]["merchant_fanin_10m"] == 1
    assert result["evidence"]["shared_ip_alone_is_hard_rule"] is False
    assert result["risk_score"] < 0.5


def test_shared_device_remains_immediate_hard_evidence():
    graph = EntityGraph()
    for index in range(2):
        row = _wire(f"C{index}", "SHARED_DEVICE", f"10.0.0.{index+1}", f"M{index}", index)
        graph.check(row)
        graph.observe(row)

    result = graph.check(_wire("C2", "SHARED_DEVICE", "10.0.0.9", "M9", 2))
    assert result["ring_detected"]
    assert result["risk_score"] >= 0.72
    assert result["component_customers"] == 3
    assert result["evidence"]["device_degree_10m"] == 3


def test_shared_ip_plus_beneficiary_convergence_is_strong_composite_evidence():
    graph = EntityGraph()
    for index in range(4):
        row = _wire(f"C{index}", f"D{index}", "10.0.0.1", "MULE", index)
        graph.check(row)
        graph.observe(row)

    result = graph.check(_wire("C4", "D4", "10.0.0.1", "MULE", 4))
    assert result["ring_detected"]
    assert result["risk_score"] >= 0.72
    assert result["evidence"]["ip_degree_10m"] == 5
    assert result["evidence"]["merchant_fanin_10m"] == 5


def test_popular_merchant_is_low_risk_without_shared_infra():
    graph = EntityGraph()
    for index in range(20):
        row = _wire(
            f"C{index}",
            f"D{index}",
            f"10.0.1.{index+1}",
            "POPULAR",
            index % 10,
        )
        graph.check(row)
        graph.observe(row)

    result = graph.check(_wire("C99", "D99", "10.0.9.9", "POPULAR", 9))
    assert not result["ring_detected"]
    assert result["risk_score"] < 0.2
    assert result["evidence"]["merchant_fanin_10m"] >= 8
