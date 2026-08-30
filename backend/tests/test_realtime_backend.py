"""Parity checks for the optional Rust rolling-feature backend."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import random

import pytest

from defense.realtime import FeatureExtractor


def _payload(index: int) -> dict:
    base = datetime(2026, 8, 30, 4, 0, tzinfo=timezone.utc)
    return {
        "customer_id": "CUST_TEST",
        "device_id": "DEV_TEST_001",
        "merchant_id": "MERCH_TEST",
        "amount": 50.0 + index * 10.0,
        "mcc": 5411 if index < 2 else 5732,
        "pos_entry_mode": "ECOM",
        "3ds_status": "Y",
        "timestamp": (base + timedelta(minutes=index * 3)).isoformat(),
    }


def _native_and_fallback(maxlen: int = 500):
    native = FeatureExtractor(maxlen=maxlen)
    fallback = FeatureExtractor(maxlen=maxlen)
    fallback._rust = None
    assert native.backend == "rust"
    assert fallback.backend == "python"
    return native, fallback


def test_python_feature_backend_contract():
    extractor = FeatureExtractor(maxlen=5)
    extractor._rust = None

    first = extractor.features(_payload(0))
    assert first["cust_txn_count_10m"] == 0
    assert first["amount_over_mean30"] == pytest.approx(1.0, abs=1e-3)

    extractor.observe(_payload(0))
    second = extractor.features(_payload(1))
    assert second["cust_txn_count_10m"] == 1
    assert second["cust_amount_sum_10m"] == 50.0
    assert second["dev_txn_count_10m"] == 1
    assert second["merch_distinct_custs_10m"] == 1


@pytest.mark.skipif(
    importlib.util.find_spec("arena_core") is None,
    reason="arena_core is built in the production container, not required for source-only tests",
)
def test_rust_backend_matches_python_fallback():
    native, fallback = _native_and_fallback(maxlen=5)

    for index in range(4):
        payload = _payload(index)
        assert native.features(payload) == fallback.features(payload)
        native.observe(payload)
        fallback.observe(payload)

    final = _payload(4)
    assert native.features(final) == fallback.features(final)
    assert native.state_sizes() == fallback.state_sizes()


@pytest.mark.skipif(
    importlib.util.find_spec("arena_core") is None,
    reason="arena_core is built in the production container, not required for source-only tests",
)
def test_rust_backend_matches_cold_start_history():
    native, fallback = _native_and_fallback()
    payload = _payload(0)
    history = [
        {"payload": {"amount": 25.0}},
        {"payload": {"amount": 75.0}},
    ]
    assert native.features(payload, history) == fallback.features(payload, history)
    assert native.features(payload, history)["amount_over_mean30"] == pytest.approx(1.0)


@pytest.mark.skipif(
    importlib.util.find_spec("arena_core") is None,
    reason="arena_core is built in the production container, not required for source-only tests",
)
def test_rust_backend_matches_after_bounded_eviction():
    native, fallback = _native_and_fallback(maxlen=3)
    for index in range(8):
        payload = _payload(index)
        assert native.features(payload) == fallback.features(payload)
        native.observe(payload)
        fallback.observe(payload)

    assert native.features(_payload(8)) == fallback.features(_payload(8))


@pytest.mark.skipif(
    importlib.util.find_spec("arena_core") is None,
    reason="arena_core is built in the production container, not required for source-only tests",
)
def test_rust_backend_matches_out_of_order_replay():
    native, fallback = _native_and_fallback(maxlen=20)
    base = datetime(2026, 8, 30, 4, 0, tzinfo=timezone.utc)
    offsets = [0, 20, 5, 40, 10, 70, 30]

    for index, minutes in enumerate(offsets):
        payload = _payload(index)
        payload["timestamp"] = (base + timedelta(minutes=minutes)).isoformat()
        assert native.features(payload) == fallback.features(payload)
        native.observe(payload)
        fallback.observe(payload)

    probe = _payload(20)
    probe["timestamp"] = (base + timedelta(minutes=45)).isoformat()
    assert native.features(probe) == fallback.features(probe)


@pytest.mark.skipif(
    importlib.util.find_spec("arena_core") is None,
    reason="arena_core is built in the production container, not required for source-only tests",
)
def test_randomized_native_python_parity():
    """Exercise ordered and unordered entity streams across many identities."""
    native, fallback = _native_and_fallback(maxlen=50)
    rng = random.Random(20260830)
    base = datetime(2026, 8, 30, 4, 0, tzinfo=timezone.utc)
    clock = 0

    for index in range(1000):
        clock += rng.randint(0, 20)
        # Occasionally replay an older event so the Rust full-scan fallback is
        # exercised, not only the chronological fast path.
        offset = clock - rng.randint(1, 600) if index % 97 == 0 and index else clock
        payload = {
            "customer_id": f"CUST_{rng.randrange(40):04d}",
            "device_id": f"DEV_{rng.randrange(70):04d}",
            "merchant_id": f"MERCH_{rng.randrange(12):02d}",
            "amount": round(rng.uniform(1.0, 1500.0), 2),
            "mcc": rng.choice([5411, 5732, 5812, 5947, 7011]),
            "pos_entry_mode": rng.choice(["ECOM", "CNP", "CHIP"]),
            "3ds_status": rng.choice(["Y", "A", "N"]),
            "timestamp": (base + timedelta(seconds=offset)).isoformat(),
        }
        assert native.features(payload) == fallback.features(payload), f"parity failed at {index}"
        native.observe(payload)
        fallback.observe(payload)

    assert native.state_sizes() == fallback.state_sizes()
