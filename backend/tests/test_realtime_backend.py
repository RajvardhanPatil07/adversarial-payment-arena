"""Parity checks for the optional Rust rolling-feature backend."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util

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
    native = FeatureExtractor(maxlen=5)
    fallback = FeatureExtractor(maxlen=5)
    fallback._rust = None

    assert native.backend == "rust"

    for index in range(4):
        payload = _payload(index)
        assert native.features(payload) == fallback.features(payload)
        native.observe(payload)
        fallback.observe(payload)

    final = _payload(4)
    assert native.features(final) == fallback.features(final)
