"""Parity/regression tests for the high-throughput batch decision path."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from defense.decision import DecisionEngine
from environment.payment_stack import PaymentEnvironment
from schemas.payment import PaymentMessage

BASE = datetime(2026, 8, 30, 4, 0, tzinfo=timezone.utc)


def _raw_payload(env: PaymentEnvironment, index: int) -> dict:
    customer_index = index % 12
    customer_id = f"CUST_{customer_index:04d}"
    customer = env.customers[customer_id]
    merchants = list(env.merchant_registry.values())
    merchant = merchants[index % len(merchants)]

    # Groups of exactly three customers share an IP, which exercises the ring
    # detector without creating an ever-growing shared-infrastructure set.
    group = customer_index // 3
    ip_address = f"10.0.{group // 256}.{group % 256 + 1}"

    return {
        "transaction_id": f"BATCH_{index:010d}",
        "customer_id": customer_id,
        "merchant_id": merchant.merchant_id,
        "mcc": merchant.mcc,
        "amount": 25.0 + float(index % 200),
        "currency": "USD",
        "pos_entry_mode": "ECOM",
        "3ds_status": "Y",
        "ip_address": ip_address,
        "ip_country": merchant.country,
        "device_id": customer.devices[0],
        "stolen_resource": "phished_credentials" if index % 17 == 0 else None,
        "timestamp": (BASE + timedelta(seconds=index * 5)).isoformat(),
    }


def test_batch_decision_matches_scalar_full_records():
    scalar_env = PaymentEnvironment(n_customers=20, seed=812)
    batch_env = PaymentEnvironment(n_customers=20, seed=812)
    scalar = DecisionEngine(environment=scalar_env)
    batch = DecisionEngine(environment=batch_env)

    raw_rows = [_raw_payload(scalar_env, index) for index in range(240)]

    scalar_records = []
    for raw in raw_rows:
        msg = PaymentMessage.model_validate(raw)
        gate = scalar_env.ingest(msg)
        assert gate["accepted"], gate["reason"]
        scalar_records.append(scalar.decide(msg))

    prepared = []
    for raw in raw_rows:
        msg = PaymentMessage.model_validate(raw)
        gate = batch_env.ingest(msg)
        assert gate["accepted"], gate["reason"]
        prepared.append(batch.prepare_for_batch(msg))
    batch_records = batch.finalize_batch(prepared)

    assert len(batch_records) == len(scalar_records)
    for expected, actual in zip(scalar_records, batch_records):
        assert actual == expected

    assert batch.scorer.extractor.state_sizes() == scalar.scorer.extractor.state_sizes()
    assert batch_env.events_seen_total == scalar_env.events_seen_total == len(raw_rows)


def test_bounded_event_retention_keeps_exact_total_count():
    env = PaymentEnvironment(
        n_customers=20,
        seed=813,
        event_stream_maxlen=7,
        gate_rejects_maxlen=3,
    )

    for index in range(50):
        result = env.ingest(PaymentMessage.model_validate(_raw_payload(env, index)))
        assert result["accepted"]

    assert env.events_seen_total == 50
    assert len(env.event_stream) == 7
    assert env.gate_rejects_total == 0
    assert len(env.gate_rejects) == 0
