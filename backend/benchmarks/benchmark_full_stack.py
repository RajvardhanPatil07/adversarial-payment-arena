"""High-volume benchmark for the complete payment-analysis pipeline.

Unlike ``benchmark_feature_state.py``, this exercises the full analytical path
used by the arena for each transaction:

1. Pydantic PaymentMessage validation;
2. Plausibility Gate + issuer enrichment/history/balance mutation;
3. rolling feature extraction/state update (Rust required here);
4. evidence-weighted entity-graph risk/check/update (Rust required here);
5. XGBoost velocity inference;
6. IsolationForest novelty inference;
7. the existing decision ladder;
8. the existing running cost matrix.

Only payload generation, model loading, WebSocket/UI serialization and LLM
attack generation are outside the measured interval. Payloads are generated in
bounded chunks so 100M-event runs do not require 100M Python dictionaries in
memory at once.

This is deliberately a throughput stress profile. The synthetic population is
compressed into a one-hour event-time window to exercise rolling state and graph
updates at high intensity, so its decision/FPR/cost mix is diagnostic telemetry,
not representative fraud-quality evidence. Quality claims belong to the separate
held-out/calibrated evaluation suite and model sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from defense.decision import DecisionEngine
from defense.novelty import NoveltyDetector
from defense.realtime import FeatureExtractor, VelocityScorer
from environment.payment_stack import PaymentEnvironment
from schemas.payment import PaymentMessage

BASE_TIME = datetime(2026, 8, 30, 4, 0, tzinfo=timezone.utc)
ONE_HOUR_US = 3_600_000_000
HUNDRED_MILLION_PER_HOUR_TPS = 100_000_000 / 3600.0


def _ipv4(first_octet: int, value: int) -> str:
    return (
        f"{first_octet}."
        f"{(value >> 16) & 255}."
        f"{(value >> 8) & 255}."
        f"{value & 255}"
    )


def _payload_chunk(
    start: int,
    count: int,
    total_events: int,
    customer_rows: list[tuple[str, str]],
    merchant_rows: list[tuple[str, int, str]],
) -> list[dict]:
    payloads: list[dict] = []
    customer_count = len(customer_rows)
    merchant_count = len(merchant_rows)

    for index in range(start, start + count):
        customer_index = index % customer_count
        customer_id, known_device = customer_rows[customer_index]
        merchant_id, mcc, merchant_country = merchant_rows[(index // 7) % merchant_count]

        # Most customers have their own stable IP. Three customers in each
        # 100-customer block deliberately share one IP so graph analysis stays
        # active while the evidence-weighted IP rule must avoid false rings.
        if customer_index % 100 < 3:
            shared_group = customer_index // 100
            ip_address = _ipv4(172, shared_group + 1)
        else:
            ip_address = _ipv4(10, customer_index + 1)

        offset_us = (index * ONE_HOUR_US) // total_events
        timestamp = BASE_TIME + timedelta(microseconds=offset_us)

        payloads.append(
            {
                "transaction_id": f"SCALE_{index:012d}",
                "customer_id": customer_id,
                "merchant_id": merchant_id,
                "mcc": mcc,
                "amount": 25.0 + float(index % 475),
                "currency": "USD",
                "pos_entry_mode": "ECOM",
                "3ds_status": "Y",
                "ip_address": ip_address,
                "ip_country": merchant_country,
                "device_id": known_device,
                "stolen_resource": (
                    "phished_credentials" if index % 77 == 0 else None
                ),
                "timestamp": timestamp,
            }
        )
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--customers", type=int, default=1_000)
    parser.add_argument("--maxlen", type=int, default=10_000)
    parser.add_argument("--event-retention", type=int, default=50_000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.events < 1:
        parser.error("--events must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.customers < 3:
        parser.error("--customers must be >= 3")
    if args.maxlen < 1:
        parser.error("--maxlen must be >= 1")
    if args.event_retention < 1:
        parser.error("--event-retention must be >= 1")

    env = PaymentEnvironment(
        n_customers=args.customers,
        seed=42,
        history_size=100,
        event_stream_maxlen=args.event_retention,
        gate_rejects_maxlen=min(args.event_retention, 10_000),
    )
    scorer = VelocityScorer()
    scorer.extractor = FeatureExtractor(env=env, maxlen=args.maxlen)
    engine = DecisionEngine(
        environment=env,
        scorer=scorer,
        novelty=NoveltyDetector(),
    )

    if engine.scorer.extractor.backend != "rust":
        raise SystemExit("arena_core must be installed for the scale benchmark")
    if engine.graph.backend != "rust":
        raise SystemExit("arena_graph_core must be installed for the scale benchmark")
    if engine.scorer.model is None:
        raise SystemExit("XGBoost model must be available for the full-stack benchmark")
    if engine.novelty.model is None:
        raise SystemExit("IsolationForest model must be available for the full-stack benchmark")

    customer_rows = [
        (customer_id, customer.devices[0])
        for customer_id, customer in env.customers.items()
    ]
    merchant_rows = [
        (merchant.merchant_id, merchant.mcc, merchant.country)
        for merchant in env.merchant_registry.values()
    ]

    totals = DecisionEngine._new_cost_totals()
    decision_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    attack_rows = 0
    ring_rows = 0
    anomaly_rows = 0
    analyzed_rows = 0
    checksum = 0.0

    prepare_seconds = 0.0
    model_and_fusion_seconds = 0.0
    cost_seconds = 0.0

    wall_started = time.perf_counter()
    for chunk_start in range(0, args.events, args.chunk_size):
        chunk_count = min(args.chunk_size, args.events - chunk_start)
        raw_payloads = _payload_chunk(
            chunk_start,
            chunk_count,
            args.events,
            customer_rows,
            merchant_rows,
        )

        prepared = []
        truths: list[str] = []

        started = time.perf_counter()
        for raw in raw_payloads:
            msg = PaymentMessage.model_validate(raw)
            gate = env.ingest(msg)
            gate_counts[gate["reason"]] += 1
            if not gate["accepted"]:
                continue

            truth = "attack" if msg.stolen_resource is not None else "legit"
            # ``env.ingest`` has already validated and serialized this exact
            # message into the internal event. Reuse that wire representation
            # instead of coercing/model-dumping the same transaction again.
            prepared.append(
                engine.prepare_wire_for_batch(gate["internal_event"]["payload"])
            )
            truths.append(truth)
        prepare_seconds += time.perf_counter() - started

        started = time.perf_counter()
        records = engine.finalize_batch(prepared)
        model_and_fusion_seconds += time.perf_counter() - started

        started = time.perf_counter()
        for record, truth in zip(records, truths):
            engine.apply_to_running_totals(totals, record, truth)
            decision_counts[record["decision"]] += 1
            attack_rows += int(truth == "attack")
            ring_rows += int(record["scores"]["ring_detected"])
            anomaly_rows += int(record["scores"]["is_anomaly"])
            checksum += (
                float(record["scores"]["velocity"])
                + float(record["scores"]["novelty_anomaly"])
                + float(record["scores"]["ring_risk"])
            )
        cost_seconds += time.perf_counter() - started
        analyzed_rows += len(records)

    wall_seconds = time.perf_counter() - wall_started
    measured_seconds = prepare_seconds + model_and_fusion_seconds + cost_seconds
    throughput = analyzed_rows / measured_seconds
    requested_run_tps = args.events / 3600.0
    target_100m_tps = HUNDRED_MILLION_PER_HOUR_TPS

    result = {
        "benchmark_profile": "synthetic-throughput-stress",
        "quality_metrics_representative": False,
        "quality_metrics_note": (
            "Decision/anomaly/cost counts are stress diagnostics only because the "
            "synthetic population is compressed into a one-hour event-time span. "
            "Use held-out calibrated evaluation/model-sweep results for fraud-quality claims."
        ),
        "scope": (
            "Pydantic validation + Plausibility Gate/enrichment/history + "
            "Rust rolling features + Rust evidence-weighted graph risk + "
            "XGBoost + IsolationForest + decision ladder + cost accounting; "
            "excludes payload generation, model loading, LLM generation and "
            "WebSocket/UI serialization"
        ),
        "events_requested": args.events,
        "events_seen_total": env.events_seen_total,
        "events_analyzed": analyzed_rows,
        "events_retained": len(env.event_stream),
        "customers": args.customers,
        "chunk_size": args.chunk_size,
        "feature_maxlen": args.maxlen,
        "feature_backend": engine.scorer.extractor.backend,
        "graph_backend": engine.graph.backend,
        "graph_risk_state_sizes": engine.graph.risk_state_sizes(),
        "xgb_model": engine.scorer.model_source,
        "iforest_model": engine.novelty.model_source,
        "synthetic_event_time_span_seconds": 3600,
        "target_requested_events_per_hour_tps": round(requested_run_tps, 2),
        "target_100m_per_hour_tps": round(target_100m_tps, 2),
        "measured_analysis_seconds": round(measured_seconds, 6),
        "wall_seconds_including_payload_generation": round(wall_seconds, 6),
        "transactions_per_second": round(throughput, 2),
        "projected_transactions_per_hour_at_measured_rate": round(throughput * 3600),
        "headroom_vs_requested_hourly_rate": round(throughput / requested_run_tps, 3),
        "headroom_vs_100m_hourly_rate": round(throughput / target_100m_tps, 3),
        "meets_requested_one_hour_target": bool(throughput >= requested_run_tps),
        "meets_100m_per_hour_target": bool(throughput >= target_100m_tps),
        "stage_seconds": {
            "schema_gate_enrichment_feature_graph_prepare": round(prepare_seconds, 6),
            "xgb_iforest_and_decision_fusion": round(model_and_fusion_seconds, 6),
            "cost_accounting_and_result_checks": round(cost_seconds, 6),
        },
        "gate_counts": dict(gate_counts),
        "decision_counts": dict(decision_counts),
        "attack_rows": attack_rows,
        "ring_rows": ring_rows,
        "anomaly_rows": anomaly_rows,
        "costs": engine.summarize_totals(totals),
        "feature_state_sizes": engine.scorer.extractor.state_sizes(),
        "graph_nodes": engine.graph.g.number_of_nodes(),
        "graph_edges": engine.graph.g.number_of_edges(),
        "checksum": round(checksum, 6),
    }

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
