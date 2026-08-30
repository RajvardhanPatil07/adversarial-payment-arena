"""Microbenchmark the rolling feature-state integration.

This intentionally benchmarks FeatureExtractor.features()+observe(), not a bare
Rust function, so Python/Rust boundary conversion and final feature-dict
construction are included. It does not benchmark XGBoost inference.

Large runs are generated in bounded chunks. Payload construction happens outside
the measured sections, so a 100M-event run does not require 100M Python dicts in
RAM and remains comparable with the earlier benchmark scope.

Run after installing the arena_core wheel:
    PYTHONPATH=backend python backend/benchmarks/benchmark_feature_state.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from defense.realtime import FeatureExtractor

WORKLOADS = ("hot", "windowed", "mixed")
BASE_TIME = datetime(2026, 8, 30, 4, 0, tzinfo=timezone.utc)


def _payload_chunk(kind: str, start: int, count: int) -> list[dict]:
    payloads: list[dict] = []
    stop = start + count

    for index in range(start, stop):
        if kind == "hot":
            customer = "CUST_HOT"
            device = "DEV_HOT_001"
            merchant = "MERCH_HOT"
            seconds = index
        elif kind == "windowed":
            customer = "CUST_WINDOW"
            device = "DEV_WINDOW_001"
            merchant = "MERCH_WINDOW"
            seconds = index * 30
        elif kind == "mixed":
            customer = f"CUST_{index % 200:04d}"
            device = f"DEV_{index % 400:04d}"
            merchant = f"MERCH_{index % 20:02d}"
            seconds = index
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(kind)

        payloads.append(
            {
                "customer_id": customer,
                "device_id": device,
                "merchant_id": merchant,
                "amount": 20.0 + float(index % 80),
                "mcc": 5411 + (index % 5),
                "pos_entry_mode": "ECOM",
                "3ds_status": "Y",
                "timestamp": (BASE_TIME + timedelta(seconds=seconds)).isoformat(),
            }
        )
    return payloads


def _run_once(
    kind: str,
    event_count: int,
    native: bool,
    maxlen: int,
    chunk_size: int,
) -> tuple[float, float]:
    extractor = FeatureExtractor(maxlen=maxlen)
    if not native:
        extractor._rust = None
    elif extractor.backend != "rust":
        raise RuntimeError("arena_core is not installed; native benchmark unavailable")

    checksum = 0.0
    measured_elapsed = 0.0

    for start in range(0, event_count, chunk_size):
        count = min(chunk_size, event_count - start)
        # Generate outside the measured section. This keeps the metric focused
        # on the feature-state path while bounding benchmark memory usage.
        payloads = _payload_chunk(kind, start, count)

        started = time.perf_counter()
        for payload in payloads:
            features = extractor.features(payload)
            checksum += features["cust_txn_count_10m"] + features["amount_over_mean30"]
            extractor.observe(payload)
        measured_elapsed += time.perf_counter() - started

    return measured_elapsed, checksum


def _measure(
    kind: str,
    event_count: int,
    native: bool,
    repeats: int,
    maxlen: int,
    chunk_size: int,
) -> dict:
    timings: list[float] = []
    checksums: list[float] = []

    # One unreported warm-up gives import caches/branch predictors a chance to
    # settle without mutating the measured extractor instances.
    warmup_count = min(1000, event_count)
    _run_once(
        kind,
        warmup_count,
        native=native,
        maxlen=maxlen,
        chunk_size=min(chunk_size, warmup_count),
    )

    for _ in range(repeats):
        elapsed, checksum = _run_once(
            kind,
            event_count,
            native=native,
            maxlen=maxlen,
            chunk_size=chunk_size,
        )
        timings.append(elapsed)
        checksums.append(checksum)

    if max(checksums) - min(checksums) > 1e-9:
        raise AssertionError("non-deterministic benchmark checksum")

    median_s = statistics.median(timings)
    return {
        "median_seconds": round(median_s, 6),
        "events_per_second": round(event_count / median_s, 2),
        "min_seconds": round(min(timings), 6),
        "max_seconds": round(max(timings), 6),
        "checksum": round(checksums[0], 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=list(WORKLOADS),
        help="Subset of workloads to run. Defaults to all three.",
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Measure only the Rust/PyO3 backend for large-volume scale runs.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.events < 1:
        parser.error("--events must be >= 1")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.maxlen < 1:
        parser.error("--maxlen must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    probe = FeatureExtractor(maxlen=args.maxlen)
    if probe.backend != "rust":
        raise SystemExit("arena_core must be installed before running this benchmark")

    workloads = list(dict.fromkeys(args.workloads))
    results = {
        "scope": "FeatureExtractor.features+observe only; excludes payload generation and XGBoost inference",
        "events_per_workload": args.events,
        "total_measured_events": args.events * len(workloads) * args.repeats,
        "repeats": args.repeats,
        "maxlen": args.maxlen,
        "chunk_size": args.chunk_size,
        "backend_mode": "rust-only" if args.native_only else "python-vs-rust",
        "selected_workloads": workloads,
        "workloads": {},
    }

    for kind in workloads:
        if args.native_only:
            results["workloads"][kind] = {
                "rust": _measure(
                    kind,
                    args.events,
                    native=True,
                    repeats=args.repeats,
                    maxlen=args.maxlen,
                    chunk_size=args.chunk_size,
                )
            }
            continue

        python_result = _measure(
            kind,
            args.events,
            native=False,
            repeats=args.repeats,
            maxlen=args.maxlen,
            chunk_size=args.chunk_size,
        )
        rust_result = _measure(
            kind,
            args.events,
            native=True,
            repeats=args.repeats,
            maxlen=args.maxlen,
            chunk_size=args.chunk_size,
        )
        if python_result["checksum"] != rust_result["checksum"]:
            raise AssertionError(f"backend checksum mismatch for {kind}")

        results["workloads"][kind] = {
            "python": python_result,
            "rust": rust_result,
            "speedup": round(
                rust_result["events_per_second"] / python_result["events_per_second"], 3
            ),
        }

    encoded = json.dumps(results, indent=2, sort_keys=True)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
