"""Microbenchmark the rolling feature-state integration.

This intentionally benchmarks FeatureExtractor.features()+observe(), not a bare
Rust function, so Python/Rust boundary conversion and the final feature-dict
construction are included. It does not benchmark XGBoost inference.

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


def _payloads(kind: str, count: int) -> list[dict]:
    base = datetime(2026, 8, 30, 4, 0, tzinfo=timezone.utc)
    payloads: list[dict] = []

    for index in range(count):
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
                "timestamp": (base + timedelta(seconds=seconds)).isoformat(),
            }
        )
    return payloads


def _run_once(payloads: list[dict], native: bool, maxlen: int) -> tuple[float, float]:
    extractor = FeatureExtractor(maxlen=maxlen)
    if not native:
        extractor._rust = None
    elif extractor.backend != "rust":
        raise RuntimeError("arena_core is not installed; native benchmark unavailable")

    checksum = 0.0
    started = time.perf_counter()
    for payload in payloads:
        features = extractor.features(payload)
        checksum += features["cust_txn_count_10m"] + features["amount_over_mean30"]
        extractor.observe(payload)
    elapsed = time.perf_counter() - started
    return elapsed, checksum


def _measure(payloads: list[dict], native: bool, repeats: int, maxlen: int) -> dict:
    timings: list[float] = []
    checksums: list[float] = []

    # One unreported warm-up gives import caches/branch predictors a chance to
    # settle without mutating the measured extractor instances.
    warmup = payloads[: min(1000, len(payloads))]
    _run_once(warmup, native=native, maxlen=maxlen)

    for _ in range(repeats):
        elapsed, checksum = _run_once(payloads, native=native, maxlen=maxlen)
        timings.append(elapsed)
        checksums.append(checksum)

    if max(checksums) - min(checksums) > 1e-9:
        raise AssertionError("non-deterministic benchmark checksum")

    median_s = statistics.median(timings)
    return {
        "median_seconds": round(median_s, 6),
        "events_per_second": round(len(payloads) / median_s, 2),
        "min_seconds": round(min(timings), 6),
        "max_seconds": round(max(timings), 6),
        "checksum": round(checksums[0], 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=500)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    probe = FeatureExtractor(maxlen=args.maxlen)
    if probe.backend != "rust":
        raise SystemExit("arena_core must be installed before running this benchmark")

    results = {
        "scope": "FeatureExtractor.features+observe only; excludes XGBoost inference",
        "events_per_workload": args.events,
        "repeats": args.repeats,
        "maxlen": args.maxlen,
        "workloads": {},
    }

    for kind in ("hot", "windowed", "mixed"):
        payloads = _payloads(kind, args.events)
        python_result = _measure(payloads, native=False, repeats=args.repeats, maxlen=args.maxlen)
        rust_result = _measure(payloads, native=True, repeats=args.repeats, maxlen=args.maxlen)
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
