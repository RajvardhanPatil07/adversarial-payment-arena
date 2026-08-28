"""Leakage-free feature builders for payment event streams.

All stateful features are causal: a row can depend on the current transaction
and earlier transactions only. Building features on a full chronological stream
is therefore safe; earlier train rows never receive information from later
validation/test rows.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Deque, Dict, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ENTRY_MODE_CODES = {"CHIP": 0, "CONTACTLESS": 1, "ECOM": 2, "CNP": 3, "SWIPE": 4}

ROW_FEATURES: Sequence[str] = (
    "log_amount",
    "hour_sin",
    "hour_cos",
    "amount_round_frac",
    "mcc_num",
    "entry_mode_code",
)

BEHAVIOURAL_FEATURES: Sequence[str] = tuple(ROW_FEATURES) + (
    "velocity_5m",
    "velocity_1h",
    "seq_index",
    "log_gap_prev",
    "device_customer_count",
    "ip_customer_count",
    "merchant_customer_count",
    "customer_device_count",
)


def _causal_state_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute sequence, velocity and entity-reuse features in event-time order.

    Entity counts include the current edge after it is observed. For example,
    the first customer seen on a device has device_customer_count=1; the second
    distinct customer later observed on that device has count=2. No future edge
    contributes to any earlier row.
    """
    n = len(df)
    velocity_5m = np.zeros(n, dtype=float)
    velocity_1h = np.zeros(n, dtype=float)
    seq_index = np.zeros(n, dtype=float)
    log_gap_prev = np.zeros(n, dtype=float)
    device_customer_count = np.ones(n, dtype=float)
    ip_customer_count = np.ones(n, dtype=float)
    merchant_customer_count = np.ones(n, dtype=float)
    customer_device_count = np.ones(n, dtype=float)

    windows_5m: DefaultDict[str, Deque[float]] = defaultdict(deque)
    windows_1h: DefaultDict[str, Deque[float]] = defaultdict(deque)
    seen_customer_events: DefaultDict[str, int] = defaultdict(int)
    last_customer_ts: Dict[str, float] = {}

    relation_specs: Tuple[Tuple[str, str, np.ndarray], ...] = (
        ("device", "customer", device_customer_count),
        ("ip", "customer", ip_customer_count),
        ("merchant", "customer", merchant_customer_count),
        ("customer", "device", customer_device_count),
    )
    relation_state: Dict[Tuple[str, str], DefaultDict[str, Set[str]]] = {
        (src, tgt): defaultdict(set)
        for src, tgt, _ in relation_specs
        if src in df.columns and tgt in df.columns
    }

    work = df.reset_index(drop=True).copy()
    order = np.argsort(work["ts"].to_numpy(dtype=float), kind="mergesort")
    for pos in order:
        row = work.iloc[int(pos)]
        ts = float(row["ts"])
        customer = str(row["customer"])

        q5 = windows_5m[customer]
        while q5 and q5[0] < ts - 300.0:
            q5.popleft()
        velocity_5m[pos] = float(len(q5))
        q5.append(ts)

        q1 = windows_1h[customer]
        while q1 and q1[0] < ts - 3600.0:
            q1.popleft()
        velocity_1h[pos] = float(len(q1))
        q1.append(ts)

        seen_customer_events[customer] += 1
        seq_index[pos] = float(seen_customer_events[customer])

        previous = last_customer_ts.get(customer)
        if previous is not None:
            log_gap_prev[pos] = float(np.log1p(max(0.0, ts - previous)))
        last_customer_ts[customer] = ts

        for src, tgt, output in relation_specs:
            state = relation_state.get((src, tgt))
            if state is None:
                output[pos] = 1.0
                continue
            source = str(row[src])
            target = str(row[tgt])
            state[source].add(target)
            output[pos] = float(len(state[source]))

    return {
        "velocity_5m": velocity_5m,
        "velocity_1h": velocity_1h,
        "seq_index": seq_index,
        "log_gap_prev": log_gap_prev,
        "device_customer_count": device_customer_count,
        "ip_customer_count": ip_customer_count,
        "merchant_customer_count": merchant_customer_count,
        "customer_device_count": customer_device_count,
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with row and strictly-causal behavioural features."""
    out = df.reset_index(drop=True).copy()
    required = {"customer", "ts", "amount", "mcc", "entry_mode"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    amount = out["amount"].to_numpy(dtype=float)
    ts = out["ts"].to_numpy(dtype=float)
    hour = (ts / 3600.0) % 24.0
    out["log_amount"] = np.log1p(np.maximum(amount, 0.0))
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["amount_round_frac"] = np.abs(amount - np.round(amount / 100.0) * 100.0) / 100.0
    out["mcc_num"] = out["mcc"].astype(float)
    out["entry_mode_code"] = out["entry_mode"].map(ENTRY_MODE_CODES).fillna(-1).astype(float)

    for name, values in _causal_state_features(out).items():
        out[name] = values
    return out


def matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"features not built: {missing}")
    return df[list(columns)].to_numpy(dtype=float)
