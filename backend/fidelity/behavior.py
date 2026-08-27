"""Feature builders: row-level features vs row+temporal+graph features.

The two feature sets exist so an experiment can answer a specific question:
does a generator that only matches row-level marginals still help a detector
that is allowed to see velocity and entity-graph features? Production fraud
systems always see those features, so evaluating augmentation on row-level
features alone flatters the weak generator.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from .temporal import velocity_counts

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all feature columns attached."""
    out = df.reset_index(drop=True).copy()
    amount = out["amount"].to_numpy(dtype=float)
    ts = out["ts"].to_numpy(dtype=float)
    hour = (ts / 3600.0) % 24.0
    out["log_amount"] = np.log1p(np.maximum(amount, 0.0))
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["amount_round_frac"] = np.abs(amount - np.round(amount / 100.0) * 100.0) / 100.0
    out["mcc_num"] = out["mcc"].astype(float)
    out["entry_mode_code"] = (
        out["entry_mode"].map(ENTRY_MODE_CODES).fillna(-1).astype(float)
    )
    out["velocity_5m"] = velocity_counts(out, 300.0, by="customer")
    out["velocity_1h"] = velocity_counts(out, 3600.0, by="customer")
    out["seq_index"] = out.groupby("customer", sort=False)["ts"].rank(method="first").astype(float)
    prev = out.sort_values("ts").groupby("customer", sort=False)["ts"].diff()
    out["log_gap_prev"] = np.log1p(prev.reindex(out.index).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float))
    for src, tgt, name in (
        ("device", "customer", "device_customer_count"),
        ("ip", "customer", "ip_customer_count"),
        ("merchant", "customer", "merchant_customer_count"),
        ("customer", "device", "customer_device_count"),
    ):
        if src in out.columns and tgt in out.columns:
            counts = out.groupby(src, sort=False)[tgt].nunique()
            out[name] = out[src].map(counts).astype(float)
        else:
            out[name] = 1.0
    return out


def matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"features not built: {missing}")
    return df[list(columns)].to_numpy(dtype=float)
