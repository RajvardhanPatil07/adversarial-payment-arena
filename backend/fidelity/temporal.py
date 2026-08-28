"""Temporal / behavioural fidelity.

Row-level fidelity asks "does one synthetic transaction look plausible?".
That is the wrong question for payment fraud, where the signal lives in
sequences: how fast, how bursty, how long a campaign runs, what time of day.
A generator that resamples columns independently can score well on marginals
and still produce a stream with no velocity structure at all -- and velocity is
exactly what a production fraud rule fires on.

All measures operate on a long DataFrame with at least:
    customer, ts (epoch seconds), amount, mcc, entry_mode
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .divergence import (
    compare_categorical,
    compare_numeric,
    composite_similarity,
)

REQUIRED_COLUMNS = ("customer", "ts", "amount")


def _require(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def inter_arrival_times(df: pd.DataFrame, by: str = "customer") -> np.ndarray:
    """Gaps in seconds between consecutive events of the same entity."""
    _require(df)
    out: List[float] = []
    for _, g in df.sort_values("ts").groupby(by, sort=False):
        ts = g["ts"].to_numpy(dtype=float)
        if ts.size > 1:
            out.extend(np.diff(ts).tolist())
    return np.asarray(out, dtype=float)


def burstiness(df: pd.DataFrame, by: str = "customer", min_events: int = 3) -> np.ndarray:
    """Goh-Barabasi burstiness B = (sd - mean) / (sd + mean) per entity, in [-1, 1].

    B near -1 is perfectly regular, 0 is Poisson, near +1 is heavily bursty.
    Attack campaigns are strongly positive; independent resampling lands near 0.
    """
    _require(df)
    vals: List[float] = []
    for _, g in df.sort_values("ts").groupby(by, sort=False):
        ts = g["ts"].to_numpy(dtype=float)
        if ts.size < min_events:
            continue
        gaps = np.diff(ts)
        mu = float(gaps.mean())
        sd = float(gaps.std())
        if mu + sd <= 0:
            continue
        vals.append((sd - mu) / (sd + mu))
    return np.asarray(vals, dtype=float)


def velocity_counts(df: pd.DataFrame, window_s: float, by: str = "customer") -> np.ndarray:
    """For each row, how many events the same entity had in the trailing window."""
    _require(df)
    counts = np.zeros(len(df), dtype=float)
    ordered = df.sort_values("ts")
    positions = {label: i for i, label in enumerate(df.index)}
    for _, g in ordered.groupby(by, sort=False):
        ts = g["ts"].to_numpy(dtype=float)
        left = np.searchsorted(ts, ts - window_s, side="left")
        c = np.arange(ts.size) - left
        for label, v in zip(g.index, c):
            counts[positions[label]] = float(v)
    return counts


def sequence_lengths(df: pd.DataFrame, by: str = "customer") -> np.ndarray:
    _require(df)
    return df.groupby(by, sort=False).size().to_numpy(dtype=float)


def campaign_durations(df: pd.DataFrame, by: str = "customer") -> np.ndarray:
    """Seconds from first to last event per entity."""
    _require(df)
    g = df.groupby(by, sort=False)["ts"]
    return (g.max() - g.min()).to_numpy(dtype=float)


def hours_of_day(df: pd.DataFrame) -> np.ndarray:
    _require(df)
    return (df["ts"].to_numpy(dtype=float) / 3600.0) % 24.0


def night_share(df: pd.DataFrame, start: float = 0.0, end: float = 6.0) -> float:
    h = hours_of_day(df)
    if h.size == 0:
        return float("nan")
    return float(np.mean((h >= start) & (h < end)))


def temporal_fidelity_report(
    real: pd.DataFrame, synth: pd.DataFrame, by: str = "customer"
) -> Dict[str, object]:
    """Compare the temporal behaviour of two event streams."""
    _require(real)
    _require(synth)
    records: List[Dict[str, object]] = [
        compare_numeric(
            "inter_arrival_seconds",
            np.log1p(inter_arrival_times(real, by)),
            np.log1p(inter_arrival_times(synth, by)),
        ),
        compare_numeric("burstiness", burstiness(real, by), burstiness(synth, by)),
    ]
    # Attack campaigns burst per RING (shared device), not per victim: a ring
    # rotates victims, which dilutes per-customer burstiness. Measuring both
    # groupings is the difference between seeing the burst and missing it.
    if "device" in real.columns and "device" in synth.columns:
        records.append(
            compare_numeric(
                "burstiness_by_device",
                burstiness(real, "device"),
                burstiness(synth, "device"),
            )
        )
        records.append(
            compare_numeric(
                "inter_arrival_seconds_by_device",
                np.log1p(inter_arrival_times(real, "device")),
                np.log1p(inter_arrival_times(synth, "device")),
            )
        )
    records += [
        compare_numeric(
            "velocity_5m", velocity_counts(real, 300.0, by), velocity_counts(synth, 300.0, by)
        ),
        compare_numeric(
            "velocity_1h", velocity_counts(real, 3600.0, by), velocity_counts(synth, 3600.0, by)
        ),
        compare_numeric(
            "sequence_length", sequence_lengths(real, by), sequence_lengths(synth, by)
        ),
        compare_numeric(
            "campaign_duration_seconds",
            np.log1p(campaign_durations(real, by)),
            np.log1p(campaign_durations(synth, by)),
        ),
        compare_numeric("hour_of_day", hours_of_day(real), hours_of_day(synth)),
    ]
    if "entry_mode" in real.columns and "entry_mode" in synth.columns:
        records.append(
            compare_categorical(
                "entry_mode_given_night",
                real.loc[hours_of_day(real) < 6, "entry_mode"],
                synth.loc[hours_of_day(synth) < 6, "entry_mode"],
            )
        )
    return {
        "family": "temporal",
        "grouped_by": by,
        "measures": records,
        "composite_similarity": composite_similarity(records),
        "real_night_share": round(night_share(real), 6),
        "synth_night_share": round(night_share(synth), 6),
        "interpretation": (
            "composite_similarity is the unweighted mean of per-measure (1 - KS) or "
            "(1 - TVD). 1.0 means indistinguishable temporal behaviour; a generator "
            "that resamples columns independently typically scores high on row-level "
            "marginals and low here."
        ),
    }
