"""Self-contained payment stream simulator and two attack generators.

The generators are deliberately lightweight (numpy+pandas only). A1 preserves
marginals while destroying identity/sequence structure. A2 regenerates ring
topology and temporal bursts while resampling row attributes jointly conditional
on generated time-of-day.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

MCCS: Sequence[int] = (5411, 5812, 5912, 4111, 4812, 5999, 6011, 7011, 5732, 4900)
ENTRY_MODES: Sequence[str] = ("CHIP", "CONTACTLESS", "ECOM", "CNP", "SWIPE")
DAY = 86400.0
HORIZON_DAYS = 30


def _hour_of(ts: np.ndarray) -> np.ndarray:
    return (ts / 3600.0) % 24.0


def _round_snap(amounts: np.ndarray, rng: np.random.Generator, share: float = 0.12) -> np.ndarray:
    out = amounts.copy()
    idx = rng.uniform(size=out.size) < share
    out[idx] = np.round(out[idx] / 100.0) * 100.0
    return np.maximum(out, 10.0)


def _entry_mode_for(mcc: int, amount: float, rng: np.random.Generator) -> str:
    if mcc in (4812, 5999, 5732):
        return "ECOM" if rng.uniform() < 0.85 else "CNP"
    if amount < 1500:
        return "CONTACTLESS" if rng.uniform() < 0.7 else "CHIP"
    return "CHIP" if rng.uniform() < 0.8 else "SWIPE"


def simulate_legit(n_rows: int = 6000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_customers = max(40, n_rows // 14)
    rows: List[Dict[str, object]] = []
    for c in range(n_customers):
        cust = f"CUST_{c:05d}"
        n_dev = 1 if rng.uniform() < 0.8 else 2
        devices = [f"DEV_{c:05d}_{d}" for d in range(n_dev)]
        ips = [f"IP_{c:05d}_{d}" for d in range(n_dev)]
        k = max(1, int(rng.poisson(n_rows / n_customers)))
        t = rng.uniform(0, 3 * DAY)
        for _ in range(k):
            t += float(rng.exponential(0.9 * DAY / 3.0))
            if t > HORIZON_DAYS * DAY:
                break
            day_frac = np.clip(rng.normal(0.55, 0.16), 0.02, 0.98)
            ts = (t // DAY) * DAY + day_frac * DAY
            mcc = int(
                rng.choice(
                    MCCS,
                    p=[0.24, 0.16, 0.07, 0.08, 0.09, 0.08, 0.06, 0.05, 0.09, 0.08],
                )
            )
            amount = float(rng.lognormal(mean=6.6, sigma=0.85))
            j = 0 if rng.uniform() < 0.85 else min(1, n_dev - 1)
            rows.append(
                {
                    "customer": cust,
                    "device": devices[j],
                    "ip": ips[j],
                    "merchant": f"MERCH_{int(rng.integers(0, 220)):04d}",
                    "mcc": mcc,
                    "amount": amount,
                    "ts": float(ts),
                    "entry_mode": _entry_mode_for(mcc, amount, rng),
                    "label": 0,
                    "attack_id": "LEGIT",
                }
            )
    df = pd.DataFrame(rows)
    df["amount"] = _round_snap(df["amount"].to_numpy(dtype=float), rng, 0.14)
    return df.sort_values("ts").reset_index(drop=True).head(n_rows)


def simulate_real_fraud(
    n_rows: int = 400,
    seed: int = 23,
    n_rings: int = 8,
    attack_id: str = "RING_CNP_BURST",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    per_ring = max(6, n_rows // n_rings)
    for r in range(n_rings):
        ring_devices = [f"DEV_RING{r}_{i}" for i in range(int(rng.integers(1, 4)))]
        ring_ips = [f"IP_RING{r}_{i}" for i in range(int(rng.integers(1, 3)))]
        victims = [f"CUST_{int(rng.integers(0, 900)):05d}" for _ in range(int(rng.integers(3, 8)))]
        merchants = [f"MERCH_{int(rng.integers(0, 40)):04d}" for _ in range(2)]
        t = float(rng.uniform(2, HORIZON_DAYS - 2)) * DAY + float(rng.uniform(0, 5)) * 3600.0
        for i in range(per_ring):
            t += (
                float(rng.uniform(15, 90))
                if rng.uniform() < 0.8
                else float(rng.uniform(1, 4) * 3600)
            )
            mcc = int(rng.choice([4812, 5999, 5732, 7011]))
            amount = float(rng.lognormal(mean=8.1, sigma=0.55))
            rows.append(
                {
                    "customer": victims[i % len(victims)],
                    "device": ring_devices[i % len(ring_devices)],
                    "ip": ring_ips[i % len(ring_ips)],
                    "merchant": merchants[i % len(merchants)],
                    "mcc": mcc,
                    "amount": amount,
                    "ts": float(t),
                    "entry_mode": "CNP" if rng.uniform() < 0.75 else "ECOM",
                    "label": 1,
                    "attack_id": attack_id,
                }
            )
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True).head(n_rows)


def synth_marginal(real_fraud: pd.DataFrame, n_rows: int, seed: int = 101) -> pd.DataFrame:
    """A1: preserve one-column marginals and deliberately destroy structure."""
    rng = np.random.default_rng(seed)
    src = real_fraud
    out = pd.DataFrame(
        {
            "customer": [f"SYN_C_{i}" for i in range(n_rows)],
            "device": [f"SYN_D_{i}" for i in range(n_rows)],
            "ip": [f"SYN_I_{i}" for i in range(n_rows)],
            "merchant": [f"SYN_M_{i}" for i in range(n_rows)],
            "mcc": rng.choice(src["mcc"].to_numpy(), size=n_rows, replace=True),
            "amount": rng.choice(src["amount"].to_numpy(dtype=float), size=n_rows, replace=True),
            "ts": rng.uniform(0.0, HORIZON_DAYS * DAY, size=n_rows),
            "entry_mode": rng.choice(src["entry_mode"].to_numpy(), size=n_rows, replace=True),
            "label": 1,
            "attack_id": "SYNTH_MARGINAL",
        }
    )
    return out.sort_values("ts").reset_index(drop=True)


def _within_device_gap_pool(src: pd.DataFrame) -> np.ndarray:
    gaps: List[float] = []
    for _, group in src.sort_values("ts").groupby("device", sort=False):
        ts = group["ts"].to_numpy(dtype=float)
        if ts.size > 1:
            gaps.extend(np.diff(ts).tolist())
    out = np.asarray([g for g in gaps if g > 0], dtype=float)
    return out if out.size else np.asarray([60.0], dtype=float)


def synth_joint_behavioural(
    real_fraud: pd.DataFrame,
    n_rows: int,
    seed: int = 101,
) -> pd.DataFrame:
    """A2: regenerate topology/bursts and preserve observable row dependencies.

    Temporal gaps are sampled only from *within-device* real gaps, avoiding
    cross-ring gaps. For each generated timestamp, amount/MCC/entry-mode are
    resampled jointly from real fraud occurring in the same 4-hour band. Amount
    receives small multiplicative jitter, so rows are not copied verbatim.
    """
    rng = np.random.default_rng(seed)
    src = real_fraud.sort_values("ts").reset_index(drop=True).copy()
    if src.empty:
        raise ValueError("real_fraud must contain at least one row")

    gaps = _within_device_gap_pool(src)
    ring_sizes = src.groupby("device", sort=False)["customer"].nunique().to_numpy(dtype=float)
    ring_sizes = ring_sizes[ring_sizes > 0]
    if ring_sizes.size == 0:
        ring_sizes = np.asarray([3.0])

    src_hours = _hour_of(src["ts"].to_numpy(dtype=float))
    hour_band = (src_hours // 4).astype(int)
    pools: Dict[int, np.ndarray] = {}
    for band in range(6):
        idx = np.flatnonzero(hour_band == band)
        if idx.size:
            pools[band] = idx
    all_idx = np.arange(len(src))

    rows: List[Dict[str, object]] = []
    ring = 0
    while len(rows) < n_rows:
        n_dev = max(1, int(rng.integers(1, 4)))
        devices = [f"GEN_D_{ring}_{i}" for i in range(n_dev)]
        ips = [f"GEN_I_{ring}_{i}" for i in range(max(1, n_dev - 1))]
        n_victims = max(2, int(rng.choice(ring_sizes)))
        victims = [f"GEN_C_{ring}_{i}" for i in range(n_victims)]
        merchants = [f"GEN_M_{ring}_{i}" for i in range(2)]

        start_idx = int(rng.choice(all_idx))
        start_hour = float(src_hours[start_idx])
        t = float(rng.integers(2, HORIZON_DAYS - 2)) * DAY + start_hour * 3600.0
        burst = min(n_rows - len(rows), max(4, int(rng.integers(6, 40))))

        for i in range(burst):
            t += float(rng.choice(gaps))
            band = int((_hour_of(np.asarray([t]))[0] // 4) % 6)
            pool = pools.get(band, all_idx)
            source_row = src.iloc[int(rng.choice(pool))]
            amount = max(
                10.0,
                float(source_row["amount"]) * float(np.exp(rng.normal(0.0, 0.035))),
            )
            rows.append(
                {
                    "customer": victims[i % len(victims)],
                    "device": devices[i % len(devices)],
                    "ip": ips[i % len(ips)],
                    "merchant": merchants[i % len(merchants)],
                    "mcc": int(source_row["mcc"]),
                    "amount": amount,
                    "ts": float(t),
                    "entry_mode": str(source_row["entry_mode"]),
                    "label": 1,
                    "attack_id": "SYNTH_JOINT_BEHAVIOURAL",
                }
            )
        ring += 1

    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True).head(n_rows)


SYNTHESIZERS = {
    "marginal": synth_marginal,
    "joint_behavioural": synth_joint_behavioural,
}
