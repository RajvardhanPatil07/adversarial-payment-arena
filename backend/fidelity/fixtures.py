"""Self-contained payment stream simulator and two attack generators.

Why this exists: the repository's own corpus builder needs faker, networkx,
pydantic and xgboost. The behavioural-fidelity experiment must be runnable with
numpy + pandas alone, or nobody will reproduce it. This module provides a
deliberately small but *structured* payment stream: entities are reused,
sequences are bursty, amounts have round-number spikes, and entry mode depends
on MCC and amount. Those are exactly the properties a naive generator destroys.

BOUNDARY: this is a simulator, not issuer data. Absolute recall numbers produced
from it are not claims about live card traffic. It exists to compare generators
under identical conditions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MCCS: Sequence[int] = (5411, 5812, 5912, 4111, 4812, 5999, 6011, 7011, 5732, 4900)
ENTRY_MODES: Sequence[str] = ("CHIP", "CONTACTLESS", "ECOM", "CNP", "SWIPE")
DAY = 86400.0
HORIZON_DAYS = 30


def _hour_of(ts: np.ndarray) -> np.ndarray:
    return (ts / 3600.0) % 24.0


def _round_snap(amounts: np.ndarray, rng: np.random.Generator, share: float = 0.12) -> np.ndarray:
    """Humans pay round numbers. Snap a share of amounts to hundreds."""
    out = amounts.copy()
    idx = rng.uniform(size=out.size) < share
    out[idx] = np.round(out[idx] / 100.0) * 100.0
    return np.maximum(out, 10.0)


def _entry_mode_for(mcc: int, amount: float, rng: np.random.Generator) -> str:
    """Entry mode is conditional on MCC and amount, never independent."""
    if mcc in (4812, 5999, 5732):
        return "ECOM" if rng.uniform() < 0.85 else "CNP"
    if amount < 1500:
        return "CONTACTLESS" if rng.uniform() < 0.7 else "CHIP"
    return "CHIP" if rng.uniform() < 0.8 else "SWIPE"


def simulate_legit(n_rows: int = 6000, seed: int = 11) -> pd.DataFrame:
    """Legitimate traffic: diurnal, per-customer sequences, stable devices."""
    rng = np.random.default_rng(seed)
    n_customers = max(40, n_rows // 14)
    rows: List[Dict[str, object]] = []
    for c in range(n_customers):
        cust = f"CUST_{c:05d}"
        n_dev = 1 if rng.uniform() < 0.8 else 2
        devices = [f"DEV_{c:05d}_{d}" for d in range(n_dev)]
        ips = [f"IP_{c:05d}_{d}" for d in range(n_dev)]
        k = max(1, int(rng.poisson(n_rows / n_customers)))
        # daily rhythm: a few purchases per day, waking hours biased
        t = rng.uniform(0, 3 * DAY)
        for _ in range(k):
            t += float(rng.exponential(0.9 * DAY / 3.0))
            if t > HORIZON_DAYS * DAY:
                break
            day_frac = np.clip(rng.normal(0.55, 0.16), 0.02, 0.98)  # ~13:00 peak
            ts = (t // DAY) * DAY + day_frac * DAY
            mcc = int(rng.choice(MCCS, p=[0.24, 0.16, 0.07, 0.08, 0.09, 0.08, 0.06, 0.05, 0.09, 0.08]))
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
    n_rows: int = 400, seed: int = 23, n_rings: int = 8, attack_id: str = "RING_CNP_BURST"
) -> pd.DataFrame:
    """Ground-truth fraud: rings share devices/IPs, burst hard, prefer night+CNP."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    per_ring = max(6, n_rows // n_rings)
    for r in range(n_rings):
        ring_devices = [f"DEV_RING{r}_{i}" for i in range(int(rng.integers(1, 4)))]
        ring_ips = [f"IP_RING{r}_{i}" for i in range(int(rng.integers(1, 3)))]
        victims = [f"CUST_{int(rng.integers(0, 900)):05d}" for _ in range(int(rng.integers(3, 8)))]
        merchants = [f"MERCH_{int(rng.integers(0, 40)):04d}" for _ in range(2)]
        t0 = float(rng.uniform(2, HORIZON_DAYS - 2)) * DAY + float(rng.uniform(0, 5)) * 3600.0
        t = t0
        for i in range(per_ring):
            # tight bursts with occasional idle gaps -> high burstiness
            t += float(rng.uniform(15, 90)) if rng.uniform() < 0.8 else float(rng.uniform(1, 4) * 3600)
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


# --------------------------------------------------------------------------
# Generators under test. Both see ONLY the fraud rows handed to them.
# --------------------------------------------------------------------------
def synth_marginal(real_fraud: pd.DataFrame, n_rows: int, seed: int = 101) -> pd.DataFrame:
    """Arm A1: resample every column independently.

    This is the parametric / template / rule-based red-team family. Marginals are
    correct by construction. Joint structure, entity reuse and burst timing are
    all destroyed: every row gets a brand-new entity and an independent timestamp.
    """
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


def synth_joint_behavioural(
    real_fraud: pd.DataFrame, n_rows: int, seed: int = 101
) -> pd.DataFrame:
    """Arm A2: preserve the joint AND the behaviour.

    - amount/hour dependence preserved by rank (copula-style) resampling
    - entry_mode sampled CONDITIONAL on amount band, not marginally
    - ring topology regenerated with the real ring-size distribution
    - burst gaps drawn from the real inter-arrival distribution

    This is a generator, not a copy: entities are new, timestamps are new, and
    amounts are perturbed within their rank neighbourhood.
    """
    rng = np.random.default_rng(seed)
    src = real_fraud.sort_values("ts").reset_index(drop=True)
    amounts = np.sort(src["amount"].to_numpy(dtype=float))
    gaps = np.diff(src["ts"].to_numpy(dtype=float))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        gaps = np.asarray([60.0])
    ring_sizes = (
        src.groupby("device", sort=False)["customer"].nunique().to_numpy(dtype=float)
    )
    ring_sizes = ring_sizes[ring_sizes > 0]
    if ring_sizes.size == 0:
        ring_sizes = np.asarray([3.0])
    real_hours = _hour_of(src["ts"].to_numpy(dtype=float))
    mcc_pool = src["mcc"].to_numpy()

    # amount-band conditional entry-mode table, learned from the real fraud
    bands = np.quantile(src["amount"].to_numpy(dtype=float), [0.0, 0.33, 0.66, 1.0])
    cond: Dict[int, np.ndarray] = {}
    for b in range(3):
        lo, hi = bands[b], bands[b + 1]
        mask = (src["amount"] >= lo) & (src["amount"] <= hi)
        pool = src.loc[mask, "entry_mode"].to_numpy()
        cond[b] = pool if pool.size else src["entry_mode"].to_numpy()

    rows: List[Dict[str, object]] = []
    ring = 0
    while len(rows) < n_rows:
        n_dev = max(1, int(rng.integers(1, 4)))
        devices = [f"GEN_D_{ring}_{i}" for i in range(n_dev)]
        ips = [f"GEN_I_{ring}_{i}" for i in range(max(1, n_dev - 1))]
        n_victims = max(2, int(rng.choice(ring_sizes)))
        victims = [f"CUST_{int(rng.integers(0, 900)):05d}" for _ in range(n_victims)]
        merchants = [f"MERCH_{int(rng.integers(0, 40)):04d}" for _ in range(2)]
        # start the campaign at an hour drawn from the real hour distribution
        h = float(rng.choice(real_hours))
        t = float(rng.integers(2, HORIZON_DAYS - 2)) * DAY + h * 3600.0
        burst = min(n_rows - len(rows), max(4, int(rng.integers(6, 40))))
        for i in range(burst):
            t += float(rng.choice(gaps))
            # rank-preserving amount draw: pick a rank, jitter within neighbourhood
            u = float(rng.uniform())
            j = int(np.clip(u * (amounts.size - 1), 0, amounts.size - 1))
            lo = max(0, j - 2)
            hi = min(amounts.size - 1, j + 2)
            amount = float(rng.uniform(amounts[lo], amounts[hi])) if hi > lo else float(amounts[j])
            band = 0 if u < 1 / 3 else (1 if u < 2 / 3 else 2)
            rows.append(
                {
                    "customer": victims[i % len(victims)],
                    "device": devices[i % len(devices)],
                    "ip": ips[i % len(ips)],
                    "merchant": merchants[i % len(merchants)],
                    "mcc": int(rng.choice(mcc_pool)),
                    "amount": amount,
                    "ts": float(t),
                    "entry_mode": str(rng.choice(cond[band])),
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
