"""Privacy attacks against the attack generators.

"It is synthetic, therefore shareable" is an assumption. A generator fit on real
fraud can memorise it, and a red-team corpus that leaks real cardholder
behaviour is a liability, not an asset. These are the cheap standard attacks;
results are published even when unflattering.

No differential privacy is claimed. There is no epsilon here.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ml.forest import RandomForestBinary, roc_auc


def _standardise(reference: np.ndarray, *others: np.ndarray):
    """Scale everything by the reference's own mean/sd so distances are comparable."""
    ref = np.asarray(reference, dtype=float)
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    out = [(ref - mu) / sd]
    for o in others:
        out.append((np.asarray(o, dtype=float) - mu) / sd)
    return tuple(out)


def nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Euclidean distance from each query row to its closest reference row."""
    q = np.asarray(query, dtype=float)
    r = np.asarray(reference, dtype=float)
    if q.size == 0 or r.size == 0:
        return np.asarray([], dtype=float)
    out = np.empty(q.shape[0], dtype=float)
    for i in range(0, q.shape[0], chunk):
        block = q[i : i + chunk]
        d = np.sqrt(((block[:, None, :] - r[None, :, :]) ** 2).sum(axis=2))
        out[i : i + chunk] = d.min(axis=1)
    return out


def duplicate_share(synth: np.ndarray, real: np.ndarray, decimals: int = 3) -> Optional[float]:
    """Share of synthetic rows that exactly match a real row after rounding."""
    s = np.asarray(synth, dtype=float)
    r = np.asarray(real, dtype=float)
    if s.size == 0 or r.size == 0:
        return None
    real_set = {tuple(np.round(row, decimals)) for row in r}
    hits = sum(1 for row in s if tuple(np.round(row, decimals)) in real_set)
    return float(hits / s.shape[0])


def membership_inference_auc(
    members: np.ndarray, non_members: np.ndarray, synth: np.ndarray
) -> Optional[float]:
    """Can an attacker tell training rows from held-out rows using only the
    synthetic sample? Score = negative distance to nearest synthetic row.

    0.5 means no detectable signal. Near 1.0 means the generator memorised.
    """
    if len(members) == 0 or len(non_members) == 0 or len(synth) == 0:
        return None
    syn, mem, non = _standardise(synth, members, non_members)
    d_mem = nearest_distances(mem, syn)
    d_non = nearest_distances(non, syn)
    y = np.concatenate([np.ones(d_mem.size, dtype=int), np.zeros(d_non.size, dtype=int)])
    scores = -np.concatenate([d_mem, d_non])
    auc = roc_auc(y, scores)
    return None if not np.isfinite(auc) else float(auc)


def attribute_inference(
    synth_X: np.ndarray,
    synth_y: Sequence[int],
    real_X: np.ndarray,
    real_y: Sequence[int],
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Train on synthetic, predict a sensitive attribute on real rows.

    Dual-use: high lift means the synthetic data is genuinely useful AND that it
    transfers real structure. Reported as a utility/risk pair, not a pure win.
    """
    sy = np.asarray(synth_y).astype(int).ravel()
    ry = np.asarray(real_y).astype(int).ravel()
    if np.unique(sy).size < 2 or ry.size == 0:
        return {"accuracy": None, "majority_baseline": None, "lift": None}
    model = RandomForestBinary(n_estimators=60, seed=seed).fit(np.asarray(synth_X, float), sy)
    pred = (model.predict_proba(np.asarray(real_X, float)) >= 0.5).astype(int)
    acc = float(np.mean(pred == ry))
    base = float(max(np.mean(ry == 1), np.mean(ry == 0)))
    return {
        "accuracy": round(acc, 6),
        "majority_baseline": round(base, 6),
        "lift": round(acc - base, 6),
    }


def _risk_label(value: Optional[float], warn: float, high: float) -> str:
    if value is None:
        return "unknown"
    if value >= high:
        return "high"
    if value >= warn:
        return "medium"
    return "low"


def privacy_audit(
    generator: str,
    synth_X: np.ndarray,
    train_members_X: np.ndarray,
    holdout_non_members_X: np.ndarray,
) -> Dict[str, object]:
    dup = duplicate_share(synth_X, train_members_X)
    syn, mem = _standardise(synth_X, train_members_X)
    dcr = nearest_distances(syn, mem)
    mi = membership_inference_auc(train_members_X, holdout_non_members_X, synth_X)
    return {
        "generator": generator,
        "exact_duplicate_share": None if dup is None else round(dup, 6),
        "distance_to_closest_real_median": None if dcr.size == 0 else round(float(np.median(dcr)), 6),
        "distance_to_closest_real_p05": None if dcr.size == 0 else round(float(np.quantile(dcr, 0.05)), 6),
        "identical_row_share": None if dcr.size == 0 else round(float(np.mean(dcr < 1e-9)), 6),
        "membership_inference_auc": None if mi is None else round(mi, 6),
        "risk": {
            "duplication": _risk_label(dup, 0.01, 0.05),
            "membership_inference": _risk_label(
                None if mi is None else abs(mi - 0.5) * 2.0, 0.20, 0.40
            ),
        },
        "reading": (
            "membership_inference_auc near 0.50 means the attack cannot distinguish "
            "training rows from held-out rows. Distance-to-closest-real near zero "
            "means near-verbatim copying even when exact duplicates are absent."
        ),
        "boundary": (
            "Small samples, one attack family, no differential privacy guarantee. "
            "A low score here is evidence of no detectable leakage at this sample "
            "size, not proof of privacy."
        ),
    }
