"""Distribution distances, numpy-only (no scipy).

Every distance here returns a raw divergence AND a bounded similarity in [0, 1],
because a fidelity dashboard that mixes unbounded and bounded measures cannot be
averaged honestly. The similarity transform used for each measure is recorded in
the returned dict so the reader can recompute it.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

_EPS = 1e-12


def _clean(a: Sequence[float]) -> np.ndarray:
    x = np.asarray(a, dtype=float).ravel()
    return x[np.isfinite(x)]


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), in [0, 1]."""
    x, y = _clean(a), _clean(b)
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.union1d(x, y)
    cx = np.searchsorted(np.sort(x), grid, side="right") / x.size
    cy = np.searchsorted(np.sort(y), grid, side="right") / y.size
    return float(np.max(np.abs(cx - cy)))


def wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D Wasserstein-1 distance via the quantile-difference integral."""
    x, y = _clean(a), _clean(b)
    if x.size == 0 or y.size == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, 1001)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def normalised_wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    """Wasserstein-1 scaled by the pooled IQR, so it is unit-free.

    CAVEAT: for a bimodal mixture the pooled IQR itself tracks the separation, so
    a large real shift can normalise to about 1.0. Read this alongside KS.
    """
    x, y = _clean(a), _clean(b)
    if x.size == 0 or y.size == 0:
        return float("nan")
    pooled = np.concatenate([x, y])
    iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
    scale = iqr if iqr > _EPS else max(float(np.std(pooled)), _EPS)
    return float(wasserstein1(x, y) / scale)


def tvd_categorical(a: Iterable, b: Iterable) -> float:
    """Total variation distance between two categorical samples, in [0, 1]."""
    va, ca = np.unique(np.asarray(list(a), dtype=object), return_counts=True)
    vb, cb = np.unique(np.asarray(list(b), dtype=object), return_counts=True)
    if ca.sum() == 0 or cb.sum() == 0:
        return float("nan")
    keys = list(dict.fromkeys(list(va) + list(vb)))
    pa = {k: v for k, v in zip(va, ca / ca.sum())}
    pb = {k: v for k, v in zip(vb, cb / cb.sum())}
    return float(0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys))


def jsd_numeric(a: Sequence[float], b: Sequence[float], bins: int = 20) -> float:
    """Jensen-Shannon divergence (base 2) on a shared histogram grid, in [0, 1]."""
    x, y = _clean(a), _clean(b)
    if x.size == 0 or y.size == 0:
        return float("nan")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if hi - lo < _EPS:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(x, bins=edges)[0].astype(float)
    q = np.histogram(y, bins=edges)[0].astype(float)
    p = p / max(p.sum(), _EPS)
    q = q / max(q.sum(), _EPS)
    m = 0.5 * (p + q)

    def _kl(u: np.ndarray, v: np.ndarray) -> float:
        mask = u > 0
        return float(np.sum(u[mask] * np.log2(u[mask] / np.maximum(v[mask], _EPS))))

    return float(max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))))


def compare_numeric(
    name: str, real: Sequence[float], synth: Sequence[float]
) -> Dict[str, object]:
    """Full numeric comparison record for one measure."""
    ks = ks_statistic(real, synth)
    nw1 = normalised_wasserstein1(real, synth)
    jsd = jsd_numeric(real, synth)
    similarity = float("nan") if not np.isfinite(ks) else float(1.0 - ks)
    r, s = _clean(real), _clean(synth)
    return {
        "measure": name,
        "kind": "numeric",
        "ks": None if not np.isfinite(ks) else round(float(ks), 6),
        "normalised_w1": None if not np.isfinite(nw1) else round(float(nw1), 6),
        "jsd": None if not np.isfinite(jsd) else round(float(jsd), 6),
        "similarity": None if not np.isfinite(similarity) else round(similarity, 6),
        "similarity_transform": "1 - KS",
        "real_median": None if r.size == 0 else round(float(np.median(r)), 6),
        "synth_median": None if s.size == 0 else round(float(np.median(s)), 6),
        "real_n": int(r.size),
        "synth_n": int(s.size),
    }


def compare_categorical(name: str, real: Iterable, synth: Iterable) -> Dict[str, object]:
    tvd = tvd_categorical(real, synth)
    return {
        "measure": name,
        "kind": "categorical",
        "tvd": None if not np.isfinite(tvd) else round(float(tvd), 6),
        "similarity": None if not np.isfinite(tvd) else round(float(1.0 - tvd), 6),
        "similarity_transform": "1 - TVD",
    }


def composite_similarity(records: Sequence[Dict[str, object]]) -> Optional[float]:
    """Unweighted mean of per-measure similarities. Unweighted on purpose: any
    weighting would let a failing measure be hidden behind a chosen weight."""
    vals = [
        float(r["similarity"])
        for r in records
        if r.get("similarity") is not None and np.isfinite(float(r["similarity"]))
    ]
    if not vals:
        return None
    return round(float(np.mean(vals)), 6)
