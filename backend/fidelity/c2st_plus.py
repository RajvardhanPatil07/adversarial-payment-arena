"""C2ST with an interval, a null test, and an explanation.

A bare C2ST AUC is not a finding. Three things are added here:

1. Bootstrap confidence interval on the out-of-fold AUC, so "0.73" cannot be
   quoted as if it were exact.
2. Permutation null test. Labels are shuffled and the whole procedure refit, so
   the reader can see what AUC this classifier reaches on data that is
   indistinguishable by construction. Without this, a small-sample AUC of 0.62
   looks like a finding when it is noise.
3. Feature attribution. Impurity-decrease importances say WHICH columns give the
   generator away, which turns a failing gate into an actionable defect report.

Also provides sliced C2ST: a global AUC can hide a generator that is realistic
on average and hopeless for one attack family or amount band.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ml.forest import RandomForestBinary, oof_scores, roc_auc

DEFAULT_GATE = 0.80


def _bootstrap_auc_ci(
    y: np.ndarray, scores: np.ndarray, n_resamples: int = 400, alpha: float = 0.05, seed: int = 7
) -> Dict[str, Optional[float]]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if pos.size < 2 or neg.size < 2:
        return {"lo": None, "hi": None}
    vals: List[float] = []
    for _ in range(n_resamples):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        a = roc_auc(y[idx], scores[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return {"lo": None, "hi": None}
    return {
        "lo": round(float(np.quantile(vals, alpha / 2)), 6),
        "hi": round(float(np.quantile(vals, 1 - alpha / 2)), 6),
    }


def c2st_report(
    real_X: np.ndarray,
    synth_X: np.ndarray,
    feature_names: Sequence[str],
    gate: float = DEFAULT_GATE,
    n_estimators: int = 90,
    k: int = 3,
    seed: int = 0,
    n_permutations: int = 12,
    top_k: int = 5,
) -> Dict[str, object]:
    """Real rows are label 1, synthetic rows are label 0.

    AUC 0.5 => indistinguishable. AUC 1.0 => trivially separable.
    """
    real_X = np.asarray(real_X, dtype=float)
    synth_X = np.asarray(synth_X, dtype=float)
    X = np.vstack([real_X, synth_X])
    y = np.concatenate([np.ones(real_X.shape[0], dtype=int), np.zeros(synth_X.shape[0], dtype=int)])
    scores, importances = oof_scores(X, y, n_estimators=n_estimators, k=k, seed=seed)
    auc = roc_auc(y, scores)
    ci = _bootstrap_auc_ci(y, scores, seed=seed + 1)

    # permutation null: same pipeline, labels destroyed
    rng = np.random.default_rng(seed + 2)
    null: List[float] = []
    for _ in range(max(0, n_permutations)):
        yp = rng.permutation(y)
        s, _ = oof_scores(X, yp, n_estimators=max(20, n_estimators // 3), k=k, seed=int(rng.integers(0, 10**6)))
        a = roc_auc(yp, s)
        if np.isfinite(a):
            null.append(a)
    if null:
        p_value = float((1.0 + sum(1 for a in null if a >= auc)) / (1.0 + len(null)))
        null_summary: Dict[str, Optional[float]] = {
            "mean": round(float(np.mean(null)), 6),
            "p95": round(float(np.quantile(null, 0.95)), 6),
            "n_permutations": len(null),
        }
    else:
        p_value = None
        null_summary = {"mean": None, "p95": None, "n_permutations": 0}

    order = np.argsort(importances)[::-1][: max(1, top_k)]
    attribution = [
        {"feature": str(feature_names[i]), "importance": round(float(importances[i]), 6)}
        for i in order
        if i < len(feature_names)
    ]
    return {
        "c2st_auc": None if not np.isfinite(auc) else round(float(auc), 6),
        "ci95": ci,
        "gate": gate,
        "passes_gate": None if not np.isfinite(auc) else bool(auc <= gate),
        "permutation_null": null_summary,
        "permutation_p_value": None if p_value is None else round(p_value, 6),
        "most_discriminative_features": attribution,
        "n_real": int(real_X.shape[0]),
        "n_synth": int(synth_X.shape[0]),
        "policy": (
            "The observed AUC is published whether or not the gate is cleared. "
            "The gate is a policy choice, not a law of statistics; the permutation "
            "null shows what this classifier scores on indistinguishable data."
        ),
    }


def sliced_c2st(
    real_X: np.ndarray,
    synth_X: np.ndarray,
    real_slice: Sequence,
    synth_slice: Sequence,
    feature_names: Sequence[str],
    gate: float = DEFAULT_GATE,
    min_rows: int = 60,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Run C2ST separately within each slice value present in both samples."""
    real_slice = np.asarray(list(real_slice), dtype=object)
    synth_slice = np.asarray(list(synth_slice), dtype=object)
    out: List[Dict[str, object]] = []
    for value in sorted(set(real_slice.tolist()) & set(synth_slice.tolist()), key=str):
        rm = real_slice == value
        sm = synth_slice == value
        if rm.sum() < min_rows or sm.sum() < min_rows:
            out.append(
                {
                    "slice": str(value),
                    "skipped": True,
                    "reason": f"fewer than {min_rows} rows on one side",
                    "n_real": int(rm.sum()),
                    "n_synth": int(sm.sum()),
                }
            )
            continue
        rep = c2st_report(
            real_X[rm], synth_X[sm], feature_names, gate=gate, seed=seed, n_permutations=0
        )
        out.append({"slice": str(value), "skipped": False, **rep})
    return out
