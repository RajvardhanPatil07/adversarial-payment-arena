"""Operating points that are calibrated honestly.

The most common way a fraud demo overstates itself is picking the threshold on
the same data it reports. Here the threshold is always pinned on a legitimate
VALIDATION split and the realised FPR is then measured on a disjoint test split.
The gap between the two is reported rather than tuned away.

numpy only, deliberately separate from the scipy-era calibration module.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def pin_threshold_at_fpr(legit_validation_scores: Sequence[float], target_fpr: float) -> float:
    """Smallest threshold whose alert rate on legitimate validation data is <= target.

    TIE HANDLING (this is not a detail -- it is the difference between a 1% and a
    100% false-positive rate). A bagged tree ensemble trained on a heavily
    imbalanced corpus assigns EXACTLY 0.0 to the large majority of legitimate
    rows. A plain quantile then lands on that tie block: `np.quantile(s, 0.99)`
    returns 0.0, and because the decision rule is `score >= tau`, every single
    legitimate row alerts. The realised FPR becomes ~100% while the code looks
    like it pinned 1%.

    So the threshold is chosen over the DISTINCT score values, taking the
    smallest whose realised validation alert rate is within budget. If even the
    largest observed score exceeds the budget (all scores tied), we step just
    above it with `nextafter`, which yields an empty alert set rather than a
    silently saturated one -- a conservative, visible failure instead of an
    invisible one.
    """
    s = np.asarray(legit_validation_scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no validation scores supplied")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("target_fpr must be strictly between 0 and 1")

    candidates = np.unique(s)  # ascending, deduplicated: tie blocks collapse
    # Alert rate is monotone non-increasing in tau, so scan upward and stop at
    # the first candidate that fits the budget.
    for tau in candidates:
        if float(np.mean(s >= tau)) <= target_fpr:
            return float(tau)
    return float(np.nextafter(candidates[-1], np.inf))


def rate_at_or_above(scores: Sequence[float], threshold: float) -> float:
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s >= threshold))


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> Optional[float]:
    """Precision implied by (recall, FPR) at a stated base rate.

    Recall and FPR are prevalence-independent; precision is not. Reporting
    precision measured on a balanced test set is the classic overstatement.
    """
    if not np.isfinite(recall) or not np.isfinite(fpr):
        return None
    tp = prevalence * recall
    fp = (1.0 - prevalence) * fpr
    if tp + fp <= 0:
        return None
    return float(tp / (tp + fp))


def operating_point(
    legit_validation: Sequence[float],
    legit_test: Sequence[float],
    fraud_test: Sequence[float],
    target_fpr: float = 0.01,
    production_prevalence: float = 0.013,
) -> Dict[str, object]:
    threshold = pin_threshold_at_fpr(legit_validation, target_fpr)
    realised_fpr = rate_at_or_above(legit_test, threshold)
    recall = rate_at_or_above(fraud_test, threshold)
    precision = precision_at_prevalence(recall, realised_fpr, production_prevalence)
    return {
        "threshold": round(float(threshold), 8),
        "threshold_source": "legitimate validation split (disjoint from test)",
        "target_fpr": target_fpr,
        "realised_test_fpr": None if not np.isfinite(realised_fpr) else round(float(realised_fpr), 6),
        "calibration_gap_pct_points": (
            None
            if not np.isfinite(realised_fpr)
            else round(100.0 * (float(realised_fpr) - target_fpr), 4)
        ),
        "recall_on_held_out_real_fraud": None if not np.isfinite(recall) else round(float(recall), 6),
        "production_prevalence": production_prevalence,
        "precision_at_production_prevalence": None if precision is None else round(precision, 6),
        "n_legit_validation": int(np.asarray(legit_validation).size),
        "n_legit_test": int(np.asarray(legit_test).size),
        "n_fraud_test": int(np.asarray(fraud_test).size),
    }


def bootstrap_mean_ci(
    values: Sequence[Optional[float]],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(float(x))], dtype=float)
    if v.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if v.size == 1:
        return {"mean": round(float(v[0]), 6), "lo": None, "hi": None, "n": 1}
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(v, v.size, replace=True).mean()) for _ in range(n_resamples)]
    return {
        "mean": round(float(v.mean()), 6),
        "lo": round(float(np.quantile(means, alpha / 2)), 6),
        "hi": round(float(np.quantile(means, 1 - alpha / 2)), 6),
        "n": int(v.size),
    }
