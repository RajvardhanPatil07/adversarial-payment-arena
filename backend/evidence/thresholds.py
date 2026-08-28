"""Operating-point calibration utilities with exact finite-sample FPR control."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def pin_threshold_at_fpr(legit_validation_scores: Sequence[float], target_fpr: float) -> float:
    """Return the smallest score threshold whose empirical ``>=`` FPR is <= target.

    Quantiles alone are not sufficient when model scores contain ties. This
    implementation evaluates the distinct score levels directly. If the highest
    score itself would exceed the target because of a tie, the returned
    threshold is the next representable float above the maximum, yielding zero
    validation alerts rather than silently violating the requested FPR.
    """
    s = np.asarray(legit_validation_scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no validation scores supplied")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("target_fpr must be strictly between 0 and 1")

    values, counts = np.unique(s, return_counts=True)
    flagged = np.cumsum(counts[::-1])[::-1]
    valid = flagged / float(s.size) <= target_fpr
    if np.any(valid):
        return float(values[np.flatnonzero(valid)[0]])
    return float(np.nextafter(values[-1], np.inf))


def rate_at_or_above(scores: Sequence[float], threshold: float) -> float:
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s >= threshold))


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> Optional[float]:
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("prevalence must be in [0, 1]")
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
    validation_fpr = rate_at_or_above(legit_validation, threshold)
    realised_fpr = rate_at_or_above(legit_test, threshold)
    recall = rate_at_or_above(fraud_test, threshold)
    precision = precision_at_prevalence(recall, realised_fpr, production_prevalence)
    return {
        "threshold": round(float(threshold), 8),
        "threshold_source": "legitimate validation split (disjoint from test)",
        "target_fpr": target_fpr,
        "realised_validation_fpr": round(float(validation_fpr), 6),
        "realised_test_fpr": None if not np.isfinite(realised_fpr) else round(float(realised_fpr), 6),
        "calibration_gap_pct_points": (
            None
            if not np.isfinite(realised_fpr)
            else round(100.0 * (float(realised_fpr) - validation_fpr), 4)
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
