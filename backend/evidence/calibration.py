"""
Leakage-free threshold calibration and prevalence-adjusted reporting.

Why this module exists
----------------------
A detector quoted as "94% recall at 0.8% false-positive rate" is close to
meaningless unless two things are stated:

1. WHERE the operating threshold came from. If the threshold that produces
   0.8% FPR was chosen on the same rows used to report 0.8% FPR, the number is
   an in-sample fit, not a measurement.
2. WHAT base rate the precision assumes. Laboratory corpora run fraud
   prevalence in the double digits; production authorisation traffic runs it
   near 1% or below. Precision does not survive that change, and the collapse
   is a property of the base rate, not of the model.

Everything here is deliberately simple arithmetic so a judge can audit it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SplitSizes:
    train: int
    validation: int
    test: int


@dataclass
class CalibrationResult:
    """One audited operating point."""

    threshold: float
    target_fpr: float
    validation_fpr: float
    validation_rows: int
    test_fpr: float
    test_rows: int
    threshold_source: str = "validation split (disjoint from test)"

    def to_dict(self) -> dict:
        return asdict(self)


def chronological_split(
    items: Sequence,
    validation_frac: float = 0.25,
    test_frac: float = 0.25,
) -> tuple[list, list, list]:
    """Split an already time-ordered sequence into train / validation / test.

    Temporal rather than random: fraud data is non-stationary, and a random
    split lets tomorrow's traffic calibrate yesterday's threshold.
    """
    if not 0.0 < validation_frac + test_frac < 1.0:
        raise ValueError("validation_frac + test_frac must be in (0, 1)")
    n = len(items)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * validation_frac))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"split too small for n={n}")
    return list(items[:n_train]), list(items[n_train:n_train + n_val]), list(items[n_train + n_val:])


def pin_threshold_at_fpr(legit_scores: Sequence[float], target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on `legit_scores` does not
    exceed `target_fpr`. Scores are 'higher is more suspicious'.
    """
    scores = np.sort(np.asarray(legit_scores, dtype=float))
    if scores.size == 0:
        raise ValueError("cannot calibrate on an empty legitimate sample")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    # Quantile at (1 - target_fpr) is the threshold that flags exactly the top
    # target_fpr proportion of legitimate traffic.
    return float(np.quantile(scores, 1.0 - target_fpr, method="higher"))


def fpr_at_threshold(legit_scores: Sequence[float], threshold: float) -> float:
    scores = np.asarray(legit_scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores >= threshold))


def recall_at_threshold(fraud_scores: Sequence[float], threshold: float) -> float:
    scores = np.asarray(fraud_scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores >= threshold))


def calibrate(
    validation_legit_scores: Sequence[float],
    test_legit_scores: Sequence[float],
    target_fpr: float = 0.01,
) -> CalibrationResult:
    """Pin the threshold on validation, then MEASURE it on test.

    The gap between `validation_fpr` and `test_fpr` is the honest estimate of
    how well the operating point generalises. Reporting only the first number
    is the error this function exists to prevent.
    """
    tau = pin_threshold_at_fpr(validation_legit_scores, target_fpr)
    return CalibrationResult(
        threshold=tau,
        target_fpr=target_fpr,
        validation_fpr=fpr_at_threshold(validation_legit_scores, tau),
        validation_rows=len(validation_legit_scores),
        test_fpr=fpr_at_threshold(test_legit_scores, tau),
        test_rows=len(test_legit_scores),
    )


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Bayes-adjusted precision for an operating point at a given base rate.

        precision = p*R / (p*R + (1-p)*FPR)

    `recall` and `fpr` are measured quantities; `prevalence` is a deployment
    assumption. Keeping them separate is the whole point: the model does not
    change when the base rate does, but the precision a fraud analyst
    experiences changes enormously.
    """
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("prevalence must be in [0, 1]")
    tp = prevalence * recall
    fp = (1.0 - prevalence) * fpr
    if tp + fp <= 0.0:
        return 0.0
    return float(tp / (tp + fp))


def alerts_per_million(recall: float, fpr: float, prevalence: float) -> dict:
    """Operational load at a base rate: what a review queue actually receives
    per million authorisations."""
    n = 1_000_000
    frauds = n * prevalence
    legits = n - frauds
    true_alerts = frauds * recall
    false_alerts = legits * fpr
    return {
        "prevalence": prevalence,
        "true_alerts": round(true_alerts, 1),
        "false_alerts": round(false_alerts, 1),
        "total_alerts": round(true_alerts + false_alerts, 1),
        "precision": round(precision_at_prevalence(recall, fpr, prevalence), 6),
        "missed_frauds": round(frauds * (1.0 - recall), 1),
    }


def prevalence_sweep(
    recall: float,
    fpr: float,
    prevalences: Sequence[float] = (0.5, 0.1, 0.05, 0.013, 0.005, 0.001),
) -> list[dict]:
    """The same detector reported across plausible base rates.

    Publishing this table is the difference between a laboratory claim and a
    deployable claim.
    """
    return [alerts_per_million(recall, fpr, float(p)) for p in prevalences]


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Nonparametric bootstrap CI for the mean of `values`.

    Used for seed-level means so that every headline number ships with an
    interval rather than a single point.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if arr.size == 1:
        v = float(arr[0])
        return {"mean": v, "lo": v, "hi": v, "n": 1}
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, alpha / 2.0)),
        "hi": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": int(arr.size),
        "method": f"nonparametric bootstrap, {n_resamples} resamples",
    }


__all__ = [
    "CalibrationResult",
    "SplitSizes",
    "alerts_per_million",
    "bootstrap_ci",
    "calibrate",
    "chronological_split",
    "fpr_at_threshold",
    "pin_threshold_at_fpr",
    "precision_at_prevalence",
    "prevalence_sweep",
    "recall_at_threshold",
]
