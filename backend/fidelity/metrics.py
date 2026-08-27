"""
Fidelity and transfer diagnostics.

Five measures, reported together and always including the ones that fail:

* `c2st_auc`                 -- can a classifier tell real from synthetic?
                                0.50 means indistinguishable; 1.00 means
                                trivially separable. This is the strictest
                                joint-distribution test and the one that
                                parametric/rule generators fail.
* `mean_jsd` / `mean_tvd`    -- marginal agreement. Easy to pass; a generator
                                that passes ONLY these is matching marginals.
* `correlation_frobenius`    -- rank-dependence agreement, i.e. the joint
                                structure the marginals cannot see.
* `tstr_ratio`               -- Train on Synthetic, Test on Real. The only
                                measure that speaks directly to usefulness.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evidence.calibration import pin_threshold_at_fpr, recall_at_threshold

from .features import ALL_COLS, CATEGORICAL_COLS, NUMERIC_COLS


def encode_frames(frames: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """One-hot encode a set of frames against a SHARED column space.

    Encoding frames separately is a classic source of silent bugs: a category
    absent from one frame produces a missing column and a shape mismatch.
    """
    sizes = [len(f) for f in frames]
    combined = pd.concat([f[ALL_COLS] for f in frames], ignore_index=True)
    encoded = pd.get_dummies(combined, columns=CATEGORICAL_COLS, dtype=float)
    out: list[pd.DataFrame] = []
    start = 0
    for size in sizes:
        out.append(encoded.iloc[start:start + size].reset_index(drop=True))
        start += size
    return out


def c2st_auc(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    seed: int = 42,
    n_splits: int = 3,
) -> dict:
    """Classifier Two-Sample Test. Cross-validated ROC-AUC of real vs synthetic."""
    x_real, x_synth = encode_frames([real, synth])
    X = pd.concat([x_real, x_synth], ignore_index=True).to_numpy(dtype=float)
    y = np.concatenate([np.zeros(len(x_real)), np.ones(len(x_synth))])

    n_splits = max(2, min(n_splits, int(min(np.bincount(y.astype(int))))))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in folds.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], proba)))

    mean_auc = float(np.mean(aucs))
    return {
        "c2st_auc": round(mean_auc, 4),
        "per_fold": [round(a, 4) for a in aucs],
        "target": 0.5,
        "passes": bool(mean_auc <= 0.65),
        "interpretation": "0.50 = indistinguishable from real; 1.00 = trivially separable",
    }


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def marginal_divergences(real: pd.DataFrame, synth: pd.DataFrame, bins: int = 20) -> dict:
    """Per-column JSD (all columns) and TVD (categorical columns)."""
    per_column: dict[str, dict] = {}

    for col in NUMERIC_COLS:
        lo = float(min(real[col].min(), synth[col].min()))
        hi = float(max(real[col].max(), synth[col].max()))
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(real[col].to_numpy(dtype=float), bins=edges)
        q, _ = np.histogram(synth[col].to_numpy(dtype=float), bins=edges)
        per_column[col] = {"type": "numeric", "jsd": round(_jsd(p, q), 6)}

    for col in CATEGORICAL_COLS:
        categories = sorted(set(real[col].astype(str)) | set(synth[col].astype(str)))
        p = np.array([float((real[col].astype(str) == c).sum()) for c in categories])
        q = np.array([float((synth[col].astype(str) == c).sum()) for c in categories])
        p_norm = p / max(p.sum(), 1e-12)
        q_norm = q / max(q.sum(), 1e-12)
        per_column[col] = {
            "type": "categorical",
            "jsd": round(_jsd(p, q), 6),
            "tvd": round(float(0.5 * np.abs(p_norm - q_norm).sum()), 6),
        }

    jsds = [v["jsd"] for v in per_column.values()]
    tvds = [v["tvd"] for v in per_column.values() if "tvd" in v]
    return {
        "mean_jsd": round(float(np.mean(jsds)), 6),
        "mean_tvd": round(float(np.mean(tvds)), 6) if tvds else None,
        "per_column": per_column,
    }


def _ordinal_frame(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Ordinal-encode categoricals using a FIXED category order so that two
    frames map onto the same numeric space."""
    out = frame[NUMERIC_COLS].astype(float).copy()
    for col, cats in categories.items():
        lookup = {c: i for i, c in enumerate(cats)}
        out[col] = frame[col].astype(str).map(lookup).fillna(-1).astype(float)
    return out


def correlation_frobenius_diff(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Frobenius norm of the difference between Spearman correlation matrices.

    Spearman rather than Pearson because we care about rank dependence, which
    is exactly what a copula models and what independent sampling erases.
    """
    categories = {
        col: sorted(set(real[col].astype(str)) | set(synth[col].astype(str)))
        for col in CATEGORICAL_COLS
    }
    r = _ordinal_frame(real, categories).corr(method="spearman").to_numpy(dtype=float)
    s = _ordinal_frame(synth, categories).corr(method="spearman").to_numpy(dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    s = np.nan_to_num(s, nan=0.0)
    diff = float(np.linalg.norm(r - s, ord="fro"))
    return {
        "correlation_frobenius_diff": round(diff, 4),
        "target": 0.0,
        "matrix_dim": int(r.shape[0]),
        "interpretation": "0.0 = identical rank-dependence structure",
    }


def fit_detector(frame: pd.DataFrame, labels: Sequence[int], seed: int = 42) -> tuple:
    """Fit a fraud detector on a framed dataset. Returns (model, columns)."""
    encoded = pd.get_dummies(frame[ALL_COLS], columns=CATEGORICAL_COLS, dtype=float)
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(encoded.to_numpy(dtype=float), np.asarray(labels, dtype=int))
    return clf, list(encoded.columns)


def score_detector(model, columns: list[str], frame: pd.DataFrame) -> np.ndarray:
    encoded = pd.get_dummies(frame[ALL_COLS], columns=CATEGORICAL_COLS, dtype=float)
    encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return model.predict_proba(encoded.to_numpy(dtype=float))[:, 1]


def tstr_report(
    real_train: tuple[pd.DataFrame, Sequence[int]],
    synth_train: tuple[pd.DataFrame, Sequence[int]],
    calibration_legit: pd.DataFrame,
    real_test: tuple[pd.DataFrame, Sequence[int]],
    target_fpr: float = 0.01,
    seed: int = 42,
) -> dict:
    """Train on Synthetic, Test on Real -- against a Train-on-Real ceiling.

    The ratio is the headline: 1.00 means synthetic training data is as useful
    as real training data; 0.50 means it buys you half the detector.

    Thresholds are pinned on `calibration_legit`, which is disjoint from
    `real_test`, so the reported recall is an out-of-sample measurement.
    """
    test_frame, test_labels = real_test
    test_labels = np.asarray(test_labels, dtype=int)

    results: dict[str, dict] = {}
    for name, (frame, labels) in (("train_on_real", real_train), ("train_on_synthetic", synth_train)):
        model, columns = fit_detector(frame, labels, seed=seed)
        calib_scores = score_detector(model, columns, calibration_legit)
        tau = pin_threshold_at_fpr(calib_scores, target_fpr)
        test_scores = score_detector(model, columns, test_frame)
        fraud_scores = test_scores[test_labels == 1]
        legit_scores = test_scores[test_labels == 0]
        results[name] = {
            "threshold": round(float(tau), 6),
            "recall": round(recall_at_threshold(fraud_scores, tau), 4),
            "fpr": round(float(np.mean(legit_scores >= tau)), 4),
            "roc_auc": round(float(roc_auc_score(test_labels, test_scores)), 4),
            "pr_auc": round(float(average_precision_score(test_labels, test_scores)), 4),
        }

    ceiling = results["train_on_real"]["recall"]
    achieved = results["train_on_synthetic"]["recall"]
    ratio = float(achieved / ceiling) if ceiling and ceiling > 0 else float("nan")
    return {
        "target_fpr": target_fpr,
        "tstr_ratio": round(ratio, 4),
        "detail": results,
        "interpretation": "ratio 1.0 = synthetic data as useful as real data for training",
    }


def fidelity_report(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 42) -> dict:
    """All distribution-level measures for one generator, pass or fail."""
    divergences = marginal_divergences(real, synth)
    c2st = c2st_auc(real, synth, seed=seed)
    corr = correlation_frobenius_diff(real, synth)
    return {
        "n_real": int(len(real)),
        "n_synthetic": int(len(synth)),
        "c2st": c2st,
        "marginals": {"mean_jsd": divergences["mean_jsd"], "mean_tvd": divergences["mean_tvd"]},
        "joint": corr,
        "per_column": divergences["per_column"],
    }


__all__ = [
    "c2st_auc",
    "correlation_frobenius_diff",
    "encode_frames",
    "fidelity_report",
    "fit_detector",
    "marginal_divergences",
    "score_detector",
    "tstr_report",
]
