"""Pure-numpy classifier primitives.

Why this file exists: every fidelity, transfer and privacy measurement in this
repository needs (a) a reasonably strong tabular binary classifier and (b) an
AUC. Depending on scikit-learn makes the evidence layer unreproducible on a
machine that cannot install it. Everything here is numpy-only, so
`make reproduce` works with numpy + pandas alone.

This is deliberately a CART/bagging implementation, not a tuned booster. The
experiments in this repository compare *data* (real vs synthetic) while holding
the classifier constant, so classifier strength is a constant, not a variable.
A stronger model would move all arms together.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Rank-based ROC AUC with correct mid-rank handling of ties."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.size != s.size:
        raise ValueError("y_true and score must have the same length")
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    sum_pos = float(ranks[pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _gini(pos_weight: float, total_weight: float) -> float:
    if total_weight <= 0:
        return 0.0
    p = pos_weight / total_weight
    return 1.0 - (p * p + (1.0 - p) * (1.0 - p))


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self) -> None:
        self.feature: int = -1
        self.threshold: float = 0.0
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.value: float = 0.0


class DecisionTreeBinary:
    """CART for binary classification with sample weights and Gini impurity."""

    def __init__(
        self,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        max_features: Optional[int] = None,
        n_bins: int = 24,
        seed: int = 0,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.n_bins = int(n_bins)
        self.rng = np.random.default_rng(seed)
        self.root: Optional[_Node] = None
        self.n_features_: int = 0
        self.importances_: Optional[np.ndarray] = None

    # -- fitting -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float).ravel()
        if w is None:
            w = np.ones(y.size, dtype=float)
        w = np.asarray(w, dtype=float).ravel()
        self.n_features_ = X.shape[1]
        self.importances_ = np.zeros(self.n_features_, dtype=float)
        self.root = self._build(X, y, w, depth=0)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ = self.importances_ / total
        return self

    def _feature_subset(self) -> np.ndarray:
        k = self.max_features or self.n_features_
        k = max(1, min(int(k), self.n_features_))
        if k == self.n_features_:
            return np.arange(self.n_features_)
        return self.rng.choice(self.n_features_, size=k, replace=False)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> Optional[Tuple[float, int, float]]:
        W = float(w.sum())
        P = float((w * y).sum())
        if W <= 0:
            return None
        parent = _gini(P, W)
        best: Optional[Tuple[float, int, float]] = None
        n = y.size
        for f in self._feature_subset():
            col = X[:, f]
            uniq = np.unique(col)
            if uniq.size < 2:
                continue
            if uniq.size > self.n_bins:
                qs = np.linspace(0.0, 1.0, self.n_bins + 2)[1:-1]
                thresholds = np.unique(np.quantile(col, qs))
            else:
                thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            if thresholds.size == 0:
                continue
            order = np.argsort(col, kind="mergesort")
            cs = col[order]
            cw = np.cumsum(w[order])
            cp = np.cumsum(w[order] * y[order])
            cut = np.searchsorted(cs, thresholds, side="right")
            for k in range(thresholds.size):
                i = int(cut[k])
                if i < self.min_samples_leaf or (n - i) < self.min_samples_leaf:
                    continue
                wl = float(cw[i - 1])
                pl = float(cp[i - 1])
                wr = W - wl
                pr = P - pl
                if wl <= 0 or wr <= 0:
                    continue
                gain = parent - (wl / W) * _gini(pl, wl) - (wr / W) * _gini(pr, wr)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, int(f), float(thresholds[k]))
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> _Node:
        node = _Node()
        W = float(w.sum())
        node.value = float((w * y).sum() / W) if W > 0 else 0.0
        if (
            depth >= self.max_depth
            or y.size < 2 * self.min_samples_leaf
            or np.all(y == y[0])
        ):
            return node
        split = self._best_split(X, y, w)
        if split is None:
            return node
        gain, feature, threshold = split
        mask = X[:, feature] <= threshold
        if mask.all() or (~mask).all():
            return node
        assert self.importances_ is not None
        self.importances_[feature] += gain * W
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y[mask], w[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], w[~mask], depth + 1)
        return node

    # -- prediction --------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            node = self.root
            while node is not None and node.feature >= 0:
                node = node.left if X[i, node.feature] <= node.threshold else node.right
            out[i] = node.value if node is not None else 0.0
        return out


class RandomForestBinary:
    """Bagged CART ensemble. Balanced class weights by default."""

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        max_features: str | int | None = "sqrt",
        bootstrap: bool = True,
        class_weight: Optional[str] = "balanced",
        n_bins: int = 24,
        seed: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.class_weight = class_weight
        self.n_bins = int(n_bins)
        self.seed = int(seed)
        self.trees: List[DecisionTreeBinary] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def _resolve_max_features(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if isinstance(mf, int):
            return max(1, min(mf, d))
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)) or 1)
        return d

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        n, d = X.shape
        mf = self._resolve_max_features(d)
        base_w = np.ones(n, dtype=float)
        if self.class_weight == "balanced":
            n_pos = max(1, int((y == 1).sum()))
            n_neg = max(1, int((y == 0).sum()))
            base_w = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
        rng = np.random.default_rng(self.seed)
        self.trees = []
        importances = np.zeros(d, dtype=float)
        for t in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                # keep both classes present in every bag
                if len(np.unique(y[idx])) < 2:
                    idx = np.concatenate(
                        [idx[:-2], rng.choice(np.flatnonzero(y == 1), 1), rng.choice(np.flatnonzero(y == 0), 1)]
                    )
            else:
                idx = np.arange(n)
            tree = DecisionTreeBinary(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                n_bins=self.n_bins,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[idx], y[idx], base_w[idx])
            self.trees.append(tree)
            if tree.importances_ is not None:
                importances += tree.importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("model is not fitted")
        acc = np.zeros(np.asarray(X).shape[0], dtype=float)
        for tree in self.trees:
            acc += tree.predict_proba(X)
        return acc / len(self.trees)


def stratified_folds(y: Sequence[int], k: int = 3, seed: int = 0) -> List[np.ndarray]:
    """Return k index arrays, each an approximately class-balanced test fold."""
    y = np.asarray(y).astype(int).ravel()
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for pos, i in enumerate(idx):
            folds[pos % k].append(int(i))
    return [np.array(sorted(f), dtype=int) for f in folds]


def oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 120,
    max_depth: int = 12,
    k: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predicted probabilities plus summed feature importances."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int).ravel()
    scores = np.zeros(y.size, dtype=float)
    importances = np.zeros(X.shape[1], dtype=float)
    for f, test_idx in enumerate(stratified_folds(y, k=k, seed=seed)):
        train_idx = np.setdiff1d(np.arange(y.size), test_idx)
        if len(np.unique(y[train_idx])) < 2 or test_idx.size == 0:
            continue
        model = RandomForestBinary(
            n_estimators=n_estimators, max_depth=max_depth, seed=seed * 100 + f
        ).fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])
        if model.feature_importances_ is not None:
            importances += model.feature_importances_
    total = importances.sum()
    return scores, (importances / total if total > 0 else importances)
