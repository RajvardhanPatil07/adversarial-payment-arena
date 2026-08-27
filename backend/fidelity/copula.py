"""
Synthesizers used as the independent variable of the transfer experiment.

Two synthesizers, one interface:

`IndependentMarginalSynthesizer` (the CONTROL arm)
    Samples every column independently from its own empirical marginal. This
    is a faithful reproduction of the rule/template/parametric approach used
    by the published red-teaming systems in this space: marginals come out
    close to real, the joint structure is destroyed.

`GaussianCopulaSynthesizer` (the TREATMENT arm)
    Maps each column to a latent Gaussian via its empirical CDF, learns the
    latent correlation matrix, samples from the correlated Gaussian, then maps
    back through the inverse CDFs. Marginals are preserved *and* the rank
    dependence structure is preserved.

Only numpy / scipy / pandas are required -- no deep generative dependency,
which keeps the reproduction command runnable on a laptop in seconds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import ALL_COLS, CATEGORICAL_COLS, NUMERIC_COLS

_EPS = 1e-6


def _clip_unit(u: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float), _EPS, 1.0 - _EPS)


class _NumericMarginal:
    """Empirical marginal with an inverse-CDF sampler."""

    def __init__(self, values: Sequence[float]) -> None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.zeros(1, dtype=float)
        self.sorted = np.sort(arr)

    def to_uniform(self, values: Sequence[float]) -> np.ndarray:
        ranks = np.searchsorted(self.sorted, np.asarray(values, dtype=float), side="right")
        return _clip_unit(ranks / (self.sorted.size + 1.0))

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        u = _clip_unit(u)
        idx = np.clip((u * self.sorted.size).astype(int), 0, self.sorted.size - 1)
        return self.sorted[idx]


class _CategoricalMarginal:
    """Empirical categorical marginal mapped onto [0, 1) intervals.

    Discrete columns are handled by *uniform jitter within the category's
    probability interval*, the standard trick for copulas over mixed data. It
    keeps the latent Gaussian continuous so the correlation estimate stays
    meaningful.
    """

    def __init__(self, values: Sequence[str]) -> None:
        series = pd.Series(list(values), dtype=object).astype(str)
        counts = series.value_counts(normalize=True).sort_index()
        if counts.empty:
            counts = pd.Series({"UNKNOWN": 1.0})
        self.categories: list[str] = [str(c) for c in counts.index]
        self.probs = counts.to_numpy(dtype=float)
        self.edges = np.cumsum(self.probs)
        self.edges[-1] = 1.0
        self.index = {c: i for i, c in enumerate(self.categories)}

    def _interval(self, j: int) -> tuple[float, float]:
        lo = float(self.edges[j - 1]) if j > 0 else 0.0
        hi = float(self.edges[j])
        if hi <= lo:
            hi = lo + _EPS
        return lo, hi

    def to_uniform(self, values: Sequence[str], rng: np.random.Generator) -> np.ndarray:
        out = np.empty(len(values), dtype=float)
        for i, raw in enumerate(values):
            j = self.index.get(str(raw), 0)
            lo, hi = self._interval(j)
            out[i] = rng.uniform(lo, hi)
        return _clip_unit(out)

    def from_uniform(self, u: np.ndarray) -> list[str]:
        u = _clip_unit(u)
        idx = np.searchsorted(self.edges, u, side="left")
        idx = np.clip(idx, 0, len(self.categories) - 1)
        return [self.categories[int(i)] for i in idx]


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone and renormalise to a
    correlation matrix. Empirical latent correlations are occasionally
    indefinite; sampling would fail without this."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-8, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


class _BaseSynthesizer:
    name = "base"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.numeric: dict[str, _NumericMarginal] = {}
        self.categorical: dict[str, _CategoricalMarginal] = {}
        self.n_train = 0
        self._fitted = False

    def _fit_marginals(self, frame: pd.DataFrame) -> None:
        for col in NUMERIC_COLS:
            self.numeric[col] = _NumericMarginal(frame[col].to_numpy(dtype=float))
        for col in CATEGORICAL_COLS:
            self.categorical[col] = _CategoricalMarginal(frame[col].tolist())
        self.n_train = int(len(frame))

    def fit(self, frame: pd.DataFrame) -> "_BaseSynthesizer":
        raise NotImplementedError

    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}.fit() must be called before sample()")


class IndependentMarginalSynthesizer(_BaseSynthesizer):
    """CONTROL arm: correct marginals, independent columns."""

    name = "independent_marginal"

    def fit(self, frame: pd.DataFrame) -> "IndependentMarginalSynthesizer":
        self._fit_marginals(frame[ALL_COLS])
        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        self._require_fit()
        data: dict[str, object] = {}
        for col in NUMERIC_COLS:
            u = self.rng.uniform(size=n)
            data[col] = self.numeric[col].from_uniform(u)
        for col in CATEGORICAL_COLS:
            u = self.rng.uniform(size=n)
            data[col] = self.categorical[col].from_uniform(u)
        return pd.DataFrame(data, columns=ALL_COLS)


class GaussianCopulaSynthesizer(_BaseSynthesizer):
    """TREATMENT arm: correct marginals AND learned rank dependence."""

    name = "gaussian_copula"

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.correlation: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "GaussianCopulaSynthesizer":
        frame = frame[ALL_COLS]
        self._fit_marginals(frame)

        latent = np.empty((len(frame), len(ALL_COLS)), dtype=float)
        for j, col in enumerate(ALL_COLS):
            if col in NUMERIC_COLS:
                u = self.numeric[col].to_uniform(frame[col].to_numpy(dtype=float))
            else:
                u = self.categorical[col].to_uniform(frame[col].tolist(), self.rng)
            latent[:, j] = norm.ppf(u)

        # Constant columns give zero variance -> guard the correlation estimate.
        stds = latent.std(axis=0)
        safe = stds > 1e-9
        corr = np.eye(len(ALL_COLS))
        if safe.sum() >= 2:
            sub = np.corrcoef(latent[:, safe], rowvar=False)
            sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
            idx = np.where(safe)[0]
            for a, ia in enumerate(idx):
                for b, ib in enumerate(idx):
                    corr[ia, ib] = sub[a, b]
        self.correlation = _nearest_psd(corr)
        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        self._require_fit()
        assert self.correlation is not None
        mean = np.zeros(len(ALL_COLS))
        latent = self.rng.multivariate_normal(mean, self.correlation, size=n, method="cholesky")
        u = norm.cdf(latent)

        data: dict[str, object] = {}
        for j, col in enumerate(ALL_COLS):
            if col in NUMERIC_COLS:
                data[col] = self.numeric[col].from_uniform(u[:, j])
            else:
                data[col] = self.categorical[col].from_uniform(u[:, j])
        return pd.DataFrame(data, columns=ALL_COLS)


SYNTHESIZERS = {
    IndependentMarginalSynthesizer.name: IndependentMarginalSynthesizer,
    GaussianCopulaSynthesizer.name: GaussianCopulaSynthesizer,
}


__all__ = [
    "GaussianCopulaSynthesizer",
    "IndependentMarginalSynthesizer",
    "SYNTHESIZERS",
]
