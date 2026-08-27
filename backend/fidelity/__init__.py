"""
Fidelity subsystem.

The competitive thesis of this repository lives here: red-teaming a payment
fraud detector only *helps* the production model if the synthetic attacks are
faithful to the JOINT distribution of real fraud, not merely to its marginals.

* `features`  -- one common feature frame for every data source
* `copula`    -- a Gaussian-copula synthesizer (joint-aware) plus an
                 independent-marginal synthesizer reproducing the standard
                 rule/template approach, used as the control arm
* `metrics`   -- C2ST, TSTR, Jensen-Shannon, total-variation and correlation
                 Frobenius diagnostics
"""

from .features import ALL_COLS, CATEGORICAL_COLS, NUMERIC_COLS  # noqa: F401

__all__ = ["ALL_COLS", "CATEGORICAL_COLS", "NUMERIC_COLS"]
