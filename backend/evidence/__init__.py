"""
Evidence subsystem: calibration, economics, and artifact provenance.

Three rules enforced by the code in this package:

1. Operating thresholds are pinned on a validation split that is disjoint from
   the split used to report performance (`calibration`).
2. Detection metrics are reported at production base rates, not only at
   laboratory prevalence (`calibration.prevalence_sweep`).
3. The cost of false positives on legitimate customers is priced in INR and
   included in every business-impact figure (`economics`).
"""

from .artifacts import ClaimLedger, read_artifact, write_artifact  # noqa: F401
from .calibration import (  # noqa: F401
    bootstrap_ci,
    calibrate,
    chronological_split,
    pin_threshold_at_fpr,
    precision_at_prevalence,
    prevalence_sweep,
)
from .economics import CostModel, evaluate_operating_point  # noqa: F401

__all__ = [
    "ClaimLedger",
    "CostModel",
    "bootstrap_ci",
    "calibrate",
    "chronological_split",
    "evaluate_operating_point",
    "pin_threshold_at_fpr",
    "precision_at_prevalence",
    "prevalence_sweep",
    "read_artifact",
    "write_artifact",
]
