"""Are the sequence-level features actually FIRING, or are they dead columns?

The wave-3 features (iat_regularity, amount_escalation, amount_band_tightness,
low_value_probe_ratio, dev_distinct_custs_1h) all need several prior events for
the same customer or device inside the lookback window. If the corpus spreads
N transactions over 1000 customers across a wide time span, almost every row has
no usable history, the feature reads its zero-default, and the column carries no
information at inference even though it looked important during training.

A feature that is non-zero on 2% of rows can still rank high in gain-based
importance (it splits a small pure subset perfectly) while contributing nothing
to out-of-sample recall. This script measures the activity rate directly.

Run: python3 experiments/diag_feature_activity.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense.realtime import FEATURE_NAMES  # noqa: E402


def report(rows: list[dict], tag: str) -> None:
    y = np.array([r["label"] for r in rows])
    print("\n=== %s (n=%d, positives=%d) ===" % (tag, len(rows), int(y.sum())))
    print("%-26s %8s %9s %9s" % ("feature", "act%", "mean|L", "mean|A"))
    for name in FEATURE_NAMES:
        col = np.array([float(r["features"].get(name, 0.0)) for r in rows])
        act = float((np.abs(col) > 1e-9).mean())
        ml = col[y == 0].mean() if (y == 0).any() else 0.0
        ma = col[y == 1].mean() if (y == 1).any() else 0.0
        flag = "  <-- DEAD" if act < 0.05 else ""
        print("%-26s %7.2f%% %9.4f %9.4f%s" % (name, 100 * act, ml, ma, flag))


def main() -> None:
    for tag, path in (("EVAL split (seed 777)", "/tmp/split_eval.pkl"),
                      ("CALIB split (seed 321)", "/tmp/split_calib.pkl")):
        report(pickle.load(open(path, "rb"))["rows"], tag)


if __name__ == "__main__":
    main()
