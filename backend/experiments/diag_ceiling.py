"""What TPR is achievable on the eval split's own features, per family?

Purpose: separate two very different diagnoses. If the XGBoost score on the eval
split's stored features reaches ~0.93 recall at a 1% false-positive budget, then
the decision ladder is throwing away recall the model already has. If it does
not, the eval split is simply harder than the in-sample number suggested and the
ladder is behaving correctly.

Run: python3 experiments/diag_ceiling.py
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense.realtime import DEFAULT_MODEL_PATH as XP  # noqa: E402
from defense.realtime import VelocityScorer  # noqa: E402


def main() -> None:
    ev = pickle.load(open("/tmp/split_eval.pkl", "rb"))
    sc = VelocityScorer()
    sc.load(XP)

    v = np.array([sc.score_from_features(r["features"]) for r in ev["rows"]])
    y = np.array([r["label"] for r in ev["rows"]])

    from sklearn.metrics import average_precision_score, roc_auc_score

    print("XGB on eval STORED features: ROC-AUC=%.4f  PR-AUC=%.4f"
          % (roc_auc_score(y, v), average_precision_score(y, v)))

    legit = v[y == 0]
    for tf in (0.005, 0.01, 0.02, 0.05):
        tau = float(np.quantile(legit, 1 - tf))
        pred = v > tau
        tp = int((pred & (y == 1)).sum())
        fn = int((~pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        tn = int((~pred & (y == 0)).sum())
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        print("  tau@%.3f -> %.5f  TPR=%6.2f%%  FPR=%5.2f%%  prec=%.3f  F1=%.3f"
              % (tf, tau, 100 * rec, 100 * fpr, prec,
                 2 * prec * rec / max(prec + rec, 1e-9)))

    tau = float(np.quantile(legit, 0.99))
    hit: Counter = Counter()
    tot: Counter = Counter()
    for r, s in zip(ev["rows"], v):
        if r["label"] == 1:
            tot[r["attack_id"]] += 1
            hit[r["attack_id"]] += int(s > tau)
    print("\nper-family recall @1%% FPR (tau=%.5f):" % tau)
    for k in sorted(tot, key=lambda x: hit[x] / tot[x]):
        print("  %-42s %3d/%3d = %6.2f%%" % (k, hit[k], tot[k], 100 * hit[k] / tot[k]))


if __name__ == "__main__":
    main()
