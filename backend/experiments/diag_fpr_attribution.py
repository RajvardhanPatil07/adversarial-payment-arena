"""Which LADDER BRANCH produces each false positive on the held-out split?

By this point the velocity layer has been cleared: the eval split's legitimate
score distribution is if anything COLDER than the calibration split's, so a tau
pinned at calib q99 cannot by itself yield 4x the target FPR. Something else in
the ladder is firing. This attributes every non-APPROVE on a legitimate row to
the exact branch that produced it, which is the only way to stop guessing.

Run: python3 experiments/diag_fpr_attribution.py
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense.decision import APPROVE, DecisionEngine  # noqa: E402
from defense.novelty import DEFAULT_MODEL_PATH as IP  # noqa: E402
from defense.realtime import DEFAULT_MODEL_PATH as XP  # noqa: E402


def main() -> None:
    calib = pickle.load(open("/tmp/split_calib.pkl", "rb"))
    ev = pickle.load(open("/tmp/split_eval.pkl", "rb"))

    eng = DecisionEngine(environment=calib["env"])
    eng.scorer.load(XP)
    eng.novelty.load(IP)
    info = eng.calibrate(calib["rows"], target_fpr=0.01)
    print("calibration:", {k: info[k] for k in (
        "achieved_validation_fpr", "achieved_validation_recall",
        "stepup_threshold", "decline_threshold", "manual_threshold",
        "ring_risk_threshold", "novelty_alone_alerts")})

    ee = DecisionEngine(environment=ev["env"])
    ee.scorer.load(XP)
    ee.novelty.load(IP)
    for a in ("stepup_threshold", "decline_threshold", "manual_threshold",
              "ring_risk_threshold", "novelty_alone_alerts"):
        setattr(ee, a, getattr(eng, a))

    fp_reasons: Counter = Counter()
    tp_reasons: Counter = Counter()
    fp = tn = tp = fn = 0
    anomaly_on_legit = 0
    for r in sorted(ev["rows"], key=lambda x: x["payload"]["timestamp"]):
        out = ee.decide(r["payload"])
        alert = out["decision"] != APPROVE
        key = out["reasons"][0].split(":")[0] if out["reasons"] else "none"
        if r["label"] == 0:
            anomaly_on_legit += int(out["scores"]["is_anomaly"])
            if alert:
                fp += 1
                fp_reasons[key] += 1
            else:
                tn += 1
        else:
            if alert:
                tp += 1
                tp_reasons[key] += 1
            else:
                fn += 1

    print("\nHELD-OUT: TPR=%.2f%%  FPR=%.2f%%  (fp=%d tn=%d tp=%d fn=%d)"
          % (100 * tp / max(tp + fn, 1), 100 * fp / max(fp + tn, 1), fp, tn, tp, fn))
    print("  is_anomaly rate on legitimate rows: %.2f%%"
          % (100 * anomaly_on_legit / max(fp + tn, 1)))
    print("\n  FALSE POSITIVES by branch:")
    for k, v in fp_reasons.most_common():
        print("    %-34s %4d  (%.2f%% of legit)" % (k, v, 100 * v / max(fp + tn, 1)))
    print("\n  TRUE POSITIVES by branch:")
    for k, v in tp_reasons.most_common():
        print("    %-34s %4d" % (k, v))


if __name__ == "__main__":
    main()
