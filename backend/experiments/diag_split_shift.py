"""Does the LEGITIMATE score distribution move between seeds?

Calibration pins tau at the 99th percentile of legitimate scores on split A and
the resulting FPR on split B is 4x the target. Two candidate explanations, and
they demand opposite fixes:

  (a) tau is fine, but split B's legitimate traffic genuinely scores higher --
      a distribution shift across seeds. Fix: calibrate on more/pooled seeds so
      tau reflects the population rather than one draw.

  (b) tau is measured on the corpus's STORED features while `decide()` recomputes
      them live, and the two disagree. Fix: make the two paths identical.

This prints the legitimate-score quantiles of both splits under BOTH feature
paths, which separates (a) from (b) directly.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense.decision import DecisionEngine  # noqa: E402
from defense.novelty import DEFAULT_MODEL_PATH as IP  # noqa: E402
from defense.realtime import DEFAULT_MODEL_PATH as XP  # noqa: E402

QS = [0.5, 0.9, 0.99, 0.995, 0.999]


def stored_scores(corpus, sc) -> np.ndarray:
    return np.array([
        sc.score_from_features(r["features"])
        for r in corpus["rows"] if r["label"] == 0
    ])


def live_scores(corpus) -> np.ndarray:
    """Recompute features through the live engine path, as decide() does."""
    eng = DecisionEngine(environment=corpus["env"])
    eng.scorer.load(XP)
    eng.novelty.load(IP)
    out = []
    for r in sorted(corpus["rows"], key=lambda x: x["payload"]["timestamp"]):
        msg = eng._coerce(r["payload"])
        wire = msg.to_wire()
        hist = eng.env.get_customer_history(msg.customer_id)
        f = eng.scorer.features(wire, hist)
        if r["label"] == 0:
            out.append(eng.scorer.score_from_features(f))
        eng.scorer.observe(wire)
    return np.array(out)


def main() -> None:
    from defense.realtime import VelocityScorer
    sc = VelocityScorer()
    sc.load(XP)

    for name in ("calib", "eval"):
        c = pickle.load(open(f"/tmp/split_{name}.pkl", "rb"))
        st = stored_scores(c, sc)
        lv = live_scores(c)
        print("\n=== %s split ===" % name)
        print("  stored features: " + "  ".join(
            "q%.3g=%.5f" % (q, np.quantile(st, q)) for q in QS))
        print("  live   features: " + "  ".join(
            "q%.3g=%.5f" % (q, np.quantile(lv, q)) for q in QS))
        print("  n_legit stored=%d live=%d" % (len(st), len(lv)))


if __name__ == "__main__":
    main()
