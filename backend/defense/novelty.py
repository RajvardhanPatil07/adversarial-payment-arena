"""
Layer 3 — Novelty detector (Isolation Forest).

Trained ONLY on legitimate baseline traffic — zero attack examples. This is
the layer that generalizes to attacks we've never seen (that's why attack 4
is withheld from every training path): anything that looks unlike honest
cardholder behavior gets quarantined for friction even when the supervised
scorer is confident.

contamination=0.02 mirrors the FP budget: we accept that ~2% of legit traffic
will trip the anomaly wire, and price that friction into the cost matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from defense.realtime import FEATURE_NAMES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "iforest_model.joblib"


class NoveltyDetector:
    """Unsupervised anomaly screen over the SAME feature space as the scorer."""

    def __init__(
        self,
        contamination: float = 0.01,
        model_path: str | Path | None = DEFAULT_MODEL_PATH,
    ) -> None:
        self.contamination = contamination
        self.model: IsolationForest | None = None
        self.model_source = "untrained"
        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ---------------- inference ---------------- #

    @staticmethod
    def vectorize(features: dict) -> list[float]:
        return [float(features[f]) for f in FEATURE_NAMES]

    def detect(self, payload: dict, features: dict) -> dict:
        """
        Mandated API. `payload` is accepted for interface symmetry (and future
        payload-conditional features); the decision uses the precomputed
        feature vector so train/infer representations can never diverge.

        Returns {is_anomaly: bool, anomaly_score: float} where anomaly_score
        is the negated decision_function (higher == weirder).
        """
        if self.model is None:
            return {"is_anomaly": False, "anomaly_score": 0.0, "model_source": "untrained"}
        x = np.array([self.vectorize(features)])
        raw = float(self.model.decision_function(x)[0])
        return {
            "is_anomaly": bool(self.model.predict(x)[0] == -1),
            "anomaly_score": round(-raw, 5),
            "model_source": self.model_source,
        }

    # ---------------- training ---------------- #

    def train(self, rows: list[dict], save_path: str | Path | None = DEFAULT_MODEL_PATH) -> dict:
        """rows: LEGIT-ONLY corpus rows [{label==0, features:dict}]."""
        X = np.array([
            self.vectorize(r["features"]) for r in rows if int(r["label"]) == 0
        ])
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
        self.model_source = "in_memory"
        metrics = {"train_rows_legit": len(X)}

        if save_path:
            import joblib

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, save_path)
            self.model_source = f"saved:{save_path}"
            metrics["saved_to"] = str(save_path)
        return metrics

    def load(self, path: str | Path) -> None:
        import joblib

        self.model = joblib.load(path)
        self.model_source = f"loaded:{path}"


__all__ = ["NoveltyDetector", "DEFAULT_MODEL_PATH"]
