"""
Layer 1 — Real-time velocity scorer (XGBoost over behavioral features).

Fraud in 2026 is sequence behavior, not row outliers: the same $120 basket is
benign as a Tuesday grocery run and screaming as the 4th CNP ticket on a
fresh device inside 10 minutes. This layer turns the event stream into a
rolling feature state and lets a shallow gradient-boosted tree do one thing
well: rank that behavioral context.

Feature discipline:
  * features are computed from state STRICTLY BEFORE the current transaction
    (observe-after-score), so training labels can never leak the present;
  * the SAME FeatureExtractor runs at train time (corpus replay) and at
    inference time — one implementation, zero skew.

Model: XGBoost binary classifier. Legit baseline = 0, attacks 1-3 = 1.
Attack 4 is deliberately withheld from training so later steps can demo
generalization to NOVEL attacks (with the Isolation Forest + graph layers
carrying the load).
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

MODE_CODE = {"ECOM": 0, "CONTACTLESS": 1, "CNP": 2, "CHIP": 3, "SWIPE": 4}
TDS_CODE = {"Y": 0, "A": 1, "N": 2}

FEATURE_NAMES = [
    # customer-centric velocity
    "cust_txn_count_10m",
    "cust_amount_sum_10m",
    "amount_over_mean30",
    "cust_mcc_distinct_1h",
    # device-centric
    "device_age_hours",
    "dev_txn_count_10m",
    # merchant-centric velocity (the compromised-endpoint tell)
    "merch_txn_count_10m",
    "merch_distinct_custs_10m",
    # static / context
    "device_known",
    "pos_entry_code",
    "tds_code",
]

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_model.json"


def _ts(wire: dict) -> datetime:
    ts = wire.get("timestamp")
    return datetime.fromisoformat(ts) if isinstance(ts, str) else ts


class FeatureExtractor:
    """
    Rolling behavioral state over the accepted-transaction stream.

    `env` (optional) enables device-binding lookups (`device_known`); without
    it the extractor still works and reports bindings as unknown.

    Deques are bounded; windows are scanned linearly which is plenty fast for
    arena-scale streams (~10^4 events).
    """

    def __init__(self, env=None, maxlen: int = 500) -> None:
        self.env = env
        self.cust_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))   # (ts, amount, mcc)
        self.dev_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))    # (ts,)
        self.merch_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))  # (ts, customer_id)
        self.dev_first_seen: dict[str, datetime] = {}

    def device_known(self, wire: dict) -> bool:
        c = self.env.customers.get(wire["customer_id"]) if self.env else None
        return bool(c and wire["device_id"] in c.devices)

    def features(self, wire: dict, history: list | None = None) -> dict:
        """Pre-transaction behavioral features for one payload (wire format)."""
        ts = _ts(wire)
        cid, did, mid = wire["customer_id"], wire["device_id"], wire["merchant_id"]
        amt = float(wire["amount"])

        cust_all = self.cust_ev[cid]
        c10 = [e for e in cust_all if ts - e[0] <= timedelta(minutes=10)]
        c1h = [e for e in cust_all if ts - e[0] <= timedelta(hours=1)]
        d10 = [e for e in self.dev_ev[did] if ts - e[0] <= timedelta(minutes=10)]
        m10 = [e for e in self.merch_ev[mid] if ts - e[0] <= timedelta(minutes=10)]

        hist_amts = [e[1] for e in cust_all]
        mean30 = float(np.mean(hist_amts)) if hist_amts else amt

        age_h = (
            0.0 if did not in self.dev_first_seen
            else (ts - self.dev_first_seen[did]).total_seconds() / 3600.0
        )

        feats = {
            "cust_txn_count_10m": len(c10),
            "cust_amount_sum_10m": round(sum(e[1] for e in c10), 2),
            "amount_over_mean30": round(amt / (mean30 + 1e-6), 3),
            "cust_mcc_distinct_1h": len({e[2] for e in c1h}),
            "device_age_hours": round(age_h, 4),
            "dev_txn_count_10m": len(d10),
            "merch_txn_count_10m": len(m10),
            "merch_distinct_custs_10m": len({e[1] for e in m10}),
            "device_known": int(self.device_known(wire)),
            "pos_entry_code": MODE_CODE[wire["pos_entry_mode"]],
            "tds_code": TDS_CODE[wire["3ds_status"]],
        }

        # Cold-start supplement: on an issuer we'd pull the ledger here; the
        # caller may pass PaymentEnvironment.get_customer_history() output.
        if history and not hist_amts:
            amts = [h["payload"]["amount"] for h in history]
            if amts:
                feats["amount_over_mean30"] = round(amt / (float(np.mean(amts)) + 1e-6), 3)
        return feats

    def observe(self, wire: dict) -> None:
        """Fold an ACCEPTED transaction into state. Call AFTER scoring."""
        ts = _ts(wire)
        self.cust_ev[wire["customer_id"]].append((ts, float(wire["amount"]), int(wire["mcc"])))
        self.dev_ev[wire["device_id"]].append((ts,))
        self.merch_ev[wire["merchant_id"]].append((ts, wire["customer_id"]))
        self.dev_first_seen.setdefault(wire["device_id"], ts)


class VelocityScorer:
    """
    XGBoost fraud-probability scorer over extracted behavioral features.

    Works in three modes:
      * trained model loaded from models/xgb_model.json (production path),
      * trained in-memory via .train(rows) (tests / notebooks),
      * untrained fallback heuristic (graceful degradation so the pipeline
        never hard-crashes before models are built — clearly flagged).
    """

    def __init__(self, model_path: str | Path | None = DEFAULT_MODEL_PATH) -> None:
        self.extractor = FeatureExtractor()
        self.model = None
        self.model_source = "untrained"
        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ---------------- feature plumbing ---------------- #

    def features(self, payload: dict, history: list | None = None) -> dict:
        return self.extractor.features(payload, history)

    def observe(self, payload: dict) -> None:
        self.extractor.observe(payload)

    @staticmethod
    def vectorize(feats: dict) -> list[float]:
        return [float(feats[f]) for f in FEATURE_NAMES]

    # ---------------- scoring ---------------- #

    def score_from_features(self, feats: dict) -> float:
        x = np.array([self.vectorize(feats)])
        if self.model is not None:
            return float(self.model.predict_proba(x)[0][1])
        return self._heuristic_score(feats)

    def score(self, payload: dict, history: list | None = None) -> float:
        """Mandated API: fraud probability in [0,1] for one wire payload."""
        return self.score_from_features(self.features(payload, history))

    @staticmethod
    def _heuristic_score(f: dict) -> float:
        """Transparent hand-weighted fallback when no model is available."""
        s = (
            0.09 * f["cust_txn_count_10m"]
            + 0.02 * f["dev_txn_count_10m"]
            + 0.05 * f["merch_distinct_custs_10m"]
            + 0.18 * min(f["amount_over_mean30"], 5.0)
            + 0.25 * (1 - f["device_known"])
            + 0.15 * (1.0 if f["device_age_hours"] < 0.1 else 0.0)
        )
        return float(min(1.0, s))

    # ---------------- training ---------------- #

    def train(self, rows: list[dict], save_path: str | Path | None = DEFAULT_MODEL_PATH) -> dict:
        """rows: [{label:int 0|1, features:dict}] from the corpus builder."""
        from xgboost import XGBClassifier

        X = np.array([self.vectorize(r["features"]) for r in rows])
        y = np.array([int(r["label"]) for r in rows])
        pos, neg = int(y.sum()), int((y == 0).sum())

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,   # regularized: resist 'unknown device => fraud' shortcuts
            gamma=0.1,
            scale_pos_weight=(neg / max(pos, 1)),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.model_source = "in_memory"
        acc = float((self.model.predict(X) == y).mean())
        metrics = {"train_rows": len(rows), "positives": pos, "train_accuracy": round(acc, 4)}

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(save_path)
            self.model_source = f"saved:{save_path}"
            metrics["saved_to"] = str(save_path)
        return metrics

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        """Persist the fitted booster (no retraining side effects)."""
        if self.model is None:
            raise RuntimeError("no trained model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        return path

    def load(self, path: str | Path) -> None:
        from xgboost import XGBClassifier

        self.model = XGBClassifier()
        self.model.load_model(path)
        self.model_source = f"loaded:{path}"


__all__ = [
    "VelocityScorer",
    "FeatureExtractor",
    "FEATURE_NAMES",
    "MODELS_DIR",
    "DEFAULT_MODEL_PATH",
]
