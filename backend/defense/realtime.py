"""Real-time rolling feature extraction and XGBoost transaction scoring.

The saved model contract is defined by ``FEATURE_NAMES``. Feature state is
observe-after-score so the current transaction never leaks into its own
features. Container builds can use the optional ``arena_core`` PyO3 extension
for rolling-window state; local development falls back to equivalent Python
bounded deques.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    from arena_core import RollingFeatureState as _RustRollingFeatureState
except ImportError:
    _RustRollingFeatureState = None

MODE_CODE = {"ECOM": 0, "CONTACTLESS": 1, "CNP": 2, "CHIP": 3, "SWIPE": 4}
TDS_CODE = {"Y": 0, "A": 1, "N": 2}

FEATURE_NAMES = [
    "cust_txn_count_10m",
    "cust_amount_sum_10m",
    "amount_over_mean30",
    "cust_mcc_distinct_1h",
    "device_age_hours",
    "dev_txn_count_10m",
    "merch_txn_count_10m",
    "merch_distinct_custs_10m",
    "device_known",
    "pos_entry_code",
    "tds_code",
]

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_model.json"


def _ts(wire: dict) -> datetime:
    value = wire.get("timestamp")
    return datetime.fromisoformat(value) if isinstance(value, str) else value


class FeatureExtractor:
    """Rolling behavioral state with an optional Rust implementation."""

    def __init__(self, env=None, maxlen: int = 500) -> None:
        self.env = env
        self.maxlen = maxlen
        self._rust = _RustRollingFeatureState(maxlen) if _RustRollingFeatureState else None

        # Pure-Python fallback. These stay unused when the native extension is
        # present, but make source checkouts work without a Rust toolchain.
        self.cust_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.dev_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.merch_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.dev_first_seen: dict[str, datetime] = {}

    @property
    def backend(self) -> str:
        return "rust" if self._rust is not None else "python"

    def device_known(self, wire: dict) -> bool:
        customer = self.env.customers.get(wire["customer_id"]) if self.env else None
        return bool(customer and wire["device_id"] in customer.devices)

    def _cold_start_ratio(self, feats: dict, amount: float, history: list | None) -> None:
        if not history:
            return
        amounts = [h["payload"]["amount"] for h in history]
        if amounts:
            feats["amount_over_mean30"] = round(
                amount / (float(np.mean(amounts)) + 1e-6), 3
            )

    def _features_rust(self, wire: dict, history: list | None) -> dict:
        ts = _ts(wire)
        amount = float(wire["amount"])
        values = self._rust.features(
            ts.timestamp(),
            wire["customer_id"],
            wire["device_id"],
            wire["merchant_id"],
            amount,
        )
        feats = {
            "cust_txn_count_10m": int(values[0]),
            "cust_amount_sum_10m": round(float(values[1]), 2),
            "amount_over_mean30": round(float(values[2]), 3),
            "cust_mcc_distinct_1h": int(values[3]),
            "device_age_hours": round(float(values[4]), 4),
            "dev_txn_count_10m": int(values[5]),
            "merch_txn_count_10m": int(values[6]),
            "merch_distinct_custs_10m": int(values[7]),
            "device_known": int(self.device_known(wire)),
            "pos_entry_code": MODE_CODE[wire["pos_entry_mode"]],
            "tds_code": TDS_CODE[wire["3ds_status"]],
        }
        if int(values[8]) == 0:
            self._cold_start_ratio(feats, amount, history)
        return feats

    def _features_python(self, wire: dict, history: list | None) -> dict:
        ts = _ts(wire)
        cid, did, mid = wire["customer_id"], wire["device_id"], wire["merchant_id"]
        amount = float(wire["amount"])

        cust_all = self.cust_ev[cid]
        c10 = [e for e in cust_all if ts - e[0] <= timedelta(minutes=10)]
        c1h = [e for e in cust_all if ts - e[0] <= timedelta(hours=1)]
        d10 = [e for e in self.dev_ev[did] if ts - e[0] <= timedelta(minutes=10)]
        m10 = [e for e in self.merch_ev[mid] if ts - e[0] <= timedelta(minutes=10)]

        hist_amounts = [e[1] for e in cust_all]
        mean = float(np.mean(hist_amounts)) if hist_amounts else amount
        age_hours = (
            0.0
            if did not in self.dev_first_seen
            else (ts - self.dev_first_seen[did]).total_seconds() / 3600.0
        )

        feats = {
            "cust_txn_count_10m": len(c10),
            "cust_amount_sum_10m": round(sum(e[1] for e in c10), 2),
            "amount_over_mean30": round(amount / (mean + 1e-6), 3),
            "cust_mcc_distinct_1h": len({e[2] for e in c1h}),
            "device_age_hours": round(age_hours, 4),
            "dev_txn_count_10m": len(d10),
            "merch_txn_count_10m": len(m10),
            "merch_distinct_custs_10m": len({e[1] for e in m10}),
            "device_known": int(self.device_known(wire)),
            "pos_entry_code": MODE_CODE[wire["pos_entry_mode"]],
            "tds_code": TDS_CODE[wire["3ds_status"]],
        }
        if not hist_amounts:
            self._cold_start_ratio(feats, amount, history)
        return feats

    def features(self, wire: dict, history: list | None = None) -> dict:
        if self._rust is not None:
            return self._features_rust(wire, history)
        return self._features_python(wire, history)

    def observe(self, wire: dict) -> None:
        ts = _ts(wire)
        if self._rust is not None:
            self._rust.observe(
                ts.timestamp(),
                wire["customer_id"],
                wire["device_id"],
                wire["merchant_id"],
                float(wire["amount"]),
                int(wire["mcc"]),
            )
            return

        self.cust_ev[wire["customer_id"]].append(
            (ts, float(wire["amount"]), int(wire["mcc"]))
        )
        self.dev_ev[wire["device_id"]].append((ts,))
        self.merch_ev[wire["merchant_id"]].append((ts, wire["customer_id"]))
        self.dev_first_seen.setdefault(wire["device_id"], ts)


class VelocityScorer:
    """XGBoost probability scorer over the rolling feature contract."""

    def __init__(self, model_path: str | Path | None = DEFAULT_MODEL_PATH) -> None:
        self.extractor = FeatureExtractor()
        self.model = None
        self.model_source = "untrained"
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def features(self, payload: dict, history: list | None = None) -> dict:
        return self.extractor.features(payload, history)

    def observe(self, payload: dict) -> None:
        self.extractor.observe(payload)

    @staticmethod
    def vectorize(feats: dict) -> list[float]:
        return [float(feats[name]) for name in FEATURE_NAMES]

    def score_from_features(self, feats: dict) -> float:
        x = np.array([self.vectorize(feats)])
        if self.model is not None:
            return float(self.model.predict_proba(x)[0][1])
        return self._heuristic_score(feats)

    def score(self, payload: dict, history: list | None = None) -> float:
        return self.score_from_features(self.features(payload, history))

    @staticmethod
    def _heuristic_score(f: dict) -> float:
        score = (
            0.09 * f["cust_txn_count_10m"]
            + 0.02 * f["dev_txn_count_10m"]
            + 0.05 * f["merch_distinct_custs_10m"]
            + 0.18 * min(f["amount_over_mean30"], 5.0)
            + 0.25 * (1 - f["device_known"])
            + 0.15 * (1.0 if f["device_age_hours"] < 0.1 else 0.0)
        )
        return float(min(1.0, score))

    def train(self, rows: list[dict], save_path: str | Path | None = DEFAULT_MODEL_PATH) -> dict:
        from xgboost import XGBClassifier

        x = np.array([self.vectorize(row["features"]) for row in rows])
        y = np.array([int(row["label"]) for row in rows])
        positives = int(y.sum())
        negatives = int((y == 0).sum())

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            gamma=0.1,
            scale_pos_weight=(negatives / max(positives, 1)),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(x, y)
        self.model_source = "in_memory"
        accuracy = float((self.model.predict(x) == y).mean())
        metrics = {
            "train_rows": len(rows),
            "positives": positives,
            "train_accuracy": round(accuracy, 4),
        }

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(save_path)
            self.model_source = f"saved:{save_path}"
            metrics["saved_to"] = str(save_path)
        return metrics

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
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
