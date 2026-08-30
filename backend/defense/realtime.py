"""Real-time behavioral feature extraction and XGBoost scoring.

The production feature contract contains the original eleven rolling/context
features plus seven sequence-shape features introduced by the calibrated
14-family corpus. The optional ``arena_core`` extension still owns the hot
10-minute/1-hour counters; an incremental bounded sequence state supplies the
new long-horizon features without rescanning customer history on every event.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import sqrt
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
    "dev_distinct_custs_1h",
    "iat_regularity",
    "amount_escalation",
    "amount_band_tightness",
    "round_amount_frac",
    "merch_youth",
    "low_value_probe_ratio",
]

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_model.json"


def _ts(wire: dict) -> datetime:
    value = wire.get("timestamp")
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _is_round_amount(amount: float) -> bool:
    return abs(amount - round(amount / 10.0) * 10.0) < 0.5


@dataclass
class _CustomerSequence:
    maxlen: int
    events: deque = field(default_factory=deque)
    ordered: bool = True
    amount_sum: float = 0.0
    amount_sq_sum: float = 0.0
    round_count: int = 0
    up_pairs: int = 0
    gap_sum: float = 0.0
    gap_sq_sum: float = 0.0
    gap_count: int = 0

    def _drop_front(self) -> None:
        if not self.events:
            return
        old_ts, old_amount = self.events[0]
        if len(self.events) >= 2:
            next_ts, next_amount = self.events[1]
            gap = (next_ts - old_ts).total_seconds()
            if gap > 0:
                self.gap_sum -= gap
                self.gap_sq_sum -= gap * gap
                self.gap_count -= 1
            if next_amount > old_amount:
                self.up_pairs -= 1
        self.events.popleft()
        self.amount_sum -= old_amount
        self.amount_sq_sum -= old_amount * old_amount
        self.round_count -= int(_is_round_amount(old_amount))

    def _purge(self, now: datetime) -> None:
        while self.events and now - self.events[0][0] > timedelta(days=30):
            self._drop_front()

    def _scan(self, now: datetime, amount: float) -> dict:
        recent = [event for event in self.events if now - event[0] <= timedelta(days=30)]
        amounts = [event[1] for event in recent]
        n = len(recent)

        regularity = 0.0
        if n >= 3:
            gaps = [
                (recent[i][0] - recent[i - 1][0]).total_seconds()
                for i in range(1, n)
            ]
            gaps = [gap for gap in gaps if gap > 0]
            if len(gaps) >= 2:
                mean_gap = float(np.mean(gaps))
                if mean_gap > 0:
                    regularity = round(
                        1.0 / (1.0 + float(np.std(gaps)) / mean_gap), 4
                    )

        escalation = 0.0
        tightness = 0.0
        if len(amounts) >= 3:
            ups = sum(
                1 for i in range(1, len(amounts)) if amounts[i] > amounts[i - 1]
            )
            escalation = round(ups / (len(amounts) - 1) - 0.5, 4)
            mean_amount = float(np.mean(amounts))
            if mean_amount > 0:
                tightness = round(
                    1.0 / (1.0 + float(np.std(amounts)) / mean_amount * 4.0), 4
                )

        band_amounts = amounts + [amount]
        round_frac = round(
            sum(1 for value in band_amounts if _is_round_amount(value))
            / max(len(band_amounts), 1),
            4,
        )
        return {
            "iat_regularity": regularity,
            "amount_escalation": escalation,
            "amount_band_tightness": tightness,
            "round_amount_frac": round_frac,
        }

    def features(self, now: datetime, amount: float) -> dict:
        monotonic = not self.events or now >= self.events[-1][0]
        if not self.ordered or not monotonic:
            return self._scan(now, amount)

        self._purge(now)
        n = len(self.events)
        regularity = 0.0
        escalation = 0.0
        tightness = 0.0

        if n >= 3 and self.gap_count >= 2:
            mean_gap = self.gap_sum / self.gap_count
            variance = max(
                self.gap_sq_sum / self.gap_count - mean_gap * mean_gap, 0.0
            )
            if mean_gap > 0:
                regularity = round(1.0 / (1.0 + sqrt(variance) / mean_gap), 4)

        if n >= 3:
            escalation = round(self.up_pairs / (n - 1) - 0.5, 4)
            mean_amount = self.amount_sum / n
            if mean_amount > 0:
                variance = max(
                    self.amount_sq_sum / n - mean_amount * mean_amount, 0.0
                )
                tightness = round(
                    1.0 / (1.0 + sqrt(variance) / mean_amount * 4.0), 4
                )

        round_frac = round(
            (self.round_count + int(_is_round_amount(amount))) / max(n + 1, 1), 4
        )
        return {
            "iat_regularity": regularity,
            "amount_escalation": escalation,
            "amount_band_tightness": tightness,
            "round_amount_frac": round_frac,
        }

    def observe(self, now: datetime, amount: float) -> None:
        monotonic = not self.events or now >= self.events[-1][0]
        if not monotonic:
            self.ordered = False

        if self.ordered:
            self._purge(now)
            if len(self.events) >= self.maxlen:
                self._drop_front()
            if self.events:
                last_ts, last_amount = self.events[-1]
                gap = (now - last_ts).total_seconds()
                if gap > 0:
                    self.gap_sum += gap
                    self.gap_sq_sum += gap * gap
                    self.gap_count += 1
                if amount > last_amount:
                    self.up_pairs += 1
            self.events.append((now, amount))
            self.amount_sum += amount
            self.amount_sq_sum += amount * amount
            self.round_count += int(_is_round_amount(amount))
            return

        if len(self.events) >= self.maxlen:
            self.events.popleft()
        self.events.append((now, amount))


@dataclass
class _DeviceSequence:
    maxlen: int
    events: deque = field(default_factory=deque)
    ordered: bool = True
    one_hour: deque = field(default_factory=deque)
    one_hour_counts: dict[str, int] = field(default_factory=dict)
    seven_day: deque = field(default_factory=deque)
    seven_day_low: int = 0

    @staticmethod
    def _dec(mapping: dict[str, int], key: str) -> None:
        value = mapping.get(key, 0) - 1
        if value <= 0:
            mapping.pop(key, None)
        else:
            mapping[key] = value

    def _purge(self, now: datetime) -> None:
        while self.one_hour and now - self.one_hour[0][0] > timedelta(hours=1):
            _, customer_id = self.one_hour.popleft()
            self._dec(self.one_hour_counts, customer_id)
        while self.seven_day and now - self.seven_day[0][0] > timedelta(days=7):
            _, amount = self.seven_day.popleft()
            self.seven_day_low -= int(amount < 50.0)

    def features(self, now: datetime) -> dict:
        monotonic = not self.events or now >= self.events[-1][0]
        if not self.ordered or not monotonic:
            one_hour_customers = {
                customer_id
                for event_ts, customer_id, _ in self.events
                if now - event_ts <= timedelta(hours=1)
            }
            seven_day_amounts = [
                amount
                for event_ts, _, amount in self.events
                if now - event_ts <= timedelta(days=7)
            ]
            ratio = 0.0
            if len(seven_day_amounts) >= 3:
                ratio = round(
                    sum(1 for amount in seven_day_amounts if amount < 50.0)
                    / len(seven_day_amounts),
                    4,
                )
            return {
                "dev_distinct_custs_1h": len(one_hour_customers),
                "low_value_probe_ratio": ratio,
            }

        self._purge(now)
        ratio = 0.0
        if len(self.seven_day) >= 3:
            ratio = round(self.seven_day_low / len(self.seven_day), 4)
        return {
            "dev_distinct_custs_1h": len(self.one_hour_counts),
            "low_value_probe_ratio": ratio,
        }

    def observe(self, now: datetime, customer_id: str, amount: float) -> None:
        monotonic = not self.events or now >= self.events[-1][0]
        if not monotonic:
            self.ordered = False

        if len(self.events) >= self.maxlen:
            self.events.popleft()
        self.events.append((now, customer_id, amount))

        if not self.ordered:
            return
        self._purge(now)
        self.one_hour.append((now, customer_id))
        self.one_hour_counts[customer_id] = self.one_hour_counts.get(customer_id, 0) + 1
        self.seven_day.append((now, amount))
        self.seven_day_low += int(amount < 50.0)


class _SequenceFeatureState:
    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self.customers: dict[str, _CustomerSequence] = {}
        self.devices: dict[str, _DeviceSequence] = {}
        self.merchant_first_seen: dict[str, datetime] = {}

    def features(self, wire: dict) -> dict:
        now = _ts(wire)
        amount = float(wire["amount"])
        cid = wire["customer_id"]
        did = wire["device_id"]
        mid = wire["merchant_id"]

        customer = self.customers.get(cid)
        if customer is None:
            customer_features = {
                "iat_regularity": 0.0,
                "amount_escalation": 0.0,
                "amount_band_tightness": 0.0,
                "round_amount_frac": round(int(_is_round_amount(amount)), 4),
            }
        else:
            customer_features = customer.features(now, amount)

        device = self.devices.get(did)
        device_features = (
            device.features(now)
            if device is not None
            else {"dev_distinct_custs_1h": 0, "low_value_probe_ratio": 0.0}
        )

        first_seen = self.merchant_first_seen.get(mid)
        age_days = 0.0 if first_seen is None else (now - first_seen).total_seconds() / 86400.0
        merchant_youth = round(1.0 / (1.0 + max(age_days, 0.0)), 4)
        return {
            **device_features,
            **customer_features,
            "merch_youth": merchant_youth,
        }

    def observe(self, wire: dict) -> None:
        now = _ts(wire)
        amount = float(wire["amount"])
        cid = wire["customer_id"]
        did = wire["device_id"]
        mid = wire["merchant_id"]

        customer = self.customers.setdefault(cid, _CustomerSequence(self.maxlen))
        customer.observe(now, amount)
        device = self.devices.setdefault(did, _DeviceSequence(self.maxlen))
        device.observe(now, cid, amount)
        self.merchant_first_seen.setdefault(mid, now)


class FeatureExtractor:
    """Rolling behavioral state with optional Rust acceleration."""

    def __init__(self, env=None, maxlen: int = 500) -> None:
        self.env = env
        self.maxlen = maxlen
        self._rust = _RustRollingFeatureState(maxlen) if _RustRollingFeatureState else None
        self._sequence = _SequenceFeatureState(maxlen)

        self.cust_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.dev_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.merch_ev: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.dev_first_seen: dict[str, datetime] = {}

    @property
    def backend(self) -> str:
        return "rust" if self._rust is not None else "python"

    def state_sizes(self) -> dict[str, int]:
        if self._rust is not None:
            customers, devices, merchants, first_seen = self._rust.state_sizes()
            return {
                "customers": int(customers),
                "devices": int(devices),
                "merchants": int(merchants),
                "devices_first_seen": int(first_seen),
            }
        return {
            "customers": len(self.cust_ev),
            "devices": len(self.dev_ev),
            "merchants": len(self.merch_ev),
            "devices_first_seen": len(self.dev_first_seen),
        }

    def device_known(self, wire: dict) -> bool:
        customer = self.env.customers.get(wire["customer_id"]) if self.env else None
        return bool(customer and wire["device_id"] in customer.devices)

    def _cold_start_ratio(self, feats: dict, amount: float, history: list | None) -> None:
        if not history:
            return
        amounts = [float(event["payload"]["amount"]) for event in history]
        if amounts:
            feats["amount_over_mean30"] = round(
                amount / (float(np.mean(amounts)) + 1e-6), 3
            )

    def _rust_values_to_features(
        self,
        wire: dict,
        history: list | None,
        amount: float,
        values,
    ) -> dict:
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
        feats.update(self._sequence.features(wire))
        return feats

    def _features_rust(self, wire: dict, history: list | None) -> dict:
        now = _ts(wire)
        amount = float(wire["amount"])
        values = self._rust.features(
            now.timestamp(),
            wire["customer_id"],
            wire["device_id"],
            wire["merchant_id"],
            amount,
        )
        return self._rust_values_to_features(wire, history, amount, values)

    def _features_python(self, wire: dict, history: list | None) -> dict:
        now = _ts(wire)
        cid, did, mid = wire["customer_id"], wire["device_id"], wire["merchant_id"]
        amount = float(wire["amount"])

        customer = self.cust_ev[cid]
        customer_count_10m = 0
        customer_amount_10m = 0.0
        customer_mcc_1h: set[int] = set()
        historical_total = 0.0
        historical_count = 0
        for event_ts, event_amount, event_mcc in customer:
            age = now - event_ts
            if age <= timedelta(minutes=10):
                customer_count_10m += 1
                customer_amount_10m += event_amount
            if age <= timedelta(hours=1):
                customer_mcc_1h.add(event_mcc)
            historical_total += event_amount
            historical_count += 1

        device_count_10m = sum(
            1
            for (event_ts,) in self.dev_ev[did]
            if now - event_ts <= timedelta(minutes=10)
        )
        merchant_count_10m = 0
        merchant_customers_10m: set[str] = set()
        for event_ts, customer_id in self.merch_ev[mid]:
            if now - event_ts <= timedelta(minutes=10):
                merchant_count_10m += 1
                merchant_customers_10m.add(customer_id)

        mean = historical_total / historical_count if historical_count else amount
        age_hours = (
            0.0
            if did not in self.dev_first_seen
            else (now - self.dev_first_seen[did]).total_seconds() / 3600.0
        )
        feats = {
            "cust_txn_count_10m": customer_count_10m,
            "cust_amount_sum_10m": round(customer_amount_10m, 2),
            "amount_over_mean30": round(amount / (mean + 1e-6), 3),
            "cust_mcc_distinct_1h": len(customer_mcc_1h),
            "device_age_hours": round(age_hours, 4),
            "dev_txn_count_10m": device_count_10m,
            "merch_txn_count_10m": merchant_count_10m,
            "merch_distinct_custs_10m": len(merchant_customers_10m),
            "device_known": int(self.device_known(wire)),
            "pos_entry_code": MODE_CODE[wire["pos_entry_mode"]],
            "tds_code": TDS_CODE[wire["3ds_status"]],
        }
        if not historical_count:
            self._cold_start_ratio(feats, amount, history)
        feats.update(self._sequence.features(wire))
        return feats

    def features(self, wire: dict, history: list | None = None) -> dict:
        if self._rust is not None:
            return self._features_rust(wire, history)
        return self._features_python(wire, history)

    def _observe_python_base(self, wire: dict) -> None:
        now = _ts(wire)
        self.cust_ev[wire["customer_id"]].append(
            (now, float(wire["amount"]), int(wire["mcc"]))
        )
        self.dev_ev[wire["device_id"]].append((now,))
        self.merch_ev[wire["merchant_id"]].append((now, wire["customer_id"]))
        self.dev_first_seen.setdefault(wire["device_id"], now)

    def features_and_observe(self, wire: dict, history: list | None = None) -> dict:
        if self._rust is None:
            feats = self._features_python(wire, history)
            self._observe_python_base(wire)
            self._sequence.observe(wire)
            return feats

        now = _ts(wire)
        amount = float(wire["amount"])
        values = self._rust.features_and_observe(
            now.timestamp(),
            wire["customer_id"],
            wire["device_id"],
            wire["merchant_id"],
            amount,
            int(wire["mcc"]),
        )
        feats = self._rust_values_to_features(wire, history, amount, values)
        self._sequence.observe(wire)
        return feats

    def observe(self, wire: dict) -> None:
        now = _ts(wire)
        if self._rust is not None:
            self._rust.observe(
                now.timestamp(),
                wire["customer_id"],
                wire["device_id"],
                wire["merchant_id"],
                float(wire["amount"]),
                int(wire["mcc"]),
            )
        else:
            self._observe_python_base(wire)
        self._sequence.observe(wire)


class VelocityScorer:
    """XGBoost fraud-probability scorer over the shared feature contract."""

    def __init__(self, model_path: str | Path | None = DEFAULT_MODEL_PATH) -> None:
        self.extractor = FeatureExtractor()
        self.model = None
        self.model_source = "untrained"
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def features(self, payload: dict, history: list | None = None) -> dict:
        return self.extractor.features(payload, history)

    def features_and_observe(self, payload: dict, history: list | None = None) -> dict:
        return self.extractor.features_and_observe(payload, history)

    def observe(self, payload: dict) -> None:
        self.extractor.observe(payload)

    @staticmethod
    def vectorize(feats: dict) -> list[float]:
        return [float(feats[name]) for name in FEATURE_NAMES]

    def score_many_from_features(self, feature_rows: list[dict]) -> list[float]:
        if not feature_rows:
            return []
        if self.model is None:
            return [self._heuristic_score(feats) for feats in feature_rows]
        x = np.asarray([self.vectorize(feats) for feats in feature_rows])
        probabilities = self.model.predict_proba(x)[:, 1]
        return [float(value) for value in probabilities]

    def score_from_features(self, feats: dict) -> float:
        if self.model is None:
            return self._heuristic_score(feats)
        x = np.asarray([self.vectorize(feats)])
        return float(self.model.predict_proba(x)[0][1])

    def score(self, payload: dict, history: list | None = None) -> float:
        return self.score_from_features(self.features(payload, history))

    @staticmethod
    def _heuristic_score(features: dict) -> float:
        score = (
            0.09 * features["cust_txn_count_10m"]
            + 0.02 * features["dev_txn_count_10m"]
            + 0.05 * features["merch_distinct_custs_10m"]
            + 0.18 * min(features["amount_over_mean30"], 5.0)
            + 0.25 * (1 - features["device_known"])
            + 0.15 * (1.0 if features["device_age_hours"] < 0.1 else 0.0)
        )
        return float(min(1.0, score))

    def train(
        self,
        rows: list[dict],
        save_path: str | Path | None = DEFAULT_MODEL_PATH,
    ) -> dict:
        from xgboost import XGBClassifier

        x = np.asarray([self.vectorize(row["features"]) for row in rows])
        y = np.asarray([int(row["label"]) for row in rows])
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
            scale_pos_weight=negatives / max(positives, 1),
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
