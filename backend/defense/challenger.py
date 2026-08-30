"""Champion/challenger shadow evaluation for fraud decisioning.

The production champion remains the existing DecisionEngine. This module trains
and scores the same graph-augmented RandomForest + One-Class SVM architecture
used by the fair model sweep, in shadow mode only. It accumulates disagreement,
FPR/TPR and action metrics without ever changing an authorization outcome.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from data.corpus_builder import build_corpus
from defense.graph import EntityGraph
from defense.realtime import FEATURE_NAMES

DECLINE_THRESHOLD = 0.85
STEPUP_THRESHOLD = 0.60
MANUAL_THRESHOLD = 0.30

GRAPH_NAMES = [
    "graph_risk",
    "graph_component_customers",
    "graph_shared_infra_count",
    "graph_max_linked_customers",
]

DEFAULT_TRAIN_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 230,
    "ATTACK_2_SYNTHETIC_MULE_RING": 230,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 240,
}


class ShadowChallenger:
    """Shadow-only graph-augmented challenger and running comparison ledger."""

    name = "random_forest+one_class_svm+graph_context"

    def __init__(self) -> None:
        self.rf: RandomForestClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.ocsvm: OneClassSVM | None = None
        self.ready = False
        self.training_metrics: dict = {"status": "untrained"}
        self.compared = 0
        self.disagreements = 0
        self.legit_rows = 0
        self.attack_rows = 0
        self.champion_legit_flagged = 0
        self.challenger_legit_flagged = 0
        self.champion_attack_flagged = 0
        self.challenger_attack_flagged = 0
        self.champion_decisions: Counter[str] = Counter()
        self.challenger_decisions: Counter[str] = Counter()

    @staticmethod
    def _graph_values(ring: dict) -> list[float]:
        shared = ring.get("shared_infra", [])
        return [
            float(ring.get("risk_score", 0.0)),
            float(ring.get("component_customers", 1)),
            float(len(shared)),
            float(max((item.get("linked_customers", 0) for item in shared), default=0)),
        ]

    @staticmethod
    def _vector(features: dict, graph_values: Iterable[float]) -> list[float]:
        return [float(features[name]) for name in FEATURE_NAMES] + [
            float(value) for value in graph_values
        ]

    @staticmethod
    def _graph_augmented_rows(rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
        graph = EntityGraph()
        x: list[list[float]] = []
        y: list[int] = []
        ordered = sorted(rows, key=lambda row: row["payload"]["timestamp"])
        for row in ordered:
            ring = graph.check(row["payload"])
            x.append(
                ShadowChallenger._vector(
                    row["features"], ShadowChallenger._graph_values(ring)
                )
            )
            y.append(int(row["label"]))
            graph.observe(row["payload"])
        return np.asarray(x, dtype=float), np.asarray(y, dtype=int)

    def fit_rows(self, rows: list[dict]) -> dict:
        """Fit the shadow models on a labeled corpus; safe to call in a worker."""
        started = time.perf_counter()
        x, y = self._graph_augmented_rows(rows)
        if x.size == 0 or np.unique(y).size < 2:
            raise ValueError("shadow challenger requires non-empty two-class training data")

        # Match benchmark_model_sweep.py exactly so "challenger" means the
        # tested architecture, not a hand-tuned cousin of it.
        rf = RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(x, y)

        legit = x[y == 0]
        scaler = StandardScaler()
        legit_scaled = scaler.fit_transform(legit)
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
        ocsvm.fit(legit_scaled)

        self.rf = rf
        self.scaler = scaler
        self.ocsvm = ocsvm
        self.ready = True
        self.training_metrics = {
            "status": "ready",
            "name": self.name,
            "train_rows": int(len(y)),
            "legit_rows": int((y == 0).sum()),
            "attack_rows": int((y == 1).sum()),
            "feature_count": int(x.shape[1]),
            "feature_names": list(FEATURE_NAMES) + GRAPH_NAMES,
            "fit_seconds": round(time.perf_counter() - started, 4),
            "matches_model_sweep_contract": True,
        }
        return dict(self.training_metrics)

    def train_default(self) -> dict:
        corpus = build_corpus(
            n_legit=4500,
            attack_counts=DEFAULT_TRAIN_COUNTS,
            seed=123,
        )
        return self.fit_rows(corpus["rows"])

    @staticmethod
    def _ladder(probability: float, anomaly: bool, ring_detected: bool) -> str:
        if ring_detected:
            return "DECLINE"
        if probability > DECLINE_THRESHOLD:
            return "DECLINE"
        if anomaly and probability <= MANUAL_THRESHOLD:
            return "MANUAL_REVIEW"
        if probability > STEPUP_THRESHOLD or anomaly:
            return "STEP_UP"
        return "APPROVE"

    @staticmethod
    def _live_graph_values(champion_record: dict) -> list[float]:
        scores = champion_record.get("scores", {})
        return [
            float(scores.get("ring_risk", 0.0)),
            float(scores.get("graph_component_customers", 1)),
            float(scores.get("graph_shared_infra_count", 0)),
            float(scores.get("graph_max_linked_customers", 0)),
        ]

    def score_record(self, champion_record: dict) -> dict | None:
        if not self.ready or self.rf is None or self.scaler is None or self.ocsvm is None:
            return None

        graph_values = self._live_graph_values(champion_record)
        vector = np.asarray(
            [self._vector(champion_record["features"], graph_values)], dtype=float
        )
        probability = float(self.rf.predict_proba(vector)[0, 1])
        anomaly_raw = float(self.ocsvm.decision_function(self.scaler.transform(vector))[0])
        anomaly = anomaly_raw < 0.0
        ring_detected = bool(champion_record.get("scores", {}).get("ring_detected"))
        decision = self._ladder(probability, anomaly, ring_detected)
        return {
            "name": self.name,
            "decision": decision,
            "fraud_probability": round(probability, 5),
            "is_anomaly": anomaly,
            "novelty_margin": round(-anomaly_raw, 5),
            "graph_context": {
                name: round(value, 4) for name, value in zip(GRAPH_NAMES, graph_values)
            },
            "shadow_only": True,
        }

    def observe(self, champion_record: dict, truth: str | None = None) -> dict | None:
        challenger = self.score_record(champion_record)
        if challenger is None:
            return None

        champion_decision = str(champion_record["decision"])
        challenger_decision = str(challenger["decision"])
        self.compared += 1
        self.disagreements += int(champion_decision != challenger_decision)
        self.champion_decisions[champion_decision] += 1
        self.challenger_decisions[challenger_decision] += 1

        if truth is not None:
            if truth == "legit":
                self.legit_rows += 1
                self.champion_legit_flagged += int(champion_decision != "APPROVE")
                self.challenger_legit_flagged += int(challenger_decision != "APPROVE")
            else:
                self.attack_rows += 1
                self.champion_attack_flagged += int(champion_decision != "APPROVE")
                self.challenger_attack_flagged += int(challenger_decision != "APPROVE")

        return {
            "champion": {
                "decision": champion_decision,
                "velocity": champion_record.get("scores", {}).get("velocity"),
                "ring_risk": champion_record.get("scores", {}).get("ring_risk"),
            },
            "challenger": challenger,
            "disagrees": champion_decision != challenger_decision,
        }

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else round(numerator / denominator, 5)

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "ready": self.ready,
            "training": dict(self.training_metrics),
            "compared": self.compared,
            "disagreements": self.disagreements,
            "disagreement_rate": self._rate(self.disagreements, self.compared),
            "simulator_truth_metrics": {
                "legit_rows": self.legit_rows,
                "attack_rows": self.attack_rows,
                "champion_fpr": self._rate(self.champion_legit_flagged, self.legit_rows),
                "challenger_fpr": self._rate(self.challenger_legit_flagged, self.legit_rows),
                "champion_tpr": self._rate(self.champion_attack_flagged, self.attack_rows),
                "challenger_tpr": self._rate(self.challenger_attack_flagged, self.attack_rows),
            },
            "champion_decisions": dict(self.champion_decisions),
            "challenger_decisions": dict(self.challenger_decisions),
            "controls_live_authorizations": False,
        }


__all__ = ["ShadowChallenger", "DEFAULT_TRAIN_COUNTS", "GRAPH_NAMES"]
