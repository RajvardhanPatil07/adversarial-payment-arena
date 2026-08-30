"""Compare production-plausible fraud model families on one reproducible corpus.

The sweep deliberately scores both detection quality and batch inference speed.
It uses the existing transaction feature contract plus an optional compact set
of local graph-context features derived from the exact EntityGraph semantics.

Heavy GNN/Transformer runtimes are not silently treated as equivalent: the
current training corpus stores per-transaction tabular features, not learned
node/edge sequence tensors. The graph-augmented candidates therefore test the
useful part we can validate today -- whether local structural context improves
held-out fraud detection enough to justify more graph learning later.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from data.corpus_builder import build_corpus
from defense.graph import EntityGraph
from defense.realtime import FEATURE_NAMES

TRAIN_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 230,
    "ATTACK_2_SYNTHETIC_MULE_RING": 230,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 240,
}
EVAL_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 80,
    "ATTACK_2_SYNTHETIC_MULE_RING": 80,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 80,
    "ATTACK_4_CNP_HIGH_VELOCITY": 80,
    "ATTACK_5_APP_SCAM_PERSONALISED": 80,
    "ATTACK_6_VPA_RENTAL_MULE": 80,
    "ATTACK_7_SYNCHRONISED_BURST_CASHOUT": 80,
    "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING": 80,
}
GRAPH_NAMES = [
    "graph_risk",
    "graph_component_customers",
    "graph_shared_infra_count",
    "graph_max_linked_customers",
]


def graph_context(rows: list[dict]) -> tuple[np.ndarray, list[dict]]:
    graph = EntityGraph()
    values: list[list[float]] = []
    rings: list[dict] = []
    for row in rows:
        ring = graph.check(row["payload"])
        shared = ring.get("shared_infra", [])
        values.append([
            float(ring["risk_score"]),
            float(ring["component_customers"]),
            float(len(shared)),
            float(max((item["linked_customers"] for item in shared), default=0)),
        ])
        rings.append(ring)
        graph.observe(row["payload"])
    return np.asarray(values, dtype=np.float64), rings


def base_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )


def supervised_models(seed: int) -> dict[str, object]:
    return {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            gamma=0.1,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            max_depth=5,
            learning_rate=0.08,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=seed,
        ),
        "small_mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-4,
            max_iter=250,
            early_stopping=True,
            random_state=seed,
        ),
    }


class NoveltyBase:
    def fit(self, x: np.ndarray) -> "NoveltyBase":
        raise NotImplementedError

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class IFNovelty(NoveltyBase):
    def __init__(self, seed: int) -> None:
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray) -> "IFNovelty":
        self.model.fit(x)
        return self

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.decision_function(x)
        return raw < 0.0, -raw


class PCANovelty(NoveltyBase):
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.threshold = 0.0

    def fit(self, x: np.ndarray) -> "PCANovelty":
        z = self.scaler.fit_transform(x)
        components = max(1, min(6, z.shape[1] - 1))
        self.pca = PCA(n_components=components, random_state=42).fit(z)
        scores = self._scores_z(z)
        self.threshold = float(np.quantile(scores, 0.99))
        return self

    def _scores_z(self, z: np.ndarray) -> np.ndarray:
        assert self.pca is not None
        recon = self.pca.inverse_transform(self.pca.transform(z))
        return np.mean((z - recon) ** 2, axis=1)

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._scores_z(self.scaler.transform(x))
        return values > self.threshold, values


class DiagonalGaussianNovelty(NoveltyBase):
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.threshold = 0.0

    def fit(self, x: np.ndarray) -> "DiagonalGaussianNovelty":
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-6)
        values = self._scores(x)
        self.threshold = float(np.quantile(values, 0.99))
        return self

    def _scores(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        z = (x - self.mean) / self.std
        return np.mean(z * z, axis=1)

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._scores(x)
        return values > self.threshold, values


class OCSVMNovelty(NoveltyBase):
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)

    def fit(self, x: np.ndarray) -> "OCSVMNovelty":
        self.model.fit(self.scaler.fit_transform(x))
        return self

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.decision_function(self.scaler.transform(x)).reshape(-1)
        return raw < 0.0, -raw


def novelty_models(seed: int) -> dict[str, NoveltyBase]:
    return {
        "isolation_forest": IFNovelty(seed),
        "pca_reconstruction": PCANovelty(),
        "diagonal_gaussian": DiagonalGaussianNovelty(),
        "one_class_svm": OCSVMNovelty(),
    }


def predict_probability(model: object, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def benchmark_call(fn, x: np.ndarray, target_rows: int = 200_000, repeats: int = 3) -> float:
    copies = max(1, math.ceil(target_rows / len(x)))
    bench = np.tile(x, (copies, 1))[:target_rows]
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(bench)
        timings.append(time.perf_counter() - started)
    elapsed = statistics.median(timings)
    return float(len(bench) / elapsed)


def final_metrics(
    scores: np.ndarray,
    anomalies: np.ndarray,
    rings: list[dict],
    rows: list[dict],
) -> dict:
    decisions: list[str] = []
    per_family: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    fp = 0
    attacks = 0
    caught = 0
    for idx, row in enumerate(rows):
        is_attack = int(row["label"]) == 1
        attack_id = row["attack_id"]
        ring = rings[idx]
        score = float(scores[idx])
        anomaly = bool(anomalies[idx])
        if ring["ring_detected"]:
            decision = "DECLINE"
        elif score > 0.85:
            decision = "DECLINE"
        elif anomaly and score <= 0.30:
            decision = "MANUAL_REVIEW"
        elif score > 0.60 or anomaly:
            decision = "STEP_UP"
        else:
            decision = "APPROVE"
        decisions.append(decision)
        if is_attack:
            attacks += 1
            family_totals[attack_id] += 1
            if decision != "APPROVE":
                caught += 1
                per_family[attack_id] += 1
        elif decision != "APPROVE":
            fp += 1

    legit = len(rows) - attacks
    fpr = fp / max(1, legit)
    tpr = caught / max(1, attacks)
    per_attack = {
        attack: round(per_family[attack] / max(1, total), 4)
        for attack, total in sorted(family_totals.items())
    }
    worst_family = min(per_attack.values(), default=0.0)
    return {
        "fpr": round(fpr, 6),
        "tpr": round(tpr, 6),
        "worst_family_tpr": round(worst_family, 6),
        "caught": caught,
        "attacks": attacks,
        "false_positives": fp,
        "legit": legit,
        "per_attack_tpr": per_attack,
        "decision_counts": dict(Counter(decisions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--train-legit", type=int, default=4500)
    parser.add_argument("--eval-legit", type=int, default=2500)
    args = parser.parse_args()

    train = build_corpus(args.train_legit, TRAIN_COUNTS, seed=123)
    evaluation = build_corpus(args.eval_legit, EVAL_COUNTS, seed=777)
    train_rows = train["rows"]
    eval_rows = evaluation["rows"]

    x_train_base = base_matrix(train_rows)
    x_eval_base = base_matrix(eval_rows)
    g_train, _ = graph_context(train_rows)
    g_eval, eval_rings = graph_context(eval_rows)
    x_train_graph = np.hstack([x_train_base, g_train])
    x_eval_graph = np.hstack([x_eval_base, g_eval])
    y_train = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int8)
    legit_mask = y_train == 0

    results: dict = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "feature_sets": {
            "base": list(FEATURE_NAMES),
            "base_plus_graph": list(FEATURE_NAMES) + GRAPH_NAMES,
        },
        "methodology": (
            "Fresh seed evaluation across attacks 1-8. Final decisions preserve the existing "
            "ring-first ladder and 0.85/0.60/0.30 score thresholds. Throughput measures "
            "batched model inference only on 200k rows."
        ),
        "deep_model_note": (
            "CNN/Transformer/TGN/GNN candidates require raw sequence/edge tensors and a deep-learning "
            "runtime not present in the production contract. They are not assigned invented scores; "
            "the base_plus_graph sweep tests validated structural context first."
        ),
        "candidates": [],
    }

    for feature_set, x_train, x_eval in (
        ("base", x_train_base, x_eval_base),
        ("base_plus_graph", x_train_graph, x_eval_graph),
    ):
        supervised = supervised_models(seed=42)
        novelty = novelty_models(seed=42)

        supervised_results: dict[str, tuple[np.ndarray, float]] = {}
        for name, model in supervised.items():
            started = time.perf_counter()
            model.fit(x_train, y_train)
            train_seconds = time.perf_counter() - started
            scores = predict_probability(model, x_eval)
            throughput = benchmark_call(lambda batch, m=model: predict_probability(m, batch), x_eval)
            supervised_results[name] = (scores, throughput)
            results["candidates"].append({
                "kind": "supervised_only",
                "feature_set": feature_set,
                "supervised": name,
                "train_seconds": round(train_seconds, 6),
                "model_inference_rows_per_second": round(throughput, 2),
            })

        novelty_results: dict[str, tuple[np.ndarray, float]] = {}
        for name, detector in novelty.items():
            started = time.perf_counter()
            detector.fit(x_train[legit_mask])
            train_seconds = time.perf_counter() - started
            anomalies, _ = detector.score(x_eval)
            target = 50_000 if name == "one_class_svm" else 200_000
            throughput = benchmark_call(lambda batch, d=detector: d.score(batch), x_eval, target_rows=target)
            novelty_results[name] = (anomalies, throughput)
            results["candidates"].append({
                "kind": "novelty_only",
                "feature_set": feature_set,
                "novelty": name,
                "train_seconds": round(train_seconds, 6),
                "model_inference_rows_per_second": round(throughput, 2),
            })

        for supervised_name, (scores, supervised_tps) in supervised_results.items():
            for novelty_name, (anomalies, novelty_tps) in novelty_results.items():
                metrics = final_metrics(scores, anomalies, eval_rings, eval_rows)
                combined_tps = 1.0 / (1.0 / supervised_tps + 1.0 / novelty_tps)
                # Quality dominates. Speed differentiates candidates that satisfy
                # the same FPR/TPR envelope; a failed FPR budget is heavily penalized.
                quality = (
                    0.55 * metrics["tpr"]
                    + 0.25 * metrics["worst_family_tpr"]
                    + 0.20 * max(0.0, 1.0 - metrics["fpr"] / 0.05)
                )
                fpr_penalty = 0.35 if metrics["fpr"] > 0.05 else 0.0
                speed_bonus = min(math.log10(max(combined_tps, 1.0)) / 8.0, 0.15)
                rank_score = quality - fpr_penalty + speed_bonus
                results["candidates"].append({
                    "kind": "combined",
                    "feature_set": feature_set,
                    "supervised": supervised_name,
                    "novelty": novelty_name,
                    "combined_model_rows_per_second": round(combined_tps, 2),
                    "rank_score": round(rank_score, 6),
                    **metrics,
                })

    combined = [row for row in results["candidates"] if row["kind"] == "combined"]
    combined.sort(
        key=lambda row: (
            row["fpr"] <= 0.05,
            row["rank_score"],
            row["tpr"],
            row["combined_model_rows_per_second"],
        ),
        reverse=True,
    )
    results["ranking"] = combined
    results["winner"] = combined[0]

    encoded = json.dumps(results, indent=2, sort_keys=True)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
