"""Decision engine for the Adversarial Payment Arena.

The engine fuses supervised behavioral scoring, novelty detection and entity-
graph evidence. The decision ladder is defined once in ``apply_ladder`` and is
used by both production inference and calibration so the measured operating
point cannot drift from the deployed one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from defense.graph import EntityGraph
from defense.novelty import NoveltyDetector
from defense.realtime import VelocityScorer
from schemas.payment import PaymentMessage

APPROVE, STEP_UP, DECLINE, MANUAL_REVIEW = (
    "APPROVE",
    "STEP_UP",
    "DECLINE",
    "MANUAL_REVIEW",
)

# Safe defaults for a fresh clone. ``calibrate`` replaces these per engine.
DECLINE_VELOCITY = 0.35
STEPUP_VELOCITY = 0.02
MANUAL_VELOCITY = 0.01
FP_FRICTION_BPS = 15.0


def apply_ladder(
    v_score: float,
    is_anomaly: bool,
    ring_risk: float,
    *,
    t_decline: float,
    t_stepup: float,
    t_manual: float,
    t_ring: float,
    novelty_alone: bool,
) -> tuple[str, list[str]]:
    """Pure decision policy shared by inference and calibration."""
    if ring_risk > t_ring:
        return DECLINE, ["ring_detected"]
    if v_score > t_decline:
        return DECLINE, [f"velocity>{round(t_decline, 4)}"]
    if is_anomaly and v_score <= t_manual:
        return MANUAL_REVIEW, ["novelty_anomaly+low_velocity"]
    if v_score > t_stepup:
        return STEP_UP, [f"velocity>{round(t_stepup, 4)}"]
    if is_anomaly and novelty_alone:
        return STEP_UP, ["novelty_anomaly"]
    return APPROVE, []


@dataclass(slots=True)
class PreparedDecision:
    """Stateful pre-model work captured for later batched model inference."""

    wire: dict
    features: dict
    ring: dict
    graph_new_edges: list[tuple[str, str]]


class DecisionEngine:
    """Own the three defense layers and their shared state."""

    def __init__(
        self,
        environment=None,
        scorer: VelocityScorer | None = None,
        novelty: NoveltyDetector | None = None,
        graph: EntityGraph | None = None,
    ) -> None:
        self.env = environment
        self.scorer = scorer or VelocityScorer()
        if environment is not None:
            self.scorer.extractor.env = environment
        self.novelty = novelty or NoveltyDetector()
        self.graph = graph or EntityGraph()

        self.decline_threshold = DECLINE_VELOCITY
        self.stepup_threshold = STEPUP_VELOCITY
        self.manual_threshold = MANUAL_VELOCITY
        self.ring_risk_threshold = 0.0
        self.novelty_alone_alerts = True

    def train(self, rows: list[dict]) -> dict:
        return {
            "xgb": self.scorer.train(rows, save_path=None),
            "iforest": self.novelty.train(rows, save_path=None),
        }

    @staticmethod
    def _coerce(payload) -> PaymentMessage:
        if isinstance(payload, PaymentMessage):
            return payload
        return PaymentMessage.model_validate(payload)

    def calibrate(
        self,
        rows: Iterable[dict],
        target_fpr: float = 0.01,
        env=None,
    ) -> dict:
        """Calibrate the *full decision stack* on a disjoint validation split.

        Stored corpus features are preferred because they were extracted against
        the environment that generated the row. When an explicit matching
        ``env`` is supplied, features may be re-extracted on a scratch scorer.
        Calibration never mutates the live graph or rolling feature state.
        """
        rows = list(rows)
        legit = [row for row in rows if int(row.get("label", 0)) == 0]
        if len(legit) < 100:
            return {
                "calibrated": False,
                "reason": f"need >=100 legitimate rows, got {len(legit)}",
            }

        ordered = sorted(rows, key=lambda row: row["payload"]["timestamp"])
        reextract = env is not None or any("features" not in row for row in ordered)

        scratch_graph = EntityGraph()
        scratch_scorer = self.scorer
        if reextract:
            scratch_scorer = copy.deepcopy(self.scorer)
            scratch_scorer.extractor.env = env if env is not None else self.env

        captured: list[tuple[float, bool, float, int]] = []
        for row in ordered:
            msg = self._coerce(row["payload"])
            wire = msg.to_wire()
            if reextract:
                source_env = env if env is not None else self.env
                history = (
                    source_env.get_customer_history(msg.customer_id)
                    if source_env is not None
                    else None
                )
                feats = scratch_scorer.features(wire, history)
                scratch_scorer.observe(wire)
            else:
                feats = row["features"]

            velocity = self.scorer.score_from_features(feats)
            novelty = self.novelty.detect(wire, feats)
            ring = scratch_graph.check(wire)
            captured.append(
                (
                    float(velocity),
                    bool(novelty["is_anomaly"]),
                    float(ring["risk_score"]) if ring["ring_detected"] else 0.0,
                    int(row.get("label", 0)),
                )
            )
            scratch_graph.observe(wire)

        legit_scores = np.asarray(
            [score for score, _, _, label in captured if label == 0], dtype=float
        )
        decline_floor = float(
            np.quantile(legit_scores, min(0.9999, 1.0 - target_fpr / 8.0))
        )
        manual_ref = float(np.quantile(legit_scores, 0.50))
        candidates = sorted(
            {
                float(np.quantile(legit_scores, q))
                for q in np.linspace(0.50, 0.99995, 600)
            }
        )
        ring_candidates = [0.0] + sorted(
            {
                round(float(risk), 3)
                for _, _, risk, _ in captured
                if risk > 0.0
            }
        ) + [1.0]

        def measured(
            tau: float,
            novelty_alone: bool,
            ring_tau: float,
        ) -> tuple[float, float]:
            t_decline = max(tau, decline_floor)
            t_manual = min(tau, manual_ref)
            fp = tn = tp = fn = 0
            for score, anomaly, ring_risk, label in captured:
                decision, _ = apply_ladder(
                    score,
                    anomaly,
                    ring_risk,
                    t_decline=t_decline,
                    t_stepup=tau,
                    t_manual=t_manual,
                    t_ring=ring_tau,
                    novelty_alone=novelty_alone,
                )
                alert = decision != APPROVE
                if label == 0:
                    fp += int(alert)
                    tn += int(not alert)
                else:
                    tp += int(alert)
                    fn += int(not alert)

            n_legit = max(fp + tn, 1)
            point = fp / n_legit
            se = (point * (1.0 - point) / n_legit) ** 0.5
            upper95 = point + 1.64 * se
            recall = tp / max(tp + fn, 1)
            return upper95, recall

        best: tuple[float, float, float, bool, float] | None = None
        for novelty_alone in (False, True):
            for ring_tau in ring_candidates:
                for tau in candidates:
                    fpr_upper, recall = measured(tau, novelty_alone, ring_tau)
                    if fpr_upper <= target_fpr and (
                        best is None or recall > best[2]
                    ):
                        best = (tau, fpr_upper, recall, novelty_alone, ring_tau)

        if best is None:
            tau = candidates[-1]
            fpr_upper, recall = measured(tau, False, 1.0)
            best = (tau, fpr_upper, recall, False, 1.0)

        tau, fpr_upper, recall, novelty_alone, ring_tau = best
        self.stepup_threshold = tau
        self.decline_threshold = max(tau, decline_floor)
        self.manual_threshold = min(tau, manual_ref)
        self.ring_risk_threshold = ring_tau
        self.novelty_alone_alerts = novelty_alone

        return {
            "calibrated": True,
            "method": "full_stack_threshold_search",
            "feature_source": "reextracted" if reextract else "corpus_stored",
            "novelty_alone_alerts": novelty_alone,
            "ring_risk_threshold": round(ring_tau, 4),
            "target_fpr": target_fpr,
            "achieved_validation_fpr_upper95": round(fpr_upper, 6),
            "achieved_validation_recall": round(recall, 6),
            "stepup_threshold": round(self.stepup_threshold, 6),
            "decline_threshold": round(self.decline_threshold, 6),
            "manual_threshold": round(self.manual_threshold, 6),
            "n_legit_validation": len(legit),
            "n_rows_validation": len(captured),
            "budget_reachable": fpr_upper <= target_fpr,
        }

    def _record_from_scores(
        self,
        wire: dict,
        feats: dict,
        v_score: float,
        nov: dict,
        ring: dict,
        graph_new_edges: list[tuple[str, str]],
    ) -> dict:
        ring_risk_for_ladder = (
            float(ring["risk_score"]) if ring["ring_detected"] else 0.0
        )
        decision, reasons = apply_ladder(
            float(v_score),
            bool(nov["is_anomaly"]),
            ring_risk_for_ladder,
            t_decline=self.decline_threshold,
            t_stepup=self.stepup_threshold,
            t_manual=self.manual_threshold,
            t_ring=self.ring_risk_threshold,
            novelty_alone=self.novelty_alone_alerts,
        )
        if reasons == ["ring_detected"]:
            reasons = [f"ring_detected:{ring['ring_id']}"]

        shared = ring.get("shared_infra", [])
        graph_component_customers = int(ring.get("component_customers", 1))
        graph_shared_infra_count = len(shared)
        graph_max_linked_customers = max(
            (int(item.get("linked_customers", 0)) for item in shared),
            default=0,
        )

        return {
            "decision": decision,
            "reasons": reasons,
            "scores": {
                "velocity": round(float(v_score), 4),
                "novelty_anomaly": nov["anomaly_score"],
                "is_anomaly": nov["is_anomaly"],
                "ring_risk": ring["risk_score"],
                "ring_detected": ring["ring_detected"],
                "ring_id": ring["ring_id"],
                "graph_component_customers": graph_component_customers,
                "graph_shared_infra_count": graph_shared_infra_count,
                "graph_max_linked_customers": graph_max_linked_customers,
            },
            "features": feats,
            "payload": wire,
            "amount": float(wire["amount"]),
            "label_hint": wire.get("stolen_resource") is not None,
            "graph_new_edges": graph_new_edges,
        }

    def decide(self, payload: PaymentMessage | dict) -> dict:
        msg = self._coerce(payload)
        wire = msg.to_wire()
        history = (
            self.env.get_customer_history(msg.customer_id) if self.env else None
        )

        feats = self.scorer.features(wire, history)
        velocity = self.scorer.score_from_features(feats)
        novelty = self.novelty.detect(wire, feats)
        ring = self.graph.check(wire)

        self.scorer.observe(wire)
        graph_new_edges = self.graph.observe(wire)
        return self._record_from_scores(
            wire, feats, velocity, novelty, ring, graph_new_edges
        )

    def prepare_for_batch(self, payload: PaymentMessage | dict) -> PreparedDecision:
        """Capture stateful analysis for one row and then advance state."""
        msg = self._coerce(payload)
        wire = msg.to_wire()
        history = (
            self.env.get_customer_history(msg.customer_id) if self.env else None
        )
        ring = self.graph.check(wire)
        feats = self.scorer.features_and_observe(wire, history)
        graph_new_edges = self.graph.observe(wire)
        return PreparedDecision(wire, feats, ring, graph_new_edges)

    def finalize_batch(self, prepared: list[PreparedDecision]) -> list[dict]:
        if not prepared:
            return []
        feature_rows = [item.features for item in prepared]
        wires = [item.wire for item in prepared]
        velocity_scores = self.scorer.score_many_from_features(feature_rows)
        novelty_rows = self.novelty.detect_many(wires, feature_rows)
        return [
            self._record_from_scores(
                item.wire,
                item.features,
                velocity,
                novelty,
                item.ring,
                item.graph_new_edges,
            )
            for item, velocity, novelty in zip(
                prepared, velocity_scores, novelty_rows
            )
        ]

    def decide_batch(
        self, payloads: Iterable[PaymentMessage | dict]
    ) -> list[dict]:
        prepared = [self.prepare_for_batch(payload) for payload in payloads]
        return self.finalize_batch(prepared)

    @staticmethod
    def _new_cost_totals() -> dict:
        return {
            "legit_volume": 0.0,
            "fp_count": 0,
            "fp_cost_usd": 0.0,
            "fn_count": 0,
            "fn_loss_usd": 0.0,
            "tp_count": 0,
            "tp_saved_usd": 0.0,
        }

    def apply_to_running_totals(self, totals: dict, record: dict, truth: str) -> None:
        amount = float(record["amount"])
        decision = record["decision"]
        if truth == "legit":
            totals["legit_volume"] += amount
            if decision != APPROVE:
                totals["fp_count"] += 1
                totals["fp_cost_usd"] += amount * FP_FRICTION_BPS / 10_000.0
        else:
            if decision == APPROVE:
                totals["fn_count"] += 1
                totals["fn_loss_usd"] += amount
            elif decision == DECLINE:
                totals["tp_count"] += 1
                totals["tp_saved_usd"] += amount

    @classmethod
    def summarize_totals(cls, totals: dict) -> dict:
        fp_usd = round(totals["fp_cost_usd"], 2)
        fn_loss = round(totals["fn_loss_usd"], 2)
        tp_saved = round(totals["tp_saved_usd"], 2)
        return {
            "fp_cost_bps": round(
                fp_usd / max(totals["legit_volume"], 1e-9) * 10_000, 2
            ),
            "fp_cost_usd": fp_usd,
            "fn_loss": fn_loss,
            "tp_saved": tp_saved,
            "net_savings": round(tp_saved - fn_loss - fp_usd, 2),
            "counts": {
                "false_positives": totals["fp_count"],
                "false_negatives": totals["fn_count"],
                "true_positives_declined": totals["tp_count"],
            },
        }

    def compute_cost_matrix(
        self,
        decisions: Iterable[dict],
        ground_truth: Iterable[str],
    ) -> dict:
        totals = self._new_cost_totals()
        for record, truth in zip(decisions, ground_truth):
            self.apply_to_running_totals(totals, record, truth)
        return self.summarize_totals(totals)


__all__ = [
    "DecisionEngine",
    "PreparedDecision",
    "apply_ladder",
    "APPROVE",
    "STEP_UP",
    "DECLINE",
    "MANUAL_REVIEW",
    "FP_FRICTION_BPS",
]
