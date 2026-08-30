"""
Decision engine — fuses the three defense layers into issuer decisions.

Ladder (documented deviation from the literal spec, see below):
    ring detected                -> DECLINE
    velocity > 0.85              -> DECLINE
    anomaly AND velocity <= 0.3  -> MANUAL_REVIEW
    velocity > 0.6 OR anomaly    -> STEP_UP (trigger 3DS)
    otherwise                    -> APPROVE

DEVIATION NOTE: the spec's ladder listed MANUAL_REVIEW *after*
`velocity > 0.6 OR novelty.is_anomaly`, which makes it unreachable dead
code (any anomaly already routes to STEP_UP). We evaluate the manual-review
condition first so the queue actually exists — identical outcomes for every
input the original ladder could produce, minus the shadowed branch.

Cost matrix (bps = basis points of transaction amount):
    FP: legit flagged (STEP_UP/DECLINE/MANUAL_REVIEW) -> 15 bps friction
    FN: attack approved                               -> 100% of amount
    TP: attack declined                               -> saves the amount
"""

from __future__ import annotations

from typing import Iterable

from defense.graph import EntityGraph
from defense.novelty import NoveltyDetector
from defense.realtime import VelocityScorer
from schemas.payment import PaymentMessage

APPROVE, STEP_UP, DECLINE, MANUAL_REVIEW = "APPROVE", "STEP_UP", "DECLINE", "MANUAL_REVIEW"

DECLINE_VELOCITY = 0.85
STEPUP_VELOCITY = 0.60
MANUAL_VELOCITY = 0.30

FP_FRICTION_BPS = 15.0


class DecisionEngine:
    """Owns the three layers + shared feature state; emits final decisions."""

    def __init__(
        self,
        environment=None,
        scorer: VelocityScorer | None = None,
        novelty: NoveltyDetector | None = None,
        graph: EntityGraph | None = None,
    ) -> None:
        self.env = environment
        # One extractor instance shared by scorer layers keeps state coherent;
        # a custom scorer brings its own only if the caller wires it so.
        self.scorer = scorer or VelocityScorer()
        # CRITICAL WIRING: the scorer's extractor needs issuer state for
        # device-binding lookups. Without this, device_known silently reads 0
        # for EVERYONE at inference — train/infer skew that nukes the FPR.
        if environment is not None:
            self.scorer.extractor.env = environment
        self.novelty = novelty or NoveltyDetector()
        self.graph = graph or EntityGraph()

    # ------------------------------------------------------------------ #
    # Training entrypoint (delegates to layers over one labeled corpus)
    # ------------------------------------------------------------------ #

    def train(self, rows: list[dict]) -> dict:
        metrics = {
            "xgb": self.scorer.train(rows, save_path=None),
            "iforest": self.novelty.train(rows, save_path=None),
        }
        return metrics

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce(payload) -> PaymentMessage:
        if isinstance(payload, PaymentMessage):
            return payload
        return PaymentMessage.model_validate(payload)

    def decide(self, payload: PaymentMessage | dict) -> dict:
        """Score one payload and emit the issuer decision.

        Observe-after-score discipline: all layer state is folded in AFTER
        the decision, never before. Newly created graph edges are returned as
        a small delta so the live API never needs to rescan the entire graph.
        """
        msg = self._coerce(payload)
        wire = msg.to_wire()
        history = (
            self.env.get_customer_history(msg.customer_id) if self.env else None
        )

        feats = self.scorer.features(wire, history)
        v_score = self.scorer.score_from_features(feats)
        nov = self.novelty.detect(wire, feats)
        ring = self.graph.check(wire)

        reasons: list[str] = []
        if ring["ring_detected"]:
            decision = DECLINE
            reasons.append(f"ring_detected:{ring['ring_id']}")
        elif v_score > DECLINE_VELOCITY:
            decision = DECLINE
            reasons.append(f"velocity>{DECLINE_VELOCITY}")
        elif nov["is_anomaly"] and v_score <= MANUAL_VELOCITY:
            decision = MANUAL_REVIEW
            reasons.append("novelty_anomaly+low_velocity")
        elif v_score > STEPUP_VELOCITY:
            decision = STEP_UP
            reasons.append(f"velocity>{STEPUP_VELOCITY}")
        elif nov["is_anomaly"]:
            decision = STEP_UP
            reasons.append("novelty_anomaly")
        else:
            decision = APPROVE

        # Fold state forward. EntityGraph.observe() returns only newly-created
        # edges; repeats only bump edge weights and produce no graph delta.
        self.scorer.observe(wire)
        graph_new_edges = self.graph.observe(wire)

        return {
            "decision": decision,
            "reasons": reasons,
            "scores": {
                "velocity": round(v_score, 4),
                "novelty_anomaly": nov["anomaly_score"],
                "is_anomaly": nov["is_anomaly"],
                "ring_risk": ring["risk_score"],
                "ring_detected": ring["ring_detected"],
                "ring_id": ring["ring_id"],
            },
            "features": feats,
            "payload": wire,
            "amount": float(wire["amount"]),
            "label_hint": wire.get("stolen_resource") is not None,
            "graph_new_edges": graph_new_edges,
        }

    # ------------------------------------------------------------------ #
    # Cost matrix
    # ------------------------------------------------------------------ #

    @staticmethod
    def _new_cost_totals() -> dict:
        return {
            "legit_volume": 0.0,
            "fp_count": 0, "fp_cost_usd": 0.0,
            "fn_count": 0, "fn_loss_usd": 0.0,
            "tp_count": 0, "tp_saved_usd": 0.0,
        }

    def apply_to_running_totals(self, totals: dict, record: dict, truth: str) -> None:
        """Fold ONE decision into running counters (live cost tracking for
        the dashboard). truth: 'legit' | anything else counts as attack."""
        amt = float(record["amount"])
        d = record["decision"]
        if truth == "legit":
            totals["legit_volume"] += amt
            if d != APPROVE:  # any friction on honest customers is FP
                totals["fp_count"] += 1
                totals["fp_cost_usd"] += amt * FP_FRICTION_BPS / 10_000.0
        else:
            if d == APPROVE:
                totals["fn_count"] += 1
                totals["fn_loss_usd"] += amt          # fraud loss: full amount
            elif d == DECLINE:
                totals["tp_count"] += 1
                totals["tp_saved_usd"] += amt         # prevented loss
            # STEP_UP / MANUAL_REVIEW on attacks: challenged — neither lost
            # nor saved until the challenge resolves.

    @classmethod
    def summarize_totals(cls, totals: dict) -> dict:
        """Wire shape for /cost_update events and final reports."""
        fp_usd = round(totals["fp_cost_usd"], 2)
        fn_loss = round(totals["fn_loss_usd"], 2)
        tp_saved = round(totals["tp_saved_usd"], 2)
        return {
            "fp_cost_bps": round(fp_usd / max(totals["legit_volume"], 1e-9) * 10_000, 2),
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
        """
        decisions: outputs of .decide(); ground_truth: aligned labels where
        'legit' means genuine traffic and anything else is attack.
        """
        totals = self._new_cost_totals()
        for rec, truth in zip(decisions, ground_truth):
            self.apply_to_running_totals(totals, rec, truth)
        return self.summarize_totals(totals)


__all__ = [
    "DecisionEngine",
    "APPROVE",
    "STEP_UP",
    "DECLINE",
    "MANUAL_REVIEW",
    "FP_FRICTION_BPS",
]
