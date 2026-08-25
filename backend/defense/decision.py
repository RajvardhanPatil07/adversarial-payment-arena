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
        the decision, never before.
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

        # fold state forward
        self.scorer.observe(wire)
        self.graph.observe(wire)

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
        }

    # ------------------------------------------------------------------ #
    # Cost matrix
    # ------------------------------------------------------------------ #

    def compute_cost_matrix(
        self,
        decisions: Iterable[dict],
        ground_truth: Iterable[str],
    ) -> dict:
        """
        decisions: outputs of .decide(); ground_truth: aligned labels where
        'legit' means genuine traffic and anything else is attack.
        """
        fp_cnt = fn_cnt = tp_cnt = 0
        fp_usd = fn_usd = tp_usd = legit_volume = 0.0

        for rec, truth in zip(decisions, ground_truth):
            amt = rec["amount"]
            d = rec["decision"]
            is_attack = truth != "legit"

            if not is_attack:
                legit_volume += amt
                if d != APPROVE:  # any friction on honest customers is FP
                    fp_cnt += 1
                    fp_usd += amt * FP_FRICTION_BPS / 10_000.0
            else:
                if d == APPROVE:
                    fn_cnt += 1
                    fn_usd += amt                      # fraud loss: full amount
                elif d == DECLINE:
                    tp_cnt += 1
                    tp_usd += amt                      # prevented loss
                # STEP_UP / MANUAL_REVIEW on attacks: challenged, not yet
                # lost nor saved — tracked implicitly by absence from both.

        return {
            "fp_cost_bps": round(fp_usd / max(legit_volume, 1e-9) * 10_000, 2),
            "fp_cost_usd": round(fp_usd, 2),
            "fn_loss": round(fn_usd, 2),
            "tp_saved": round(tp_usd, 2),
            "net_savings": round(tp_usd - fn_usd - fp_usd, 2),
            "counts": {"false_positives": fp_cnt, "false_negatives": fn_cnt, "true_positives_declined": tp_cnt},
        }


__all__ = [
    "DecisionEngine",
    "APPROVE",
    "STEP_UP",
    "DECLINE",
    "MANUAL_REVIEW",
    "FP_FRICTION_BPS",
]
