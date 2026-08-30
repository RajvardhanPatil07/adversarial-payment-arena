"""
Decision engine — fuses the three defense layers into issuer decisions.

Ladder (documented deviation from the literal spec, see below). Every
threshold named here is CALIBRATED by DecisionEngine.calibrate() on a
validation split, never hand-set:
    ring detected AND ring_risk > t_ring  -> DECLINE
    velocity > t_decline                  -> DECLINE
    anomaly AND velocity <= t_manual      -> MANUAL_REVIEW
    velocity > t_stepup                   -> STEP_UP (trigger 3DS)
    anomaly AND novelty_alone_alerts      -> STEP_UP
    otherwise                             -> APPROVE

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

# Operating thresholds.
#
# These were hand-set constants (0.85 / 0.60 / 0.30) chosen when the scorer used
# eleven point-in-time features and the corpus contained no benign anomalies.
# Both of those changed: the feature set is now eighteen-dimensional and the
# legitimate class deliberately contains fraud-shaped-but-genuine traffic, which
# moves the whole score distribution. Hand-set constants silently became
# mis-calibrated -- recall collapsed while the numbers on the page stayed the
# same, which is precisely the failure mode this repository argues against.
#
# They are kept as DEFAULTS (a fresh clone with the shipped model needs some
# operating point) but `DecisionEngine.calibrate()` overrides them from a
# legitimate validation split, pinning STEP_UP at a target FPR. That is how a
# threshold should be chosen: measured on held-out legitimate traffic, never
# guessed, and never fitted on the evaluation split.
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
    """The decision ladder as ONE pure function. Returns (decision, reasons).

    This exists because the ladder was previously written twice: once in
    `decide()` for production and once inside `calibrate()`'s search loop to
    predict the false-positive rate. The two drifted, and the drift was
    invisible -- calibration modelled `alert = ring or v > tau or anomaly`, but
    production also alerted via the MANUAL_REVIEW branch (anomaly AND low
    velocity), an alert source the search did not know existed. So calibration
    optimised a policy that was never the one deployed, reported a validation
    FPR of 0.95%, and the same thresholds measured 4.45% on held-out traffic.

    A calibration routine that simulates a different ladder than the one that
    runs is not calibration. Both callers now go through here, so the predicted
    operating point and the delivered operating point cannot disagree.

    `ring_risk` is 0.0 when the graph layer reports no ring, so the default
    `t_ring=0.0` with a strict `>` reproduces decline-on-any-ring.
    """
    reasons: list[str] = []
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
    return APPROVE, reasons


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

    def calibrate(
        self,
        rows: Iterable[dict],
        target_fpr: float = 0.01,
        env=None,
    ) -> dict:
        """Pin the operating thresholds on a validation split, FULL-STACK.

        Four design decisions, each of which the naive version gets wrong in a
        way that is easy to miss and expensive to believe:

        1. It reads the features the CORPUS already computed instead of
           recomputing them. This is the subtle one, and it silently invalidated
           every number this method produced before the fix. `build_corpus`
           returns `{"rows", "env"}`: each row's `features` were extracted by a
           FeatureExtractor bound to THAT corpus's PaymentEnvironment, which owns
           the device-binding registry and customer profiles. Recomputing them
           here resolved every lookup against `self.env` -- the TRAINING
           environment, built from a different seed -- so no device in the
           calibration split was ever on file. Measured: 728 of 820 rows
           disagreed on `device_known`, mean 0.000 recomputed against 0.888
           stored. Every honest customer looked like they had just appeared on an
           unrecognised handset, the legitimate score distribution collapsed onto
           the fraud side of the cliff, and tau was fitted to that wreckage. The
           validation FPR still printed 1.1% because the corruption was
           self-consistent WITHIN the split; held-out FPR came back 6.6%. A
           calibration that reports success on data it has silently mangled is
           worse than no calibration, and it is the exact failure this
           repository exists to argue against. Pass `env=` to force a live
           re-extraction when the caller genuinely has the matching environment.

        2. It calibrates the WHOLE STACK, not the XGBoost score. A quantile of
           the supervised score ignores that the graph and novelty layers also
           raise alerts, so the achieved false-positive rate lands above target
           -- the model is calibrated and the *system* is not. The full ladder is
           replayed per candidate and the measured non-APPROVE rate on
           legitimate rows is what gets controlled.

        3. It SEARCHES rather than inverting a quantile. The score distribution
           of a class-weighted booster is sharply bimodal, so adjacent quantiles
           sit on opposite sides of a cliff. A search picks the largest
           achievable operating point inside budget and reports what it hit.

        4. It runs on a SCRATCH stack, not on `self`. The replay is stateful:
           `scorer.observe` and `graph.observe` fold every calibration row into
           the live engine's history. Calibrating used to leave the engine
           carrying 1,656 phantom transactions -- validation traffic
           masquerading as production history at inference time. The scratch
           copies below keep calibration read-only with respect to the engine.

        `rows` should be a MIXED stream with the same composition as production
        traffic, drawn from a seed disjoint from both training and evaluation.
        Calibrating on the rows the numbers are reported on is leakage.
        """
        import copy

        import numpy as np

        rows = list(rows)
        legit = [r for r in rows if int(r.get("label", 0)) == 0]
        if len(legit) < 100:
            return {
                "calibrated": False,
                "reason": f"need >=100 legitimate rows, got {len(legit)}",
            }

        ordered = sorted(rows, key=lambda x: x["payload"]["timestamp"])

        # Decide the feature source ONCE, explicitly, and report which was used.
        # Trusting stored features is the default because the corpus extracted
        # them against the environment that actually generated the traffic.
        reextract = env is not None or any("features" not in r for r in ordered)

        # ---- scratch stack: calibration must not mutate the live engine ---- #
        scratch_graph = EntityGraph()
        scratch_scorer = self.scorer
        if reextract:
            scratch_scorer = copy.deepcopy(self.scorer)
            scratch_scorer.extractor.env = env if env is not None else self.env

        captured: list[tuple[float, bool, bool, int]] = []
        for r in ordered:
            msg = self._coerce(r["payload"])
            wire = msg.to_wire()
            if reextract:
                src = env if env is not None else self.env
                history = src.get_customer_history(msg.customer_id) if src else None
                feats = scratch_scorer.features(wire, history)
                scratch_scorer.observe(wire)
            else:
                feats = r["features"]
            v = self.scorer.score_from_features(feats)
            nov = self.novelty.detect(wire, feats)
            ring = scratch_graph.check(wire)
            captured.append((
                float(v), bool(nov["is_anomaly"]),
                float(ring["risk_score"]) if ring["ring_detected"] else 0.0,
                int(r.get("label", 0)),
            ))
            scratch_graph.observe(wire)

        def measured(tau: float, novelty_alone: bool, ring_tau: float) -> tuple[float, float]:
            """(FPR, recall) on the calibration split for one candidate policy.

            `novelty_alone` toggles whether an Isolation Forest hit may raise an
            alert BY ITSELF, or only when the supervised score corroborates it.
            An unsupervised layer trained on legitimate traffic fires on any
            unusual-but-genuine behaviour -- which is precisely what the
            benign-anomaly rows are -- so making this a calibrated choice rather
            than a hardcoded rule is the honest way to resolve it.

            `ring_tau` is the graph layer's own admission threshold. The ladder
            used to DECLINE on `ring_detected` unconditionally, which put a hard
            floor under the achievable false-positive rate that no supervised
            threshold could lower: the shared_family_device and flash_sale_crowd
            benign anomalies are, structurally, three-plus cardholders behind one
            device or IP -- the same topology as a mule ring. With the branch
            fixed at `always`, a 1% budget was simply unreachable and calibration
            could only report failure. Thresholding on the ring's own risk score
            lets the search trade graph sensitivity against the budget.
            """
            # Derived exactly as they will be at deployment, from the same tau.
            t_decline = max(tau, decline_floor)
            t_manual = min(tau, manual_ref)
            fp = tn = tp = fn = 0
            for v, anom, ring_risk, label in captured:
                # THE production ladder, not a re-implementation of it.
                decision, _ = apply_ladder(
                    v, anom, ring_risk,
                    t_decline=t_decline, t_stepup=tau, t_manual=t_manual,
                    t_ring=ring_tau, novelty_alone=novelty_alone,
                )
                alert = decision != APPROVE
                if label == 0:
                    fp += alert
                    tn += not alert
                else:
                    tp += alert
                    fn += not alert
            n_legit_eval = max(fp + tn, 1)
            point = fp / n_legit_eval
            # UPPER CONFIDENCE BOUND, not the point estimate, is what gets
            # compared against the budget. tau lives in the far tail: at a 1%
            # target on 2,000 legitimate rows, the decision is being made on the
            # strength of ~20 observations, so the point estimate carries a
            # standard error of roughly 0.2pp and the search will happily pick
            # whichever candidate got lucky. Selecting the argmax of a noisy
            # estimate is optimisation against sampling noise, and it shows up as
            # the calibrated FPR landing consistently above target out of sample.
            #
            # Wilson-style one-sided bound: penalise each candidate by the
            # uncertainty in its own FPR estimate, so a threshold only wins if it
            # is inside budget with margin rather than by luck.
            se = (point * (1.0 - point) / n_legit_eval) ** 0.5
            upper = point + 1.64 * se  # ~95% one-sided
            return (upper, tp / max(tp + fn, 1))

        # ---- search: largest recall whose measured FPR stays in budget ----- #
        legit_scores = np.array([v for v, _, _, lab in captured if lab == 0])
        # Derived thresholds are pinned BEFORE the search so `measured()` scores
        # candidates under exactly the policy that will later be installed.
        # A decline costs strictly more than a step-up, so it must be strictly
        # rarer: placed deeper into the same distribution.
        decline_floor = float(
            np.quantile(legit_scores, min(0.9999, 1.0 - target_fpr / 8.0))
        )
        # MANUAL_REVIEW is the low-velocity + novelty branch, so its threshold
        # must sit BELOW the step-up line: it catches rows the supervised model
        # is quiet about but the novelty layer flags. A median (as before) put it
        # above tau, which made the branch unreachable -- dead code with a test
        # asserting it worked.
        manual_ref = float(np.quantile(legit_scores, 0.50))
        candidates = sorted({
            float(np.quantile(legit_scores, q))
            for q in np.linspace(0.50, 0.99995, 600)
        })
        # Ring candidates span "trust the graph completely" (0.0, the old
        # unconditional behaviour) through to "never alert on the graph alone"
        # (1.0), so the previous policy stays inside the search space and is
        # chosen when it is genuinely optimal rather than assumed.
        ring_risks = sorted({
            round(float(r), 3) for _, _, r, _ in captured if r > 0.0
        })
        ring_candidates = [0.0] + ring_risks + [1.0]

        best = None
        for novelty_alone in (False, True):
            for ring_tau in ring_candidates:
                for tau in candidates:
                    fpr, rec = measured(tau, novelty_alone, ring_tau)
                    if fpr <= target_fpr and (best is None or rec > best[2]):
                        best = (tau, fpr, rec, novelty_alone, ring_tau)

        if best is None:
            # Budget unreachable even at the most conservative policy: report the
            # floor honestly rather than silently exceeding the budget.
            tau = candidates[-1]
            fpr, rec = measured(tau, False, 1.0)
            best = (tau, fpr, rec, False, 1.0)

        tau, achieved_fpr_upper, achieved_recall, novelty_alone, ring_tau = best
        self.stepup_threshold = tau
        self.novelty_alone_alerts = novelty_alone
        self.ring_risk_threshold = ring_tau
        self.decline_threshold = max(tau, decline_floor)
        self.manual_threshold = min(tau, manual_ref)

        return {
            "calibrated": True,
            "method": "full_stack_threshold_search",
            "feature_source": "reextracted" if reextract else "corpus_stored",
            "novelty_alone_alerts": novelty_alone,
            "ring_risk_threshold": round(ring_tau, 4),
            "target_fpr": target_fpr,
            # Reported as the 95% upper bound it actually is, so the number on
            # the page is the pessimistic one rather than the flattering one.
            "achieved_validation_fpr_upper95": round(achieved_fpr_upper, 6),
            "achieved_validation_recall": round(achieved_recall, 6),
            "stepup_threshold": round(tau, 6),
            "decline_threshold": round(self.decline_threshold, 6),
            "manual_threshold": round(self.manual_threshold, 6),
            "n_legit_validation": len(legit),
            "n_rows_validation": len(captured),
            "budget_reachable": achieved_fpr_upper <= target_fpr,
        }

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

        # Instance thresholds, so a calibrated engine overrides the defaults
        # without mutating module state for every other engine in the process.
        t_decline = getattr(self, "decline_threshold", DECLINE_VELOCITY)
        t_stepup = getattr(self, "stepup_threshold", STEPUP_VELOCITY)
        t_manual = getattr(self, "manual_threshold", MANUAL_VELOCITY)

        # The graph layer's admission threshold is calibrated, not assumed. An
        # unconditional decline-on-ring puts a floor under the false-positive
        # rate that no other threshold can lower, because a shared family tablet
        # and a mule ring have the same topology: several cardholders behind one
        # device. Default 0.0 preserves the original decline-on-any-ring
        # behaviour for an uncalibrated engine.
        t_ring = getattr(self, "ring_risk_threshold", 0.0)

        # ONE ladder, shared with calibrate(). See apply_ladder(): duplicating
        # it is how the calibrated operating point silently stopped matching the
        # deployed one.
        ring_risk = float(ring["risk_score"]) if ring["ring_detected"] else 0.0
        decision, reasons = apply_ladder(
            v_score, bool(nov["is_anomaly"]), ring_risk,
            t_decline=t_decline, t_stepup=t_stepup, t_manual=t_manual,
            t_ring=t_ring,
            novelty_alone=getattr(self, "novelty_alone_alerts", True),
        )
        if reasons == ["ring_detected"]:
            reasons = [f"ring_detected:{ring['ring_id']}"]

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
