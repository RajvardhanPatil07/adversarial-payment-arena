"""Four-action payment policy economics.

The policy receives a fraud score plus an *observable/predicted* APP-candidate
flag. Ground-truth fraud subtype is used only for outcome accounting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence

import numpy as np

APPROVE = "APPROVE"
STEP_UP = "STEP_UP"
COOLING_OFF = "COOLING_OFF"
DECLINE = "DECLINE"
ACTIONS = (APPROVE, STEP_UP, COOLING_OFF, DECLINE)


@dataclass(frozen=True)
class ActionCostModel:
    avg_fraud_amount_inr: float = 18_500.0
    avg_app_scam_amount_inr: float = 42_000.0
    chargeback_admin_inr: float = 1_200.0
    avg_legit_amount_inr: float = 2_400.0
    interchange_margin: float = 0.012
    insult_support_cost_inr: float = 210.0
    insult_churn_probability: float = 0.019
    customer_lifetime_value_inr: float = 9_800.0
    step_up_cost_inr: float = 6.0
    step_up_abandon_probability: float = 0.07
    cooling_off_cost_inr: float = 38.0
    cooling_off_abandon_probability: float = 0.03
    manual_review_cost_inr: float = 95.0
    step_up_blocks_unauthorised: float = 0.82
    step_up_blocks_app_scam: float = 0.11
    cooling_off_blocks_unauthorised: float = 0.86
    cooling_off_blocks_app_scam: float = 0.58
    decline_blocks_app_scam: float = 0.58

    def insult_cost_per_decline(self) -> float:
        return (
            self.insult_support_cost_inr
            + self.avg_legit_amount_inr * self.interchange_margin
            + self.insult_churn_probability * self.customer_lifetime_value_inr
        )

    def friction_cost_on_legit(self, action: str) -> float:
        if action == APPROVE:
            return 0.0
        if action == STEP_UP:
            return self.step_up_cost_inr + self.step_up_abandon_probability * (
                self.avg_legit_amount_inr * self.interchange_margin + self.insult_support_cost_inr
            )
        if action == COOLING_OFF:
            return (
                self.cooling_off_cost_inr
                + self.manual_review_cost_inr
                + self.cooling_off_abandon_probability
                * (self.avg_legit_amount_inr * self.interchange_margin + self.insult_support_cost_inr)
            )
        if action == DECLINE:
            return self.insult_cost_per_decline()
        raise ValueError(f"unknown action {action!r}")

    def fraud_loss(self, action: str, is_app_truth: bool) -> float:
        amount = self.avg_app_scam_amount_inr if is_app_truth else self.avg_fraud_amount_inr
        gross = amount + self.chargeback_admin_inr
        if action == APPROVE:
            return gross
        if action == STEP_UP:
            blocked = (
                self.step_up_blocks_app_scam
                if is_app_truth
                else self.step_up_blocks_unauthorised
            )
            return gross * (1.0 - blocked) + self.step_up_cost_inr
        if action == COOLING_OFF:
            blocked = (
                self.cooling_off_blocks_app_scam
                if is_app_truth
                else self.cooling_off_blocks_unauthorised
            )
            return gross * (1.0 - blocked) + self.cooling_off_cost_inr + self.manual_review_cost_inr
        if action == DECLINE:
            if is_app_truth:
                return gross * (1.0 - self.decline_blocks_app_scam)
            return 0.0
        raise ValueError(f"unknown action {action!r}")


def choose_action(
    score: float,
    is_app_candidate: bool,
    t_step_up: float,
    t_cooling: float,
    t_decline: float,
    app_carve_out: bool = True,
) -> str:
    if score >= t_decline:
        return COOLING_OFF if (is_app_candidate and app_carve_out) else DECLINE
    if score >= t_cooling:
        return COOLING_OFF
    if score >= t_step_up:
        return STEP_UP
    return APPROVE


def evaluate_policy(
    scores: Sequence[float],
    labels: Sequence[int],
    app_candidates: Sequence[bool],
    app_truth: Sequence[bool],
    t_step_up: float,
    t_cooling: float,
    t_decline: float,
    model: Optional[ActionCostModel] = None,
    app_carve_out: bool = True,
) -> Dict[str, object]:
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).astype(int).ravel()
    candidates = np.asarray(app_candidates).astype(bool).ravel()
    truth = np.asarray(app_truth).astype(bool).ravel()
    if not (s.size == y.size == candidates.size == truth.size):
        raise ValueError("scores, labels, app_candidates and app_truth must be the same length")

    m = model or ActionCostModel()
    counts = {a: 0 for a in ACTIONS}
    legit_counts = {a: 0 for a in ACTIONS}
    fraud_cost = app_cost = non_app_cost = friction_cost = 0.0

    for i in range(s.size):
        action = choose_action(
            float(s[i]),
            bool(candidates[i]),
            t_step_up,
            t_cooling,
            t_decline,
            app_carve_out,
        )
        counts[action] += 1
        if y[i] == 1:
            loss = m.fraud_loss(action, bool(truth[i]))
            fraud_cost += loss
            if bool(truth[i]):
                app_cost += loss
            else:
                non_app_cost += loss
        else:
            legit_counts[action] += 1
            friction_cost += m.friction_cost_on_legit(action)

    n_legit = max(1, int((y == 0).sum()))
    total = fraud_cost + friction_cost
    return {
        "thresholds": {
            "step_up": round(float(t_step_up), 6),
            "cooling_off": round(float(t_cooling), 6),
            "decline": round(float(t_decline), 6),
        },
        "app_carve_out": bool(app_carve_out),
        "action_counts": counts,
        "legit_action_counts": legit_counts,
        "legit_friction_rate": round(float((n_legit - legit_counts[APPROVE]) / n_legit), 6),
        "legit_hard_decline_rate": round(float(legit_counts[DECLINE] / n_legit), 6),
        "total_cost_inr": round(total, 2),
        "fraud_cost_inr": round(fraud_cost, 2),
        "app_scam_cost_inr": round(app_cost, 2),
        "non_app_fraud_cost_inr": round(non_app_cost, 2),
        "friction_and_insult_cost_inr": round(friction_cost, 2),
        "insult_share_of_total_cost": round(friction_cost / total, 6) if total > 0 else None,
        "cost_model": asdict(m),
    }


def sweep_policies(
    scores: Sequence[float],
    labels: Sequence[int],
    app_candidates: Sequence[bool],
    app_truth: Sequence[bool],
    grid: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
    model: Optional[ActionCostModel] = None,
    two_action_baseline_threshold: Optional[float] = None,
) -> Dict[str, object]:
    """Search a policy family that *actually contains* the two-action baseline.

    Both carve-out states are searched. Therefore the collapsed threshold triple
    ``(t, t, t, app_carve_out=False)`` is exactly the approve/decline baseline.
    """
    m = model or ActionCostModel()
    candidates = {float(g) for g in grid}
    if two_action_baseline_threshold is not None:
        candidates.add(float(two_action_baseline_threshold))
    ordered = sorted(candidates)

    results: List[Dict[str, object]] = []
    for a, b, c in combinations_with_replacement(ordered, 3):
        for carve_out in (False, True):
            results.append(
                evaluate_policy(
                    scores,
                    labels,
                    app_candidates,
                    app_truth,
                    a,
                    b,
                    c,
                    m,
                    app_carve_out=carve_out,
                )
            )
    results.sort(key=lambda r: float(r["total_cost_inr"]))
    best = results[0]

    baseline = baseline_with_carve_out = None
    saving = app_saving = saving_vs_carve_out = None
    baseline_reachable = two_action_baseline_threshold is None
    if two_action_baseline_threshold is not None:
        t = float(two_action_baseline_threshold)
        baseline = evaluate_policy(
            scores, labels, app_candidates, app_truth, t, t, t, m, app_carve_out=False
        )
        baseline["policy"] = "two_action_approve_or_decline_no_app_carve_out"
        baseline_with_carve_out = evaluate_policy(
            scores, labels, app_candidates, app_truth, t, t, t, m, app_carve_out=True
        )
        baseline_with_carve_out["policy"] = "two_action_plus_predicted_app_carve_out"

        baseline_reachable = any(
            r["app_carve_out"] is False
            and all(abs(float(r["thresholds"][name]) - t) <= 1e-6 for name in ("step_up", "cooling_off", "decline"))
            for r in results
        )
        saving = round(float(baseline["total_cost_inr"]) - float(best["total_cost_inr"]), 2)
        app_saving = round(
            float(baseline["app_scam_cost_inr"]) - float(best["app_scam_cost_inr"]), 2
        )
        saving_vs_carve_out = round(
            float(baseline_with_carve_out["total_cost_inr"]) - float(best["total_cost_inr"]), 2
        )

    return {
        "n_policies_evaluated": len(results),
        "best_policy": best,
        "two_action_baseline": baseline,
        "two_action_baseline_with_app_carve_out": baseline_with_carve_out,
        "saving_vs_two_action_inr": saving,
        "saving_vs_two_action_with_carve_out_inr": saving_vs_carve_out,
        "saving_on_app_scam_subset_inr": app_saving,
        "baseline_is_reachable_by_this_family": bool(baseline_reachable),
        "frontier": results[:8],
    }
