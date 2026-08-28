"""Leakage-free four-action policy experiment with a learned APP-candidate model.

Important separation:
- fraud threshold is calibrated on validation legitimate rows;
- APP-candidate status on test is *predicted* from observable features;
- policy thresholds/carve-out are selected on validation outcomes;
- the selected policy is frozen before test evaluation;
- ground-truth APP subtype is used only to score outcomes, never to choose an action.

Run:
    python backend/experiments/run_action_policy.py
    python backend/experiments/run_action_policy.py --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import pandas as pd

from evidence.actions import ActionCostModel, evaluate_policy, sweep_policies
from evidence.artifacts import ClaimLedger, write_artifact
from evidence.thresholds import pin_threshold_at_fpr, rate_at_or_above
from fidelity.behavior import BEHAVIOURAL_FEATURES, build_features, matrix
from fidelity.fixtures import simulate_legit, simulate_real_fraud
from ml.forest import RandomForestBinary

DEFAULT_SEEDS = [11, 23, 37]
TARGET_FPR = 0.01
APP_CANDIDATE_TARGET_FPR = 0.02
DECLINE_EFFICACY_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.58)
POLICY_GRID = (0.15, 0.30, 0.50, 0.70, 0.90)
COMMAND = "python backend/experiments/run_action_policy.py"


def build_app_scam_stream(legit: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Simulated APP scam using a customer's real device/IP and an observable mule payment."""
    rng = np.random.default_rng(seed)
    victims = legit.sample(n=min(n_rows, len(legit)), random_state=seed).reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    for i in range(len(victims)):
        v = victims.iloc[i]
        rows.append(
            {
                "customer": v["customer"],
                "device": v["device"],
                "ip": v["ip"],
                "merchant": f"MULE_{int(rng.integers(0, 60)):03d}",
                "mcc": 6011,
                "amount": float(rng.lognormal(mean=10.2, sigma=0.5)),
                "ts": float(v["ts"]) + float(rng.uniform(600, 6 * 3600)),
                "entry_mode": "ECOM",
                "label": 1,
                "attack_id": "APP_SCAM_UPI",
            }
        )
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def _app_truth(frame: pd.DataFrame) -> np.ndarray:
    return frame["attack_id"].to_numpy() == "APP_SCAM_UPI"


def _candidate_metrics(candidates: np.ndarray, truth: np.ndarray) -> Dict[str, object]:
    candidates = np.asarray(candidates, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(candidates & truth))
    fp = int(np.sum(candidates & ~truth))
    fn = int(np.sum(~candidates & truth))
    tn = int(np.sum(~candidates & ~truth))
    return {
        "rows": int(truth.size),
        "candidates": int(candidates.sum()),
        "true_app": int(truth.sum()),
        "recall": None if tp + fn == 0 else round(tp / (tp + fn), 6),
        "false_positive_rate": None if fp + tn == 0 else round(fp / (fp + tn), 6),
        "precision": None if tp + fp == 0 else round(tp / (tp + fp), 6),
    }


def _evaluate_frozen(
    selected: Dict[str, object],
    scores: np.ndarray,
    labels: np.ndarray,
    app_candidates: np.ndarray,
    app_truth: np.ndarray,
    baseline_threshold: float,
    model: ActionCostModel,
) -> Dict[str, object]:
    t = selected["thresholds"]
    chosen = evaluate_policy(
        scores,
        labels,
        app_candidates,
        app_truth,
        float(t["step_up"]),
        float(t["cooling_off"]),
        float(t["decline"]),
        model,
        app_carve_out=bool(selected["app_carve_out"]),
    )
    baseline = evaluate_policy(
        scores,
        labels,
        app_candidates,
        app_truth,
        baseline_threshold,
        baseline_threshold,
        baseline_threshold,
        model,
        app_carve_out=False,
    )
    baseline["policy"] = "two_action_approve_or_decline_no_app_carve_out"
    return {
        "selected_policy": chosen,
        "two_action_baseline": baseline,
        "saving_vs_two_action_inr": round(
            float(baseline["total_cost_inr"]) - float(chosen["total_cost_inr"]), 2
        ),
        "saving_on_app_scam_subset_inr": round(
            float(baseline["app_scam_cost_inr"]) - float(chosen["app_scam_cost_inr"]), 2
        ),
    }


def run_seed(
    seed: int,
    n_legit: int = 5000,
    n_ring_fraud: int = 300,
    n_app_scam: int = 200,
    n_estimators: int = 60,
) -> Dict[str, object]:
    legit = simulate_legit(n_legit, seed=seed)
    ring = simulate_real_fraud(n_ring_fraud, seed=seed + 1)
    app = build_app_scam_stream(legit, n_app_scam, seed=seed + 2)

    stream = pd.concat([legit, ring, app], ignore_index=True).sort_values("ts").reset_index(drop=True)
    feats = build_features(stream)
    n = len(feats)
    train = feats.iloc[: int(0.6 * n)].copy()
    val = feats.iloc[int(0.6 * n) : int(0.8 * n)].copy()
    test = feats.iloc[int(0.8 * n) :].copy()

    fraud_model = RandomForestBinary(
        n_estimators=n_estimators, max_depth=12, seed=seed
    ).fit(matrix(train, BEHAVIOURAL_FEATURES), train["label"].to_numpy(int))

    val_scores = fraud_model.predict_proba(matrix(val, BEHAVIOURAL_FEATURES))
    test_scores = fraud_model.predict_proba(matrix(test, BEHAVIOURAL_FEATURES))
    val_labels = val["label"].to_numpy(int)
    test_labels = test["label"].to_numpy(int)

    val_legit_scores = val_scores[val_labels == 0]
    baseline_threshold = pin_threshold_at_fpr(val_legit_scores, TARGET_FPR)
    validation_fpr = rate_at_or_above(val_legit_scores, baseline_threshold)

    train_app_truth = _app_truth(train).astype(int)
    if np.unique(train_app_truth).size < 2:
        raise RuntimeError("APP-candidate training split does not contain both classes")
    app_model = RandomForestBinary(
        n_estimators=max(24, n_estimators // 2), max_depth=10, seed=seed + 1000
    ).fit(matrix(train, BEHAVIOURAL_FEATURES), train_app_truth)

    val_app_truth = _app_truth(val)
    test_app_truth = _app_truth(test)
    val_app_scores = app_model.predict_proba(matrix(val, BEHAVIOURAL_FEATURES))
    test_app_scores = app_model.predict_proba(matrix(test, BEHAVIOURAL_FEATURES))
    val_non_app_scores = val_app_scores[~val_app_truth]
    app_candidate_threshold = pin_threshold_at_fpr(
        val_non_app_scores, APP_CANDIDATE_TARGET_FPR
    )
    val_app_candidates = val_app_scores >= app_candidate_threshold
    test_app_candidates = test_app_scores >= app_candidate_threshold

    validation_search = sweep_policies(
        val_scores,
        val_labels,
        val_app_candidates,
        val_app_truth,
        grid=POLICY_GRID,
        model=ActionCostModel(),
        two_action_baseline_threshold=baseline_threshold,
    )
    selected = validation_search["best_policy"]
    test_evaluation = _evaluate_frozen(
        selected,
        test_scores,
        test_labels,
        test_app_candidates,
        test_app_truth,
        baseline_threshold,
        ActionCostModel(),
    )

    sensitivity = []
    for efficacy in DECLINE_EFFICACY_GRID:
        cost_model = ActionCostModel(decline_blocks_app_scam=efficacy)
        search = sweep_policies(
            val_scores,
            val_labels,
            val_app_candidates,
            val_app_truth,
            grid=POLICY_GRID,
            model=cost_model,
            two_action_baseline_threshold=baseline_threshold,
        )
        frozen = _evaluate_frozen(
            search["best_policy"],
            test_scores,
            test_labels,
            test_app_candidates,
            test_app_truth,
            baseline_threshold,
            cost_model,
        )
        sensitivity.append(
            {
                "decline_blocks_app_scam": efficacy,
                "saving_vs_true_two_action_inr": frozen["saving_vs_two_action_inr"],
                "saving_on_app_scam_subset_inr": frozen["saving_on_app_scam_subset_inr"],
                "selected_policy": frozen["selected_policy"],
            }
        )

    return {
        "seed": seed,
        "two_action_threshold_at_1pct_validation_fpr": round(float(baseline_threshold), 8),
        "realised_validation_fpr": round(float(validation_fpr), 6),
        "app_candidate_threshold": round(float(app_candidate_threshold), 8),
        "app_candidate_validation_metrics": _candidate_metrics(val_app_candidates, val_app_truth),
        "app_candidate_test_metrics": _candidate_metrics(test_app_candidates, test_app_truth),
        "validation_search": validation_search,
        "test_evaluation": test_evaluation,
        "app_decline_efficacy_sensitivity": sensitivity,
        "test_composition": {
            "rows": int(len(test)),
            "legit": int((test_labels == 0).sum()),
            "ring_fraud": int(((test_labels == 1) & (~test_app_truth)).sum()),
            "app_scam": int(test_app_truth.sum()),
        },
    }


def _break_even_interval(summary: List[Dict[str, object]]) -> Dict[str, object] | None:
    for left, right in zip(summary, summary[1:]):
        a = float(left["mean_saving_vs_true_two_action_inr"])
        b = float(right["mean_saving_vs_true_two_action_inr"])
        if (a >= 0 > b) or (a <= 0 < b):
            return {
                "between_decline_blocks_app_scam": [
                    left["decline_blocks_app_scam"],
                    right["decline_blocks_app_scam"],
                ],
                "note": "coarse interval from the published sensitivity grid; not an interpolated estimate",
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="small deterministic CI run")
    args = parser.parse_args()

    if args.smoke:
        seeds = [11]
        n_legit, n_ring, n_app, trees = 900, 90, 70, 24
        command = f"{COMMAND} --smoke"
    else:
        seeds = DEFAULT_SEEDS
        n_legit, n_ring, n_app, trees = 5000, 300, 200, 60
        command = COMMAND

    per_seed = [
        run_seed(s, n_legit=n_legit, n_ring_fraud=n_ring, n_app_scam=n_app, n_estimators=trees)
        for s in seeds
    ]

    savings = [r["test_evaluation"]["saving_vs_two_action_inr"] for r in per_seed]
    app_savings = [r["test_evaluation"]["saving_on_app_scam_subset_inr"] for r in per_seed]
    selected_costs = [r["test_evaluation"]["selected_policy"]["total_cost_inr"] for r in per_seed]
    base_costs = [r["test_evaluation"]["two_action_baseline"]["total_cost_inr"] for r in per_seed]

    sensitivity_summary: List[Dict[str, object]] = []
    for i, efficacy in enumerate(DECLINE_EFFICACY_GRID):
        vals = [
            r["app_decline_efficacy_sensitivity"][i]["saving_vs_true_two_action_inr"]
            for r in per_seed
        ]
        sensitivity_summary.append(
            {
                "decline_blocks_app_scam": efficacy,
                "mean_saving_vs_true_two_action_inr": round(float(np.mean(vals)), 2),
            }
        )

    mean_saving = round(float(np.mean(savings)), 2)
    headline = (
        "FOUR-ACTION POLICY IMPROVES HELD-OUT COST"
        if mean_saving > 0
        else (
            "FOUR-ACTION POLICY MATCHES HELD-OUT BASELINE"
            if mean_saving == 0
            else "FOUR-ACTION POLICY DOES NOT IMPROVE HELD-OUT COST"
        )
    )

    payload = {
        "experiment": "four_action_policy_economics_leakage_free",
        "question": "Does a validation-selected four-action policy beat approve/decline on held-out test traffic?",
        "action_set": ["APPROVE", "STEP_UP", "COOLING_OFF", "DECLINE"],
        "protocol": {
            "seeds": seeds,
            "fraud_threshold": "pinned on legitimate validation rows with exact empirical FPR control under ties",
            "app_candidate": "random-forest prediction from observable causal features; attack_id is never passed to choose_action",
            "policy_selection": "thresholds and app_carve_out selected on validation outcomes only, then frozen for test",
            "baseline": "approve/decline at the validation-pinned fraud threshold with app_carve_out=False",
        },
        "mean_total_cost_four_action_inr": round(float(np.mean(selected_costs)), 2),
        "mean_total_cost_two_action_inr": round(float(np.mean(base_costs)), 2),
        "mean_saving_inr": mean_saving,
        "mean_saving_on_app_scam_subset_inr": round(float(np.mean(app_savings)), 2),
        "baseline_reachable_by_four_action_family": all(
            r["validation_search"]["baseline_is_reachable_by_this_family"] for r in per_seed
        ),
        "headline_result": headline,
        "sensitivity_summary": sensitivity_summary,
        "break_even_interval": _break_even_interval(sensitivity_summary),
        "per_seed": per_seed,
        "boundaries": [
            "All ActionCostModel rates remain assumptions, not measured issuer figures.",
            "APP-candidate quality is measured on held-out test traffic and includes false positives/false negatives.",
            "Policy selection uses validation outcomes; test outcomes are used only once for final evaluation.",
            "The simulator is not issuer traffic, so absolute INR and recall values are illustrative.",
            "The baseline is a literal member of the searched family: equal thresholds with app_carve_out=False.",
        ],
    }

    path = write_artifact("action_policy", payload, seeds=seeds, command=command)
    ledger = ClaimLedger()
    ledger.add(
        claim="The four-action policy result is measured without oracle APP labels or test-set policy tuning.",
        artifact="action_policy",
        field="protocol, per_seed.app_candidate_test_metrics, per_seed.test_evaluation",
        derivation="Train APP-candidate model on train, calibrate on validation, choose policy on validation, freeze and score on test.",
        boundary="The APP labels and economics are simulated.",
    )
    ledger.add(
        claim="The two-action baseline is reachable by the searched family.",
        artifact="action_policy",
        field="baseline_reachable_by_four_action_family",
        derivation="The search includes app_carve_out=False and the collapsed threshold triple at the pinned baseline threshold.",
        boundary="Reachability is structural; it does not imply the baseline is optimal.",
    )
    ledger.write(command=command)

    print(f"wrote {path}")
    print(f"  four-action mean test cost INR {payload['mean_total_cost_four_action_inr']}")
    print(f"  two-action  mean test cost INR {payload['mean_total_cost_two_action_inr']}")
    print(f"  mean held-out saving       INR {payload['mean_saving_inr']}")
    print(f"  headline                       {payload['headline_result']}")
    print(f"  baseline reachable             {payload['baseline_reachable_by_four_action_family']}")


if __name__ == "__main__":
    main()
