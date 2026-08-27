"""Four-action policy economics, including UPI-style APP scams.

An approve/decline model cannot express the control that actually works against
authorised push payment scams. The customer is genuine, the device is genuine,
the authorisation is genuine, so a hard decline mostly insults good traffic while
the victim retries elsewhere. Delay and confirmation recover more value than
blocking. This prices that difference instead of asserting it.

RESULT: at the default assumptions the hypothesis is NOT supported. See the
`reading` and `boundaries` fields of the artifact.

Run:  python backend/experiments/run_action_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from typing import Dict, List

import numpy as np
import pandas as pd

from evidence.actions import ActionCostModel, sweep_policies
from evidence.artifacts import ClaimLedger, write_artifact
from evidence.thresholds import pin_threshold_at_fpr
from fidelity.behavior import BEHAVIOURAL_FEATURES, build_features, matrix
from fidelity.fixtures import simulate_legit, simulate_real_fraud
from ml.forest import RandomForestBinary

SEEDS = [11, 23, 37]
N_LEGIT = 5000
N_RING_FRAUD = 300
N_APP_SCAM = 200
TARGET_FPR = 0.01
DECLINE_EFFICACY_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.58)
COMMAND = "python backend/experiments/run_action_policy.py"


def build_app_scam_stream(legit: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """APP scam: the real customer, on their own device and IP, pushes a large
    payment to a mule beneficiary after being socially engineered.

    Note what is deliberately absent: no device sharing, no IP anomaly, no
    impossible velocity. That is exactly why device-centric ring detection
    misses this class, and why a decline is the wrong control for it.
    """
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


def run_seed(seed: int) -> Dict[str, object]:
    legit = simulate_legit(N_LEGIT, seed=seed)
    ring = simulate_real_fraud(N_RING_FRAUD, seed=seed + 1)
    app = build_app_scam_stream(legit, N_APP_SCAM, seed=seed + 2)

    stream = (
        pd.concat([legit, ring, app], ignore_index=True).sort_values("ts").reset_index(drop=True)
    )
    feats = build_features(stream)
    n = len(feats)
    train = feats.iloc[: int(0.6 * n)]
    val = feats.iloc[int(0.6 * n) : int(0.8 * n)]
    test = feats.iloc[int(0.8 * n) :]

    model = RandomForestBinary(n_estimators=60, max_depth=12, seed=seed).fit(
        matrix(train, BEHAVIOURAL_FEATURES), train["label"].to_numpy(int)
    )
    val_legit = val[val["label"] == 0]
    baseline_threshold = pin_threshold_at_fpr(
        model.predict_proba(matrix(val_legit, BEHAVIOURAL_FEATURES)), TARGET_FPR
    )
    scores = model.predict_proba(matrix(test, BEHAVIOURAL_FEATURES))
    labels = test["label"].to_numpy(int)
    is_app = test["attack_id"].to_numpy() == "APP_SCAM_UPI"

    sweep = sweep_policies(
        scores,
        labels,
        is_app,
        grid=(0.15, 0.30, 0.50, 0.70, 0.90),
        model=ActionCostModel(),
        two_action_baseline_threshold=baseline_threshold,
    )
    # Sensitivity on the single assumption the carve-out depends on: how much of
    # an APP scam a hard decline actually prevents. At 0.58 (equal to a
    # cooling-off) declining dominates by construction.
    sensitivity = []
    for d in DECLINE_EFFICACY_GRID:
        sw = sweep_policies(
            scores,
            labels,
            is_app,
            grid=(0.15, 0.30, 0.50, 0.70, 0.90),
            model=ActionCostModel(decline_blocks_app_scam=d),
            two_action_baseline_threshold=baseline_threshold,
        )
        sensitivity.append(
            {
                "decline_blocks_app_scam": d,
                "saving_vs_true_two_action_inr": sw["saving_vs_two_action_inr"],
                "saving_on_app_scam_subset_inr": sw["saving_on_app_scam_subset_inr"],
                "cooling_off_actions_in_best_policy": sw["best_policy"]["action_counts"][
                    "COOLING_OFF"
                ],
            }
        )

    return {
        "seed": seed,
        "two_action_threshold_at_1pct_fpr": round(float(baseline_threshold), 8),
        "app_decline_efficacy_sensitivity": sensitivity,
        "test_composition": {
            "rows": int(len(test)),
            "legit": int((labels == 0).sum()),
            "ring_fraud": int(((labels == 1) & (~is_app)).sum()),
            "app_scam": int(is_app.sum()),
        },
        "sweep": sweep,
    }


def main() -> None:
    per_seed = [run_seed(s) for s in SEEDS]
    savings = [r["sweep"]["saving_vs_two_action_inr"] for r in per_seed]
    app_savings = [r["sweep"]["saving_on_app_scam_subset_inr"] for r in per_seed]
    best_costs = [r["sweep"]["best_policy"]["total_cost_inr"] for r in per_seed]
    sensitivity_summary = []
    break_even = None
    for i, d in enumerate(DECLINE_EFFICACY_GRID):
        vals = [r["app_decline_efficacy_sensitivity"][i]["saving_vs_true_two_action_inr"] for r in per_seed]
        vals = [v for v in vals if v is not None]
        mean_saving = round(float(np.mean(vals)), 2) if vals else None
        sensitivity_summary.append(
            {"decline_blocks_app_scam": d, "mean_saving_vs_true_two_action_inr": mean_saving}
        )
        if break_even is None and mean_saving is not None and mean_saving > 0:
            break_even = d
    base_costs = [r["sweep"]["two_action_baseline"]["total_cost_inr"] for r in per_seed]
    payload = {
        "experiment": "four_action_policy_economics",
        "question": "Does a four-action policy beat approve/decline once APP scams are in the mix?",
        "action_set": ["APPROVE", "STEP_UP", "COOLING_OFF", "DECLINE"],
        "protocol": {
            "seeds": SEEDS,
            "two_action_baseline": "threshold pinned at 1% FPR on legitimate validation rows, evaluated with app_carve_out=False",
            "four_action": "non-decreasing threshold triples over the grid PLUS the pinned baseline threshold, cheapest total cost selected",
            "app_scam_rule": "APP candidates are never hard-declined on score alone",
        },
        "mean_total_cost_four_action_inr": round(float(np.mean(best_costs)), 2),
        "mean_total_cost_two_action_inr": round(float(np.mean(base_costs)), 2),
        "mean_saving_inr": round(float(np.mean([s for s in savings if s is not None])), 2),
        "mean_saving_on_app_scam_subset_inr": round(
            float(np.mean([s for s in app_savings if s is not None])), 2
        ),
        "baseline_reachable_by_four_action_family": all(
            r["sweep"]["baseline_is_reachable_by_this_family"] for r in per_seed
        ),
        "headline_result": "HYPOTHESIS NOT SUPPORTED at the default assumptions",
        "reading": (
            "This experiment refutes its own hypothesis, and the refutation is more "
            "useful than a confirmation would have been. Against a TRUE approve/decline "
            "baseline (no APP carve-out), the four-action optimum is not cheaper: the "
            "cost-minimising policy collapses to a plain score cascade and STEP_UP is "
            "never selected. The cause is localised to one assumption. When a hard "
            "decline is credited with the same APP-scam prevention as a cooling-off "
            "(both 0.58), declining strictly dominates -- identical prevention, none of "
            "the review cost -- so no search could ever justify the carve-out. The "
            "sensitivity sweep prices that assumption instead of asserting it: the "
            "carve-out only pays once a decline is materially worse at stopping an "
            "authorised push payment than a delay is, which is precisely the empirical "
            "question an issuer would have to answer from its own retry data."
        ),
        "break_even_decline_efficacy": break_even,
        "sensitivity_summary": sensitivity_summary,
        "per_seed": per_seed,
        "boundaries": [
            "Every rate in ActionCostModel is an assumption, not a measured issuer figure. The artifact carries the full model so any number can be recomputed.",
            "Friction effectiveness against APP scams (58% for cooling-off) is the single most influential assumption and is not measured here.",
            "Simulated APP scams have no true victim-behaviour ground truth; they encode a hypothesis about what the pattern looks like.",
            "Cost totals are for the simulated test split only and do not scale linearly to portfolio volumes.",
            "The baseline reported here is a TRUE approve/decline with no APP carve-out. An earlier version gave the baseline the same carve-out as the treatment arm, which made the carve-out's value identically zero by construction.",
            "An earlier version of this sweep used strictly distinct thresholds, could not reproduce the baseline, and reported a NEGATIVE saving of about INR 193,770. That was a search-space bug, not a finding.",
            "STEP_UP is never selected at any assumption tested. Either the step-up cost is too high relative to its 0.82 prevention, or a three-tier ladder is genuinely redundant here. This is not evidence that step-up is useless in production.",
            "The default decline_blocks_app_scam of 0.58 makes the carve-out impossible to justify. It is kept as the default deliberately, so the artifact reports a negative result rather than a tuned positive one.",
        ],
    }
    path = write_artifact("action_policy", payload, seeds=SEEDS, command=COMMAND)
    ledger = ClaimLedger()
    ledger.add(
        claim="At our default cost assumptions the four-action policy does NOT beat a true approve/decline baseline; the APP carve-out only pays once a hard decline is assumed to be materially worse than a delay at stopping an authorised push payment scam.",
        artifact="action_policy",
        field="mean_total_cost_four_action_inr, mean_total_cost_two_action_inr, break_even_decline_efficacy, sensitivity_summary",
        derivation="Cheapest non-decreasing threshold triple over the grid plus the pinned 1% FPR threshold, using combinations_with_replacement so collapsed triples reproduce approve/decline; compared on identical held-out scores.",
        boundary="A negative result at one set of assumptions, not a general proof. The break-even point is reported so the claim can be re-evaluated against real retry data.",
    )
    ledger.add(
        claim="Whether hard declines are the wrong control for APP scams is decidable, and it reduces to one measurable quantity: how often a declined victim completes the payment through another rail.",
        artifact="action_policy",
        field="sensitivity_summary, break_even_decline_efficacy",
        derivation="decline_blocks_app_scam is swept from 0.10 to 0.58 and the policy search is re-run at each value; the sign of the saving flips inside that range.",
        boundary="Reflects a modelling assumption about victim retry behaviour, not an observed recovery rate.",
    )
    ledger.write(command=COMMAND)
    print(f"wrote {path}")
    print(f"  four-action mean cost INR {payload['mean_total_cost_four_action_inr']}")
    print(f"  two-action  mean cost INR {payload['mean_total_cost_two_action_inr']}")
    print(f"  mean saving           INR {payload['mean_saving_inr']}")
    print(f"  of which APP subset   INR {payload['mean_saving_on_app_scam_subset_inr']}")
    print(f"  headline              {payload['headline_result']}")
    print(f"  break-even decline efficacy: {payload['break_even_decline_efficacy']}")
    for row in payload["sensitivity_summary"]:
        print(f"    decline_blocks_app_scam={row['decline_blocks_app_scam']:.2f} -> mean saving INR {row['mean_saving_vs_true_two_action_inr']}")
    print(f"  baseline reachable    {payload['baseline_reachable_by_four_action_family']}")
    for r in per_seed:
        bp = r["sweep"]["best_policy"]
        print(
            f"  seed {r['seed']}: thresholds {bp['thresholds']} "
            f"legit_friction={bp['legit_friction_rate']} hard_decline={bp['legit_hard_decline_rate']}"
        )
        print(f"           actions {bp['action_counts']}")


if __name__ == "__main__":
    main()
