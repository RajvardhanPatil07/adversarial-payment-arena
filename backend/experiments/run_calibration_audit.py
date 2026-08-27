"""
CALIBRATION AUDIT -- where the threshold came from, and what it costs.

This script exists to make one class of claim impossible to make carelessly in
this repository: a recall number quoted at a false-positive rate that was
fitted on the same rows used to report it.

It does three things:

1. Splits legitimate traffic temporally into train / validation / test and pins
   the operating threshold on VALIDATION ONLY, then measures the realised
   false-positive rate on TEST. The gap between the two is reported.
2. Sweeps the same fixed operating point across plausible fraud base rates,
   from laboratory prevalence down to 0.1%, showing what precision an analyst
   would actually experience.
3. Prices each base rate in INR including the insult cost of falsely declining
   a legitimate customer.

Reproduce
---------
    python backend/experiments/run_calibration_audit.py
    make calibration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from data.corpus_builder import build_corpus  # noqa: E402
from evidence.artifacts import write_artifact  # noqa: E402
from evidence.calibration import (  # noqa: E402
    bootstrap_ci,
    calibrate,
    chronological_split,
    precision_at_prevalence,
    prevalence_sweep,
    recall_at_threshold,
)
from evidence.economics import CostModel, evaluate_operating_point  # noqa: E402
from fidelity.features import frame_from_rows  # noqa: E402
from fidelity.metrics import fit_detector, score_detector  # noqa: E402

SEEDS = [11, 23, 37]
N_LEGIT = 6000
ATTACK_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 130,
    "ATTACK_2_SYNTHETIC_MULE_RING": 130,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 130,
}
FRAUD_TRAIN_N = 90
FPR_BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05]
HEADLINE_FPR = 0.01
PRODUCTION_PREVALENCE = 0.013


def run_seed(seed: int) -> dict:
    corpus = build_corpus(n_legit=N_LEGIT, attack_counts=ATTACK_COUNTS, seed=seed)
    rows = corpus["rows"]

    def by_time(collection):
        return sorted(collection, key=lambda r: r["payload"]["timestamp"])

    legit_rows = by_time([r for r in rows if r["label"] == 0])
    fraud_rows = by_time([r for r in rows if r["label"] == 1])

    legit_train, legit_validation, legit_test = chronological_split(
        legit_rows, validation_frac=0.25, test_frac=0.25
    )
    fraud_train = fraud_rows[:FRAUD_TRAIN_N]
    fraud_test = fraud_rows[FRAUD_TRAIN_N:]

    legit_train_frame = frame_from_rows(legit_train)
    legit_validation_frame = frame_from_rows(legit_validation)
    legit_test_frame = frame_from_rows(legit_test)
    fraud_train_frame = frame_from_rows(fraud_train)
    fraud_test_frame = frame_from_rows(fraud_test)

    train_frame = pd.concat([legit_train_frame, fraud_train_frame], ignore_index=True)
    train_labels = np.concatenate(
        [np.zeros(len(legit_train_frame), dtype=int), np.ones(len(fraud_train_frame), dtype=int)]
    )

    model, columns = fit_detector(train_frame, train_labels, seed=seed)
    validation_scores = score_detector(model, columns, legit_validation_frame)
    test_legit_scores = score_detector(model, columns, legit_test_frame)
    test_fraud_scores = score_detector(model, columns, fraud_test_frame)

    operating_points = []
    for budget in FPR_BUDGETS:
        audit = calibrate(validation_scores, test_legit_scores, target_fpr=budget)
        recall = recall_at_threshold(test_fraud_scores, audit.threshold)
        entry = audit.to_dict()
        entry.update(
            {
                "recall_on_held_out_fraud": round(recall, 4),
                "calibration_gap_pct_points": round((audit.test_fpr - audit.validation_fpr) * 100, 3),
                "precision_at_production_prevalence": round(
                    precision_at_prevalence(recall, audit.test_fpr, PRODUCTION_PREVALENCE), 4
                ),
            }
        )
        operating_points.append(entry)

    return {
        "seed": seed,
        "splits": {
            "legit_train": len(legit_train),
            "legit_validation": len(legit_validation),
            "legit_test": len(legit_test),
            "fraud_train": len(fraud_train),
            "fraud_test": len(fraud_test),
        },
        "operating_points": operating_points,
    }


def main() -> dict:
    command = "python backend/experiments/run_calibration_audit.py"
    print(f"running calibration audit over seeds {SEEDS} ...")
    seed_results = [run_seed(seed) for seed in SEEDS]

    aggregated = {}
    for index, budget in enumerate(FPR_BUDGETS):
        aggregated[f"fpr_budget_{budget}"] = {
            "target_fpr": budget,
            "realised_test_fpr": bootstrap_ci(
                [r["operating_points"][index]["test_fpr"] for r in seed_results]
            ),
            "recall_on_held_out_fraud": bootstrap_ci(
                [r["operating_points"][index]["recall_on_held_out_fraud"] for r in seed_results]
            ),
            "calibration_gap_pct_points": bootstrap_ci(
                [r["operating_points"][index]["calibration_gap_pct_points"] for r in seed_results]
            ),
            "precision_at_production_prevalence": bootstrap_ci(
                [
                    r["operating_points"][index]["precision_at_production_prevalence"]
                    for r in seed_results
                ]
            ),
        }

    headline_recall = aggregated[f"fpr_budget_{HEADLINE_FPR}"]["recall_on_held_out_fraud"]["mean"]
    headline_fpr = aggregated[f"fpr_budget_{HEADLINE_FPR}"]["realised_test_fpr"]["mean"]

    cost_model = CostModel()
    sweep = prevalence_sweep(headline_recall, headline_fpr)
    priced_sweep = []
    for row in sweep:
        priced = evaluate_operating_point(
            recall=headline_recall,
            fpr=headline_fpr,
            prevalence=row["prevalence"],
            model=cost_model,
        )
        priced_sweep.append(
            {
                "prevalence": row["prevalence"],
                "precision": row["precision"],
                "true_alerts_per_million": row["true_alerts"],
                "false_alerts_per_million": row["false_alerts"],
                "missed_frauds_per_million": row["missed_frauds"],
                "net_benefit_inr_per_million": priced["net_benefit_inr"],
                "insult_cost_inr_per_million": priced["insult_cost_inr"],
            }
        )

    payload = {
        "protocol": {
            "threshold_source": "legitimate validation split, temporally disjoint from test",
            "why": "pinning a threshold on the rows used to report it converts a measurement into a fit",
            "seeds": SEEDS,
            "fpr_budgets": FPR_BUDGETS,
            "headline_fpr": HEADLINE_FPR,
            "production_prevalence": PRODUCTION_PREVALENCE,
        },
        "aggregated": aggregated,
        "per_seed": seed_results,
        "prevalence_and_cost_sweep": priced_sweep,
        "headline": {
            "recall_at_1pct_fpr": aggregated[f"fpr_budget_{HEADLINE_FPR}"]["recall_on_held_out_fraud"],
            "realised_fpr": aggregated[f"fpr_budget_{HEADLINE_FPR}"]["realised_test_fpr"],
            "calibration_gap": aggregated[f"fpr_budget_{HEADLINE_FPR}"]["calibration_gap_pct_points"],
            "precision_at_production_prevalence": aggregated[f"fpr_budget_{HEADLINE_FPR}"][
                "precision_at_production_prevalence"
            ],
        },
        "boundaries": [
            "The calibration gap measures threshold generalisation inside the corpus window only.",
            "Prevalence is a deployment assumption; the sweep shows sensitivity rather than a single truth.",
            "INR cost rates are stated assumptions and can be overridden in evidence.economics.CostModel.",
        ],
    }
    write_artifact("calibration_audit", payload, seeds=SEEDS, command=command)

    print(json.dumps(payload["headline"], indent=2, default=str))
    print("prevalence and cost sweep:")
    print(json.dumps(priced_sweep, indent=2, default=str))
    return payload


if __name__ == "__main__":
    main()
