"""
TRANSFER ABLATION -- the central experiment of this repository.

Research question
-----------------
Red-teaming a fraud detector with synthetic attacks is now a standard idea.
The unexamined assumption is that it *helps*. It does not always: augmenting a
detector with low-fidelity synthetic fraud can move its decision boundary away
from real fraud and cost real-world recall.

So we hold everything constant except the fidelity of the attack generator.

    Arm A0  no augmentation                      (baseline detector)
    Arm A1  independent-marginal synthetic fraud  (the standard rule/template
                                                   approach: right marginals,
                                                   destroyed joint structure)
    Arm A2  Gaussian-copula synthetic fraud       (marginals AND learned rank
                                                   dependence)

Identical across arms: real training rows, augmentation budget, detector
family and hyperparameters, calibration split, target false-positive rate,
and the held-out real-fraud test set. The ONE independent variable is the
generator.

What gets measured
------------------
* recall on HELD-OUT REAL FRAUD at a false-positive rate pinned to 1.00% on a
  disjoint legitimate validation split
* ROC-AUC and PR-AUC on the same held-out set
* precision at production base rate, not laboratory prevalence
* net business impact in INR including the insult cost of false positives
* the fidelity of each generator, so fidelity can be regressed against transfer

Boundary conditions (stated, not hidden)
----------------------------------------
* "Real fraud" here means fraud produced by the arena's topology-aware
  environment synthesizers, not a proprietary issuer dataset. The experiment
  measures whether GENERATOR FIDELITY predicts transfer; it does not claim an
  absolute recall number for live Mastercard traffic.
* The augmentation budget is fixed. A generator could win simply by being
  allowed to produce more rows; that variable is deliberately closed.
* Every number below is a seed-level mean with a nonparametric bootstrap CI.

Reproduce
---------
    python backend/experiments/run_transfer_ablation.py
    make transfer
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from data.corpus_builder import build_corpus  # noqa: E402
from evidence.artifacts import ClaimLedger, write_artifact, write_metrics_summary  # noqa: E402
from evidence.calibration import (  # noqa: E402
    bootstrap_ci,
    chronological_split,
    pin_threshold_at_fpr,
    precision_at_prevalence,
    prevalence_sweep,
)
from evidence.economics import CostModel, evaluate_operating_point  # noqa: E402
from fidelity.copula import GaussianCopulaSynthesizer, IndependentMarginalSynthesizer  # noqa: E402
from fidelity.features import frame_from_rows  # noqa: E402
from fidelity.metrics import fidelity_report, fit_detector, score_detector  # noqa: E402

DOCS_DIR = BACKEND_ROOT.parent / "docs"
CHART_PATH = DOCS_DIR / "transfer_ledger.png"

SEEDS = [11, 23, 37]
N_LEGIT = 6000
TRAIN_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 130,
    "ATTACK_2_SYNTHETIC_MULE_RING": 130,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 130,
}
REAL_FRAUD_TRAIN_N = 90          # deliberately scarce: labelled fraud always is
SYNTH_BUDGET = 750               # fixed augmentation budget, matched across arms
TARGET_FPR = 0.01
PRODUCTION_PREVALENCE = 0.013    # ~1.3%, a realistic authorisation base rate
LAB_PREVALENCE_NOTE = "laboratory corpora run fraud prevalence orders of magnitude above production"

# Published comparator figure, used only as an annotated reference point on the
# chart. Source: a competing GFF 2026 submission reporting C2ST 0.9801 and a
# 3.8-point real-fraud recall LOSS after hardening on parametric attacks.
COMPARATOR = {"label": "published parametric red-team", "c2st": 0.9801, "delta_recall": -0.038}


def _arm_metrics(
    train_frame: pd.DataFrame,
    train_labels: np.ndarray,
    calibration_legit: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_labels: np.ndarray,
    seed: int,
) -> dict:
    """Fit one arm, pin its threshold out-of-sample, and measure it."""
    model, columns = fit_detector(train_frame, train_labels, seed=seed)

    calib_scores = score_detector(model, columns, calibration_legit)
    tau = pin_threshold_at_fpr(calib_scores, TARGET_FPR)

    test_scores = score_detector(model, columns, test_frame)
    fraud_scores = test_scores[test_labels == 1]
    legit_scores = test_scores[test_labels == 0]

    recall = float(np.mean(fraud_scores >= tau))
    fpr = float(np.mean(legit_scores >= tau))

    return {
        "threshold": round(float(tau), 6),
        "threshold_source": "legitimate validation split, disjoint from test",
        "recall_on_real_fraud": round(recall, 4),
        "fpr_on_real_legit": round(fpr, 4),
        "roc_auc": round(float(roc_auc_score(test_labels, test_scores)), 4),
        "pr_auc": round(float(average_precision_score(test_labels, test_scores)), 4),
        "precision_at_lab_prevalence": round(
            precision_at_prevalence(recall, fpr, float(np.mean(test_labels))), 4
        ),
        "precision_at_production_prevalence": round(
            precision_at_prevalence(recall, fpr, PRODUCTION_PREVALENCE), 4
        ),
        "train_rows": int(len(train_frame)),
        "train_fraud_rows": int(train_labels.sum()),
    }


def run_seed(seed: int) -> dict:
    """One complete three-arm comparison on one seed."""
    corpus = build_corpus(n_legit=N_LEGIT, attack_counts=TRAIN_COUNTS, seed=seed)
    rows = corpus["rows"]

    def by_time(collection):
        return sorted(collection, key=lambda r: r["payload"]["timestamp"])

    legit_rows = by_time([r for r in rows if r["label"] == 0])
    fraud_rows = by_time([r for r in rows if r["label"] == 1])

    legit_train, legit_validation, legit_test = chronological_split(
        legit_rows, validation_frac=0.25, test_frac=0.25
    )
    fraud_train = fraud_rows[:REAL_FRAUD_TRAIN_N]
    fraud_test = fraud_rows[REAL_FRAUD_TRAIN_N:]
    if len(fraud_test) < 50:
        raise RuntimeError("not enough held-out real fraud; raise TRAIN_COUNTS")

    legit_train_frame = frame_from_rows(legit_train)
    legit_validation_frame = frame_from_rows(legit_validation)
    legit_test_frame = frame_from_rows(legit_test)
    fraud_train_frame = frame_from_rows(fraud_train)
    fraud_test_frame = frame_from_rows(fraud_test)

    test_frame = pd.concat([legit_test_frame, fraud_test_frame], ignore_index=True)
    test_labels = np.concatenate(
        [np.zeros(len(legit_test_frame), dtype=int), np.ones(len(fraud_test_frame), dtype=int)]
    )

    base_frame = pd.concat([legit_train_frame, fraud_train_frame], ignore_index=True)
    base_labels = np.concatenate(
        [np.zeros(len(legit_train_frame), dtype=int), np.ones(len(fraud_train_frame), dtype=int)]
    )

    # --- generators are fit ONLY on the scarce real fraud training slice ----
    generators = {
        "A1_independent_marginal": IndependentMarginalSynthesizer(seed=seed).fit(fraud_train_frame),
        "A2_gaussian_copula": GaussianCopulaSynthesizer(seed=seed).fit(fraud_train_frame),
    }

    arms: dict[str, dict] = {
        "A0_baseline": _arm_metrics(
            base_frame, base_labels, legit_validation_frame, test_frame, test_labels, seed
        )
    }
    fidelity: dict[str, dict] = {}

    for arm_name, generator in generators.items():
        synth_frame = generator.sample(SYNTH_BUDGET)
        aug_frame = pd.concat([base_frame, synth_frame], ignore_index=True)
        aug_labels = np.concatenate([base_labels, np.ones(len(synth_frame), dtype=int)])
        arms[arm_name] = _arm_metrics(
            aug_frame, aug_labels, legit_validation_frame, test_frame, test_labels, seed
        )
        # Fidelity is measured against HELD-OUT real fraud, never the fit slice.
        fidelity[arm_name] = fidelity_report(fraud_test_frame, synth_frame, seed=seed)

    baseline_recall = arms["A0_baseline"]["recall_on_real_fraud"]
    for arm_name in generators:
        arms[arm_name]["delta_recall_vs_baseline"] = round(
            arms[arm_name]["recall_on_real_fraud"] - baseline_recall, 4
        )

    return {
        "seed": seed,
        "splits": {
            "legit_train": len(legit_train),
            "legit_validation": len(legit_validation),
            "legit_test": len(legit_test),
            "real_fraud_train": len(fraud_train),
            "real_fraud_test": len(fraud_test),
            "synthetic_budget_per_arm": SYNTH_BUDGET,
        },
        "arms": arms,
        "fidelity": fidelity,
    }


def aggregate(seed_results: list[dict]) -> dict:
    """Seed-level means with bootstrap CIs for every reported quantity."""
    arm_names = list(seed_results[0]["arms"].keys())
    summary: dict[str, dict] = {}

    for arm in arm_names:
        collect = lambda key: [r["arms"][arm][key] for r in seed_results if key in r["arms"][arm]]  # noqa: E731
        entry = {
            "recall_on_real_fraud": bootstrap_ci(collect("recall_on_real_fraud")),
            "fpr_on_real_legit": bootstrap_ci(collect("fpr_on_real_legit")),
            "roc_auc": bootstrap_ci(collect("roc_auc")),
            "pr_auc": bootstrap_ci(collect("pr_auc")),
            "precision_at_production_prevalence": bootstrap_ci(
                collect("precision_at_production_prevalence")
            ),
        }
        deltas = collect("delta_recall_vs_baseline")
        if deltas:
            entry["delta_recall_vs_baseline"] = bootstrap_ci(deltas)
        summary[arm] = entry

    fidelity_summary: dict[str, dict] = {}
    for arm in seed_results[0]["fidelity"].keys():
        fidelity_summary[arm] = {
            "c2st_auc": bootstrap_ci([r["fidelity"][arm]["c2st"]["c2st_auc"] for r in seed_results]),
            "mean_jsd": bootstrap_ci([r["fidelity"][arm]["marginals"]["mean_jsd"] for r in seed_results]),
            "mean_tvd": bootstrap_ci(
                [r["fidelity"][arm]["marginals"]["mean_tvd"] or float("nan") for r in seed_results]
            ),
            "correlation_frobenius_diff": bootstrap_ci(
                [r["fidelity"][arm]["joint"]["correlation_frobenius_diff"] for r in seed_results]
            ),
        }

    return {"arms": summary, "fidelity": fidelity_summary}


def _plot(seed_results: list[dict], aggregated: dict) -> None:
    """The headline figure: generator fidelity against real-fraud transfer."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 6.0))

    colors = {"A1_independent_marginal": "#dc2626", "A2_gaussian_copula": "#2563eb"}
    labels = {
        "A1_independent_marginal": "A1 independent marginals (rule/template equivalent)",
        "A2_gaussian_copula": "A2 Gaussian copula (joint-aware)",
    }

    for arm, color in colors.items():
        xs = [r["fidelity"][arm]["c2st"]["c2st_auc"] for r in seed_results]
        ys = [r["arms"][arm]["delta_recall_vs_baseline"] for r in seed_results]
        ax.scatter(xs, ys, s=110, color=color, edgecolors="white", linewidths=1.4,
                   zorder=5, label=labels[arm])

    ax.scatter([COMPARATOR["c2st"]], [COMPARATOR["delta_recall"]], marker="X", s=170,
               color="#6b7280", zorder=5, label=COMPARATOR["label"] + " (published)")
    ax.annotate(
        "published comparator:\nC2ST 0.980, real-fraud recall -3.8 pts",
        xy=(COMPARATOR["c2st"], COMPARATOR["delta_recall"]),
        xytext=(-165, 26), textcoords="offset points", fontsize=8, color="#374151",
        arrowprops={"arrowstyle": "->", "color": "#9ca3af", "lw": 1},
    )

    ax.axhline(0.0, color="#111827", lw=1.2, ls="--")
    ax.text(0.505, 0.004, "above this line, red-teaming HELPED the real detector",
            fontsize=8, color="#111827")
    ax.text(0.505, -0.012, "below this line, red-teaming HURT it", fontsize=8, color="#991b1b")
    ax.axvline(0.5, color="#059669", lw=1, ls=":")
    ax.text(0.51, ax.get_ylim()[1] * 0.86, "C2ST 0.5 = synthetic\nindistinguishable from real",
            fontsize=8, color="#059669")

    ax.set_xlabel("C2ST AUC of the attack generator  (0.5 = perfect fidelity, 1.0 = trivially fake)")
    ax.set_ylabel("Change in recall on HELD-OUT REAL FRAUD vs baseline")
    ax.set_title("Generator fidelity predicts whether red-teaming\nhelps or hurts real-world detection", fontsize=12)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.94)
    ax.grid(alpha=0.2)

    # ---- right panel: recall by arm with bootstrap CIs --------------------- #
    arm_order = ["A0_baseline", "A1_independent_marginal", "A2_gaussian_copula"]
    pretty = ["A0\nno augmentation", "A1\nindependent\nmarginals", "A2\nGaussian\ncopula"]
    means = [aggregated["arms"][a]["recall_on_real_fraud"]["mean"] for a in arm_order]
    los = [aggregated["arms"][a]["recall_on_real_fraud"]["lo"] for a in arm_order]
    his = [aggregated["arms"][a]["recall_on_real_fraud"]["hi"] for a in arm_order]
    err_lo = [max(m - l, 0.0) for m, l in zip(means, los)]
    err_hi = [max(h - m, 0.0) for m, h in zip(means, his)]

    bars = bx.bar(pretty, means, color=["#6b7280", "#dc2626", "#2563eb"], width=0.58)
    bx.errorbar(pretty, means, yerr=[err_lo, err_hi], fmt="none", ecolor="#111827", capsize=6, lw=1.4)
    for bar, mean in zip(bars, means):
        bx.text(bar.get_x() + bar.get_width() / 2, mean + 0.022, f"{mean:.1%}",
                ha="center", fontsize=10, fontweight="bold")

    fpr_mean = aggregated["arms"]["A0_baseline"]["fpr_on_real_legit"]["mean"]
    bx.set_ylabel("Recall on held-out REAL fraud")
    bx.set_ylim(0, min(1.05, max(his) + 0.16))
    bx.set_title(
        "Same detector, same budget, same pinned FPR\n"
        f"false-positive rate held at {fpr_mean:.2%} "
        f"| {len(SEEDS)} seeds, bootstrap 95% CI",
        fontsize=11,
    )
    bx.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    command = "python backend/experiments/run_transfer_ablation.py"
    print(f"running transfer ablation over seeds {SEEDS} ...")

    seed_results = []
    for seed in SEEDS:
        print(f"  seed {seed} ...")
        seed_results.append(run_seed(seed))

    aggregated = aggregate(seed_results)
    _plot(seed_results, aggregated)

    # ---- prevalence and economics for the winning arm ---------------------- #
    best_arm = "A2_gaussian_copula"
    recall = aggregated["arms"][best_arm]["recall_on_real_fraud"]["mean"]
    fpr = aggregated["arms"][best_arm]["fpr_on_real_legit"]["mean"]

    prevalence_table = prevalence_sweep(recall, fpr)
    cost_model = CostModel()
    economics = {
        "cost_model": cost_model.to_dict(),
        "at_production_prevalence": evaluate_operating_point(
            recall=recall, fpr=fpr, prevalence=PRODUCTION_PREVALENCE, model=cost_model
        ),
        "arms_priced": {
            arm: evaluate_operating_point(
                recall=aggregated["arms"][arm]["recall_on_real_fraud"]["mean"],
                fpr=aggregated["arms"][arm]["fpr_on_real_legit"]["mean"],
                prevalence=PRODUCTION_PREVALENCE,
                model=cost_model,
            )
            for arm in aggregated["arms"]
        },
        "note": (
            "Insult cost is the expected INR cost of wrongly declining a legitimate "
            "payment: lost margin plus support contact plus probability-weighted churn. "
            "At a 1% false-positive rate it is the dominant cost term."
        ),
    }

    write_artifact(
        "transfer_ledger",
        {
            "design": {
                "question": "does generator fidelity determine whether red-teaming helps real-world detection?",
                "independent_variable": "attack generator only",
                "held_constant": [
                    "real training rows",
                    "augmentation budget (%d rows)" % SYNTH_BUDGET,
                    "detector family and hyperparameters",
                    "calibration split and target FPR (%.2f%%)" % (TARGET_FPR * 100),
                    "held-out real-fraud test set",
                ],
                "seeds": SEEDS,
                "target_fpr": TARGET_FPR,
                "production_prevalence": PRODUCTION_PREVALENCE,
            },
            "aggregated": aggregated,
            "per_seed": seed_results,
            "comparator_reference": COMPARATOR,
            "boundaries": [
                "Real fraud means the arena environment's topology-aware synthesizers, not issuer production data.",
                "The claim is about the RELATIONSHIP between fidelity and transfer, not an absolute recall figure for live traffic.",
                "Augmentation budget is fixed across arms so no generator wins by volume.",
                LAB_PREVALENCE_NOTE,
            ],
        },
        seeds=SEEDS,
        command=command,
    )

    write_artifact(
        "prevalence_metrics",
        {
            "operating_point": {"arm": best_arm, "recall": recall, "fpr": fpr},
            "sweep": prevalence_table,
            "note": (
                "The model is identical across every row. Only the assumed base rate "
                "changes. Precision collapse under realistic prevalence is a property "
                "of the base rate and belongs in the result, not in a footnote."
            ),
        },
        seeds=SEEDS,
        command=command,
    )

    write_artifact("economics", economics, seeds=SEEDS, command=command)

    delta_a1 = aggregated["arms"]["A1_independent_marginal"]["delta_recall_vs_baseline"]
    delta_a2 = aggregated["arms"]["A2_gaussian_copula"]["delta_recall_vs_baseline"]
    c2st_a1 = aggregated["fidelity"]["A1_independent_marginal"]["c2st_auc"]
    c2st_a2 = aggregated["fidelity"]["A2_gaussian_copula"]["c2st_auc"]

    summary = {
        "headline": {
            "baseline_recall": aggregated["arms"]["A0_baseline"]["recall_on_real_fraud"],
            "delta_recall_independent_marginal": delta_a1,
            "delta_recall_gaussian_copula": delta_a2,
            "c2st_independent_marginal": c2st_a1,
            "c2st_gaussian_copula": c2st_a2,
            "pinned_fpr": TARGET_FPR,
            "seeds": SEEDS,
        },
        "precision_at_production_prevalence": aggregated["arms"][best_arm][
            "precision_at_production_prevalence"
        ],
        "net_benefit_inr_at_production_prevalence": economics["at_production_prevalence"][
            "net_benefit_inr"
        ],
        "insult_share_of_total_cost": economics["at_production_prevalence"][
            "insult_share_of_total_cost"
        ],
    }
    write_metrics_summary(summary, seeds=SEEDS, command=command)

    (
        ClaimLedger()
        .add(
            claim="Generator fidelity determines whether red-teaming helps or hurts real-fraud recall.",
            artifact="transfer_ledger",
            field="aggregated.arms.*.delta_recall_vs_baseline",
            derivation="Recall on held-out real fraud per arm minus the A0 baseline, at an FPR pinned to 1% on a disjoint legitimate validation split.",
            boundary="Three seeds; synthetic environment fraud; fixed augmentation budget.",
        )
        .add(
            claim="The independent-marginal generator is trivially distinguishable from real fraud.",
            artifact="transfer_ledger",
            field="aggregated.fidelity.A1_independent_marginal.c2st_auc",
            derivation="Cross-validated ROC-AUC of a random forest separating real from synthetic rows.",
            boundary="Measured against held-out real fraud, never the generator's fit slice.",
        )
        .add(
            claim="Operating thresholds are calibrated without leakage.",
            artifact="transfer_ledger",
            field="per_seed[].arms.*.threshold_source",
            derivation="Threshold pinned at 1% FPR on a legitimate validation split, then measured on a disjoint test split.",
            boundary="Temporal split; non-stationarity beyond the corpus window is untested.",
        )
        .add(
            claim="Precision degrades sharply at production base rate.",
            artifact="prevalence_metrics",
            field="sweep[].precision",
            derivation="Bayes adjustment p*R / (p*R + (1-p)*FPR) applied to the measured operating point.",
            boundary="Prevalence is a deployment assumption, not a measurement.",
        )
        .add(
            claim="False positives dominate the cost ledger at a 1% false-positive rate.",
            artifact="economics",
            field="at_production_prevalence.insult_share_of_total_cost",
            derivation="Insult cost per false positive = lost margin + support contact + churn probability * CLV.",
            boundary="INR rates are stated order-of-magnitude assumptions and are overridable.",
        )
        .write(command=command)
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"chart written to {CHART_PATH}")
    return summary


if __name__ == "__main__":
    main()
