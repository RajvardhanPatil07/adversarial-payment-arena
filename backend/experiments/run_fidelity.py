"""
FIDELITY LAB -- how realistic are the attacks we generate, measured five ways.

A red-team generator that passes only marginal tests (KS, histogram overlap) is
not realistic; it is *individually* plausible and *jointly* impossible. Real
fraud lives in the dependence structure: amount conditioned on entry mode
conditioned on 3DS outcome conditioned on hour.

This script measures both generators on:

    1. C2ST AUC                     joint separability  (strictest)
    2. mean JSD                     marginal agreement  (easiest)
    3. mean TVD                     categorical agreement
    4. correlation Frobenius diff   rank-dependence structure
    5. TSTR ratio                   usefulness for training a real detector

The acceptance gate for this repository is C2ST AUC <= 0.80 for the copula
generator. If the gate is not cleared, the number is published anyway. A
fidelity lab that only reports its passes is marketing.

Reproduce
---------
    python backend/experiments/run_fidelity.py
    make fidelity
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

from data.corpus_builder import build_corpus  # noqa: E402
from evidence.artifacts import write_artifact  # noqa: E402
from evidence.calibration import bootstrap_ci, chronological_split  # noqa: E402
from fidelity.copula import SYNTHESIZERS  # noqa: E402
from fidelity.features import frame_from_rows  # noqa: E402
from fidelity.metrics import fidelity_report, tstr_report  # noqa: E402

DOCS_DIR = BACKEND_ROOT.parent / "docs"
CHART_PATH = DOCS_DIR / "fidelity_lab.png"

SEEDS = [11, 23, 37]
N_LEGIT = 5000
ATTACK_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 130,
    "ATTACK_2_SYNTHETIC_MULE_RING": 130,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 130,
}
FRAUD_FIT_N = 90
SYNTH_N = 750
TARGET_FPR = 0.01
C2ST_GATE = 0.80


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
    fraud_fit = fraud_rows[:FRAUD_FIT_N]
    fraud_holdout = fraud_rows[FRAUD_FIT_N:]

    legit_train_frame = frame_from_rows(legit_train)
    legit_validation_frame = frame_from_rows(legit_validation)
    legit_test_frame = frame_from_rows(legit_test)
    fraud_fit_frame = frame_from_rows(fraud_fit)
    fraud_holdout_frame = frame_from_rows(fraud_holdout)

    test_frame = pd.concat([legit_test_frame, fraud_holdout_frame], ignore_index=True)
    test_labels = np.concatenate(
        [np.zeros(len(legit_test_frame), dtype=int), np.ones(len(fraud_holdout_frame), dtype=int)]
    )

    real_train_frame = pd.concat([legit_train_frame, fraud_fit_frame], ignore_index=True)
    real_train_labels = np.concatenate(
        [np.zeros(len(legit_train_frame), dtype=int), np.ones(len(fraud_fit_frame), dtype=int)]
    )

    per_generator: dict[str, dict] = {}
    for name, cls in SYNTHESIZERS.items():
        generator = cls(seed=seed).fit(fraud_fit_frame)
        synth_frame = generator.sample(SYNTH_N)

        # Distribution diagnostics against HELD-OUT real fraud.
        report = fidelity_report(fraud_holdout_frame, synth_frame, seed=seed)

        # Usefulness: swap real fraud for synthetic fraud in the training set.
        synth_train_frame = pd.concat([legit_train_frame, synth_frame], ignore_index=True)
        synth_train_labels = np.concatenate(
            [np.zeros(len(legit_train_frame), dtype=int), np.ones(len(synth_frame), dtype=int)]
        )
        report["tstr"] = tstr_report(
            real_train=(real_train_frame, real_train_labels),
            synth_train=(synth_train_frame, synth_train_labels),
            calibration_legit=legit_validation_frame,
            real_test=(test_frame, test_labels),
            target_fpr=TARGET_FPR,
            seed=seed,
        )
        per_generator[name] = report

    return {"seed": seed, "generators": per_generator}


def aggregate(seed_results: list[dict]) -> dict:
    names = list(seed_results[0]["generators"].keys())
    out: dict[str, dict] = {}
    for name in names:
        rows = [r["generators"][name] for r in seed_results]
        out[name] = {
            "c2st_auc": bootstrap_ci([r["c2st"]["c2st_auc"] for r in rows]),
            "mean_jsd": bootstrap_ci([r["marginals"]["mean_jsd"] for r in rows]),
            "mean_tvd": bootstrap_ci([r["marginals"]["mean_tvd"] or float("nan") for r in rows]),
            "correlation_frobenius_diff": bootstrap_ci(
                [r["joint"]["correlation_frobenius_diff"] for r in rows]
            ),
            "tstr_ratio": bootstrap_ci([r["tstr"]["tstr_ratio"] for r in rows]),
        }
    return out


def _plot(aggregated: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    measures = [
        ("c2st_auc", "C2ST AUC\n(0.5 ideal)", 0.5),
        ("mean_jsd", "mean JSD\n(0 ideal)", 0.0),
        ("mean_tvd", "mean TVD\n(0 ideal)", 0.0),
        ("correlation_frobenius_diff", "corr Frobenius\n(0 ideal)", 0.0),
        ("tstr_ratio", "TSTR ratio\n(1 ideal)", 1.0),
    ]
    names = list(aggregated.keys())
    colors = {"independent_marginal": "#dc2626", "gaussian_copula": "#2563eb"}

    fig, axes = plt.subplots(1, len(measures), figsize=(17.5, 4.6))
    for ax, (key, title, ideal) in zip(axes, measures):
        means = [aggregated[n][key]["mean"] for n in names]
        los = [aggregated[n][key]["lo"] for n in names]
        his = [aggregated[n][key]["hi"] for n in names]
        err = [[max(m - l, 0) for m, l in zip(means, los)], [max(h - m, 0) for m, h in zip(means, his)]]
        bars = ax.bar(range(len(names)), means, color=[colors.get(n, "#6b7280") for n in names], width=0.55)
        ax.errorbar(range(len(names)), means, yerr=err, fmt="none", ecolor="#111827", capsize=5, lw=1.2)
        ax.axhline(ideal, color="#059669", ls=":", lw=1.4)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        "Fidelity Lab: five measures, reported together. Dotted green line is the ideal value.\n"
        "Marginal measures (JSD, TVD) are easy to pass; C2ST and correlation structure are not.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    command = "python backend/experiments/run_fidelity.py"
    print(f"running fidelity lab over seeds {SEEDS} ...")
    seed_results = [run_seed(seed) for seed in SEEDS]
    aggregated = aggregate(seed_results)
    _plot(aggregated)

    copula_c2st = aggregated["gaussian_copula"]["c2st_auc"]["mean"]
    gate_cleared = bool(copula_c2st <= C2ST_GATE)

    payload = {
        "measures": [
            "C2ST AUC (joint separability, 0.5 ideal)",
            "mean JSD (marginal, 0 ideal)",
            "mean TVD (categorical, 0 ideal)",
            "correlation Frobenius difference (rank dependence, 0 ideal)",
            "TSTR ratio (usefulness for training, 1 ideal)",
        ],
        "acceptance_gate": {
            "metric": "gaussian_copula C2ST AUC",
            "threshold": C2ST_GATE,
            "observed": round(copula_c2st, 4),
            "cleared": gate_cleared,
            "policy": "the observed value is published whether or not the gate is cleared",
        },
        "aggregated": aggregated,
        "per_seed": seed_results,
        "boundaries": [
            "Fidelity is measured against held-out arena fraud, not issuer production data.",
            "A copula captures rank dependence, not higher-order interaction or sequence structure.",
            "Passing marginal tests alone does not establish realism; that is the point of reporting all five.",
        ],
    }
    write_artifact("fidelity_report", payload, seeds=SEEDS, command=command)

    print(json.dumps({"acceptance_gate": payload["acceptance_gate"], "aggregated": aggregated}, indent=2, default=str))
    print(f"chart written to {CHART_PATH}")
    return payload


if __name__ == "__main__":
    main()
