"""Reproducible behavioural-fidelity audit for the two lightweight generators.

The generator is fit only on an earlier fraud slice and evaluated against later
held-out fraud. Stateful features are causal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np

from evidence.artifacts import write_artifact
from fidelity.behavior import BEHAVIOURAL_FEATURES, ROW_FEATURES, build_features, matrix
from fidelity.c2st_plus import c2st_report
from fidelity.divergence import compare_categorical, compare_numeric, composite_similarity
from fidelity.fixtures import simulate_real_fraud, synth_joint_behavioural, synth_marginal
from fidelity.graphstats import graph_fidelity_report
from fidelity.temporal import temporal_fidelity_report

COMMAND = "python backend/experiments/run_behavioural_fidelity.py"
SEEDS = [11, 23, 37]


def row_fidelity_report(real, synth):
    real_features = build_features(real)
    synth_features = build_features(synth)
    records = [
        compare_numeric("log_amount", real_features["log_amount"], synth_features["log_amount"]),
        compare_numeric(
            "amount_round_frac",
            real_features["amount_round_frac"],
            synth_features["amount_round_frac"],
        ),
        compare_categorical("mcc", real["mcc"], synth["mcc"]),
        compare_categorical("entry_mode", real["entry_mode"], synth["entry_mode"]),
    ]
    return {"measures": records, "composite_similarity": composite_similarity(records)}


def run_seed(seed: int, n_real: int = 700, c2st_trees: int = 70, permutations: int = 8):
    real = simulate_real_fraud(n_real, seed=seed)
    cut = max(80, int(0.45 * len(real)))
    fit_real = real.iloc[:cut].copy()
    heldout_real = real.iloc[cut:].copy()
    n_synth = len(heldout_real)

    generators = {
        "A1_marginal": synth_marginal(fit_real, n_synth, seed=seed + 101),
        "A2_joint_behavioural": synth_joint_behavioural(
            fit_real, n_synth, seed=seed + 202
        ),
    }
    out = {}
    for name, synth in generators.items():
        real_f = build_features(heldout_real)
        synth_f = build_features(synth)
        out[name] = {
            "row": row_fidelity_report(heldout_real, synth),
            "temporal": temporal_fidelity_report(heldout_real, synth),
            "graph": graph_fidelity_report(heldout_real, synth),
            "c2st_row": c2st_report(
                matrix(real_f, ROW_FEATURES),
                matrix(synth_f, ROW_FEATURES),
                ROW_FEATURES,
                n_estimators=c2st_trees,
                seed=seed,
                n_permutations=permutations,
            ),
            "c2st_behavioural": c2st_report(
                matrix(real_f, BEHAVIOURAL_FEATURES),
                matrix(synth_f, BEHAVIOURAL_FEATURES),
                BEHAVIOURAL_FEATURES,
                n_estimators=c2st_trees,
                seed=seed + 1,
                n_permutations=permutations,
            ),
        }
    return {
        "seed": seed,
        "fit_real_rows": int(len(fit_real)),
        "heldout_real_rows": int(len(heldout_real)),
        "generators": out,
    }


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return None if not values else round(float(np.mean(values)), 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = [11] if args.smoke else SEEDS
    n_real = 260 if args.smoke else 700
    trees = 24 if args.smoke else 70
    permutations = 1 if args.smoke else 8
    command = COMMAND + (" --smoke" if args.smoke else "")

    per_seed = [run_seed(s, n_real, trees, permutations) for s in seeds]
    summary = {}
    for name in ("A1_marginal", "A2_joint_behavioural"):
        summary[name] = {
            "row_composite": _mean(
                [r["generators"][name]["row"]["composite_similarity"] for r in per_seed]
            ),
            "temporal_composite": _mean(
                [r["generators"][name]["temporal"]["composite_similarity"] for r in per_seed]
            ),
            "graph_composite": _mean(
                [r["generators"][name]["graph"]["composite_similarity"] for r in per_seed]
            ),
            "c2st_row_auc": _mean(
                [r["generators"][name]["c2st_row"]["c2st_auc"] for r in per_seed]
            ),
            "c2st_behavioural_auc": _mean(
                [r["generators"][name]["c2st_behavioural"]["c2st_auc"] for r in per_seed]
            ),
        }

    payload = {
        "experiment": "behavioural_fidelity_heldout",
        "protocol": {
            "generator_fit": "earlier real-fraud slice only",
            "fidelity_reference": "later held-out real fraud",
            "features": "strictly causal event-time features",
            "seeds": seeds,
        },
        "summary": summary,
        "per_seed": per_seed,
        "boundaries": [
            "The reference fraud is simulated, not issuer data.",
            "C2ST is a diagnostic, not a privacy guarantee or a deployment metric.",
        ],
    }
    path = write_artifact("behavioural_fidelity", payload, seeds=seeds, command=command)
    print(f"wrote {path}")
    for name, values in summary.items():
        print(name, values)


if __name__ == "__main__":
    main()
