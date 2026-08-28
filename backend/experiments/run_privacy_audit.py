"""Reproducible privacy audit for the lightweight synthetic-fraud generators."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np

from evidence.artifacts import write_artifact
from evidence.privacy import attribute_inference, privacy_audit
from fidelity.behavior import ROW_FEATURES, build_features, matrix
from fidelity.fixtures import simulate_real_fraud, synth_joint_behavioural, synth_marginal

COMMAND = "python backend/experiments/run_privacy_audit.py"
SEEDS = [11, 23, 37]
ATTRIBUTE_FEATURES = tuple(c for c in ROW_FEATURES if c != "entry_mode_code")


def _sensitive_entry_mode(frame):
    return frame["entry_mode"].isin(["ECOM", "CNP"]).to_numpy(dtype=int)


def run_seed(seed: int, n_real: int = 700):
    real = simulate_real_fraud(n_real, seed=seed)
    cut = max(80, int(0.55 * len(real)))
    members = real.iloc[:cut].copy()
    non_members = real.iloc[cut:].copy()
    n_synth = len(non_members)

    generators = {
        "A1_marginal": synth_marginal(members, n_synth, seed=seed + 101),
        "A2_joint_behavioural": synth_joint_behavioural(
            members, n_synth, seed=seed + 202
        ),
    }

    mem_f = build_features(members)
    non_f = build_features(non_members)
    out = {}
    for name, synth in generators.items():
        syn_f = build_features(synth)
        audit = privacy_audit(
            name,
            matrix(syn_f, ROW_FEATURES),
            matrix(mem_f, ROW_FEATURES),
            matrix(non_f, ROW_FEATURES),
        )
        audit["attribute_inference_entry_mode"] = attribute_inference(
            matrix(syn_f, ATTRIBUTE_FEATURES),
            _sensitive_entry_mode(synth),
            matrix(non_f, ATTRIBUTE_FEATURES),
            _sensitive_entry_mode(non_members),
            seed=seed,
        )
        out[name] = audit
    return {"seed": seed, "generators": out}


def _mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return None if not vals else round(float(np.mean(vals)), 6)


def _max(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return None if not vals else round(float(np.max(vals)), 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = [11] if args.smoke else SEEDS
    n_real = 260 if args.smoke else 700
    command = COMMAND + (" --smoke" if args.smoke else "")

    per_seed = [run_seed(s, n_real) for s in seeds]
    summary = {}
    for name in ("A1_marginal", "A2_joint_behavioural"):
        memberships = [
            r["generators"][name]["membership_inference_auc"] for r in per_seed
        ]
        duplicates = [r["generators"][name]["exact_duplicate_share"] for r in per_seed]
        lifts = [
            r["generators"][name]["attribute_inference_entry_mode"]["lift"]
            for r in per_seed
        ]
        summary[name] = {
            "membership_inference_auc_mean": _mean(memberships),
            "membership_inference_auc_max": _max(memberships),
            "duplicate_share_mean": _mean(duplicates),
            "attribute_inference_lift_mean": _mean(lifts),
            "attribute_inference_lift_max": _max(lifts),
        }

    payload = {
        "experiment": "synthetic_fraud_privacy_audit",
        "protocol": {
            "members": "earlier real-fraud rows used to fit generator",
            "non_members": "later held-out real-fraud rows",
            "seeds": seeds,
        },
        "summary": summary,
        "per_seed": per_seed,
        "boundary": (
            "These attacks can detect leakage but cannot prove privacy. "
            "No differential-privacy guarantee is claimed."
        ),
    }
    path = write_artifact("privacy_audit", payload, seeds=seeds, command=command)
    print(f"wrote {path}")
    for name, values in summary.items():
        print(name, values)


if __name__ == "__main__":
    main()
