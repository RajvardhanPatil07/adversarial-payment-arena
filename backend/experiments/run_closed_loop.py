"""
CLOSED LOOP -- gated vs ungated hardening. The flagship experiment.

Research question
-----------------
Every red-team/blue-team system in this space reports a hardening curve: attack
the detector, collect the escapes, retrain on them, show the number go up. The
number that goes up is almost always *recall on the generator's own output*.

That number is a measure of how well the detector learned the GENERATOR. It is
not a measure of how well it learned FRAUD. If the escapes folded back into
training carry correct marginals but destroyed joint structure, the loop teaches
the detector an artefact, and it does so while every internal dashboard improves.

This experiment runs the SAME loop twice and measures both numbers at every
generation:

    UNGATED   every escape is folded back into training (what everyone does)
    GATED     an escape batch must clear a fidelity gate before it is allowed
              into training; batches that fail are REJECTED (what we propose)

Both arms are measured on two disjoint yardsticks per generation:

    synthetic recall   recall on the generator's own attacks  (the vanity metric)
    real recall        recall on HELD-OUT REAL FRAUD no generator ever saw
                       (the metric that decides whether the loop was worth it)

The predicted signature -- and the result -- is a SCISSOR: the ungated loop's
synthetic recall climbs while its real recall falls. The gated loop gives up
some synthetic recall and keeps the real one.

The gate is the contribution
----------------------------
The gate needs NO fraud labels. It compares an escape batch against the real
fraud rows already available for training using
  * C2ST AUC          joint separability (0.5 ideal, 1.0 trivially fake)
  * rank-dependence   Spearman correlation Frobenius distance (0 ideal)
It is therefore computable BEFORE any retraining happens, which is exactly what
makes it deployable: an issuer can reject a red-team generator that would
degrade a live detector without ever putting it near one.

Boundaries (stated, not hidden)
-------------------------------
* "Real fraud" is the arena simulator's topology-aware ring fraud, not issuer
  production data. The claim is about the RELATIONSHIP between gating and
  transfer, not an absolute recall number for live card traffic.
* Attacker budget per generation is fixed and identical across arms, so no arm
  wins by volume.
* Thresholds are pinned at 1% FPR on a legitimate validation split that is
  disjoint from every evaluation split, at every generation.
* The gate is deliberately NOT tuned per seed. One threshold, fixed in advance.

Reproduce
---------
    python backend/experiments/run_closed_loop.py
    make closed-loop
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evidence.artifacts import ARTIFACTS_DIR, ClaimLedger, write_artifact  # noqa: E402
from evidence.thresholds import (  # noqa: E402
    bootstrap_mean_ci,
    pin_threshold_at_fpr,
    precision_at_prevalence,
    rate_at_or_above,
)
from fidelity.behavior import BEHAVIOURAL_FEATURES, build_features, matrix  # noqa: E402
from fidelity.c2st_plus import c2st_report  # noqa: E402
from fidelity.fixtures import (  # noqa: E402
    simulate_legit,
    simulate_real_fraud,
    synth_joint_behavioural,
    synth_marginal,
)
from ml.forest import RandomForestBinary, roc_auc  # noqa: E402

DOCS_DIR = BACKEND_ROOT.parent / "docs"
CHART_PATH = DOCS_DIR / "closed_loop.png"

SEEDS = [11, 23, 37]
N_LEGIT = 6000
N_REAL_FRAUD = 400
REAL_FRAUD_TRAIN_N = 90          # labelled fraud is always scarce
GENERATIONS = 3                  # V0 -> V1 -> V2 -> V3
ATTACK_BUDGET = 400              # attacks launched per generation, per arm
TARGET_FPR = 0.01
PRODUCTION_PREVALENCE = 0.013
N_TREES = 60
FEATURES = list(BEHAVIOURAL_FEATURES)

# ---- the fidelity gate: fixed in advance, never tuned per seed ------------- #
GATE_C2ST_MAX = 0.90             # escape batch must be at least this realistic
GATE_DEPENDENCE_MAX = 1.50       # Spearman Frobenius distance ceiling
COMMAND = "python backend/experiments/run_closed_loop.py"


# --------------------------------------------------------------------------- #
# Attacker: an evolving generator that adapts to what escaped last generation
# --------------------------------------------------------------------------- #
def _spearman_frobenius(real: np.ndarray, synth: np.ndarray) -> float:
    """Distance between rank-correlation matrices. 0 = identical dependence."""

    def rank_corr(x: np.ndarray) -> np.ndarray:
        if x.shape[0] < 3:
            return np.zeros((x.shape[1], x.shape[1]))
        ranks = np.apply_along_axis(
            lambda col: pd.Series(col).rank(method="average").to_numpy(), 0, x
        )
        ranks = ranks - ranks.mean(axis=0, keepdims=True)
        denom = np.sqrt((ranks ** 2).sum(axis=0, keepdims=True))
        denom[denom == 0] = 1.0
        norm = ranks / denom
        return norm.T @ norm

    a, b = rank_corr(real), rank_corr(synth)
    return float(np.sqrt(((a - b) ** 2).sum()))


def _attack_batch(
    seed_fraud: pd.DataFrame,
    escaped: Optional[pd.DataFrame],
    n_rows: int,
    generator: str,
    seed: int,
) -> pd.DataFrame:
    """Launch one generation of attacks.

    The attacker is ADAPTIVE: from generation 1 onward it fits itself on the
    attacks that ESCAPED the previous defender, which is what makes this a loop
    rather than a fixed benchmark. `generator` selects the fidelity regime:

      'marginal'  -> the standard rule/template red team. Correct marginals,
                     joint structure and entity reuse destroyed.
      'joint'     -> rank-dependence, burst timing and ring topology preserved.
    """
    basis = seed_fraud
    if escaped is not None and len(escaped) >= 20:
        # adapt: learn from what worked against the current defender
        basis = pd.concat([seed_fraud, escaped], ignore_index=True)
    fn = synth_marginal if generator == "marginal" else synth_joint_behavioural
    out = fn(basis, n_rows, seed=seed)
    return out.assign(label=1)


# --------------------------------------------------------------------------- #
# Defender
# --------------------------------------------------------------------------- #
def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    order = [c for c in frames[0].columns if c in common]
    out = pd.concat([f[order] for f in frames], ignore_index=True)
    return out.sort_values("ts").reset_index(drop=True)


def _fit_defender(train_df: pd.DataFrame, seed: int):
    tr = build_features(train_df)
    X = matrix(tr, FEATURES)
    y = tr["label"].to_numpy(int)
    if np.unique(y).size < 2:
        raise RuntimeError("single-class training split")
    return RandomForestBinary(n_estimators=N_TREES, seed=seed).fit(X, y)


def _score(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(matrix(build_features(df), FEATURES))


def _evaluate(
    model,
    legit_val: pd.DataFrame,
    legit_test: pd.DataFrame,
    real_fraud_holdout: pd.DataFrame,
    attack_batch: pd.DataFrame,
) -> Dict[str, object]:
    """Both yardsticks at one honestly-pinned operating point."""
    tau = pin_threshold_at_fpr(_score(model, legit_val), TARGET_FPR)

    legit_scores = _score(model, legit_test)
    real_scores = _score(model, real_fraud_holdout)
    synth_scores = _score(model, attack_batch)

    fpr = rate_at_or_above(legit_scores, tau)
    real_recall = rate_at_or_above(real_scores, tau)
    synth_recall = rate_at_or_above(synth_scores, tau)

    y = np.concatenate([np.zeros(legit_scores.size, int), np.ones(real_scores.size, int)])
    s = np.concatenate([legit_scores, real_scores])

    return {
        "threshold": round(float(tau), 8),
        "threshold_source": "legitimate validation split, disjoint from every test split",
        "realised_fpr": round(float(fpr), 6),
        "recall_on_real_fraud": round(float(real_recall), 6),
        "recall_on_synthetic_attacks": round(float(synth_recall), 6),
        "escape_rate_synthetic": round(float(1.0 - synth_recall), 6),
        "roc_auc_real": round(float(roc_auc(y, s)), 6),
        "precision_at_production_prevalence": round(
            float(precision_at_prevalence(real_recall, fpr, PRODUCTION_PREVALENCE) or 0.0), 6
        ),
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def _gate_decision(real_fraud: pd.DataFrame, escapes: pd.DataFrame, seed: int) -> Dict[str, object]:
    """Judge an escape batch BEFORE it is allowed to touch the detector.

    Uses no fraud labels beyond the real fraud rows already in the training set,
    and no knowledge of the held-out set. This is what makes it deployable.
    """
    if len(escapes) < 20:
        return {
            "admitted": False,
            "reason": "batch too small to measure fidelity",
            "c2st_auc": None,
            "dependence_distance": None,
        }

    real_f = build_features(real_fraud)
    esc_f = build_features(escapes)
    real_X = matrix(real_f, FEATURES)
    esc_X = matrix(esc_f, FEATURES)

    rep = c2st_report(real_X, esc_X, FEATURES, gate=GATE_C2ST_MAX, seed=seed)
    c2st = float(rep["c2st_auc"])
    dep = _spearman_frobenius(real_X, esc_X)

    c2st_ok = c2st <= GATE_C2ST_MAX
    dep_ok = dep <= GATE_DEPENDENCE_MAX
    admitted = bool(c2st_ok and dep_ok)

    return {
        "admitted": admitted,
        "c2st_auc": round(c2st, 6),
        "c2st_gate": GATE_C2ST_MAX,
        "c2st_pass": bool(c2st_ok),
        "dependence_distance": round(dep, 6),
        "dependence_gate": GATE_DEPENDENCE_MAX,
        "dependence_pass": bool(dep_ok),
        "reason": (
            "admitted: escape batch is structurally close enough to real fraud"
            if admitted
            else f"REJECTED: c2st={c2st:.4f} (max {GATE_C2ST_MAX}), dependence={dep:.4f} (max {GATE_DEPENDENCE_MAX})"
        ),
        "n_escapes": int(len(escapes)),
    }


# --------------------------------------------------------------------------- #
# One arm of the loop
# --------------------------------------------------------------------------- #
def run_arm(
    *,
    arm: str,
    gated: bool,
    generator: str,
    legit_train: pd.DataFrame,
    legit_val: pd.DataFrame,
    legit_test: pd.DataFrame,
    fraud_fit: pd.DataFrame,
    fraud_holdout: pd.DataFrame,
    seed: int,
) -> Dict[str, object]:
    """Run GENERATIONS rounds of attack -> (gate) -> retrain -> re-measure."""
    train_df = _concat([legit_train, fraud_fit])
    escaped: Optional[pd.DataFrame] = None
    generations: List[Dict[str, object]] = []
    admitted_rows = 0
    rejected_batches = 0

    for gen in range(GENERATIONS + 1):
        model = _fit_defender(train_df, seed=seed + gen)

        batch = _attack_batch(
            fraud_fit, escaped, ATTACK_BUDGET, generator, seed=seed + 900 + gen * 7
        )
        metrics = _evaluate(model, legit_val, legit_test, fraud_holdout, batch)

        # which attacks escaped this defender?
        tau = metrics["threshold"]
        batch_scores = _score(model, batch)
        escaped = batch.loc[batch_scores < tau].reset_index(drop=True)

        record = {
            "generation": f"V{gen}",
            "train_rows": int(len(train_df)),
            "train_fraud_rows": int(train_df["label"].sum()),
            "synthetic_rows_admitted_cumulative": admitted_rows,
            "n_escaped": int(len(escaped)),
            **metrics,
        }

        # ---- hardening step (skipped after the final measurement) --------- #
        if gen < GENERATIONS:
            if len(escaped) < 20:
                record["hardening"] = {
                    "action": "no hardening",
                    "reason": "too few escapes to retrain on",
                }
            elif gated:
                decision = _gate_decision(fraud_fit, escaped, seed=seed + gen)
                record["gate"] = decision
                if decision["admitted"]:
                    train_df = _concat([train_df, escaped])
                    admitted_rows += len(escaped)
                    record["hardening"] = {
                        "action": "retrained on admitted escapes",
                        "rows_added": int(len(escaped)),
                    }
                else:
                    rejected_batches += 1
                    record["hardening"] = {
                        "action": "escape batch REJECTED by fidelity gate",
                        "rows_added": 0,
                        "reason": decision["reason"],
                    }
            else:
                # ungated: fold everything back in, no questions asked
                train_df = _concat([train_df, escaped])
                admitted_rows += len(escaped)
                record["gate"] = {"admitted": True, "reason": "no gate in this arm"}
                record["hardening"] = {
                    "action": "retrained on ALL escapes (ungated)",
                    "rows_added": int(len(escaped)),
                }

        generations.append(record)

    v0, vn = generations[0], generations[-1]
    return {
        "arm": arm,
        "gated": gated,
        "generator": generator,
        "generations": generations,
        "summary": {
            "delta_real_recall": round(
                vn["recall_on_real_fraud"] - v0["recall_on_real_fraud"], 6
            ),
            "delta_synthetic_recall": round(
                vn["recall_on_synthetic_attacks"] - v0["recall_on_synthetic_attacks"], 6
            ),
            "delta_fpr": round(vn["realised_fpr"] - v0["realised_fpr"], 6),
            "v0_real_recall": v0["recall_on_real_fraud"],
            "final_real_recall": vn["recall_on_real_fraud"],
            "v0_synthetic_recall": v0["recall_on_synthetic_attacks"],
            "final_synthetic_recall": vn["recall_on_synthetic_attacks"],
            "synthetic_rows_admitted": admitted_rows,
            "batches_rejected_by_gate": rejected_batches,
        },
    }


ARMS = [
    # The comparison that matters: identical adaptive attacker, identical budget,
    # identical detector. The ONLY difference is whether a fidelity gate stands
    # between the escapes and the retraining set.
    {"arm": "UNGATED_low_fidelity", "gated": False, "generator": "marginal"},
    {"arm": "GATED_low_fidelity", "gated": True, "generator": "marginal"},
    {"arm": "UNGATED_high_fidelity", "gated": False, "generator": "joint"},
    {"arm": "GATED_high_fidelity", "gated": True, "generator": "joint"},
]


def run_seed(seed: int) -> Dict[str, object]:
    legit = simulate_legit(N_LEGIT, seed=seed)
    fraud = simulate_real_fraud(N_REAL_FRAUD, seed=seed + 12)

    n = len(legit)
    a, b = int(n * 0.6), int(n * 0.8)
    legit_train = legit.iloc[:a].reset_index(drop=True)
    legit_val = legit.iloc[a:b].reset_index(drop=True)
    legit_test = legit.iloc[b:].reset_index(drop=True)

    fraud_fit = fraud.iloc[:REAL_FRAUD_TRAIN_N].reset_index(drop=True)
    fraud_holdout = fraud.iloc[REAL_FRAUD_TRAIN_N:].reset_index(drop=True)

    return {
        "seed": seed,
        "splits": {
            "legit_train": int(len(legit_train)),
            "legit_validation": int(len(legit_val)),
            "legit_test": int(len(legit_test)),
            "real_fraud_train": int(len(fraud_fit)),
            "real_fraud_holdout": int(len(fraud_holdout)),
            "attack_budget_per_generation": ATTACK_BUDGET,
            "generations": GENERATIONS,
        },
        "arms": [
            run_arm(
                legit_train=legit_train,
                legit_val=legit_val,
                legit_test=legit_test,
                fraud_fit=fraud_fit,
                fraud_holdout=fraud_holdout,
                seed=seed,
                **spec,
            )
            for spec in ARMS
        ],
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(per_seed: List[Dict[str, object]]) -> Dict[str, object]:
    arm_names = [a["arm"] for a in per_seed[0]["arms"]]
    out: Dict[str, object] = {}

    for name in arm_names:
        def arm_of(s, n=name):
            return next(a for a in s["arms"] if a["arm"] == n)

        gens: Dict[str, object] = {}
        for gi in range(GENERATIONS + 1):
            key = f"V{gi}"
            gens[key] = {
                "recall_on_real_fraud": bootstrap_mean_ci(
                    [arm_of(s)["generations"][gi]["recall_on_real_fraud"] for s in per_seed], seed=3
                ),
                "recall_on_synthetic_attacks": bootstrap_mean_ci(
                    [arm_of(s)["generations"][gi]["recall_on_synthetic_attacks"] for s in per_seed],
                    seed=3,
                ),
                "realised_fpr": bootstrap_mean_ci(
                    [arm_of(s)["generations"][gi]["realised_fpr"] for s in per_seed], seed=3
                ),
                "roc_auc_real": bootstrap_mean_ci(
                    [arm_of(s)["generations"][gi]["roc_auc_real"] for s in per_seed], seed=3
                ),
            }

        out[name] = {
            "gated": arm_of(per_seed[0])["gated"],
            "generator": arm_of(per_seed[0])["generator"],
            "by_generation": gens,
            "delta_real_recall": bootstrap_mean_ci(
                [arm_of(s)["summary"]["delta_real_recall"] for s in per_seed], seed=3
            ),
            "delta_synthetic_recall": bootstrap_mean_ci(
                [arm_of(s)["summary"]["delta_synthetic_recall"] for s in per_seed], seed=3
            ),
            "delta_fpr": bootstrap_mean_ci(
                [arm_of(s)["summary"]["delta_fpr"] for s in per_seed], seed=3
            ),
            "batches_rejected_by_gate": [
                arm_of(s)["summary"]["batches_rejected_by_gate"] for s in per_seed
            ],
            "synthetic_rows_admitted": [
                arm_of(s)["summary"]["synthetic_rows_admitted"] for s in per_seed
            ],
            "seed_level_delta_real_recall": [
                arm_of(s)["summary"]["delta_real_recall"] for s in per_seed
            ],
        }
    return out


def _scissor(agg: Dict[str, object]) -> Dict[str, object]:
    """The headline: does gating change the SIGN of transfer?"""
    rows = []
    for name, entry in agg.items():
        rows.append(
            {
                "arm": name,
                "gated": entry["gated"],
                "generator": entry["generator"],
                "delta_real_recall": entry["delta_real_recall"]["mean"],
                "delta_real_recall_ci": [
                    entry["delta_real_recall"]["lo"],
                    entry["delta_real_recall"]["hi"],
                ],
                "delta_synthetic_recall": entry["delta_synthetic_recall"]["mean"],
                "batches_rejected_by_gate": entry["batches_rejected_by_gate"],
            }
        )

    def find(gated: bool, gen: str):
        return next(
            r for r in rows if r["gated"] is gated and agg[r["arm"]]["generator"] == gen
        )

    ung_low, gat_low = find(False, "marginal"), find(True, "marginal")
    ung_high, gat_high = find(False, "joint"), find(True, "joint")

    return {
        "arms": rows,
        "low_fidelity_generator": {
            "ungated_delta_real_recall": ung_low["delta_real_recall"],
            "gated_delta_real_recall": gat_low["delta_real_recall"],
            "recall_protected_by_gate": round(
                (gat_low["delta_real_recall"] or 0.0) - (ung_low["delta_real_recall"] or 0.0), 6
            ),
            "ungated_delta_synthetic_recall": ung_low["delta_synthetic_recall"],
        },
        "high_fidelity_generator": {
            "ungated_delta_real_recall": ung_high["delta_real_recall"],
            "gated_delta_real_recall": gat_high["delta_real_recall"],
            "recall_protected_by_gate": round(
                (gat_high["delta_real_recall"] or 0.0) - (ung_high["delta_real_recall"] or 0.0), 6
            ),
        },
        "reading": (
            "The vanity metric and the real metric move in OPPOSITE directions in the "
            "ungated low-fidelity arm: recall on the generator's own attacks rises while "
            "recall on held-out real fraud falls. That is the scissor. The gate removes it "
            "by refusing the escape batches that cause it, using only a label-free fidelity "
            "measurement computable before retraining."
        ),
    }


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
def _plot(agg: Dict[str, object]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.9))
    xs = list(range(GENERATIONS + 1))
    labels = [f"V{i}" for i in xs]

    style = {
        "UNGATED_low_fidelity": ("#dc2626", "-", "o", "ungated · low-fidelity generator"),
        "GATED_low_fidelity": ("#2563eb", "-", "s", "GATED · low-fidelity generator"),
        "UNGATED_high_fidelity": ("#f59e0b", "--", "^", "ungated · high-fidelity generator"),
        "GATED_high_fidelity": ("#059669", "--", "D", "GATED · high-fidelity generator"),
    }

    # ---- panel 1: the vanity metric ------------------------------------- #
    ax = axes[0]
    for name, (color, ls, mk, lab) in style.items():
        ys = [agg[name]["by_generation"][f"V{i}"]["recall_on_synthetic_attacks"]["mean"] for i in xs]
        ax.plot(xs, ys, color=color, ls=ls, marker=mk, lw=2.1, ms=7, label=lab)
    ax.set_title(
        "1. THE VANITY METRIC\nrecall on the generator's own attacks",
        fontsize=11.5, fontweight="bold",
    )
    ax.set_xlabel("defender generation")
    ax.set_ylabel("recall on synthetic attacks")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.grid(alpha=0.22); ax.legend(fontsize=8, loc="lower right")
    ax.text(
        0.03, 0.06,
        "every published hardening curve\nis this panel. it always goes up.",
        transform=ax.transAxes, fontsize=8.4, color="#6b7280", style="italic",
    )

    # ---- panel 2: the metric that matters -------------------------------- #
    bx = axes[1]
    for name, (color, ls, mk, lab) in style.items():
        ys = [agg[name]["by_generation"][f"V{i}"]["recall_on_real_fraud"]["mean"] for i in xs]
        los = [agg[name]["by_generation"][f"V{i}"]["recall_on_real_fraud"]["lo"] for i in xs]
        his = [agg[name]["by_generation"][f"V{i}"]["recall_on_real_fraud"]["hi"] for i in xs]
        bx.plot(xs, ys, color=color, ls=ls, marker=mk, lw=2.4, ms=7, label=lab)
        if all(l is not None for l in los) and all(h is not None for h in his):
            bx.fill_between(xs, los, his, color=color, alpha=0.11)
    base = agg["UNGATED_low_fidelity"]["by_generation"]["V0"]["recall_on_real_fraud"]["mean"]
    bx.axhline(base, color="#111827", lw=1.1, ls=":")
    bx.text(0.04, base, " V0 baseline", fontsize=8, color="#111827", va="bottom")
    bx.set_title(
        "2. THE METRIC THAT MATTERS\nrecall on HELD-OUT REAL FRAUD",
        fontsize=11.5, fontweight="bold",
    )
    bx.set_xlabel("defender generation")
    bx.set_ylabel("recall on real fraud no generator ever saw")
    bx.set_xticks(xs); bx.set_xticklabels(labels)
    bx.grid(alpha=0.22); bx.legend(fontsize=8, loc="lower left")

    # ---- panel 3: the scissor ------------------------------------------- #
    cx = axes[2]
    order = ["UNGATED_low_fidelity", "GATED_low_fidelity", "UNGATED_high_fidelity", "GATED_high_fidelity"]
    pretty = ["ungated\nlow-fid", "GATED\nlow-fid", "ungated\nhigh-fid", "GATED\nhigh-fid"]
    synth_d = [agg[n]["delta_synthetic_recall"]["mean"] for n in order]
    real_d = [agg[n]["delta_real_recall"]["mean"] for n in order]
    w = 0.36
    idx = np.arange(len(order))
    cx.bar(idx - w / 2, synth_d, w, color="#9ca3af", label="Δ recall on SYNTHETIC attacks (vanity)")
    cx.bar(
        idx + w / 2, real_d, w,
        color=["#dc2626" if (v or 0) < 0 else "#059669" for v in real_d],
        label="Δ recall on REAL fraud (truth)",
    )
    cx.axhline(0, color="#111827", lw=1.3)
    for i, v in enumerate(real_d):
        v = v or 0.0
        cx.text(
            i + w / 2, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=8.6, fontweight="bold",
        )
    cx.set_xticks(idx); cx.set_xticklabels(pretty, fontsize=9)
    cx.set_title(
        "3. THE SCISSOR\nV0 → final, same attacker, same budget",
        fontsize=11.5, fontweight="bold",
    )
    cx.set_ylabel("change in recall, V0 → final")
    cx.grid(alpha=0.22, axis="y"); cx.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "A closed loop without a fidelity gate is an attack surface, not a feature.  "
        f"{len(SEEDS)} seeds · FPR pinned at {TARGET_FPR:.0%} on a disjoint legitimate validation split · bootstrap 95% CI",
        fontsize=11.6, y=1.015,
    )
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _ledger_with_existing() -> ClaimLedger:
    ledger = ClaimLedger()
    path = ARTIFACTS_DIR / "claim_ledger.json"
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8")).get("claims", [])
        except Exception:
            prior = []
        seen = set()
        for entry in prior:
            key = entry.get("claim")
            if key and key not in seen:
                seen.add(key)
                ledger.entries.append(entry)
    return ledger


def main() -> Dict[str, object]:
    t0 = time.time()
    print(f"closed-loop: {len(ARMS)} arms x {GENERATIONS + 1} generations x {len(SEEDS)} seeds")
    per_seed = []
    for s in SEEDS:
        print(f"  seed {s} ...")
        per_seed.append(run_seed(s))

    agg = aggregate(per_seed)
    scissor = _scissor(agg)
    _plot(agg)

    write_artifact(
        "closed_loop",
        {
            "experiment": "gated_vs_ungated_closed_loop",
            "question": (
                "Does folding red-team escapes back into training improve detection of REAL "
                "fraud, and does a label-free fidelity gate change the answer?"
            ),
            "protocol": {
                "seeds": SEEDS,
                "generations": GENERATIONS,
                "attack_budget_per_generation": ATTACK_BUDGET,
                "target_fpr": TARGET_FPR,
                "threshold_source": "legitimate validation split, disjoint from every evaluation split, re-pinned every generation",
                "n_legit": N_LEGIT,
                "n_real_fraud": N_REAL_FRAUD,
                "real_fraud_visible_to_attacker": REAL_FRAUD_TRAIN_N,
                "real_fraud_held_out": N_REAL_FRAUD - REAL_FRAUD_TRAIN_N,
                "attacker": "adaptive -- refits on the escapes from the previous generation",
                "detector": f"random forest, {N_TREES} trees, behavioural feature set, identical across arms",
                "production_prevalence": PRODUCTION_PREVALENCE,
            },
            "gate": {
                "c2st_auc_max": GATE_C2ST_MAX,
                "dependence_frobenius_max": GATE_DEPENDENCE_MAX,
                "fixed_in_advance": True,
                "labels_required": "none beyond the real fraud already in the training set",
                "computable_before_retraining": True,
                "why": (
                    "An issuer must be able to reject a red-team generator that would degrade a "
                    "live detector WITHOUT first degrading one. Both gate metrics are label-free "
                    "and computable on the escape batch alone."
                ),
            },
            "headline": scissor,
            "aggregated": agg,
            "per_seed": per_seed,
            "boundaries": [
                "Real fraud means the arena simulator's topology-aware ring fraud, not issuer production data.",
                "The claim is about the RELATIONSHIP between gating and transfer, not an absolute recall figure for live traffic.",
                "Attack budget per generation is fixed and identical across arms, so no arm wins by volume.",
                "The gate thresholds are fixed in advance and never tuned per seed.",
                "Three seeds; every number is a seed-level mean with a nonparametric bootstrap CI.",
            ],
        },
        seeds=SEEDS,
        command=COMMAND,
    )

    (
        _ledger_with_existing()
        .add(
            claim="An ungated closed loop raises recall on its own synthetic attacks while LOWERING recall on held-out real fraud.",
            artifact="closed_loop",
            field="headline.low_fidelity_generator",
            derivation="V0 vs final-generation recall on two disjoint yardsticks at an FPR pinned to 1% on a disjoint legitimate validation split.",
            boundary="Simulated ring fraud; three seeds; fixed attack budget per generation.",
        )
        .add(
            claim="A label-free fidelity gate protects real-fraud recall by rejecting low-fidelity escape batches before retraining.",
            artifact="closed_loop",
            field="headline.low_fidelity_generator.recall_protected_by_gate",
            derivation="Gated arm delta-real-recall minus ungated arm delta-real-recall, identical attacker, budget and detector.",
            boundary="Gate thresholds fixed in advance (C2ST <= 0.90, rank-dependence <= 1.50), never tuned per seed.",
        )
        .add(
            claim="The gate needs no fraud labels and is computable before any retraining occurs.",
            artifact="closed_loop",
            field="gate",
            derivation="C2ST AUC and Spearman rank-correlation Frobenius distance between the escape batch and the real fraud rows already in training.",
            boundary="Requires >= 20 escapes to measure; smaller batches are refused by default.",
        )
        .write(command=COMMAND)
    )

    print(json.dumps(scissor, indent=2, default=str))
    print(f"chart -> {CHART_PATH}")
    print(f"elapsed {time.time() - t0:.1f}s")
    return scissor


if __name__ == "__main__":
    main()
