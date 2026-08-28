"""
Evidence API -- serves the artifact set to the UI.

Artifacts are generated offline by experiment scripts; requests never recompute
metrics. The allow-list prevents path traversal and exposes the exact command
used to reproduce each artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_ROOT.parent / "artifacts"

router = APIRouter(prefix="/api/evidence", tags=["evidence"])

KNOWN_ARTIFACTS = {
    "metrics": "Flat summary of every headline number.",
    "transfer_ledger": "Three-arm transfer ablation: does generator fidelity determine real-fraud transfer?",
    "fidelity_report": "Five fidelity measures per generator, including the ones that fail.",
    "calibration_audit": "Threshold provenance and the validation-to-test calibration gap.",
    "prevalence_metrics": "The same operating point reported across plausible fraud base rates.",
    "economics": "INR business impact including the insult cost of false positives.",
    "behavioural_fidelity": "Held-out row, temporal and graph fidelity for the lightweight generators.",
    "privacy_audit": "Membership, duplication and attribute-inference audit of synthetic fraud.",
    "action_policy": "Validation-selected four-action policy evaluated once on held-out test traffic.",
    "claim_ledger": "Every public claim mapped to artifact, field, derivation and boundary.",
}

REPRODUCE_COMMANDS = {
    "calibration_audit": "python backend/experiments/run_calibration_audit.py",
    "fidelity_report": "python backend/experiments/run_fidelity.py",
    "transfer_ledger": "python backend/experiments/run_transfer_ablation.py",
    "prevalence_metrics": "python backend/experiments/run_transfer_ablation.py",
    "economics": "python backend/experiments/run_transfer_ablation.py",
    "metrics": "python backend/experiments/run_transfer_ablation.py",
    "behavioural_fidelity": "python backend/experiments/run_behavioural_fidelity.py",
    "privacy_audit": "python backend/experiments/run_privacy_audit.py",
    "action_policy": "python backend/experiments/run_action_policy.py",
    "claim_ledger": "make reproduce",
    "all": "make reproduce",
}


def _load(name: str) -> dict:
    path = ARTIFACTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"artifact '{name}' has not been generated yet",
                "reproduce": REPRODUCE_COMMANDS.get(name, REPRODUCE_COMMANDS["all"]),
            },
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"artifact '{name}' is not valid JSON: {exc}")


@router.get("/index")
async def list_artifacts() -> dict:
    entries = []
    for name, description in KNOWN_ARTIFACTS.items():
        path = ARTIFACTS_DIR / f"{name}.json"
        exists = path.exists()
        generated_at = None
        if exists:
            try:
                generated_at = (
                    json.loads(path.read_text(encoding="utf-8"))
                    .get("provenance", {})
                    .get("generated_at")
                )
            except Exception:
                generated_at = None
        entries.append(
            {
                "name": name,
                "description": description,
                "available": exists,
                "generated_at": generated_at,
                "reproduce": REPRODUCE_COMMANDS.get(name, REPRODUCE_COMMANDS["all"]),
            }
        )
    available = sum(1 for e in entries if e["available"])
    return {
        "artifact_count": len(entries),
        "available_count": available,
        "complete": available == len(entries),
        "reproduce_all": REPRODUCE_COMMANDS["all"],
        "artifacts": entries,
    }


@router.get("/summary")
async def summary() -> dict:
    metrics = _load("metrics")
    headline = metrics.get("headline", {})
    return {
        "provenance": metrics.get("provenance", {}),
        "pinned_fpr": headline.get("pinned_fpr"),
        "seeds": headline.get("seeds", []),
        "baseline_recall": headline.get("baseline_recall"),
        "delta_recall_independent_marginal": headline.get("delta_recall_independent_marginal"),
        "delta_recall_gaussian_copula": headline.get("delta_recall_gaussian_copula"),
        "c2st_independent_marginal": headline.get("c2st_independent_marginal"),
        "c2st_gaussian_copula": headline.get("c2st_gaussian_copula"),
        "precision_at_production_prevalence": metrics.get("precision_at_production_prevalence"),
        "net_benefit_inr_at_production_prevalence": metrics.get(
            "net_benefit_inr_at_production_prevalence"
        ),
        "insult_share_of_total_cost": metrics.get("insult_share_of_total_cost"),
        "thesis": (
            "Closing the red-team loop is not sufficient. Whether the loop improves "
            "real-world detection depends on the fidelity of the attack generator, and "
            "low-fidelity augmentation can measurably reduce recall on real fraud."
        ),
    }


@router.get("/claims")
async def claims() -> dict:
    return _load("claim_ledger")


@router.get("/artifact/{name}")
async def artifact(name: str) -> dict:
    if name not in KNOWN_ARTIFACTS:
        raise HTTPException(
            status_code=404,
            detail={"error": f"unknown artifact '{name}'", "known": sorted(KNOWN_ARTIFACTS)},
        )
    return _load(name)


__all__ = ["KNOWN_ARTIFACTS", "REPRODUCE_COMMANDS", "router"]
