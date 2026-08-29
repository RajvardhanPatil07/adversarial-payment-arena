"""
Evidence API -- serves the artifact set to the UI.

Additive router. It reads only files under `artifacts/` that were produced by
the experiment scripts; it never computes metrics on request. That separation
is deliberate: numbers shown in the UI are the same numbers a judge can open in
a JSON file and regenerate with a published command.

Mount:
    from api.evidence import router as evidence_router
    app.include_router(evidence_router)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_ROOT.parent / "artifacts"

router = APIRouter(prefix="/api/evidence", tags=["evidence"])

# Only these names are servable. Prevents path traversal and accidental
# exposure of anything else that lands in the artifacts directory.
KNOWN_ARTIFACTS = {
    "metrics": "Flat summary of every headline number.",
    "transfer_ledger": "Three-arm transfer ablation: does generator fidelity determine real-fraud transfer?",
    "fidelity_report": "Five fidelity measures per generator, including the ones that fail.",
    "calibration_audit": "Threshold provenance and the validation-to-test calibration gap.",
    "prevalence_metrics": "The same operating point reported across plausible fraud base rates.",
    "economics": "INR business impact including the insult cost of false positives.",
    "claim_ledger": "Every public claim mapped to artifact, field, derivation and boundary.",
    "closed_loop": "HEADLINE: gated vs ungated retraining loops. Shows the fidelity scissor -- "
                   "an ungated loop climbs on its own synthetic attacks while falling on real fraud.",
    "family_coverage": "Per-family detection recall, leave-one-family-out zero-day generalisation, "
                       "and which defense layer fires for each attack family.",
    "latency": "Measured inline decision latency percentiles against the 100ms authorisation budget.",
}

REPRODUCE_COMMANDS = {
    "calibration_audit": "python backend/experiments/run_calibration_audit.py",
    "fidelity_report": "python backend/experiments/run_fidelity.py",
    "transfer_ledger": "python backend/experiments/run_transfer_ablation.py",
    "prevalence_metrics": "python backend/experiments/run_transfer_ablation.py",
    "economics": "python backend/experiments/run_transfer_ablation.py",
    "metrics": "python backend/experiments/run_transfer_ablation.py",
    "claim_ledger": "python backend/experiments/run_transfer_ablation.py",
    "closed_loop": "python backend/experiments/run_closed_loop.py",
    "family_coverage": "python backend/experiments/run_family_coverage.py",
    "latency": "python backend/experiments/run_latency.py",
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
    """Which artifacts exist, what each one proves, and how to regenerate it."""
    entries = []
    for name, description in KNOWN_ARTIFACTS.items():
        path = ARTIFACTS_DIR / f"{name}.json"
        exists = path.exists()
        generated_at = None
        if exists:
            try:
                generated_at = json.loads(path.read_text(encoding="utf-8")).get(
                    "provenance", {}
                ).get("generated_at")
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
    """The four numbers that matter, plus their boundary conditions.

    Returns 404 with a reproduce command rather than fabricating placeholders
    when the evidence set has not been generated.
    """
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
    """The claim ledger: claim, supporting artifact field, derivation, boundary."""
    return _load("claim_ledger")


@router.get("/artifact/{name}")
async def artifact(name: str) -> dict:
    """Raw artifact JSON for one allow-listed artifact name."""
    if name not in KNOWN_ARTIFACTS:
        raise HTTPException(
            status_code=404,
            detail={"error": f"unknown artifact '{name}'", "known": sorted(KNOWN_ARTIFACTS)},
        )
    return _load(name)


__all__ = ["KNOWN_ARTIFACTS", "REPRODUCE_COMMANDS", "router"]
