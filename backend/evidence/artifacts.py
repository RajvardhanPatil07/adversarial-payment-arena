"""
Artifact writing and the claim-to-artifact ledger.

Every headline number this project reports must be traceable to a file on
disk that a judge can open, plus the exact command that regenerated it. A
claim without an artifact is an opinion.

Layout produced under `artifacts/`:

    artifacts/
      calibration_audit.json     operating points, thresholds, split sizes
      fidelity_report.json       per-generator distribution diagnostics
      transfer_ledger.json       the three-arm transfer experiment
      prevalence_metrics.json    base-rate adjusted reporting
      economics.json             INR business impact incl. insult cost
      claim_ledger.json          claim -> artifact -> field -> derivation
      metrics.json               flat summary of every headline number
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def provenance(seeds: list[int] | None = None, command: str | None = None) -> dict:
    """Reproduction metadata stamped onto every artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seeds": seeds or [],
        "command": command or "",
    }


def write_artifact(
    name: str,
    payload: dict[str, Any],
    seeds: list[int] | None = None,
    command: str | None = None,
) -> Path:
    """Write `artifacts/<name>.json` with provenance attached."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.json"
    document = {"provenance": provenance(seeds=seeds, command=command), **payload}
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return path


def read_artifact(name: str) -> dict:
    path = ARTIFACTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing artifact {path}. Run `make reproduce` to regenerate the evidence set."
        )
    return json.loads(path.read_text(encoding="utf-8"))


class ClaimLedger:
    """Maps every public claim to the artifact field that supports it, and to
    the boundary condition under which it holds.

    The boundary is mandatory. A claim shipped without its limits is the thing
    that gets a fraud system deployed badly.
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(
        self,
        claim: str,
        artifact: str,
        field: str,
        derivation: str,
        boundary: str,
    ) -> "ClaimLedger":
        entry = {
            "claim": claim,
            "artifact": f"artifacts/{artifact}.json",
            "field": field,
            "derivation": derivation,
            "boundary": boundary,
        }
        for i, existing in enumerate(self.entries):
            if existing.get("claim") == claim:
                self.entries[i] = entry
                return self
        self.entries.append(entry)
        return self

    def write(self, command: str | None = None) -> Path:
        return write_artifact(
            "claim_ledger",
            {"claim_count": len(self.entries), "claims": self.entries},
            command=command,
        )


def write_metrics_summary(summary: dict, seeds: list[int] | None = None, command: str | None = None) -> Path:
    """Flat, machine-readable summary of headline numbers for CI and the UI."""
    return write_artifact("metrics", summary, seeds=seeds, command=command)


__all__ = [
    "ARTIFACTS_DIR",
    "ClaimLedger",
    "SCHEMA_VERSION",
    "provenance",
    "read_artifact",
    "write_artifact",
    "write_metrics_summary",
]
