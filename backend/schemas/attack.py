"""
Attack spec schema — the taxonomy layer of the arena.

An AttackSpec is a *campaign definition*, not a script: it declares who the
attacker pretends to be, what they paid for their tooling, what the payoff
model looks like, and the operational envelope they operate in. The attacker
LLM reasons over this spec; the Plausibility Gate independently enforces the
physics of the world. Specs are authored as YAML and validated into this
Pydantic model on load — same contract discipline as PaymentMessage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from schemas.payment import PosEntryMode, StolenResourceType, ThreeDSStatus


class EconomicModel(BaseModel):
    """The fraudster's P&L. Drives the gate's economic-viability floor and
    gives the LLM a rational-agent objective (breakeven math) instead of
    cartoonish 'steal everything' behavior."""

    model_config = ConfigDict(extra="forbid")

    acquisition_cost_usd: float = Field(gt=0, description="Street cost of tools/stolen resources.")
    expected_payoff_usd: float = Field(gt=0, description="Expected gross extractable value per campaign.")
    breakeven_txns: int = Field(gt=0, description="Successful txns needed to recoup cost.")


class OperationalConstraints(BaseModel):
    """The operational envelope the agent must stay inside. These mirror what
    the gate checks — but we do NOT hand the gate's verdict logic to the LLM;
    it learns from rejection feedback like a real adversary would."""

    model_config = ConfigDict(extra="forbid")

    stolen_resource: Optional[StolenResourceType] = Field(
        default=None, description="Acquisition vector claimed on generated payloads."
    )
    pos_entry_modes: list[PosEntryMode] = Field(description="Rails the campaign uses.")
    preferred_three_ds: list[ThreeDSStatus] = Field(description="3DS outcomes the agent aims for.")
    target_verticals: list[str] = Field(
        default_factory=list,
        description="Merchant categories in scope (must exist in the env registry).",
    )
    min_amount_usd: float = Field(gt=0)
    max_amount_usd: float = Field(gt=0)

    @property
    def amount_band(self) -> tuple[float, float]:
        return self.min_amount_usd, self.max_amount_usd


class AttackSpec(BaseModel):
    """Root taxonomy object loaded from backend/attack_specs/*.yaml."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^ATTACK_[A-Z0-9_]+$")
    attack_name: str
    taxon: str = Field(description="Taxonomy class, e.g. account_takeover | synthetic | cnp_velocity.")
    description: str
    preconditions: str = Field(description="What the fraudster must already possess.")
    economic_model: EconomicModel
    constraints: OperationalConstraints
    evasion_notes: str = Field(default="", description="Known issuer screens the agent should reason about.")


def load_attack_spec(path: str | Path) -> AttackSpec:
    """Load + validate one attack YAML. Raises pydantic.ValidationError on drift."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AttackSpec.model_validate(raw)


__all__ = ["AttackSpec", "EconomicModel", "OperationalConstraints", "load_attack_spec"]
