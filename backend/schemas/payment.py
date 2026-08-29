"""
Payment message schema — the canonical transaction payload for the arena.

Every message that enters the simulated issuer stack MUST validate against
`PaymentMessage`. This is the Pydantic v2 "contract" that OpenRouter structured
outputs will be forced into later (Step 3): if the LLM attacker agent emits
free-text JSON, it dies here at the boundary, never inside the simulation.

Field names intentionally mirror simplified ISO 8583 / ISO 20022 concepts.
See docs/ISO_MAPPING.md (later step) for the field-by-field mapping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    """Timezone-aware 'now' — naive datetimes are a footgun in event loops."""
    return datetime.now(timezone.utc)


class PosEntryMode(str, Enum):
    """How the card/payment instrument was presented at the point of interaction.

    Maps loosely to ISO 8583 DE-22 (POS Entry Mode):
      - ECOM:       e-commerce / card-not-present over the internet (81/07)
      - CONTACTLESS: tap-to-pay, card-present via NFC (05/07)
      - CNP:        manual key-entry card-not-present (phone order, etc.)
      - CHIP:       EMV chip dip, card-present (05)
      - SWIPE:      magnetic stripe, card-present (00/01) — legacy
    """

    ECOM = "ECOM"
    CONTACTLESS = "CONTACTLESS"
    CNP = "CNP"
    CHIP = "CHIP"
    SWIPE = "SWIPE"


class ThreeDSStatus(str, Enum):
    """3-D Secure authentication result carried on the authorization.

    Maps to the 3DS ecosystem fields in ISO 8583 (DE-44-ish / 20022 PA.87):
      - Y: cardholder fully authenticated (frictionless or challenge passed)
      - A: attempted / ACS not available — issuer accepted liability shift attempt
      - N: no 3DS performed (normal for card-present rails)
    """

    Y = "Y"
    A = "A"
    N = "N"


class StolenResourceType(str, Enum):
    """
    The acquisition vector behind an attack transaction.

    Legitimate traffic NEVER sets this field — only the attacker agent does.
    The Plausibility Gate uses it for ECONOMIC VIABILITY checks: a fraudster
    who paid $200 for a synthetic identity will not burn it on a $4 coffee.
    """

    CLONED_CARD = "cloned_card"                    # magstripe clone / shimmer dump (~$25)
    PHISHED_CREDENTIALS = "phished_credentials"    # phish-kit harvested login (~$8)
    CLONED_VOICE = "cloned_voice"                  # voice-clone for IVR takeover (~$50)
    OTP_INTERCEPT = "otp_intercept"                # SIM-swap / relay OTP (~$60)
    SYNTHETIC_IDENTITY = "synthetic_identity"      # full synthetic ID kit (~$200)
    FULLZ = "fullz"                                # name+SSN+card DOB bundle (~$30)
    STOLEN_DEVICE = "stolen_device"                # physical device w/ session (~$300)
    COMPROMISED_MERCHANT = "compromised_merchant"  # injected/pwned merchant endpoint (~$500)
    # A *real* account, KYC-clean, rented for a rotation window from a recruited
    # holder (student, gig worker, "commission work" victim). This is NOT a
    # synthetic identity: nothing was fabricated and no ID kit was purchased,
    # which is exactly why it is cheap and why the economic floor is low. It is
    # the standard cash-out layer on real-time rails (UPI VPA rental).
    RENTED_ACCOUNT = "rented_account"              # rented KYC-clean payee (~$45/window)


# Estimated street cost (USD) of acquiring each stolen resource. Used by the
# Plausibility Gate's ECONOMIC VIABILITY check: amount < cost => implausible.
RESOURCE_COST_TABLE_USD: dict[StolenResourceType, float] = {
    StolenResourceType.CLONED_CARD: 25.00,
    StolenResourceType.PHISHED_CREDENTIALS: 8.00,
    StolenResourceType.CLONED_VOICE: 50.00,
    StolenResourceType.OTP_INTERCEPT: 60.00,
    StolenResourceType.SYNTHETIC_IDENTITY: 200.00,
    StolenResourceType.FULLZ: 30.00,
    StolenResourceType.STOLEN_DEVICE: 300.00,
    StolenResourceType.COMPROMISED_MERCHANT: 500.00,
    StolenResourceType.RENTED_ACCOUNT: 45.00,
}


class PaymentMessage(BaseModel):
    """
    One authorization request flowing through the simulated payment stack.

    `model_config.extra="forbid"` is deliberate: any field the LLM invents
    that isn't part of the contract causes a validation error instead of
    silently passing junk into the defense stack.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # ---- Identity -----------------------------------------------------------
    transaction_id: str = Field(
        default_factory=lambda: uuid4().hex[:16].upper(),
        description="Unique authorization identifier (STAN-like).",
    )
    customer_id: str = Field(description="Issuer customer identifier, e.g. CUST_0421.")

    # ---- Merchant -----------------------------------------------------------
    merchant_id: str = Field(description="Registered merchant identifier, e.g. MERCH_ELEC_BESTBUY.")
    mcc: int = Field(ge=0, le=9999, description="Merchant Category Code (ISO 8583 DE-18).")

    # ---- Money --------------------------------------------------------------
    amount: float = Field(gt=0, description="Transaction amount in major units.")
    currency: str = Field(default="USD", min_length=3, max_length=3, description="ISO 4217 alpha-3.")

    # ---- Rail / entry metadata ----------------------------------------------
    pos_entry_mode: PosEntryMode = Field(description="Presentation mode (DE-22 analogue).")
    three_ds_status: ThreeDSStatus = Field(
        alias="3ds_status",
        description="3-D Secure result: Y / A / N.",
    )
    ip_address: str = Field(description="Client IP as seen by the acquirer (ECOM/CNP only in reality).")
    ip_country: str = Field(
        min_length=2,
        max_length=2,
        description="GeoIP-resolved country of ip_address (simulated enrichment; real stacks use MMDB lookups).",
    )
    device_id: str = Field(min_length=6, description="Device fingerprint bound (or not) to the customer.")

    # ---- Attack context (attacker-only) --------------------------------------
    stolen_resource: Optional[StolenResourceType] = Field(
        default=None,
        description=(
            "Acquisition vector used by the attacker agent. None for legitimate "
            "traffic. Drives the gate's economic-viability floor."
        ),
    )

    timestamp: datetime = Field(default_factory=_utcnow, description="Event time (UTC).")

    @field_validator("ip_address")
    @classmethod
    def _sane_ip(cls, v: str) -> str:
        # Cheap structural check — full inet_ntop parsing is overkill for sim.
        parts = v.split(".")
        if len(parts) != 4:
            raise ValueError(f"ip_address must be dotted quad, got {v!r}")
        return v

    def to_wire(self) -> dict:
        """Serialize using wire aliases ('3ds_status'), for JSONL + WebSocket."""
        return self.model_dump(by_alias=True, mode="json")


__all__ = [
    "PaymentMessage",
    "PosEntryMode",
    "ThreeDSStatus",
    "StolenResourceType",
    "RESOURCE_COST_TABLE_USD",
]
