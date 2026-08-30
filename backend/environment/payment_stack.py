"""
Simulated issuer payment stack + Plausibility Gate.

The `PaymentEnvironment` is the "world" of the arena:
  * it owns all state (merchant registry, customer profiles, device bindings,
    transaction ledgers, balances),
  * it is the ONLY entry point for authorization traffic (`ingest`),
  * and it runs the three-check Plausibility Gate that models how a real
    fraudster behaves: attacks must be economically viable, operationally
    realistic, and rail-feasible — otherwise the campaign is a waste of the
    fraudster's money, so we refuse to even simulate them.

Gate checks (fail-fast, first failure wins):
  1. ECONOMIC VIABILITY  -> reason code "economic_infeasible"
     amount < street cost of the claimed stolen resource.
  2. REALISM CRITIC      -> reason code "metadata_incoherent"
     MCC / IP-geo / device metadata must cohere with the merchant registry.
  3. RAIL FEASIBILITY    -> reason code "rail_infeasible"
     pos_entry_mode x 3ds_status pairings must be physically possible.

NOTE ON SCOPE: the gate models *attacker operational realism*, not fraud
risk. Whether an accepted message IS fraud is the defense stack's job
(Steps later). Keeping these two concerns separate is what lets us measure
FPR against the FP budget honestly.
"""

from __future__ import annotations

import random
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

from faker import Faker
from pydantic import BaseModel, Field

from schemas.payment import (
    RESOURCE_COST_TABLE_USD,
    PaymentMessage,
    PosEntryMode,
    StolenResourceType,
    ThreeDSStatus,
)

# ---------------------------------------------------------------------------
# Reason codes returned by the Plausibility Gate
# ---------------------------------------------------------------------------

REASON_OK = "ok"                          # passed all gate checks
REASON_ECONOMIC = "economic_infeasible"   # check 1 failed
REASON_METADATA = "metadata_incoherent"   # check 2 failed
REASON_RAIL = "rail_infeasible"           # check 3 failed


# ---------------------------------------------------------------------------
# Registry entities
# ---------------------------------------------------------------------------


class Merchant(BaseModel):
    """A merchant registered with the acquirer. The REALISM CRITIC checks
    incoming payloads against this ground truth."""

    merchant_id: str
    name: str
    mcc: int = Field(description="Merchant Category Code (DE-18).")
    country: str = Field(default="US")
    city: str
    category: str = Field(description="Coarse vertical: grocery|electronics|travel|ecommerce.")
    is_online: bool = Field(description="Card-not-present-capable storefront?")


class CustomerProfile(BaseModel):
    """Issuer-side customer record. Device bindings are the anchor for
    'known good device' reasoning in the defense layer."""

    customer_id: str
    name: str
    email: str
    country: str = Field(default="US")
    city: str
    state: str
    balance: float = Field(ge=0, description="Current available balance (USD).")
    devices: list[str] = Field(default_factory=list, description="Bound device fingerprints.")
    kyc_tier: str = Field(default="standard", description="kyc_light|standard|enhanced.")
    created_at: datetime


# ---------------------------------------------------------------------------
# Seed data builders
# ---------------------------------------------------------------------------

# 20 mock merchants across four verticals. MCCs are real-ish codes:
#   5411/5451/5499 grocery, 5732/5045 electronics, 3005 airline, 7011 hotel,
#   3351 car rental, 4722 travel agency, 5947 gift shop, 5815 digital goods,
#   5968 subscription/continuity, 5812 restaurants/delivery.
_SEED_MERCHANTS: list[dict] = [
    # --- Grocery (physical) ---
    dict(merchant_id="MERCH_GROC_SAFewayX", name="SafeWay X Markets", mcc=5411, city="San Francisco", category="grocery", is_online=False),
    dict(merchant_id="MERCH_GROC_WHOLEBASKET", name="WholeBasket Grocers", mcc=5411, city="Austin", category="grocery", is_online=False),
    dict(merchant_id="MERCH_GROC_FRESHDEPOT", name="Fresh Depot Foods", mcc=5451, city="Chicago", category="grocery", is_online=False),
    dict(merchant_id="MERCH_GROC_CORNERBODEGA", name="Corner Bodega 24h", mcc=5499, city="New York", category="grocery", is_online=False),
    dict(merchant_id="MERCH_GROC_COSTCLUB", name="CostClub Wholesale", mcc=5411, city="Seattle", category="grocery", is_online=False),
    # --- Electronics (mostly physical) ---
    dict(merchant_id="MERCH_ELEC_BESTBUYX", name="BestBuyX Electronics", mcc=5732, city="Minneapolis", category="electronics", is_online=False),
    dict(merchant_id="MERCH_ELEC_NEWEGGX", name="NewEggX Components", mcc=5045, city="Los Angeles", category="electronics", is_online=True),
    dict(merchant_id="MERCH_ELEC_MICROCENTERX", name="MicroCenterX", mcc=5732, city="Columbus", category="electronics", is_online=False),
    dict(merchant_id="MERCH_ELEC_GADGETHUB", name="GadgetHub Online", mcc=5999, city="Denver", category="electronics", is_online=True),
    # --- Travel ---
    dict(merchant_id="MERCH_TRAV_DELTAAIR", name="DeltaAir Booking", mcc=3005, city="Atlanta", category="travel", is_online=True),
    dict(merchant_id="MERCH_TRAV_JETAIR", name="JetAir Reservations", mcc=3058, city="Boston", category="travel", is_online=True),
    dict(merchant_id="MERCH_TRAV_MARRIOTSTAY", name="MarriottStay Hotels", mcc=7011, city="Bethesda", category="travel", is_online=True),
    dict(merchant_id="MERCH_TRAV_HERTZGO", name="HertzGo Car Rental", mcc=3351, city="Miami", category="travel", is_online=False),
    dict(merchant_id="MERCH_TRAV_EXPEDIX", name="Expedix Travel Agency", mcc=4722, city="Phoenix", category="travel", is_online=True),
    # --- E-commerce / digital ---
    dict(merchant_id="MERCH_ECOM_SHOPIFYX", name="ShopifyX Marketplace", mcc=5968, city="Ottawa", country="CA", category="ecommerce", is_online=True),
    dict(merchant_id="MERCH_ECOM_ETSYARTS", name="EtsyArts Gifts", mcc=5947, city="Brooklyn", category="ecommerce", is_online=True),
    dict(merchant_id="MERCH_DIGI_SPOTIFLY", name="Spotifly Subscriptions", mcc=5815, city="Stockholm", country="SE", category="ecommerce", is_online=True),
    dict(merchant_id="MERCH_DIGI_STEAMPLAY", name="SteamPlay Games", mcc=5815, city="Bellevue", category="ecommerce", is_online=True),
    dict(merchant_id="MERCH_FOOD_DOORRUNNER", name="DoorRunner Delivery", mcc=5812, city="San Francisco", category="ecommerce", is_online=True),
    dict(merchant_id="MERCH_ECOM_ZONMARKET", name="ZonMarket Online", mcc=5942, city="Las Vegas", category="ecommerce", is_online=True),
]


def build_merchant_registry() -> dict[str, Merchant]:
    """Return merchant_id -> Merchant for the 20 seeded merchants."""
    return {m["merchant_id"]: Merchant(**m) for m in _SEED_MERCHANTS}


def build_customer_profiles(count: int = 1000, seed: int = 42) -> dict[str, CustomerProfile]:
    """Generate `count` synthetic-but-plausible customer profiles via Faker.

    Each customer binds 1-3 devices (the defense layer's graph will connect
    customers <-> devices <-> merchants; shared devices between two profiles
    would be a ring signal — we deliberately do NOT share them at seed time).

    `created_at` is anchored to a FIXED date (not datetime.now) so a seeded
    registry is identical across processes — the same reason corpus_builder
    pins _ATTACK_BASE. It is metadata only (no feature reads it today), but
    pinning it keeps the environment free of any wall-clock read.
    """
    fake = Faker("en_US")
    Faker.seed(seed)
    rng = random.Random(seed)

    profiles: dict[str, CustomerProfile] = {}
    now = datetime(2026, 6, 2, 0, 0, 0, tzinfo=timezone.utc)  # fixed anchor; see docstring
    for i in range(count):
        cid = f"CUST_{i:04d}"
        n_devices = rng.choices([1, 2, 3], weights=[55, 33, 12])[0]
        profiles[cid] = CustomerProfile(
            customer_id=cid,
            name=fake.unique.name(),
            email=fake.unique.email(),
            city=fake.city(),
            state=fake.state_abbr(),
            balance=round(rng.uniform(500.00, 25_000.00), 2),
            devices=[f"DEV_{uuid_hex(rng)}" for _ in range(n_devices)],
            kyc_tier=rng.choice(["kyc_light", "standard", "standard", "enhanced"]),
            created_at=now - timedelta(days=rng.randint(90, 1100)),
        )
    return profiles


def uuid_hex(rng: random.Random) -> str:
    """Random 10-hex-char token from a seeded RNG (keeps generation reproducible)."""
    return f"{rng.getrandbits(40):010x}"


# ---------------------------------------------------------------------------
# Plausibility Gate
# ---------------------------------------------------------------------------


class PlausibilityGate:
    """
    Three fail-fast checks applied to every inbound PaymentMessage.

    This encodes the attacker-economics thesis of the whole arena: a real
    fraudster optimizes ROI. Payloads that couldn't exist as a rational
    attack are rejected BEFORE entering the stack, which keeps the defense
    layer's metrics honest (we only score plausible traffic).
    """

    @staticmethod
    def check_economic_viability(payload: PaymentMessage) -> Optional[str]:
        """Check 1: does the amount clear the acquisition cost of the resource?

        Legit traffic carries stolen_resource=None and auto-passes. An
        attacker burning a $200 synthetic identity on a $4.99 coffee is not
        a threat model — it's charity.
        """
        if payload.stolen_resource is None:
            return None
        floor = RESOURCE_COST_TABLE_USD[payload.stolen_resource]
        if payload.amount < floor:
            return REASON_ECONOMIC
        return None

    @staticmethod
    def check_realism(payload: PaymentMessage, registry: dict[str, Merchant]) -> Optional[str]:
        """Check 2: metadata coherence against the merchant registry.

        - unknown merchant => incoherent
        - MCC mismatch vs registry => incoherent (classic BIN/MCC abuse tell)
        - GeoIP country != merchant country => incoherent for our sim universe
          (real stacks allow cross-border, but here merchants are domestic-
          acquiring by construction; document this simplification honestly)
        - empty/garbage device id is impossible post-schema, so the remaining
          device signal (binding novelty) is deferred to the defense layer.
        """
        merchant = registry.get(payload.merchant_id)
        if merchant is None:
            return REASON_METADATA
        if payload.mcc != merchant.mcc:
            return REASON_METADATA
        if payload.ip_country != merchant.country:
            return REASON_METADATA
        return None

    @staticmethod
    def check_rail_feasibility(payload: PaymentMessage) -> tuple[Optional[str], list[str]]:
        """Check 3: physical pairing rules between entry mode and 3DS.

        Returns (reason_or_None, risk_flags). "A" (attempted) on a CNP rail
        is allowed but flagged — liability-shift gaming is a known pattern.

        Pairing truth table:
          CONTACTLESS : 3DS must be N  (tap never runs 3DS)
          CHIP / SWIPE: 3DS must be N  (card-present rails don't do 3DS)
          ECOM / CNP  : 3DS in {Y,A,N} ("A" adds a suspicion flag)
        """
        flags: list[str] = []
        mode, tds = payload.pos_entry_mode, payload.three_ds_status

        if mode in (PosEntryMode.CONTACTLESS, PosEntryMode.CHIP, PosEntryMode.SWIPE):
            if tds != ThreeDSStatus.N:
                return REASON_RAIL, flags
        else:  # ECOM or CNP — card-not-present rails
            if tds == ThreeDSStatus.A:
                flags.append("3ds_attempted_suspect")
        return None, flags


# ---------------------------------------------------------------------------
# Payment Environment
# ---------------------------------------------------------------------------


class PaymentEnvironment:
    """
    The world state of the arena plus the single ingest chokepoint.

    State owned here (all in-memory, reset per run):
      * merchant registry (20 merchants)
      * customer profiles (1000 by default)
      * per-customer accepted-transaction ledgers (bounded to last 100)
      * accepted/rejected event retention for replay/UI diagnostics
      * running balances (informational; funds sufficiency is NOT a gate
        check — declines are the defense stack's job, and conflating them
        would pollute the FPR budget measurement)

    ``event_stream_maxlen`` and ``gate_rejects_maxlen`` are optional retention
    bounds for production/scale runs. ``None`` preserves the historical full
    in-memory stream used by experiments. Total counters are maintained either
    way, so observability does not depend on retained-history size.
    """

    def __init__(
        self,
        n_customers: int = 1000,
        seed: int = 42,
        history_size: int = 100,
        event_stream_maxlen: int | None = None,
        gate_rejects_maxlen: int | None = None,
    ) -> None:
        if event_stream_maxlen is not None and event_stream_maxlen < 1:
            raise ValueError("event_stream_maxlen must be >= 1 or None")
        if gate_rejects_maxlen is not None and gate_rejects_maxlen < 1:
            raise ValueError("gate_rejects_maxlen must be >= 1 or None")

        self.rng = random.Random(seed)
        self.merchant_registry = build_merchant_registry()
        self.customers = build_customer_profiles(n_customers, seed=seed)

        # These views are immutable for the lifetime of the simulated issuer
        # world. Building them once avoids Pydantic serialization and repeated
        # list scans in the per-authorization hot path.
        self._merchant_enrichment: dict[str, dict] = {
            merchant_id: merchant.model_dump()
            for merchant_id, merchant in self.merchant_registry.items()
        }
        self._customer_devices: dict[str, frozenset[str]] = {
            customer_id: frozenset(customer.devices)
            for customer_id, customer in self.customers.items()
        }

        self._history_size = history_size
        self._customer_ledgers: dict[str, deque] = {
            cid: deque(maxlen=history_size) for cid in self.customers
        }
        self.event_stream = (
            deque(maxlen=event_stream_maxlen)
            if event_stream_maxlen is not None
            else []
        )
        self.gate_rejects = (
            deque(maxlen=gate_rejects_maxlen)
            if gate_rejects_maxlen is not None
            else []
        )
        self.events_seen_total = 0
        self.gate_rejects_total = 0

    # ------------------------------------------------------------------ #
    # Ingest path
    # ------------------------------------------------------------------ #

    def ingest(self, payload: PaymentMessage | dict) -> dict:
        """Run one authorization through the Plausibility Gate.

        Returns:
            {
              "accepted": bool,
              "reason":   "ok" | "economic_infeasible" | "metadata_incoherent"
                          | "rail_infeasible",
              "internal_event": {...}   # enriched event for ledger/stream/UI
            }

        Raises TypeError for non-PaymentMessage input that fails coercion —
        schema violations die loudly at the boundary (see constraints).
        """
        if isinstance(payload, dict):
            payload = PaymentMessage.model_validate(payload)
        elif not isinstance(payload, PaymentMessage):
            raise TypeError(f"ingest expects PaymentMessage|dict, got {type(payload)!r}")

        flags: list[str] = []

        # ---- Check 1: ECONOMIC VIABILITY --------------------------------
        reason = PlausibilityGate.check_economic_viability(payload)
        if reason is None:
            # ---- Check 2: REALISM CRITIC --------------------------------
            reason = PlausibilityGate.check_realism(payload, self.merchant_registry)
        if reason is None:
            # ---- Check 3: RAIL FEASIBILITY ------------------------------
            reason, rail_flags = PlausibilityGate.check_rail_feasibility(payload)
            flags.extend(rail_flags)

        accepted = reason is None
        internal_event = self._enrich(payload, accepted, reason or REASON_OK, flags)

        self.events_seen_total += 1
        self.event_stream.append(internal_event)
        if accepted:
            self._post_acceptance(payload, internal_event)
        else:
            self.gate_rejects_total += 1
            self.gate_rejects.append(internal_event)

        return {"accepted": accepted, "reason": internal_event["gate_reason"], "internal_event": internal_event}

    # ------------------------------------------------------------------ #
    # History access
    # ------------------------------------------------------------------ #

    def get_customer_history(self, customer_id: str) -> list[dict]:
        """Return the retained ACCEPTED transaction history for a customer."""
        return list(self._customer_ledgers.get(customer_id, ()))

    def is_device_known(self, customer_id: str, device_id: str) -> bool:
        """O(1) issuer binding lookup used by enrichment/feature extraction."""
        return device_id in self._customer_devices.get(customer_id, ())

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _enrich(self, payload: PaymentMessage, accepted: bool, reason: str, flags: list[str]) -> dict:
        """Build the internal event: payload + environment context.

        `device_known` is enrichment only — a new device is NOT a gate reject;
        novel-device scoring belongs to the defense stack (novelty.py).
        """
        merchant = self._merchant_enrichment.get(payload.merchant_id)
        customer = self.customers.get(payload.customer_id)
        return {
            "type": "transaction",
            "accepted": accepted,
            "gate_reason": reason,
            "risk_flags": flags,
            "payload": payload.to_wire(),
            "enrichment": {
                # Give each event its own mapping so downstream diagnostic code
                # can mutate an event without corrupting the cached registry view.
                "merchant": dict(merchant) if merchant is not None else None,
                "customer_country": customer.country if customer else None,
                "device_known": self.is_device_known(
                    payload.customer_id, payload.device_id
                ),
                "balance_before": customer.balance if customer else None,
            },
            "decision": None,  # filled by the defense stack in later steps
        }

    def _post_acceptance(self, payload: PaymentMessage, event: dict) -> None:
        """Mutate world state for an accepted auth: decrement balance, append ledger."""
        customer = self.customers[payload.customer_id]
        customer.balance = round(max(0.0, customer.balance - payload.amount), 2)
        event["enrichment"]["balance_after"] = customer.balance
        self._customer_ledgers[payload.customer_id].append(event)


__all__ = [
    "PaymentEnvironment",
    "PlausibilityGate",
    "Merchant",
    "CustomerProfile",
    "build_merchant_registry",
    "build_customer_profiles",
    "REASON_OK",
    "REASON_ECONOMIC",
    "REASON_METADATA",
    "REASON_RAIL",
]
