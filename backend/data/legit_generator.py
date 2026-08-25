"""
Legitimate traffic generator.

Produces the "clean" baseline of the arena: Faker-driven, distribution-shaped
cardholder behavior that is coherent WITH THE MERCHANT REGISTRY BY
CONSTRUCTION (MCC and geo are copied from the chosen merchant, so the
REALISM CRITIC can only pass it — any gate reject here is a bug, not noise).

This file is a generator wired into the event loop, NOT a static dataset:
every run produces fresh traffic through PaymentEnvironment.ingest(), and the
JSONL artifact is just the persisted trace of that run.

Rail mix target (per spec): 70% ECOM (mostly 3DS=Y), 20% CONTACTLESS,
10% CNP. Amounts follow per-vertical lognormal-ish distributions clamped to
sane card-present/CNP ranges.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from faker import Faker

# Allow `python data/legit_generator.py` from anywhere: put backend/ on sys.path.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from environment.payment_stack import PaymentEnvironment, Merchant  # noqa: E402
from schemas.payment import (
    PaymentMessage,
    PosEntryMode,
    ThreeDSStatus,
)

DATA_DIR = Path(__file__).resolve().parent
BASELINE_PATH = DATA_DIR / "legit_baseline.jsonl"

# Per-vertical amount distributions: (mu, sigma) for lognormal + clamp range.
_AMOUNT_DIST = {
    "grocery":     (3.6, 0.7, 4.00, 320.00),
    "electronics": (5.2, 0.9, 25.00, 2500.00),
    "travel":      (6.4, 0.7, 90.00, 4000.00),
    "ecommerce":   (4.3, 1.0, 8.00, 900.00),
}

# Which verticals each entry mode plausibly touches. CONTACTLESS is
# card-PRESENT => physical verticals only; ECOM/CNP => online merchants.
_MODE_VERTICALS = {
    PosEntryMode.ECOM: ["grocery", "electronics", "travel", "ecommerce", "ecommerce"],
    PosEntryMode.CONTACTLESS: ["grocery", "grocery", "electronics", "travel"],
    PosEntryMode.CNP: ["travel", "ecommerce", "ecommerce"],
}

# 3DS outcome mix per mode — realistic frictionless-heavy profile with a
# sprinkle of attempted ("A") auths so the defense layer sees them in baseline.
_TDS_MIX = {
    PosEntryMode.ECOM: [(ThreeDSStatus.Y, 0.75), (ThreeDSStatus.A, 0.10), (ThreeDSStatus.N, 0.15)],
    PosEntryMode.CNP: [(ThreeDSStatus.Y, 0.40), (ThreeDSStatus.A, 0.15), (ThreeDSStatus.N, 0.45)],
    # Card-present rails never run 3DS; enforced below regardless of this table.
    PosEntryMode.CONTACTLESS: [(ThreeDSStatus.N, 1.0)],
    PosEntryMode.CHIP: [(ThreeDSStatus.N, 1.0)],
    PosEntryMode.SWIPE: [(ThreeDSStatus.N, 1.0)],
}


def _sample_amount(rng: random.Random, category: str) -> float:
    mu, sigma, lo, hi = _AMOUNT_DIST[category]
    amt = float("inf")
    while not (lo <= amt <= hi):  # resample until inside clamp — keeps dist honest
        amt = rng.lognormvariate(mu, sigma)
    return round(amt, 2)


def _pick_merchant(env: PaymentEnvironment, rng: random.Random, mode: PosEntryMode) -> Merchant:
    vertical = rng.choice(_MODE_VERTICALS[mode])
    candidates = [m for m in env.merchant_registry.values() if m.category == vertical]
    return rng.choice(candidates)


def build_legit_payload(
    env: PaymentEnvironment,
    rng: random.Random,
    fake: Faker,
    mode: PosEntryMode | None = None,
    timestamp: datetime | None = None,
) -> PaymentMessage:
    """Construct ONE coherent legitimate payload.

    Coherence strategy (why the gate must accept this):
      * merchant sampled from registry -> mcc/ip_country copied FROM it
      * device drawn from customer's bound set 85% of the time; new-device
        churn 15% (legal — novelty scoring is the defense layer's job)
      * rail pairing follows the gate's truth table by construction
    """
    mode = mode or rng.choice(
        [PosEntryMode.ECOM] * 70 + [PosEntryMode.CONTACTLESS] * 20 + [PosEntryMode.CNP] * 10
    )
    customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
    merchant = _pick_merchant(env, rng, mode)

    # 85% bound device / 15% new-device churn (legal; novelty scoring is the
    # defense layer's concern, not the gate's).
    device_id = (
        rng.choice(customer.devices)
        if rng.random() < 0.85
        else f"DEV_{fake.uuid4().replace('-', '')[:10]}"
    )

    tds = rng.choices([s for s, _ in _TDS_MIX[mode]], weights=[w for _, w in _TDS_MIX[mode]])[0]
    timestamp = timestamp or datetime.now(timezone.utc) - timedelta(
        minutes=rng.randint(0, 60 * 24 * 90)  # spread over trailing 90 days
    )

    return PaymentMessage(
        transaction_id=f"{rng.getrandbits(64):016X}",
        customer_id=customer.customer_id,
        merchant_id=merchant.merchant_id,
        mcc=merchant.mcc,                      # <- copied from registry: coherent
        amount=_sample_amount(rng, merchant.category),
        pos_entry_mode=mode,
        **{"3ds_status": tds.value},
        ip_address=fake.ipv4_public(),
        ip_country=merchant.country,           # <- copied from registry: coherent
        device_id=device_id,
        stolen_resource=None,                  # <- legit traffic never claims a vector
        timestamp=timestamp,
    )


def generate_baseline(n: int = 10_000, seed: int = 42) -> list[dict]:
    """Generate n legit txns through the live gate; persist + return payloads.

    Asserts 100% gate acceptance — the generator is coherent-by-construction,
    so ANY reject indicates an environment/generator contract bug.
    """
    env = PaymentEnvironment(n_customers=1000, seed=seed)
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    out: list[dict] = []
    rejects: list[dict] = []

    for _ in range(n):
        msg = build_legit_payload(env, rng, fake)
        result = env.ingest(msg)
        if result["accepted"]:
            out.append({**msg.to_wire(), "label": "legit"})
        else:
            rejects.append(result["internal_event"])

    assert not rejects, f"Gate rejected {len(rejects)} 'coherent' legit txns: " \
                        f"{rejects[0]['gate_reason']} — fix generator/environment contract."

    BASELINE_PATH.write_text(
        "\n".join(json.dumps(row, default=str) for row in out) + "\n",
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    rows = generate_baseline(10_000)
    print(f"wrote {len(rows)} accepted legit txns -> {BASELINE_PATH}")
    mix = {}
    for r in rows:
        mix[r["pos_entry_mode"]] = mix.get(r["pos_entry_mode"], 0) + 1
    print("rail mix:", {k: f"{v / len(rows):.0%}" for k, v in sorted(mix.items())})
