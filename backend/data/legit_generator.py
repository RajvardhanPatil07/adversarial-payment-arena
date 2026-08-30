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

# Fixed clock anchor for the trailing-90-day legit history, so corpora are
# bit-reproducible across processes at a given seed (see corpus_builder
# _ATTACK_BASE for the same reasoning). Legit history trails this anchor;
# attacks fire ~12h before it, keeping the whole corpus internally ordered.
_LEGIT_NOW = datetime(2026, 6, 2, 0, 0, 0, tzinfo=timezone.utc)

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

    # ~94% bound device / ~6% new-device churn. Real cardholder token churn
    # is low; a higher rate would make "unknown device" too weak a fraud
    # prior for the defense layers (measured: 15% churn pushed legit FPR
    # past budget because attacks are ~100% unknown-device).
    device_id = (
        rng.choice(customer.devices)
        if rng.random() < 0.94
        else f"DEV_{fake.uuid4().replace('-', '')[:10]}"
    )

    tds = rng.choices([s for s, _ in _TDS_MIX[mode]], weights=[w for _, w in _TDS_MIX[mode]])[0]
    timestamp = timestamp or _LEGIT_NOW - timedelta(
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


# --------------------------------------------------------------------------- #
# HARD NEGATIVES — legitimate traffic that LOOKS like fraud
# --------------------------------------------------------------------------- #
#
# Why this exists
# ---------------
# Without this, every legitimate transaction in the corpus is a single
# well-behaved event spread over 90 days, while every attack is bursty and
# mostly on an unknown device. The two classes become linearly separable on
# `device_known` and `cust_txn_count_10m` alone, any classifier reaches ~0.999
# recall, and the benchmark stops measuring anything: a saturated metric cannot
# rank two defences.
#
# Real issuers do not have that luxury. Their false positives come from a small
# set of well-known *benign anomalies* -- the honeymoon customer on a new phone
# abroad, the family sharing one tablet, the Black-Friday burst, the genuine
# high-ticket purchase. Those are the transactions that generate insult cost,
# and they are structurally indistinguishable from fraud on the naive features.
#
# So we generate them ON PURPOSE, label them LEGIT, and let them punish the
# model. Every profile below is a real, documented benign pattern, and each one
# collides with a specific attack family:
#
#   travel_abroad_burst   collides with ATTACK_1  (new device + burst + geo)
#   shared_family_device  collides with ATTACK_2  (one device, several people)
#   flash_sale_crowd      collides with ATTACK_3  (many customers, one merchant)
#   subscription_batch    collides with ATTACK_4  (rapid low-value cadence)
#   big_ticket_purchase   collides with ATTACK_5  (amount far over own baseline)
#   payday_regular        collides with ATTACK_11 (machine-regular cadence)
#
# This makes the benchmark harder on purpose. The recall number goes DOWN and
# becomes informative, which is the entire point.

_HARD_NEGATIVE_PROFILES = [
    "travel_abroad_burst",
    "shared_family_device",
    "flash_sale_crowd",
    "subscription_batch",
    "big_ticket_purchase",
    "payday_regular",
]


def build_hard_negatives(
    env: PaymentEnvironment,
    rng: random.Random,
    fake: Faker,
    n: int,
    profile: str | None = None,
) -> list[PaymentMessage]:
    """Generate `n` LEGITIMATE payloads that mimic fraud signatures.

    Returns PaymentMessages carrying `stolen_resource=None` (they are genuinely
    legitimate) that nonetheless move the same features the attack families
    move. Coherent-by-construction against the Plausibility Gate: mcc and
    ip_country are always copied from the chosen merchant.
    """
    out: list[PaymentMessage] = []
    while len(out) < n:
        kind = profile or rng.choice(_HARD_NEGATIVE_PROFILES)

        # --- 1. Honeymoon abroad: new phone, new country, holiday burst ----- #
        # Collides with voice-clone ATO: unknown device + rapid CNP tickets.
        if kind == "travel_abroad_burst":
            customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
            device = f"DEV_{fake.uuid4().replace('-', '')[:10]}"   # genuinely new phone
            ip = fake.ipv4_public()
            base = _LEGIT_NOW - timedelta(days=rng.randint(1, 60))
            for _ in range(min(rng.randint(3, 6), n - len(out))):
                base += timedelta(seconds=rng.randint(120, 900))
                merchant = _pick_merchant(env, rng, PosEntryMode.ECOM)
                out.append(PaymentMessage(
                    transaction_id=f"{rng.getrandbits(64):016X}",
                    customer_id=customer.customer_id,
                    merchant_id=merchant.merchant_id,
                    mcc=merchant.mcc,
                    amount=_sample_amount(rng, merchant.category),
                    pos_entry_mode=PosEntryMode.ECOM,
                    **{"3ds_status": ThreeDSStatus.Y.value},
                    ip_address=ip,
                    ip_country=merchant.country,
                    device_id=device,
                    stolen_resource=None,
                    timestamp=base,
                ))

        # --- 2. One household tablet used by several family members --------- #
        # Collides with the mule ring: ONE device across several customer ids.
        elif kind == "shared_family_device":
            family = rng.sample(sorted(env.customers.keys()), rng.randint(2, 4))
            device = f"DEV_{fake.uuid4().replace('-', '')[:10]}"
            base = _LEGIT_NOW - timedelta(days=rng.randint(1, 85))
            for cid in family:
                if len(out) >= n:
                    break
                for _ in range(rng.randint(1, 2)):
                    if len(out) >= n:
                        break
                    base += timedelta(seconds=rng.randint(300, 3600))
                    merchant = _pick_merchant(env, rng, PosEntryMode.ECOM)
                    out.append(PaymentMessage(
                        transaction_id=f"{rng.getrandbits(64):016X}",
                        customer_id=cid,
                        merchant_id=merchant.merchant_id,
                        mcc=merchant.mcc,
                        amount=_sample_amount(rng, merchant.category),
                        pos_entry_mode=PosEntryMode.ECOM,
                        **{"3ds_status": ThreeDSStatus.Y.value},
                        ip_address=fake.ipv4_public(),
                        ip_country=merchant.country,
                        device_id=device,
                        stolen_resource=None,
                        timestamp=base,
                    ))

        # --- 3. Flash sale: a crowd converges on one merchant in minutes ---- #
        # Collides with merchant compromise AND merchant bust-out.
        elif kind == "flash_sale_crowd":
            merchant = _pick_merchant(env, rng, PosEntryMode.ECOM)
            crowd = rng.sample(sorted(env.customers.keys()), min(rng.randint(12, 20), len(env.customers)))
            base = _LEGIT_NOW - timedelta(days=rng.randint(1, 80))
            for cid in crowd:
                if len(out) >= n:
                    break
                customer = env.customers[cid]
                base += timedelta(seconds=rng.randint(5, 45))
                out.append(PaymentMessage(
                    transaction_id=f"{rng.getrandbits(64):016X}",
                    customer_id=cid,
                    merchant_id=merchant.merchant_id,
                    mcc=merchant.mcc,
                    amount=_sample_amount(rng, merchant.category),
                    pos_entry_mode=PosEntryMode.ECOM,
                    **{"3ds_status": rng.choice([ThreeDSStatus.Y, ThreeDSStatus.N]).value},
                    ip_address=fake.ipv4_public(),
                    ip_country=merchant.country,
                    device_id=rng.choice(customer.devices),
                    stolen_resource=None,
                    timestamp=base,
                ))

        # --- 4. Monthly subscription batch billed back-to-back ------------- #
        # Collides with card testing: rapid low-value cadence on one card.
        elif kind == "subscription_batch":
            customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
            device = rng.choice(customer.devices)
            base = _LEGIT_NOW - timedelta(days=rng.randint(1, 85))
            for _ in range(min(rng.randint(3, 6), n - len(out))):
                base += timedelta(seconds=rng.randint(10, 120))
                merchant = _pick_merchant(env, rng, PosEntryMode.ECOM)
                out.append(PaymentMessage(
                    transaction_id=f"{rng.getrandbits(64):016X}",
                    customer_id=customer.customer_id,
                    merchant_id=merchant.merchant_id,
                    mcc=merchant.mcc,
                    amount=round(rng.uniform(4.99, 24.99), 2),   # small, like a probe
                    pos_entry_mode=PosEntryMode.ECOM,
                    **{"3ds_status": ThreeDSStatus.N.value},      # MIT, no challenge
                    ip_address=fake.ipv4_public(),
                    ip_country=merchant.country,
                    device_id=device,
                    stolen_resource=None,
                    timestamp=base,
                ))

        # --- 5. Genuine high-ticket purchase far over own baseline --------- #
        # Collides with the APP scam: amount wildly outside personal history.
        elif kind == "big_ticket_purchase":
            customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
            merchant = _pick_merchant(env, rng, PosEntryMode.ECOM)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                # A real laptop / holiday: 8-20x this customer's typical ticket.
                amount=round(_sample_amount(rng, merchant.category) * rng.uniform(8.0, 20.0), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                **{"3ds_status": ThreeDSStatus.Y.value},
                ip_address=fake.ipv4_public(),
                ip_country=merchant.country,
                device_id=rng.choice(customer.devices),
                stolen_resource=None,
                timestamp=_LEGIT_NOW - timedelta(days=rng.randint(1, 85)),
            ))

        # --- 6. Payday standing orders: machine-regular cadence ------------ #
        # Collides with agent scope expansion: fixed-period, low-jitter timing.
        else:
            customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
            device = rng.choice(customer.devices)
            base = _LEGIT_NOW - timedelta(days=rng.randint(30, 88))
            period = rng.randint(3600, 7200)
            for _ in range(min(rng.randint(3, 5), n - len(out))):
                base += timedelta(seconds=period + rng.randint(-45, 45))
                merchant = _pick_merchant(env, rng, PosEntryMode.ECOM)
                out.append(PaymentMessage(
                    transaction_id=f"{rng.getrandbits(64):016X}",
                    customer_id=customer.customer_id,
                    merchant_id=merchant.merchant_id,
                    mcc=merchant.mcc,
                    amount=_sample_amount(rng, merchant.category),
                    pos_entry_mode=PosEntryMode.ECOM,
                    **{"3ds_status": ThreeDSStatus.Y.value},
                    ip_address=fake.ipv4_public(),
                    ip_country=merchant.country,
                    device_id=device,
                    stolen_resource=None,
                    timestamp=base,
                ))
    return out[:n]


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
