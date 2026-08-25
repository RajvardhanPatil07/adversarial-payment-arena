"""
Gate acceptance tests — Step 2 exit criteria.

1. 100 freshly generated legit txns must pass the gate 100%.
2. 5 deliberately incoherent payloads must each fail with the EXACT reason code.
3. Bonus contract checks: ECOM+"A" passes but carries the suspicion flag;
   contactless never runs 3DS.
"""

import random
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from data.legit_generator import build_legit_payload  # noqa: E402
from environment.payment_stack import (  # noqa: E402
    REASON_ECONOMIC,
    REASON_METADATA,
    REASON_RAIL,
    PaymentEnvironment,
)
from faker import Faker  # noqa: E402
from schemas.payment import PaymentMessage, PosEntryMode, StolenResourceType, ThreeDSStatus  # noqa: E402


@pytest.fixture(scope="module")
def env() -> PaymentEnvironment:
    return PaymentEnvironment(n_customers=1000, seed=7)


@pytest.fixture(scope="module")
def rng() -> random.Random:
    return random.Random(7)


# --------------------------------------------------------------------------- #
# 1. Legit traffic sails through
# --------------------------------------------------------------------------- #


def test_100_legit_txns_all_pass(env: PaymentEnvironment, rng: random.Random):
    fake = Faker()
    Faker.seed(7)
    for _ in range(100):
        msg = build_legit_payload(env, rng, fake)
        result = env.ingest(msg)
        assert result["accepted"], f"legit txn rejected: {result['reason']}"
        assert result["reason"] == "ok"
    # environment state actually mutated: history is queryable
    some_customer = msg.customer_id
    assert len(env.get_customer_history(some_customer)) >= 1


def test_ecom_attempted_3ds_passes_but_is_flagged(env: PaymentEnvironment, rng: random.Random):
    """'A' on a CNP rail is legal but suspicious — flag, don't reject."""
    fake = Faker()
    Faker.seed(11)
    for _ in range(50):  # resample until we draw an "A" from the mix
        msg = build_legit_payload(env, rng, fake, mode=PosEntryMode.ECOM)
        if msg.three_ds_status == ThreeDSStatus.A:
            break
    assert msg.three_ds_status == ThreeDSStatus.A, "mix never produced an attempted 3DS"
    result = env.ingest(msg)
    assert result["accepted"]
    assert "3ds_attempted_suspect" in result["internal_event"]["risk_flags"]


def test_contactless_never_carries_3ds(env: PaymentEnvironment, rng: random.Random):
    fake = Faker()
    Faker.seed(13)
    for _ in range(25):
        msg = build_legit_payload(env, rng, fake, mode=PosEntryMode.CONTACTLESS)
        assert msg.three_ds_status == ThreeDSStatus.N


# --------------------------------------------------------------------------- #
# 2. Deliberately bad payloads fail with exact reason codes
# --------------------------------------------------------------------------- #


def _base_kwargs(env: PaymentEnvironment) -> dict:
    """A known-good skeleton we mutate into each failure mode."""
    customer = next(iter(env.customers.values()))
    merchant = next(
        m for m in env.merchant_registry.values() if m.category == "grocery" and not m.is_online
    )
    return dict(
        customer_id=customer.customer_id,
        merchant_id=merchant.merchant_id,
        mcc=merchant.mcc,
        amount=42.00,
        pos_entry_mode=PosEntryMode.CONTACTLESS.value,
        **{"3ds_status": ThreeDSStatus.N.value},
        ip_address="198.51.100.24",
        ip_country=merchant.country,
        device_id=customer.devices[0],
    )


def test_bad_payloads_fail_with_exact_codes(env: PaymentEnvironment):
    base = _base_kwargs(env)

    bad_cases = [
        # (label, overrides, expected_reason_code)
        (
            "wrong MCC vs merchant registry",
            {"mcc": 5732},  # electronics MCC on a grocery merchant
            REASON_METADATA,
        ),
        (
            "contactless + 3DS=Y (rail-infeasible pairing)",
            {"3ds_status": ThreeDSStatus.Y.value},
            REASON_RAIL,
        ),
        (
            "$49.99 burn of a $200 synthetic identity",
            {
                "pos_entry_mode": PosEntryMode.ECOM.value,
                "3ds_status": ThreeDSStatus.Y.value,
                "amount": 49.99,
                "stolen_resource": StolenResourceType.SYNTHETIC_IDENTITY.value,
            },
            REASON_ECONOMIC,
        ),
        (
            "GeoIP country mismatch vs merchant acquiring country",
            {"ip_country": "FR"},
            REASON_METADATA,
        ),
        (
            "unknown ghost merchant",
            {"merchant_id": "MERCH_GHOST_999", "mcc": 5411},
            REASON_METADATA,
        ),
    ]

    seen_codes = set()
    for label, overrides, expected in bad_cases:
        payload = PaymentMessage.model_validate({**base, **overrides})
        result = env.ingest(payload)
        assert not result["accepted"], f"{label}: expected reject, got accept"
        assert result["reason"] == expected, f"{label}: wanted {expected}, got {result['reason']}"
        assert result["internal_event"]["gate_reason"] == expected
        seen_codes.add(expected)

    # all three gate checks exercised end-to-end
    assert seen_codes == {REASON_METADATA, REASON_RAIL, REASON_ECONOMIC}


def test_rejects_do_not_mutate_balances_or_history(env: PaymentEnvironment):
    base = _base_kwargs(env)
    cid = base["customer_id"]

    balance_before = env.customers[cid].balance
    history_before = list(env.get_customer_history(cid))

    payload = PaymentMessage.model_validate({**base, "mcc": 5732})
    result = env.ingest(payload)
    assert not result["accepted"]

    assert env.customers[cid].balance == balance_before
    assert env.get_customer_history(cid) == history_before


def test_history_capped_at_last_100(env: PaymentEnvironment, rng: random.Random):
    fake = Faker()
    Faker.seed(21)
    # hammer one specific customer with >100 accepted txns
    target = next(iter(env.customers))
    for i in range(130):
        msg = build_legit_payload(env, rng, fake)
        msg = msg.model_copy(update={"customer_id": target})
        result = env.ingest(msg)
        assert result["accepted"]
    history = env.get_customer_history(target)
    assert len(history) == 100  # deque(maxlen=100) keeps exactly the tail
