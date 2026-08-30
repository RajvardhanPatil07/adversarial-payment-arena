"""
Labeled-corpus builder for the defense stack — NO static datasets.

Every row is generated through the live event loop:
  legit rows  <- legit_generator payloads ingested into PaymentEnvironment
  attack rows <- topology-aware synthesizers driven by the AttackSpec YAMLs,
                 also gated through PaymentEnvironment (a reject here is a bug)

Each row carries the PRE-transaction feature vector computed by the exact
same FeatureExtractor used at inference (observe-after-score replay), plus
its label. Training therefore has zero feature-skew by construction.

Attack coverage: specs 1-3 are LABELED TRAINING data. Attack 4 is deliberately
NEVER generated here — later steps use it as the held-out novel attack that
only the unsupervised layers (Isolation Forest + graph) can hope to catch.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from data.legit_generator import build_hard_negatives, build_legit_payload  # noqa: E402
from defense.realtime import FeatureExtractor  # noqa: E402
from environment.payment_stack import PaymentEnvironment  # noqa: E402
from faker import Faker  # noqa: E402
from schemas.attack import AttackSpec, load_attack_spec  # noqa: E402
from schemas.payment import PaymentMessage, PosEntryMode, ThreeDSStatus  # noqa: E402

SPECS_DIR = BACKEND_ROOT / "attack_specs"
SPEC_FILES = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": SPECS_DIR / "attack_1_mfa_reset_voice_clone.yaml",
    "ATTACK_2_SYNTHETIC_MULE_RING": SPECS_DIR / "attack_2_synthetic_mule_ring.yaml",
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": SPECS_DIR / "attack_3_prompt_injected_merchant.yaml",
    # ZERO-DAY: generatable for the holdout experiment, NEVER in TRAIN_COUNTS.
    "ATTACK_4_CNP_HIGH_VELOCITY": SPECS_DIR / "attack_4_cnp_high_velocity.yaml",
    # India-first real-time-rail families (taxonomy T-12, T-14, T-17, T-09).
    # Each one defeats a DIFFERENT defensive signal, which is the point of
    # adding them: breadth here means coverage of failure modes, not row count.
    "ATTACK_5_APP_SCAM_PERSONALISED": SPECS_DIR / "attack_5_app_scam_personalised.yaml",
    "ATTACK_6_VPA_RENTAL_MULE": SPECS_DIR / "attack_6_vpa_rental_mule.yaml",
    "ATTACK_7_SYNCHRONISED_BURST_CASHOUT": SPECS_DIR / "attack_7_synchronised_burst_cashout.yaml",
    "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING": SPECS_DIR / "attack_8_learned_threshold_structuring.yaml",
    # Wave 3 (taxonomy T-05, T-06, T-19, T-11, T-04, T-20). These extend
    # coverage OFF the card-velocity axis: an authentication-layer attack that
    # passes 3DS, a policy-layer attack on the exemption band, an agentic
    # mandate-drift attack, a sequence-level rule evasion, an acquiring-side
    # bust-out, and a model-layer attack on the defender itself.
    "ATTACK_9_OTP_RELAY_VISHING": SPECS_DIR / "attack_9_otp_relay_vishing.yaml",
    "ATTACK_10_EXEMPTION_BAND_ABUSE": SPECS_DIR / "attack_10_exemption_band_abuse.yaml",
    "ATTACK_11_AGENTIC_SCOPE_EXPANSION": SPECS_DIR / "attack_11_agentic_scope_expansion.yaml",
    "ATTACK_12_GEO_VELOCITY_ITINERARY": SPECS_DIR / "attack_12_geo_velocity_itinerary.yaml",
    "ATTACK_13_MERCHANT_BUSTOUT": SPECS_DIR / "attack_13_merchant_bustout.yaml",
    "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE": SPECS_DIR / "attack_14_adversarial_boundary_probe.yaml",
}

# Attacks fire "recently"; legit history trails 90 days. A FIXED anchor (not
# datetime.now) keeps the corpus bit-reproducible across processes at a given
# seed: time-derived features (hour_sin/hour_cos/dow) must not drift with the
# wall clock, or `make reproduce` produces different numbers every run. The
# date is arbitrary but must stay in the past relative to any real run.
_ATTACK_BASE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Shared synthesis helpers
# --------------------------------------------------------------------------- #


def _merchant_for(env: PaymentEnvironment, rng: random.Random, verticals: list[str], online: bool | None):
    cands = [
        m for m in env.merchant_registry.values()
        if m.category in verticals
        and m.country == "US"
        and (online is None or m.is_online == online)
    ]
    if not cands:  # defensive fallback: any US merchant
        cands = [m for m in env.merchant_registry.values() if m.country == "US"]
    return rng.choice(cands)


def _new_device(rng: random.Random) -> str:
    return f"DEV_{rng.getrandbits(40):010x}"


# --------------------------------------------------------------------------- #
# Per-spec synthesizers (all coherent-by-construction vs the Plausibility Gate)
# --------------------------------------------------------------------------- #


def synth_attack_1_voice_clone(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """Bursty ATO monetization: per victim, a brand-new device fires 2-4 CNP
    tickets minutes apart. Signals: unknown device + personal velocity burst +
    amount far over the victim's baseline."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
        device = _new_device(rng)                      # session born post-reset
        ip = fake.ipv4_public()
        merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
        tds_pool = [t for t in cons.preferred_three_ds]
        for _ in range(min(rng.randint(2, 4), n - len(out))):
            cursor += timedelta(seconds=rng.randint(60, 540))
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(rng.uniform(lo, hi), 2),
                pos_entry_mode=PosEntryMode.CNP,
                **{"3ds_status": rng.choice(tds_pool).value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_2_mule_ring(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """Cash-out topology: rings of 8-14 synthetic customers SHARE one device
    and ONE egress IP. Signals: graph ring (primary) + device velocity."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        ring_size = min(rng.randint(8, 14), n - len(out))
        device, ip = _new_device(rng), fake.ipv4_public()
        merchant = _merchant_for(env, rng, cons.target_verticals, online=False)
        mules = rng.sample(sorted(env.customers.keys()), ring_size)
        for cid in mules:
            cursor += timedelta(seconds=rng.randint(30, 180))
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=cid,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(rng.uniform(lo, hi), 2),
                pos_entry_mode=cons.pos_entry_modes[0],
                **{"3ds_status": ThreeDSStatus.N.value},   # tap rail physics
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_3_compromised_merchant(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """Merchant-endpoint burst: MANY distinct honest-looking customers converge
    on ONE MID, mostly on their own bound devices, passing 3DS. Per-payload
    signals are weak BY DESIGN; merchant-centric velocity is the tell."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
    victims = iter(rng.sample(sorted(env.customers.keys()), min(n, len(env.customers))))
    while len(out) < n:
        try:
            cid = next(victims)
        except StopIteration:
            cid = rng.choice(sorted(env.customers.keys()))
        customer = env.customers[cid]
        known = rng.random() < 0.9                      # harvested sessions
        device = rng.choice(customer.devices) if known else _new_device(rng)
        cursor += timedelta(seconds=rng.randint(20, 90))
        amount = min(max(rng.gauss((lo + hi) / 2 + 130, 180), lo), hi)
        out.append(PaymentMessage(
            transaction_id=f"{rng.getrandbits(64):016X}",
            customer_id=cid,
            merchant_id=merchant.merchant_id,
            mcc=merchant.mcc,
            amount=round(amount, 2),
            pos_entry_mode=cons.pos_entry_modes[0],
            **{"3ds_status": rng.choice([t for t in cons.preferred_three_ds]).value},
            ip_address=fake.ipv4_public(),
            ip_country=merchant.country,
            device_id=device,
            stolen_resource=cons.stolen_resource.value,
            timestamp=cursor,
        ))
    return out


def synth_attack_4_card_testing(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """THE ZERO-DAY (never trained on). Bot-farm card testing: ONE shared
    device + egress IP fires rapid small ECOM tickets across dozens of
    victim cards. Tells are structural — infra sharing, merchant velocity,
    and an ECOM/3DS=N amount mix unlike honest checkout traffic."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        device, ip = _new_device(rng), fake.ipv4_public()   # one bot box per wave
        merchants = [
            _merchant_for(env, rng, cons.target_verticals, online=True)
            for _ in range(2)
        ]
        victims = rng.sample(sorted(env.customers.keys()), min(rng.randint(10, 20), len(env.customers)))
        for cid in victims:
            customer = env.customers[cid]
            for _ in range(2):                               # validate + drain attempt
                if len(out) >= n:
                    break
                merchant = rng.choice(merchants)
                known = rng.random() < 0.15                  # some victims' own sessions
                device_use = rng.choice(customer.devices) if known else device
                cursor += timedelta(seconds=rng.randint(20, 60))
                tds = ThreeDSStatus.N if rng.random() < 0.85 else ThreeDSStatus.A
                out.append(PaymentMessage(
                    transaction_id=f"{rng.getrandbits(64):016X}",
                    customer_id=cid,
                    merchant_id=merchant.merchant_id,
                    mcc=merchant.mcc,
                    amount=round(rng.uniform(lo, hi), 2),
                    pos_entry_mode=cons.pos_entry_modes[0],
                    **{"3ds_status": tds.value},
                    ip_address=ip,
                    ip_country=merchant.country,
                    device_id=device_use,
                    stolen_resource=cons.stolen_resource.value,
                    timestamp=cursor,
                ))
    return out


def synth_attack_5_app_scam(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-12 APP SCAM. The hardest family in the taxonomy, because nothing is
    stolen: the GENUINE customer, on their OWN bound device, from a normal IP,
    passing 3DS, authorises the payment themselves.

    Deliberately weak on every classic signal -- known device, 3DS=Y, no infra
    sharing. The only residue is behavioural: a novel high-value beneficiary and
    an escalating 1-3 payment sequence over tens of minutes as the coercion
    proceeds. This family exists to prove the stack is not merely a
    stolen-credential detector."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
        # The victim's OWN device: APP scams do not involve device compromise.
        device = rng.choice(customer.devices)
        ip = fake.ipv4_public()
        # A novel beneficiary the victim has never paid before.
        merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
        # Escalating sequence: the scammer asks for more once the first clears.
        escalation = 1.0
        for _ in range(min(rng.randint(1, 3), n - len(out))):
            cursor += timedelta(seconds=rng.randint(420, 1800))  # minutes of coercion
            amount = min(max(rng.uniform(lo, hi) * escalation, lo), hi)
            escalation *= rng.uniform(1.15, 1.6)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(amount, 2),
                pos_entry_mode=PosEntryMode.CNP,
                # The victim PASSES the challenge. 3DS proves presence, not intent.
                **{"3ds_status": ThreeDSStatus.Y.value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_6_vpa_mule(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-14 VPA-RENTAL MULE. Fan-IN topology, the mirror image of ATTACK_2's
    shared-device ring.

    Here the shared entity is the BENEFICIARY, not the device: many unrelated
    senders, each on their own device and IP, converge on a small rented pool of
    payee endpoints. Per-account velocity stays normal by construction -- each
    sender contributes only one or two events. Only beneficiary-side convergence
    reveals it, which is precisely the signal a per-account monitor cannot see."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        # A small rented pool absorbing a wide fan-in.
        pool = [_merchant_for(env, rng, cons.target_verticals, online=True) for _ in range(rng.randint(1, 2))]
        fan_in = min(rng.randint(9, 16), n - len(out))
        senders = rng.sample(sorted(env.customers.keys()), min(fan_in, len(env.customers)))
        for cid in senders:
            customer = env.customers[cid]
            # Each sender uses their OWN device: no infra sharing on the send side.
            device = rng.choice(customer.devices)
            cursor += timedelta(seconds=rng.randint(45, 240))
            # MCC and country must come from the SAME payee that receives the
            # credit, or the Plausibility Gate rejects it as metadata-incoherent.
            payee = rng.choice(pool)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=cid,
                merchant_id=payee.merchant_id,
                mcc=payee.mcc,
                amount=round(rng.uniform(lo, hi), 2),
                pos_entry_mode=PosEntryMode.CNP,
                **{"3ds_status": ThreeDSStatus.N.value},   # push credit, no challenge
                ip_address=fake.ipv4_public(),
                ip_country=payee.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_7_burst_cashout(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-17 SYNCHRONISED BURST. Breaks the INDEPENDENCE assumption rather than
    any threshold.

    Many accounts, each with its own device and IP, fire inside one tight
    window (seconds apart across the pool). No per-account counter moves: each
    account transacts once or twice. The signature is cross-entity temporal
    clustering, which a per-row classifier structurally cannot represent -- and
    that architectural point is why this family is reported separately."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        # One coordinated wave: a shared merchant pool, a shared instant.
        merchants = [_merchant_for(env, rng, cons.target_verticals, online=False) for _ in range(rng.randint(2, 3))]
        wave = min(rng.randint(12, 22), n - len(out))
        members = rng.sample(sorted(env.customers.keys()), min(wave, len(env.customers)))
        cursor += timedelta(hours=rng.randint(6, 30))     # waves are far apart
        wave_start = cursor
        for cid in members:
            customer = env.customers[cid]
            # Distinct infrastructure per member: nothing is shared but TIME.
            device = _new_device(rng)
            # Sub-minute coordination: the whole wave lands in one window.
            ts = wave_start + timedelta(seconds=rng.randint(0, 90))
            merchant = rng.choice(merchants)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=cid,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(rng.uniform(lo, hi), 2),
                pos_entry_mode=PosEntryMode.CONTACTLESS,
                **{"3ds_status": ThreeDSStatus.N.value},   # card-present physics
                ip_address=fake.ipv4_public(),
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=ts,
            ))
        cursor = wave_start + timedelta(seconds=120)
    return out


def synth_attack_8_threshold_structuring(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-09 LEARNED THRESHOLD STRUCTURING. A model-layer attack: the target is
    the defender's decision boundary, not the payment rail.

    The agent has estimated the review line empirically, so amounts cluster in a
    tight band just under a NON-round discovered value. That is the inversion
    worth noting: legitimate human spending is full of round numbers, and classic
    hand-guessed structuring clusters AT round numbers. Learned structuring
    avoids them, so the residue is an abnormally low round-number frequency plus
    abnormally low amount variance."""
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        # The empirically-discovered ceiling: deliberately not a round number.
        ceiling = round(rng.uniform(lo + (hi - lo) * 0.55, hi), 2)
        band_width = (hi - lo) * rng.uniform(0.04, 0.09)   # tight by design
        customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
        device = _new_device(rng)
        ip = fake.ipv4_public()
        for _ in range(min(rng.randint(3, 6), n - len(out))):
            cursor += timedelta(seconds=rng.randint(300, 2400))  # patient, not bursty
            merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
            amount = ceiling - rng.uniform(0.0, band_width)
            # Actively avoid round numbers: nudge off any multiple of 10.
            if abs(amount - round(amount / 10.0) * 10.0) < 0.5:
                amount -= rng.uniform(1.7, 4.3)
            amount = min(max(amount, lo), hi)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(amount, 2),
                pos_entry_mode=PosEntryMode.ECOM,
                # Aims for the frictionless / attempted exemption band.
                **{"3ds_status": ThreeDSStatus.A.value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_9_otp_relay(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-05 OTP-RELAY VISHING. The authorisation carries a PASSED 3DS challenge.

    This family exists to break the equivalence between "3DS=Y" and "cardholder
    present". The victim really did authenticate -- they read the code to an
    agent on the phone. So the only residue is that the pass arrives from a
    device the customer has never used, seconds after that device first appears.

    Contrast with ATTACK_5 (APP scam), which also carries 3DS=Y but on the
    victim's OWN device. Same challenge outcome, opposite device signal: the
    pair forces the stack to reason about them separately.
    """
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
        # The OPERATOR's hardware, not the victim's: brand new to this customer.
        device = _new_device(rng)
        ip = fake.ipv4_public()
        # The relay window is short: the code expires in about 90 seconds, so
        # the whole monetisation happens inside a few minutes.
        for _ in range(min(rng.randint(1, 3), n - len(out))):
            cursor += timedelta(seconds=rng.randint(40, 210))
            merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(rng.uniform(lo, hi), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                # PASSED challenge. Fraudulently obtained, but genuinely passed.
                **{"3ds_status": ThreeDSStatus.Y.value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_10_exemption_band(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-06 EXEMPTION-BAND ABUSE. A policy-layer attack, not a rail attack.

    Every transaction is small, exemption-eligible and never challenged. No
    individual event is anomalous -- that is the design. The tell is
    distributional: a tight amount band pressed against an invisible policy
    ceiling, at a patient cadence that trips no velocity counter.

    Deliberately the HARDEST family for a per-row scorer in this repository,
    because per-row there is genuinely almost nothing to see.
    """
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
        # A phished credential does not imply a new device in every case: the
        # operator may be replaying from a proxy that resolves to the victim's
        # own fingerprint. Mixed, so the family is not trivially device-detected.
        device = rng.choice(customer.devices) if rng.random() < 0.35 else _new_device(rng)
        ip = fake.ipv4_public()
        for _ in range(min(rng.randint(4, 8), n - len(out))):
            # Patient: tens of minutes apart, never a burst.
            cursor += timedelta(seconds=rng.randint(900, 5400))
            merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                # Hug the ceiling: the whole population sits in a narrow band.
                amount=round(rng.uniform(hi - (hi - lo) * 0.30, hi), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                # Never challenged: the exemption was granted.
                **{"3ds_status": rng.choice([ThreeDSStatus.N, ThreeDSStatus.A]).value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_11_agentic_scope(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-19 AGENT SCOPE EXPANSION. An attack that did not exist pre-GenAI.

    A delegated agent starts inside its mandate and drifts out of it. The
    device is the agent's -- stable, bound and legitimately trusted throughout,
    so device binding provides no signal at all.

    Two signatures a human shopper never produces:
      * a monotonic escalation slope across the sequence, and
      * machine-regular inter-arrival timing (low jitter around a fixed period).
    """
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
        # The agent's bound device: legitimately trusted, never rotated.
        device = rng.choice(customer.devices)
        ip = fake.ipv4_public()
        # Mandate drift: start near the granted floor, escalate past it.
        amount = lo * rng.uniform(1.0, 1.4)
        # Machine cadence: a fixed period with only small jitter.
        period = rng.randint(3600, 9000)
        for _ in range(min(rng.randint(4, 7), n - len(out))):
            cursor += timedelta(seconds=period + rng.randint(-90, 90))
            merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
            amount = min(amount * rng.uniform(1.25, 1.75), hi)
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(max(amount, lo), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                # Stored-credential / merchant-initiated: challenge already held.
                **{"3ds_status": ThreeDSStatus.Y.value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_12_geo_itinerary(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-11 GEO-VELOCITY SPOOF. Defeats impossible-travel rules by satisfying
    them pairwise.

    The sequence is a fabricated itinerary: a travel booking, then spend
    consistent with an origin, then spend after a plausible elapsed flight
    time. Every ADJACENT pair is feasible, so a rule comparing consecutive
    events never fires. Only sequence-level reasoning against the customer's
    own 90-day history reveals that the itinerary is novel.
    """
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        customer = env.customers[f"CUST_{rng.randrange(len(env.customers)):04d}"]
        device = _new_device(rng)
        ip = fake.ipv4_public()
        # Leg 1: the booking itself, which makes everything after it "expected".
        travel = _merchant_for(env, rng, ["travel"], online=True)
        legs = [travel]
        # Legs 2..k: spend along the fabricated route.
        for _ in range(rng.randint(1, 3)):
            legs.append(_merchant_for(env, rng, cons.target_verticals, online=True))
        for merchant in legs:
            if len(out) >= n:
                break
            # A realistic flight-time gap: each pair passes geo-velocity.
            cursor += timedelta(seconds=rng.randint(7200, 21600))
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(rng.uniform(lo, hi), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                **{"3ds_status": rng.choice([ThreeDSStatus.N, ThreeDSStatus.A]).value},
                ip_address=ip,
                # Coherence: geo must come from the merchant the gate checks.
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_13_merchant_bustout(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-04 MERCHANT BUST-OUT. The only ACQUIRING-side family in the repo.

    The victim is the acquirer, not a cardholder. Many unrelated cardholders,
    each on their own device, converge on ONE merchant with high-value tickets
    inside a short window. Per-CARD there is nothing to see -- most of these
    cards touch this merchant exactly once, which is what a first purchase at a
    new store looks like.

    Contrast with ATTACK_3 (compromised merchant): that one is a *real*
    merchant whose endpoint was injected, so the merchant has legitimate
    history. Here the merchant itself is the fraud, and the ticket sizes are
    high rather than normal.
    """
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        # ONE shell MID absorbs the whole wave.
        shell = _merchant_for(env, rng, cons.target_verticals, online=True)
        wave = min(rng.randint(10, 18), n - len(out))
        cardholders = rng.sample(sorted(env.customers.keys()), min(wave, len(env.customers)))
        for cid in cardholders:
            customer = env.customers[cid]
            # Each cardholder on their OWN device: no infra sharing to find.
            device = rng.choice(customer.devices)
            cursor += timedelta(seconds=rng.randint(60, 300))
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=cid,
                merchant_id=shell.merchant_id,
                mcc=shell.mcc,
                # High tickets: the bust-out extracts value before clawback.
                amount=round(rng.uniform(lo + (hi - lo) * 0.45, hi), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                **{"3ds_status": rng.choice([ThreeDSStatus.Y, ThreeDSStatus.N]).value},
                ip_address=fake.ipv4_public(),
                ip_country=shell.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


def synth_attack_14_boundary_probe(
    env: PaymentEnvironment, rng: random.Random, fake: Faker, spec: AttackSpec, n: int
) -> list[PaymentMessage]:
    """T-20 ADVERSARIAL BOUNDARY PROBING. The family that attacks the DEFENDER.

    Two phases, and the two-phase structure IS the signature:

      probe   -- a spread of cheap authorisations sweeping the amount axis,
                 treating approve/decline as an oracle to locate the boundary;
      exploit -- a few high-value transactions wearing the metadata profile
                 that the probes established as safe.

    A per-transaction scorer sees the exploit event as ordinary, because by
    construction it was chosen to look ordinary. Only the probe-then-exploit
    SEQUENCE from shared infrastructure gives it away. This is the family that
    closes the red-team loop: it attacks the artefact the Defend pillar ships.
    """
    cons = spec.constraints
    lo, hi = cons.amount_band
    out: list[PaymentMessage] = []
    cursor = _ATTACK_BASE
    while len(out) < n:
        # One reconnaissance rig: shared device + egress across both phases.
        device, ip = _new_device(rng), fake.ipv4_public()
        victims = rng.sample(sorted(env.customers.keys()), min(rng.randint(6, 12), len(env.customers)))

        # ---- phase 1: probe. Cheap, dense, sweeping the amount axis. ---- #
        probe_ceiling = lo + (hi - lo) * 0.10
        n_probe = min(rng.randint(6, 11), n - len(out))
        for i in range(n_probe):
            cid = victims[i % len(victims)]
            merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
            cursor += timedelta(seconds=rng.randint(15, 75))
            # A deliberate sweep, not a random draw: this is reconnaissance.
            amount = lo + (probe_ceiling - lo) * (i / max(n_probe - 1, 1))
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=cid,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(max(amount, lo), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                **{"3ds_status": ThreeDSStatus.N.value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))

        # ---- phase 2: exploit. Few, high-value, wearing the safe profile. ---- #
        for _ in range(min(rng.randint(2, 4), n - len(out))):
            cid = rng.choice(victims)
            merchant = _merchant_for(env, rng, cons.target_verticals, online=True)
            # A pause: the operator reads the oracle before committing.
            cursor += timedelta(seconds=rng.randint(600, 2400))
            out.append(PaymentMessage(
                transaction_id=f"{rng.getrandbits(64):016X}",
                customer_id=cid,
                merchant_id=merchant.merchant_id,
                mcc=merchant.mcc,
                amount=round(rng.uniform(hi * 0.55, hi), 2),
                pos_entry_mode=PosEntryMode.ECOM,
                # The profile the probes proved survivable.
                **{"3ds_status": ThreeDSStatus.A.value},
                ip_address=ip,
                ip_country=merchant.country,
                device_id=device,
                stolen_resource=cons.stolen_resource.value,
                timestamp=cursor,
            ))
    return out


_SYNTHESIZERS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": synth_attack_1_voice_clone,
    "ATTACK_2_SYNTHETIC_MULE_RING": synth_attack_2_mule_ring,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": synth_attack_3_compromised_merchant,
    "ATTACK_4_CNP_HIGH_VELOCITY": synth_attack_4_card_testing,
    "ATTACK_5_APP_SCAM_PERSONALISED": synth_attack_5_app_scam,
    "ATTACK_6_VPA_RENTAL_MULE": synth_attack_6_vpa_mule,
    "ATTACK_7_SYNCHRONISED_BURST_CASHOUT": synth_attack_7_burst_cashout,
    "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING": synth_attack_8_threshold_structuring,
    # Wave 3: authentication-layer, policy-layer, agentic and model-layer
    # surfaces. Each defeats a control class none of families 1-8 touch.
    "ATTACK_9_OTP_RELAY_VISHING": synth_attack_9_otp_relay,
    "ATTACK_10_EXEMPTION_BAND_ABUSE": synth_attack_10_exemption_band,
    "ATTACK_11_AGENTIC_SCOPE_EXPANSION": synth_attack_11_agentic_scope,
    "ATTACK_12_GEO_VELOCITY_ITINERARY": synth_attack_12_geo_itinerary,
    "ATTACK_13_MERCHANT_BUSTOUT": synth_attack_13_merchant_bustout,
    "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE": synth_attack_14_boundary_probe,
}


# --------------------------------------------------------------------------- #
# Corpus assembly
# --------------------------------------------------------------------------- #


def build_corpus(
    n_legit: int,
    attack_counts: dict[str, int],
    seed: int = 42,
    hard_negative_frac: float = 0.18,
    txns_per_customer: float = 6.0,
) -> dict:
    """Generate a labeled corpus through the live event loop.

    Returns {"rows": [...], "env": PaymentEnvironment} — the env rides along
    so the DecisionEngine gets correct device-binding lookups at inference.

    `hard_negative_frac` reserves that share of the legitimate rows for BENIGN
    ANOMALIES (see legit_generator.build_hard_negatives): real cardholder
    behaviour that moves the same features as fraud — a new phone abroad, a
    shared family tablet, a flash-sale crowd, a subscription batch, a genuine
    big-ticket purchase, a payday standing order.

    This is deliberately adversarial to our OWN detector. Without it the two
    classes separate on `device_known` and `cust_txn_count_10m` alone, recall
    pins at ~0.999, and the benchmark can no longer distinguish a good defence
    from a lucky one. A saturated metric is not evidence. Set to 0.0 to
    reproduce the old easy corpus for comparison.

    `txns_per_customer` fixes the population size at `n_legit / txns_per_customer`
    instead of a hardcoded 1000. This is not cosmetic — it was a silent
    train/serve skew that cost ~7 points of out-of-sample ROC-AUC:

        train: 6000 legit / 1000 customers = 6.0 txns each
        eval:  1200 legit / 1000 customers = 1.2 txns each

    Every sequence-level feature (iat_regularity, amount_escalation,
    amount_band_tightness, low_value_probe_ratio) needs three or more prior
    events for the same entity inside the lookback window. At 6.0 txns per
    customer they fire during training and earn high gain-based importance; at
    1.2 they read their zero-default on ~96% of rows. The model was scored on a
    feature vector that does not exist at evaluation time. Measured effect:
    in-sample ROC-AUC 0.9894 against out-of-sample 0.9251, with four of the
    seven sequence features active on under 5% of eval rows.

    Holding this constant across train / calibration / evaluation is what makes
    those features real rather than decorative. A feature that is non-zero on 2%
    of rows can still rank in the top three by gain -- it splits a small pure
    subset perfectly -- while contributing nothing to held-out recall.
    """
    n_customers = max(20, int(round(n_legit / max(txns_per_customer, 0.1))))
    env = PaymentEnvironment(n_customers=n_customers, seed=seed)
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    n_hard = int(round(n_legit * max(0.0, min(hard_negative_frac, 1.0))))
    n_easy = n_legit - n_hard

    items: list[tuple[PaymentMessage, str]] = []
    for _ in range(n_easy):
        items.append((build_legit_payload(env, rng, fake), "LEGIT"))
    # Benign anomalies carry the SAME label as ordinary legitimate traffic:
    # they are legitimate. The tag rides along only so the evidence layer can
    # report false positives on them separately (that is the insult-cost story).
    for msg in build_hard_negatives(env, rng, fake, n_hard):
        items.append((msg, "LEGIT_HARD"))
    for spec_id, count in attack_counts.items():
        spec = load_attack_spec(SPEC_FILES[spec_id])
        items.extend((m, spec_id) for m in _SYNTHESIZERS[spec_id](env, rng, fake, spec, count))

    # Chronological replay: features must reflect pre-txn state in stream order.
    items.sort(key=lambda pair: pair[0].timestamp)

    extractor = FeatureExtractor(env)
    rows: list[dict] = []
    for msg, attack_id in items:
        result = env.ingest(msg)
        if not result["accepted"]:  # coherent-by-construction: this is a bug alarm
            raise AssertionError(f"{attack_id} payload rejected: {result['reason']} :: {msg.to_wire()}")
        wire = msg.to_wire()
        feats = extractor.features(wire)
        extractor.observe(wire)
        is_legit = attack_id in ("LEGIT", "LEGIT_HARD")
        rows.append({
            "label": 0 if is_legit else 1,
            "attack_id": "LEGIT" if is_legit else attack_id,
            # Retained so the evidence layer can price false positives on
            # benign anomalies separately from ordinary legitimate traffic.
            "legit_kind": ("hard" if attack_id == "LEGIT_HARD" else "easy") if is_legit else None,
            "payload": wire,
            "features": feats,
        })
    return {"rows": rows, "env": env}


# --------------------------------------------------------------------------- #
# CLI: build corpus -> train layers -> save models -> self-evaluation printout
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from collections import Counter

    from defense.decision import APPROVE, DecisionEngine

    # ATTACK_4 (card testing) stays the held-out zero-day: never trained on, so
    # the holdout experiment measures architecture rather than memorisation.
    # Everything else is trained, because the newer families cover control
    # classes (authentication, policy, agentic, acquiring, model-layer) that the
    # original three do not touch at all.
    TRAIN_COUNTS = {
        "ATTACK_1_MFA_RESET_VOICE_CLONE": 150,
        "ATTACK_2_SYNTHETIC_MULE_RING": 150,
        "ATTACK_3_PROMPT_INJECTED_MERCHANT": 150,
        "ATTACK_5_APP_SCAM_PERSONALISED": 120,
        "ATTACK_6_VPA_RENTAL_MULE": 120,
        "ATTACK_7_SYNCHRONISED_BURST_CASHOUT": 120,
        "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING": 120,
        "ATTACK_9_OTP_RELAY_VISHING": 120,
        "ATTACK_10_EXEMPTION_BAND_ABUSE": 120,
        "ATTACK_11_AGENTIC_SCOPE_EXPANSION": 120,
        "ATTACK_12_GEO_VELOCITY_ITINERARY": 120,
        "ATTACK_13_MERCHANT_BUSTOUT": 120,
        "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE": 120,
    }
    # DENSITY IS A PROTOCOL CONSTANT, not a free parameter. Every split below
    # passes the same txns_per_customer, so the sequence-level features carry the
    # same amount of usable history at train, calibration and evaluation time.
    # Letting it drift is a train/serve skew that costs ~7 points of ROC-AUC
    # while every in-sample number still looks excellent (see build_corpus).
    # ~24 transactions per cardholder over the trailing 90 days is roughly two
    # a week: a realistic retail cardholder, and enough history for the
    # behavioural statistics (regularity, escalation, band tightness) to be
    # defined on a meaningful share of rows rather than reading their
    # zero-default. At 6.0 they were dead columns; see build_corpus.
    TXNS_PER_CUSTOMER = 24.0

    print("building training corpus (with hard negatives) ...")
    train = build_corpus(
        n_legit=6000, attack_counts=TRAIN_COUNTS, seed=123,
        txns_per_customer=TXNS_PER_CUSTOMER,
    )

    engine = DecisionEngine(environment=train["env"])
    metrics = engine.train(train["rows"])
    print("xgb:", metrics["xgb"])
    print("iforest:", metrics["iforest"])

    # persist artifacts for the server step / run.sh
    from defense.novelty import DEFAULT_MODEL_PATH as IF_PATH
    from defense.realtime import DEFAULT_MODEL_PATH as XGB_PATH

    print("saved xgb ->", engine.scorer.save(XGB_PATH))
    engine.novelty.train([r for r in train["rows"] if r["label"] == 0], save_path=IF_PATH)

    # CALIBRATION on a third seed, disjoint from both train and eval. The
    # thresholds shipped with the model must come from somewhere measured;
    # hand-set constants silently decay every time the feature set changes.
    # Calibration split matches the EVALUATION split's size and composition, not
    # just its density. Quantile estimation in the far tail is the whole job
    # here: tau sits at the 99th percentile of legitimate scores, so it is
    # determined by the top ~20 rows of a 2,000-row split and inherits their
    # sampling noise. Matching n and the attack mix keeps the estimate on the
    # same footing as the traffic it will police.
    print("building calibration corpus (seed 321, disjoint) ...")
    calib = build_corpus(
        n_legit=2000, attack_counts={k: 30 for k in TRAIN_COUNTS}, seed=321,
        txns_per_customer=TXNS_PER_CUSTOMER,
    )
    calib_info = engine.calibrate(calib["rows"], target_fpr=0.01)
    print("calibration:", calib_info)

    print("building evaluation corpus (fresh seeds) ...")
    ev_counts = {k: 30 for k in TRAIN_COUNTS}
    ev = build_corpus(
        n_legit=2000, attack_counts=ev_counts, seed=777,
        txns_per_customer=TXNS_PER_CUSTOMER,
    )
    # Evaluation gets a FRESH engine that loads the saved models from disk,
    # rather than reusing the training engine's scorer object. Passing
    # `scorer=engine.scorer` looks like a harmless optimisation -- same weights,
    # one less load -- but the scorer owns the FeatureExtractor, and that
    # extractor is still carrying every one of the 7,650 training transactions in
    # its per-customer, per-device and per-merchant history. Those customers
    # reappear in the eval split, so their first eval transaction is scored
    # against months of phantom history, and `merch_first_seen` already knows
    # every merchant. Measured effect: held-out FPR read 4.25% with the shared
    # scorer against 0.85% with a clean one, on identical thresholds. State
    # leakage across the train/eval boundary flatters or wrecks the numbers
    # depending on which way the history points, and either way it is not a
    # measurement of the deployed system.
    engine_eval = DecisionEngine(environment=ev["env"])
    engine_eval.scorer.load(XGB_PATH)
    engine_eval.novelty.load(IF_PATH)
    # Carry ONLY the calibrated operating point across -- never the state.
    for _attr in ("stepup_threshold", "decline_threshold", "manual_threshold",
                  "ring_risk_threshold", "novelty_alone_alerts"):
        setattr(engine_eval, _attr, getattr(engine, _attr))

    records, truths = [], []
    for r in sorted(ev["rows"], key=lambda r: r["payload"]["timestamp"]):
        records.append(engine_eval.decide(r["payload"]))
        truths.append("legit" if r["label"] == 0 else r["attack_id"])

    flagged = sum(1 for rec, t in zip(records, truths) if t == "legit" and rec["decision"] != APPROVE)
    caught = sum(1 for rec, t in zip(records, truths) if t != "legit" and rec["decision"] != APPROVE)
    declined = sum(1 for rec, t in zip(records, truths) if t != "legit" and rec["decision"] == "DECLINE")
    n_legit_ev = sum(1 for t in truths if t == "legit")
    n_attack_ev = len(truths) - n_legit_ev
    cost = engine_eval.compute_cost_matrix(records, ["legit" if t == "legit" else "attack" for t in truths])

    print(f"FPR  : {flagged}/{n_legit_ev} = {flagged / n_legit_ev:.2%}")
    print(f"TPR  : {caught}/{n_attack_ev} = {caught / n_attack_ev:.2%} (non-approve), declined={declined}")
    print("per-attack caught:", Counter(
        t for rec, t in zip(records, truths)
        if t != "legit" and rec["decision"] != APPROVE
    ))
    print("cost matrix:", cost)
