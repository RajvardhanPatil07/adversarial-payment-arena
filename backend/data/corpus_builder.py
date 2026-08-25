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

from data.legit_generator import build_legit_payload  # noqa: E402
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
}

# Attacks fire "recently"; legit history trails 90 days. A 12h base keeps
# even the longest synthesized campaign strictly in the past.
_ATTACK_BASE = datetime.now(timezone.utc) - timedelta(hours=12)


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


_SYNTHESIZERS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": synth_attack_1_voice_clone,
    "ATTACK_2_SYNTHETIC_MULE_RING": synth_attack_2_mule_ring,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": synth_attack_3_compromised_merchant,
}


# --------------------------------------------------------------------------- #
# Corpus assembly
# --------------------------------------------------------------------------- #


def build_corpus(
    n_legit: int,
    attack_counts: dict[str, int],
    seed: int = 42,
) -> dict:
    """Generate a labeled corpus through the live event loop.

    Returns {"rows": [...], "env": PaymentEnvironment} — the env rides along
    so the DecisionEngine gets correct device-binding lookups at inference.
    """
    env = PaymentEnvironment(n_customers=1000, seed=seed)
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    items: list[tuple[PaymentMessage, str]] = []
    for _ in range(n_legit):
        items.append((build_legit_payload(env, rng, fake), "LEGIT"))
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
        rows.append({
            "label": 0 if attack_id == "LEGIT" else 1,
            "attack_id": attack_id,
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

    TRAIN_COUNTS = {
        "ATTACK_1_MFA_RESET_VOICE_CLONE": 230,
        "ATTACK_2_SYNTHETIC_MULE_RING": 230,
        "ATTACK_3_PROMPT_INJECTED_MERCHANT": 240,
    }
    print("building training corpus ...")
    train = build_corpus(n_legit=4500, attack_counts=TRAIN_COUNTS, seed=123)

    engine = DecisionEngine(environment=train["env"])
    metrics = engine.train(train["rows"])
    print("xgb:", metrics["xgb"])
    print("iforest:", metrics["iforest"])

    # persist artifacts for the server step / run.sh
    from defense.novelty import DEFAULT_MODEL_PATH as IF_PATH
    from defense.realtime import DEFAULT_MODEL_PATH as XGB_PATH

    print("saved xgb ->", engine.scorer.save(XGB_PATH))
    engine.novelty.train([r for r in train["rows"] if r["label"] == 0], save_path=IF_PATH)

    print("building evaluation corpus (fresh seeds) ...")
    ev_counts = {
        "ATTACK_1_MFA_RESET_VOICE_CLONE": 34,
        "ATTACK_2_SYNTHETIC_MULE_RING": 33,
        "ATTACK_3_PROMPT_INJECTED_MERCHANT": 33,
    }
    ev = build_corpus(n_legit=1000, attack_counts=ev_counts, seed=777)
    engine_eval = DecisionEngine(environment=ev["env"], scorer=engine.scorer, novelty=engine.novelty)

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
