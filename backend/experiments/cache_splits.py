"""Cache the calibration and evaluation splits so diagnostics don't rebuild them.

Density (txns_per_customer) is held identical across splits on purpose -- see
build_corpus's docstring for why letting it drift silently cost ~7 points of
out-of-sample ROC-AUC.

Run: python3 experiments/cache_splits.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.corpus_builder import build_corpus  # noqa: E402

FAMILIES = [
    "ATTACK_1_MFA_RESET_VOICE_CLONE",
    "ATTACK_2_SYNTHETIC_MULE_RING",
    "ATTACK_3_PROMPT_INJECTED_MERCHANT",
    "ATTACK_5_APP_SCAM_PERSONALISED",
    "ATTACK_6_VPA_RENTAL_MULE",
    "ATTACK_7_SYNCHRONISED_BURST_CASHOUT",
    "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING",
    "ATTACK_9_OTP_RELAY_VISHING",
    "ATTACK_10_EXEMPTION_BAND_ABUSE",
    "ATTACK_11_AGENTIC_SCOPE_EXPANSION",
    "ATTACK_12_GEO_VELOCITY_ITINERARY",
    "ATTACK_13_MERCHANT_BUSTOUT",
    "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE",
]

TXNS_PER_CUSTOMER = 24.0

SPLITS = {
    "calib": dict(n_legit=2000, attack_counts={k: 15 for k in FAMILIES}, seed=321),
    "eval": dict(n_legit=2000, attack_counts={k: 30 for k in FAMILIES}, seed=777),
}


def main() -> None:
    for name, kw in SPLITS.items():
        c = build_corpus(txns_per_customer=TXNS_PER_CUSTOMER, **kw)
        with open(f"/tmp/split_{name}.pkl", "wb") as fh:
            pickle.dump(c, fh)
        pos = sum(r["label"] for r in c["rows"])
        print("%-6s %5d rows  %4d positives  %4d customers"
              % (name, len(c["rows"]), pos, len(c["env"].customers)))


if __name__ == "__main__":
    main()
