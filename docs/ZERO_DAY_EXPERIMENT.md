# Zero-Day Holdout Experiment

**Research question:** when an attack class the supervised model has *never
seen* hits the stack, does defense collapse — or do the unsupervised layers
hold the line?

## Protocol

| Ingredient | Detail |
|---|---|
| Supervised training set | Legit baseline + ATTACK_1 (voice-clone ATO) + ATTACK_2 (mule ring) + ATTACK_3 (compromised merchant) |
| **Withheld from ALL training** | **ATTACK_4 — CNP high-velocity card testing** |
| Holdout stream (chronological, fresh engine state) | 500 legit transactions, then 50 zero-day transactions |
| Layer attribution (overlapping flags) | XGB: velocity > 0.60 · iForest: anomaly flag · Graph: `ring_detected` |

The zero-day's shape: one bot device + egress IP firing ~$35-$400 ECOM tickets
across dozens of victim cards at 20-60s cadence, mostly skipping 3DS. Every
payload passes the Plausibility Gate by construction — this is operationally
realistic fraud, not garbage input.

## Results

> **The supervised XGBoost model never saw attack 4. It caught 94%.
> The unsupervised Isolation Forest, trained only on legitimate traffic,
> caught 92% as anomalies**. **Overall, the decisioning
> stack stopped 94% of the zero-day campaign while holding false
> positives on held-out legit traffic to 0.8%.**

This demonstrates robustness to novel GenAI-powered attacks: attackers can
invent new campaign shapes faster than we can label them, so the defense
cannot depend on supervision alone.

| Layer | Detection rate on zero-day | Notes |
|---|---|---|
| XGBoost (supervised) | 94% (47/50) | Generalized partially via merchant/device velocity features — no ATTACK_4 rows in training |
| XGBoost at DECLINE threshold (>0.85) | 94% | Conservative read |
| Isolation Forest (unsupervised, legit-only training) | 92% (46/50) | Amount/velocity mix unlike honest checkout traffic |
| EntityGraph (unsupervised) | 42% (21/50) | Shared bot device links >=3 victim profiles; warms up after first contacts |
| **DecisionEngine overall (non-APPROVE)** | **94% (47/50)** | 47 DECLINED, 0 STEP_UP |
| Held-out legit FPR | 0.8% | Within the <2% budget |

Financial view on the zero-day window alone: **$11,002.38 saved**
(declined), **$477.75 lost** (approved). Transactions flagged *only*
by the Isolation Forest — the pure zero-day catches — are highlighted in
amber in `docs/zero_day_results.png` (txn indices: (none — overlaps with other layers)).

## Honest caveats

* Simulated world, synthetic distributions: absolute numbers are directional,
  the ARCHITECTURE claim is the point.
* Layer flags overlap; per-layer rates don't sum to the overall TPR.
* The graph needs 3+ shared-infra contacts before a ring exists — the first
  transactions of any burst are structurally hard.
* One zero-day class was tested. The claim is "the design generalizes to
  unseen classes", demonstrated once, not proven universally.

## Reproduce

```bash
backend/.venv/bin/python backend/experiments/zero_day_holdout.py
```
