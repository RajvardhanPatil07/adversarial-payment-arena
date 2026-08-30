# Judge's Guide — Five Minutes to the Point

Everything below is live in the repo. No number here is typed by hand; each is
emitted by `make reproduce` into `artifacts/` and mapped in
[`artifacts/claim_ledger.json`](../artifacts/claim_ledger.json).

## The one idea

> **A closed-loop red team without a fidelity gate is an attack surface, not a
> feature.** Folding a low-fidelity generator's escapes back into training makes
> every dashboard number improve while recall on *real* fraud falls. A label-free
> fidelity gate, computable before retraining, removes that failure mode.

## See it in 60 seconds

1. Open the dashboard and press **▶ GUIDED DEMO**. A synthetic mule ring is built
   live; the narration advances only on real events (gate verdicts, graph ring
   detection, declines, final cost).
2. Open **/evidence**. The *fidelity scissor* panel shows the headline: an ungated
   loop loses **−35.8 pts** of real-fraud recall while gaining **+86 pts** on its own
   synthetic attacks; the same loop with the gate on loses **−0.5 pts**.

## The three numbers that carry the claim

| Claim | Number (current artifacts) | Reproduce |
|---|---|---|
| Fidelity separates the generators | C2ST AUC **0.873** (copula) vs **0.964** (independent); 0.5 = indistinguishable from real | `python backend/experiments/run_fidelity.py` |
| An ungated closed loop degrades real-fraud recall | **−35.8 pts** real recall while **+86 pts** on its own attacks (the scissor) | `python backend/experiments/run_closed_loop.py` |
| The gate protects that recall | gated arm **−0.5 pts**; **+35.3 pts** of recall protected, gate needs no fraud labels | `python backend/experiments/run_closed_loop.py` |

Supporting economics, at a 1.3% production base rate and 1% FPR: net
**₹22.9Cr per million authorisations**, with wrongly-declined legitimate payments
**≈ 60%** of total cost — which is why the cost matrix is asymmetric.
The four-layer stack scores inline at **p99 15.1 ms** against a 100 ms
authorisation budget.

## The honest boundaries (we state these so you don't have to find them)

- "Real fraud" here is the arena's topology-aware held-out fraud, not issuer
  production data. The claim is about the *relationship* between fidelity and
  transfer, not an absolute live-traffic recall figure.
- On this corpus the unaugmented baseline sits at ~0.996 recall, so the transfer
  ablation is at a ceiling: both synthetic arms show ~0 delta this run. We report
  the ceiling rather than manufacture a positive delta; the closed-loop scissor is
  the operational demonstration of the gate's value.
- Three seeds; every figure is a seed-level mean with a nonparametric bootstrap CI.

## Verify any number yourself

```bash
make reproduce   # regenerates every artifact in artifacts/ with a provenance stamp
```

Each artifact records the git SHA, seeds, Python version and the exact command that
produced it. The `/evidence` page prints the reproduce command next to every claim.
