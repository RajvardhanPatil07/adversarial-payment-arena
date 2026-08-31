# Judge Guide — 90 Seconds to the Point

> **Most red-team loops celebrate when recall on their own synthetic attacks rises. We prove that
> the same loop can silently destroy recall on held-out real fraud—and our label-free fidelity gate
> prevents 35.3 points of that damage before retraining.**

**Submission links:** **Live Demo — PLACEHOLDER: add final public URL** ·
[`90-second Judge Path`](#90-second-judge-path) ·
[`Evidence`](../artifacts/closed_loop.json) ·
[`Walkthrough`](Solution_Walkthrough_Adversarial_Payment_Arena.docx) ·
[`Reproduce`](#reproduce) ·
**Video — PLACEHOLDER: add final public URL** ·
**PDF — PLACEHOLDER: add final public URL**

## The three numbers

| What moved | Current committed result |
|---|---:|
| Recall on the loop’s own synthetic attacks | **+85.7 pts** |
| Recall on held-out arena fraud | **−35.8 pts** |
| Real-fraud recall protected by the fidelity gate | **+35.3 pts** |

**Boundary:** held-out arena fraud is simulated evaluation data from the arena, not issuer
production traffic. The submission demonstrates a controlled failure mode and prevention mechanism;
it does not claim production validation.

## 90-second Judge Path

1. Open the homepage. The fidelity-scissor sentence and all three numbers should be legible within
   ten seconds, even if the live backend is still waking.
2. Select **Start the 90-second demo**. If the engine is cold, the action queues and the page says
   “Waking the live defense engine—usually 10–15 seconds.”
3. Follow the event-driven narration: attacker generation → plausibility gate → graph discovery →
   defense action → financial outcome. Each step advances only after the matching application event.
4. When the campaign completes, select **Verify the evidence**. The evidence page opens with the
   fidelity scissor, followed by provenance, seeds, boundaries, claim ledger, and reproduction commands.

## Why the gate works before retraining

The gate compares each synthetic escape batch with known fraud structure using C2ST and Spearman
rank-dependence distance. These checks require no new outcome labels and run before retraining, so an
issuer can reject a low-fidelity batch before it degrades the detector.

## Evidence and boundaries

- Headline artifact: [`artifacts/closed_loop.json`](../artifacts/closed_loop.json), generated with seeds
  11, 23, and 37 by `python backend/experiments/run_closed_loop.py`.
- Claim mapping: [`artifacts/claim_ledger.json`](../artifacts/claim_ledger.json).
- The ungated arm’s −35.8-point interval is wide (three seeds: [−82.9, 0.0] pts). We report the
  observed mean and interval; we do not present it as production confidence.
- The gated arm changes held-out arena-fraud recall by −0.5 pts, so the measured protected difference
  is +35.3 pts under the fixed simulator, attacker budget, detector, and thresholds.
- The separate transfer ablation is at a recall ceiling. It is retained as an honest negative result,
  not used to inflate the headline.

## Reproduce

```bash
make reproduce
```

Every generated artifact records the git SHA, Python version, seeds, and exact command. For the
headline alone, run `python backend/experiments/run_closed_loop.py`.

## 60-second signed-out-browser audit

- **0–10s:** open the final URL in a private window; confirm the thesis, +85.7, −35.8, +35.3, and
  simulated-data boundary are visible without scrolling.
- **10–20s:** confirm **Start the 90-second demo** is actionable. On cold start, verify the waking copy;
  on failure, verify the retry control and working Evidence/Reproduce links.
- **20–35s:** start the demo; confirm narration advances on visible attacker, gate, graph, and defense events.
- **35–45s:** confirm the ring becomes visually connected and defense decisions/financial outcome update.
- **45–55s:** open **Verify the evidence**; confirm the scissor appears first and the artifact timestamp,
  git SHA, seeds, boundaries, and command are present.
- **55–60s:** check mobile width, keyboard focus, browser console, and that no authenticated session or API key
  is required for the deterministic offline demo.

## 90-second presentation script

**0–12s — Thesis.** “Most red-team loops celebrate when recall on their own synthetic attacks rises.
Ours exposes the dangerous opposite movement: synthetic recall rises 85.7 points while recall on
held-out arena fraud falls 35.8 points.”

**12–20s — Boundary.** “Held-out arena fraud is simulated evaluation data, not issuer production
traffic. This is a controlled, reproducible safety result—not a production-validation claim.”

**20–28s — Start.** “I’ll launch one synthetic mule-ring campaign. The walkthrough advances on real
application events, not a timer.” Select **Start the 90-second demo**.

**28–48s — Attack and plausibility.** “The attacker generates payments constrained by fraud economics,
metadata, and payment-rail rules. Passing the plausibility gate means each payment is coherent; it does
not mean the batch is safe training data.”

**48–63s — Graph and defense.** “Individually these payments look ordinary. Shared devices and IPs make
the coordinated ring visible, and the defense contains it while tracking fraud loss and customer friction.”

**63–75s — Scissor.** “The tempting next step is to retrain on every escape. That is exactly where the
ungated loop fails: its own synthetic scoreboard improves while held-out arena-fraud recall collapses.”

**75–84s — Prevention.** “Our label-free fidelity gate checks the escape batch before retraining and
rejects unsafe synthetic data, protecting 35.3 points of real-fraud recall in this experiment.”

**84–90s — Proof.** Select **Verify the evidence**. “Every number maps to a committed artifact with seeds,
provenance, boundaries, and one-command reproduction.”
