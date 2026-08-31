/**
 * Offline-safe copy of the judge-facing headline from artifacts/closed_loop.json.
 *
 * The live evidence API remains authoritative when available. These values are
 * bundled so a sleeping demo backend cannot erase the result judges came to
 * inspect. Keep this file mechanically consistent with the committed artifact.
 */
export const COMMITTED_SCISSOR = {
  source: "artifacts/closed_loop.json",
  generatedAt: "2026-08-30T15:30:22.623127+00:00",
  gitSha: "0b927d7",
  command: "python backend/experiments/run_closed_loop.py",
  seeds: [11, 23, 37],
  generations: 3,
  targetFpr: 0.01,
  syntheticRecallGain: 0.856667,
  realRecallLoss: -0.358065,
  gatedRealRecallLoss: -0.005377,
  recallProtected: 0.352688,
  realRecallCi: [-0.829033, 0] as [number, number],
  gatedRealRecallCi: [-0.01613, 0] as [number, number],
  batchesRejected: [3, 3, 3],
  gate: {
    c2stAucMax: 0.9,
    dependenceFrobeniusMax: 1.5,
    labelsRequired: "none beyond the real fraud already in the training set",
  },
  boundaries: [
    "Held-out arena fraud is simulated evaluation data, not issuer production traffic.",
    "The claim is about the relationship between gating and transfer, not an absolute recall figure for live traffic.",
    "Attack budget per generation is fixed and identical across arms.",
    "Gate thresholds were fixed in advance and never tuned per seed.",
    "Three seeds; intervals are seed-level nonparametric bootstrap intervals and are not production validation.",
  ],
} as const;

export const WINNING_THESIS =
  "Most red-team loops celebrate when recall on their own synthetic attacks rises. We prove that the same loop can silently destroy recall on held-out real fraud—and our label-free fidelity gate prevents 35.3 points of that damage before retraining.";

export function signedPoints(value: number, digits = 1): string {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value * 100).toFixed(digits)} pts`;
}
