"use client";

/**
 * HardeningLab — the fidelity scissor, rendered from artifacts only.
 *
 * Reads /api/evidence/closed-loop (backend allow-lists the closed_loop.json
 * artifact produced by `python backend/experiments/run_closed_loop.py`) and
 * renders gated vs ungated retraining side-by-side. Like the rest of the
 * evidence page, a missing artifact produces a reproduce instruction, never
 * a placeholder number.
 */

import { useEffect, useState } from "react";

import { backendHttpUrl } from "@/lib/backend";
import { COMMITTED_SCISSOR } from "@/lib/committed-evidence";

type Arm = {
  arm: string;
  gated: boolean;
  generator: string;
  delta_real_recall: number;
  delta_real_recall_ci: [number, number];
  delta_synthetic_recall: number;
  batches_rejected_by_gate: number[];
};

type ClosedLoop = {
  provenance?: { generated_at?: string; git_sha?: string; seeds?: number[]; command?: string };
  question?: string;
  protocol?: { seeds?: number[]; generations?: number; target_fpr?: number };
  gate?: {
    c2st_auc_max?: number;
    dependence_frobenius_max?: number;
    labels_required?: string;
    computable_before_retraining?: boolean;
    why?: string;
  };
  arms?: Arm[];
  low_fidelity_generator?: {
    ungated_delta_real_recall?: number;
    gated_delta_real_recall?: number;
    recall_protected_by_gate?: number;
    ungated_delta_synthetic_recall?: number;
  };
  reading?: string;
  boundaries?: string[];
  scissor?: {
    synthetic_recall_gain_ungated?: number;
    real_recall_loss_ungated?: number;
    real_recall_loss_gated?: number;
    recall_protected_by_gate?: number;
  };
};

const signedPts = (v?: number, digits = 1) => {
  if (v === undefined || v === null || Number.isNaN(v)) return "--";
  const sign = v > 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)} pts`;
};

const styles = {
  h2: { fontSize: 20, fontWeight: 650, margin: "34px 0 6px" } as const,
  sub: { fontSize: 13.5, color: "#94a3b8", margin: "0 0 16px", lineHeight: 1.6, maxWidth: 860 } as const,
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16, marginBottom: 20 } as const,
  card: { background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: "18px 18px 16px" } as const,
  cardLabel: { fontSize: 12, color: "#94a3b8", marginBottom: 8, lineHeight: 1.4 } as const,
  cardValue: { fontSize: 26, fontWeight: 700, letterSpacing: "-0.01em" } as const,
  cardCi: { fontSize: 11.5, color: "#64748b", marginTop: 6, fontFamily: "ui-monospace, monospace" } as const,
  cardNote: { fontSize: 11.5, color: "#94a3b8", marginTop: 10, lineHeight: 1.5 } as const,
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 } as const,
  th: {
    textAlign: "left",
    padding: "10px 12px",
    borderBottom: "1px solid #1f2937",
    color: "#94a3b8",
    fontWeight: 600,
    fontSize: 11.5,
    letterSpacing: "0.06em",
    textTransform: "uppercase",
  } as const,
  td: { padding: "12px", borderBottom: "1px solid #161e2e", verticalAlign: "top", lineHeight: 1.55 } as const,
  mono: { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 11.5 } as const,
  code: {
    display: "inline-block",
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 6,
    padding: "3px 8px",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: 11.5,
    color: "#93c5fd",
  } as const,
  pill: (ok: boolean) =>
    ({
      display: "inline-block",
      fontSize: 11,
      fontWeight: 600,
      padding: "2px 9px",
      borderRadius: 999,
      background: ok ? "#052e1a" : "#2a1113",
      color: ok ? "#4ade80" : "#fca5a5",
      border: `1px solid ${ok ? "#166534" : "#7f1d1d"}`,
    }) as const,
};

export function HardeningLab() {
  const [data, setData] = useState<ClosedLoop>({
    provenance: {
      generated_at: COMMITTED_SCISSOR.generatedAt,
      git_sha: COMMITTED_SCISSOR.gitSha,
      seeds: [...COMMITTED_SCISSOR.seeds],
      command: COMMITTED_SCISSOR.command,
    },
    protocol: {
      seeds: [...COMMITTED_SCISSOR.seeds],
      generations: COMMITTED_SCISSOR.generations,
      target_fpr: COMMITTED_SCISSOR.targetFpr,
    },
    gate: {
      c2st_auc_max: COMMITTED_SCISSOR.gate.c2stAucMax,
      dependence_frobenius_max: COMMITTED_SCISSOR.gate.dependenceFrobeniusMax,
      labels_required: COMMITTED_SCISSOR.gate.labelsRequired,
      computable_before_retraining: true,
    },
    low_fidelity_generator: {
      ungated_delta_real_recall: COMMITTED_SCISSOR.realRecallLoss,
      gated_delta_real_recall: COMMITTED_SCISSOR.gatedRealRecallLoss,
      recall_protected_by_gate: COMMITTED_SCISSOR.recallProtected,
      ungated_delta_synthetic_recall: COMMITTED_SCISSOR.syntheticRecallGain,
    },
    boundaries: [...COMMITTED_SCISSOR.boundaries],
  });
  const [usingCommittedFallback, setUsingCommittedFallback] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch(`${backendHttpUrl()}/api/evidence/closed-loop`, { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setUsingCommittedFallback(false);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const low = data.low_fidelity_generator ?? {};
  const arms = data.arms ?? [];

  return (
    <>
      <h2 id="fidelity-scissor" style={styles.h2}>The fidelity scissor</h2>
      <p style={styles.sub}>
        The label-free gate compares each synthetic escape batch with known fraud structure using
        C2ST and rank dependence. Because both checks run before retraining and need no new outcome
        labels, an unsafe batch can be rejected before it damages the detector.
      </p>

      <section style={styles.grid}>
        <div style={{ ...styles.card, border: "1px solid #7f1d1d" }}>
          <div style={styles.cardLabel}>
            Synthetic attack recall rises
          </div>
          <div style={{ ...styles.cardValue, color: "#4ade80" }}>
            {signedPts(low.ungated_delta_synthetic_recall)}
            <span style={{ fontSize: 13, color: "#94a3b8", fontWeight: 500 }}> on synthetic attacks</span>
          </div>
          <div style={{ ...styles.cardValue, color: "#f87171", fontSize: 20, marginTop: 4 }}>
            {signedPts(low.ungated_delta_real_recall)}
            <span style={{ fontSize: 13, color: "#94a3b8", fontWeight: 500 }}>
              {" "}on held-out arena fraud
            </span>
          </div>
          <div style={styles.cardNote}>
            The two yardsticks move in opposite directions in the ungated loop. Held-out arena fraud is
            simulated evaluation data, not issuer production traffic.
          </div>
        </div>

        <div style={{ ...styles.card, border: "1px solid #166534" }}>
          <div style={styles.cardLabel}>Same loop, fidelity gate on — the gate refuses the escape batches</div>
          <div style={{ ...styles.cardValue, color: "#4ade80" }}>
            {signedPts(low.recall_protected_by_gate)}
            <span style={{ fontSize: 13, color: "#94a3b8", fontWeight: 500 }}> real-fraud recall protected</span>
          </div>
          <div style={{ ...styles.cardValue, color: "#93c5fd", fontSize: 20, marginTop: 4 }}>
            {signedPts(low.gated_delta_real_recall)}
            <span style={{ fontSize: 13, color: "#94a3b8", fontWeight: 500 }}> gated change on held-out arena fraud</span>
          </div>
          <div style={styles.cardNote}>
            {data.gate?.why ??
              "Both gate metrics are label-free and computable on the escape batch alone, before retraining."}
          </div>
        </div>
      </section>

      {data.gate && (
        <p style={styles.sub}>
          The gate (fixed in advance, never tuned per seed): C2ST AUC ≤ {data.gate.c2st_auc_max} · rank-dependence
          error ≤ {data.gate.dependence_frobenius_max} · labels required:{" "}
          {data.gate.labels_required ?? "none"}.{" "}
          {data.gate.computable_before_retraining
            ? "Computable before retraining — an issuer can reject a generator without first degrading a live detector."
            : ""}
        </p>
      )}

      {arms.length > 0 && (
        <div style={{ overflowX: "auto", marginBottom: 20 }}><table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Arm</th>
              <th style={styles.th}>Generator</th>
              <th style={styles.th}>Gate</th>
              <th style={styles.th}>Δ recall, real fraud</th>
              <th style={styles.th}>Δ recall, synthetic</th>
              <th style={styles.th}>Batches rejected</th>
            </tr>
          </thead>
          <tbody>
            {arms.map((a) => (
              <tr key={a.arm}>
                <td style={{ ...styles.td, ...styles.mono, color: "#93c5fd" }}>{a.arm}</td>
                <td style={{ ...styles.td, color: "#cbd5e1" }}>{a.generator}</td>
                <td style={styles.td}>
                  <span style={styles.pill(a.gated)}>{a.gated ? "gated" : "ungated"}</span>
                </td>
                <td
                  style={{
                    ...styles.td,
                    ...styles.mono,
                    color: a.delta_real_recall < -0.05 ? "#f87171" : "#4ade80",
                  }}
                >
                  {signedPts(a.delta_real_recall)}
                  <div style={styles.cardCi}>
                    [{signedPts(a.delta_real_recall_ci?.[0])}, {signedPts(a.delta_real_recall_ci?.[1])}]
                  </div>
                </td>
                <td style={{ ...styles.td, ...styles.mono, color: "#cbd5e1" }}>
                  {signedPts(a.delta_synthetic_recall)}
                </td>
                <td style={{ ...styles.td, ...styles.mono, color: "#cbd5e1" }}>
                  {a.batches_rejected_by_gate?.join(" / ") ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table></div>
      )}

      {data.reading && (
        <p style={{ ...styles.sub, borderTop: "1px solid #2563eb", paddingTop: 14 }}>{data.reading}</p>
      )}

      {data.boundaries && data.boundaries.length > 0 && (
        <ul style={{ ...styles.sub, paddingLeft: 18, marginBottom: 8 }}>
          {data.boundaries.map((b) => (
            <li key={b} style={{ marginBottom: 4 }}>
              {b}
            </li>
          ))}
        </ul>
      )}

      <p style={styles.sub}>
        Reproduce: <span style={styles.code}>python backend/experiments/run_closed_loop.py</span> — seeds{" "}
        {data.protocol?.seeds?.join(", ") ?? "—"}, {data.protocol?.generations ?? "—"} generations, FPR pinned
        at {data.protocol?.target_fpr ?? "—"}. Artifact <span style={styles.code}>artifacts/closed_loop.json</span>,
        git <span style={styles.code}>{data.provenance?.git_sha ?? COMMITTED_SCISSOR.gitSha}</span>.
        {usingCommittedFallback ? " Showing the committed artifact snapshot while the live evidence API wakes." : ""}
      </p>
    </>
  );
}
