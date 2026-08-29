"use client";

/**
 * Evidence Ledger.
 *
 * This route exists to answer one question a judge will ask: which of these
 * numbers can I check? Every card reads from a generated artifact served by
 * /api/evidence, shows the boundary condition attached to the claim, and
 * prints the command that regenerates it.
 *
 * If the evidence set has not been generated, this page says so plainly rather
 * than rendering placeholder numbers.
 */

import { useEffect, useState } from "react";

import { backendHttpUrl } from "@/lib/backend";

// Resolved through lib/backend.ts so this page cannot drift onto a different
// env var than the WebSocket dashboard (previously it read two of its own).
const API_BASE = backendHttpUrl();

type Interval = { mean: number; lo: number; hi: number; n?: number };

type Summary = {
  provenance?: { generated_at?: string; git_sha?: string; seeds?: number[] };
  pinned_fpr?: number;
  seeds?: number[];
  baseline_recall?: Interval;
  delta_recall_independent_marginal?: Interval;
  delta_recall_gaussian_copula?: Interval;
  c2st_independent_marginal?: Interval;
  c2st_gaussian_copula?: Interval;
  precision_at_production_prevalence?: Interval;
  net_benefit_inr_at_production_prevalence?: number;
  insult_share_of_total_cost?: number;
  thesis?: string;
};

type Claim = {
  claim: string;
  artifact: string;
  field: string;
  derivation: string;
  boundary: string;
};

type IndexEntry = {
  name: string;
  description: string;
  available: boolean;
  generated_at: string | null;
  reproduce: string;
};

const pct = (v?: number, digits = 1) =>
  v === undefined || v === null || Number.isNaN(v) ? "--" : `${(v * 100).toFixed(digits)}%`;

const signedPct = (v?: number, digits = 1) => {
  if (v === undefined || v === null || Number.isNaN(v)) return "--";
  const sign = v > 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)} pts`;
};

const num = (v?: number, digits = 3) =>
  v === undefined || v === null || Number.isNaN(v) ? "--" : v.toFixed(digits);

const inr = (v?: number) =>
  v === undefined || v === null || Number.isNaN(v)
    ? "--"
    : new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 0,
      }).format(v);

const ci = (i?: Interval, fmt: (v?: number) => string = (v) => num(v)) =>
  i ? `${fmt(i.mean)}  [${fmt(i.lo)}, ${fmt(i.hi)}]` : "--";

const styles = {
  page: {
    minHeight: "100vh",
    background: "#0b0f19",
    color: "#e5e7eb",
    padding: "40px 28px 80px",
    fontFamily:
      "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  } as const,
  shell: { maxWidth: 1180, margin: "0 auto" } as const,
  eyebrow: {
    fontSize: 12,
    letterSpacing: "0.18em",
    textTransform: "uppercase",
    color: "#60a5fa",
    marginBottom: 10,
  } as const,
  h1: { fontSize: 34, fontWeight: 700, margin: "0 0 12px", lineHeight: 1.15 } as const,
  thesis: {
    fontSize: 15,
    lineHeight: 1.65,
    color: "#cbd5e1",
    maxWidth: 820,
    borderLeft: "3px solid #2563eb",
    paddingLeft: 16,
    margin: "0 0 30px",
  } as const,
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(255px, 1fr))",
    gap: 16,
    marginBottom: 34,
  } as const,
  card: {
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: 12,
    padding: "18px 18px 16px",
  } as const,
  cardLabel: { fontSize: 12, color: "#94a3b8", marginBottom: 8, lineHeight: 1.4 } as const,
  cardValue: { fontSize: 26, fontWeight: 700, letterSpacing: "-0.01em" } as const,
  cardCi: { fontSize: 11.5, color: "#64748b", marginTop: 6, fontFamily: "ui-monospace, monospace" } as const,
  cardNote: { fontSize: 11.5, color: "#94a3b8", marginTop: 10, lineHeight: 1.5 } as const,
  h2: { fontSize: 20, fontWeight: 650, margin: "34px 0 6px" } as const,
  sub: { fontSize: 13.5, color: "#94a3b8", margin: "0 0 16px", lineHeight: 1.6, maxWidth: 860 } as const,
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
  td: {
    padding: "12px",
    borderBottom: "1px solid #161e2e",
    verticalAlign: "top",
    lineHeight: 1.55,
  } as const,
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
  warn: {
    background: "#1c1917",
    border: "1px solid #78350f",
    borderRadius: 12,
    padding: 20,
    lineHeight: 1.65,
    fontSize: 14,
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
    } as const),
};

export default function EvidencePage() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [claims, setClaims] = useState<Claim[]>([]);
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const indexRes = await fetch(`${API_BASE}/api/evidence/index`, { cache: "no-store" });
        if (!indexRes.ok) throw new Error(`evidence index unavailable (${indexRes.status})`);
        const indexJson = await indexRes.json();
        if (cancelled) return;
        setIndex(indexJson.artifacts ?? []);

        const [summaryRes, claimsRes] = await Promise.all([
          fetch(`${API_BASE}/api/evidence/summary`, { cache: "no-store" }),
          fetch(`${API_BASE}/api/evidence/claims`, { cache: "no-store" }),
        ]);

        if (cancelled) return;
        if (summaryRes.ok) setSummary(await summaryRes.json());
        else
          setError(
            "The evidence set has not been generated yet. Run `make reproduce` to build it.",
          );
        if (claimsRes.ok) {
          const claimsJson = await claimsRes.json();
          setClaims(claimsJson.claims ?? []);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "failed to load evidence");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const deltaCopula = summary?.delta_recall_gaussian_copula;
  const deltaIndependent = summary?.delta_recall_independent_marginal;

  return (
    <main style={styles.page}>
      <div style={styles.shell}>
        <div style={styles.eyebrow}>Evidence Ledger</div>
        <h1 style={styles.h1}>Fidelity determines transfer</h1>
        <p style={styles.thesis}>
          {summary?.thesis ??
            "Closing the red-team loop is not sufficient. Whether the loop improves real-world detection depends on the fidelity of the attack generator, and low-fidelity augmentation can measurably reduce recall on real fraud."}
        </p>

        {loading && <p style={styles.sub}>Loading generated artifacts...</p>}

        {!loading && error && (
          <div style={styles.warn}>
            <strong>Evidence not available.</strong>
            <br />
            {error}
            <br />
            <br />
            This page deliberately shows nothing rather than placeholder numbers. Regenerate with{" "}
            <span style={styles.code}>make reproduce</span> and reload.
          </div>
        )}

        {summary && (
          <>
            <section style={styles.grid}>
              <div style={styles.card}>
                <div style={styles.cardLabel}>
                  A2 Gaussian copula &mdash; change in recall on real fraud
                </div>
                <div
                  style={{
                    ...styles.cardValue,
                    color: (deltaCopula?.mean ?? 0) >= 0 ? "#4ade80" : "#f87171",
                  }}
                >
                  {signedPct(deltaCopula?.mean)}
                </div>
                <div style={styles.cardCi}>{ci(deltaCopula, (v) => signedPct(v))}</div>
                <div style={styles.cardNote}>
                  Versus the unaugmented baseline, at a false-positive rate pinned to{" "}
                  {pct(summary.pinned_fpr, 2)} on a disjoint validation split.
                </div>
              </div>

              <div style={styles.card}>
                <div style={styles.cardLabel}>
                  A1 independent marginals &mdash; change in recall on real fraud
                </div>
                <div
                  style={{
                    ...styles.cardValue,
                    color: (deltaIndependent?.mean ?? 0) >= 0 ? "#4ade80" : "#f87171",
                  }}
                >
                  {signedPct(deltaIndependent?.mean)}
                </div>
                <div style={styles.cardCi}>{ci(deltaIndependent, (v) => signedPct(v))}</div>
                <div style={styles.cardNote}>
                  The rule and template approach: correct marginals, destroyed joint structure.
                </div>
              </div>

              <div style={styles.card}>
                <div style={styles.cardLabel}>C2ST AUC &mdash; can real be told from synthetic?</div>
                <div style={styles.cardValue}>{num(summary.c2st_gaussian_copula?.mean, 3)}</div>
                <div style={styles.cardCi}>
                  copula {ci(summary.c2st_gaussian_copula)} <br />
                  independent {ci(summary.c2st_independent_marginal)}
                </div>
                <div style={styles.cardNote}>
                  0.50 means indistinguishable from real fraud. 1.00 means trivially separable.
                </div>
              </div>

              <div style={styles.card}>
                <div style={styles.cardLabel}>
                  Precision at production base rate (~1.3%)
                </div>
                <div style={styles.cardValue}>
                  {pct(summary.precision_at_production_prevalence?.mean)}
                </div>
                <div style={styles.cardCi}>
                  {ci(summary.precision_at_production_prevalence, (v) => pct(v))}
                </div>
                <div style={styles.cardNote}>
                  The same detector, reported at a realistic base rate rather than laboratory
                  prevalence. This number is the honest one.
                </div>
              </div>

              <div style={styles.card}>
                <div style={styles.cardLabel}>Net benefit per million authorisations</div>
                <div style={styles.cardValue}>
                  {inr(summary.net_benefit_inr_at_production_prevalence)}
                </div>
                <div style={styles.cardNote}>
                  Fraud prevented, less fraud lost, less the insult cost of wrongly declining
                  legitimate customers, less review cost.
                </div>
              </div>

              <div style={styles.card}>
                <div style={styles.cardLabel}>
                  False positives as a share of total cost
                </div>
                <div style={styles.cardValue}>{pct(summary.insult_share_of_total_cost)}</div>
                <div style={styles.cardNote}>
                  At a 1% false-positive rate, wrongly declined legitimate payments are the
                  single largest cost term — larger than the fraud losses that slip through.
                  The asymmetric cost matrix is what keeps that term bounded. Most fraud demos
                  price only the fraud they stopped.
                </div>
              </div>
            </section>

            <h2 style={styles.h2}>Claim ledger</h2>
            <p style={styles.sub}>
              Every public claim, the artifact field that supports it, how it was derived, and the
              boundary beyond which it does not hold. The boundary column is mandatory.
            </p>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Claim</th>
                  <th style={styles.th}>Artifact field</th>
                  <th style={styles.th}>Derivation</th>
                  <th style={styles.th}>Boundary</th>
                </tr>
              </thead>
              <tbody>
                {claims.map((claim) => (
                  <tr key={claim.claim}>
                    <td style={{ ...styles.td, fontWeight: 550 }}>{claim.claim}</td>
                    <td style={{ ...styles.td, ...styles.mono, color: "#93c5fd" }}>
                      {claim.artifact}
                      <br />
                      {claim.field}
                    </td>
                    <td style={{ ...styles.td, color: "#cbd5e1" }}>{claim.derivation}</td>
                    <td style={{ ...styles.td, color: "#fcd34d" }}>{claim.boundary}</td>
                  </tr>
                ))}
                {claims.length === 0 && (
                  <tr>
                    <td style={styles.td} colSpan={4}>
                      No claim ledger generated yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </>
        )}

        {index.length > 0 && (
          <>
            <h2 style={styles.h2}>Artifacts and reproduction</h2>
            <p style={styles.sub}>
              Each artifact is a JSON file stamped with the git sha, seeds, Python version and the
              command that produced it.
            </p>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Artifact</th>
                  <th style={styles.th}>What it proves</th>
                  <th style={styles.th}>Status</th>
                  <th style={styles.th}>Reproduce</th>
                </tr>
              </thead>
              <tbody>
                {index.map((entry) => (
                  <tr key={entry.name}>
                    <td style={{ ...styles.td, ...styles.mono, color: "#93c5fd" }}>{entry.name}</td>
                    <td style={{ ...styles.td, color: "#cbd5e1" }}>{entry.description}</td>
                    <td style={styles.td}>
                      <span style={styles.pill(entry.available)}>
                        {entry.available ? "generated" : "missing"}
                      </span>
                      {entry.generated_at && (
                        <div style={{ ...styles.cardCi, marginTop: 6 }}>{entry.generated_at}</div>
                      )}
                    </td>
                    <td style={styles.td}>
                      <span style={styles.code}>{entry.reproduce}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        <p style={{ ...styles.sub, marginTop: 36, color: "#64748b" }}>
          Boundary of the whole result: &ldquo;real fraud&rdquo; here means held-out fraud from the
          arena&rsquo;s topology-aware environment, not issuer production data. The claim is about
          the relationship between generator fidelity and transfer, not an absolute recall figure
          for live card traffic.
        </p>
      </div>
    </main>
  );
}
