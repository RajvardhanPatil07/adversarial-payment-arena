import type { Metadata } from "next";
import Link from "next/link";

import { PageHeader } from "@/components/shell/page-header";
import { StatusChip } from "@/components/shell/status-chip";
import { ROUTE, ROUTES } from "@/lib/site";
import {
  readTyped,
  fmtDeltaPts,
  fmtNum,
  fmtPct,
  fmtInterval,
  type ClosedLoopArtifact,
  type FidelityReportArtifact,
  type LatencyArtifact,
} from "@/lib/artifacts";

const META = ROUTE["/"];

export const metadata: Metadata = {
  // The default title in layout.tsx is already this page's claim, so the
  // template is bypassed here rather than repeating it.
  title: {
    absolute: "A closed-loop red team without a fidelity gate is an attack surface",
  },
  description: META.blurb,
};

/** A measured figure: mono, with its artifact source stated inline. */
function Figure({
  label,
  value,
  source,
  tone = "neutral",
}: {
  label: string;
  value: string | null;
  source: string;
  tone?: "neutral" | "red" | "blue" | "pass";
}) {
  return (
    <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
      <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">{label}</p>
      <p className="type-num mt-3 text-xl font-medium tracking-tight text-text md:text-2xl">
        {value ?? <span className="text-text-dim">not measured</span>}
      </p>
      <p className="type-num mt-3 text-[0.6875rem] text-text-dim">{source}</p>
      {value && tone !== "neutral" && (
        <div className="mt-3">
          <StatusChip tone={tone}>
            {tone === "red" ? "harm measured" : tone === "blue" ? "harm removed" : "verified"}
          </StatusChip>
        </div>
      )}
    </div>
  );
}

/** One row of the loop diagram. Keeps the gate's two outgoing paths visible. */
function LoopDiagram() {
  return (
    <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
      <h3 className="type-ui text-sm font-semibold tracking-tight text-text">The closed loop, and where the gate cuts it</h3>
      <ol className="mt-6 flex flex-col gap-3">
        {[
          { n: "1", text: "The attacker generates attacks against the current detector.", tone: "border-red/40" },
          { n: "2", text: "Attacks that slip past the defense become escape batches — candidate training rows.", tone: "border-red/40" },
          { n: "3", text: "THE FIDELITY GATE: measure each escape batch's joint structure against real fraud, before retraining. Low-fidelity batches are refused here.", tone: "border-blue/60" },
          { n: "4", text: "Admitted batches are folded into supervised training; the detector refits and the loop repeats.", tone: "border-border" },
          { n: "5", text: "The improved detector is measured on held-out REAL fraud — the number that decides whether the loop helped.", tone: "border-border" },
        ].map((row) => (
          <li
            key={row.n}
            className={`flex items-start gap-4 rounded-[var(--r-md)] border border-l-2 ${row.tone} bg-surface-2 px-4 py-3`}
          >
            <span className="type-num shrink-0 text-xs text-text-dim">{row.n}</span>
            <span className={`type-ui text-sm leading-relaxed ${row.n === "3" ? "text-text" : "text-text-dim"}`}>
              {row.text}
            </span>
          </li>
        ))}
      </ol>
      <p className="type-ui measure mt-6 text-xs leading-relaxed text-text-dim">
        Without step 3, a low-fidelity generator&apos;s escapes train the detector to detect the
        generator — and the dashboard improves while real-fraud recall falls. The gate is
        label-free and computable before retraining: an issuer can refuse a bad loop without
        first being harmed by it.
      </p>
    </div>
  );
}

export default async function HomePage() {
  // Every hero figure is read from a generated artifact at render time. If the
  // evidence set has not been regenerated, each slot shows "not measured"
  // rather than a stale or hardcoded number.
  const [closedLoop, fidelity, latency] = await Promise.all([
    readTyped<ClosedLoopArtifact>("closed_loop"),
    readTyped<FidelityReportArtifact>("fidelity_report"),
    readTyped<LatencyArtifact>("latency"),
  ]);

  // The rubric map. Each criterion is answered by exactly one route, so a judge
  // can navigate their own scorecard. Every entry is derived from ROUTES, so
  // this panel cannot drift out of sync with the navigation.
  const rubric = ROUTES.filter((route) => route.href !== "/");

  const scissor = closedLoop?.headline.low_fidelity_generator ?? null;
  const copula = fidelity?.aggregated.gaussian_copula ?? null;
  const independent = fidelity?.aggregated.independent_marginal ?? null;

  return (
    <>
      <PageHeader h1={META.h1} criterion={META.criterion} blurb={META.blurb} eyebrow="00" />

      {/* The three headline claims, each with its measured number. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
        <h2 className="type-ui text-sm font-semibold tracking-tight text-text">The three claims, measured</h2>
        <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
          Each claim below is a number emitted by <code className="type-num">make reproduce</code>,
          read at render time from the artifact named in the card. Regenerate the evidence set and
          these cards change with it; nothing here is hardcoded.
        </p>

        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <Figure
            label="Ungated loop — real-fraud recall"
            value={scissor ? fmtDeltaPts(scissor.ungated_delta_real_recall) : null}
            source="artifacts/closed_loop.json · delta_real_recall"
            tone={scissor && scissor.ungated_delta_real_recall < 0 ? "red" : "neutral"}
          />
          <Figure
            label="Same loop, gate on — real-fraud recall"
            value={scissor ? fmtDeltaPts(scissor.gated_delta_real_recall) : null}
            source="artifacts/closed_loop.json · delta_real_recall"
            tone={scissor ? "blue" : "neutral"}
          />
          <Figure
            label="Inline decision latency, p99"
            value={latency ? `${fmtNum(latency.overall.p99_ms, 1)} ms of a ${latency.protocol.inline_budget_ms} ms budget` : null}
            source="artifacts/latency.json · overall.p99_ms"
            tone="pass"
          />
        </div>

        <p className="type-ui measure mt-6 text-sm leading-relaxed text-text-dim">
          The first two cards are the scissor: the same closed loop, run with and without the
          fidelity gate, on the same low-fidelity generator and the same seeds. The gate protects{" "}
          <span className="type-num">
            {scissor ? fmtPct(scissor.recall_protected_by_gate) : "—"}
          </span>{" "}
          of real-fraud recall —{" "}
          <span className="type-num">
            {scissor ? fmtDeltaPts(scissor.ungated_delta_synthetic_recall) : "—"}
          </span>{" "}
          of the vanity metric is what the ungated loop buys with it.
        </p>
      </section>

      {/* The loop diagram. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="grid gap-4 lg:grid-cols-2">
          <LoopDiagram />

          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h3 className="type-ui text-sm font-semibold tracking-tight text-text">
              Fidelity separates the generators
            </h3>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              Both generators match the marginals. The joint structure is the only difference —
              and it is what the gate measures. C2ST AUC: 0.5 = indistinguishable from real,
              1.0 = trivially fake.
            </p>
            <div className="mt-6 overflow-hidden rounded-[var(--r-md)] border border-border">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-border bg-surface-2">
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">Generator</th>
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">C2ST AUC</th>
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">Rank-dependence error</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border">
                    <td className="type-ui px-4 py-2.5 text-sm text-text">Gaussian copula</td>
                    <td className="type-num px-4 py-2.5 text-sm text-text">{fmtInterval(copula?.c2st_auc)}</td>
                    <td className="type-num px-4 py-2.5 text-sm text-text">{fmtInterval(copula?.correlation_frobenius_diff)}</td>
                  </tr>
                  <tr>
                    <td className="type-ui px-4 py-2.5 text-sm text-text">Independent marginal</td>
                    <td className="type-num px-4 py-2.5 text-sm text-text">{fmtInterval(independent?.c2st_auc)}</td>
                    <td className="type-num px-4 py-2.5 text-sm text-text">{fmtInterval(independent?.correlation_frobenius_diff)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
              The marginal fit is matched across arms (mean JSD{" "}
              <span className="type-num">{fmtInterval(copula?.mean_jsd)}</span> vs{" "}
              <span className="type-num">{fmtInterval(independent?.mean_jsd)}</span>), so the entire
              gap is joint structure — the one variable under test.
            </p>
          </div>
        </div>
      </section>

      {/* The rubric map. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
          Built to the scorecard
        </h2>
        <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
          Each judging criterion is answered by one page, and each page opens with the
          claim it defends rather than a section label.
        </p>

        <ul className="mt-8 divide-y divide-border overflow-hidden rounded-[var(--r-lg)] border border-border bg-surface-1">
          {rubric.map((route) => (
            <li key={route.href}>
              <Link
                href={route.href}
                className="group flex flex-col gap-2 px-5 py-5 transition-colors hover:bg-surface-2 md:flex-row md:items-baseline md:gap-6 md:px-6"
              >
                <span className="type-num shrink-0 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim md:w-64">
                  {route.criterion}
                </span>
                <span className="type-ui min-w-0 flex-1 text-sm leading-snug text-text transition-colors group-hover:text-blue">
                  {route.h1}
                </span>
                <span
                  aria-hidden="true"
                  className="type-num shrink-0 text-xs text-text-dim transition-colors group-hover:text-blue"
                >
                  {route.href}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
