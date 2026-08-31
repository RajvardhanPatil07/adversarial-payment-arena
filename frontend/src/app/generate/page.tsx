import type { Metadata } from "next";

import { PageHeader } from "@/components/shell/page-header";
import { StatusChip } from "@/components/shell/status-chip";
import { ROUTE } from "@/lib/site";
import {
  readTyped,
  fmtInterval,
  fmtNum,
  type FidelityReportArtifact,
  type BehaviouralFidelityArtifact,
} from "@/lib/artifacts";

const META = ROUTE["/generate"];

export const metadata: Metadata = {
  title: "Generate",
  description: META.blurb,
};

/** Display name for a generator arm key in the fidelity report. */
const GENERATOR_LABELS: Record<string, string> = {
  gaussian_copula: "Gaussian copula (joint structure preserved)",
  independent_marginal: "Independent marginal (joint structure destroyed)",
};

/** The five measures, with their ideal value and what a bad value means. */
const MEASURE_NOTES: Record<string, string> = {
  c2st_auc: "0.5 ideal — a classifier separating real from synthetic. High = trivially fake.",
  mean_jsd: "0 ideal — mean Jensen-Shannon divergence across column marginals.",
  mean_tvd: "0 ideal — mean total-variation distance across categorical marginals.",
  correlation_frobenius_diff:
    "0 ideal — Frobenius norm of the correlation-matrix difference. Captures rank dependence, which marginals cannot see.",
  tstr_ratio:
    "1 ideal — train-on-synthetic, test-on-real recall. A generator can be distinguishable and still useful, or vice versa.",
};

export default async function GeneratePage() {
  const [fidelity, behavioural] = await Promise.all([
    readTyped<FidelityReportArtifact>("fidelity_report"),
    readTyped<BehaviouralFidelityArtifact>("behavioural_fidelity"),
  ]);

  const gate = fidelity?.acceptance_gate ?? null;
  const arms = fidelity?.aggregated ?? null;
  const inversion = behavioural?.ordering ?? null;

  return (
    <>
      <PageHeader h1={META.h1} criterion={META.criterion} blurb={META.blurb} eyebrow="02" />

      {/* The gate: thresholds fixed in advance, published whether or not cleared. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
        <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
          The fidelity gate
        </h2>
        <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
          Label-free and computable on an escape batch alone, before retraining: an issuer can
          refuse a red-team generator that would degrade a live detector without first being
          harmed by it. The thresholds were fixed in advance and are never tuned per seed.
        </p>

        {gate ? (
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
              <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                Gate metric
              </p>
              <p className="type-num mt-3 text-sm text-text">{gate.metric}</p>
              <p className="type-num mt-3 text-[0.6875rem] text-text-dim">
                artifacts/fidelity_report.json
              </p>
            </div>
            <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
              <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                Threshold (fixed in advance)
              </p>
              <p className="type-num mt-3 text-xl font-medium tracking-tight text-text">
                {fmtNum(gate.threshold, 2)}
              </p>
              <p className="type-num mt-3 text-[0.6875rem] text-text-dim">
                ≤ this AUC admits the batch
              </p>
            </div>
            <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
              <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                Observed
              </p>
              <div className="mt-3 flex items-center gap-3">
                <p className="type-num text-xl font-medium tracking-tight text-text">
                  {fmtNum(gate.observed, 3)}
                </p>
                <StatusChip tone={gate.cleared ? "pass" : "warn"}>
                  {gate.cleared ? "cleared" : "not cleared"}
                </StatusChip>
              </div>
              <p className="type-num mt-3 text-[0.6875rem] text-text-dim">
                published either way, by policy
              </p>
            </div>
          </div>
        ) : (
          <div className="type-ui mt-8 rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 text-sm text-text-dim">
            The gate is read from <code className="type-num">artifacts/fidelity_report.json</code>,
            which has not been generated yet. Run <code className="type-num">make reproduce</code>.
          </div>
        )}

        {gate && (
          <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
            {gate.policy.charAt(0).toUpperCase() + gate.policy.slice(1)}.
          </p>
        )}
      </section>

      {/* Five measures per generator. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="flex flex-col gap-3 md:flex-row md:items-baseline md:justify-between">
          <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
            Five fidelity measures, per generator
          </h2>
          <p className="type-ui text-xs text-text-dim">
            Mean over three seeds, nonparametric bootstrap 95% intervals.
          </p>
        </div>

        {arms ? (
          <>
            <div className="mt-6 overflow-x-auto rounded-[var(--r-lg)] border border-border bg-surface-1">
              <table className="w-full min-w-[760px] text-left">
                <thead>
                  <tr className="border-b border-border bg-surface-2">
                    <th className="type-ui px-4 py-3 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      Measure
                    </th>
                    {Object.keys(arms).map((arm) => (
                      <th key={arm} className="type-ui px-4 py-3 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        {GENERATOR_LABELS[arm] ?? arm}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {["c2st_auc", "mean_jsd", "mean_tvd", "correlation_frobenius_diff", "tstr_ratio"].map(
                    (measure) => (
                      <tr key={measure} className="border-b border-border last:border-b-0">
                        <td className="px-4 py-3 align-top">
                          <p className="type-num text-sm text-text">{measure}</p>
                          <p className="type-ui mt-1 max-w-[36ch] text-[0.6875rem] leading-relaxed text-text-dim">
                            {MEASURE_NOTES[measure]}
                          </p>
                        </td>
                        {Object.keys(arms).map((arm) => (
                          <td key={arm} className="type-num px-4 py-3 align-top text-sm text-text">
                            {fmtInterval(arms[arm]?.[measure])}
                          </td>
                        ))}
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
            <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
              The marginals (JSD, TVD) are matched across arms by construction; the separation is
              entirely in the joint measures (C2ST, rank dependence). That is the point: a generator
              can pass every marginal test and still be trivially separable from real fraud.
            </p>
          </>
        ) : (
          <div className="type-ui mt-6 rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 text-sm text-text-dim">
            The measures are read from{" "}
            <code className="type-num">artifacts/fidelity_report.json</code>, which has not been
            generated yet.
          </div>
        )}
      </section>

      {/* The ordering inversion — row-level fidelity does not predict transfer. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
          <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
            Row-level fidelity does not predict transfer
          </h2>
          <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
            Ranking the same two generators by row-level fidelity vs by their measured transfer to
            held-out real fraud produces opposite orderings — measured, not asserted.
          </p>

          {inversion ? (
            <>
              <div className="mt-6 overflow-x-auto rounded-[var(--r-md)] border border-border">
                <table className="w-full min-w-[560px] text-left">
                  <thead>
                    <tr className="border-b border-border bg-surface-2">
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Ranking
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Best first
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border">
                      <td className="type-ui px-4 py-2.5 text-sm text-text">By row-level fidelity</td>
                      <td className="type-num px-4 py-2.5 text-sm text-text-dim">
                        {inversion.rank_by_row_level_fidelity.join("  ›  ")}
                      </td>
                    </tr>
                    <tr>
                      <td className="type-ui px-4 py-2.5 text-sm text-text">By transfer to real fraud</td>
                      <td className="type-num px-4 py-2.5 text-sm text-text-dim">
                        {inversion.rank_by_transfer.join("  ›  ")}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              {inversion.ordering_inversion_detected && (
                <div className="mt-4">
                  <StatusChip tone="red">ordering inversion detected</StatusChip>
                </div>
              )}
              <p className="type-num mt-4 text-[0.6875rem] text-text-dim">
                artifacts/behavioural_fidelity.json · ordering
              </p>
            </>
          ) : (
            <p className="type-ui mt-6 text-sm text-text-dim">
              Read from <code className="type-num">artifacts/behavioural_fidelity.json</code>, not
              generated yet.
            </p>
          )}
        </div>
      </section>

      {/* Boundaries. */}
      {fidelity?.boundaries && (
        <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
          <div className="rounded-[var(--r-md)] border border-border bg-surface-2 p-5">
            <h3 className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Boundary conditions
            </h3>
            <ul className="type-ui measure mt-3 list-disc space-y-1.5 pl-5 text-xs leading-relaxed text-text-dim">
              {fidelity.boundaries.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          </div>
        </section>
      )}
    </>
  );
}
