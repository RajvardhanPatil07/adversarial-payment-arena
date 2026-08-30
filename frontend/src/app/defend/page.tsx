import type { Metadata } from "next";

import { PageHeader } from "@/components/shell/page-header";
import { StatusChip } from "@/components/shell/status-chip";
import { ROUTE } from "@/lib/site";
import {
  readTyped,
  fmtInterval,
  fmtNum,
  fmtPct,
  type ClosedLoopArtifact,
  type FamilyCoverageArtifact,
  type PrevalenceArtifact,
  type LatencyArtifact,
  type CalibrationArtifact,
} from "@/lib/artifacts";

const META = ROUTE["/defend"];

export const metadata: Metadata = {
  title: "Defend",
  description: META.blurb,
};

/** Compact recall bar: proportion of the track filled, labelled with its value. */
function RecallBar({ value }: { value: number | null }) {
  if (value === null) {
    return <span className="type-num text-sm text-text-dim">not measured</span>;
  }
  const pctFilled = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="flex items-center gap-3">
      <div className="h-1.5 w-28 overflow-hidden rounded-full bg-surface-3">
        <div className="h-full rounded-full bg-blue" style={{ width: `${pctFilled}%` }} />
      </div>
      <span className="type-num text-sm text-text">{fmtPct(value)}</span>
    </div>
  );
}

export default async function DefendPage() {
  const [closedLoop, coverage, prevalence, latency, calibration] = await Promise.all([
    readTyped<ClosedLoopArtifact>("closed_loop"),
    readTyped<FamilyCoverageArtifact>("family_coverage"),
    readTyped<PrevalenceArtifact>("prevalence_metrics"),
    readTyped<LatencyArtifact>("latency"),
    readTyped<CalibrationArtifact>("calibration_audit"),
  ]);

  // The hardening curve: the ungated low-fidelity arm's per-generation recall
  // on real fraud. This is the loop doing damage, visible generation by
  // generation — the honest counterpart to the "training on escapes" story.
  const ungated = closedLoop?.aggregated?.UNGATED_low_fidelity ?? null;
  const generations = ungated?.by_generation ?? null;
  const genIds = generations ? Object.keys(generations).sort() : [];

  const weakest = coverage?.summary?.weakest_family_when_withheld ?? null;
  const sweep = prevalence?.sweep ?? null;
  const productionRow = sweep?.find((r) => r.prevalence === 0.013) ?? null;

  return (
    <>
      <PageHeader h1={META.h1} criterion={META.criterion} blurb={META.blurb} eyebrow="03" />

      {/* Zero-day generalisation headline. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
        <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
          Leave-one-family-out: detection without memorisation
        </h2>
        <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
          Recall on a family the supervised model trained on is not evidence of generalisation.
          The withheld column is: measured with that family absent from supervised training, so
          whatever catches it is the unsupervised novelty layer and the entity graph.
        </p>

        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
            <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Mean recall · family in training
            </p>
            <p className="type-num mt-3 text-xl font-medium tracking-tight text-text">
              {coverage ? fmtPct(coverage.summary.mean_recall_family_in_training) : <span className="text-text-dim">not measured</span>}
            </p>
            <p className="type-num mt-3 text-[0.6875rem] text-text-dim">artifacts/family_coverage.json</p>
          </div>
          <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
            <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Mean recall · family withheld (zero-day)
            </p>
            <p className="type-num mt-3 text-xl font-medium tracking-tight text-text">
              {coverage ? fmtPct(coverage.summary.mean_recall_family_withheld_zero_day) : <span className="text-text-dim">not measured</span>}
            </p>
            <p className="type-num mt-3 text-[0.6875rem] text-text-dim">artifacts/family_coverage.json</p>
          </div>
          <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
            <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Weakest family when withheld
            </p>
            {weakest ? (
              <>
                <p className="type-ui mt-3 text-sm font-medium text-text">{weakest.label}</p>
                <p className="type-num mt-1 text-sm text-text-dim">{fmtPct(weakest.recall)}</p>
              </>
            ) : (
              <p className="type-num mt-3 text-sm text-text-dim">not measured</p>
            )}
            <p className="type-num mt-3 text-[0.6875rem] text-text-dim">artifacts/family_coverage.json</p>
          </div>
        </div>

        <div className="mt-4">
          {weakest && (
            <StatusChip tone="warn" title="The honest number, not the flattering one.">
              weakest zero-day: {weakest.label} — {fmtPct(weakest.recall)}
            </StatusChip>
          )}
        </div>
      </section>

      {/* The hardening curve, drawn from per-generation recall. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
          <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
            What an ungated loop does to real-fraud recall, generation by generation
          </h2>
          <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
            The ungated low-fidelity arm, retrained on its own escapes each generation. Recall on
            held-out real fraud, with the 95% bootstrap interval shown. This is the harm the gate
            refuses.
          </p>

          {genIds.length > 0 ? (
            <div className="mt-6 overflow-x-auto rounded-[var(--r-md)] border border-border">
              <table className="w-full min-w-[560px] text-left">
                <thead>
                  <tr className="border-b border-border bg-surface-2">
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      Generation
                    </th>
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      Recall on held-out real fraud
                    </th>
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      95% interval
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {genIds.map((g) => {
                    const row = generations?.[g]?.recall_on_real_fraud;
                    return (
                      <tr key={g} className="border-b border-border last:border-b-0">
                        <td className="type-num px-4 py-2.5 text-sm text-text">{g}</td>
                        <td className="px-4 py-2.5">
                          <RecallBar value={row?.mean ?? null} />
                        </td>
                        <td className="type-num px-4 py-2.5 text-sm text-text-dim">
                          {row ? `${fmtNum(row.lo, 3)}–${fmtNum(row.hi, 3)}` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="type-ui mt-6 text-sm text-text-dim">
              Read from <code className="type-num">artifacts/closed_loop.json</code>, not generated
              yet. Run <code className="type-num">make reproduce</code>.
            </p>
          )}

          {ungated && (
            <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
              Net effect over the loop: {fmtInterval(ungated.delta_real_recall)} on real-fraud
              recall. The same loop with the gate on{" "}
              {closedLoop?.headline
                ? `holds ${fmtInterval(closedLoop.aggregated?.GATED_low_fidelity?.delta_real_recall)}`
                : "— the gated arm is not measured"}
              . Every arm had the same attack budget per generation, so no arm wins by volume.
            </p>
          )}
        </div>
      </section>

      {/* Precision at a realistic base rate. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
          <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
            Precision under a production base rate
          </h2>
          <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
            The model is identical in every row; only the assumed base rate changes. Precision
            collapse under realistic prevalence is a property of the base rate and belongs in the
            result, not a footnote.
          </p>

          {sweep ? (
            <>
              <div className="mt-6 overflow-x-auto rounded-[var(--r-md)] border border-border">
                <table className="w-full min-w-[560px] text-left">
                  <thead>
                    <tr className="border-b border-border bg-surface-2">
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Assumed prevalence
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Precision
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Missed frauds (per run)
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {sweep.map((row) => {
                      const isProduction = row.prevalence === 0.013;
                      return (
                        <tr
                          key={row.prevalence}
                          className={`border-b border-border last:border-b-0 ${isProduction ? "bg-blue-dim/30" : ""}`}
                        >
                          <td className="type-num px-4 py-2.5 text-sm text-text">
                            {fmtPct(row.prevalence)}
                            {isProduction && (
                              <span className="type-num ml-2 text-[0.6875rem] text-blue">production</span>
                            )}
                          </td>
                          <td className="type-num px-4 py-2.5 text-sm text-text">
                            {fmtPct(row.precision)}
                          </td>
                          <td className="type-num px-4 py-2.5 text-sm text-text-dim">
                            {fmtNum(row.missed_frauds, 1)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {productionRow && (
                <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
                  At the production prevalence of 1.3%: precision{" "}
                  <span className="type-num">{fmtPct(productionRow.precision)}</span>. The
                  operating point is pinned at 1% FPR on a legitimate validation split, disjoint
                  from every evaluation split.
                </p>
              )}
            </>
          ) : (
            <p className="type-ui mt-6 text-sm text-text-dim">
              Read from <code className="type-num">artifacts/prevalence_metrics.json</code>, not
              generated yet.
            </p>
          )}
        </div>
      </section>

      {/* Inline latency — feasibility of live authorisation. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              Fits inside an inline authorisation budget
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              The full four-layer decision stack — feature assembly, XGBoost, Isolation Forest,
              entity-graph ring check, thresholding — measured per transaction on the exact call
              the WebSocket server makes.
            </p>
            {latency ? (
              <div className="mt-6 grid grid-cols-2 gap-4">
                <div>
                  <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">p50</p>
                  <p className="type-num mt-1 text-lg text-text">{fmtNum(latency.overall.p50_ms, 1)} ms</p>
                </div>
                <div>
                  <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">p95</p>
                  <p className="type-num mt-1 text-lg text-text">{fmtNum(latency.overall.p95_ms, 1)} ms</p>
                </div>
                <div>
                  <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">p99</p>
                  <p className="type-num mt-1 text-lg text-text">{fmtNum(latency.overall.p99_ms, 1)} ms</p>
                </div>
                <div>
                  <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">Budget</p>
                  <p className="type-num mt-1 text-lg text-text-dim">
                    {fmtNum(latency.protocol.inline_budget_ms, 0)} ms
                  </p>
                </div>
              </div>
            ) : (
              <p className="type-ui mt-6 text-sm text-text-dim">
                Read from <code className="type-num">artifacts/latency.json</code>, not generated
                yet.
              </p>
            )}
            <p className="type-num mt-4 text-[0.6875rem] text-text-dim">
              artifacts/latency.json · n={latency ? latency.overall.n : "—"} scored transactions
            </p>
          </div>

          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              Calibration, audited without leakage
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              The threshold is pinned on a validation split that is temporally disjoint from the
              test split, because pinning a threshold on the rows used to report it converts a
              measurement into a fit.
            </p>
            {calibration ? (
              <dl className="mt-6 space-y-4">
                <div>
                  <dt className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                    Recall at 1% FPR, held-out fraud
                  </dt>
                  <dd className="type-num mt-1 text-sm text-text">
                    {fmtInterval(calibration.headline.recall_at_1pct_fpr)}
                  </dd>
                </div>
                <div>
                  <dt className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                    Realised test FPR
                  </dt>
                  <dd className="type-num mt-1 text-sm text-text">
                    {fmtInterval(calibration.headline.realised_fpr)}
                  </dd>
                </div>
                <div>
                  <dt className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                    Calibration gap (pct points)
                  </dt>
                  <dd className="type-num mt-1 text-sm text-text">
                    {fmtInterval(calibration.headline.calibration_gap)}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="type-ui mt-6 text-sm text-text-dim">
                Read from <code className="type-num">artifacts/calibration_audit.json</code>, not
                generated yet.
              </p>
            )}
            <p className="type-num mt-4 text-[0.6875rem] text-text-dim">artifacts/calibration_audit.json</p>
          </div>
        </div>
      </section>

      {/* Boundaries, stated rather than hidden. */}
      {(coverage?.boundaries ?? calibration?.boundaries) && (
        <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
          <div className="rounded-[var(--r-md)] border border-border bg-surface-2 p-5">
            <h3 className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Boundary conditions
            </h3>
            <ul className="type-ui measure mt-3 list-disc space-y-1.5 pl-5 text-xs leading-relaxed text-text-dim">
              {[...(coverage?.boundaries ?? []), ...(calibration?.boundaries ?? [])].map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>
        </section>
      )}
    </>
  );
}
