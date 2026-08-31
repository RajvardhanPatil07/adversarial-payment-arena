import type { Metadata } from "next";

import { PageHeader } from "@/components/shell/page-header";
import { StatusChip } from "@/components/shell/status-chip";
import { Boundary } from "@/components/evidence/boundary";
import { Claim } from "@/components/evidence/claim";
import { HardeningCurve, type SeedSeries } from "@/components/evidence/hardening-curve";
import { LatencyBudget } from "@/components/evidence/latency-budget";
import { Reveal } from "@/components/evidence/reveal";
import { loadArtifact } from "@/lib/artifacts";
import { fmtInterval, fmtNum, fmtPct } from "@/lib/format";
import { ROUTE } from "@/lib/site";

const META = ROUTE["/defend"];

export const metadata: Metadata = {
  title: "Defend",
  description: META.blurb,
};

const CL = "artifacts/closed_loop.json";
const FC = "artifacts/family_coverage.json";

export default async function DefendPage() {
  const [closedLoopR, coverageR, prevalenceR, latencyR, calibrationR, metricsR] = await Promise.all([
    loadArtifact("closed_loop"),
    loadArtifact("family_coverage"),
    loadArtifact("prevalence_metrics"),
    loadArtifact("latency"),
    loadArtifact("calibration_audit"),
    loadArtifact("metrics"),
  ]);
  const closedLoop = closedLoopR.ok ? closedLoopR.data : null;
  const coverage = coverageR.ok ? coverageR.data : null;
  const prevalence = prevalenceR.ok ? prevalenceR.data : null;
  const latency = latencyR.ok ? latencyR.data : null;
  const calibration = calibrationR.ok ? calibrationR.data : null;
  const metrics = metricsR.ok ? metricsR.data : null;

  // The hardening curve: both low-fidelity arms, per seed and mean.
  const generations = closedLoop
    ? Object.keys(closedLoop.aggregated["UNGATED_low_fidelity"]?.by_generation ?? {}).sort()
    : [];

  const seedSeries = (armName: string): SeedSeries[] =>
    closedLoop
      ? closedLoop.per_seed.map((seed) => ({
          seed: seed.seed,
          values: seed.arms
            .find((a) => a.arm === armName)
            ?.generations.map((g) => g.recall_on_real_fraud) ?? null,
        }))
      : [];

  const meanSeries = (armName: string): number[] | null => {
    const arm = closedLoop?.aggregated[armName];
    if (!arm) return null;
    const series = generations.map((g) => arm.by_generation[g]?.recall_on_real_fraud.mean ?? null);
    return series.every((v) => v !== null) ? (series as number[]) : null;
  };

  const ungatedMean = meanSeries("UNGATED_low_fidelity");
  const gatedMean = meanSeries("GATED_low_fidelity");
  const curveReady = closedLoop !== null && ungatedMean !== null && gatedMean !== null;

  const weakest = coverage?.summary?.weakest_family_when_withheld ?? null;
  const sweep = prevalence?.sweep ?? null;
  const opFpr = prevalence?.operating_point.fpr ?? null;

  const lat = latency?.overall ?? null;

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

        <Reveal>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <Claim
              variant="card"
              label="Mean recall · family in training"
              value={coverage ? fmtPct(coverage.summary.mean_recall_family_in_training) : null}
              interpretation="Per-family non-APPROVE rate with the family present in supervised training, averaged across the fourteen families."
              artifactPath={`${FC} · summary.mean_recall_family_in_training`}
              reproduceCmd="make coverage"
              tone="blue"
            />
            <Claim
              variant="card"
              label="Mean recall · family withheld (zero-day)"
              value={coverage ? fmtPct(coverage.summary.mean_recall_family_withheld_zero_day) : null}
              interpretation="The same families absent from supervised training: the honest column, measured with architecture only."
              artifactPath={`${FC} · summary.mean_recall_family_withheld_zero_day`}
              reproduceCmd="make coverage"
              tone="blue"
            />
            <Claim
              variant="card"
              label="Weakest family when withheld"
              value={weakest ? fmtPct(weakest.recall) : null}
              interpretation={weakest ? `${weakest.label} — it defeats ${weakest.defeats}.` : "The weakest zero-day generaliser."}
              artifactPath={`${FC} · summary.weakest_family_when_withheld`}
              reproduceCmd="make coverage"
              tone="neutral"
            />
          </div>
        </Reveal>

        <div className="mt-4">
          {weakest && (
            <StatusChip tone="warn" title="The honest number, not the flattering one.">
              weakest zero-day: {weakest.label} — {fmtPct(weakest.recall)}
            </StatusChip>
          )}
        </div>
      </section>

      {/* The hardening curve: per-seed paths, mean paths, 1% FPR reference. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              What the loop does to real-fraud recall, generation by generation
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              The same low-fidelity generator, ungated and gated. Thin lines are individual seeds;
              bold lines are the seed means. Ungated, the per-seed paths scatter and fall as
              low-fidelity escapes pollute retraining; gated, they stay together.
            </p>

            {curveReady ? (
              <div className="mt-6">
                <HardeningCurve
                  generations={generations}
                  ungatedSeeds={seedSeries("UNGATED_low_fidelity")}
                  gatedSeeds={seedSeries("GATED_low_fidelity")}
                  ungatedMean={ungatedMean}
                  gatedMean={gatedMean}
                  fprReference={calibration?.protocol.headline_fpr ?? null}
                  artifactPath={`${CL} · per_seed[].arms[].generations[] + aggregated`}
                />
              </div>
            ) : (
              <p className="type-ui mt-6 text-sm text-text-dim">
                Read from <code className="type-num">artifacts/closed_loop.json</code>, which is
                unavailable. Run <code className="type-num">make reproduce</code> to build it.
              </p>
            )}

            {closedLoop && (
              <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
                Net effect over the loop: ungated{" "}
                <span className="type-num">{fmtInterval(closedLoop.aggregated["UNGATED_low_fidelity"]?.delta_real_recall ?? null)}</span>{" "}
                · gated{" "}
                <span className="type-num">{fmtInterval(closedLoop.aggregated["GATED_low_fidelity"]?.delta_real_recall ?? null)}</span>{" "}
                on real-fraud recall. Every arm had the same attack budget per generation, so no
                arm wins by volume.
              </p>
            )}
          </div>
        </Reveal>
      </section>

      {/* Precision at a realistic base rate + the recommended operating point. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              Precision under a production base rate
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              {prevalence?.note ??
                "The model is identical in every row; only the assumed base rate changes."}{" "}
              The false-positive rate is the operating point&apos;s, pinned on a legitimate
              validation split disjoint from every evaluation split.
            </p>

            {sweep ? (
              <div className="mt-6 overflow-x-auto rounded-[var(--r-md)] border border-border">
                <table className="w-full min-w-[640px] text-left">
                  <thead>
                    <tr className="border-b border-border bg-surface-2">
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Assumed prevalence
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        FPR (operating point)
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
                          <td className="type-num px-4 py-2.5 text-sm text-text-dim">
                            {opFpr !== null ? fmtPct(opFpr) : "—"}
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
            ) : (
              <p className="type-ui mt-6 text-sm text-text-dim">
                Read from <code className="type-num">artifacts/prevalence_metrics.json</code>, not
                generated yet.
              </p>
            )}

            {/* The recommended operating point — every term measured. */}
            <div className="mt-6 rounded-[var(--r-md)] border border-blue/40 bg-blue-dim/20 p-4">
              <p className="type-ui text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-blue">
                Recommended operating point
              </p>
              {calibration && metrics ? (
                <div className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  <div>
                    <p className="type-ui text-xs text-text-dim">target FPR</p>
                    <p className="type-num mt-1 text-sm text-text">
                      {fmtPct(calibration.protocol.headline_fpr)}
                    </p>
                  </div>
                  <div>
                    <p className="type-ui text-xs text-text-dim">realised test FPR</p>
                    <p className="type-num mt-1 text-sm text-text">
                      {fmtInterval(calibration.headline.realised_fpr)}
                    </p>
                  </div>
                  <div>
                    <p className="type-ui text-xs text-text-dim">recall on held-out fraud</p>
                    <p className="type-num mt-1 text-sm text-text">
                      {fmtInterval(calibration.headline.recall_at_1pct_fpr)}
                    </p>
                  </div>
                  <div>
                    <p className="type-ui text-xs text-text-dim">
                      precision at production prevalence
                    </p>
                    <p className="type-num mt-1 text-sm text-text">
                      {fmtInterval(metrics.precision_at_production_prevalence)}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="type-num mt-2 text-sm text-text-dim">— not measured</p>
              )}
              <p className="type-ui measure mt-3 text-xs leading-relaxed text-text-dim">
                Pinned on a validation split disjoint from training and evaluation, re-pinned for
                every seed and every withheld family — the calibration gap is audited, not
                assumed:{" "}
                <span className="type-num">
                  {calibration ? fmtInterval(calibration.headline.calibration_gap) : "—"}
                </span>{" "}
                percentage points.
              </p>
            </div>
          </div>
        </Reveal>
      </section>

      {/* Inline latency + calibration audit. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="grid gap-4 lg:grid-cols-2">
          <Reveal className="h-full">
            <div className="h-full rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
              <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
                Fits inside an inline authorisation budget
              </h2>
              <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
                The full four-layer decision stack — feature assembly, XGBoost, Isolation Forest,
                entity-graph ring check, thresholding — measured per transaction on the exact call
                the WebSocket server makes.
              </p>
              {lat ? (
                <div className="mt-6">
                  <LatencyBudget
                    p50={lat.p50_ms}
                    p95={lat.p95_ms}
                    p99={lat.p99_ms}
                    budget={latency?.protocol.inline_budget_ms ?? null}
                    n={lat.n}
                    artifactPath="artifacts/latency.json · overall"
                  />
                </div>
              ) : (
                <p className="type-ui mt-6 text-sm text-text-dim">
                  Read from <code className="type-num">artifacts/latency.json</code>, not generated
                  yet.
                </p>
              )}
            </div>
          </Reveal>

          <Reveal className="h-full">
            <div className="h-full rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
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
          </Reveal>
        </div>
      </section>

      {/* Boundaries, stated rather than hidden. */}
      {(coverage?.boundaries || calibration?.boundaries || closedLoop?.boundaries) && (
        <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
          <Boundary
            items={[
              ...(coverage?.boundaries ?? []),
              ...(calibration?.boundaries ?? []),
              ...(closedLoop?.boundaries ?? []),
            ]}
          />
        </section>
      )}
    </>
  );
}
