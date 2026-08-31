import type { Metadata } from "next";

import { ArtifactChip } from "@/components/evidence/artifact-chip";
import { Boundary } from "@/components/evidence/boundary";
import { Claim } from "@/components/evidence/claim";
import { LatencyBudget } from "@/components/evidence/latency-budget";
import { Reveal } from "@/components/evidence/reveal";
import { Scissor } from "@/components/evidence/scissor";
import { PageHeader } from "@/components/shell/page-header";
import { loadArtifact } from "@/lib/artifacts";
import { fmtInterval, fmtInr, fmtNum, fmtPct } from "@/lib/format";
import { ROUTE } from "@/lib/site";

const META = ROUTE["/evidence"];

export const metadata: Metadata = {
  title: "Evidence",
  description: META.blurb,
};

const CL = "artifacts/closed_loop.json";

/**
 * SECTION 1D: the explicit missing state. A failed validation renders this,
 * naming the artifact and the exact missing field paths — never a fallback
 * number, never a silent empty section.
 */
function ArtifactUnavailable({ name, missing }: { name: string; missing: string[] }) {
  return (
    <div className="rounded-[var(--r-md)] border border-warn/40 bg-surface-1 p-4">
      <p className="type-ui text-sm font-semibold text-warn">artifact unavailable</p>
      <p className="type-ui measure mt-2 text-xs leading-relaxed text-text-dim">
        <span className="type-num">{name}</span> failed validation. Missing:{" "}
        <span className="type-num">{missing.join(", ")}</span>. Run{" "}
        <code className="type-num">make reproduce</code> then{" "}
        <code className="type-num">npm run snapshot:artifacts</code> and reload. This page
        deliberately shows nothing rather than placeholder numbers.
      </p>
    </div>
  );
}

export default async function EvidencePage() {
  const [ledgerR, calibrationR, economicsR, latencyR, closedLoopR] = await Promise.all([
    loadArtifact("claim_ledger"),
    loadArtifact("calibration_audit"),
    loadArtifact("economics"),
    loadArtifact("latency"),
    loadArtifact("closed_loop"),
  ]);

  const ledger = ledgerR.ok ? ledgerR.data : null;
  const calibration = calibrationR.ok ? calibrationR.data : null;
  const economics = economicsR.ok ? economicsR.data : null;
  const latency = latencyR.ok ? latencyR.data : null;
  const closedLoop = closedLoopR.ok ? closedLoopR.data : null;

  // -- Scissor wiring: mean real + synthetic recall per generation, both arms. --
  const generations = closedLoop
    ? Object.keys(closedLoop.aggregated["UNGATED_low_fidelity"]?.by_generation ?? {}).sort()
    : [];

  const realSeries = (armName: string): number[] | null => {
    const arm = closedLoop?.aggregated[armName];
    if (!arm) return null;
    const series = generations.map((g) => arm.by_generation[g]?.recall_on_real_fraud.mean ?? null);
    return series.every((v) => v !== null) ? (series as number[]) : null;
  };

  /** Seed-mean recall on the loop's own synthetic attacks, per generation. */
  const syntheticSeries = (armName: string): number[] | null => {
    if (!closedLoop) return null;
    const series = generations.map((_, gi) => {
      const values = closedLoop.per_seed.map(
        (s) =>
          s.arms.find((a) => a.arm === armName)?.generations[gi]
            ?.recall_on_synthetic_attacks ?? null,
      );
      return values.every((v) => v !== null)
        ? (values as number[]).reduce((a, b) => a + b, 0) / values.length
        : null;
    });
    return series.every((v) => v !== null) ? (series as number[]) : null;
  };

  const ungatedReal = realSeries("UNGATED_low_fidelity");
  const ungatedSynthetic = syntheticSeries("UNGATED_low_fidelity");
  const gatedReal = realSeries("GATED_low_fidelity");
  const gatedSynthetic = syntheticSeries("GATED_low_fidelity");

  // -- Calibration sweep rows, ordered by target FPR. --
  const calibrationRows = calibration
    ? Object.entries(calibration.aggregated)
        .map(([key, row]) => ({ key, row }))
        .sort((a, b) => a.row.target_fpr - b.row.target_fpr)
    : [];

  const headlineBudget = calibration?.protocol.headline_fpr ?? null;
  const econCounts = economics?.at_production_prevalence.counts ?? null;
  const econDerived = economics?.cost_model.derived ?? null;
  const econOp = economics?.at_production_prevalence.operating_point ?? null;
  const lat = latency?.overall ?? null;

  return (
    <>
      <PageHeader h1={META.h1} criterion={META.criterion} blurb={META.blurb} eyebrow="05" />

      {/* The claim ledger: the spine of this page. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
        <h2 className="type-ui text-sm font-semibold tracking-tight text-text">The claim ledger</h2>
        <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
          Every public claim, the artifact field that supports it, how the number was derived, and
          the boundary beyond which it does not hold. The boundary column is mandatory — a claim
          without a stated limit is marketing, not evidence.
        </p>

        {ledger ? (
          <Reveal>
            <div className="mt-6 overflow-x-auto rounded-[var(--r-md)] border border-border">
              <table className="w-full min-w-[900px] text-left">
                <thead>
                  <tr className="border-b border-border bg-surface-2">
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      Claim
                    </th>
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      Artifact · field
                    </th>
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      Derivation
                    </th>
                    <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                      Boundary
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {ledger.claims.map((c) => (
                    <tr key={c.claim} className="border-b border-border align-top last:border-b-0">
                      <td className="type-ui measure px-4 py-3 text-xs leading-relaxed text-text">
                        {c.claim}
                      </td>
                      <td className="px-4 py-3">
                        <ArtifactChip path={c.artifact} note={c.field} />
                        <p className="type-num mt-1.5 text-[0.6875rem] leading-relaxed text-text-dim">
                          {c.field}
                        </p>
                      </td>
                      <td className="type-ui measure px-4 py-3 text-xs leading-relaxed text-text-dim">
                        {c.derivation}
                      </td>
                      <td className="type-ui measure px-4 py-3 text-xs leading-relaxed text-warn">
                        {c.boundary}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="type-ui mt-3 text-xs text-text-faint">
              {ledger.claims.length} claims · each row links to the raw JSON it is read from ·
              regenerate the whole set with <code className="type-num">make reproduce</code>
            </p>
          </Reveal>
        ) : (
          <div className="mt-6">
            <ArtifactUnavailable
              name="claim_ledger.json"
              missing={ledgerR.ok ? [] : ledgerR.missing}
            />
          </div>
        )}
      </section>

      {/* Calibration audit: the anti-leakage sweep. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              Calibration, audited without leakage
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              The threshold is pinned on a legitimate validation split temporally disjoint from the
              test split, because pinning a threshold on the rows used to report it converts a
              measurement into a fit. The gap between target and realised FPR is then audited at
              every budget, with bootstrap intervals.
            </p>

            {calibration ? (
              <div className="mt-6 overflow-x-auto rounded-[var(--r-md)] border border-border">
                <table className="w-full min-w-[820px] text-left">
                  <thead>
                    <tr className="border-b border-border bg-surface-2">
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Target FPR
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Realised test FPR
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Recall · held-out fraud
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Calibration gap
                      </th>
                      <th className="type-ui px-4 py-2.5 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                        Precision @ production prevalence
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {calibrationRows.map(({ key, row }) => {
                      const isHeadline =
                        headlineBudget !== null && row.target_fpr === headlineBudget;
                      return (
                        <tr
                          key={key}
                          className={`border-b border-border last:border-b-0 ${isHeadline ? "bg-blue-dim/30" : ""}`}
                        >
                          <td className="type-num px-4 py-2.5 text-sm text-text">
                            {fmtPct(row.target_fpr)}
                            {isHeadline && (
                              <span className="type-num ml-2 text-[0.6875rem] text-blue">headline</span>
                            )}
                          </td>
                          <td className="type-num px-4 py-2.5 text-sm text-text-dim">
                            {fmtInterval(row.realised_test_fpr)}
                          </td>
                          <td className="type-num px-4 py-2.5 text-sm text-text">
                            {fmtInterval(row.recall_on_held_out_fraud)}
                          </td>
                          <td className="type-num px-4 py-2.5 text-sm text-text-dim">
                            {fmtInterval(row.calibration_gap_pct_points)}
                          </td>
                          <td className="type-num px-4 py-2.5 text-sm text-text">
                            {fmtInterval(row.precision_at_production_prevalence)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="mt-6">
                <ArtifactUnavailable
              name="calibration_audit.json"
              missing={calibrationR.ok ? [] : calibrationR.missing}
            />
              </div>
            )}

            {calibration && (
              <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
                All intervals are nonparametric bootstrap 95% CIs over the recorded seeds. The
                headline operating point pins{" "}
                <span className="type-num">{fmtPct(calibration.protocol.headline_fpr)}</span> FPR
                and reports precision at the production prevalence assumption of{" "}
                <span className="type-num">
                  {fmtPct(calibration.protocol.production_prevalence)}
                </span>
                .
              </p>
            )}
          </div>
        </Reveal>
      </section>

      {/* The scissor, on the evidence record. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              The scissor, on the record
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              The vanity metric and the real metric move in opposite directions in the ungated
              low-fidelity arm: recall on the generator&apos;s own attacks rises while recall on
              held-out real fraud falls. The gate removes it by refusing the escape batches that
              cause it — using only a label-free fidelity measurement computable before retraining.
            </p>
            {closedLoop && ungatedReal && ungatedSynthetic && gatedReal && gatedSynthetic ? (
              <div className="mt-6">
                <Scissor
                  generations={generations}
                  ungatedReal={ungatedReal}
                  ungatedSynthetic={ungatedSynthetic}
                  gatedReal={gatedReal}
                  gatedSynthetic={gatedSynthetic}
                  artifactPath={`${CL} · aggregated.*.by_generation + per_seed`}
                />
              </div>
            ) : (
              <div className="mt-6">
                <ArtifactUnavailable
                  name="closed_loop.json"
                  missing={closedLoopR.ok ? ["aggregated.*.by_generation"] : closedLoopR.missing}
                />
              </div>
            )}
          </div>
        </Reveal>
      </section>

      {/* Economics + the false-positive cost panel. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              The cost ledger, priced asymmetrically
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              Most fraud demos price only the fraud they stopped. This ledger prices all four
              cells of the confusion matrix at the production prevalence assumption, including the
              insult cost of wrongly declining a legitimate customer — support contact, lost
              interchange margin, and churn probability priced against lifetime value.
            </p>

            {economics && econCounts && econDerived && econOp ? (
              <>
                {/* The asymmetric cost matrix (SECTION 8.4). */}
                <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="rounded-[var(--r-md)] border border-blue/40 bg-blue-dim/20 p-4">
                    <p className="type-ui text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-blue">
                      True positives · declined fraud
                    </p>
                    <p className="type-num mt-2 text-2xl text-text">{fmtNum(econCounts.tp, 0)}</p>
                    <p className="type-ui mt-2 text-xs leading-relaxed text-text-dim">
                      Value <span className="type-num">{fmtInr(econDerived.value_per_true_positive_inr)}</span>{" "}
                      each — the fraud amount, since recovery on declined fraud is assumed total.
                    </p>
                  </div>
                  <div className="rounded-[var(--r-md)] border border-red/40 bg-red-dim/40 p-4">
                    <p className="type-ui text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-red">
                      False positives · insulted good customers
                    </p>
                    <p className="type-num mt-2 text-2xl text-text">{fmtNum(econCounts.fp, 0)}</p>
                    <p className="type-ui mt-2 text-xs leading-relaxed text-text-dim">
                      Cost <span className="type-num">{fmtInr(econDerived.insult_cost_per_false_positive_inr)}</span>{" "}
                      each — support contact plus lost margin plus churn risk priced at CLV.
                    </p>
                  </div>
                  <div className="rounded-[var(--r-md)] border border-red/40 bg-red-dim/40 p-4">
                    <p className="type-ui text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-red">
                      False negatives · fraud that slipped through
                    </p>
                    <p className="type-num mt-2 text-2xl text-text">{fmtNum(econCounts.fn, 0)}</p>
                    <p className="type-ui mt-2 text-xs leading-relaxed text-text-dim">
                      Loss <span className="type-num">{fmtInr(econDerived.loss_per_false_negative_inr)}</span>{" "}
                      each — the full fraud amount plus chargeback admin, unrecovered.
                    </p>
                  </div>
                  <div className="rounded-[var(--r-md)] border border-border bg-surface-2 p-4">
                    <p className="type-ui text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-text-dim">
                      True negatives · untouched good traffic
                    </p>
                    <p className="type-num mt-2 text-2xl text-text">{fmtNum(econCounts.tn, 0)}</p>
                    <p className="type-ui mt-2 text-xs leading-relaxed text-text-dim">
                      The silent majority; the whole point is to leave them alone.
                    </p>
                  </div>
                </div>

                {/* Headline economics claims, all from economics.json. */}
                <div className="mt-4 grid gap-4 md:grid-cols-3">
                  <Claim
                    variant="card"
                    label="Net benefit per million authorisations"
                    value={fmtInr(economics.at_production_prevalence.net_benefit_inr)}
                    interpretation="Fraud prevented, less fraud lost, less the insult cost of wrongly declining legitimate customers, less review cost — all four cells, priced."
                    artifactPath="artifacts/economics.json · at_production_prevalence.net_benefit_inr"
                    reproduceCmd="make reproduce"
                    tone="blue"
                  />
                  <Claim
                    variant="card"
                    label="False positives as a share of total cost"
                    value={fmtPct(economics.at_production_prevalence.insult_share_of_total_cost)}
                    interpretation="Wrongly declined legitimate payments are the single largest cost term — larger than the fraud losses that slip through. The asymmetric cost matrix is what keeps that term bounded."
                    artifactPath="artifacts/economics.json · at_production_prevalence.insult_share_of_total_cost"
                    reproduceCmd="make reproduce"
                    tone="neutral"
                  />
                  <Claim
                    variant="card"
                    label="Operating point this is priced at"
                    value={
                      econOp
                        ? `${fmtPct(econOp.recall)} recall · ${fmtPct(econOp.fpr)} FPR`
                        : null
                    }
                    interpretation={`Assumed prevalence ${fmtPct(econOp?.prevalence ?? null)} on a volume of ${fmtNum(econOp?.volume ?? null, 0)} authorisations per run.`}
                    artifactPath="artifacts/economics.json · at_production_prevalence.operating_point"
                    reproduceCmd="make reproduce"
                    tone="neutral"
                  />
                </div>
              </>
            ) : (
              <div className="mt-6">
                <ArtifactUnavailable
              name="economics.json"
              missing={economicsR.ok ? [] : economicsR.missing}
            />
              </div>
            )}
          </div>
        </Reveal>
      </section>

      {/* Latency percentiles vs budget. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
              Latency percentiles vs the inline budget
            </h2>
            <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
              The full four-layer decision stack — feature assembly with sliding-window velocity
              counters, XGBoost, Isolation Forest, the entity-graph ring check, thresholding —
              measured per transaction on the exact call the WebSocket server makes.
            </p>
            {lat && latency ? (
              <div className="mt-6">
                <LatencyBudget
                  p50={lat.p50_ms}
                  p95={lat.p95_ms}
                  p99={lat.p99_ms}
                  budget={latency.protocol.inline_budget_ms}
                  n={lat.n}
                  artifactPath="artifacts/latency.json · overall"
                />
              </div>
            ) : (
              <div className="mt-6">
                <ArtifactUnavailable
              name="latency.json"
              missing={latencyR.ok ? [] : latencyR.missing}
            />
              </div>
            )}
          </div>
        </Reveal>
      </section>

      {/* Every boundary, stated rather than hidden. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        {calibration ? (
          <Boundary items={calibration.boundaries} />
        ) : (
          <Boundary>
            Boundary statements for the calibration audit are read from{" "}
            <code className="type-num">artifacts/calibration_audit.json</code>, which is currently
            unavailable.
          </Boundary>
        )}
        <div className="mt-4">
          <Boundary title="Boundary of the whole result">
            &ldquo;Real fraud&rdquo; here means held-out fraud from the arena&apos;s
            topology-aware environment, not issuer production data. The claim is about the
            relationship between generator fidelity and transfer, not an absolute recall figure
            for live card traffic. Every per-claim boundary is stated in the ledger above; the
            per-cell INR rates in the cost panel are order-of-magnitude assumptions and are
            overridable.
          </Boundary>
        </div>
      </section>
    </>
  );
}
