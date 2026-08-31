import type { Metadata } from "next";
import Link from "next/link";

import { PageHeader } from "@/components/shell/page-header";
import { StatusChip } from "@/components/shell/status-chip";
import { Claim } from "@/components/evidence/claim";
import { LoopDiagram } from "@/components/evidence/loop-diagram";
import { Reveal } from "@/components/evidence/reveal";
import { Scissor } from "@/components/evidence/scissor";
import { ROUTE, ROUTES } from "@/lib/site";
import { loadArtifact } from "@/lib/artifacts";
import { fmtDeltaPts, fmtNum, fmtPct } from "@/lib/format";
import type { ClosedLoopDoc } from "@/lib/validators";

const META = ROUTE["/"];

export const metadata: Metadata = {
  // The default title in layout.tsx is already this page's claim, so the
  // template is bypassed here rather than repeating it.
  title: {
    absolute: "A closed-loop red team without a fidelity gate is an attack surface",
  },
  description: META.blurb,
};

/** Generation ids shared by every arm, e.g. ["V0","V1","V2","V3"]. */
const GENERATIONS = ["V0", "V1", "V2", "V3"] as const;

/**
 * Mean a per-seed measure across seeds, per generation — arithmetic over
 * measured values only (the same mean the artifact's aggregated block
 * records), never an invented or rounded number. Null when the arm is absent.
 */
function meanSeries(
  perSeed: ClosedLoopDoc["per_seed"],
  armName: string,
  key: "recall_on_real_fraud" | "recall_on_synthetic_attacks",
): number[] | null {
  const rows: number[][] = [];
  for (const seed of perSeed) {
    const arm = seed.arms.find((a) => a.arm === armName);
    if (!arm) continue;
    rows.push(arm.generations.map((g) => g[key]));
  }
  if (rows.length === 0) return null;
  const len = Math.min(...rows.map((r) => r.length));
  if (len === 0) return null;
  return Array.from({ length: len }, (_, i) => {
    const vals = rows.map((r) => r[i]);
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  });
}

/** "3 / seed" when every seed rejected the same count; "3·3·3" otherwise. */
function fmtBatchesRejected(rejected: number[] | undefined): string | null {
  if (!rejected || rejected.length === 0) return null;
  const allSame = rejected.every((n) => n === rejected[0]);
  return allSame ? `${rejected[0]} / seed` : rejected.join("·");
}

export default async function HomePage() {
  // Every hero figure is read from a generated artifact at render time. If the
  // evidence set has not been regenerated, each slot shows "not measured"
  // rather than a stale or hardcoded number. Nothing above the fold touches
  // the backend: the page is fully painted from the artifact files.
  const [closedLoopR, fidelityR, latencyR] = await Promise.all([
    loadArtifact("closed_loop"),
    loadArtifact("fidelity_report"),
    loadArtifact("latency"),
  ]);
  const closedLoop = closedLoopR.ok ? closedLoopR.data : null;
  const fidelity = fidelityR.ok ? fidelityR.data : null;
  const latency = latencyR.ok ? latencyR.data : null;

  // The rubric map. Each criterion is answered by exactly one route, so a judge
  // can navigate their own scorecard. Every entry is derived from ROUTES, so
  // this panel cannot drift out of sync with the navigation.
  const rubric = ROUTES.filter((route) => route.href !== "/");

  const lowFi = closedLoop?.headline.low_fidelity_generator ?? null;
  const gatedLowFiArm = closedLoop?.headline.arms.find((a) => a.arm === "GATED_low_fidelity");
  const copula = fidelity?.aggregated.gaussian_copula ?? null;
  const independent = fidelity?.aggregated.independent_marginal ?? null;

  // Loop diagram wiring — all four values measured, read at render time.
  const gateThreshold = closedLoop ? fmtNum(closedLoop.gate.c2st_auc_max, 3) : null;
  const gateObserved = fidelity ? fmtNum(fidelity.acceptance_gate.observed, 3) : null;
  const rejectedC2st = independent ? fmtNum(independent.c2st_auc.mean, 3) : null;
  const batchesRejected = fmtBatchesRejected(gatedLowFiArm?.batches_rejected_by_gate);

  // Scissor wiring — real + synthetic recall per generation, both arms.
  const generations = GENERATIONS as unknown as string[];
  const ungatedReal =
    closedLoop && closedLoop.aggregated["UNGATED_low_fidelity"]
      ? generations.map(
          (g) => closedLoop.aggregated["UNGATED_low_fidelity"].by_generation[g]?.recall_on_real_fraud.mean ?? null,
        )
      : null;
  const gatedReal =
    closedLoop && closedLoop.aggregated["GATED_low_fidelity"]
      ? generations.map(
          (g) => closedLoop.aggregated["GATED_low_fidelity"].by_generation[g]?.recall_on_real_fraud.mean ?? null,
        )
      : null;
  const ungatedSynthetic = closedLoop ? meanSeries(closedLoop.per_seed, "UNGATED_low_fidelity", "recall_on_synthetic_attacks") : null;
  const gatedSynthetic = closedLoop ? meanSeries(closedLoop.per_seed, "GATED_low_fidelity", "recall_on_synthetic_attacks") : null;
  const scissorReady =
    ungatedReal !== null &&
    ungatedSynthetic !== null &&
    gatedReal !== null &&
    gatedSynthetic !== null &&
    [ungatedReal, ungatedSynthetic, gatedReal, gatedSynthetic].every((s) => s.every((v) => v !== null));

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

        <Reveal>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <Claim
              variant="card"
              label="Ungated loop — real-fraud recall"
              value={lowFi ? fmtDeltaPts(lowFi.ungated_delta_real_recall) : null}
              interpretation="The low-fidelity generator's escapes, folded back into training, change recall on held-out real fraud by this much."
              artifactPath="artifacts/closed_loop.json · headline.low_fidelity_generator.ungated_delta_real_recall"
              reproduceCmd="make loop"
              tone={lowFi && lowFi.ungated_delta_real_recall < 0 ? "red" : "neutral"}
            />
            <Claim
              variant="card"
              label="Same loop, gate on — real-fraud recall"
              value={lowFi ? fmtDeltaPts(lowFi.gated_delta_real_recall) : null}
              interpretation="The identical loop with the fidelity gate enabled: the low-fidelity batches are refused and real-fraud recall is protected."
              artifactPath="artifacts/closed_loop.json · headline.low_fidelity_generator.gated_delta_real_recall"
              reproduceCmd="make loop"
              tone="blue"
            />
            <Claim
              variant="card"
              label="Inline decision latency, p99"
              value={latency ? `${fmtNum(latency.overall.p99_ms, 1)} ms of a ${latency.protocol.inline_budget_ms} ms budget` : null}
              interpretation="The full four-layer decision stack, scored inline, stays inside the authorisation window at the 99th percentile."
              artifactPath="artifacts/latency.json · overall.p99_ms"
              reproduceCmd="make latency"
              tone="pass"
            />
          </div>
        </Reveal>

        {/* SECTION 8.3 — WHY THIS IS HARD, beside the metrics. */}
        <div className="mt-4 rounded-[var(--r-md)] border border-border bg-surface-2 px-4 py-3">
          <p className="type-ui text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-text-dim">
            Why this is hard
          </p>
          <p className="type-ui measure mt-2 text-xs leading-relaxed text-text-dim">
            The ungated loop&apos;s harm is invisible on every number the loop itself controls —
            synthetic-attack recall goes <span className="type-num">up</span> while real-fraud recall
            goes <span className="type-num">down</span>. It only shows on a yardstick the loop
            cannot touch: held-out real fraud. That is why each card names its artifact path, and
            why the gate is label-free and fixed in advance — it must fire before the loop can
            grade its own homework.
          </p>
        </div>

        <p className="type-ui measure mt-6 text-sm leading-relaxed text-text-dim">
          The first two cards are the scissor: the same closed loop, run with and without the
          fidelity gate, on the same low-fidelity generator and the same seeds. The gate protects{" "}
          <span className="type-num">
            {lowFi ? fmtPct(lowFi.recall_protected_by_gate) : "—"}
          </span>{" "}
          of real-fraud recall —{" "}
          <span className="type-num">
            {lowFi ? fmtDeltaPts(lowFi.ungated_delta_synthetic_recall) : "—"}
          </span>{" "}
          of the vanity metric is what the ungated loop buys with it.
        </p>
      </section>

      {/* The loop diagram — SVG, wired to the measured gate values. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h3 className="type-ui text-sm font-semibold tracking-tight text-text">
              The closed loop, and where the gate cuts it
            </h3>
            <p className="type-ui measure mt-2 text-xs leading-relaxed text-text-dim">
              Every number in the diagram is read from an artifact at render time: the gate&apos;s
              threshold, the observed fidelity of the admitted generator, and the fidelity of the
              generator whose batches were rejected.
            </p>
            <div className="mt-6">
              <LoopDiagram
                gateThreshold={gateThreshold}
                gateObserved={gateObserved}
                rejectedC2st={rejectedC2st}
                batchesRejected={batchesRejected}
                provenanceNote="closed_loop.json + fidelity_report.json · seeds 11 · 23 · 37"
              />
            </div>
            <p className="type-ui measure mt-6 text-xs leading-relaxed text-text-dim">
              Without the gate, a low-fidelity generator&apos;s escapes train the detector to detect
              the generator — and the dashboard improves while real-fraud recall falls. The gate is
              label-free and computable before retraining: an issuer can refuse a bad loop without
              first being harmed by it.
            </p>
          </div>
        </Reveal>
      </section>

      {/* The scissor — the vanity metric against the real one. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
          <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
            <h3 className="type-ui text-sm font-semibold tracking-tight text-text">
              The scissor: what the ungated loop does to the two recalls
            </h3>
            <p className="type-ui measure mt-2 text-xs leading-relaxed text-text-dim">
              Same loop, same generator, same seeds — run ungated and gated. Each line is the
              seed-mean recall per generation, read from closed_loop.json.
            </p>
            <div className="mt-6">
              {scissorReady ? (
                <Scissor
                  generations={generations}
                  ungatedReal={ungatedReal as number[]}
                  ungatedSynthetic={ungatedSynthetic as number[]}
                  gatedReal={gatedReal as number[]}
                  gatedSynthetic={gatedSynthetic as number[]}
                  artifactPath="artifacts/closed_loop.json · per_seed[].arms[].generations[]"
                />
              ) : (
                <p className="type-num text-sm text-text-dim">
                  — artifact unavailable: closed_loop.json · per_seed[].arms[].generations[]
                </p>
              )}
            </div>
          </div>
        </Reveal>
      </section>

      {/* Fidelity separates the generators. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <Reveal>
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
                    <td className="type-num px-4 py-2.5 text-sm text-text">
                      {copula ? `${fmtNum(copula.c2st_auc.mean, 3)} (${fmtNum(copula.c2st_auc.lo, 3)}–${fmtNum(copula.c2st_auc.hi, 3)})` : "— not measured"}
                    </td>
                    <td className="type-num px-4 py-2.5 text-sm text-text">
                      {copula ? `${fmtNum(copula.correlation_frobenius_diff.mean, 3)} (${fmtNum(copula.correlation_frobenius_diff.lo, 3)}–${fmtNum(copula.correlation_frobenius_diff.hi, 3)})` : "— not measured"}
                    </td>
                  </tr>
                  <tr>
                    <td className="type-ui px-4 py-2.5 text-sm text-text">Independent marginal</td>
                    <td className="type-num px-4 py-2.5 text-sm text-text">
                      {independent ? `${fmtNum(independent.c2st_auc.mean, 3)} (${fmtNum(independent.c2st_auc.lo, 3)}–${fmtNum(independent.c2st_auc.hi, 3)})` : "— not measured"}
                    </td>
                    <td className="type-num px-4 py-2.5 text-sm text-text">
                      {independent ? `${fmtNum(independent.correlation_frobenius_diff.mean, 3)} (${fmtNum(independent.correlation_frobenius_diff.lo, 3)}–${fmtNum(independent.correlation_frobenius_diff.hi, 3)})` : "— not measured"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
              The marginal fit is matched across arms (mean JSD{" "}
              <span className="type-num">{copula ? fmtNum(copula.mean_jsd.mean, 3) : "—"}</span> vs{" "}
              <span className="type-num">{independent ? fmtNum(independent.mean_jsd.mean, 3) : "—"}</span>),
              so the entire gap is joint structure — the one variable under test.
            </p>
            <div className="mt-4">
              <StatusChip tone={fidelity?.acceptance_gate.cleared ? "pass" : "warn"}>
                {fidelity?.acceptance_gate.cleared
                  ? "acceptance gate cleared"
                  : "acceptance gate not cleared — published either way, by policy"}
              </StatusChip>
            </div>
          </div>
        </Reveal>
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
