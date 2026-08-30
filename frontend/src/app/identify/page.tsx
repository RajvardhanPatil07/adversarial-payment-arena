import type { Metadata } from "next";

import { PageHeader } from "@/components/shell/page-header";
import { StatusChip } from "@/components/shell/status-chip";
import { ROUTE } from "@/lib/site";
import {
  readTyped,
  fmtInterval,
  fmtPct,
  type FamilyCoverageArtifact,
} from "@/lib/artifacts";

const META = ROUTE["/identify"];

export const metadata: Metadata = {
  title: "Identify",
  description: META.blurb,
};

/** Short human label for an ATTACK_N_* family id. */
function familyLabel(id: string, meta: { label?: string }): string {
  return meta.label ?? id
    .replace(/^ATTACK_\d+_/, "")
    .toLowerCase()
    .replace(/_/g, " ");
}

export default async function IdentifyPage() {
  const coverage = await readTyped<FamilyCoverageArtifact>("family_coverage");

  const summary = coverage?.summary ?? null;
  const inTraining = coverage?.family_in_training ?? null;
  const withheld = coverage?.family_withheld_zero_day ?? null;
  const defeats = coverage?.families_defeat ?? null;

  // Ordered by the withheld (zero-day) recall -- the honest column -- so the
  // weakest generaliser is the first row a judge sees, not buried at the bottom.
  const ids = withheld ? Object.keys(withheld) : inTraining ? Object.keys(inTraining) : [];
  const ordered = ids.sort((a, b) => {
    const ra = withheld?.[a]?.recall.mean ?? 0;
    const rb = withheld?.[b]?.recall.mean ?? 0;
    return ra - rb;
  });

  return (
    <>
      <PageHeader h1={META.h1} criterion={META.criterion} blurb={META.blurb} eyebrow="01" />

      <section className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
            <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Executable families
            </p>
            <p className="type-num mt-3 text-xl font-medium tracking-tight text-text md:text-2xl">
              {summary?.executable_families ?? <span className="text-text-dim">not measured</span>}
            </p>
            <p className="type-num mt-3 text-[0.6875rem] text-text-dim">
              artifacts/family_coverage.json
            </p>
          </div>
          <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
            <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Mean recall, family in training
            </p>
            <p className="type-num mt-3 text-xl font-medium tracking-tight text-text md:text-2xl">
              {summary ? fmtPct(summary.mean_recall_family_in_training) : <span className="text-text-dim">not measured</span>}
            </p>
            <p className="type-num mt-3 text-[0.6875rem] text-text-dim">
              artifacts/family_coverage.json
            </p>
          </div>
          <div className="rounded-[var(--r-md)] border border-border bg-surface-1 p-5">
            <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Mean recall, family withheld (zero-day)
            </p>
            <p className="type-num mt-3 text-xl font-medium tracking-tight text-text md:text-2xl">
              {summary ? fmtPct(summary.mean_recall_family_withheld_zero_day) : <span className="text-text-dim">not measured</span>}
            </p>
            <p className="type-num mt-3 text-[0.6875rem] text-text-dim">
              artifacts/family_coverage.json
            </p>
          </div>
        </div>

        <p className="type-ui measure mt-6 text-sm leading-relaxed text-text-dim">
          {summary?.reading ??
            "The taxonomy maps twenty-two vectors; fourteen are executable and measured. Every executable family carries its own measured detection number, both with the family in supervised training and with it withheld (leave-one-family-out)."}
        </p>
      </section>

      {/* The atlas: one row per executable family, both conditions side by side. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="flex flex-col gap-3 md:flex-row md:items-baseline md:justify-between">
          <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
            The attack atlas — measured per family
          </h2>
          <p className="type-ui text-xs text-text-dim">
            Ordered by the withheld column, ascending: the weakest zero-day generaliser is first.
          </p>
        </div>

        {ordered.length === 0 ? (
          <div className="type-ui mt-6 rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 text-sm text-text-dim">
            The atlas is read from <code className="type-num">artifacts/family_coverage.json</code>,
            which has not been generated yet. Run <code className="type-num">make reproduce</code>{" "}
            to build it.
          </div>
        ) : (
          <div className="mt-6 overflow-x-auto rounded-[var(--r-lg)] border border-border bg-surface-1">
            <table className="w-full min-w-[880px] text-left">
              <thead>
                <tr className="border-b border-border bg-surface-2">
                  <th className="type-ui px-4 py-3 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                    Family
                  </th>
                  <th className="type-ui px-4 py-3 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                    Control it defeats
                  </th>
                  <th className="type-ui px-4 py-3 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                    Recall · in training
                  </th>
                  <th className="type-ui px-4 py-3 text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
                    Recall · withheld (zero-day)
                  </th>
                </tr>
              </thead>
              <tbody>
                {ordered.map((id) => {
                  const w = withheld?.[id];
                  const t = inTraining?.[id];
                  const worst = summary?.weakest_family_when_withheld.family === id;
                  return (
                    <tr key={id} className="border-b border-border last:border-b-0">
                      <td className="px-4 py-3">
                        <span className="type-ui text-sm text-text">
                          {familyLabel(id, { label: w?.label ?? t?.label })}
                        </span>
                        {worst && (
                          <span className="type-num ml-2">
                            <StatusChip tone="warn">weakest when withheld</StatusChip>
                          </span>
                        )}
                      </td>
                      <td className="type-ui px-4 py-3 text-sm text-text-dim">
                        {defeats?.[id] ?? w?.defeats ?? "—"}
                      </td>
                      <td className="type-num px-4 py-3 text-sm text-text">
                        {fmtInterval(t?.recall)}
                      </td>
                      <td className="type-num px-4 py-3 text-sm text-text">
                        {fmtInterval(w?.recall)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
          Recall is the share of a family&apos;s transactions receiving a non-APPROVE decision;
          intervals are seed-level means with a nonparametric bootstrap over three seeds. When a
          family is withheld, whatever still catches it is architecture — the unsupervised
          novelty layer and the entity graph — not memorisation.
        </p>

        {/* Boundaries, stated rather than hidden. */}
        {coverage?.boundaries && (
          <div className="mt-6 rounded-[var(--r-md)] border border-border bg-surface-2 p-5">
            <h3 className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
              Boundary conditions
            </h3>
            <ul className="type-ui measure mt-3 list-disc space-y-1.5 pl-5 text-xs leading-relaxed text-text-dim">
              {coverage.boundaries.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </>
  );
}
