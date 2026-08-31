import type { Metadata } from "next";

import { PageHeader } from "@/components/shell/page-header";
import { StatusChip } from "@/components/shell/status-chip";
import { Boundary } from "@/components/evidence/boundary";
import { Claim } from "@/components/evidence/claim";
import { Reveal } from "@/components/evidence/reveal";
import { AttackAtlas, type AtlasEntry } from "@/components/evidence/attack-atlas";
import { loadArtifact } from "@/lib/artifacts";
import { fmtPct } from "@/lib/format";
import { TAXONOMY } from "@/data/taxonomy";
import { ROUTE } from "@/lib/site";

const META = ROUTE["/identify"];
const ARTIFACT = "artifacts/family_coverage.json";

export const metadata: Metadata = {
  title: "Identify",
  description: META.blurb,
};

export default async function IdentifyPage() {
  const coverageR = await loadArtifact("family_coverage");
  const coverage = coverageR.ok ? coverageR.data : null;

  const summary = coverage?.summary ?? null;
  const inTraining = coverage?.family_in_training ?? null;
  const withheld = coverage?.family_withheld_zero_day ?? null;

  // Join the 22-taxon taxonomy to the measured families. The prose facts are
  // static; every measurement is read from the artifact at render time. A
  // [SPEC] taxon carries no family key, so it can never pick up another
  // family's number — it renders "not yet executable", never a placeholder.
  const entries: AtlasEntry[] = TAXONOMY.map((taxon) => {
    const fam = taxon.family;
    const t = fam ? (inTraining?.[fam] ?? null) : null;
    const w = fam ? (withheld?.[fam] ?? null) : null;
    const defeatsFromArtifact = fam ? (coverage?.families_defeat[fam] ?? null) : null;
    return {
      id: taxon.id,
      title: taxon.title,
      channel: taxon.channel,
      rail: taxon.rail,
      genaiEnabler: taxon.genaiEnabler,
      defeats: defeatsFromArtifact ?? taxon.defeats,
      executable: taxon.executable,
      recallInTraining: taxon.executable ? (t?.recall ?? null) : null,
      recallWithheld: taxon.executable ? (w?.recall ?? null) : null,
      artifactPath: taxon.executable
        ? `${ARTIFACT} · family_in_training["${fam}"].recall`
        : ARTIFACT,
      reproduceCmd: "make coverage",
    };
  });

  // Weakest zero-day generaliser first: order by withheld recall, ascending,
  // then by in-training recall, then by id for a stable order. [SPEC] taxa sort
  // last so a judge first meets the families that carry measurements.
  entries.sort((a, b) => {
    if (a.executable !== b.executable) return a.executable ? -1 : 1;
    const ra = a.recallWithheld?.mean ?? 0;
    const rb = b.recallWithheld?.mean ?? 0;
    if (ra !== rb) return ra - rb;
    const ta = a.recallInTraining?.mean ?? 0;
    const tb = b.recallInTraining?.mean ?? 0;
    if (ta !== tb) return ta - tb;
    return a.id.localeCompare(b.id);
  });

  const weakest = summary?.weakest_family_when_withheld ?? null;

  return (
    <>
      <PageHeader h1={META.h1} criterion={META.criterion} blurb={META.blurb} eyebrow="01" />

      <section className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
        <Reveal>
          <div className="grid gap-4 md:grid-cols-3">
            <Claim
              variant="card"
              label="Executable families"
              value={summary ? `${summary.executable_families}` : null}
              interpretation="Fourteen of the twenty-two mapped taxa are executable: generated, admitted through the Plausibility Gate, and individually measured."
              artifactPath={`${ARTIFACT} · summary.executable_families`}
              reproduceCmd="make coverage"
              tone="neutral"
            />
            <Claim
              variant="card"
              label="Mean recall, family in training"
              value={summary ? fmtPct(summary.mean_recall_family_in_training) : null}
              interpretation="Per-family non-APPROVE rate with the family present in supervised training, averaged across families."
              artifactPath={`${ARTIFACT} · summary.mean_recall_family_in_training`}
              reproduceCmd="make coverage"
              tone="blue"
            />
            <Claim
              variant="card"
              label="Mean recall, family withheld (zero-day)"
              value={summary ? fmtPct(summary.mean_recall_family_withheld_zero_day) : null}
              interpretation="The same families absent from supervised training: whatever still catches them is architecture, not memorisation."
              artifactPath={`${ARTIFACT} · summary.mean_recall_family_withheld_zero_day`}
              reproduceCmd="make coverage"
              tone="blue"
            />
          </div>
        </Reveal>

        <p className="type-ui measure mt-6 text-sm leading-relaxed text-text-dim">
          {summary?.reading ??
            "The taxonomy maps twenty-two vectors; fourteen are executable and measured. Every executable family carries its own measured detection number, both with the family in supervised training and with it withheld (leave-one-family-out)."}
        </p>

        {weakest && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <StatusChip tone="warn">weakest when withheld</StatusChip>
            <span className="type-ui text-xs text-text-dim">
              {weakest.label} — {fmtPct(weakest.recall)} when absent from supervised training; it defeats{" "}
              {weakest.defeats}.
            </span>
          </div>
        )}
      </section>

      {/* The atlas: 22 taxa, filters, per-family measured recall on every
          executable card. */}
      <section className="mx-auto w-full max-w-[1400px] px-4 pb-12 md:px-6 md:pb-16">
        <div className="flex flex-col gap-3 md:flex-row md:items-baseline md:justify-between">
          <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
            The attack atlas — 22 mapped, 14 executable
          </h2>
          <p className="type-ui text-xs text-text-dim">
            Ordered by withheld (zero-day) recall, ascending: the weakest generaliser is first.
          </p>
        </div>

        <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">
          Stated honestly: fourteen of twenty-two are executable. Claiming twenty-two{" "}
          <em>implemented</em> attacks would be the kind of number this repository is built to
          argue against. Each unmapped row names its fields and its target signal, so each is an
          afternoon of work rather than a research question.
        </p>

        {coverageR.ok ? (
          <div className="mt-6">
            <AttackAtlas entries={entries} />
          </div>
        ) : (
          <div className="type-ui mt-6 rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 text-sm text-text-dim">
            The atlas is read from <code className="type-num">artifacts/family_coverage.json</code>,
            which is unavailable. Run <code className="type-num">make reproduce</code> to build it.
            {coverageR.ok === false && (
              <span className="type-num mt-2 block text-xs text-text-faint">
                missing: {coverageR.missing.join(", ")}
              </span>
            )}
          </div>
        )}

        <p className="type-ui measure mt-4 text-xs leading-relaxed text-text-dim">
          Recall is the share of a family&apos;s transactions receiving a non-APPROVE decision;
          intervals are seed-level means with a nonparametric bootstrap over three seeds. When a
          family is withheld, whatever still catches it is architecture — the unsupervised
          novelty layer and the entity graph — not memorisation.
        </p>

        {coverage?.boundaries && (
          <div className="mt-6">
            <Boundary title="Boundary conditions" items={coverage.boundaries} />
          </div>
        )}
      </section>
    </>
  );
}
