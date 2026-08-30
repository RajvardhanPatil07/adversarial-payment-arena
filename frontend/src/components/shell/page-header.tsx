/**
 * PageHeader — the claim at the top of every route.
 *
 * The H1 is always a sentence making a claim, never a noun label, and it is
 * always paired with the judging criterion the page answers. That pairing is
 * the point: a judge should be able to see which part of their scorecard they
 * are looking at without reading the body.
 *
 * Presentational and props-only.
 */

import type { Criterion } from "@/lib/site";

export interface PageHeaderProps {
  /** The claim. Rendered as the page's only H1. */
  h1: string;
  /** The judging criterion this page answers. */
  criterion: Criterion;
  /** One-paragraph framing, constrained to the 68ch prose measure. */
  blurb?: string;
  /** Optional slot for status chips or actions, right-aligned on desktop. */
  aside?: React.ReactNode;
  /** Optional eyebrow, e.g. a section number. */
  eyebrow?: string;
}

export function PageHeader({ h1, criterion, blurb, aside, eyebrow }: PageHeaderProps) {
  return (
    <header className="border-b border-border bg-surface-1">
      <div className="mx-auto max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
        <div className="flex flex-col gap-8 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <p className="type-num text-[0.6875rem] uppercase tracking-[0.12em] text-text-dim">
              {eyebrow ? `${eyebrow} · ` : ""}
              answers: {criterion}
            </p>

            {/* The claim. Balanced wrapping keeps it legible as a still image,
             * which matters because these pages get screenshotted into decks. */}
            <h1 className="type-ui measure mt-4 text-pretty text-2xl font-semibold leading-[1.2] tracking-tight text-text md:text-4xl md:leading-[1.15]">
              {h1}
            </h1>

            {blurb && (
              <p className="type-ui measure mt-5 text-sm leading-relaxed text-text-dim md:text-base">
                {blurb}
              </p>
            )}
          </div>

          {aside && <div className="flex shrink-0 flex-wrap items-center gap-2">{aside}</div>}
        </div>
      </div>
    </header>
  );
}
