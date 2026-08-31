/**
 * SectionPending — an honest marker for a route whose evidence sections are
 * not wired yet.
 *
 * This exists so the navigation is never broken during the phased build: every
 * route in the nav resolves to a real, styled, legible page from the first
 * phase onward. It states plainly what will appear and which artifact will
 * supply it, and it renders NO numbers at all -- a placeholder metric would be
 * indistinguishable from a fabricated one.
 */

export interface SectionPendingProps {
  /** What this route will contain, in plain prose. */
  summary: string;
  /** Artifact filenames that will supply this route's numbers. */
  sources: readonly string[];
}

export function SectionPending({ summary, sources }: SectionPendingProps) {
  return (
    <section className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-6 md:py-16">
      <div className="rounded-[var(--r-lg)] border border-border bg-surface-1 p-6 md:p-8">
        <h2 className="type-ui text-sm font-semibold tracking-tight text-text">
          Evidence sections for this route are not wired yet
        </h2>
        <p className="type-ui measure mt-3 text-sm leading-relaxed text-text-dim">{summary}</p>

        <div className="mt-6">
          <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
            Will read from
          </p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {sources.map((source) => (
              <li
                key={source}
                className="type-num rounded-[var(--r-sm)] border border-border bg-surface-2 px-2 py-1 text-[0.6875rem] text-text-dim"
              >
                {source}
              </li>
            ))}
          </ul>
        </div>

        <p className="type-ui measure mt-6 text-xs leading-relaxed text-text-dim">
          No number is shown above, by design. Every figure on this site is read at
          runtime from a generated artifact, so a placeholder value would be
          indistinguishable from a fabricated one.
        </p>
      </div>
    </section>
  );
}
