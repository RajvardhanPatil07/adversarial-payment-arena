"use client";

/**
 * <ArtifactChip> — the address of one number.
 *
 * Mono path + a link that opens the raw JSON + a copy button. The link targets
 * the STATIC snapshot under /artifacts/ (copied verbatim by
 * `npm run snapshot:artifacts`), so it works with the backend cold and serves
 * byte-identical JSON to what the pages render from.
 */

import { SITE } from "@/lib/site";

export interface ArtifactChipProps {
  /** Repo-relative artifact path, e.g. "artifacts/closed_loop.json". */
  path: string;
  /** Where the chip is used; lets pages add context in a title tooltip. */
  note?: string;
}

/** "artifacts/closed_loop.json" -> "/artifacts/closed_loop.json" (static copy). */
function snapshotHref(path: string): string {
  const trimmed = path.replace(/^\.?\//, "");
  return `/${trimmed}`;
}

export function ArtifactChip({ path, note }: ArtifactChipProps) {
  return (
    <span className="inline-flex items-center gap-2 rounded-[var(--r-sm)] border border-border bg-surface-1 py-0.5 pl-2 pr-1 align-middle">
      <a
        href={snapshotHref(path)}
        target="_blank"
        rel="noreferrer"
        title={(note ? `${note} — ` : "") + `Open the raw JSON this number is read from: ${path}`}
        className="type-num text-[0.6875rem] text-text-dim underline decoration-border-hi underline-offset-2 transition-colors hover:text-blue"
      >
        {path}
      </a>
      <button
        type="button"
        onClick={() => {
          void navigator.clipboard?.writeText(`${SITE.repo}/blob/main/${path.replace(/^\.?\//, "")}`);
        }}
        title={`Copy the GitHub link to ${path}`}
        aria-label={`Copy link to ${path}`}
        className="type-num rounded-[var(--r-sm)] border border-border px-1.5 text-[0.625rem] text-text-faint transition-colors hover:border-border-hi hover:text-text"
      >
        ⧉
      </button>
    </span>
  );
}
