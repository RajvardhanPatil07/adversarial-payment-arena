"use client";

/**
 * <Claim> — one measured number, fully sourced.
 *
 * The unit of discourse on this site. A Claim renders:
 *   - an uppercase, letterspaced label,
 *   - the measured value in mono tabular figures, tone-coloured,
 *   - a 95% CI when the measurement is an interval,
 *   - one plain-English interpretation sentence,
 *   - a footer chip row: the artifact path (opens the raw JSON) and the
 *     reproduce command (copy-on-click).
 *
 * A null/absent value renders "— not measured", never a fallback number.
 * The missing artifact path is also console.warn'd so a broken evidence set
 * is discoverable from the devtools console as well as the page.
 *
 * Presentational only: every string and number arrives as a prop.
 */

import { fmtInterval, type Interval } from "@/lib/format";

import { ArtifactChip } from "./artifact-chip";

export type ClaimTone = "neutral" | "red" | "blue" | "pass" | "fail";

export interface ClaimProps {
  label: string;
  /** The measured value, pre-formatted by the caller (loader formatters). */
  value: string | null;
  /** Bootstrap 95% interval, when the measurement is an interval. */
  ci?: Interval | null;
  unit?: string;
  /** One plain-English sentence: what this number means. */
  interpretation: string;
  /** Artifact path as recorded in the repo, e.g. "artifacts/closed_loop.json". */
  artifactPath: string;
  /** Command that regenerates this number. Copy-on-click. */
  reproduceCmd?: string;
  tone?: ClaimTone;
  /** hero = page-centrepiece size, card = grid tile, inline = row density. */
  variant?: "hero" | "card" | "inline";
}

const TONE_TEXT: Record<ClaimTone, string> = {
  neutral: "text-text",
  red: "text-red",
  blue: "text-blue",
  pass: "text-pass",
  fail: "text-fail",
};

const VALUE_SIZE: Record<NonNullable<ClaimProps["variant"]>, string> = {
  hero: "text-4xl md:text-5xl",
  card: "text-3xl",
  inline: "text-base",
};

export function Claim({
  label,
  value,
  ci = null,
  unit = "",
  interpretation,
  artifactPath,
  reproduceCmd,
  tone = "neutral",
  variant = "card",
}: ClaimProps) {
  if (value === null || value === undefined || value === "") {
    // SECTION 1B: the missing path is surfaced, not papered over.
    console.warn(`[evidence] artifact field missing: ${artifactPath}`);
    return (
      <div className="flex flex-col gap-2">
        <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">{label}</p>
        <p className={`type-num font-semibold ${TONE_TEXT[tone]} ${VALUE_SIZE[variant]}`}>—</p>
        <p className="type-ui text-xs text-text-dim">not measured</p>
        <ArtifactChip path={artifactPath} />
      </div>
    );
  }

  const ciText = fmtInterval(ci, "");

  return (
    <div className="flex flex-col gap-2">
      <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">{label}</p>
      <p className={`type-num font-semibold ${TONE_TEXT[tone]} ${VALUE_SIZE[variant]}`}>
        {value}
        {unit && !value.includes(unit) && <span className="ml-1 text-base font-normal text-text-dim">{unit}</span>}
      </p>
      {ciText && <p className="type-num text-xs text-text-dim">95% CI {ciText}</p>}
      <p className="type-ui measure text-xs text-text-dim">{interpretation}</p>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <ArtifactChip path={artifactPath} />
        {reproduceCmd && <ReproduceChip cmd={reproduceCmd} />}
      </div>
    </div>
  );
}

function ReproduceChip({ cmd }: { cmd: string }) {
  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard?.writeText(cmd);
      }}
      title={`Copy: ${cmd}`}
      className="type-num rounded-[var(--r-sm)] border border-border bg-surface-1 px-2 py-1 text-[0.6875rem] text-text-dim transition-colors hover:border-border-hi hover:text-text"
    >
      $ {cmd}
    </button>
  );
}
