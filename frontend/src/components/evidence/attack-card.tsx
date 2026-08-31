"use client";

/**
 * <AttackCard> — one taxon of the attack atlas (SECTION 5).
 *
 * Static prose facts (title, channel, rail, GenAI enabler, what it defeats)
 * arrive from the taxonomy data; the MEASURED recall arrives from
 * family_coverage.json via the page as pre-validated props. A [SPEC] taxon
 * with no measurement renders an explicit "not yet executable" pill — never
 * a placeholder number.
 */

import { fmtPct, type Interval } from "@/lib/format";

import { Claim } from "./claim";

export interface AttackCardProps {
  /** Taxon id, e.g. "T-12". */
  id: string;
  title: string;
  /** Payment channel, e.g. "UPI" / "card-not-present". Static prose. */
  channel: string;
  /** Rail or surface attacked, e.g. "issuing" / "acquiring". Static prose. */
  rail: string;
  /** What GenAI changed about this fraud. Static prose from the taxonomy. */
  genaiEnabler: string;
  /** The defensive signal the attack is designed to defeat. */
  defeats: string;
  /** Executable [CODE] taxon with a measured family behind it. */
  executable: boolean;
  /** Measured recall with the family in supervised training, or null. */
  recallInTraining: Interval | null;
  /** Measured recall with the family withheld (zero-day), or null. */
  recallWithheld: Interval | null;
  /** Artifact path for the Claim footers and console warnings. */
  artifactPath: string;
  /** Command that regenerates the measurement. */
  reproduceCmd?: string;
}

export function AttackCard({
  id,
  title,
  channel,
  rail,
  genaiEnabler,
  defeats,
  executable,
  recallInTraining,
  recallWithheld,
  artifactPath,
  reproduceCmd,
}: AttackCardProps) {
  return (
    <article className="flex flex-col gap-3 rounded-[var(--r-lg)] border border-border bg-surface-1 p-4">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="type-num text-sm font-semibold text-text-faint">{id}</span>
        <h3 className="type-ui text-sm font-semibold text-text">{title}</h3>
        <span className="ms-auto flex flex-wrap gap-1.5">
          <Badge tone="neutral">{channel}</Badge>
          <Badge tone="neutral">{rail}</Badge>
          {executable ? (
            <Badge tone="pass">executable</Badge>
          ) : (
            <Badge tone="warn">not yet executable</Badge>
          )}
        </span>
      </header>

      <p className="type-ui measure text-xs text-text-dim">
        <span className="text-text-faint">GenAI enabler — </span>
        {genaiEnabler}
      </p>
      <p className="type-ui measure text-xs text-text-dim">
        <span className="text-text-faint">Defeats — </span>
        {defeats}
      </p>

      {executable && (
        <div className="mt-1 grid gap-3 border-t border-border pt-3 sm:grid-cols-2">
          <Claim
            variant="inline"
            label="recall · in training"
            value={fmtPct(recallInTraining?.mean ?? null)}
            ci={recallInTraining}
            interpretation="Share of this family's transactions receiving a non-APPROVE decision, with the family present in supervised training."
            artifactPath={artifactPath}
            reproduceCmd={reproduceCmd}
            tone="blue"
          />
          <Claim
            variant="inline"
            label="recall · withheld (zero-day)"
            value={fmtPct(recallWithheld?.mean ?? null)}
            ci={recallWithheld}
            interpretation="Same family absent from supervised training: whatever still catches it is architecture, not memorisation."
            artifactPath={artifactPath}
            reproduceCmd={reproduceCmd}
            tone="blue"
          />
        </div>
      )}
    </article>
  );
}

function Badge({ tone, children }: { tone: "neutral" | "pass" | "warn"; children: string }) {
  const tones: Record<typeof tone, string> = {
    neutral: "border-border-hi text-text-dim",
    pass: "border-pass/40 text-pass",
    warn: "border-warn/40 text-warn",
  };
  return (
    <span
      className={`type-ui rounded-full border px-2 py-0.5 text-[0.625rem] uppercase tracking-[0.06em] ${tones[tone]}`}
    >
      {children}
    </span>
  );
}
