/**
 * Status chip — a small, monospaced fact about the running system.
 *
 * Presentational and props-only. `tone` is restricted to the two accent hues
 * plus the neutral/pass/warn states, so a chip can never introduce a third
 * accent colour.
 */

export type ChipTone = "neutral" | "red" | "blue" | "pass" | "warn";

const TONE_CLASS: Record<ChipTone, string> = {
  neutral: "border-border text-text-dim",
  red: "border-red/40 bg-red-dim/40 text-red",
  blue: "border-blue/40 bg-blue-dim/40 text-blue",
  pass: "border-pass/40 text-pass",
  warn: "border-warn/40 text-warn",
};

export interface StatusChipProps {
  children: React.ReactNode;
  tone?: ChipTone;
  /** Longer explanation surfaced as a native tooltip and to screen readers. */
  title?: string;
}

export function StatusChip({ children, tone = "neutral", title }: StatusChipProps) {
  return (
    <span
      title={title}
      className={`type-num inline-flex items-center gap-1.5 whitespace-nowrap rounded-[var(--r-sm)] border px-2 py-1 text-[0.6875rem] leading-none tracking-tight ${TONE_CLASS[tone]}`}
    >
      {children}
    </span>
  );
}
