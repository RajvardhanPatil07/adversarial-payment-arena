/**
 * Client-safe value formatters.
 *
 * Pure functions over plain shapes — no node:fs, no server imports — so
 * "use client" components (Claim, Scissor, HardeningCurve, AttackCard) share
 * EXACTLY the formatting rules the server pages use, from one module.
 *
 * Rule (SECTION 1B): these formatters never invent a number. A null/absent
 * input formats to null, and the component renders "— not measured".
 */

/** Bootstrap 95% CI shape — shared by nearly every measured field. */
export interface Interval {
  mean: number;
  lo: number;
  hi: number;
  n: number;
  method?: string;
}

/** Is `v` a plausible Interval? Used by validators and client components alike. */
export function isInterval(v: unknown): v is Interval {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.mean === "number" &&
    typeof o.lo === "number" &&
    typeof o.hi === "number" &&
    typeof o.n === "number"
  );
}

/** Format an Interval as "mean (lo–hi)" in the caller's unit. Mono-rendered. */
export function fmtInterval(i: Interval | null | undefined, unit = ""): string | null {
  if (!i) return null;
  const f = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
  return `${f(i.mean)}${unit} (${f(i.lo)}–${f(i.hi)})`;
}

/** Format a bare number, or null when the value is absent. */
export function fmtNum(v: number | null | undefined, digits = 3): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return v.toFixed(digits);
}

/** Signed percentage-point delta, e.g. "-35.8 pts". Null when absent. */
export function fmtDeltaPts(v: number | null | undefined): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  const sign = v > 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(1)} pts`;
}

/** Absolute percentage, e.g. "48.3%". Null when absent. */
export function fmtPct(v: number | null | undefined, digits = 1): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return `${(v * 100).toFixed(digits)}%`;
}

/** Indian-rupee amount, e.g. "₹2,29,33,60,80". Null when absent. */
export function fmtInr(v: number | null | undefined): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return `₹${new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(v)}`;
}

/** Strip the ISO timestamp to its date for compact display. */
export function fmtDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  return iso.slice(0, 10);
}
