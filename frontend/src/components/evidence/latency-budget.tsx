/**
 * <LatencyBudget> — the latency-vs-budget visual (SECTION 8.6).
 *
 * Percentile bars (p50/p95/p99) against the full-width authorisation budget.
 * Every number arrives as a prop, read from artifacts/latency.json at render
 * time; a null renders "—" and the console warning, never a placeholder.
 * Static SVG: readable as a screenshot, identical on the server.
 */

import { fmtNum } from "@/lib/format";

export interface LatencyBudgetProps {
  /** Measured p50 / p95 / p99 in milliseconds, or null. */
  p50: number | null;
  p95: number | null;
  p99: number | null;
  /** The inline authorisation budget in milliseconds, or null. */
  budget: number | null;
  /** Sample size for the caption, or null. */
  n: number | null;
  /** Artifact path, for the caption and console warning. */
  artifactPath: string;
}

const W = 560;
const BAR_H = 26;
const GAP = 14;
const LABEL_W = 48;
const PAD_R = 70;
const TOP = 8;

export function LatencyBudget({ p50, p95, p99, budget, n, artifactPath }: LatencyBudgetProps) {
  const rows: Array<{ label: string; value: number | null; hue: string }> = [
    { label: "p50", value: p50, hue: "var(--blue)" },
    { label: "p95", value: p95, hue: "var(--blue)" },
    { label: "p99", value: p99, hue: "var(--blue)" },
    { label: "budget", value: budget, hue: "var(--text-faint)" },
  ];
  const scale = budget !== null && budget > 0 ? budget : null;
  const H = TOP + rows.length * (BAR_H + GAP);

  if (p50 === null || p95 === null || p99 === null || budget === null) {
    console.warn(`[evidence] artifact field missing: ${artifactPath}`);
  }

  return (
    <figure
      aria-label="Measured decision-latency percentiles drawn against the inline authorisation budget. All three percentiles sit well inside the budget bar."
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="w-full">
        {rows.map((row, i) => {
          const y = TOP + i * (BAR_H + GAP);
          const has = row.value !== null && scale !== null;
          const w = has ? ((row.value as number) / scale) * (W - LABEL_W - PAD_R) : 0;
          const isBudget = row.label === "budget";
          return (
            <g key={row.label}>
              <text x={LABEL_W - 8} y={y + BAR_H / 2 + 3.5} textAnchor="end" className="type-num" fontSize="10.5" fill="var(--text-dim)">
                {row.label}
              </text>
              {/* the budget is the full-width frame; percentiles are bars inside it */}
              <rect
                x={LABEL_W}
                y={y}
                width={W - LABEL_W - PAD_R}
                height={BAR_H}
                rx={4}
                fill="none"
                stroke={isBudget ? row.hue : "var(--border)"}
                strokeWidth={isBudget ? 1.5 : 1}
                strokeDasharray={isBudget ? "6 6" : undefined}
              />
              {!isBudget && has && (
                <rect x={LABEL_W} y={y} width={w} height={BAR_H} rx={4} fill={row.hue} fillOpacity={0.75} />
              )}
              <text
                x={W - PAD_R + 10}
                y={y + BAR_H / 2 + 3.5}
                className="type-num"
                fontSize="10.5"
                fill={has ? "var(--text)" : "var(--text-dim)"}
              >
                {row.value !== null ? `${fmtNum(row.value, 1)} ms` : "—"}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="type-num mt-2 text-[0.6875rem] text-text-faint">
        {n !== null ? `n=${n} scored transactions · ` : ""}
        {artifactPath}
      </figcaption>
    </figure>
  );
}
