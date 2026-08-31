/**
 * <HardeningCurve> — real-fraud recall across loop generations, per seed.
 *
 * Thin translucent per-seed paths (ungated in the attacker hue, gated in the
 * defender hue) behind a bold mean path per arm, with a flat dashed reference
 * line at the calibrated 1% legitimate-FPR budget. Static by design: readable
 * as a screenshot, identical on the server.
 *
 * Every series arrives as a prop, read from closed_loop.json by the page.
 * A null/absent series is skipped with a console warning — never invented.
 */

export interface SeedSeries {
  seed: number;
  /** Real-fraud recall per generation, or null when the seed is absent. */
  values: number[] | null;
}

export interface HardeningCurveProps {
  /** Generation labels along the x-axis, e.g. ["V0","V1","V2","V3"]. */
  generations: string[];
  /** Ungated arm per-seed real-fraud recall (attacker hue). */
  ungatedSeeds: SeedSeries[];
  /** Gated arm per-seed real-fraud recall (defender hue). */
  gatedSeeds: SeedSeries[];
  /** Ungated arm mean real-fraud recall per generation, or null. */
  ungatedMean: number[] | null;
  /** Gated arm mean real-fraud recall per generation, or null. */
  gatedMean: number[] | null;
  /** Calibrated legitimate-FPR reference (e.g. 0.01), or null. */
  fprReference: number | null;
  /** Artifact path for the console warning on missing series. */
  artifactPath: string;
}

const W = 720;
const H = 320;
const PAD_L = 52;
const PAD_R = 24;
const PAD_T = 24;
const PAD_B = 40;

const Y_TOP = 1.05;
const Y_BOT = 0;

function y(v: number): number {
  return PAD_T + (H - PAD_T - PAD_B) * (1 - (v - Y_BOT) / (Y_TOP - Y_BOT));
}

function x(i: number, n: number): number {
  const span = W - PAD_L - PAD_R;
  return n <= 1 ? PAD_L + span / 2 : PAD_L + (span * i) / (n - 1);
}

function pts(values: number[], n: number): string {
  return values.map((v, i) => `${x(i, n).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
}

function warn(path: string, what: string): void {
  console.warn(`[evidence] artifact field missing: ${path} (${what})`);
}

function series(values: number[] | null | undefined, n: number, artifactPath: string, what: string): number[] | null {
  if (values && values.length === n) return values;
  warn(artifactPath, what);
  return null;
}

export function HardeningCurve({
  generations,
  ungatedSeeds,
  gatedSeeds,
  ungatedMean,
  gatedMean,
  fprReference,
  artifactPath,
}: HardeningCurveProps) {
  const n = generations.length;

  return (
    <figure
      aria-label="Real-fraud recall across loop generations. Ungated arm: per-seed paths fall away as low-fidelity batches pollute retraining. Gated arm: per-seed paths stay together. A dashed reference marks the 1% legitimate false-positive-rate budget."
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="w-full">
        {/* y gridlines */}
        {[1.0, 0.75, 0.5, 0.25].map((g) => (
          <g key={g}>
            <line x1={PAD_L} x2={W - PAD_R} y1={y(g)} y2={y(g)} stroke="var(--border)" strokeWidth="1" />
            <text x={PAD_L - 8} y={y(g) + 3.5} textAnchor="end" className="type-num" fontSize="10" fill="var(--text-faint)">
              {g.toFixed(2)}
            </text>
          </g>
        ))}

        {/* x-axis generation labels */}
        {generations.map((g, i) => (
          <text key={g} x={x(i, n)} y={H - 14} textAnchor="middle" className="type-num" fontSize="10.5" fill="var(--text-dim)">
            {g}
          </text>
        ))}

        {/* flat dashed 1% FPR reference */}
        {fprReference !== null && (
          <g>
            <line
              x1={PAD_L}
              x2={W - PAD_R}
              y1={y(fprReference)}
              y2={y(fprReference)}
              stroke="var(--text-faint)"
              strokeWidth="1.5"
              strokeDasharray="6 6"
            />
            <text x={W - PAD_R} y={y(fprReference) - 6} textAnchor="end" className="type-num" fontSize="10" fill="var(--text-faint)">
              1% legit FPR reference
            </text>
          </g>
        )}

        {/* thin translucent per-seed paths */}
        {ungatedSeeds.map((s) => {
          const v = series(s.values, n, artifactPath, `ungated seed ${s.seed}`);
          return v ? (
            <polyline key={`u${s.seed}`} points={pts(v, n)} fill="none" stroke="var(--red)" strokeOpacity="0.35" strokeWidth="1.5" />
          ) : null;
        })}
        {gatedSeeds.map((s) => {
          const v = series(s.values, n, artifactPath, `gated seed ${s.seed}`);
          return v ? (
            <polyline key={`g${s.seed}`} points={pts(v, n)} fill="none" stroke="var(--blue)" strokeOpacity="0.35" strokeWidth="1.5" />
          ) : null;
        })}

        {/* bold mean paths */}
        {(() => {
          const m = series(ungatedMean, n, artifactPath, "ungated mean");
          return m ? <polyline points={pts(m, n)} fill="none" stroke="var(--red)" strokeWidth="3" /> : null;
        })()}
        {(() => {
          const m = series(gatedMean, n, artifactPath, "gated mean");
          return m ? <polyline points={pts(m, n)} fill="none" stroke="var(--blue)" strokeWidth="3" /> : null;
        })()}

        {/* on-plot arm labels — no legend, per SECTION 5 */}
        <text x={x(0, n)} y={PAD_T - 8} textAnchor="start" className="type-num" fontSize="10.5" fontWeight="700" fill="var(--red)">
          ungated
        </text>
        <text x={W - PAD_R} y={PAD_T - 8} textAnchor="end" className="type-num" fontSize="10.5" fontWeight="700" fill="var(--blue)">
          gated
        </text>

        {/* frame */}
        <rect x={PAD_L} y={PAD_T} width={W - PAD_L - PAD_R} height={H - PAD_T - PAD_B} fill="none" stroke="var(--border)" strokeWidth="1" />
      </svg>
      <figcaption className="type-num mt-2 text-center text-[0.6875rem] text-text-faint">
        real-fraud recall per generation · thin = one seed, bold = seed mean · {artifactPath}
      </figcaption>
    </figure>
  );
}
