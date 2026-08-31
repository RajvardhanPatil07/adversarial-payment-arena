/**
 * <Scissor> — the two-panel divergence chart.
 *
 * Left panel: the UNGATED loop. Recall on the loop's own synthetic attacks
 * (the vanity metric) rises while recall on held-out REAL fraud falls — the
 * two lines scissor apart. Right panel: the GATED loop, where the fidelity
 * gate rejects the low-fidelity batches and the two metrics move together.
 *
 * Per SECTION 5 there is NO legend: each line is labelled on-plot, and the
 * measured deltas are annotated on the plot itself. Every number arrives as a
 * prop; the component renders "—" and warns to the console when a series is
 * missing rather than drawing an invented line.
 *
 * Static by design (no animation, no hooks): it must be readable as a
 * screenshot, and it renders identically on the server.
 */

export interface ScissorProps {
  /** Generation labels along the x-axis, e.g. ["V0","V1","V2","V3"]. */
  generations: string[];
  /** Ungated arm: recall on held-out REAL fraud per generation, or null. */
  ungatedReal: number[] | null;
  /** Ungated arm: recall on the loop's own synthetic attacks, or null. */
  ungatedSynthetic: number[] | null;
  /** Gated arm: recall on held-out REAL fraud per generation, or null. */
  gatedReal: number[] | null;
  /** Gated arm: recall on the loop's own synthetic attacks, or null. */
  gatedSynthetic: number[] | null;
  /** Where the artifact path should point, for the console warning. */
  artifactPath: string;
}

const W = 460; // panel width
const H = 300; // panel height
const PAD_L = 56;
const PAD_R = 16;
const PAD_T = 40;
const PAD_B = 36;

/** Map a recall value (0..1) to panel y. Headroom above 1.0 for line labels. */
function y(v: number): number {
  const top = 1.12; // chart ceiling, leaves label room
  const floor = 0;
  return PAD_T + (H - PAD_T - PAD_B) * (1 - (v - floor) / (top - floor));
}

function x(i: number, n: number): number {
  const span = W - PAD_L - PAD_R;
  return n <= 1 ? PAD_L + span / 2 : PAD_L + (span * i) / (n - 1);
}

function pts(series: number[], n: number): string {
  return series.map((v, i) => `${x(i, n).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
}

function missing(paths: string): number[] | null {
  console.warn(`[evidence] artifact field missing: ${paths}`);
  return null;
}

function Panel({
  title,
  sub,
  real,
  synthetic,
  generations,
  artifactPath,
  dx,
}: {
  title: string;
  sub: string;
  real: number[] | null;
  synthetic: number[] | null;
  generations: string[];
  artifactPath: string;
  dx: number;
}) {
  const n = generations.length;
  const r = real && real.length === n ? real : null;
  const s = synthetic && synthetic.length === n ? synthetic : null;
  if (!r || !s) missing(artifactPath);

  // On-plot delta annotations: first→last of each series.
  const realDelta = r && r.length >= 2 ? r[r.length - 1] - r[0] : null;
  const synDelta = s && s.length >= 2 ? s[s.length - 1] - s[0] : null;

  return (
    <g transform={`translate(${dx},0)`}>
      <text x={W / 2} y={16} textAnchor="middle" className="type-ui" fontSize="13" fontWeight="700" fill="var(--text)">
        {title}
      </text>
      <text x={W / 2} y={32} textAnchor="middle" className="type-num" fontSize="10" fill="var(--text-dim)">
        {sub}
      </text>

      {/* gridlines at 1.0 and 0.5 */}
      {[1.0, 0.5].map((g) => (
        <g key={g}>
          <line x1={PAD_L} x2={W - PAD_R} y1={y(g)} y2={y(g)} stroke="var(--border)" strokeWidth="1" />
          <text x={PAD_L - 8} y={y(g) + 3.5} textAnchor="end" className="type-num" fontSize="10" fill="var(--text-faint)">
            {g.toFixed(1)}
          </text>
        </g>
      ))}

      {/* x-axis labels */}
      {generations.map((g, i) => (
        <text key={g} x={x(i, n)} y={H - 14} textAnchor="middle" className="type-num" fontSize="10.5" fill="var(--text-dim)">
          {g}
        </text>
      ))}

      {/* REAL line — defender hue; the metric that matters */}
      {r && (
        <g>
          <polyline points={pts(r, n)} fill="none" stroke="var(--blue)" strokeWidth="2.5" />
          {r.map((v, i) => (
            <circle key={i} cx={x(i, n)} cy={y(v)} r="3" fill="var(--blue)" />
          ))}
          <text
            x={x(0, n)}
            y={y(r[0]) - 10}
            textAnchor="middle"
            className="type-num"
            fontSize="10.5"
            fontWeight="700"
            fill="var(--blue)"
          >
            real fraud
          </text>
        </g>
      )}

      {/* SYNTHETIC line — attacker hue; the vanity metric */}
      {s && (
        <g>
          <polyline points={pts(s, n)} fill="none" stroke="var(--red)" strokeWidth="2.5" />
          {s.map((v, i) => (
            <circle key={i} cx={x(i, n)} cy={y(v)} r="3" fill="var(--red)" />
          ))}
          <text
            x={x(n - 1, n)}
            y={y(s[s.length - 1]) - 10}
            textAnchor="middle"
            className="type-num"
            fontSize="10.5"
            fontWeight="700"
            fill="var(--red)"
          >
            its own synthetic
          </text>
        </g>
      )}

      {/* on-plot deltas — the divergence, stated in points */}
      {realDelta !== null && (
        <text
          x={W / 2}
          y={H - 44}
          textAnchor="middle"
          className="type-num"
          fontSize="10.5"
          fontWeight="700"
          fill="var(--blue)"
        >
          real {realDelta >= 0 ? "+" : "−"}
          {Math.abs(realDelta * 100).toFixed(1)} pts
        </text>
      )}
      {synDelta !== null && (
        <text
          x={W / 2}
          y={H - 28}
          textAnchor="middle"
          className="type-num"
          fontSize="10.5"
          fontWeight="700"
          fill="var(--red)"
        >
          synthetic {synDelta >= 0 ? "+" : "−"}
          {Math.abs(synDelta * 100).toFixed(1)} pts
        </text>
      )}

      {/* panel frame */}
      <rect x={PAD_L} y={PAD_T} width={W - PAD_L - PAD_R} height={H - PAD_T - PAD_B} fill="none" stroke="var(--border)" strokeWidth="1" />
    </g>
  );
}

export function Scissor({ generations, ungatedReal, ungatedSynthetic, gatedReal, gatedSynthetic, artifactPath }: ScissorProps) {
  return (
    <figure
      aria-label="Two panels. Ungated loop: recall on the loop's own synthetic attacks rises while recall on held-out real fraud falls — the metrics diverge. Gated loop: the fidelity gate rejects low-fidelity batches and both metrics stay together."
    >
      <svg viewBox={`0 0 ${W * 2} ${H}`} role="img" className="w-full">
        <Panel
          dx={0}
          title="UNGATED"
          sub="low-fidelity generator · no gate"
          real={ungatedReal}
          synthetic={ungatedSynthetic}
          generations={generations}
          artifactPath={artifactPath}
        />
        <Panel
          dx={W}
          title="GATED"
          sub="fidelity gate rejects low-fidelity batches"
          real={gatedReal}
          synthetic={gatedSynthetic}
          generations={generations}
          artifactPath={artifactPath}
        />
      </svg>
      <figcaption className="type-ui measure mt-3 text-xs text-text-dim">
        The vanity metric and the real metric move in opposite directions when the loop is ungated; the fidelity gate
        removes the divergence.
      </figcaption>
    </figure>
  );
}
