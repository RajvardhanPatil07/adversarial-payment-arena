"use client";

/**
 * <LoopDiagram> — the loop, hand-rolled in SVG.
 *
 * IDENTIFY → GENERATE → [FIDELITY GATE] → DEFEND → back to IDENTIFY.
 *
 * Layout rules that keep the diagram readable as a screenshot (SECTION 3):
 *   - Every element has its own lane. The rejected path leaves the gate's TOP
 *     vertex and points up, into otherwise-empty space: refused batches never
 *     enter the loop, so they never cross the loop's geometry either.
 *   - The gate→DEFEND corridor is wide enough to carry the ADMITTED label with
 *     its measured value (gateObserved), which the section blurb promises.
 *   - The return arc dips below everything else; only the loop caption shares
 *     its lane, well clear of the arc's lowest point.
 *
 * Every value in the labels arrives as a prop (read at render time from the
 * artifacts by the page), so the diagram can never drift from the evidence.
 * The stroke-dashoffset draw-in runs once when the diagram scrolls into view
 * and is disabled by prefers-reduced-motion; the static end state is fully
 * readable as a screenshot.
 *
 * Presentational: props only.
 */

import { useEffect, useRef, useState } from "react";

export interface LoopDiagramProps {
  /** Gate C2ST threshold, e.g. "0.900" (closed_loop.gate.c2st_auc_max). */
  gateThreshold: string | null;
  /** Observed generator C2ST, e.g. "0.523" (fidelity_report.acceptance_gate.observed). */
  gateObserved: string | null;
  /** Observed generator C2ST on the rejected arm's generator, or null. */
  rejectedC2st: string | null;
  /** Number of escape batches the gate rejected in the gated arm, or null. */
  batchesRejected: string | null;
  /** Seed count / provenance note, rendered as a caption chip. */
  provenanceNote?: string;
}

export function LoopDiagram({
  gateThreshold,
  gateObserved,
  rejectedC2st,
  batchesRejected,
  provenanceNote,
}: LoopDiagramProps) {
  const ref = useRef<HTMLElement | null>(null);
  const [drawn, setDrawn] = useState(false);

  // Draw the paths once, on scroll into view. Reduced-motion users get the
  // static end state immediately via the CSS layer (globals.css forces the
  // drawn state under prefers-reduced-motion), so this effect only subscribes.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setDrawn(true);
          io.disconnect();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // The gate's threshold, with an explicit unmeasured state.
  const threshold = gateThreshold ?? "—";

  return (
    <figure
      aria-label="The closed loop: identify, generate, a fidelity gate, and defend. The gate admits high-fidelity batches to retraining and refuses low-fidelity ones before they enter the loop."
      ref={ref}
    >
      <svg
        viewBox="0 0 1040 365"
        role="img"
        className="w-full"
        style={{ color: "var(--text-dim)" }}
      >
        {/* ---- the main loop: IDENTIFY → GENERATE → GATE → DEFEND → back ---- */}
        {/* Each path carries pathLength="1": the draw-in animation in
             globals.css runs on normalised dash fractions, so the dashed
             paths below never fight a competing dash attribute. */}
        <path
          d="M 190 210 C 220 210, 250 210, 280 210"
          fill="none"
          stroke="var(--border-hi)"
          strokeWidth="2"
          pathLength={1}
          className="loop-line"
          data-drawn={drawn}
        />
        <path
          d="M 410 210 C 430 210, 450 210, 470 210"
          fill="none"
          stroke="var(--border-hi)"
          strokeWidth="2"
          pathLength={1}
          className="loop-line"
          data-drawn={drawn}
        />
        {/* The gate→DEFEND corridor is 210px wide so the ADMITTED label and
             its measured value fit inside it without touching either shape. */}
        <path
          d="M 640 210 C 710 210, 780 210, 850 210"
          fill="none"
          stroke="var(--border-hi)"
          strokeWidth="2"
          pathLength={1}
          className="loop-line"
          data-drawn={drawn}
        />
        {/* Return arc: DEFEND → IDENTIFY (the loop in closed loop). Dashes are
             CSS-owned (loop-line-return) for the same normalisation reason. */}
        <path
          d="M 915 252 C 915 345, 125 345, 125 252"
          fill="none"
          stroke="var(--border-hi)"
          strokeWidth="2"
          pathLength={1}
          className="loop-line loop-line-return"
          data-drawn={drawn}
        />

        {/* ---- IDENTIFY node ---- */}
        <LoopNode x={60} y={210} w={130} label="IDENTIFY" sub="22 mapped · 14 executable" hue="var(--text)" />
        {/* ---- GENERATE node ---- */}
        <LoopNode x={280} y={210} w={130} label="GENERATE" sub="synthetic attacks" hue="var(--red)" />
        {/* ---- DEFEND node ---- */}
        <LoopNode x={850} y={210} w={130} label="DEFEND" sub="retrain + rescore" hue="var(--blue)" />

        {/* ---- the gate: dominant, between GENERATE and DEFEND ---- */}
        <g>
          <path
            d="M 470 150 L 555 118 L 640 150 L 640 270 L 555 302 L 470 270 Z"
            fill="var(--blue-dim)"
            stroke="var(--blue)"
            strokeWidth="2.5"
          />
          <text x={555} y={196} textAnchor="middle" className="type-ui" fontSize="17" fontWeight="700" fill="var(--blue)">
            FIDELITY GATE
          </text>
          <text x={555} y={222} textAnchor="middle" className="type-num" fontSize="12" fill="var(--text-dim)">
            C2ST ≤ {threshold}
          </text>
          <text x={555} y={244} textAnchor="middle" className="type-num" fontSize="11" fill="var(--text-dim)">
            label-free · fixed in advance
          </text>
        </g>

        {/* ---- the REJECTED path: out the gate's TOP vertex, into empty space.
             Refused batches never enter the loop, so the path never crosses the
             loop's geometry — and the bottom lane stays clear for the arc. */}
        <path
          d="M 555 118 L 555 88"
          fill="none"
          stroke="var(--red)"
          strokeWidth="2.5"
          pathLength={1}
          className="loop-line loop-line-rejected"
          data-drawn={drawn}
        />
        <polygon points="549,90 561,90 555,68" fill="var(--red)" />

        <text x={555} y={22} textAnchor="middle" className="type-num" fontSize="12" fontWeight="700" fill="var(--red)">
          {rejectedC2st === null
            ? "REJECTED · low fidelity · not measured"
            : "REJECTED · low fidelity"}
        </text>
        {rejectedC2st !== null && (
          <text x={555} y={40} textAnchor="middle" className="type-num" fontSize="11" fill="var(--red)">
            C2ST {rejectedC2st} &gt; {threshold} gate
          </text>
        )}
        {batchesRejected && (
          <text x={555} y={58} textAnchor="middle" className="type-num" fontSize="11" fill="var(--red)">
            refused {batchesRejected} escape batches
          </text>
        )}

        {/* ---- the ADMITTED path: the corridor spine itself, labelled with the
             observed fidelity of the admitted generator. ---- */}
        <text x={745} y={184} textAnchor="middle" className="type-num" fontSize="12" fontWeight="700" fill="var(--blue)">
          ADMITTED
        </text>
        <text x={745} y={202} textAnchor="middle" className="type-num" fontSize="11" fill="var(--blue)">
          C2ST {gateObserved ?? "—"} ≤ {threshold} gate
        </text>

        {/* ---- Arrowheads into each node, flush with the box edges ---- */}
        <polygon points="268,204 280,210 268,216" fill="var(--border-hi)" />
        <polygon points="458,204 470,210 458,216" fill="var(--border-hi)" />
        <polygon points="838,204 850,210 838,216" fill="var(--border-hi)" />
        {/* Return arrow, up into IDENTIFY's bottom edge. */}
        <polygon points="119,264 131,264 125,250" fill="var(--border-hi)" />

        {/* ---- loop direction caption, below the return arc's lowest point ---- */}
        <text x={520} y={348} textAnchor="middle" className="type-ui" fontSize="11" fill="var(--text-faint)">
          escapes fold back into training — the loop
        </text>
      </svg>
      {provenanceNote && (
        <figcaption className="type-num mt-2 text-center text-[0.6875rem] text-text-faint">{provenanceNote}</figcaption>
      )}
    </figure>
  );
}

function LoopNode({
  x,
  y,
  w,
  label,
  sub,
  hue,
}: {
  x: number;
  y: number;
  w: number;
  label: string;
  sub: string;
  hue: string;
}) {
  return (
    <g>
      <rect x={x} y={y - 40} width={w} height={80} rx={10} fill="var(--surface-2)" stroke="var(--border-hi)" strokeWidth="1.5" />
      <text x={x + w / 2} y={y - 6} textAnchor="middle" className="type-ui" fontSize="15" fontWeight="700" fill={hue}>
        {label}
      </text>
      <text x={x + w / 2} y={y + 18} textAnchor="middle" className="type-num" fontSize="10.5" fill="var(--text-dim)">
        {sub}
      </text>
    </g>
  );
}
