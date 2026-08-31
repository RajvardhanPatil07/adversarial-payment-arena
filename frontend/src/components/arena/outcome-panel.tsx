"use client";

import { CircleCheck, Fingerprint, ShieldCheck, TriangleAlert } from "lucide-react";
import Link from "next/link";

import type { CampaignSummaryData, CostUpdate } from "@/lib/arena-types";

function usd(value: number): string {
  const sign = value < 0 ? "−" : "";
  return `${sign}$${Math.abs(value).toLocaleString("en-US", {
    maximumFractionDigits: 0,
  })}`;
}

export function OutcomePanel({
  costs,
  summary,
  decisions,
  running,
}: {
  costs: CostUpdate | null;
  summary: CampaignSummaryData | null;
  decisions: Array<{ decision: string; scores: { ring_detected: boolean; ring_risk: number } }>;
  running: boolean;
}) {
  const blocked = decisions.filter((decision) => decision.decision === "DECLINE").length;
  const ringHits = decisions.filter((decision) => decision.scores.ring_detected).length;
  const peakRingRisk = decisions.reduce((peak, decision) => Math.max(peak, decision.scores.ring_risk), 0);
  const finished = Boolean(summary);
  const status = finished ? "Campaign contained" : ringHits > 0 ? "Coordinated ring exposed" : running ? "Tracing the attack" : "Defense standing by";
  const positive = (costs?.net_savings ?? 0) >= 0;

  return (
    <section
      aria-label="Defense outcome"
      className={`outcome-panel ${finished ? "outcome-panel--contained" : ringHits > 0 ? "outcome-panel--hot" : ""}`}
    >
      <div className="flex min-w-0 items-center gap-3">
        <div className={`outcome-icon ${finished ? "outcome-icon--contained" : ringHits > 0 ? "outcome-icon--active" : ""}`}>
          {finished ? <CircleCheck aria-hidden="true" /> : ringHits > 0 ? <ShieldCheck aria-hidden="true" /> : <Fingerprint aria-hidden="true" />}
        </div>
        <div className="min-w-0">
          <p className="text-xs font-medium text-zinc-400">Defense outcome</p>
          <h2 className="truncate text-base font-semibold tracking-[-0.02em] text-zinc-50 sm:text-lg">
            {status}
          </h2>
        </div>
      </div>

      <div className="outcome-metrics">
        <div>
          <span>Ring signals</span>
          <strong key={ringHits} className={`metric-value ${ringHits > 0 && !finished ? "text-red-300" : "text-zinc-100"}`}>{ringHits}</strong>
        </div>
        <div>
          <span>Payments blocked</span>
          <strong key={blocked} className="metric-value">{blocked}</strong>
        </div>
        <div>
          <span>Peak ring risk</span>
          <strong key={peakRingRisk} className="metric-value">{Math.round(peakRingRisk * 100)}%</strong>
        </div>
        <div>
          <span>Net effect</span>
          <strong key={costs?.net_savings ?? 0} className={`metric-value ${positive ? "text-emerald-300" : "text-red-300"}`}>
            {usd(costs?.net_savings ?? 0)}
          </strong>
        </div>
      </div>

      {finished ? (
        <Link href="/evidence" className="outcome-link">
          Verify the result <span aria-hidden="true">→</span>
        </Link>
      ) : running ? (
        <span className="outcome-live"><i /> Live evidence</span>
      ) : (
        <span className="outcome-ready"><TriangleAlert aria-hidden="true" /> Awaiting campaign</span>
      )}
    </section>
  );
}
