"use client";

/**
 * CostPanel — the running cost matrix in bps, the business language of the
 * arena. Fed by cost_update events (which include ambient legit traffic,
 * so false positives get priced, not hidden).
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { CostUpdate } from "@/lib/arena-types";

function usd(n: number): string {
  const sign = n < 0 ? "−" : "";
  return `${sign}$${Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function CostPanel({ costs }: { costs: CostUpdate | null }) {
  const c = costs ?? {
    fp_cost_bps: 0,
    fp_cost_usd: 0,
    fn_loss: 0,
    tp_saved: 0,
    net_savings: 0,
    counts: { false_positives: 0, false_negatives: 0, true_positives_declined: 0 },
  };
  const positive = c.net_savings >= 0;

  return (
    <Card className="border-border bg-surface-1">
      <CardHeader className="pb-2">
        <CardTitle className="type-ui flex items-center justify-between text-sm">
          Cost Matrix
          <span className="type-num text-[9.5px] font-normal text-text-dim">
            FP 15bps · FN 100% · TP saves
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-[var(--r-sm)] border border-warn/50 bg-warn/5 p-1.5">
            <div className="type-num text-base font-bold text-warn">{c.fp_cost_bps.toFixed(1)}</div>
            <div className="type-ui text-[9px] uppercase tracking-wide text-text-dim">FP cost (bps)</div>
            <div className="type-num text-[9px] text-text-dim">
              {usd(c.fp_cost_usd)} · {c.counts.false_positives} flagged
            </div>
          </div>
          <div className="rounded-[var(--r-sm)] border border-fail/50 bg-fail/5 p-1.5">
            <div className="type-num text-base font-bold text-fail">{usd(c.fn_loss)}</div>
            <div className="type-ui text-[9px] uppercase tracking-wide text-text-dim">FN loss</div>
            <div className="type-num text-[9px] text-text-dim">{c.counts.false_negatives} missed</div>
          </div>
          <div className="rounded-[var(--r-sm)] border border-pass/50 bg-pass/5 p-1.5">
            <div className="type-num text-base font-bold text-pass">{usd(c.tp_saved)}</div>
            <div className="type-ui text-[9px] uppercase tracking-wide text-text-dim">TP saved</div>
            <div className="type-num text-[9px] text-text-dim">
              {c.counts.true_positives_declined} blocked
            </div>
          </div>
        </div>
        <Separator className="bg-border" />
        <div className="flex items-baseline justify-between">
          <span className="type-ui text-[10px] uppercase tracking-wide text-text-dim">Net savings</span>
          <span
            className={`type-num text-xl font-bold ${positive ? "text-pass" : "text-fail"}`}
          >
            {usd(c.net_savings)}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
