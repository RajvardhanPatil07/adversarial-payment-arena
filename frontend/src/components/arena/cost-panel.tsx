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

/** Tile-scale money: cents do not fit a 99px tile, and OutcomePanel rounds the same way. */
function usdWhole(n: number): string {
  const sign = n < 0 ? "−" : "";
  return `${sign}$${Math.abs(n).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

const TILE_VALUE = "overflow-hidden text-nowrap font-mono text-sm font-bold tabular-nums";
const TILE_META = "font-mono text-[9px] leading-snug text-muted-foreground tabular-nums";

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
    <Card className="border-zinc-800 bg-zinc-950/60">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-sm">
          Business impact
          <span className="text-[9.5px] font-normal text-muted-foreground">
            customer friction priced in
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="min-w-0 rounded border border-amber-900/60 bg-amber-500/5 p-1.5">
            <div className={`${TILE_VALUE} text-amber-400`}>{c.fp_cost_bps.toFixed(1)}</div>
            <div className="text-[9px] uppercase tracking-wide text-muted-foreground">Friction cost</div>
            <div className={TILE_META}>
              <div>{usdWhole(c.fp_cost_usd)}</div>
              <div>{c.counts.false_positives} flagged</div>
            </div>
          </div>
          <div className="min-w-0 rounded border border-red-900/60 bg-red-500/5 p-1.5">
            <div className={`${TILE_VALUE} text-red-400`}>{usdWhole(c.fn_loss)}</div>
            <div className="text-[9px] uppercase tracking-wide text-muted-foreground">Fraud missed</div>
            <div className={TILE_META}>{c.counts.false_negatives} missed</div>
          </div>
          <div className="min-w-0 rounded border border-emerald-900/60 bg-emerald-500/5 p-1.5">
            <div className={`${TILE_VALUE} text-emerald-400`}>{usdWhole(c.tp_saved)}</div>
            <div className="text-[9px] uppercase tracking-wide text-muted-foreground">Fraud blocked</div>
            <div className={TILE_META}>{c.counts.true_positives_declined} blocked</div>
          </div>
        </div>
        <Separator className="bg-zinc-800" />
        <div className="flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Net savings</span>
          <span
            className={`font-mono text-xl font-bold ${positive ? "text-emerald-400" : "text-red-400"}`}
          >
            {usd(c.net_savings)}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
