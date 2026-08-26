"use client";

/**
 * AttackerFeed — the "Generate" phase, visible.
 * Planner/operator reasoning, generated payloads, and Plausibility Gate
 * verdicts (the gate lighting up red is the point of this whole demo).
 */

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { PaymentPayload } from "@/lib/arena-types";

export interface ThoughtRow {
  id: number;
  role: "PLANNER" | "OPERATOR";
  text: string;
  txn?: number;
}

export interface PayloadRow {
  id: number;
  payload: PaymentPayload;
  txn?: number;
}

export interface CheckRow {
  id: number;
  ok: boolean;
  reason: string;
  attempt?: number;
  txn?: number;
}

const REASON_STYLE: Record<string, "destructive" | "secondary" | "outline"> = {
  ok: "secondary",
  economic_infeasible: "destructive",
  metadata_incoherent: "destructive",
  rail_infeasible: "destructive",
};

function money(n: number): string {
  return `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function PayloadCard({ row }: { row: PayloadRow }) {
  const p = row.payload;
  return (
    <div className="rounded-md border border-l-2 border-l-amber-500 bg-amber-500/5 p-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-amber-400">{p.merchant_id.replace("MERCH_", "")}</span>
        <span className="text-foreground">{money(p.amount)}</span>
      </div>
      <div className="mt-0.5 flex flex-wrap gap-x-3">
        <span>{p.customer_id}</span>
        <span>{p.pos_entry_mode}</span>
        <span>3DS:{p["3ds_status"]}</span>
        <span>mcc {p.mcc}</span>
        {p.stolen_resource ? (
          <span className="text-red-400">⚠ {p.stolen_resource}</span>
        ) : null}
      </div>
    </div>
  );
}

export function AttackerFeed({
  thoughts,
  payloads,
  checks,
}: {
  thoughts: ThoughtRow[];
  payloads: PayloadRow[];
  checks: CheckRow[];
}) {
  return (
    <Card className="flex min-h-0 flex-1 flex-col border-zinc-800 bg-zinc-950/60">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          Attacker Stream
          <Badge variant="outline" className="font-mono text-[9px] text-red-400 border-red-900">
            stealth/ox-alpha
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 overflow-y-auto pr-1">
        <Tabs defaultValue="stream" className="flex h-full flex-col gap-2">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="stream" className="text-xs">Stream</TabsTrigger>
            <TabsTrigger value="gate" className="text-xs">
              Gate Verdicts
              {checks.length > 0 && (
                <span className="ml-1 text-[9px] text-red-400">
                  {checks.filter((c) => !c.ok).length}✗
                </span>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="stream" className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
            {thoughts.length === 0 && payloads.length === 0 && (
              <p className="pt-8 text-center text-xs text-muted-foreground">
                No campaign running. Pick an attack and press LAUNCH.
              </p>
            )}
            {thoughts.map((t) => (
              <div
                key={`t-${t.id}`}
                className={`rounded-md border p-2 text-[11px] leading-snug ${
                  t.role === "PLANNER"
                    ? "border-violet-900 bg-violet-500/10 text-violet-200"
                    : "border-zinc-800 bg-zinc-900/60 text-zinc-300"
                }`}
              >
                <span
                  className={`mr-1.5 font-mono text-[9px] font-bold ${
                    t.role === "PLANNER" ? "text-violet-400" : "text-red-400"
                  }`}
                >
                  {t.role}
                  {t.txn ? ` #${t.txn}` : ""}
                </span>
                {t.text}
              </div>
            )).reverse()}
            {payloads.map((p) => (
              <PayloadCard key={`p-${p.id}`} row={p} />
            )).reverse()}
          </TabsContent>

          <TabsContent value="gate" className="min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-1">
            {checks.length === 0 && (
              <p className="pt-8 text-center text-xs text-muted-foreground">
                Gate verdicts will appear here.
              </p>
            )}
            {checks.slice().reverse().map((c) => (
              <div
                key={`c-${c.id}`}
                className={`flex items-center justify-between rounded border px-2 py-1.5 font-mono text-[10.5px] ${
                  c.ok ? "border-emerald-900 bg-emerald-500/5" : "border-red-900 bg-red-500/10"
                }`}
              >
                <span className="text-muted-foreground">
                  txn {c.txn ?? "—"} · attempt {c.attempt ?? "—"}
                </span>
                <Badge variant={REASON_STYLE[c.reason] ?? "outline"} className="text-[9px]">
                  {c.ok ? "PASS" : c.reason}
                </Badge>
              </div>
            ))}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
