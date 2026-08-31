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
  role: "PLANNER" | "OPERATOR" | "SYSTEM";
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
    <div className="type-num rounded-[var(--r-md)] border border-l-2 border-l-red bg-red/5 p-2 text-[10.5px] leading-relaxed text-text-dim">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-red">{p.merchant_id.replace("MERCH_", "")}</span>
        <span className="text-text">{money(p.amount)}</span>
      </div>
      <div className="mt-0.5 flex flex-wrap gap-x-3">
        <span>{p.customer_id}</span>
        <span>{p.pos_entry_mode}</span>
        <span>3DS:{p["3ds_status"]}</span>
        <span>mcc {p.mcc}</span>
        {p.stolen_resource ? (
          <span className="text-fail">⚠ {p.stolen_resource}</span>
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
    <Card className="flex min-h-0 flex-1 flex-col border-border bg-surface-1">
      <CardHeader className="pb-2">
        <CardTitle className="type-ui flex items-center gap-2 text-sm">
          Attacker Stream
          <Badge variant="outline" className="type-ui text-[9px] text-red border-red/50">
            LLM red team
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 overflow-y-auto pr-1">
        <Tabs defaultValue="stream" className="flex h-full flex-col gap-2">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="stream" className="type-ui text-xs">Stream</TabsTrigger>
            <TabsTrigger value="gate" className="type-ui text-xs">
              Gate Verdicts
              {checks.length > 0 && (
                <span className="type-num ml-1 text-[9px] text-red">
                  {checks.filter((c) => !c.ok).length}✗
                </span>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="stream" className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
            {thoughts.length === 0 && payloads.length === 0 && (
              <p className="type-ui pt-8 text-center text-xs text-text-dim">
                No campaign running. Pick an attack and press LAUNCH.
              </p>
            )}
            {thoughts.map((t) => (
              <div
                key={`t-${t.id}`}
                className={`type-ui rounded-[var(--r-md)] border p-2 text-[11px] leading-snug ${
                  t.role === "PLANNER"
                    ? "border-red/50 bg-red-dim text-text"
                    : t.role === "SYSTEM"
                      ? "border-warn/50 bg-warn/10 text-text"
                      : "border-border bg-surface-2 text-text-dim"
                }`}
              >
                <span
                  className={`type-num mr-1.5 text-[9px] font-bold ${
                    t.role === "PLANNER"
                      ? "text-red"
                      : t.role === "SYSTEM"
                        ? "text-warn"
                        : "text-red"
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
              <p className="type-ui pt-8 text-center text-xs text-text-dim">
                Gate verdicts will appear here.
              </p>
            )}
            {checks.slice().reverse().map((c) => (
              <div
                key={`c-${c.id}`}
                className={`type-num flex items-center justify-between rounded-[var(--r-sm)] border px-2 py-1.5 text-[10.5px] ${
                  c.ok ? "border-blue/50 bg-blue-dim" : "border-red/50 bg-red-dim"
                }`}
              >
                <span className="text-text-dim">
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
