"use client";

/**
 * DefenseFeed — the "Defend" phase, visible.
 * Every accepted payload's final issuer decision with the three layer
 * scores that produced it. DECLINEs and ring hits get loud colors.
 */

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Decision, DecisionScores } from "@/lib/arena-types";

export interface DecisionRow {
  id: number;
  txn?: number | null;
  decision: Decision;
  reasons: string[];
  scores: DecisionScores;
  amount: number;
  payload?: { merchant_id: string; customer_id: string; amount: number };
}

const DECISION_STYLE: Record<Decision, string> = {
  APPROVE: "border-emerald-800 bg-emerald-500/10 text-emerald-400",
  STEP_UP: "border-amber-700 bg-amber-500/10 text-amber-400",
  DECLINE: "border-red-800 bg-red-500/10 text-red-400",
  MANUAL_REVIEW: "border-violet-800 bg-violet-500/10 text-violet-400",
};

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

export function DefenseFeed({ rows }: { rows: DecisionRow[] }) {
  return (
    <Card className="flex min-h-0 flex-1 flex-col border-zinc-800 bg-zinc-950/60">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">
          Defense Decisions
          <span className="ml-2 text-[10px] font-normal text-muted-foreground">
            behavioral · anomaly · network
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 overflow-y-auto">
        {rows.length === 0 ? (
          <p className="pt-10 text-center text-xs text-muted-foreground">
            Decisions render here in real time.
          </p>
        ) : (
          <Table>
            <TableHeader className="sticky top-0 bg-zinc-950">
              <TableRow className="border-zinc-800 hover:bg-transparent">
                <TableHead className="h-8 text-[10px] text-muted-foreground">TXN</TableHead>
                <TableHead className="h-8 text-[10px] text-muted-foreground">DECISION</TableHead>
                <TableHead className="h-8 text-[10px] text-muted-foreground" title="Behavioral velocity score">VELOCITY</TableHead>
                <TableHead className="h-8 text-[10px] text-muted-foreground" title="Novelty and anomaly score">ANOMALY</TableHead>
                <TableHead className="h-8 text-[10px] text-muted-foreground" title="Coordinated entity-ring risk">NETWORK</TableHead>
                <TableHead className="h-8 text-[10px] text-muted-foreground">WHY</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.slice().reverse().map((r) => (
                <TableRow key={r.id} className="defense-row-enter border-zinc-900">
                  <TableCell className="py-2 font-mono text-[11px] text-muted-foreground">
                    {r.txn ?? "—"}
                  </TableCell>
                  <TableCell className="py-2">
                    <span
                      className={`inline-block rounded border px-1.5 py-0.5 font-mono text-[9.5px] font-bold ${DECISION_STYLE[r.decision]}`}
                    >
                      {r.decision}
                    </span>
                  </TableCell>
                  <TableCell className="py-2 font-mono text-[11px]">
                    {r.scores.velocity.toFixed(2)}
                  </TableCell>
                  <TableCell className="py-2 font-mono text-[11px]">
                    {r.scores.is_anomaly ? (
                      <span className="text-emerald-400">{pct(r.scores.novelty_anomaly)}</span>
                    ) : (
                      <span className="text-muted-foreground">{r.scores.novelty_anomaly.toFixed(2)}</span>
                    )}
                  </TableCell>
                  <TableCell className="py-2 font-mono text-[11px]">
                    {r.scores.ring_detected ? (
                      <Badge variant="destructive" className="px-1 py-0 text-[9px]">RING</Badge>
                    ) : (
                      <span className="text-muted-foreground">{r.scores.ring_risk.toFixed(2)}</span>
                    )}
                  </TableCell>
                  <TableCell className="max-w-[150px] truncate py-2 text-[10px] text-muted-foreground" title={r.reasons.join(", ")}>
                    {r.reasons.length > 0 ? r.reasons.join(", ") : "clean"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
