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
  APPROVE: "border-pass/60 bg-pass/10 text-pass",
  STEP_UP: "border-warn/60 bg-warn/10 text-warn",
  DECLINE: "border-fail/60 bg-fail/10 text-fail",
  MANUAL_REVIEW: "border-blue/60 bg-blue/10 text-blue",
};

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

export function DefenseFeed({ rows }: { rows: DecisionRow[] }) {
  return (
    <Card className="flex min-h-0 flex-1 flex-col border-border bg-surface-1">
      <CardHeader className="pb-2">
        <CardTitle className="type-ui text-sm">
          Defense Decisions
          <span className="type-ui ml-2 text-[10px] font-normal text-text-dim">
            velocity · iForest · graph
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 overflow-y-auto">
        {rows.length === 0 ? (
          <p className="type-ui pt-10 text-center text-xs text-text-dim">
            Decisions render here in real time.
          </p>
        ) : (
          <Table>
            <TableHeader className="sticky top-0 bg-surface-1">
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="type-ui h-8 text-[10px] text-text-dim">TXN</TableHead>
                <TableHead className="type-ui h-8 text-[10px] text-text-dim">DECISION</TableHead>
                <TableHead className="type-ui h-8 text-[10px] text-text-dim">VEL</TableHead>
                <TableHead className="type-ui h-8 text-[10px] text-text-dim">ANOM</TableHead>
                <TableHead className="type-ui h-8 text-[10px] text-text-dim">RING</TableHead>
                <TableHead className="type-ui h-8 text-[10px] text-text-dim">WHY</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.slice().reverse().map((r) => (
                <TableRow key={r.id} className="border-border">
                  <TableCell className="type-num py-1.5 text-[10.5px] text-text-dim">
                    {r.txn ?? "—"}
                  </TableCell>
                  <TableCell className="py-1.5">
                    <span
                      className={`type-num inline-block rounded-[var(--r-sm)] border px-1.5 py-0.5 text-[9.5px] font-bold ${DECISION_STYLE[r.decision]}`}
                    >
                      {r.decision}
                    </span>
                  </TableCell>
                  <TableCell className="type-num py-1.5 text-[10.5px]">
                    {r.scores.velocity.toFixed(2)}
                  </TableCell>
                  <TableCell className="type-num py-1.5 text-[10.5px]">
                    {r.scores.is_anomaly ? (
                      <span className="text-pass">{pct(r.scores.novelty_anomaly)}</span>
                    ) : (
                      <span className="text-text-dim">{r.scores.novelty_anomaly.toFixed(2)}</span>
                    )}
                  </TableCell>
                  <TableCell className="type-num py-1.5 text-[10.5px]">
                    {r.scores.ring_detected ? (
                      <Badge variant="destructive" className="px-1 py-0 text-[9px]">RING</Badge>
                    ) : (
                      <span className="text-text-dim">{r.scores.ring_risk.toFixed(2)}</span>
                    )}
                  </TableCell>
                  <TableCell className="type-num max-w-[150px] truncate py-1.5 text-[9.5px] text-text-dim">
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
