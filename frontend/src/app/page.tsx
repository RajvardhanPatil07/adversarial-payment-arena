"use client";

/**
 * The Adversarial Payment Arena — SOC dashboard (single page).
 *
 * Left:   the attacker's mind (planner/operator thoughts, payloads, gate verdicts)
 *         + the running cost matrix in bps.
 * Center: the entity graph lighting up as rings form + AI field notes.
 * Right:  the defense stack's decisions with per-layer scores.
 *
 * All live data arrives over one WebSocket (src/lib/ws.ts); this component
 * owns the connection lifecycle and folds events into reducer state.
 */

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { toast } from "sonner";

import { AnalystPanel, type AnalystStats } from "@/components/arena/analyst-panel";
import { AttackerFeed, type CheckRow, type PayloadRow, type ThoughtRow } from "@/components/arena/attacker-feed";
import { ControlBar } from "@/components/arena/control-bar";
import { CostPanel } from "@/components/arena/cost-panel";
import { DefenseFeed, type DecisionRow } from "@/components/arena/defense-feed";
import { EntityGraphCanvas } from "@/components/arena/graph-canvas";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ArenaEvent, CampaignSummaryData, CostUpdate, GraphEdge, GraphNode } from "@/lib/arena-types";
import { ArenaSocket, backendWsUrl, type ConnState } from "@/lib/ws";

// --------------------------------------------------------------------------- //
// State
// --------------------------------------------------------------------------- //

interface ArenaState {
  thoughts: ThoughtRow[];
  payloads: PayloadRow[];
  checks: CheckRow[];
  decisions: DecisionRow[];
  costs: CostUpdate | null;
  summary: CampaignSummaryData | null;
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  running: boolean;
}

const initialState: ArenaState = {
  thoughts: [],
  payloads: [],
  checks: [],
  decisions: [],
  costs: null,
  summary: null,
  graphNodes: [],
  graphEdges: [],
  running: false,
};

const CAPS = { thoughts: 120, payloads: 60, checks: 150, decisions: 150, graphNodes: 160, graphEdges: 220 };

type Action = { type: "reset" } | { type: "event"; event: ArenaEvent };

function capped<T>(arr: T[], cap: number): T[] {
  return arr.length > cap ? arr.slice(arr.length - cap) : arr;
}

function reducer(state: ArenaState, action: Action): ArenaState {
  switch (action.type) {
    case "reset":
      return {
        ...state,
        thoughts: [],
        payloads: [],
        checks: [],
        decisions: [],
        summary: null,
        running: true,
      };
    case "event": {
      const e = action.event;
      switch (e.type) {
        case "agent_thought":
          return {
            ...state,
            thoughts: capped(
              [...state.thoughts, { id: Date.now() + Math.random(), role: e.role, text: e.data, txn: e.txn_index }],
              CAPS.thoughts,
            ),
          };
        case "payload_generated":
          return {
            ...state,
            payloads: capped(
              [...state.payloads, { id: Date.now() + Math.random(), payload: e.data, txn: e.txn_index }],
              CAPS.payloads,
            ),
          };
        case "plausibility_check":
          return {
            ...state,
            checks: capped(
              [
                ...state.checks,
                {
                  id: Date.now() + Math.random(),
                  ok: e.data.accepted,
                  reason: e.data.reason,
                  attempt: e.data.attempt,
                  txn: e.txn_index,
                },
              ],
              CAPS.checks,
            ),
          };
        case "defense_decision":
          return {
            ...state,
            decisions: capped(
              [
                ...state.decisions,
                {
                  id: Date.now() + Math.random(),
                  txn: e.txn_index,
                  decision: e.decision,
                  reasons: e.reasons,
                  scores: e.scores,
                  amount: e.amount,
                },
              ],
              CAPS.decisions,
            ),
          };
        case "cost_update": {
          const costs: CostUpdate = {
            fp_cost_bps: e.fp_cost_bps,
            fp_cost_usd: e.fp_cost_usd,
            fn_loss: e.fn_loss,
            tp_saved: e.tp_saved,
            net_savings: e.net_savings,
            counts: e.counts,
          };
          return { ...state, costs };
        }
        case "graph_update": {
          const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));
          for (const n of e.nodes) nodeMap.set(n.id, n);
          const edgeMap = new Map(
            state.graphEdges.map((x) => [`${x.source}->${x.target}`, x] as const),
          );
          for (const x of e.edges) edgeMap.set(`${x.source}->${x.target}`, x);
          const graphNodes = capped([...nodeMap.values()], CAPS.graphNodes);
          const keepIds = new Set(graphNodes.map((n) => n.id));
          const graphEdges = capped(
            [...edgeMap.values()].filter((x) => keepIds.has(x.source) && keepIds.has(x.target)),
            CAPS.graphEdges,
          );
          return { ...state, graphNodes, graphEdges };
        }
        case "campaign_summary":
          return { ...state, summary: e.data, running: false };
        case "campaign_accepted":
          return state; // control-side only
        default:
          return state;
      }
    }
  }
}

// --------------------------------------------------------------------------- //
// Page
// --------------------------------------------------------------------------- //

export default function ArenaPage() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [conn, setConn] = useState<ConnState>("idle");
  const [turbo, setTurbo] = useState(false);
  const socketRef = useRef<ArenaSocket | null>(null);
  const ringToastedRef = useRef(false);

  // ---- WebSocket lifecycle: connect once, reconnect forever ---------------
  useEffect(() => {
    const handle = (e: ArenaEvent) => {
      // side-effects that must not live in the reducer
      if (e.type === "defense_decision") {
        if (e.decision === "DECLINE") {
          toast.error(`TXN ${e.txn_index ?? "?"} DECLINED`, {
            description: e.reasons.join(", ") || "policy",
            duration: 3500,
          });
        }
        if (e.scores.ring_detected && !ringToastedRef.current) {
          ringToastedRef.current = true;
          toast.warning("Mule ring detected in entity graph", {
            description: `shared infra across 3+ profiles (${e.scores.ring_id ?? "?"})`,
          });
        }
      }
      if (e.type === "campaign_summary") {
        toast.success(`Campaign complete — ${e.data.accepted}/${e.data.txn_slots} landed`, {
          description: `net vs tooling: $${e.data.net_vs_tooling_usd.toLocaleString()} · ${e.data.llm_calls} LLM calls`,
          duration: 8000,
        });
      }
      if (e.type === "error") {
        toast.error("Backend rejected the request", { description: String(e.data) });
      }
      dispatch({ type: "event", event: e });
    };

    const socket = new ArenaSocket(backendWsUrl(), { onEvent: handle, onState: setConn });
    socketRef.current = socket;
    socket.connect();
    return () => socket.close();
  }, []);

  const launch = useCallback(
    (attackFile: string, size: number) => {
      dispatch({ type: "reset" });
      const ok = socketRef.current?.send({
        type: "start_campaign",
        attack_file: attackFile,
        campaign_size: size,
        ...(turbo ? { sleep_s: 0 } : {}),
      });
      if (!ok) toast.error("Socket not open yet", { description: "wait for LIVE badge" });
    },
    [turbo],
  );

  const getStats = useCallback((): AnalystStats => {
    return {
      costs: state.costs,
      decisions: state.decisions.slice(-20).map((d) => ({
        decision: d.decision,
        velocity: d.scores.velocity,
        ring: d.scores.ring_detected,
        amount: d.amount,
      })),
      summary: state.summary as Record<string, unknown> | null,
    };
  }, [state]);

  return (
    <main className="mx-auto flex h-screen max-w-[1800px] flex-col gap-3 p-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="font-mono text-lg font-bold tracking-tight text-zinc-100">
            ADVERSARIAL PAYMENT ARENA
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              Identify → Generate → Defend
            </span>
          </h1>
          <p className="text-[11px] text-muted-foreground">
            LLM fraud campaigns vs. a simulated issuer stack — every thought, gate verdict, and decision, live.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="border-zinc-700 font-mono text-[9.5px] text-zinc-400">
            XGBoost · iForest · NetworkX
          </Badge>
          <Badge variant="outline" className="border-red-900 font-mono text-[9.5px] text-red-400">
            red team: LLM (OpenRouter)
          </Badge>
          <a
            href="/evidence"
            className="font-mono text-[9.5px] text-emerald-400 underline underline-offset-2 transition-colors hover:text-emerald-300"
          >
            evidence &amp; claim ledger →
          </a>
        </div>
      </header>

      <ControlBar conn={conn} running={state.running} turbo={turbo} onTurboChange={setTurbo} onLaunch={launch} />

      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[360px_1fr_420px]">
        {/* ---- left: attacker mind + money ---- */}
        <div className="flex min-h-0 flex-col gap-3">
          <AttackerFeed thoughts={state.thoughts} payloads={state.payloads} checks={state.checks} />
          <CostPanel costs={state.costs} />
        </div>

        {/* ---- center: the graph + analyst ---- */}
        <div className="flex min-h-0 flex-col gap-3">
          <Card className="flex min-h-0 flex-1 flex-col overflow-hidden border-zinc-800 bg-zinc-950/60">
            <CardHeader className="pb-1">
              <CardTitle className="text-sm">
                Entity Graph
                <span className="ml-2 font-mono text-[10px] font-normal text-muted-foreground">
                  red = shared infra (ring signal) · {state.graphNodes.length} nodes
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="min-h-0 flex-1 p-0">
              <EntityGraphCanvas nodes={state.graphNodes} edges={state.graphEdges} className="h-full w-full" />
            </CardContent>
          </Card>
          <div className="h-[230px]">
            <AnalystPanel getStats={getStats} />
          </div>
        </div>

        {/* ---- right: defense + campaign summary ---- */}
        <div className="flex min-h-0 flex-col gap-3">
          <DefenseFeed rows={state.decisions} />
          {state.summary && (
            <Card className="border-zinc-800 bg-zinc-950/60">
              <CardHeader className="pb-1">
                <CardTitle className="text-sm">Last Campaign</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[10.5px] text-muted-foreground">
                <span>spec</span>
                <span className="text-right text-zinc-300">{state.summary.spec_id.replace("ATTACK_", "")}</span>
                <span>landed</span>
                <span className="text-right text-zinc-300">
                  {state.summary.accepted}/{state.summary.txn_slots} ({Math.round(state.summary.accept_rate * 100)}%)
                </span>
                <span>gate rejects</span>
                <span className="text-right text-zinc-300">
                  {Object.entries(state.summary.gate_rejects).map(([k, v]) => `${k}:${v}`).join(" ") || "0"}
                </span>
                <span>LLM calls</span>
                <span className="text-right text-zinc-300">{state.summary.llm_calls}</span>
                <span>gross attempted</span>
                <span className="text-right text-amber-400">${state.summary.gross_value_usd.toLocaleString()}</span>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </main>
  );
}
