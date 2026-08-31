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
import { ArrowRight, Network, RefreshCcw, ShieldCheck, WifiOff } from "lucide-react";
import Link from "next/link";
import { toast } from "sonner";

import { AnalystPanel, type AnalystStats } from "@/components/arena/analyst-panel";
import { AttackerFeed, type CheckRow, type PayloadRow, type ThoughtRow } from "@/components/arena/attacker-feed";
import { ControlBar } from "@/components/arena/control-bar";
import { CostPanel } from "@/components/arena/cost-panel";
import { DefenseFeed, type DecisionRow } from "@/components/arena/defense-feed";
import { EntityGraphCanvas } from "@/components/arena/graph-canvas";
import { JudgeContext } from "@/components/arena/judge-context";
import { OutcomePanel } from "@/components/arena/outcome-panel";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ArenaEvent, CampaignSummaryData, CostUpdate, GraphEdge, GraphNode } from "@/lib/arena-types";
import { COMMITTED_SCISSOR, signedPoints, WINNING_THESIS } from "@/lib/committed-evidence";
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

type MotionBeat = "attacker" | "gate" | "graph" | "decision" | "contained";

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

// --------------------------------------------------------------------------- //
// Guided demo ("Judge Mode")
// --------------------------------------------------------------------------- //

const GUIDED_ATTACK = "attack_2_synthetic_mule_ring";
const GUIDED_SIZE = 25;

/**
 * The scripted 90-second walkthrough. Steps advance on real WebSocket events,
 * never on timers, so the narration cannot drift ahead of the fight.
 */
const GUIDED_STEPS: string[] = [
  "Attacker generation: the red team is assembling a synthetic mule ring across apparently unrelated accounts.",
  "Plausibility gate: each generated payment must satisfy fraud economics, metadata, and payment-rail rules.",
  "Graph discovery: the payments look coherent alone while shared infrastructure accumulates between them.",
  "Graph discovery: shared devices and IP addresses expose the coordinated ring that transaction scoring misses.",
  "Defense action: the stack contains the ring while pricing fraud loss against legitimate-customer friction.",
  "Financial outcome recorded. Now verify the fidelity scissor that determines whether these escapes are safe for retraining.",
];

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
  const [guidedStep, setGuidedStep] = useState<number | null>(null);
  const [guidedQueued, setGuidedQueued] = useState(false);
  const [motionBeat, setMotionBeat] = useState<{ kind: MotionBeat; sequence: number } | null>(null);
  const socketRef = useRef<ArenaSocket | null>(null);
  const ringToastedRef = useRef(false);
  const guidedRef = useRef<number | null>(null);

  const setGuided = useCallback((step: number | null) => {
    guidedRef.current = step;
    setGuidedStep(step);
  }, []);

  // ---- WebSocket lifecycle: connect once, reconnect forever ---------------
  useEffect(() => {
    const handle = (e: ArenaEvent) => {
      if (guidedRef.current !== null) {
        const beat: MotionBeat | null =
          e.type === "agent_thought" ? "attacker" :
          e.type === "plausibility_check" && e.data.accepted ? "gate" :
          e.type === "graph_update" ? "graph" :
          e.type === "defense_decision" ? "decision" :
          e.type === "campaign_summary" ? "contained" : null;
        if (beat) setMotionBeat((current) => ({ kind: beat, sequence: (current?.sequence ?? 0) + 1 }));
      }

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

      // Guided demo narration: advance only on the event that proves the step.
      const step = guidedRef.current;
      if (step !== null) {
        if (step === 0 && e.type === "payload_generated") setGuided(1);
        else if (step === 1 && e.type === "plausibility_check" && e.data.accepted) setGuided(2);
        else if (step === 2 && e.type === "defense_decision" && e.scores.ring_detected) setGuided(3);
        else if (step === 3 && e.type === "defense_decision" && e.decision === "DECLINE") setGuided(4);
        else if (e.type === "campaign_summary" && step < GUIDED_STEPS.length - 1) {
          setGuided(GUIDED_STEPS.length - 1);
        }
      }

      dispatch({ type: "event", event: e });
    };

    const socket = new ArenaSocket(backendWsUrl(), { onEvent: handle, onState: setConn });
    socketRef.current = socket;
    socket.connect();
    return () => socket.close();
  }, [setGuided]);

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

  const sendGuidedCampaign = useCallback(() => {
    dispatch({ type: "reset" });
    const ok = socketRef.current?.send({
      type: "start_campaign",
      attack_file: GUIDED_ATTACK,
      campaign_size: GUIDED_SIZE,
      sleep_s: 0,
    });
    if (ok) {
      setGuided(0);
      setGuidedQueued(false);
    } else {
      setGuidedQueued(true);
    }
  }, [setGuided]);

  const startGuidedDemo = useCallback(() => {
    if (conn === "open") {
      sendGuidedCampaign();
      return;
    }
    setGuidedQueued(true);
    if (conn === "closed" || conn === "idle") socketRef.current?.connect();
  }, [conn, sendGuidedCampaign]);

  useEffect(() => {
    if (conn === "open" && guidedQueued) sendGuidedCampaign();
  }, [conn, guidedQueued, sendGuidedCampaign]);

  const retryConnection = useCallback(() => {
    setGuidedQueued(true);
    socketRef.current?.connect();
  }, []);

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

  const hasActivity = state.running || state.decisions.length > 0 || state.thoughts.length > 0 || Boolean(state.summary);
  const ringDecision = state.decisions.find((decision) => decision.scores.ring_detected);
  const ringInsight = ringDecision
    ? `Coordinated infrastructure detected${ringDecision.scores.ring_id ? ` · ${ringDecision.scores.ring_id}` : ""}`
    : state.running
      ? "Watching for shared devices, IP addresses, and merchants"
      : "The graph will connect infrastructure that isolated payment scoring cannot see";
  const guidedFocus = guidedStep === null
    ? null
    : guidedStep <= 1
      ? "attacker"
      : guidedStep <= 3
        ? "graph"
        : guidedStep === 4
          ? "defense"
          : "outcome";

  return (
    <main className="arena-shell">
      <header className="arena-header">
        <Link href="/" className="arena-brand" aria-label="Adversarial Payment Arena home">
          <span className="arena-mark"><ShieldCheck aria-hidden="true" /></span>
          <span>
            <strong>Adversarial Payment Arena</strong>
            <small>Closed-loop fraud defense</small>
          </span>
        </Link>
        <nav className="arena-nav" aria-label="Primary navigation">
          <a href="#arena">Live arena</a>
          <Link href="/evidence">Evidence <ArrowRight aria-hidden="true" /></Link>
        </nav>
      </header>

      {!hasActivity && (
        <section className="arena-briefing">
          <div className="arena-briefing__copy">
            <h1>{WINNING_THESIS}</h1>
            <p className="arena-briefing__lede">
              Watch one synthetic mule-ring campaign become visible, contained, and safe to learn from.
            </p>
            <div className="arena-briefing__actions">
              <Button
                onClick={startGuidedDemo}
                aria-describedby={conn === "open" ? undefined : "engine-status"}
                className="h-14 gap-2 bg-emerald-400 px-6 text-sm font-semibold text-emerald-950 shadow-lg shadow-emerald-950/30 hover:bg-emerald-300"
              >
                {guidedQueued ? "Start when engine is ready" : "Start the 90-second demo"}
                <ArrowRight className="size-4" />
              </Button>
              <div className="arena-briefing__secondary">
                <Link href="/evidence" className="briefing-link">Verify evidence</Link>
                <Link href="/evidence#reproduce" className="briefing-link">Reproduce results</Link>
              </div>
            </div>
            {conn !== "open" && (
              <div id="engine-status" className={`engine-status ${conn === "closed" ? "engine-status--error" : ""}`} role="status" aria-live="polite">
                {conn === "closed" ? <WifiOff aria-hidden="true" /> : <span className="engine-status__spinner" aria-hidden="true" />}
                <span>
                  <b>{conn === "closed" ? "The live engine has not connected." : "Waking the live defense engine—usually 10–15 seconds."}</b>
                  <small>{conn === "closed" ? "The evidence remains available while the arena reconnects." : "You can queue the demo now or inspect the evidence while it wakes."}</small>
                </span>
                {conn === "closed" && (
                  <button type="button" onClick={retryConnection}>
                    <RefreshCcw aria-hidden="true" /> Retry connection
                  </button>
                )}
              </div>
            )}
          </div>
          <div className="arena-proof" aria-label="Headline fidelity-scissor result">
            <div data-tone="synthetic"><b>{signedPoints(COMMITTED_SCISSOR.syntheticRecallGain)}</b><span>recall on synthetic attacks</span></div>
            <div data-tone="harm"><b>{signedPoints(COMMITTED_SCISSOR.realRecallLoss)}</b><span>recall on held-out arena fraud</span></div>
            <div data-tone="protected"><b>{signedPoints(COMMITTED_SCISSOR.recallProtected)}</b><span>real-fraud recall protected by the gate</span></div>
            <p>
              <ShieldCheck aria-hidden="true" />
              <span><b>Evaluation boundary</b> Held-out arena fraud is simulated evaluation data, not issuer production traffic.</span>
            </p>
          </div>
        </section>
      )}

      {hasActivity && <JudgeContext />}

      <section
        id="arena"
        className={`space-y-3 scroll-mt-4 ${guidedFocus ? `arena-guided arena-guided--${guidedFocus}` : ""}`}
        data-motion-beat={motionBeat?.kind}
      >
        {hasActivity && (
          <ControlBar
            conn={conn}
            running={state.running}
            turbo={turbo}
            onTurboChange={setTurbo}
            onLaunch={launch}
            onGuidedDemo={startGuidedDemo}
          />
        )}

        {guidedStep !== null && (
          <div className="guided-narration" role="status" aria-live="polite">
            <div className="guided-progress" aria-label={`Guided demo step ${guidedStep + 1} of ${GUIDED_STEPS.length}`}>
              {GUIDED_STEPS.map((_, index) => (
                <span key={index} className={index === guidedStep ? "is-current" : index < guidedStep ? "is-complete" : ""} />
              ))}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-medium text-emerald-400">Step {guidedStep + 1} of {GUIDED_STEPS.length}</p>
              <p className="text-sm leading-relaxed text-zinc-200">{GUIDED_STEPS[guidedStep]}</p>
            </div>
            {guidedStep === GUIDED_STEPS.length - 1 ? (
              <div className="guided-result">
                <span><b>{signedPoints(COMMITTED_SCISSOR.syntheticRecallGain)}</b> synthetic</span>
                <span><b>{signedPoints(COMMITTED_SCISSOR.realRecallLoss)}</b> held-out arena fraud</span>
                <span><b>{signedPoints(COMMITTED_SCISSOR.recallProtected)}</b> protected</span>
                <Link href="/evidence" className="outcome-link">Verify the evidence <ArrowRight className="size-3.5" /></Link>
              </div>
            ) : (
              <button onClick={() => setGuided(null)} className="text-xs text-zinc-500 hover:text-zinc-200">Dismiss</button>
            )}
          </div>
        )}

        <div className="guided-stage guided-stage--outcome">
          <OutcomePanel costs={state.costs} summary={state.summary} decisions={state.decisions} running={state.running} />
        </div>

        <div className="arena-workspace">
          {motionBeat && (
            <div key={motionBeat.sequence} className={`event-tracer event-tracer--${motionBeat.kind}`} aria-hidden="true">
              <span />
            </div>
          )}
          <aside className="arena-workspace__attacker guided-stage guided-stage--attacker">
            <AttackerFeed thoughts={state.thoughts} payloads={state.payloads} checks={state.checks} />
            <CostPanel costs={state.costs} />
          </aside>

          <section className="arena-workspace__graph guided-stage guided-stage--graph">
            <Card className="graph-stage">
              <CardHeader className="graph-stage__header">
                <div>
                  <CardTitle className="text-base text-zinc-50">Entity intelligence</CardTitle>
                  <p className={`mt-1 text-xs ${ringDecision ? "text-red-300" : "text-zinc-500"}`}>{ringInsight}</p>
                </div>
                <div className="graph-legend" aria-label="Entity graph legend">
                  <span data-kind="customer">Customer</span><span data-kind="device">Device</span>
                  <span data-kind="ip">IP</span><span data-kind="merchant">Merchant</span>
                </div>
              </CardHeader>
              <CardContent className="relative min-h-0 flex-1 p-0">
                {state.graphNodes.length === 0 && (
                  <div className="graph-empty">
                    <Network aria-hidden="true" />
                    <strong>No suspicious connections yet</strong>
                    <span>Start the guided demo to watch unrelated accounts converge into one fraud ring.</span>
                  </div>
                )}
                <EntityGraphCanvas nodes={state.graphNodes} edges={state.graphEdges} className="absolute inset-0" />
              </CardContent>
            </Card>
            <div className="h-[168px]"><AnalystPanel getStats={getStats} /></div>
          </section>

          <aside className="arena-workspace__defense guided-stage guided-stage--defense">
            <DefenseFeed rows={state.decisions} />
            {state.summary && (
              <Card className="campaign-recap">
                <CardHeader className="pb-2"><CardTitle className="text-sm">Campaign recap</CardTitle></CardHeader>
                <CardContent className="campaign-recap__grid">
                  <span><small>Plausible payments</small><b>{state.summary.accepted}/{state.summary.txn_slots}</b></span>
                  <span><small>Gross attempted</small><b>${state.summary.gross_value_usd.toLocaleString()}</b></span>
                  <span><small>Gate rejects</small><b>{Object.values(state.summary.gate_rejects).reduce((sum, value) => sum + value, 0)}</b></span>
                  <span><small>Net after tooling</small><b>${state.summary.net_vs_tooling_usd.toLocaleString()}</b></span>
                </CardContent>
              </Card>
            )}
          </aside>
        </div>
      </section>

      <footer className="arena-footer">
        <span>Simulation only · no live payment traffic</span>
        <Link href="/evidence">Reproduce every result with the evidence ledger <ArrowRight aria-hidden="true" /></Link>
      </footer>
    </main>
  );
}
