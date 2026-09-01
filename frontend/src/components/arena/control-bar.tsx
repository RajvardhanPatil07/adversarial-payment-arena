"use client";

/**
 * ControlBar — campaign controls + live backend status.
 * Attack selector comes from /api/attacks; the spec brief (cost / payoff /
 * breakeven / rails) is fetched from /api/load_attack so judges see the
 * taxonomy driving the agent.
 */

import { useEffect, useState } from "react";
import { ChevronDown, FlaskConical, Play, SlidersHorizontal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { getHealth, listAttacks, loadAttack, type LoadAttackResponse } from "@/lib/api";
import type { ConnState } from "@/lib/ws";

const STATE_STYLE: Record<ConnState, { label: string; cls: string }> = {
  idle: { label: "IDLE", cls: "border-zinc-700 text-zinc-400" },
  connecting: { label: "CONNECTING", cls: "border-amber-700 text-amber-400 animate-pulse" },
  open: { label: "LIVE", cls: "border-emerald-700 text-emerald-400" },
  closed: { label: "RECONNECTING", cls: "border-red-800 text-red-400 animate-pulse" },
};

function readableAttack(value: string): string {
  return value
    .replace(/^attack_\d+_/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function ControlBar({
  conn,
  running,
  turbo,
  onTurboChange,
  onLaunch,
  onGuidedDemo,
}: {
  conn: ConnState;
  running: boolean;
  turbo: boolean;
  onTurboChange: (v: boolean) => void;
  onLaunch: (attackFile: string, size: number) => void;
  onGuidedDemo: () => void;
}) {
  const [attacks, setAttacks] = useState<string[]>([]);
  const [selected, setSelected] = useState("attack_2_synthetic_mule_ring");
  const [size, setSize] = useState(12);
  const [spec, setSpec] = useState<LoadAttackResponse | null>(null);
  const [models, setModels] = useState<{ xgb: string; iforest: string } | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    listAttacks()
      .then((r) => setAttacks(r.attacks))
      .catch(() => setAttacks([]));
    getHealth()
      .then((h) => setModels(h.models))
      .catch(() => setModels(null));
  }, []);

  useEffect(() => {
    if (!selected) return;
    loadAttack(selected).then(setSpec).catch(() => setSpec(null));
  }, [selected]);

  const st = STATE_STYLE[conn];

  return (
    <div className="scenario-control">
      <div className="scenario-control__bar">
        <div className="flex min-w-0 items-center gap-3">
          <Badge variant="outline" className={`shrink-0 font-mono text-[10px] ${st.cls}`}>{st.label}</Badge>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-zinc-100">{readableAttack(selected)}</p>
            <p className="hidden text-xs text-zinc-500 sm:block">Recommended scenario · {size} payment attempts</p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            disabled={running}
            onClick={() => setExpanded((value) => !value)}
            aria-expanded={expanded}
            className="hidden h-9 gap-2 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 sm:flex"
          >
            <SlidersHorizontal className="size-3.5" /> Configure
            <ChevronDown className={`size-3.5 transition-transform ${expanded ? "rotate-180" : ""}`} />
          </Button>
          <Button
            size="sm"
            disabled={running}
            onClick={onGuidedDemo}
            aria-describedby={conn === "open" ? undefined : "control-connection-status"}
            className="h-9 gap-2 bg-emerald-400 px-4 font-semibold text-emerald-950 hover:bg-emerald-300"
          >
            <Play className="size-3.5 fill-current" />
            {running ? "Campaign live" : conn === "open" ? "Guided demo" : "Wake & start demo"}
          </Button>
        </div>
      </div>

      {conn !== "open" && (
        <p id="control-connection-status" className="scenario-control__connection" role="status" aria-live="polite">
          {conn === "closed"
            ? "Connection interrupted. Choose “Wake & start demo” to retry; evidence remains available."
            : "Waking the live defense engine—usually 10–15 seconds."}
        </p>
      )}

      {expanded && (
        <div className="scenario-control__details">
          <label className="scenario-field min-w-[240px] flex-1">
            <span>Attack scenario</span>
            <select value={selected} onChange={(e) => setSelected(e.target.value)} disabled={running}>
              {(attacks.length > 0 ? attacks : ["attack_2_synthetic_mule_ring"]).map((attack) => (
                <option key={attack} value={attack}>{readableAttack(attack)}</option>
              ))}
            </select>
          </label>

          <label className="scenario-field">
            <span>Campaign size</span>
            <Input
              type="number"
              min={1}
              max={200}
              value={size}
              disabled={running}
              onChange={(e) => setSize(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
              className="h-10 w-24 border-zinc-700 bg-zinc-900 font-mono text-sm"
            />
          </label>

          <label className="flex items-center gap-3 rounded-xl bg-zinc-900/70 px-3 py-2">
            <Switch checked={turbo} onCheckedChange={onTurboChange} disabled={running} />
            <span><b className="block text-xs text-zinc-200">Fast offline run</b><small className="text-zinc-500">Skip model pacing</small></span>
          </label>

          <Button
            size="sm"
            variant="outline"
            disabled={running || conn !== "open"}
            onClick={() => onLaunch(selected, size)}
            className="h-10 gap-2 border-red-900 bg-red-950/40 px-4 text-red-200 hover:bg-red-950"
          >
            <FlaskConical className="size-4" /> Launch custom campaign
          </Button>
        </div>
      )}

      {spec && (
        <div className="scenario-brief">
          <span><b>Attack economics</b> ${spec.spec.economic_model.acquisition_cost_usd} cost → ${spec.spec.economic_model.expected_payoff_usd} expected payoff</span>
          <span><b>Payment rails</b> {spec.spec.constraints.pos_entry_modes.join(" / ")} · ${spec.spec.constraints.min_amount_usd}–{spec.spec.constraints.max_amount_usd}</span>
          {models && (
            <span className="ml-auto font-mono text-[10px] text-zinc-500">
              velocity {models.xgb.includes("loaded") ? "ready" : "fallback"} · anomaly {models.iforest.includes("loaded") ? "ready" : "fallback"}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
