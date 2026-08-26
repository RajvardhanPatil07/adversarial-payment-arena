"use client";

/**
 * ControlBar — campaign controls + live backend status.
 * Attack selector comes from /api/attacks; the spec brief (cost / payoff /
 * breakeven / rails) is fetched from /api/load_attack so judges see the
 * taxonomy driving the agent.
 */

import { useEffect, useState } from "react";
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

export function ControlBar({
  conn,
  running,
  turbo,
  onTurboChange,
  onLaunch,
}: {
  conn: ConnState;
  running: boolean;
  turbo: boolean;
  onTurboChange: (v: boolean) => void;
  onLaunch: (attackFile: string, size: number) => void;
}) {
  const [attacks, setAttacks] = useState<string[]>([]);
  const [selected, setSelected] = useState("attack_1");
  const [size, setSize] = useState(25);
  const [spec, setSpec] = useState<LoadAttackResponse | null>(null);
  const [models, setModels] = useState<{ xgb: string; iforest: string } | null>(null);

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
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
        <Badge variant="outline" className={`font-mono text-[10px] ${st.cls}`}>{st.label}</Badge>

        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={running}
          className="h-8 rounded-md border border-zinc-700 bg-zinc-900 px-2 font-mono text-xs text-zinc-200"
        >
          {(attacks.length > 0 ? attacks : ["attack_1"]).map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>

        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          size
          <Input
            type="number"
            min={1}
            max={200}
            value={size}
            disabled={running}
            onChange={(e) => setSize(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
            className="h-8 w-20 border-zinc-700 bg-zinc-900 font-mono text-xs"
          />
        </label>

        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          turbo (offline)
          <Switch checked={turbo} onCheckedChange={onTurboChange} disabled={running} />
        </label>

        <Button
          size="sm"
          disabled={running || conn !== "open"}
          onClick={() => onLaunch(selected, size)}
          className="ml-auto bg-red-600 font-mono text-xs font-bold tracking-wider text-white hover:bg-red-500"
        >
          {running ? "CAMPAIGN LIVE…" : "▶ LAUNCH CAMPAIGN"}
        </Button>
      </div>

      {spec && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-[10.5px] text-muted-foreground">
          <span className="font-semibold text-zinc-300">{spec.spec.attack_name}</span>
          <span>·</span>
          <span className="font-mono">
            cost ${spec.spec.economic_model.acquisition_cost_usd} → payoff $
            {spec.spec.economic_model.expected_payoff_usd} · breakeven{" "}
            {spec.spec.economic_model.breakeven_txns} txns
          </span>
          <span>·</span>
          <span className="font-mono">
            rails {spec.spec.constraints.pos_entry_modes.join("/")} · 3DS{" "}
            {spec.spec.constraints.preferred_three_ds.join("/")} · ${spec.spec.constraints.min_amount_usd}-
            {spec.spec.constraints.max_amount_usd}
          </span>
          {models && (
            <span className="ml-auto font-mono text-[9px]">
              xgb:{models.xgb.includes("loaded") ? "✓" : "fallback"} iforest:
              {models.iforest.includes("loaded") ? "✓" : "fallback"}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
