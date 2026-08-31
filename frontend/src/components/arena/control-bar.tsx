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
  idle: { label: "IDLE", cls: "border-border-hi text-text-dim" },
  connecting: { label: "CONNECTING", cls: "border-warn/70 text-warn animate-pulse" },
  open: { label: "LIVE", cls: "border-pass/70 text-pass" },
  closed: { label: "RECONNECTING", cls: "border-fail/70 text-fail animate-pulse" },
};

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
      <div className="flex flex-wrap items-center gap-3 rounded-[var(--r-md)] border border-border bg-surface-1 p-3">
        <Badge variant="outline" className={`type-ui text-[10px] font-semibold tracking-wide ${st.cls}`}>{st.label}</Badge>

        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={running}
          className="type-num h-8 rounded-[var(--r-md)] border border-border-hi bg-surface-2 px-2 text-xs text-text"
        >
          {(attacks.length > 0 ? attacks : ["attack_2_synthetic_mule_ring"]).map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>

        <label className="type-ui flex items-center gap-1.5 text-xs text-text-dim">
          size
          <Input
            type="number"
            min={1}
            max={200}
            value={size}
            disabled={running}
            onChange={(e) => setSize(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
            className="type-num h-8 w-20 border-border-hi bg-surface-2 text-xs"
          />
        </label>

        <label className="type-ui flex items-center gap-1.5 text-xs text-text-dim">
          turbo (offline)
          <Switch checked={turbo} onCheckedChange={onTurboChange} disabled={running} />
        </label>

        <Button
          size="sm"
          disabled={running || conn !== "open"}
          onClick={onGuidedDemo}
          className="type-ui ml-auto rounded-[var(--r-md)] bg-blue text-xs font-bold tracking-wider text-bg hover:bg-blue/85"
        >
          ▶ GUIDED DEMO
        </Button>

        <Button
          size="sm"
          disabled={running || conn !== "open"}
          onClick={() => onLaunch(selected, size)}
          className="type-ui rounded-[var(--r-md)] bg-red text-xs font-bold tracking-wider text-bg hover:bg-red/85"
        >
          {running ? "CAMPAIGN LIVE…" : "▶ LAUNCH CAMPAIGN"}
        </Button>
      </div>

      {spec && (
        <div className="type-ui flex flex-wrap items-center gap-2 rounded-[var(--r-md)] border border-border bg-surface-2 px-3 py-2 text-[10.5px] text-text-dim">
          <span className="type-num font-semibold text-text">{spec.spec.attack_name}</span>
          <span>·</span>
          <span className="type-num">
            cost ${spec.spec.economic_model.acquisition_cost_usd} → payoff $
            {spec.spec.economic_model.expected_payoff_usd} · breakeven{" "}
            {spec.spec.economic_model.breakeven_txns} txns
          </span>
          <span>·</span>
          <span className="type-num">
            rails {spec.spec.constraints.pos_entry_modes.join("/")} · 3DS{" "}
            {spec.spec.constraints.preferred_three_ds.join("/")} · ${spec.spec.constraints.min_amount_usd}-
            {spec.spec.constraints.max_amount_usd}
          </span>
          {models && (
            <span className="type-num ml-auto text-[9px] text-text-dim">
              xgb:{models.xgb.includes("loaded") ? "✓" : "fallback"} iforest:
              {models.iforest.includes("loaded") ? "✓" : "fallback"}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
