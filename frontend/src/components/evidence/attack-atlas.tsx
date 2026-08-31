"use client";

/**
 * <AttackAtlas> — the 22-taxon atlas with filters (PHASE 5).
 *
 * Client component: channel chips, rail chips and an executable-only toggle.
 * All card content arrives as pre-joined props from the server page — every
 * measured value was read from family_coverage.json at render time; this
 * component only filters what is already present and never fetches.
 */

import { useMemo, useState } from "react";

import { AttackCard, type AttackCardProps } from "./attack-card";

export type AtlasEntry = AttackCardProps & {
  /** Filter keys — prose facts from the taxonomy data. */
  channel: string;
  rail: string;
};

export function AttackAtlas({ entries }: { entries: readonly AtlasEntry[] }) {
  const [channel, setChannel] = useState<string | null>(null);
  const [rail, setRail] = useState<string | null>(null);
  const [execOnly, setExecOnly] = useState(false);

  const channels = useMemo(() => [...new Set(entries.map((e) => e.channel))].sort(), [entries]);
  const rails = useMemo(() => [...new Set(entries.map((e) => e.rail))].sort(), [entries]);

  const visible = entries.filter(
    (e) =>
      (channel === null || e.channel === channel) &&
      (rail === null || e.rail === rail) &&
      (!execOnly || e.executable),
  );

  return (
    <div className="flex flex-col gap-6">
      {/* filter chips */}
      <div className="flex flex-col gap-3 rounded-[var(--r-md)] border border-border bg-surface-1 p-4">
        <FilterRow
          label="channel"
          options={channels}
          active={channel}
          onSelect={(v) => setChannel(v === channel ? null : v)}
        />
        <FilterRow
          label="rail"
          options={rails}
          active={rail}
          onSelect={(v) => setRail(v === rail ? null : v)}
        />
        <div className="flex items-center gap-3">
          <span className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-faint">status</span>
          <button
            type="button"
            aria-pressed={execOnly}
            onClick={() => setExecOnly((v) => !v)}
            className={`type-ui rounded-full border px-3 py-1 text-[0.6875rem] uppercase tracking-[0.06em] transition-colors ${
              execOnly
                ? "border-pass/60 bg-pass/10 text-pass"
                : "border-border-hi text-text-dim hover:border-border-hi hover:text-text"
            }`}
          >
            executable only
          </button>
          {(channel !== null || rail !== null || execOnly) && (
            <button
              type="button"
              onClick={() => {
                setChannel(null);
                setRail(null);
                setExecOnly(false);
              }}
              className="type-ui text-[0.6875rem] uppercase tracking-[0.06em] text-text-dim underline decoration-border-hi underline-offset-2 transition-colors hover:text-text"
            >
              clear all
            </button>
          )}
          <span className="type-num ms-auto text-xs text-text-dim" aria-live="polite">
            {visible.length} / {entries.length} shown
          </span>
        </div>
      </div>

      {/* the atlas grid */}
      {visible.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {visible.map((entry) => (
            <AttackCard key={entry.id} {...entry} />
          ))}
        </div>
      ) : (
        <p className="type-ui rounded-[var(--r-md)] border border-border bg-surface-1 p-6 text-sm text-text-dim">
          No taxon matches the current filters.
        </p>
      )}
    </div>
  );
}

function FilterRow({
  label,
  options,
  active,
  onSelect,
}: {
  label: string;
  options: string[];
  active: string | null;
  onSelect: (v: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="type-ui w-16 shrink-0 text-[0.6875rem] uppercase tracking-[0.08em] text-text-faint">{label}</span>
      {options.map((opt) => (
        <button
          key={opt}
          type="button"
          aria-pressed={active === opt}
          onClick={() => onSelect(opt)}
          className={`type-ui rounded-full border px-3 py-1 text-[0.6875rem] uppercase tracking-[0.06em] transition-colors ${
            active === opt
              ? "border-blue/60 bg-blue/10 text-blue"
              : "border-border-hi text-text-dim hover:border-border-hi hover:text-text"
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}
