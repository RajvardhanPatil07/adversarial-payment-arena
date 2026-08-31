"use client";

/**
 * AnalystPanel — Vercel AI SDK wiring. POSTs the live session stats to
 * /api/analyst (which streams from an OpenRouter model — the same LLM that
 * plays the attacker, set via OPENROUTER_MODEL) and renders the tokens as
 * they arrive. Graceful when unkeyed: the route returns 500 and we surface
 * it as a toast-able error line.
 */

import { useCallback, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { CostUpdate, Decision } from "@/lib/arena-types";

export interface AnalystStats {
  costs: CostUpdate | null;
  decisions: { decision: Decision; velocity: number; ring: boolean; amount: number }[];
  summary: Record<string, unknown> | null;
}

export function AnalystPanel({ getStats }: { getStats: () => AnalystStats }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    setBusy(true);
    setText("");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch("/api/analyst", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stats: getStats() }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        const detail = await res.json().catch(() => ({}));
        setText(`⚠ ${detail.error ?? `analyst route failed (${res.status})`}`);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        setText((prev) => prev + decoder.decode(value, { stream: true }));
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setText(`⚠ ${(err as Error).message}`);
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }, [getStats]);

  return (
    <Card className="flex min-h-0 flex-col border-zinc-800 bg-zinc-950/60">
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm">Analyst brief</CardTitle>
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => (busy ? abortRef.current?.abort() : void run())}
          className="h-7 border-zinc-700 px-2 font-mono text-[10px] text-zinc-300 hover:bg-zinc-800"
        >
          {busy ? "Stop" : "Summarize the fight"}
        </Button>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 overflow-y-auto">
        {text ? (
          <p className="whitespace-pre-wrap text-xs leading-relaxed text-zinc-300">{text}</p>
        ) : (
          <p className="text-xs leading-relaxed text-muted-foreground">
            Generate a plain-language read of the current session: cost, decision mix,
            and which defense layers exposed the attack.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
