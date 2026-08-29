/**
 * /api/analyst — streams a fraud-ops narration of the live session.
 *
 * Uses the Vercel AI SDK (streamText) pointed at an OpenRouter model — by
 * default the same free reasoning model that plays the attacker (override
 * with OPENROUTER_MODEL; `stealth/ox-alpha` when that slug is served to the
 * account), now briefing the defenders. Requires OPENROUTER_API_KEY in the
 * frontend environment (.env.local); without it the route fails loudly and
 * the UI shows a toast-able error instead of pretending.
 */

import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";
import type { AnalystStats } from "@/components/arena/analyst-panel";

export async function POST(req: Request) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return Response.json(
      { error: "OPENROUTER_API_KEY not set in frontend env — analyst offline" },
      { status: 500 },
    );
  }

  let stats: AnalystStats;
  try {
    stats = (await req.json()).stats as AnalystStats;
  } catch {
    return Response.json({ error: "body must be JSON {stats}" }, { status: 400 });
  }

  const openai = createOpenAI({
    apiKey,
    baseURL: process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1",
    headers: {
      "HTTP-Referer": "https://github.com/adversarial-payment-arena",
      "X-Title": "Adversarial Payment Arena",
    },
  });

  const result = streamText({
    model: openai(process.env.OPENROUTER_MODEL ?? "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    system:
      "You are a senior fraud-ops analyst narrating a live adversarial payment simulation. " +
      "In under 130 words: name the attack pattern the decisions suggest, call out the most " +
      "important cost-matrix number, and give ONE concrete tuning recommendation. Terse, " +
      "confident, no preamble, no markdown headers.",
    prompt: `Live session stats:\n${JSON.stringify(stats, null, 2)}\n\nBrief the SOC team.`,
  });

  return result.toTextStreamResponse();
}
