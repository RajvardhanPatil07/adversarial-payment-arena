/**
 * Thin REST helpers for the FastAPI control endpoints. The live fight
 * flows over WebSocket (ws.ts); these are only for bootstrap data.
 */

import type { CampaignSummaryData, GraphEdge, GraphNode, HealthInfo, MerchantInfo } from "@/lib/arena-types";
import { backendHttpUrl } from "@/lib/ws";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${backendHttpUrl()}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<HealthInfo> {
  return getJson<HealthInfo>("/api/health");
}

export function getMerchants(): Promise<MerchantInfo[]> {
  return getJson<MerchantInfo[]>("/api/merchants");
}

export function listAttacks(): Promise<{ attacks: string[] }> {
  return getJson<{ attacks: string[] }>("/api/attacks");
}

export function getGraphSnapshot(): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }> {
  return getJson<{ nodes: GraphNode[]; edges: GraphEdge[] }>("/api/graph/snapshot");
}

export interface LoadAttackResponse {
  filename: string;
  spec: {
    spec_id: string;
    attack_name: string;
    taxon: string;
    description: string;
    preconditions: string;
    economic_model: { acquisition_cost_usd: number; expected_payoff_usd: number; breakeven_txns: number };
    constraints: {
      pos_entry_modes: string[];
      preferred_three_ds: string[];
      target_verticals: string[];
      min_amount_usd: number;
      max_amount_usd: number;
    };
  };
}

export function loadAttack(filename: string): Promise<LoadAttackResponse> {
  return fetch(`${backendHttpUrl()}/api/load_attack`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
    cache: "no-store",
  }).then((res) => {
    if (!res.ok) throw new Error(`/api/load_attack -> ${res.status}`);
    return res.json() as Promise<LoadAttackResponse>;
  });
}

export type { CampaignSummaryData };
