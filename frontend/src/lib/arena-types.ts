/**
 * Wire protocol shared with the FastAPI backend (backend/main.py).
 * One source of truth for every event that can cross the WebSocket,
 * plus the REST payload shapes.
 */

export type Decision = "APPROVE" | "STEP_UP" | "DECLINE" | "MANUAL_REVIEW";

export interface PaymentPayload {
  transaction_id: string;
  customer_id: string;
  merchant_id: string;
  mcc: number;
  amount: number;
  currency: string;
  pos_entry_mode: string;
  "3ds_status": "Y" | "A" | "N";
  ip_address: string;
  ip_country: string;
  device_id: string;
  stolen_resource: string | null;
  timestamp: string;
}

export interface DecisionScores {
  velocity: number;
  novelty_anomaly: number;
  is_anomaly: boolean;
  ring_risk: number;
  ring_detected: boolean;
  ring_id: string | null;
}

export interface CostUpdate {
  fp_cost_bps: number;
  fp_cost_usd: number;
  fn_loss: number;
  tp_saved: number;
  net_savings: number;
  counts: {
    false_positives: number;
    false_negatives: number;
    true_positives_declined: number;
  };
}

export interface GraphNode {
  id: string; // "C:CUST_0001" | "D:DEV_..." | "I:1.2.3.4" | "M:MERCH_..."
  type: "customer" | "device" | "ip" | "merchant" | "unknown";
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface CampaignSummaryData {
  spec_id: string;
  txn_slots: number;
  accepted: number;
  accept_rate: number;
  attempts: number;
  gate_rejects: Record<string, number>;
  malformed: number;
  llm_calls: number;
  gross_value_usd: number;
  net_vs_tooling_usd: number;
}

export type ArenaEvent =
  | { type: "campaign_accepted"; spec: string; size: number; attack_file: string }
  | { type: "campaign_start"; data: { spec_id: string; size: number } }
  | { type: "agent_thought"; role: "PLANNER" | "OPERATOR"; data: string; txn_index?: number }
  | { type: "payload_generated"; data: PaymentPayload; txn_index?: number }
  | {
      type: "plausibility_check";
      data: { accepted: boolean; reason: string; risk_flags?: string[]; attempt?: number };
      txn_index?: number;
    }
  | { type: "system_feedback"; data: string; txn_index?: number }
  | {
      type: "defense_decision";
      txn_index?: number | null;
      decision: Decision;
      reasons: string[];
      scores: DecisionScores;
      amount: number;
    }
  | ({ type: "cost_update" } & CostUpdate)
  | { type: "graph_update"; nodes: GraphNode[]; edges: GraphEdge[] }
  | { type: "campaign_summary"; data: CampaignSummaryData }
  | { type: "txn_abandoned"; txn_index: number; data: { reason: string } }
  | { type: "error"; data: string }
  | { type: "pong" };
// NOTE: no open-ended fallback member — it would break discriminated-union
// narrowing everywhere. Unknown wire frames are cast at the WS parse boundary.

export interface MerchantInfo {
  merchant_id: string;
  name: string;
  mcc: number;
  country: string;
  city: string;
  category: string;
  is_online: boolean;
}

export interface HealthInfo {
  status: string;
  models: { xgb: string; iforest: string };
  events_seen: number;
  costs: CostUpdate;
}
