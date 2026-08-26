/**
 * ArenaSocket — resilient WebSocket client for the SOC dashboard.
 *
 * Responsibilities:
 *  - JSON event parsing with a safe fallback (unknown frames are dropped,
 *    never crash the UI),
 *  - automatic reconnection with exponential backoff + jitter,
 *  - heartbeat pings so dead connections surface within one interval,
 *  - connection-state callbacks for the UI badge.
 *
 * The backend protocol is request/response per campaign; this client is
 * deliberately dumb plumbing — all state lives in the React reducer.
 */

import type { ArenaEvent } from "@/lib/arena-types";

export type ConnState = "idle" | "connecting" | "open" | "closed";

export interface ArenaSocketHandlers {
  onEvent: (event: ArenaEvent) => void;
  onState: (state: ConnState) => void;
}

const MAX_BACKOFF_MS = 5000;
const BASE_BACKOFF_MS = 600;
const HEARTBEAT_MS = 20_000;

export class ArenaSocket {
  private ws: WebSocket | null = null;
  private retry = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private closedByUser = false;

  constructor(
    private readonly url: string,
    private readonly handlers: ArenaSocketHandlers,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.clearTimers();
    this.handlers.onState("connecting");

    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.onopen = () => {
      this.retry = 0;
      this.handlers.onState("open");
      this.heartbeat = setInterval(() => this.send({ type: "ping" }), HEARTBEAT_MS);
    };

    ws.onmessage = (frame) => {
      try {
        // Unknown frame shapes are tolerated here (cast) and dropped by the
        // reducer's default case — one bad packet never kills the feed.
        this.handlers.onEvent(JSON.parse(frame.data as string) as ArenaEvent);
      } catch {
        // non-JSON frame — ignore
      }
    };

    ws.onclose = () => {
      this.clearTimers();
      this.handlers.onState("closed");
      if (!this.closedByUser) this.scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose always follows onerror; reconnect logic lives there
      ws.close();
    };
  }

  send(msg: object): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
      return true;
    }
    return false;
  }

  close(): void {
    this.closedByUser = true;
    this.clearTimers();
    this.ws?.close();
    this.ws = null;
    this.handlers.onState("idle");
  }

  private scheduleReconnect(): void {
    const delay = Math.min(BASE_BACKOFF_MS * 2 ** this.retry, MAX_BACKOFF_MS);
    this.retry += 1;
    this.reconnectTimer = setTimeout(() => this.connect(), delay + Math.random() * 250);
  }

  private clearTimers(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.heartbeat) clearInterval(this.heartbeat);
    this.reconnectTimer = null;
    this.heartbeat = null;
  }
}

export function backendHttpUrl(): string {
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
}

export function backendWsUrl(): string {
  return process.env.NEXT_PUBLIC_BACKEND_WS_URL ?? "ws://localhost:8000/ws";
}
