/**
 * Single source of truth for backend URL resolution.
 *
 * Why this file exists
 * --------------------
 * Three separate call sites had each grown their own env-var convention
 * (NEXT_PUBLIC_BACKEND_URL, NEXT_PUBLIC_API_BASE_URL, NEXT_PUBLIC_API_URL),
 * with a hardcoded `ws://localhost:8000/ws` fallback for the socket. That is
 * fine on a laptop and broken the moment the prototype is deployed:
 *
 *   1. An operator setting one variable silently leaves the others on
 *      localhost, so half the UI works and half shows a connection error --
 *      the worst possible failure mode for a live demo.
 *   2. `ws://` from an `https://` page is mixed content. Browsers block it
 *      outright, with no fallback and a console-only error.
 *
 * So: one variable, one derivation, scheme inferred from the page.
 *
 * Configuration
 * -------------
 * Set NEXT_PUBLIC_BACKEND_URL to the backend origin, e.g.
 *     NEXT_PUBLIC_BACKEND_URL=https://arena-api.example.com
 * The WebSocket URL is derived from it (https -> wss, http -> ws), so it can
 * never drift out of sync with the HTTP origin. Override it only if the socket
 * genuinely lives elsewhere, via NEXT_PUBLIC_BACKEND_WS_URL.
 *
 * The legacy names are still read, so existing deployments keep working, but
 * they are deprecated and resolve through this one function.
 */

const DEFAULT_LOCAL = "http://localhost:8000";

/** Strip any trailing slashes so callers can always append "/path". */
function normalise(origin: string): string {
  return origin.replace(/\/+$/, "");
}

/**
 * The backend HTTP origin.
 *
 * Resolution order: canonical name, then the two legacy aliases, then the
 * local-dev default.
 */
export function backendHttpUrl(): string {
  const configured =
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    process.env.NEXT_PUBLIC_API_URL;

  if (configured && configured.trim()) return normalise(configured.trim());

  // No configuration. In the browser, prefer the page's own origin over
  // localhost when we are clearly not on a dev host: a deployed bundle that
  // falls back to localhost produces a confusing "backend down" state, whereas
  // same-origin at least works behind a reverse proxy or rewrite.
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    const isLocal = host === "localhost" || host === "127.0.0.1" || host === "[::1]";
    if (!isLocal) return normalise(window.location.origin);
  }
  return DEFAULT_LOCAL;
}

/**
 * The backend WebSocket URL, derived from the HTTP origin so the scheme can
 * never be mismatched (https page + ws:// socket = blocked mixed content).
 */
export function backendWsUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_BACKEND_WS_URL;
  if (explicit && explicit.trim()) return normalise(explicit.trim());

  const http = backendHttpUrl();
  const ws = http.startsWith("https://")
    ? http.replace(/^https:\/\//, "wss://")
    : http.replace(/^http:\/\//, "ws://");
  return `${ws}/ws`;
}

/** Join the backend origin with an API path. */
export function backendUrl(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${backendHttpUrl()}${suffix}`;
}
