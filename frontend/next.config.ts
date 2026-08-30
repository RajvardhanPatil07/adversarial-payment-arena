import type { NextConfig } from "next";

/**
 * Two build modes, one config.
 *
 * DEV / default: a normal Next server. The /api/analyst route streams LLM
 * narration, which needs a server runtime, so this mode keeps full capability
 * for local development.
 *
 * STATIC_EXPORT=1: emits a static bundle to `out/`, which the Docker image
 * hands to FastAPI. This is how the deployed prototype ships -- one origin
 * serving UI, REST and the WebSocket, so there is no CORS surface and no
 * https-page/ws-socket mixed-content trap.
 *
 * The cost of the export mode is explicit: route handlers under src/app/api
 * cannot be prerendered, so the analyst narration is unavailable in the
 * container. That is an acceptable trade -- it is a commentary flourish, while
 * the WebSocket arena and the evidence endpoints are the substance, and both
 * are served by FastAPI. The UI degrades with a visible notice rather than
 * silently failing.
 */
const isStaticExport = process.env.STATIC_EXPORT === "1";

const nextConfig: NextConfig = {
  ...(isStaticExport
    ? {
        output: "export" as const,
        // FastAPI's StaticFiles(html=True) resolves /evidence -> evidence/index.html,
        // so directory-style output is what the server expects.
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),
  // Dev-only: the sandbox proxies dev traffic through its own hostname, and
  // Next blocks cross-origin dev-resource requests by default. Allowed here
  // once rather than re-echoed as a warning on every request.
  allowedDevOrigins: [
    "3000-ivxqw04vwssako2dtg5d0-d0b9e1e2.sandbox.novita.ai",
    "3000-ivxqw04vwssako2dtg5d0-d0b9e1e2.e2b.dev",
  ],
  // Surface the mode to the client so the UI can explain a disabled feature
  // instead of showing an unexplained error.
  env: {
    NEXT_PUBLIC_STATIC_EXPORT: isStaticExport ? "1" : "0",
  },
};

export default nextConfig;
