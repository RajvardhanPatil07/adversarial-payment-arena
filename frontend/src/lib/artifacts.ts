/**
 * The artifact loader — the only bridge between generated evidence and the UI.
 *
 * Every measured number rendered anywhere in this application is read through
 * this module, at render time, from the JSON files `make reproduce` emits into
 * `artifacts/`. Nothing is hardcoded, defaulted or estimated: if an artifact is
 * missing or a field is absent, the loader returns null and the page renders an
 * explicit "not measured" state — a missing number must be indistinguishable
 * from an honestly-absent one, never from a fabricated one.
 *
 * Both execution modes read from the same directory:
 *   - `next dev`: cwd is frontend/, so the repo root is one level up.
 *   - static export / container: the Dockerfile copies `artifacts/` to /app
 *     and the UI build runs with cwd /ui, so the repo root is /app. The env
 *     override exists for that layout; the default works for dev and CI.
 *
 * Server-only by construction (node:fs). Pages using it are server components;
 * the one interactive page (/arena) does not import this module.
 */

/**
 * Formatters and the Interval shape live in the client-safe `format.ts` and are
 * re-exported here so server pages keep one import site. Client components MUST
 * import from `@/lib/format` directly — this module is server-only (node:fs).
 */
export {
  fmtInterval,
  fmtNum,
  fmtDeltaPts,
  fmtPct,
  fmtInr,
  fmtDate,
  isInterval,
  type Interval,
} from "./format";
import { VALIDATORS } from "./validators";
import type { Interval } from "./format";

import { promises as fs } from "node:fs";
import path from "node:path";

/** Artifact filename union — the full evidence set, named at the type level. */
export type ArtifactName =
  | "action_policy"
  | "behavioural_fidelity"
  | "calibration_audit"
  | "claim_ledger"
  | "closed_loop"
  | "economics"
  | "family_coverage"
  | "fidelity_report"
  | "latency"
  | "metrics"
  | "prevalence_metrics"
  | "privacy_audit"
  | "transfer_ledger";

/** The directory the artifacts live in, overridable for the container build. */
function artifactRoot(): string {
  return process.env.ARTIFACTS_DIR ?? path.resolve(process.cwd(), "..", "artifacts");
}

/** Cache: a page's render reads several artifacts; a build reads each once. */
const cache = new Map<ArtifactName, unknown>();

/**
 * Read one artifact as untyped JSON. Returns null (never throws) when the
 * file is missing or unparseable, so a not-yet-generated evidence set degrades
 * into explicit "not measured" states rather than a crashed page.
 */
export async function readArtifact(name: ArtifactName): Promise<unknown> {
  const hit = cache.get(name);
  if (hit !== undefined) return hit;

  try {
    const raw = await fs.readFile(path.join(artifactRoot(), `${name}.json`), "utf8");
    const parsed: unknown = JSON.parse(raw);
    cache.set(name, parsed);
    return parsed;
  } catch {
    return null;
  }
}

/** Read one artifact with a per-artifact schema type. Null when absent. */
export function readTyped<T>(name: ArtifactName): Promise<T | null> {
  return readArtifact(name) as Promise<T | null>;
}

/** The validated document type each artifact name maps to. */
export type ValidatedDoc<N extends ArtifactName> = ReturnType<(typeof VALIDATORS)[N]>;

type AnyValidated = { ok: true; data: unknown } | { ok: false; missing: string[] };

/**
 * The PHASE 3 loader: read + validate + discriminate, in one call.
 *
 * Returns `{ok, data}` only when the per-artifact validator found every field
 * the UI consumes; otherwise `{ok: false, missing}` listing the dot-paths, so
 * the page can render an explicit "artifact unavailable" state (SECTION 1D)
 * naming exactly what is absent — a missing number stays discoverable, never
 * silently defaulted. The validator result is cached with the document.
 */
export async function loadArtifact<N extends ArtifactName>(
  name: N,
): Promise<Extract<ValidatedDoc<N>, { ok: true }> | { ok: false; missing: string[] }> {
  const doc = await readArtifact(name);
  if (doc === null) {
    return { ok: false, missing: [`${name}.json (file missing or unparseable)`] };
  }
  const result = VALIDATORS[name](doc) as AnyValidated;
  return result.ok
    ? { ok: true, data: result.data } as Extract<ValidatedDoc<N>, { ok: true }>
    : { ok: false, missing: result.missing };
}

/**
 * The manifest validation pass: every artifact, validated, with its missing
 * paths. PHASE 3 requires proving the pipeline by printing which artifacts
 * validated and which fields are missing; this is the single function that
 * produces that proof, and the root layout reuses it for the Nav chip so the
 * chip means "validated", not merely "present".
 */
export async function validateManifest(): Promise<
  Array<{ name: ArtifactName; ok: boolean; missing: string[] }>
> {
  const out: Array<{ name: ArtifactName; ok: boolean; missing: string[] }> = [];
  for (const name of ALL_ARTIFACTS) {
    const r = await loadArtifact(name);
    out.push({ name, ok: r.ok, missing: r.ok ? [] : r.missing });
  }
  return out;
}

// --------------------------------------------------------------------------- //
// Formatters + Interval: see ./format (re-exported above).
// --------------------------------------------------------------------------- //
// Typed views over the artifacts the evidence pages consume.
// --------------------------------------------------------------------------- //

export interface Provenance {
  schema_version: string;
  generated_at: string;
  git_sha: string;
  command: string;
  seeds: number[];
}

export interface ClosedLoopArtifact {
  provenance: Provenance;
  gate: {
    c2st_auc_max: number;
    dependence_frobenius_max: number;
    fixed_in_advance: boolean;
    labels_required: string;
    computable_before_retraining: boolean;
    why: string;
  };
  headline: {
    arms: Array<{
      arm: string;
      gated: boolean;
      generator: string;
      delta_real_recall: number;
      delta_real_recall_ci: [number, number];
      delta_synthetic_recall: number;
      batches_rejected_by_gate: number[];
    }>;
    low_fidelity_generator: {
      ungated_delta_real_recall: number;
      gated_delta_real_recall: number;
      recall_protected_by_gate: number;
      ungated_delta_synthetic_recall: number;
    };
    high_fidelity_generator: {
      ungated_delta_real_recall: number;
      gated_delta_real_recall: number;
      recall_protected_by_gate: number;
    };
    reading: string;
  };
  aggregated: Record<
    string,
    {
      gated: boolean;
      generator: string;
      by_generation: Record<string, { recall_on_real_fraud: Interval }>;
      delta_real_recall: Interval;
    }
  >;
  boundaries: string[];
}

export interface FidelityReportArtifact {
  provenance: Provenance;
  measures: string[];
  acceptance_gate: {
    metric: string;
    threshold: number;
    observed: number;
    cleared: boolean;
    policy: string;
  };
  aggregated: Record<string, Record<string, Interval>>;
  boundaries: string[];
}

export interface FamilyCoverageArtifact {
  provenance: Provenance;
  families_defeat: Record<string, string>;
  summary: {
    executable_families: number;
    mean_recall_family_in_training: number;
    mean_recall_family_withheld_zero_day: number;
    weakest_family_when_withheld: {
      family: string;
      label: string;
      recall: number;
      defeats: string;
    };
    reading: string;
  };
  family_in_training: Record<
    string,
    { recall: Interval; label: string; defeats: string }
  >;
  family_withheld_zero_day: Record<
    string,
    {
      recall: Interval;
      label: string;
      defeats: string;
      layer_hits: Record<string, Interval>;
    }
  >;
  boundaries: string[];
}

export interface LatencyArtifact {
  provenance: Provenance;
  protocol: { inline_budget_ms: number; n_scored: number };
  overall: {
    n: number;
    mean_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    max_ms: number;
  };
}

export interface PrevalenceArtifact {
  provenance: Provenance;
  operating_point: { arm: string; recall: number; fpr: number };
  sweep: Array<{
    prevalence: number;
    precision: number;
    missed_frauds: number;
  }>;
  note: string;
}

export interface CalibrationArtifact {
  provenance: Provenance;
  headline: {
    recall_at_1pct_fpr: Interval;
    realised_fpr: Interval;
    calibration_gap: Interval;
  };
  boundaries: string[];
}

export interface MetricsArtifact {
  provenance: Provenance;
  headline: {
    baseline_recall: Interval;
    delta_recall_independent_marginal: Interval;
    delta_recall_gaussian_copula: Interval;
  };
  precision_at_production_prevalence: Interval;
  net_benefit_inr_at_production_prevalence: number;
  insult_share_of_total_cost: number;
}

export interface BehaviouralFidelityArtifact {
  provenance: Provenance;
  headline: Record<string, Record<string, Interval>>;
  delta_recall_pct_points_vs_A0: Record<string, number>;
  fidelity_composites: Record<string, Record<string, number>>;
  ordering: {
    rank_by_row_level_fidelity: string[];
    rank_by_transfer: string[];
    ordering_inversion_detected: boolean;
  };
  boundaries: string[];
}

// --------------------------------------------------------------------------- //
// Site-wide provenance — feeds the Nav chip and the Footer stamp.
// --------------------------------------------------------------------------- //

export interface SiteProvenance {
  /** Artifacts present and VALIDATED (PHASE 3 manifest pass). Null when none. */
  artifactCount: number | null;
  /** Git SHA of the newest artifact's provenance. Null when none. */
  gitSha: string | null;
  /** generated_at of the newest artifact. Null when none. */
  generatedAt: string | null;
  /** Seeds common to the measured artifacts, for the footer stamp. Null when absent. */
  seeds: number[] | null;
}

const ALL_ARTIFACTS: readonly ArtifactName[] = [
  "action_policy",
  "behavioural_fidelity",
  "calibration_audit",
  "claim_ledger",
  "closed_loop",
  "economics",
  "family_coverage",
  "fidelity_report",
  "latency",
  "metrics",
  "prevalence_metrics",
  "privacy_audit",
  "transfer_ledger",
];

interface MinimalProvenance {
  provenance?: { generated_at?: string; git_sha?: string };
}

/**
 * One pass over the evidence set: counts validated artifacts and takes the
 * newest generated_at and its git SHA. Called once by the root layout so the
 * Nav chip and the Footer stamp agree with each other by construction.
 */
export async function readSiteProvenance(): Promise<SiteProvenance> {
  let count = 0;
  let newestAt: string | null = null;
  let newestSha: string | null = null;
  let seeds: number[] | null = null;

  for (const name of ALL_ARTIFACTS) {
    // PHASE 3: the chip counts VALIDATED artifacts, not merely present ones.
    const r = await loadArtifact(name);
    if (!r.ok) continue;
    count += 1;
    const doc = r.data as unknown as MinimalProvenance & { provenance?: { seeds?: unknown } };
    const at = doc.provenance?.generated_at;
    if (typeof at === "string" && (newestAt === null || at > newestAt)) {
      newestAt = at;
      const sha = doc.provenance?.git_sha;
      newestSha = typeof sha === "string" ? sha : null;
    }
    // Seeds for the footer stamp: keep the first non-empty list. Some artifacts
    // record an empty seed list by design (e.g. the claim ledger); a union of
    // differing lists would be an invented number, so the first wins.
    const docSeeds = doc.provenance?.seeds;
    if (
      seeds === null &&
      Array.isArray(docSeeds) &&
      docSeeds.length > 0 &&
      docSeeds.every((s) => typeof s === "number")
    ) {
      seeds = docSeeds as number[];
    }
  }

  return {
    artifactCount: count > 0 ? count : null,
    gitSha: newestSha,
    generatedAt: newestAt,
    seeds,
  };
}
