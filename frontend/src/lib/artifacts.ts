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

// --------------------------------------------------------------------------- //
// Bootstrap CI shape — shared by nearly every measured field.
// --------------------------------------------------------------------------- //

export interface Interval {
  mean: number;
  lo: number;
  hi: number;
  n: number;
  method?: string;
}

/** Format an Interval as "mean (lo–hi)" in the caller's unit. Mono-rendered. */
export function fmtInterval(i: Interval | null | undefined, unit = ""): string | null {
  if (!i) return null;
  const f = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
  return `${f(i.mean)}${unit} (${f(i.lo)}–${f(i.hi)})`;
}

/** Format a bare number, or null when the value is absent. */
export function fmtNum(v: number | null | undefined, digits = 3): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return v.toFixed(digits);
}

/** Signed percentage-point delta, e.g. "-35.8 pts". Null when absent. */
export function fmtDeltaPts(v: number | null | undefined): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  const sign = v > 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(1)} pts`;
}

/** Absolute percentage, e.g. "48.3%". Null when absent. */
export function fmtPct(v: number | null | undefined, digits = 1): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return `${(v * 100).toFixed(digits)}%`;
}

/** Strip the ISO timestamp to its date for compact display. */
export function fmtDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  return iso.slice(0, 10);
}

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
  /** Artifacts present and parseable in the snapshot. Null when none. */
  artifactCount: number | null;
  /** Git SHA of the newest artifact's provenance. Null when none. */
  gitSha: string | null;
  /** generated_at of the newest artifact. Null when none. */
  generatedAt: string | null;
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

  for (const name of ALL_ARTIFACTS) {
    const doc = (await readArtifact(name)) as MinimalProvenance | null;
    if (!doc || typeof doc !== "object" || !doc.provenance) continue;
    count += 1;
    const at = doc.provenance.generated_at;
    if (typeof at === "string" && (newestAt === null || at > newestAt)) {
      newestAt = at;
      const sha = doc.provenance.git_sha;
      newestSha = typeof sha === "string" ? sha : null;
    }
  }

  return {
    artifactCount: count > 0 ? count : null,
    gitSha: newestSha,
    generatedAt: newestAt,
  };
}
