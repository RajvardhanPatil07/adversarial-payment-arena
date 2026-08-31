/**
 * Hand-rolled per-artifact validators (SECTION 1D / PHASE 3).
 *
 * No schema library (SECTION 1: no new dependencies), no `any`, no
 * @ts-ignore: each validator walks the exact fields the UI consumes and
 * collects dot-paths for everything missing. A validated artifact is
 * returned as the typed interface; a failed one returns the missing-field
 * list so the page can render an explicit "artifact unavailable" state.
 *
 * The validators deliberately tolerate EXTRA fields (the JSON files are far
 * richer than the interfaces) — they police presence and shape, not absence.
 */

import { isInterval, type Interval } from "./format";

// --------------------------------------------------------------------------- //
// The discriminated result every loader call returns.
// --------------------------------------------------------------------------- //

export type Validated<T> = { ok: true; data: T } | { ok: false; missing: string[] };

// --------------------------------------------------------------------------- //
// Tiny validator toolkit.
// --------------------------------------------------------------------------- //

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isArr = (v: unknown): v is unknown[] => Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && !Number.isNaN(v);

/** Collects missing dot-paths; ok when empty. */
class Missing {
  readonly paths: string[] = [];
  add(path: string): void {
    this.paths.push(path);
  }
  get ok(): boolean {
    return this.paths.length === 0;
  }
}

/** Dot-path lookup. Returns undefined (never throws) for absent segments. */
function at(root: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((cur, seg) => (isObj(cur) ? cur[seg] : undefined), root);
}

function obj(m: Missing, root: unknown, path: string): Obj | null {
  const v = at(root, path);
  if (isObj(v)) return v;
  m.add(path);
  return null;
}

function arr(m: Missing, root: unknown, path: string): unknown[] | null {
  const v = at(root, path);
  if (isArr(v)) return v;
  m.add(path);
  return null;
}

function num(m: Missing, root: unknown, path: string): number | null {
  const v = at(root, path);
  if (isNum(v)) return v;
  m.add(path);
  return null;
}

function str(m: Missing, root: unknown, path: string): string | null {
  const v = at(root, path);
  if (typeof v === "string") return v;
  m.add(path);
  return null;
}

function bool(m: Missing, root: unknown, path: string): boolean | null {
  const v = at(root, path);
  if (typeof v === "boolean") return v;
  m.add(path);
  return null;
}

function strArr(m: Missing, root: unknown, path: string): string[] | null {
  const v = at(root, path);
  if (isArr(v) && v.every((s) => typeof s === "string")) return v as string[];
  m.add(path);
  return null;
}

function interval(m: Missing, root: unknown, path: string): Interval | null {
  const v = at(root, path);
  if (isInterval(v)) return v;
  m.add(path);
  return null;
}

/** Element helper: check one key on an array element, reporting `label.key`. */
function elStr(m: Missing, el: unknown, key: string, label: string): string | null {
  const v = isObj(el) ? el[key] : undefined;
  if (typeof v === "string") return v;
  m.add(`${label}.${key}`);
  return null;
}

/** Element helper: check one interval key on an array/record element, reporting `label.key`. */
function elInterval(m: Missing, el: unknown, key: string, label: string): Interval | null {
  const v = isObj(el) ? el[key] : undefined;
  if (isInterval(v)) return v;
  m.add(`${label}.${key}`);
  return null;
}

function elNum(m: Missing, el: unknown, key: string, label: string): number | null {
  const v = isObj(el) ? el[key] : undefined;
  if (isNum(v)) return v;
  m.add(`${label}.${key}`);
  return null;
}

function elBool(m: Missing, el: unknown, key: string, label: string): boolean | null {
  const v = isObj(el) ? el[key] : undefined;
  if (typeof v === "boolean") return v;
  m.add(`${label}.${key}`);
  return null;
}

// --------------------------------------------------------------------------- //
// Shared shapes.
// --------------------------------------------------------------------------- //

/** The lite contract every artifact satisfies at minimum. */
export interface LiteArtifact {
  provenance: { generated_at: string; git_sha: string };
  boundaries: string[];
}

/** Provenance shape checked on EVERY artifact. */
function checkProvenance(m: Missing, doc: unknown, label: string): void {
  if (str(m, doc, "provenance.generated_at") === null) m.add(`${label}.provenance`);
  str(m, doc, "provenance.git_sha");
}

function validateLite(doc: unknown): Validated<LiteArtifact> {
  const m = new Missing();
  checkProvenance(m, doc, "artifact");
  strArr(m, doc, "boundaries");
  return m.ok ? { ok: true, data: doc as LiteArtifact } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// closed_loop.json
// --------------------------------------------------------------------------- //

export interface ClosedLoopDoc {
  provenance: { generated_at: string; git_sha: string; seeds: number[] };
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
  per_seed: Array<{
    seed: number;
    arms: Array<{
      arm: string;
      gated: boolean;
      generator: string;
      generations: Array<{
        generation: string;
        recall_on_real_fraud: number;
        recall_on_synthetic_attacks: number;
      }>;
    }>;
  }>;
  boundaries: string[];
}

export function validateClosedLoop(doc: unknown): Validated<ClosedLoopDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "closed_loop");
  const seeds = arr(m, doc, "provenance.seeds");
  if (seeds && !seeds.every(isNum)) m.add("provenance.seeds");

  num(m, doc, "gate.c2st_auc_max");
  num(m, doc, "gate.dependence_frobenius_max");
  bool(m, doc, "gate.fixed_in_advance");
  str(m, doc, "gate.labels_required");
  bool(m, doc, "gate.computable_before_retraining");
  str(m, doc, "gate.why");

  const arms = arr(m, doc, "headline.arms");
  if (arms) {
    if (arms.length === 0) m.add("headline.arms (empty)");
    arms.forEach((el, i) => {
      const label = `headline.arms[${i}]`;
      elStr(m, el, "arm", label);
      elBool(m, el, "gated", label);
      elStr(m, el, "generator", label);
      elNum(m, el, "delta_real_recall", label);
      elNum(m, el, "delta_synthetic_recall", label);
      const ci = isObj(el) ? el["delta_real_recall_ci"] : undefined;
      if (!(isArr(ci) && ci.length === 2 && ci.every(isNum))) m.add(`${label}.delta_real_recall_ci`);
      const rej = isObj(el) ? el["batches_rejected_by_gate"] : undefined;
      if (!(isArr(rej) && rej.every(isNum))) m.add(`${label}.batches_rejected_by_gate`);
    });
  }

  num(m, doc, "headline.low_fidelity_generator.ungated_delta_real_recall");
  num(m, doc, "headline.low_fidelity_generator.gated_delta_real_recall");
  num(m, doc, "headline.low_fidelity_generator.recall_protected_by_gate");
  num(m, doc, "headline.low_fidelity_generator.ungated_delta_synthetic_recall");
  num(m, doc, "headline.high_fidelity_generator.ungated_delta_real_recall");
  num(m, doc, "headline.high_fidelity_generator.gated_delta_real_recall");
  num(m, doc, "headline.high_fidelity_generator.recall_protected_by_gate");
  str(m, doc, "headline.reading");

  const aggregated = obj(m, doc, "aggregated");
  if (aggregated) {
    if (Object.keys(aggregated).length === 0) m.add("aggregated (empty)");
    for (const [armName, arm] of Object.entries(aggregated)) {
      const label = `aggregated.${armName}`;
      if (!isObj(arm)) {
        m.add(label);
        continue;
      }
      if (typeof arm["gated"] !== "boolean") m.add(`${label}.gated`);
      if (typeof arm["generator"] !== "string") m.add(`${label}.generator`);
      interval(m, arm, "delta_real_recall");
      const byGen = arm["by_generation"];
      if (!isObj(byGen) || Object.keys(byGen).length === 0) {
        m.add(`${label}.by_generation`);
      } else {
        for (const gen of Object.keys(byGen)) {
          interval(m, byGen, `${gen}.recall_on_real_fraud`);
        }
      }
    }
  }

  const perSeed = arr(m, doc, "per_seed");
  if (perSeed) {
    perSeed.forEach((seedEl, s) => {
      const sLabel = `per_seed[${s}]`;
      elNum(m, seedEl, "seed", sLabel);
      const seedArms = isObj(seedEl) ? seedEl["arms"] : undefined;
      if (!isArr(seedArms)) {
        m.add(`${sLabel}.arms`);
        return;
      }
      seedArms.forEach((armEl, a) => {
        const aLabel = `${sLabel}.arms[${a}]`;
        elStr(m, armEl, "arm", aLabel);
        elBool(m, armEl, "gated", aLabel);
        elStr(m, armEl, "generator", aLabel);
        const gens = isObj(armEl) ? armEl["generations"] : undefined;
        if (!isArr(gens)) {
          m.add(`${aLabel}.generations`);
          return;
        }
        gens.forEach((genEl, g) => {
          const gLabel = `${aLabel}.generations[${g}]`;
          elStr(m, genEl, "generation", gLabel);
          elNum(m, genEl, "recall_on_real_fraud", gLabel);
          elNum(m, genEl, "recall_on_synthetic_attacks", gLabel);
        });
      });
    });
  }

  strArr(m, doc, "boundaries");
  return m.ok ? { ok: true, data: doc as ClosedLoopDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// fidelity_report.json
// --------------------------------------------------------------------------- //

export interface FidelityReportDoc {
  provenance: { generated_at: string; git_sha: string };
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

export function validateFidelityReport(doc: unknown): Validated<FidelityReportDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "fidelity_report");
  strArr(m, doc, "measures");
  str(m, doc, "acceptance_gate.metric");
  num(m, doc, "acceptance_gate.threshold");
  num(m, doc, "acceptance_gate.observed");
  bool(m, doc, "acceptance_gate.cleared");
  str(m, doc, "acceptance_gate.policy");
  const aggregated = obj(m, doc, "aggregated");
  if (aggregated) {
    for (const [gen, measures] of Object.entries(aggregated)) {
      if (!isObj(measures)) {
        m.add(`aggregated.${gen}`);
        continue;
      }
      for (const measure of Object.keys(measures)) {
        interval(m, measures, measure);
      }
    }
  }
  strArr(m, doc, "boundaries");
  return m.ok ? { ok: true, data: doc as FidelityReportDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// family_coverage.json
// --------------------------------------------------------------------------- //

export interface FamilyCoverageDoc {
  provenance: { generated_at: string; git_sha: string };
  families_defeat: Record<string, string>;
  summary: {
    executable_families: number;
    mean_recall_family_in_training: number;
    mean_recall_family_withheld_zero_day: number;
    weakest_family_when_withheld: { family: string; label: string; recall: number; defeats: string };
    reading: string;
  };
  family_in_training: Record<string, { recall: Interval; label: string; defeats: string }>;
  family_withheld_zero_day: Record<string, { recall: Interval; label: string; defeats: string }>;
  boundaries: string[];
}

export function validateFamilyCoverage(doc: unknown): Validated<FamilyCoverageDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "family_coverage");

  const defeat = obj(m, doc, "families_defeat");
  if (defeat) {
    for (const [k, v] of Object.entries(defeat)) {
      if (typeof v !== "string") m.add(`families_defeat.${k}`);
    }
  }

  num(m, doc, "summary.executable_families");
  num(m, doc, "summary.mean_recall_family_in_training");
  num(m, doc, "summary.mean_recall_family_withheld_zero_day");
  str(m, doc, "summary.weakest_family_when_withheld.family");
  str(m, doc, "summary.weakest_family_when_withheld.label");
  num(m, doc, "summary.weakest_family_when_withheld.recall");
  str(m, doc, "summary.weakest_family_when_withheld.defeats");
  str(m, doc, "summary.reading");

  const inTraining = obj(m, doc, "family_in_training");
  if (inTraining) {
    for (const [fam, v] of Object.entries(inTraining)) {
      const label = `family_in_training.${fam}`;
      if (!isObj(v)) {
        m.add(label);
        continue;
      }
      interval(m, v, "recall");
      if (typeof v["label"] !== "string") m.add(`${label}.label`);
      if (typeof v["defeats"] !== "string") m.add(`${label}.defeats`);
    }
  }

  const withheld = obj(m, doc, "family_withheld_zero_day");
  if (withheld) {
    for (const [fam, v] of Object.entries(withheld)) {
      const label = `family_withheld_zero_day.${fam}`;
      if (!isObj(v)) {
        m.add(label);
        continue;
      }
      interval(m, v, "recall");
      if (typeof v["label"] !== "string") m.add(`${label}.label`);
      if (typeof v["defeats"] !== "string") m.add(`${label}.defeats`);
    }
  }

  strArr(m, doc, "boundaries");
  return m.ok ? { ok: true, data: doc as FamilyCoverageDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// latency.json
// --------------------------------------------------------------------------- //

export interface LatencyDoc {
  provenance: { generated_at: string; git_sha: string };
  protocol: { inline_budget_ms: number; n_scored: number };
  overall: { n: number; mean_ms: number; p50_ms: number; p95_ms: number; p99_ms: number; max_ms: number };
}

export function validateLatency(doc: unknown): Validated<LatencyDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "latency");
  num(m, doc, "protocol.inline_budget_ms");
  num(m, doc, "protocol.n_scored");
  num(m, doc, "overall.n");
  num(m, doc, "overall.mean_ms");
  num(m, doc, "overall.p50_ms");
  num(m, doc, "overall.p95_ms");
  num(m, doc, "overall.p99_ms");
  num(m, doc, "overall.max_ms");
  return m.ok ? { ok: true, data: doc as LatencyDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// prevalence_metrics.json
// --------------------------------------------------------------------------- //

export interface PrevalenceDoc {
  provenance: { generated_at: string; git_sha: string };
  operating_point: { arm: string; recall: number; fpr: number };
  sweep: Array<{ prevalence: number; precision: number; missed_frauds: number }>;
  note: string;
}

export function validatePrevalence(doc: unknown): Validated<PrevalenceDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "prevalence_metrics");
  str(m, doc, "operating_point.arm");
  num(m, doc, "operating_point.recall");
  num(m, doc, "operating_point.fpr");
  const sweep = arr(m, doc, "sweep");
  if (sweep) {
    sweep.forEach((el, i) => {
      const label = `sweep[${i}]`;
      elNum(m, el, "prevalence", label);
      elNum(m, el, "precision", label);
      elNum(m, el, "missed_frauds", label);
    });
  }
  str(m, doc, "note");
  return m.ok ? { ok: true, data: doc as PrevalenceDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// calibration_audit.json
// --------------------------------------------------------------------------- //

/** One row of the FPR-budget sweep (`aggregated.fpr_budget_*`). */
export interface CalibrationBudgetRow {
  target_fpr: number;
  realised_test_fpr: Interval;
  recall_on_held_out_fraud: Interval;
  calibration_gap_pct_points: Interval;
  precision_at_production_prevalence: Interval;
}

export interface CalibrationDoc {
  provenance: { generated_at: string; git_sha: string };
  protocol: { headline_fpr: number; production_prevalence: number };
  headline: {
    recall_at_1pct_fpr: Interval;
    realised_fpr: Interval;
    calibration_gap: Interval;
  };
  /** Keyed "fpr_budget_0.001" … "fpr_budget_0.05" (extra keys tolerated). */
  aggregated: Record<string, CalibrationBudgetRow>;
  boundaries: string[];
}

export function validateCalibration(doc: unknown): Validated<CalibrationDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "calibration_audit");
  num(m, doc, "protocol.headline_fpr");
  num(m, doc, "protocol.production_prevalence");
  interval(m, doc, "headline.recall_at_1pct_fpr");
  interval(m, doc, "headline.realised_fpr");
  interval(m, doc, "headline.calibration_gap");
  strArr(m, doc, "boundaries");
  const agg = obj(m, doc, "aggregated");
  if (agg) {
    for (const [key, row] of Object.entries(agg)) {
      if (!isObj(row)) {
        m.add(`aggregated.${key}`);
        continue;
      }
      const label = `aggregated.${key}`;
      elNum(m, row, "target_fpr", label);
      elInterval(m, row, "realised_test_fpr", `${label}.realised_test_fpr`);
      elInterval(m, row, "recall_on_held_out_fraud", `${label}.recall_on_held_out_fraud`);
      elInterval(m, row, "calibration_gap_pct_points", `${label}.calibration_gap_pct_points`);
      elInterval(m, row, "precision_at_production_prevalence", `${label}.precision_at_production_prevalence`);
    }
  }
  return m.ok ? { ok: true, data: doc as CalibrationDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// metrics.json
// --------------------------------------------------------------------------- //

export interface MetricsDoc {
  provenance: { generated_at: string; git_sha: string };
  headline: {
    baseline_recall: Interval;
    delta_recall_independent_marginal: Interval;
    delta_recall_gaussian_copula: Interval;
  };
  precision_at_production_prevalence: Interval;
  net_benefit_inr_at_production_prevalence: number;
  insult_share_of_total_cost: number;
}

export function validateMetrics(doc: unknown): Validated<MetricsDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "metrics");
  interval(m, doc, "headline.baseline_recall");
  interval(m, doc, "headline.delta_recall_independent_marginal");
  interval(m, doc, "headline.delta_recall_gaussian_copula");
  interval(m, doc, "precision_at_production_prevalence");
  num(m, doc, "net_benefit_inr_at_production_prevalence");
  num(m, doc, "insult_share_of_total_cost");
  return m.ok ? { ok: true, data: doc as MetricsDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// behavioural_fidelity.json
// --------------------------------------------------------------------------- //

export interface BehaviouralFidelityDoc {
  provenance: { generated_at: string; git_sha: string };
  headline: Record<string, Record<string, Interval>>;
  delta_recall_pct_points_vs_A0: Record<string, number>;
  ordering: {
    rank_by_row_level_fidelity: string[];
    rank_by_transfer: string[];
    ordering_inversion_detected: boolean;
  };
  boundaries: string[];
}

export function validateBehaviouralFidelity(doc: unknown): Validated<BehaviouralFidelityDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "behavioural_fidelity");
  const headline = obj(m, doc, "headline");
  if (headline && Object.keys(headline).length === 0) m.add("headline (empty)");
  const deltas = obj(m, doc, "delta_recall_pct_points_vs_A0");
  if (deltas) {
    for (const [k, v] of Object.entries(deltas)) {
      if (!isNum(v)) m.add(`delta_recall_pct_points_vs_A0.${k}`);
    }
  }
  strArr(m, doc, "ordering.rank_by_row_level_fidelity");
  strArr(m, doc, "ordering.rank_by_transfer");
  bool(m, doc, "ordering.ordering_inversion_detected");
  strArr(m, doc, "boundaries");
  return m.ok ? { ok: true, data: doc as BehaviouralFidelityDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// claim_ledger.json
// --------------------------------------------------------------------------- //

export interface ClaimLedgerDoc {
  provenance: { generated_at: string; git_sha: string };
  claim_count: number;
  claims: Array<{
    claim: string;
    artifact: string;
    field: string;
    derivation: string;
    boundary: string;
  }>;
}

export function validateClaimLedger(doc: unknown): Validated<ClaimLedgerDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "claim_ledger");
  num(m, doc, "claim_count");
  const claims = arr(m, doc, "claims");
  if (claims) {
    claims.forEach((el, i) => {
      const label = `claims[${i}]`;
      elStr(m, el, "claim", label);
      elStr(m, el, "artifact", label);
      elStr(m, el, "field", label);
      elStr(m, el, "derivation", label);
      elStr(m, el, "boundary", label);
    });
  }
  return m.ok ? { ok: true, data: doc as ClaimLedgerDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// economics.json
// --------------------------------------------------------------------------- //

export interface EconomicsDoc {
  provenance: { generated_at: string; git_sha: string };
  cost_model: {
    derived: {
      insult_cost_per_false_positive_inr: number;
      loss_per_false_negative_inr: number;
      value_per_true_positive_inr: number;
    };
  };
  at_production_prevalence: {
    counts: { tp: number; fp: number; fn: number; tn: number };
    fraud_prevented_inr: number;
    fraud_lost_inr: number;
    insult_cost_inr: number;
    review_cost_inr: number;
    net_benefit_inr: number;
    insult_share_of_total_cost: number;
    operating_point: { recall: number; fpr: number; prevalence: number; volume: number };
  };
}

export function validateEconomics(doc: unknown): Validated<EconomicsDoc> {
  const m = new Missing();
  checkProvenance(m, doc, "economics");
  num(m, doc, "cost_model.derived.insult_cost_per_false_positive_inr");
  num(m, doc, "cost_model.derived.loss_per_false_negative_inr");
  num(m, doc, "cost_model.derived.value_per_true_positive_inr");
  num(m, doc, "at_production_prevalence.counts.tp");
  num(m, doc, "at_production_prevalence.counts.fp");
  num(m, doc, "at_production_prevalence.counts.fn");
  num(m, doc, "at_production_prevalence.counts.tn");
  num(m, doc, "at_production_prevalence.fraud_prevented_inr");
  num(m, doc, "at_production_prevalence.fraud_lost_inr");
  num(m, doc, "at_production_prevalence.insult_cost_inr");
  num(m, doc, "at_production_prevalence.review_cost_inr");
  num(m, doc, "at_production_prevalence.net_benefit_inr");
  num(m, doc, "at_production_prevalence.insult_share_of_total_cost");
  num(m, doc, "at_production_prevalence.operating_point.recall");
  num(m, doc, "at_production_prevalence.operating_point.fpr");
  num(m, doc, "at_production_prevalence.operating_point.prevalence");
  num(m, doc, "at_production_prevalence.operating_point.volume");
  return m.ok ? { ok: true, data: doc as EconomicsDoc } : { ok: false, missing: m.paths };
}

// --------------------------------------------------------------------------- //
// The registry: every artifact name -> its validator.
// --------------------------------------------------------------------------- //

export const VALIDATORS = {
  action_policy: validateLite,
  behavioural_fidelity: validateBehaviouralFidelity,
  calibration_audit: validateCalibration,
  claim_ledger: validateClaimLedger,
  closed_loop: validateClosedLoop,
  economics: validateEconomics,
  family_coverage: validateFamilyCoverage,
  fidelity_report: validateFidelityReport,
  latency: validateLatency,
  metrics: validateMetrics,
  prevalence_metrics: validatePrevalence,
  privacy_audit: validateLite,
  transfer_ledger: validateLite,
} as const;
