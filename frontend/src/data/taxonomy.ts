/**
 * The 22-taxon attack taxonomy — static PROSE facts only.
 *
 * Transcribed from docs/ATTACK_TAXONOMY.md (read-only; not copied at runtime so
 * the frontend has no dependency outside its own tree). This file contains NO
 * measurements: every number on the atlas is read from
 * artifacts/family_coverage.json at render time and joined onto these entries
 * by the identify page. A [SPEC] taxon carries no family key at all, so it can
 * never accidentally pick up another family's measurement.
 *
 * Status legend: executable = [CODE] spec in backend/attack_specs/ — the 14
 * families that are generated and measured; the other 8 are [SPEC] mapped.
 */

/** The five organising rails; also the filter chips on the atlas. */
export type TaxonRail =
  | "issuing"
  | "acquiring"
  | "real-time rails"
  | "agentic commerce"
  | "post-authorisation"
  | "model layer";

export const RAILS: readonly TaxonRail[] = [
  "issuing",
  "acquiring",
  "real-time rails",
  "agentic commerce",
  "post-authorisation",
  "model layer",
];

export interface Taxon {
  id: string;
  title: string;
  /** Payment channel / surface, for the channel filter chips. */
  channel: string;
  /** Rail or surface attacked, for the rail filter chips. */
  rail: TaxonRail;
  /** What GenAI changed about this fraud — the organising principle. */
  genaiEnabler: string;
  /** The defensive signal the attack is designed to defeat (taxonomy prose). */
  defeats: string;
  /** [CODE] executable + measured, or [SPEC] mapped only. */
  executable: boolean;
  /** The family key in family_coverage.json, for executable taxa only. */
  family?: string;
}

export const TAXONOMY: readonly Taxon[] = [
  // ---- Layer 1 — identity and onboarding ----
  {
    id: "T-01",
    title: "Synthetic identity bust-out",
    channel: "account onboarding",
    rail: "issuing",
    genaiEnabler: "Generated coherent identity histories that survive KYC document checks and age gracefully before the bust",
    defeats: "history-length and account-age heuristics",
    executable: true,
    family: "ATTACK_2_SYNTHETIC_MULE_RING",
  },
  {
    id: "T-02",
    title: "Deepfake liveness injection",
    channel: "account onboarding",
    rail: "issuing",
    genaiEnabler: "Video and voice synthesis defeats selfie-liveness at onboarding and at step-up",
    defeats: "biometric step-up as a terminal control",
    executable: false,
  },
  {
    id: "T-03",
    title: "Voice-clone MFA reset",
    channel: "call-centre / MFA",
    rail: "issuing",
    genaiEnabler: "Cloned cardholder voice passes call-centre verification, resetting credentials before any transaction",
    defeats: "device-binding and step-up trust",
    executable: true,
    family: "ATTACK_1_MFA_RESET_VOICE_CLONE",
  },
  {
    id: "T-04",
    title: "AI-assisted document forgery for merchant onboarding",
    channel: "merchant onboarding",
    rail: "acquiring",
    genaiEnabler: "Generated registration and bank-proof documents onboard a fake merchant fast",
    defeats: "merchant vetting at acquiring",
    executable: true,
    family: "ATTACK_13_MERCHANT_BUSTOUT",
  },
  // ---- Layer 2 — authentication and authorisation ----
  {
    id: "T-05",
    title: "OTP-relay vishing at scale",
    channel: "3DS / OTP",
    rail: "issuing",
    genaiEnabler: "Conversational agents run thousands of simultaneous, personalised OTP-extraction calls",
    defeats: "treating a passed 3DS challenge as proof of cardholder presence",
    executable: true,
    family: "ATTACK_9_OTP_RELAY_VISHING",
  },
  {
    id: "T-06",
    title: "3DS frictionless-flow abuse",
    channel: "3DS frictionless",
    rail: "issuing",
    genaiEnabler: "Attacks are shaped to stay inside the risk-based-authentication exemption band",
    defeats: "threshold-based step-up policy",
    executable: true,
    family: "ATTACK_10_EXEMPTION_BAND_ABUSE",
  },
  {
    id: "T-07",
    title: "Session hijack and drain",
    channel: "card-not-present",
    rail: "issuing",
    genaiEnabler: "Automated post-hijack behaviour that imitates the victim's own transaction rhythm",
    defeats: "device-consistency signals",
    executable: false,
  },
  // ---- Layer 3 — card-not-present and transaction shaping ----
  {
    id: "T-08",
    title: "Adaptive card-testing swarm",
    channel: "card-not-present",
    rail: "issuing",
    genaiEnabler: "The agent reads decline reason codes and rewrites amount, MCC and cadence policy between attempts",
    defeats: "fixed velocity rules",
    executable: true,
    family: "ATTACK_4_CNP_HIGH_VELOCITY",
  },
  {
    id: "T-09",
    title: "Amount structuring below review thresholds",
    channel: "card-not-present",
    rail: "issuing",
    genaiEnabler: "Learned, per-issuer estimation of the review threshold rather than a guessed round number",
    defeats: "static amount thresholds",
    executable: true,
    family: "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING",
  },
  {
    id: "T-10",
    title: "MCC laundering",
    channel: "card-not-present",
    rail: "issuing",
    genaiEnabler: "Category selection optimised against the issuer's own observed decline surface",
    defeats: "MCC risk weighting",
    executable: false,
  },
  {
    id: "T-11",
    title: "Geo-velocity spoof with plausible travel",
    channel: "card-not-present",
    rail: "issuing",
    genaiEnabler: "Generated itineraries that make impossible travel look like a real trip",
    defeats: "impossible-travel rules",
    executable: true,
    family: "ATTACK_12_GEO_VELOCITY_ITINERARY",
  },
  // ---- Layer 4 — real-time rails and UPI-era patterns (India-first) ----
  {
    id: "T-12",
    title: "AI-personalised APP scam (authorised push payment)",
    channel: "UPI / instant rails",
    rail: "real-time rails",
    genaiEnabler: "Scam narratives personalised from scraped public data; the victim authorises the payment themselves",
    defeats: "every control premised on the cardholder being the victim of a *stolen* credential",
    executable: true,
    family: "ATTACK_5_APP_SCAM_PERSONALISED",
  },
  {
    id: "T-13",
    title: "Digital-arrest / impersonation coercion",
    channel: "UPI / instant rails",
    rail: "real-time rails",
    genaiEnabler: "Synthetic authority voices and documents sustain multi-hour coercion",
    defeats: "single-transaction risk scoring",
    executable: false,
  },
  {
    id: "T-14",
    title: "VPA-rental mule networks",
    channel: "UPI / instant rails",
    rail: "real-time rails",
    genaiEnabler: "Automated recruitment and rotation of rented payment addresses",
    defeats: "per-account monitoring without graph context",
    executable: true,
    family: "ATTACK_6_VPA_RENTAL_MULE",
  },
  {
    id: "T-15",
    title: "Fan-out dispersal and layered multi-hop",
    channel: "UPI / instant rails",
    rail: "real-time rails",
    genaiEnabler: "Route planning that keeps every hop individually unremarkable",
    defeats: "thresholds applied per transaction",
    executable: false,
  },
  {
    id: "T-16",
    title: "QR and collect-request redirection",
    channel: "QR / collect-request",
    rail: "real-time rails",
    genaiEnabler: "Generated payee identities and QR overlays that survive visual inspection",
    defeats: "payee-name verification by eye",
    executable: false,
  },
  {
    id: "T-17",
    title: "Synchronised burst cash-out",
    channel: "UPI / instant rails",
    rail: "real-time rails",
    genaiEnabler: "Coordinated timing across many mules within one detection window",
    defeats: "independence assumptions between accounts",
    executable: true,
    family: "ATTACK_7_SYNCHRONISED_BURST_CASHOUT",
  },
  // ---- Layer 5 — merchant, agentic commerce and model-layer attacks ----
  {
    id: "T-18",
    title: "Prompt-injected merchant / agentic checkout hijack",
    channel: "agentic checkout",
    rail: "agentic commerce",
    genaiEnabler: "A poisoned merchant catalogue or page redirects an AI shopping agent's checkout",
    defeats: "the assumption that the buyer is a human making a choice",
    executable: true,
    family: "ATTACK_3_PROMPT_INJECTED_MERCHANT",
  },
  {
    id: "T-19",
    title: "Agent scope expansion",
    channel: "delegated payment agent",
    rail: "agentic commerce",
    genaiEnabler: "A delegated payment agent drifts beyond its mandate through intent manipulation",
    defeats: "consent captured once at delegation time",
    executable: true,
    family: "ATTACK_11_AGENTIC_SCOPE_EXPANSION",
  },
  {
    id: "T-20",
    title: "GenAI collusive merchant ring",
    channel: "merchant ring",
    rail: "agentic commerce",
    genaiEnabler: "Coordinated fake merchants and fake customers generate mutually reinforcing transaction history",
    defeats: "reputation built from transaction volume",
    executable: true,
    family: "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE",
  },
  {
    id: "T-21",
    title: "Synthetic-evidence refund and chargeback abuse",
    channel: "disputes / chargebacks",
    rail: "post-authorisation",
    genaiEnabler: "Generated receipts, photos and delivery evidence win disputes",
    defeats: "manual evidence review",
    executable: false,
  },
  {
    id: "T-22",
    title: "Model-layer attack: poisoning the feedback loop",
    channel: "retraining corpus",
    rail: "model layer",
    genaiEnabler: "Attacks crafted to be *labelled* wrongly, corrupting the retraining set itself",
    defeats: "closed-loop retraining without label provenance",
    executable: false,
  },
];

/** Distinct channel values, derived — feeds the channel filter chips. */
export const CHANNELS: readonly string[] = [...new Set(TAXONOMY.map((t) => t.channel))].sort();
