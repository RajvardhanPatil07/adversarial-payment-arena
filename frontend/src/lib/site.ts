/**
 * Route + framing metadata for the judge-facing evidence document.
 *
 * One source of truth so the nav, the footer and (later) the guided tour
 * cannot drift out of sync with the actual App Router tree.
 *
 * Every `h1` here is a SENTENCE MAKING A CLAIM rather than a noun label, and
 * every route names the judging criterion it answers. Nothing in this file is
 * a measured value -- headline numbers are read from artifacts at runtime, so
 * no figure is ever hardcoded here.
 */

/** The five judging criteria this submission is built against. */
export type Criterion =
  | "diversity of attacks identified"
  | "fidelity of simulated attacks"
  | "detection efficacy"
  | "novelty"
  | "real-world feasibility in live payments";

export interface RouteMeta {
  /** App Router path. */
  href: string;
  /** Short nav label. */
  nav: string;
  /** The page's H1 -- a claim, never a label. */
  h1: string;
  /** Which judging criterion this page answers. */
  criterion: Criterion;
  /**
   * One-line framing shown under the H1 and in the hero's scroll narrative.
   * Prose only: assertions about what the page shows, never a measurement.
   */
  blurb: string;
}

export const ROUTES: readonly RouteMeta[] = [
  {
    href: "/",
    nav: "Thesis",
    h1: "A closed-loop red team without a fidelity gate is an attack surface, not a feature.",
    criterion: "novelty",
    blurb:
      "Folding a low-fidelity generator's escapes back into training makes every dashboard number improve while recall on real fraud falls. A label-free fidelity gate, computable before retraining, removes that failure mode.",
  },
  {
    href: "/identify",
    nav: "Identify",
    // Phase 0 finding: the repository maps 22 taxa but only 14 are executable
    // and measured. docs/ATTACK_TAXONOMY.md states plainly that claiming 22
    // implemented attacks "would be the kind of number this repository is
    // built to argue against", so the H1 says mapped-vs-executable outright.
    h1: "Twenty-two attack vectors mapped, fourteen executable, and the measured recall for each.",
    criterion: "diversity of attacks identified",
    blurb:
      "Each family was chosen because it defeats a different control class, so breadth is not fourteen variations of velocity abuse. Every executable family carries its own measured detection number.",
  },
  {
    href: "/generate",
    nav: "Generate",
    h1: "Correct marginals are not fidelity. The joint structure is.",
    criterion: "fidelity of simulated attacks",
    blurb:
      "A generator can match every column histogram and still be trivially separable from real fraud. Five fidelity measures are reported per generator, and the gate's thresholds were fixed in advance.",
  },
  {
    href: "/defend",
    nav: "Defend",
    h1: "Hardening measured on attack families the model never saw.",
    criterion: "detection efficacy",
    blurb:
      "Recall on a family the supervised model trained on is not evidence of generalisation. Every family is also measured with that family withheld from supervised training.",
  },
  {
    href: "/arena",
    nav: "Arena",
    h1: "Watch the loop run.",
    criterion: "real-world feasibility in live payments",
    blurb:
      "The live red-team/blue-team console: attacker reasoning on the left, the entity graph in the centre, the defense stack's decisions on the right.",
  },
  {
    href: "/evidence",
    nav: "Evidence",
    h1: "Every number has an address.",
    criterion: "real-world feasibility in live payments",
    blurb:
      "Each claim maps to an artifact, a field inside it, the command that regenerates it, and the boundary condition that limits it.",
  },
] as const;

/** Every route path, as a literal union. */
export type RoutePath = (typeof ROUTES)[number]["href"];

/**
 * Route metadata keyed by path.
 *
 * Pages index this directly (`ROUTE.identify`), which is exact-typed and
 * therefore needs no non-null assertion at the call site -- `ROUTES.find()`
 * would return `RouteMeta | undefined` and force a `!` into every page.
 */
export const ROUTE: Record<RoutePath, RouteMeta> = ROUTES.reduce(
  (acc, route) => {
    acc[route.href] = route;
    return acc;
  },
  {} as Record<RoutePath, RouteMeta>,
);

/**
 * Static, non-measured facts about the submission's operating envelope.
 * These are properties of the design, not experimental results, so they are
 * safe to state in the shell. Anything measured is read from an artifact.
 */
export const SITE = {
  name: "Adversarial Payment Arena",
  /** Nav status chips. `synthetic-only` and `offline lab` are design facts. */
  chips: {
    synthetic: "synthetic-only",
    offline: "offline lab",
  },
  footer: {
    reproduce: "make reproduce",
    licence: "MIT",
    dataNotice: "no cardholder data — fully synthetic",
  },
  /** Repo URL, used for the artifact-source links in the footer. */
  repo: "https://github.com/RajvardhanPatil07/adversarial-payment-arena",
} as const;
