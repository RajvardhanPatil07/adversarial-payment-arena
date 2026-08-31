"use client";

/**
 * JUDGE MODE — the ~90s guided tour (SECTION 8, competitive edge #1).
 *
 * A judge opening this repository for the first time has one question: "what
 * am I looking at?" This tour answers it in six steps, one per route, in the
 * order the argument is made. It is opened from the nav button or the
 * `?tour=1` deep link (for sharing a judged walkthrough directly).
 *
 * Steps are DERIVED from `ROUTES` in lib/site.ts, so the tour can never drift
 * out of sync with the actual App Router tree or the H1 each page makes.
 *
 * Deliberately minimal: a floating card, progress dots, prev/next, ESC to
 * close. No timers, no autoplay — a judge reads at their own pace.
 */

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { ROUTES } from "@/lib/site";

/** One step per route, in argument order. */
const STEPS = ROUTES.map((route) => ({
  href: route.href,
  nav: route.nav,
  h1: route.h1,
  blurb: route.blurb,
  criterion: route.criterion,
}));

export const START_EVENT = "judge-mode:start";

export function JudgeTour() {
  /** Active step index, or null when the tour is closed. */
  const [step, setStep] = useState<number | null>(null);
  /** Guard so the ?tour=1 deep link starts the tour exactly once. */
  const linkHandled = useRef(false);

  useEffect(() => {
    /**
     * The tour is opened by a custom event, dispatched by the nav button and —
     * once — by the ?tour=1 deep link. State changes happen inside the event
     * handler, never synchronously in the effect body (react-hooks rules).
     */
    const start = () => {
      setStep((current) => (current === null ? 0 : current));
    };
    window.addEventListener(START_EVENT, start);

    if (!linkHandled.current) {
      linkHandled.current = true;
      if (new URLSearchParams(window.location.search).get("tour") === "1") {
        window.dispatchEvent(new Event(START_EVENT));
      }
    }

    return () => {
      window.removeEventListener(START_EVENT, start);
    };
  }, []);

  /** ESC closes the tour — a keyboard user must be able to leave it. */
  useEffect(() => {
    if (step === null) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setStep(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  if (step === null) return null;

  const current = STEPS[step];
  if (!current) return null;
  const first = step === 0;
  const last = step === STEPS.length - 1;

  return (
    // The dim backdrop closes the tour on click; the card stops propagation.
    <div
      role="dialog"
      aria-label="Judge mode tour"
      onClick={() => setStep(null)}
      className="fixed inset-0 z-[70] flex items-end justify-center bg-bg/70 px-4 pb-6 backdrop-blur-[2px] md:pb-10"
    >
      <div
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-xl rounded-[var(--r-lg)] border border-border-hi bg-surface-1 p-5 shadow-2xl md:p-6"
      >
        <p className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
          Judge mode · step {step + 1} of {STEPS.length} · {current.criterion}
        </p>

        <h2 className="type-ui mt-2 text-base font-semibold leading-snug tracking-tight text-text">
          {current.nav}: {current.h1}
        </h2>
        <p className="type-ui measure mt-2 text-sm leading-relaxed text-text-dim">
          {current.blurb}
        </p>

        {/* Progress dots: position in the argument, also direct navigation. */}
        <div className="mt-4 flex items-center gap-1.5" role="tablist" aria-label="Tour steps">
          {STEPS.map((s, i) => (
            <button
              key={s.href}
              type="button"
              role="tab"
              aria-selected={i === step}
              aria-label={`Step ${i + 1}: ${s.nav}`}
              onClick={() => setStep(i)}
              className={`h-1.5 rounded-[var(--r-sm)] transition-all ${
                i === step ? "w-6 bg-blue" : "w-3 bg-border-hi hover:bg-text-faint"
              }`}
            />
          ))}
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setStep(first ? null : step - 1)}
            className="type-ui rounded-[var(--r-sm)] border border-border px-3 py-1.5 text-sm text-text-dim transition-colors hover:border-border-hi hover:text-text"
          >
            {first ? "Close" : "Back"}
          </button>
          <Link
            href={current.href}
            onClick={() => setStep(null)}
            className="type-ui rounded-[var(--r-sm)] px-3 py-1.5 text-sm text-blue transition-colors hover:bg-blue-dim"
          >
            Open {current.nav}
          </Link>
          <button
            type="button"
            autoFocus
            onClick={() => setStep(last ? null : step + 1)}
            className="type-ui ml-auto rounded-[var(--r-sm)] bg-surface-3 px-3 py-1.5 text-sm text-text transition-colors hover:bg-border"
          >
            {last ? "Finish" : "Next"}
          </button>
        </div>

        <p className="type-ui mt-3 text-[0.6875rem] text-text-faint">
          ESC closes the tour · shareable deep link: <span className="type-num">?tour=1</span>
        </p>
      </div>
    </div>
  );
}
