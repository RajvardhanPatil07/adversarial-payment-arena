"use client";

/**
 * Persistent top navigation.
 *
 * Layout: product name left, the six routes centre, status chips right.
 *
 * The manifest chip reports how many artifacts the STATIC snapshot contains.
 * Phase 1 renders the shell only, so it is passed in as a prop and rendered as
 * "not measured" when absent -- the artifact loader that supplies it lands in
 * Phase 3. It is never defaulted to a number, because a fabricated artifact
 * count would be a fabricated claim about the evidence base.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { START_EVENT } from "@/components/shell/judge-tour";
import { StatusChip } from "@/components/shell/status-chip";
import { ROUTES, SITE } from "@/lib/site";

/** Opens the guided tour by dispatching the start event JudgeTour listens for. */
function startTour(): void {
  window.dispatchEvent(new Event(START_EVENT));
}

export interface NavProps {
  /**
   * Number of artifacts that passed manifest validation, or null when the
   * manifest has not been read yet. Null renders an explicit unmeasured state.
   */
  validatedArtifacts?: number | null;
}

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function Nav({ validatedArtifacts = null }: NavProps) {
  const pathname = usePathname();

  /**
   * The mobile sheet must close on navigation, so a tap never leaves it open
   * over the page it just navigated to.
   *
   * That is done by DERIVING the open state from the route rather than by
   * resetting it in an effect: storing the pathname the sheet was opened at and
   * treating a change as closed. An effect calling setState here would trigger
   * a cascading render on every navigation (react-hooks/set-state-in-effect),
   * and is the documented "you might not need an effect" case.
   */
  const [openedAt, setOpenedAt] = useState<string | null>(null);
  const open = openedAt === pathname;

  const setOpen = (next: boolean): void => {
    setOpenedAt(next ? pathname : null);
  };

  const manifestChip =
    validatedArtifacts === null ? (
      <StatusChip tone="neutral" title="Artifact manifest has not been read on this page yet.">
        manifest · <span className="text-text-dim">not measured</span>
      </StatusChip>
    ) : (
      <StatusChip tone="blue" title="Every artifact in the static snapshot passed schema validation.">
        manifest validated · {validatedArtifacts} artifacts
      </StatusChip>
    );

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-bg/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-4 px-4 md:px-6">
        {/* Left: product name */}
        <Link
          href="/"
          className="type-ui shrink-0 text-sm font-semibold tracking-tight text-text transition-colors hover:text-blue"
        >
          {SITE.name}
        </Link>

        {/* Centre: routes */}
        <nav aria-label="Primary" className="hidden flex-1 justify-center lg:flex">
          <ul className="flex items-center gap-1">
            {ROUTES.map((route) => {
              const active = isActive(pathname, route.href);
              return (
                <li key={route.href}>
                  <Link
                    href={route.href}
                    aria-current={active ? "page" : undefined}
                    title={route.criterion}
                    className={`type-ui rounded-[var(--r-sm)] px-3 py-1.5 text-sm transition-colors ${
                      active
                        ? "bg-surface-2 text-text"
                        : "text-text-dim hover:bg-surface-1 hover:text-text"
                    }`}
                  >
                    {route.nav}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Right: status chips */}
        <div className="ml-auto hidden items-center gap-2 lg:flex">
          {/* JUDGE MODE: the ~90s guided tour (SECTION 8). */}
          <button
            type="button"
            onClick={startTour}
            title="A ~90 second guided tour of the whole argument, one step per page."
            className="type-ui rounded-[var(--r-sm)] border border-blue/50 bg-blue-dim/40 px-3 py-1.5 text-sm text-blue transition-colors hover:border-blue hover:bg-blue-dim"
          >
            Judge mode
          </button>
          <StatusChip tone="neutral" title="No cardholder data. Every transaction is generated.">
            {SITE.chips.synthetic}
          </StatusChip>
          <StatusChip tone="neutral" title="Runs with no connection to a production payment network.">
            {SITE.chips.offline}
          </StatusChip>
          {manifestChip}
        </div>

        {/* Mobile toggle */}
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-controls="mobile-nav"
          className="type-ui ml-auto rounded-[var(--r-sm)] border border-border px-3 py-1.5 text-sm text-text-dim transition-colors hover:border-border-hi hover:text-text lg:hidden"
        >
          {open ? "Close" : "Menu"}
        </button>
      </div>

      {/* Mobile sheet */}
      {open && (
        <div id="mobile-nav" className="border-t border-border bg-surface-1 lg:hidden">
          <nav aria-label="Primary" className="mx-auto max-w-[1400px] px-4 py-3">
            <ul className="flex flex-col gap-1">
              {ROUTES.map((route) => {
                const active = isActive(pathname, route.href);
                return (
                  <li key={route.href}>
                    <Link
                      href={route.href}
                      aria-current={active ? "page" : undefined}
                      className={`type-ui flex flex-col gap-0.5 rounded-[var(--r-sm)] px-3 py-2 transition-colors ${
                        active ? "bg-surface-2 text-text" : "text-text-dim hover:bg-surface-2"
                      }`}
                    >
                      <span className="text-sm font-medium">{route.nav}</span>
                      <span className="type-num text-[0.6875rem] text-text-dim">
                        {route.criterion}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
            <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3">
              <button
                type="button"
                onClick={startTour}
                className="type-ui rounded-[var(--r-sm)] border border-blue/50 bg-blue-dim/40 px-3 py-1.5 text-sm text-blue transition-colors hover:border-blue hover:bg-blue-dim"
              >
                Judge mode
              </button>
              <StatusChip>{SITE.chips.synthetic}</StatusChip>
              <StatusChip>{SITE.chips.offline}</StatusChip>
              {manifestChip}
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}
