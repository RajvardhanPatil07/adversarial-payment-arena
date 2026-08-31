/**
 * <Boundary> — the "--warn bordered callout" from SECTION 5.
 *
 * Every result on this site is allowed to be partial; the boundary states
 * exactly how far it reaches. Deliberately NOT tone-mapped to red/blue:
 * boundaries are neither attacker nor defender, they are epistemic.
 *
 * Server-safe (no hooks): renders identically in server and client trees.
 */

import type { ReactNode } from "react";

export interface BoundaryProps {
  /** Override the default "Boundary of this result" heading. */
  title?: string;
  /** Boundary sentences as a list (preferred — one <li> per boundary). */
  items?: readonly string[];
  /** Or a single free-form child. */
  children?: ReactNode;
}

export function Boundary({ title = "Boundary of this result", items, children }: BoundaryProps) {
  return (
    <aside
      aria-label={title}
      className="measure rounded-[var(--r-md)] border border-warn/40 bg-surface-1 px-4 py-3"
    >
      <p className="type-ui text-[0.6875rem] font-semibold uppercase tracking-[0.08em] text-warn">{title}</p>
      {items && items.length > 0 ? (
        <ul className="mt-2 list-disc space-y-1 pl-4">
          {items.map((b) => (
            <li key={b} className="type-ui text-xs text-text-dim">
              {b}
            </li>
          ))}
        </ul>
      ) : (
        <div className="type-ui mt-2 text-xs text-text-dim">{children}</div>
      )}
    </aside>
  );
}
