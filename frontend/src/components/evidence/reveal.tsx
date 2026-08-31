"use client";

/**
 * <Reveal> — the one subtle 200–300ms scroll reveal per section (SECTION 3).
 *
 * The server renders NO data-reveal attribute, so the CSS default (visible)
 * applies and content is legible with JavaScript disabled. Only after
 * hydration does the pending state get set, immediately followed by the
 * IntersectionObserver flip — so JS users get the transition and no-JS users
 * get the end state. prefers-reduced-motion never enters the pending state
 * (and the global CSS kills transitions besides).
 */

import { useEffect, useRef, type ReactNode } from "react";

export function Reveal({ children, className }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }
    el.dataset.reveal = "pending";
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          el.dataset.reveal = "revealed";
          io.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={ref} className={`reveal ${className ?? ""}`}>
      {children}
    </div>
  );
}
