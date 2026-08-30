/**
 * Persistent footer — the provenance stamp.
 *
 * Verifiability is a visible feature, so the footer carries the build's git
 * SHA, the artifact count, the evidence generation timestamp, the command that
 * regenerates everything, the licence, and the synthetic-data notice.
 *
 * Phase 0 found that artifact provenance SHAs are NOT uniform across the
 * evidence set (four different SHAs across 13 artifacts). A single footer SHA
 * would therefore misrepresent the evidence base, so this component labels its
 * SHA explicitly as the UI build and defers per-artifact provenance to the
 * evidence ledger. Values arrive as props and render an explicit
 * "not measured" when absent -- never a placeholder value.
 */

import { SITE } from "@/lib/site";

export interface FooterProps {
  /** Git SHA of the commit this UI was built from. */
  gitSha?: string | null;
  /** Number of artifacts present in the static snapshot. */
  artifactCount?: number | null;
  /** ISO timestamp of the newest artifact's `provenance.generated_at`. */
  evidenceGeneratedAt?: string | null;
}

/** Renders a value, or an explicit unmeasured state. Never a fallback number. */
function Value({ children }: { children: string | number | null | undefined }) {
  if (children === null || children === undefined || children === "") {
    return (
      <span className="text-text-dim">
        <span aria-hidden="true">—</span>
        <span className="ml-1.5 text-[0.6875rem]">not measured</span>
      </span>
    );
  }
  return <span className="text-text-dim">{children}</span>;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="type-ui text-[0.6875rem] uppercase tracking-[0.08em] text-text-dim">
        {label}
      </dt>
      <dd className="type-num text-xs">{children}</dd>
    </div>
  );
}

export function Footer({
  gitSha = null,
  artifactCount = null,
  evidenceGeneratedAt = null,
}: FooterProps) {
  return (
    <footer className="mt-auto border-t border-border bg-surface-1">
      <div className="mx-auto max-w-[1400px] px-4 py-8 md:px-6">
        <dl className="grid grid-cols-2 gap-6 md:grid-cols-3 lg:grid-cols-5">
          <Row label="UI build">
            <Value>{gitSha}</Value>
          </Row>
          <Row label="Artifacts">
            <Value>{artifactCount}</Value>
          </Row>
          <Row label="Evidence generated">
            <Value>{evidenceGeneratedAt}</Value>
          </Row>
          <Row label="Reproduce">
            <code className="text-text-dim">{SITE.footer.reproduce}</code>
          </Row>
          <Row label="Licence">
            <a
              href={`${SITE.repo}/blob/main/LICENSE`}
              target="_blank"
              rel="noreferrer"
              className="text-text-dim underline decoration-border-hi underline-offset-2 transition-colors hover:text-blue"
            >
              {SITE.footer.licence}
            </a>
          </Row>
        </dl>

        <p className="type-ui mt-6 border-t border-border pt-4 text-xs text-text-dim">
          {SITE.footer.dataNotice}. Per-artifact provenance — the git SHA, seeds and
          command behind each individual number — is listed on the evidence ledger,
          because the artifacts were not all generated at the same commit.
        </p>
      </div>
    </footer>
  );
}
