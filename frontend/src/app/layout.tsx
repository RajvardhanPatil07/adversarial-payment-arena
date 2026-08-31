import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";

import "./globals.css";
// Imported after globals.css so the design tokens win over the inherited
// shadcn dark palette for the properties both files declare.
import "@/styles/tokens.css";

import { Footer } from "@/components/shell/footer";
import { JudgeTour } from "@/components/shell/judge-tour";
import { Nav } from "@/components/shell/nav";
import { readSiteProvenance, fmtDate } from "@/lib/artifacts";
import { SITE } from "@/lib/site";

/**
 * Typography rule: mono == measured, sans == asserted.
 *
 * Inter carries prose, JetBrains Mono carries every measured number. The
 * variables are composed in front of the `--font-ui` / `--font-mono` token
 * stacks in globals.css, so the tokens remain the declared fallback.
 */
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "A closed-loop red team without a fidelity gate is an attack surface",
    template: `%s — ${SITE.name}`,
  },
  description:
    "Folding a low-fidelity generator's escapes back into training makes every dashboard number improve while recall on real fraud falls. A label-free fidelity gate, computable before retraining, removes that failure mode.",
  // OG / social card (SECTION 8, edge #8): the card states the thesis, the
  // criteria and the verifiability rule — no numbers, because numbers belong
  // to artifacts and are rendered at runtime, not baked into metadata.
  openGraph: {
    type: "website",
    siteName: SITE.name,
    title: "A closed-loop red team without a fidelity gate is an attack surface",
    description:
      "Adversarial Payment Arena — 22 attack families mapped, 14 measured, a label-free fidelity gate that blocks the closed-loop failure mode. Every number links to the artifact it was read from.",
  },
  twitter: {
    card: "summary",
    title: "A closed-loop red team without a fidelity gate is an attack surface",
    description:
      "Adversarial Payment Arena — a closed-loop red team for payment fraud, with a label-free fidelity gate. Every number has an address.",
  },
};

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // The provenance stamp is shared by the Nav's manifest chip and the
  // Footer, so it is read once here. Both agree by construction, and a
  // missing evidence set renders explicit "not measured" states.
  const provenance = await readSiteProvenance();

  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-bg text-text">
        {/* Keyboard users should not have to tab the whole nav on every page. */}
        <a
          href="#main"
          className="type-ui sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[60] focus:rounded-[var(--r-sm)] focus:border focus:border-blue focus:bg-surface-2 focus:px-3 focus:py-2 focus:text-sm focus:text-text"
        >
          Skip to content
        </a>

        <Nav validatedArtifacts={provenance.artifactCount} />

        {/* JUDGE MODE: the ~90s guided tour. Rendered once, site-wide. */}
        <JudgeTour />

        <main id="main" className="flex flex-1 flex-col">
          {children}
        </main>

        <Footer
          gitSha={provenance.gitSha}
          artifactCount={provenance.artifactCount}
          evidenceGeneratedAt={fmtDate(provenance.generatedAt)}
          seeds={provenance.seeds}
        />

        <Toaster theme="dark" richColors closeButton position="bottom-right" />
      </body>
    </html>
  );
}
