import type { Metadata } from "next";
import { Geist_Mono, Manrope } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";

import "./globals.css";

const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Adversarial Payment Arena — Fidelity-Gated Fraud Defense",
  description:
    "See the fidelity scissor: synthetic-attack recall rises while held-out simulated fraud recall falls, and a label-free gate prevents the damage.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${manrope.variable} ${geistMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background">
        {children}
        <Toaster theme="dark" richColors closeButton position="bottom-right" />
      </body>
    </html>
  );
}
