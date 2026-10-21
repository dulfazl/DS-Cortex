import type { Metadata } from "next";
import Link from "next/link";
import { MetaBadge } from "@/components/MetaBadge";
import "./globals.css";

export const metadata: Metadata = {
  title: "DS Cortex",
  description: "Agents that audit your data, debate candidate pipelines with evidence, and explain every decision.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col">
        <header className="border-b border-border bg-surface">
          <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <span className="grid h-7 w-7 place-items-center rounded-md bg-accent text-sm text-white">DS</span>
              <span>DS Cortex</span>
            </Link>
            <nav className="flex items-center gap-4 text-sm text-muted">
              <Link href="/" className="hover:text-text">
                New analysis
              </Link>
              <Link href="/replay" className="hover:text-text">
                Replays
              </Link>
            </nav>
            <div className="ml-auto">
              <MetaBadge />
            </div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">{children}</main>
        <footer className="border-t border-border py-4 text-center text-xs text-muted">
          Every recommendation carries tool-run evidence and a computed confidence. Nothing is decided silently.
        </footer>
      </body>
    </html>
  );
}
