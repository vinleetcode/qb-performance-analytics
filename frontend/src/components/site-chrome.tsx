"use client";
import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { GITHUB_URL } from "@/lib/config";
import { cn } from "@/lib/utils";
import { BRAND, Logo } from "./brand";

export { BRAND };

const NAV = [
  { href: "/prospects", label: "Prospects" },
  { href: "/backtesting", label: "Backtesting" },
  { href: "/methodology", label: "Methodology" },
  { href: "/case-study", label: "Case study" },
];

const isOn = (path: string, href: string) => path === href || path.startsWith(`${href}/`);

function DesktopNav({ path }: { path: string }) {
  return (
    <nav aria-label="Main" className="hidden items-center gap-1 text-sm md:flex">
      {NAV.map((n) => (
        <Link
          key={n.href}
          href={n.href}
          aria-current={isOn(path, n.href) ? "page" : undefined}
          className={cn(
            "rounded px-2.5 py-1.5 text-muted-foreground hover:text-chalk",
            isOn(path, n.href) && "text-chalk underline decoration-projection decoration-2 underline-offset-[10px]",
          )}
        >
          {n.label}
        </Link>
      ))}
      <a href={GITHUB_URL} className="rounded px-2.5 py-1.5 text-muted-foreground hover:text-chalk" target="_blank" rel="noreferrer">
        GitHub
      </a>
    </nav>
  );
}

function MobileNav({ path }: { path: string }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  const close = () => setOpen(false);
  return (
    <div className="md:hidden">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="mobile-menu"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex size-10 items-center justify-center rounded-md border border-line text-chalk"
      >
        {open ? <X className="size-5" aria-hidden /> : <Menu className="size-5" aria-hidden />}
        <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
      </button>
      {open && (
        <nav id="mobile-menu" aria-label="Main" className="absolute inset-x-0 top-full z-40 border-b border-line bg-background/98 px-4 pb-4 shadow-xl shadow-black/40 backdrop-blur">
          <ul className="divide-y divide-line">
            {NAV.map((n) => (
              <li key={n.href}>
                <Link
                  href={n.href}
                  onClick={close}
                  aria-current={isOn(path, n.href) ? "page" : undefined}
                  className={cn("block py-3 font-display text-[22px] font-semibold", isOn(path, n.href) ? "text-chalk" : "text-chalk/75")}
                >
                  {n.label}
                </Link>
              </li>
            ))}
            <li>
              <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="block py-3 font-display text-[22px] font-semibold text-chalk/75">
                GitHub
              </a>
            </li>
          </ul>
        </nav>
      )}
    </div>
  );
}

function ActiveNav() {
  const path = usePathname();
  return (
    <>
      <DesktopNav path={path} />
      <MobileNav path={path} />
    </>
  );
}

export function SiteHeader() {
  return (
    <header className="relative z-30 border-b border-line bg-background">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-raised focus:px-3 focus:py-2">
        Skip to content
      </a>
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-x-6 px-4 py-3 sm:px-6">
        <Link href="/" aria-label={`${BRAND}, home`} className="rounded-sm py-1 text-chalk">
          <Logo size={24} />
        </Link>
        {/* the active-link state reads the URL, which Next.js only knows at request time */}
        <Suspense fallback={<DesktopNav path="" />}>
          <ActiveNav />
        </Suspense>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-10 text-sm text-muted-foreground sm:grid-cols-[2fr_1fr] sm:px-6">
        <div className="measure space-y-3">
          <p className="text-chalk">
            <Logo size={20} />
          </p>
          <p>
            Forecasts come from models trained on drafted quarterbacks since 2005 and tested on later draft classes. They are
            estimates with wide ranges, not scouting grades.
          </p>
          <p>
            College data provided by{" "}
            <a href="https://collegefootballdata.com" target="_blank" rel="noreferrer" className="underline hover:text-chalk">CollegeFootballData.com</a>.
            NFL data from{" "}
            <a href="https://github.com/nflverse/nflverse-data" target="_blank" rel="noreferrer" className="underline hover:text-chalk">nflverse</a>{" "}
            (<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer" className="underline hover:text-chalk">CC BY 4.0</a>).
            A portfolio project, not affiliated with or endorsed by the NFL, any team, school or conference.
          </p>
        </div>
        <ul className="flex flex-col gap-1.5 sm:items-end">
          <li><a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-chalk">Source code on GitHub</a></li>
          <li><Link href="/methodology" className="hover:text-chalk">Methodology and accuracy</Link></li>
          <li><Link href="/case-study" className="hover:text-chalk">Case study</Link></li>
          <li><Link href="/credits" className="hover:text-chalk">Data and photo credits</Link></li>
        </ul>
      </div>
    </footer>
  );
}
