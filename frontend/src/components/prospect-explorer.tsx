"use client";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Users } from "lucide-react";
import { useProspectSearch } from "@/lib/api/queries";
import type { ProspectSummary } from "@/lib/api/types";
import { Button } from "@/components/ui/button";
import { ErrorState, SlowLoading } from "@/components/status";
import { draftLine, useDebounced } from "@/components/prospect-search";
import { PhotoBanner } from "@/components/photo-banner";
import { PHOTOS } from "@/lib/images";

const PAGE = 20;
const YEARS = Array.from({ length: 2026 - 2000 + 1 }, (_, i) => 2026 - i);

export function availability(p: ProspectSummary) {
  if (p.kind === "drafted") return "Pre- and post-draft forecasts, comparisons, historical backtest";
  return "Pre-draft forecast and comparisons (no NFL draft record)";
}

export function ProspectRow({ p }: { p: ProspectSummary }) {
  return (
    <li className="border-t border-line">
      <Link href={`/prospects/${p.player_id}`} className="grid gap-1 px-1 py-4 hover:bg-panel/60 sm:grid-cols-[1.4fr_1fr_1.2fr] sm:items-baseline sm:gap-4">
        <span className="font-display text-[22px] font-semibold text-chalk">
          {p.name}
          {p.shares_name_with_other_prospects && (
            <span className="ml-2 inline-flex items-center gap-1 align-middle font-sans text-[12px] font-normal text-caution">
              <Users className="size-3" aria-hidden /> shares a name with another prospect
            </span>
          )}
        </span>
        <span className="text-[14px] text-chalk/90">
          {p.college ?? "College unknown"}
          <span className="block text-[13px] text-muted-foreground">{draftLine(p)}</span>
        </span>
        <span className="text-[13px] text-muted-foreground">{availability(p)}</span>
      </Link>
    </li>
  );
}

export function ProspectExplorer() {
  const router = useRouter();
  const path = usePathname();
  const sp = useSearchParams();
  const [text, setText] = useState(sp.get("q") ?? "");
  const q = useDebounced(text.trim(), 250);
  const year = sp.get("year") ? Number(sp.get("year")) : undefined;
  const drafted = sp.get("drafted") === "1";
  const page = Math.max(0, Number(sp.get("page") ?? 0) || 0);

  const push = (next: Record<string, string | undefined>) => {
    const n = new URLSearchParams(sp.toString());
    for (const [k, v] of Object.entries(next)) {
      if (v === undefined || v === "") n.delete(k);
      else n.set(k, v);
    }
    router.replace(n.size ? `${path}?${n}` : path, { scroll: false });
  };
  useEffect(() => {
    if ((sp.get("q") ?? "") !== q) push({ q: q || undefined, page: undefined });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);

  const res = useProspectSearch(q, { draftYear: year, draftedOnly: drafted, page, limit: PAGE });
  const total = res.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));

  return (
    <>
      <PhotoBanner
        photo={PHOTOS.yardLine}
        priority
        title="The prospect database"
        intro={
          <p>
            Every drafted quarterback from 2000 to 2026, plus FBS quarterbacks whose final college season is in the data. Open a
            report for draft and NFL career forecasts.
          </p>
        }
      />
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <form className=" grid gap-3 sm:grid-cols-[1fr_11rem_auto]" role="search" onSubmit={(e) => e.preventDefault()}>
        <div>
          <label htmlFor="q" className="block text-[13px] text-muted-foreground">Name</label>
          <input
            id="q"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. Mendoza, Josh Allen"
            autoComplete="off"
            className="mt-1 h-11 w-full rounded-md border border-line bg-panel px-3 text-chalk placeholder:text-muted-foreground"
          />
        </div>
        <div>
          <label htmlFor="year" className="block text-[13px] text-muted-foreground">Draft class</label>
          <select
            id="year"
            value={year ?? ""}
            onChange={(e) => push({ year: e.target.value || undefined, page: undefined })}
            className="mt-1 h-11 w-full rounded-md border border-line bg-panel px-3 text-chalk"
          >
            <option value="">All classes</option>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>
        <label className="flex h-11 items-center gap-2 self-end text-[14px] text-chalk">
          <input
            type="checkbox"
            checked={drafted}
            onChange={(e) => push({ drafted: e.target.checked ? "1" : undefined, page: undefined })}
            className="size-4 accent-[var(--projection)]"
          />
          Drafted only
        </label>
      </form>

      <div className="mt-8" aria-live="polite">
        {res.isPending ? (
          <SlowLoading label="Loading prospects" lines={6} />
        ) : res.isError ? (
          <ErrorState error={res.error} onRetry={() => res.refetch()} />
        ) : (
          <>
            <p className="text-[13px] text-muted-foreground">
              {total === 0 ? "No quarterbacks match these filters." : `${total.toLocaleString()} quarterback${total === 1 ? "" : "s"}`}
              {total > 0 && ` (page ${page + 1} of ${pages})`}
            </p>
            {total === 0 ? (
              <p className="mt-4 text-sm text-muted-foreground">
                Try part of a last name, or clear the draft class filter. College quarterbacks appear only if they threw at least 100
                passes in a final FBS season.
              </p>
            ) : (
              <ul className={`mt-3 border-b border-line ${res.isPlaceholderData ? "opacity-60" : ""}`}>
                {res.data.items.map((p) => (
                  <ProspectRow key={p.player_id} p={p} />
                ))}
              </ul>
            )}
            {pages > 1 && (
              <nav aria-label="Pagination" className="mt-6 flex items-center gap-3">
                <Button variant="outline" disabled={page === 0} onClick={() => push({ page: page - 1 ? String(page - 1) : undefined })}>
                  <ChevronLeft aria-hidden /> Previous
                </Button>
                <Button variant="outline" disabled={page + 1 >= pages} onClick={() => push({ page: String(page + 1) })}>
                  Next <ChevronRight aria-hidden />
                </Button>
              </nav>
            )}
          </>
        )}
      </div>
      </div>
    </>
  );
}
