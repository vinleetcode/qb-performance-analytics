"use client";
import Link from "next/link";
import { useQueries } from "@tanstack/react-query";
import { api } from "@/lib/api/client";
import { qk, retryPolicy, useComparisons } from "@/lib/api/queries";
import type { Comparable, ProspectDetail } from "@/lib/api/types";
import { fmt1, fmtInt } from "@/lib/format";
import { ErrorState, SlowLoading, Unavailable } from "@/components/status";
import { InfoTip } from "@/components/info-tip";
import { initials, num } from "./player-header";
import { Section } from "./section";

const COLS = [
  ["completion_pct", "Cmp %"],
  ["yards_per_attempt", "YPA"],
  ["td_pct", "TD %"],
  ["int_pct", "INT %"],
  ["rush_yards_per_game_last3", "Rush yds/g"],
] as const;

function Outcome({ c }: { c: Comparable }) {
  const o = c.nfl_outcome;
  if (o.window_complete && o.starts_first_5_seasons != null) return <>{o.starts_first_5_seasons} starts in NFL seasons 1 to 5</>;
  if (o.starts_to_date != null) return <>{o.starts_to_date} starts so far (five-season window still open)</>;
  return <>No NFL outcome recorded</>;
}

export function ComparisonsSection({ id, detail }: { id: string; detail: ProspectDetail }) {
  const q = useComparisons(id);
  const comps = q.data?.status === "available" ? q.data.comparables : [];
  const details = useQueries({
    queries: comps.map((c) => ({
      queryKey: qk.prospect(c.player_id ?? ""),
      queryFn: () => api.prospect(c.player_id as string),
      enabled: !!c.player_id,
      staleTime: 60 * 60 * 1000,
      retry: retryPolicy,
    })),
  });
  const maxD = Math.max(...comps.map((c) => c.distance), 1);
  return (
    <Section
      id="comparisons"
      title="Similar college profiles"
      intro="The five earlier drafted quarterbacks whose final college seasons look most like his. A way to read the profile, not a forecast."
    >
      {q.isPending ? (
        <SlowLoading label="Loading comparisons" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => q.refetch()} />
      ) : q.data.status !== "available" ? (
        <Unavailable title="No comparisons available" reason={q.data.reason} />
      ) : (
        <div className="space-y-8">
          <ol className="space-y-6">
            {comps.map((c, i) => (
              <li key={`${c.name}-${c.draft_year}`} className="grid gap-3 border-t border-line pt-5 sm:grid-cols-[3.5rem_1fr_14rem]">
                <span aria-hidden className="hidden size-14 items-center justify-center rounded-full border border-line bg-panel font-display text-[20px] font-semibold text-chalk sm:flex">
                  {initials(c.name)}
                </span>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <span className="sr-only">Comparison {i + 1}:</span>
                    {c.player_id ? (
                      <Link href={`/prospects/${c.player_id}`} className="text-[22px] font-display font-semibold text-chalk hover:underline">
                        {c.name}
                      </Link>
                    ) : (
                      <span className="text-[22px] font-display font-semibold text-chalk">{c.name}</span>
                    )}
                    <span className="text-sm text-muted-foreground">
                      {c.college}, {c.draft_year} draft{c.overall_pick != null ? `, pick ${c.overall_pick}` : ""}
                    </span>
                  </div>
                  <p className="mt-2 text-[14px] text-chalk/90">
                    <span className="text-muted-foreground">Most alike: </span>
                    {c.most_similar_on}
                  </p>
                  <p className="text-[14px] text-chalk/90">
                    <span className="text-muted-foreground">Biggest differences ({c.name.split(" ").slice(-1)[0]} vs this prospect): </span>
                    {c.biggest_differences}
                  </p>
                  <p className="mt-1 text-[14px]">
                    <span className="text-muted-foreground">NFL outcome (observed): </span>
                    <span className="text-observed-text">
                      <Outcome c={c} />
                    </span>
                  </p>
                </div>
                <div className="self-center">
                  <div className="flex items-center justify-between text-[12px] text-muted-foreground">
                    <span>Profile distance</span>
                    <span className="num">{c.distance.toFixed(2)}</span>
                  </div>
                  <div className="mt-1 h-2 rounded-[2px] bg-panel" aria-hidden>
                    <div className="h-full rounded-[2px] bg-reference" style={{ width: `${(c.distance / maxD) * 100}%` }} />
                  </div>
                </div>
              </li>
            ))}
          </ol>
          <p className="flex items-start gap-1 text-[13px] text-muted-foreground">
            <InfoTip label="profile distance">
              Distance between standardized college profiles (completion %, YPA, TD and INT rates, volume, rushing, schedule,
              team context, age, height). Smaller means more alike. It is not a probability, and a comp&apos;s NFL career says
              little about this player&apos;s: in backtests, comps&apos; outcomes predicted 32+ starts only weakly.
            </InfoTip>
            Shorter bars mean closer profiles. Comparisons use only earlier draft classes and never use NFL results to pick comps.
          </p>
          <ProfileDots
            prospect={{ name: detail.prospect.name, fs: detail.final_season_statistics as Record<string, unknown> }}
            comps={comps.map((c, i) => ({ name: c.name, fs: details[i]?.data?.final_season_statistics as Record<string, unknown> | undefined }))}
          />
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full min-w-[560px] text-left text-[13px]">
              <caption className="px-3 pt-3 text-left text-sm font-medium text-chalk">Final college season, side by side</caption>
              <thead className="text-muted-foreground">
                <tr className="border-b border-line">
                  <th scope="col" className="px-3 py-2 font-medium">Quarterback</th>
                  {COLS.map(([, l]) => (
                    <th key={l} scope="col" className="px-3 py-2 font-medium">{l}</th>
                  ))}
                  <th scope="col" className="px-3 py-2 font-medium">Att</th>
                </tr>
              </thead>
              <tbody>
                {[{ name: `${detail.prospect.name} (this prospect)`, fs: detail.final_season_statistics, me: true }, ...comps.map((c, i) => ({ name: c.name, fs: details[i]?.data?.final_season_statistics, me: false }))].map((r) => (
                  <tr key={r.name} className={`border-t border-line/60 ${r.me ? "bg-panel" : ""}`}>
                    <th scope="row" className={`px-3 py-1.5 text-left ${r.me ? "font-semibold text-chalk" : "font-normal"}`}>{r.name}</th>
                    {COLS.map(([k]) => (
                      <td key={k} className="num px-3 py-1.5">{fmt1(num(r.fs as Record<string, unknown> | undefined, k))}</td>
                    ))}
                    <td className="num px-3 py-1.5">{fmtInt(num(r.fs as Record<string, unknown> | undefined, "attempts"))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Section>
  );
}

type Row = { name: string; fs: Record<string, unknown> | undefined };

/** Final-season college statistics as dot strips: the prospect (chalk ring) against each comparison (blue dots). */
function ProfileDots({ prospect, comps }: { prospect: Row; comps: Row[] }) {
  const loaded = comps.filter((c) => c.fs);
  if (!loaded.length) return null;
  return (
    <figure aria-labelledby="dots-cap">
      <figcaption id="dots-cap" className="mb-4 text-sm font-medium text-chalk">
        College profile against the five comparisons (final season)
      </figcaption>
      <div className="space-y-4">
        {COLS.map(([k, label]) => {
          const pts = [prospect, ...loaded].map((r) => ({ name: r.name, v: num(r.fs, k) })).filter((r): r is { name: string; v: number } => r.v != null);
          const me = num(prospect.fs, k);
          if (pts.length < 2 || me == null) return null;
          const lo = Math.min(...pts.map((p) => p.v));
          const hi = Math.max(...pts.map((p) => p.v));
          const pad = (hi - lo) * 0.08 || 1;
          const x = (v: number) => `${((v - (lo - pad)) / (hi - lo + 2 * pad)) * 100}%`;
          const desc = pts.map((p) => `${p.name} ${fmt1(p.v)}`).join(", ");
          return (
            <div key={k} className="grid grid-cols-[6.5rem_1fr] items-center gap-3 sm:grid-cols-[8rem_1fr]">
              <span className="text-[13px] text-muted-foreground">{label}</span>
              <div role="img" aria-label={`${label}: ${desc}`} className="relative h-8">
                <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
                {loaded.map((c) => {
                  const v = num(c.fs, k);
                  return v == null ? null : (
                    <span
                      key={c.name}
                      title={`${c.name}: ${fmt1(v)}`}
                      className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-projection ring-2 ring-background"
                      style={{ left: x(v) }}
                    />
                  );
                })}
                <span
                  title={`${prospect.name}: ${fmt1(me)}`}
                  className="absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-chalk bg-background"
                  style={{ left: x(me) }}
                />
                <span className="num absolute -top-1 text-[11px] text-chalk" style={{ left: x(me), transform: "translateX(-50%)" }}>
                  {fmt1(me)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-muted-foreground" aria-hidden>
        <li className="flex items-center gap-1.5"><span className="inline-block size-3 rounded-full border-2 border-chalk" /> This prospect</li>
        <li className="flex items-center gap-1.5"><span className="inline-block size-3 rounded-full bg-projection" /> Comparisons (hover for names)</li>
      </ul>
    </figure>
  );
}
