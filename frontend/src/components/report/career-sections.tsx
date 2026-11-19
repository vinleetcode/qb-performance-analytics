"use client";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ForecastResponse } from "@/lib/api/types";
import { fmtInt, fmtProb, sentence, statLabel } from "@/lib/format";
import { RulerLegend } from "@/components/field-ruler";
import { ChartFrame } from "@/components/chart-frame";
import { Unavailable } from "@/components/status";
import { ProbBar } from "./draft-section";
import { Section } from "./section";
import { HeadlineStats, IntervalExplorer, ProjectionTable, ReadingGuide } from "./production-forecast";


export function StartsSection({ f }: { f: ForecastResponse }) {
  const c = f.career;
  const s = c.start_probabilities;
  return (
    <Section
      id="starts"
      title="Starting chances"
      intro="Probability of reaching each number of regular-season starts in his first five NFL seasons."
    >
      {c.status !== "available" || !s ? (
        <Unavailable title="Starting probabilities are not available" reason={c.reason} />
      ) : (
        <div className="space-y-6">
          <div className="space-y-4" role="list" aria-label="Probability of reaching start totals in NFL seasons 1 to 5">
            {(
              [
                ["16+ starts", s.p_16_plus, "About one full season as the starter"],
                ["32+ starts", s.p_32_plus, "About two seasons as the starter"],
                ["64+ starts", s.p_64_plus, "About four seasons as the starter"],
              ] as const
            ).map(([l, p, sub], i) => (
              <div role="listitem" key={l}>
                <ProbBar label={l} sub={sub} p={p} color={["var(--projection)", "rgb(76 139 209 / 0.72)", "rgb(76 139 209 / 0.5)"][i]} />
              </div>
            ))}
          </div>
          <p className="text-[13px] text-muted-foreground">
            {s.p_any_nfl_game != null && <>Chance of playing in at least one NFL game: {fmtProb(s.p_any_nfl_game)}. </>}
            The three probabilities are nested: reaching 64 starts implies reaching 32 and 16. Window: {s.window}.
          </p>
          {c.reference_v2?.status === "available" && c.reference_v2.p_32_plus != null && (
            <p className="text-[13px] text-muted-foreground">
              Reference: the earlier career-success model ({c.reference_v2.model_version}), which uses college data only and
              never the draft slot, gives {fmtProb(c.reference_v2.p_32_plus)} for 32+ starts.
            </p>
          )}
        </div>
      )}
    </Section>
  );
}

export function ProductionSection({ f }: { f: ForecastResponse }) {
  const c = f.career;
  const by = useMemo(() => new Map(c.five_year_totals.map((s) => [s.stat, s])), [c.five_year_totals]);
  return (
    <Section
      id="production"
      title="Five-season production"
      intro="What his first five NFL seasons could add up to: the expected total for each statistic and the ranges it should fall in."
    >
      {c.status !== "available" ? (
        <Unavailable title="Production forecasts are not available" reason={c.reason} />
      ) : (
        <div className="space-y-10">
          {c.hypothetical && (
            <p className="rounded-md border-l-2 border-caution bg-panel/60 px-3 py-2 text-sm text-chalk">
              Hypothetical: these forecasts assume a draft slot you chose, not his real one.
            </p>
          )}
          <div className="space-y-3">
            <HeadlineStats stats={by} />
            <RulerLegend reference={c.five_year_totals.some((s) => s.upper_bound_beyond_training_max) ? "Historical maximum" : undefined} />
            {c.five_year_totals.some((s) => s.upper_bound_beyond_training_max) && (
              <p className="text-[13px] text-muted-foreground">
                Where a dashed line appears, the upper range runs past the largest five-season total any of the 183 training
                quarterbacks reached. Read those upper ends as &ldquo;at least the historical maximum&rdquo;, not as precise values.
              </p>
            )}
          </div>
          <IntervalExplorer stats={by} />
          <ReadingGuide stats={c.five_year_totals} />
          <ProjectionTable stats={c.five_year_totals} />
          <YearByYear f={f} />
          {c.efficiency_note && <p className="text-[13px] text-muted-foreground">{sentence(c.efficiency_note)}</p>}
        </div>
      )}
    </Section>
  );
}

const YEAR_STATS = ["pass_yards", "games_started", "pass_tds", "interceptions", "rush_yards"];

function YearByYear({ f }: { f: ForecastResponse }) {
  const y = f.career.year_by_year;
  const [stat, setStat] = useState("pass_yards");
  if (!y || y.status !== "available")
    return <Unavailable title="Season-by-season path not available" reason={y?.reason} />;
  const data = y.seasons.map((s) => {
    const v = s.stats[stat];
    return { season: `Season ${s.season}`, expected: v.expected, err: [v.expected - v.interval_80.lower, v.interval_80.upper - v.expected], lo: v.interval_80.lower, hi: v.interval_80.upper };
  });
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2" role="group" aria-label="Statistic for the season-by-season chart">
        {YEAR_STATS.map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setStat(k)}
            aria-pressed={stat === k}
            className={`rounded px-2.5 py-1 text-[13px] ${stat === k ? "bg-raised text-chalk" : "text-muted-foreground hover:text-chalk"}`}
          >
            {statLabel(k)}
          </button>
        ))}
      </div>
      <ChartFrame
        title={`${statLabel(stat)} by NFL season, expected value with 80% range`}
        description={`Season-by-season forecast of ${statLabel(stat).toLowerCase()}: ${data.map((d) => `${d.season} ${fmtInt(d.expected)} (80% range ${fmtInt(d.lo)} to ${fmtInt(d.hi)})`).join("; ")}.`}
        columns={["Season", "Expected", "80% low", "80% high"]}
        rows={data.map((d) => [d.season, fmtInt(d.expected), fmtInt(d.lo), fmtInt(d.hi)])}
      >
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--line)" strokeDasharray="2 4" />
              <XAxis dataKey="season" tick={{ fill: "var(--slate-text)", fontSize: 12 }} axisLine={{ stroke: "var(--line)" }} tickLine={false} />
              <YAxis tick={{ fill: "var(--slate-text)", fontSize: 12 }} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => fmtInt(v)} />
              <Tooltip
                cursor={{ fill: "rgb(233 229 219 / 0.05)" }}
                contentStyle={{ background: "var(--raised)", border: "1px solid var(--line)", borderRadius: 5, fontSize: 13 }}
                formatter={(v) => [fmtInt(Number(v)), "Expected"]}
              />
              <Bar dataKey="expected" fill="var(--projection)" radius={[4, 4, 0, 0]} maxBarSize={56} isAnimationActive={false}>
                <ErrorBar dataKey="err" width={10} stroke="var(--chalk)" strokeWidth={1.5} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartFrame>
      <p className="mt-2 text-[12px] text-muted-foreground">
        Whiskers show the 80% range for each season. {sentence(y.method)} {sentence(y.interval_note)}
      </p>
    </div>
  );
}
