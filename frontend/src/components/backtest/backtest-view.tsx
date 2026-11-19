"use client";
import Link from "next/link";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import type { BacktestResponse, BacktestStat, Mode } from "@/lib/api/types";
import { CLASS_INFO } from "@/lib/backtest";
import { fmtInt, fmtProb, MODE_LABEL, sentence, statLabel } from "@/lib/format";
import { FieldRuler, RulerLegend } from "@/components/field-ruler";
import { ChartFrame } from "@/components/chart-frame";
import { cn } from "@/lib/utils";

export { CLASS_INFO };

export function ClassificationBanner({ b }: { b: BacktestResponse }) {
  const info = CLASS_INFO[b.classification.status];
  const Icon = info.icon;
  return (
    <div className={cn("flex gap-3 rounded-md border-l-4 bg-panel p-4", info.tone.split(" ")[0])} data-testid="classification">
      <Icon className={cn("mt-0.5 size-5 shrink-0", info.tone.split(" ")[1])} aria-hidden />
      <div>
        <p className="font-medium text-chalk">{info.title}</p>
        <p className="mt-1 text-[14px] text-muted-foreground">{sentence(b.classification.explanation)}</p>
        {b.classification.status === "retrospective_illustration_in_sample" && (
          <p className="mt-1 text-[14px] text-caution">
            The model that produced these numbers was trained on this player. Do not read the gaps below as forecast accuracy.
          </p>
        )}
      </div>
    </div>
  );
}

const GROUPS: { title: string; stats: string[] }[] = [
  { title: "Yards", stats: ["pass_yards", "rush_yards"] },
  { title: "Touchdowns and interceptions", stats: ["pass_tds", "interceptions", "rush_tds"] },
  { title: "Games", stats: ["games_started", "games_played"] },
];

function GroupChart({ title, stats, mode, inSample }: { title: string; stats: BacktestStat[]; mode: Mode; inSample: boolean }) {
  const data = stats.map((s) => ({
    stat: statLabel(s.stat),
    predicted: s.predicted,
    actual: s.actual,
    // whiskers stop at zero on the chart (a -30 rushing-yard bound would otherwise stretch the axis); the data table keeps exact values
    err: [Math.max(0, s.predicted - Math.max(0, s.interval_80.lower)), Math.max(0, s.interval_80.upper - s.predicted)],
    lo: s.interval_80.lower,
    hi: s.interval_80.upper,
  }));
  const predName = inSample ? "In-sample fit" : "Predicted";
  return (
    <ChartFrame
      title={`${title}: ${predName.toLowerCase()} vs actual (${MODE_LABEL[mode].toLowerCase()} view)`}
      description={data.map((d) => `${d.stat}: ${predName.toLowerCase()} ${fmtInt(d.predicted)} (80% range ${fmtInt(d.lo)} to ${fmtInt(d.hi)}), actual ${fmtInt(d.actual)}`).join("; ")}
      columns={["Statistic", predName, "80% low", "80% high", "Actual"]}
      rows={data.map((d) => [d.stat, fmtInt(d.predicted), fmtInt(d.lo), fmtInt(d.hi), fmtInt(d.actual)])}
    >
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={3}>
            <CartesianGrid vertical={false} stroke="var(--line)" strokeDasharray="2 4" />
            <XAxis dataKey="stat" tick={{ fill: "var(--slate-text)", fontSize: 12 }} axisLine={{ stroke: "var(--line)" }} tickLine={false} />
            <YAxis tick={{ fill: "var(--slate-text)", fontSize: 12 }} axisLine={false} tickLine={false} width={52} tickFormatter={(v) => fmtInt(v)} />
            <Tooltip
              cursor={{ fill: "rgb(233 229 219 / 0.05)" }}
              contentStyle={{ background: "var(--raised)", border: "1px solid var(--line)", borderRadius: 5, fontSize: 13 }}
              formatter={(v, n) => [fmtInt(Number(v)), n]}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: "var(--slate-text)" }} />
            <Bar dataKey="predicted" name={predName} fill="var(--projection)" radius={[4, 4, 0, 0]} maxBarSize={44} isAnimationActive={false}>
              <ErrorBar dataKey="err" width={8} stroke="var(--chalk)" strokeWidth={1.5} />
            </Bar>
            <Bar dataKey="actual" name="Actual" fill="var(--observed)" radius={[4, 4, 0, 0]} maxBarSize={44} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

function Inside({ s, complete }: { s: BacktestStat; complete: boolean }) {
  if (!complete || s.inside_80 == null)
    return (
      <span className="inline-flex items-center gap-1 text-muted-foreground">
        <CircleDashed className="size-3.5" aria-hidden /> Window open
      </span>
    );
  return s.inside_80 ? (
    <span className="inline-flex items-center gap-1 text-good">
      <CheckCircle2 className="size-3.5" aria-hidden /> Inside
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-destructive">
      <XCircle className="size-3.5" aria-hidden /> Outside
    </span>
  );
}

export function BacktestView({ b }: { b: BacktestResponse }) {
  const modes = (Object.keys(b.views) as Mode[]).filter((m) => m === "pre_draft" || m === "post_draft");
  const [mode, setMode] = useState<Mode>(modes.includes("pre_draft") ? "pre_draft" : modes[0]);
  const v = b.views[mode];
  const inSample = b.classification.in_sample;
  const complete = b.actual_window_complete;
  const by = new Map(v.statistics.map((s) => [s.stat, s]));
  const p = b.player as Record<string, string | number>;
  return (
    <div className="space-y-10">
      <div>
        <p className="text-[15px] text-muted-foreground">
          {String(p.college)}, {String(p.draft_year)} draft, round {String(p.round)}, pick {String(p.overall_pick)}
        </p>
        <h2 className="mt-1 text-[clamp(36px,6vw,60px)] leading-none text-chalk">
          <Link href={`/prospects/${b.player_id}`} className="hover:underline">{String(p.name)}</Link>
        </h2>
      </div>
      <ClassificationBanner b={b} />
      <dl className="grid gap-x-8 gap-y-3 text-[14px] sm:grid-cols-3">
        <div>
          <dt className="text-muted-foreground">Forecast made</dt>
          <dd className="text-chalk">{b.prediction_cutoff}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Trained on</dt>
          <dd className="text-chalk">
            {b.training_classes ? `Draft classes ${b.training_classes[0]} to ${b.training_classes[1]} (${b.n_training_qbs} QBs)` : "Final model (includes this player)"}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Compared with</dt>
          <dd className="text-chalk">
            {complete ? "His first five NFL seasons (complete)" : `His first ${b.observed_seasons} NFL season${b.observed_seasons === 1 ? "" : "s"} (five-season window still open)`}
          </dd>
        </div>
      </dl>

      <div className="flex flex-wrap items-center gap-4">
        <div role="radiogroup" aria-label="Backtest view" className="inline-flex rounded-md border border-line bg-panel p-1">
          {modes.map((m) => (
            <button
              key={m}
              role="radio"
              aria-checked={mode === m}
              onClick={() => setMode(m)}
              className={`rounded px-4 py-1.5 text-sm font-medium ${mode === m ? "bg-chalk text-background" : "text-muted-foreground hover:text-chalk"}`}
            >
              {MODE_LABEL[m]}
            </button>
          ))}
        </div>
        <p className="text-[13px] text-muted-foreground">
          {mode === "pre_draft" ? "College data only, as it would have looked before the draft." : "Also uses his actual draft slot."}{" "}
          Source: {v.source.replace(/\.$/, "")}.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        {GROUPS.map((g) => (
          <GroupChart key={g.title} title={g.title} mode={mode} inSample={inSample} stats={g.stats.map((s) => by.get(s)).filter((s): s is BacktestStat => !!s)} />
        ))}
      </div>

      <div className="space-y-6">
        <RulerLegend actual />
        <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2">
          {v.statistics.map((s) => (
            <div key={s.stat}>
              <div className="flex items-baseline justify-between text-[14px]">
                <span className="text-chalk">{statLabel(s.stat)}</span>
                <span className="num text-muted-foreground">
                  {inSample ? "fit" : "predicted"} {fmtInt(s.predicted)}, actual <span className="text-observed-text">{fmtInt(s.actual)}</span>
                </span>
              </div>
              <FieldRuler
                label={`${statLabel(s.stat)}: ${inSample ? "in-sample fit" : "prediction"} vs actual`}
                expected={s.predicted}
                intervals={[s.interval_80]}
                actual={s.actual}
                showScale={false}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full min-w-[640px] text-left text-[13px]">
          <caption className="px-3 pt-3 text-left text-sm font-medium text-chalk">
            {inSample ? "In-sample fit vs actual (not forecast accuracy)" : "Predicted vs actual, NFL seasons 1 to 5"}
          </caption>
          <thead className="text-muted-foreground">
            <tr className="border-b border-line">
              <th scope="col" className="px-3 py-2 font-medium">Statistic</th>
              <th scope="col" className="px-3 py-2 font-medium">{inSample ? "Fit" : "Predicted"}</th>
              <th scope="col" className="px-3 py-2 font-medium">80% range</th>
              <th scope="col" className="px-3 py-2 font-medium">Actual</th>
              <th scope="col" className="px-3 py-2 font-medium">Absolute error</th>
              <th scope="col" className="px-3 py-2 font-medium">In 80% range?</th>
            </tr>
          </thead>
          <tbody>
            {v.statistics.map((s) => (
              <tr key={s.stat} className="border-t border-line/60">
                <th scope="row" className="px-3 py-1.5 font-normal text-chalk">{statLabel(s.stat)}</th>
                <td className="num px-3 py-1.5">{fmtInt(s.predicted)}</td>
                <td className="num px-3 py-1.5">{fmtInt(s.interval_80.lower)} to {fmtInt(s.interval_80.upper)}</td>
                <td className="num px-3 py-1.5 text-observed-text">
                  {fmtInt(s.actual)}
                  {!complete && <span className="text-muted-foreground"> so far</span>}
                </td>
                <td className="num px-3 py-1.5">{s.error != null ? fmtInt(Math.abs(s.error)) : "Not yet known"}</td>
                <td className="px-3 py-1.5"><Inside s={s} complete={complete} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <h3 className="text-[20px] text-chalk">Starting chances {inSample ? "(in-sample)" : "at the time"}</h3>
        <ul className="mt-3 grid gap-3 sm:grid-cols-3">
          {v.start_probabilities.map((s) => (
            <li key={s.threshold} className="rounded-md border border-line p-3">
              <p className="text-[13px] text-muted-foreground">{s.threshold}+ starts in seasons 1 to 5</p>
              <p className="num text-[28px] font-semibold text-chalk">{fmtProb(s.predicted)}</p>
              <p className="text-[13px]">
                Happened:{" "}
                <span className="text-observed-text">{s.actual_event === true ? "yes" : s.actual_event === false ? "no" : "not yet known"}</span>
              </p>
            </li>
          ))}
        </ul>
      </div>
      <p className="text-[13px] text-muted-foreground">
        Only five-season totals were forecast and tested. Season-by-season and three-season comparisons are not shown because no
        such forecasts were backtested.
      </p>
    </div>
  );
}
