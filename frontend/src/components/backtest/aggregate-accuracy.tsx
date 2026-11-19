"use client";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { usePerformance } from "@/lib/api/queries";
import type { Mode } from "@/lib/api/types";
import { avgSkill } from "@/lib/backtest";
import { fmt1, fmtInt, fmtPct, MODE_LABEL, statLabel } from "@/lib/format";
import { ChartFrame } from "@/components/chart-frame";
import { ErrorState, SlowLoading } from "@/components/status";
import { InfoTip } from "@/components/info-tip";

export { avgSkill };

/** Aggregate accuracy over every eligible historical backtest, exactly as reported by /api/models/performance. */
export function AggregateAccuracy() {
  const q = usePerformance();
  const [mode, setMode] = useState<Mode>("pre_draft");
  if (q.isPending) return <SlowLoading label="Loading accuracy summary" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const s = q.data.sections;
  const rows = s.production.errors.filter((e) => e.view === mode);
  const cov = s.interval_calibration.mean_coverage[mode];
  // skill = 1 - MAE_model / MAE_median, computed from the two MAEs the backend reports
  const data = rows.map((r) => ({ stat: statLabel(r.statistic), skill: 1 - r.mae_nested_selection / r.mae_historical_median }));
  const passYds = rows.find((r) => r.statistic === "pass_yards");
  return (
    <div className="space-y-6">
      <p className="measure text-[15px] text-chalk/90">
        One player&apos;s forecast says little about a model. These are the averages over all {s.production.test_qbs} drafted
        quarterbacks in the {s.production.test_classes[0]} to {s.production.test_classes[1]} classes, each forecast by models
        trained only on earlier classes whose outcomes were known at the time.
      </p>
      <div role="radiogroup" aria-label="Accuracy view" className="inline-flex rounded-md border border-line bg-panel p-1">
        {(["pre_draft", "post_draft"] as Mode[]).map((m) => (
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
      <ChartFrame
        title={`Error reduction against always guessing the historical median (${MODE_LABEL[mode].toLowerCase()})`}
        description={data.map((d) => `${d.stat}: ${fmtPct(d.skill)}`).join("; ")}
        columns={["Statistic", "Error reduction vs median"]}
        rows={data.map((d) => [d.stat, fmtPct(d.skill)])}
      >
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
              <CartesianGrid horizontal={false} stroke="var(--line)" strokeDasharray="2 4" />
              <XAxis type="number" tickFormatter={(v) => fmtPct(v)} tick={{ fill: "var(--slate-text)", fontSize: 12 }} axisLine={false} tickLine={false} domain={[-0.1, 0.5]} />
              <YAxis type="category" dataKey="stat" width={110} interval={0} tick={{ fill: "var(--slate-text)", fontSize: 12 }} axisLine={false} tickLine={false} />
              <ReferenceLine x={0} stroke="var(--reference)" />
              <Tooltip
                cursor={{ fill: "rgb(233 229 219 / 0.05)" }}
                contentStyle={{ background: "var(--raised)", border: "1px solid var(--line)", borderRadius: 5, fontSize: 13 }}
                formatter={(v) => [fmtPct(Number(v)), "Error reduction"]}
              />
              <Bar dataKey="skill" fill="var(--projection)" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartFrame>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full min-w-[600px] text-left text-[13px]">
          <caption className="px-3 pt-3 text-left text-sm font-medium text-chalk">Mean absolute error, five-season totals ({MODE_LABEL[mode].toLowerCase()})</caption>
          <thead className="text-muted-foreground">
            <tr className="border-b border-line">
              <th scope="col" className="px-3 py-2 font-medium">Statistic</th>
              <th scope="col" className="px-3 py-2 font-medium">
                Model, as backtested{" "}
                <InfoTip label="model error">
                  The same procedure that produced the individual backtests on this page: the model family was chosen inside earlier
                  classes only. This is the fair number.
                </InfoTip>
              </th>
              <th scope="col" className="px-3 py-2 font-medium">Historical median</th>
              <th scope="col" className="px-3 py-2 font-medium">
                Shipped model family{" "}
                <InfoTip label="hindsight number">
                  The family the live model uses, applied to every class. It was chosen after seeing these classes, so it is
                  optimistic.
                </InfoTip>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.statistic} className="border-t border-line/60">
                <th scope="row" className="px-3 py-1.5 font-normal text-chalk">{statLabel(r.statistic)}</th>
                <td className="num px-3 py-1.5">{r.mae_nested_selection < 100 ? fmt1(r.mae_nested_selection) : fmtInt(r.mae_nested_selection)}</td>
                <td className="num px-3 py-1.5">{r.mae_historical_median < 100 ? fmt1(r.mae_historical_median) : fmtInt(r.mae_historical_median)}</td>
                <td className="num px-3 py-1.5 text-muted-foreground">{r.mae_shipped_family < 100 ? fmt1(r.mae_shipped_family) : fmtInt(r.mae_shipped_family)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="measure text-[14px] text-muted-foreground">
        The 80% ranges contained the real total {fmtPct(cov.coverage80)} of the time on average ({fmtPct(cov.coverage50)} for the 50%
        range, {fmtPct(cov.coverage90)} for the 90% range). On average the errors are {fmtPct(avgSkill(s.production.errors, "pre_draft"))}{" "}
        smaller than the median&apos;s before the draft and {fmtPct(avgSkill(s.production.errors, "post_draft"))} smaller once the
        draft slot is known: teams give early picks the starting job. Large misses are normal. The average error in five-season
        passing yards is {fmtInt(passYds?.mae_nested_selection)} yards in this view.
      </p>
    </div>
  );
}
