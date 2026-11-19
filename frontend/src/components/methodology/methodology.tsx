"use client";
import Link from "next/link";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { usePerformance } from "@/lib/api/queries";
import type { PerformanceSections } from "@/lib/api/types";
import { fmt3, fmtPct, statLabel } from "@/lib/format";
import { ChartFrame } from "@/components/chart-frame";
import { ErrorState, SlowLoading } from "@/components/status";
import { GITHUB_URL } from "@/lib/config";
import { PHOTOS } from "@/lib/images";
import { PhotoBanner } from "@/components/photo-banner";

const axis = { fill: "var(--slate-text)", fontSize: 12 };
const tip = { background: "var(--raised)", border: "1px solid var(--line)", borderRadius: 5, fontSize: 13 };

function Block({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="grid gap-6 border-t border-line py-12 lg:grid-cols-[15rem_1fr]">
      <h2 id={`${id}-h`} className="text-[30px] text-chalk">{title}</h2>
      <div className="min-w-0 space-y-5 text-[15px] text-chalk/90">{children}</div>
    </section>
  );
}

function Status({ text }: { text?: string }) {
  if (!text) return null;
  return <p className="rounded-md border-l-2 border-reference bg-panel/60 px-3 py-2 text-[13px] text-muted-foreground">How this was evaluated: {text}.</p>;
}

export function EmbargoTimeline() {
  const years = Array.from({ length: 2018 - 2005 + 1 }, (_, i) => 2005 + i);
  return (
    <figure aria-labelledby="tl-cap">
      <figcaption id="tl-cap" className="mb-3 text-sm font-medium text-chalk">Example: forecasting the 2018 draft class</figcaption>
      <ol className="flex gap-[3px]" aria-label="Draft classes 2005 to 2018 and how each is used">
        {years.map((y) => {
          const role = y <= 2013 ? "train" : y < 2018 ? "gap" : "test";
          return (
            <li key={y} className="min-w-0 flex-1">
              <div
                className={`h-7 rounded-[2px] ${role === "train" ? "bg-projection" : role === "gap" ? "border border-dashed border-line" : "bg-observed"}`}
                aria-hidden
              />
              <span className="num mt-1 block truncate text-center text-[11px] text-muted-foreground">{String(y).slice(2)}</span>
              <span className="sr-only">
                {y}: {role === "train" ? "used for training" : role === "gap" ? "not used: five NFL seasons not finished by 2018" : "forecast"}
              </span>
            </li>
          );
        })}
      </ol>
      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-muted-foreground" aria-hidden>
        <li className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-[2px] bg-projection" /> Trained on (2005 to 2013)</li>
        <li className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-[2px] border border-dashed border-line" /> Outcomes not yet known in 2018</li>
        <li className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-[2px] bg-observed" /> Forecast, then scored</li>
      </ul>
    </figure>
  );
}

function DraftProbChart({ s }: { s: PerformanceSections["draft_probability"] }) {
  const label: Record<string, string> = {
    A_college: "College stats only",
    B_college_invited: "+ Combine invitation",
    C_college_invited_measures: "+ invitation and measurements",
  };
  const data = s.models.map((m) => ({ name: label[m.variant], auc: m.auc, err: [m.auc - m.auc_lo, m.auc_hi - m.auc], brier: m.brier, lo: m.auc_lo, hi: m.auc_hi }));
  return (
    <ChartFrame
      title="Draft probability: ranking accuracy (AUC, where 0.5 is chance and 1 is perfect) with 95% intervals"
      description={data.map((d) => `${d.name}: AUC ${fmt3(d.auc)} (95% interval ${fmt3(d.lo)} to ${fmt3(d.hi)}), Brier ${fmt3(d.brier)}`).join("; ")}
      columns={["Inputs", "AUC", "95% low", "95% high", "Brier score"]}
      rows={data.map((d) => [d.name, fmt3(d.auc), fmt3(d.lo), fmt3(d.hi), fmt3(d.brier)])}
    >
      <div className="h-60 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 28, bottom: 4, left: 8 }}>
            <CartesianGrid horizontal={false} stroke="var(--line)" strokeDasharray="2 4" />
            <XAxis type="number" domain={[0.5, 1]} tick={axis} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={170} interval={0} tick={axis} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={tip} cursor={{ fill: "rgb(233 229 219 / 0.05)" }} formatter={(v) => [fmt3(Number(v)), "AUC"]} />
            <Bar dataKey="auc" fill="var(--projection)" radius={[0, 4, 4, 0]} maxBarSize={22} isAnimationActive={false}>
              <ErrorBar dataKey="err" width={6} stroke="var(--chalk)" strokeWidth={1.5} direction="x" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

function StarterChart({ s }: { s: PerformanceSections["starter_probabilities"] }) {
  const data = [16, 32, 64].flatMap((k) =>
    (["pre_draft", "post_draft"] as const).map((fs) => {
      const m = s.find((r) => r.feature_set === fs && r.threshold === k && r.model === "nested_logit_pava")!;
      const b = s.find((r) => r.feature_set === fs && r.threshold === k && r.model === "base_rate")!;
      return { name: `${k}+ ${fs === "pre_draft" ? "pre" : "post"}`, model: m.brier, base: b.brier, auc: m.auc, n: m.n, pos: m.n_pos };
    }),
  );
  return (
    <ChartFrame
      title="Starting probabilities: Brier score (lower is better), model vs base rate"
      description={data.map((d) => `${d.name}-draft: model ${fmt3(d.model)}, base rate ${fmt3(d.base)}, AUC ${fmt3(d.auc)}, ${d.pos} of ${d.n} reached it`).join("; ")}
      columns={["Threshold and view", "Model Brier", "Base-rate Brier", "AUC", "Reached / QBs"]}
      rows={data.map((d) => [d.name, fmt3(d.model), fmt3(d.base), fmt3(d.auc), `${d.pos} / ${d.n}`])}
    >
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={3}>
            <CartesianGrid vertical={false} stroke="var(--line)" strokeDasharray="2 4" />
            <XAxis dataKey="name" tick={axis} axisLine={{ stroke: "var(--line)" }} tickLine={false} />
            <YAxis tick={axis} axisLine={false} tickLine={false} width={44} />
            <Tooltip contentStyle={tip} cursor={{ fill: "rgb(233 229 219 / 0.05)" }} formatter={(v, n) => [fmt3(Number(v)), n]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="model" name="Model" fill="var(--projection)" radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
            <Bar dataKey="base" name="Base rate" fill="var(--reference)" radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

function SkillChart({ s }: { s: PerformanceSections["production"] }) {
  const stats = Array.from(new Set(s.errors.map((e) => e.statistic)));
  const data = stats.map((st) => {
    const pre = s.errors.find((e) => e.statistic === st && e.view === "pre_draft")!;
    const post = s.errors.find((e) => e.statistic === st && e.view === "post_draft")!;
    return {
      stat: statLabel(st),
      pre: 1 - pre.mae_nested_selection / pre.mae_historical_median,
      post: 1 - post.mae_nested_selection / post.mae_historical_median,
    };
  });
  return (
    <ChartFrame
      title="Five-season totals: error reduction vs the historical median, before and after the draft"
      description={data.map((d) => `${d.stat}: ${fmtPct(d.pre)} before the draft, ${fmtPct(d.post)} after`).join("; ")}
      columns={["Statistic", "Pre-draft", "Post-draft"]}
      rows={data.map((d) => [d.stat, fmtPct(d.pre), fmtPct(d.post)])}
    >
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }} barGap={2}>
            <CartesianGrid horizontal={false} stroke="var(--line)" strokeDasharray="2 4" />
            <XAxis type="number" domain={[-0.1, 0.5]} tickFormatter={(v) => fmtPct(v)} tick={axis} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="stat" width={110} interval={0} tick={axis} axisLine={false} tickLine={false} />
            <ReferenceLine x={0} stroke="var(--reference)" />
            <Tooltip contentStyle={tip} cursor={{ fill: "rgb(233 229 219 / 0.05)" }} formatter={(v, n) => [fmtPct(Number(v)), n]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="pre" name="Pre-draft" fill="rgb(76 139 209 / 0.45)" stroke="var(--projection)" radius={[0, 3, 3, 0]} maxBarSize={10} isAnimationActive={false} />
            <Bar dataKey="post" name="Post-draft" fill="var(--projection)" radius={[0, 3, 3, 0]} maxBarSize={10} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

function CoverageChart({ s }: { s: PerformanceSections["interval_calibration"] }) {
  const rows = s.coverage_by_statistic;
  const stats = Array.from(new Set(rows.map((r) => r.target)));
  const data = stats.map((t) => ({
    stat: statLabel(t),
    pre: rows.find((r) => r.target === t && r.feature_set === "pre_draft")?.coverage80 ?? 0,
    post: rows.find((r) => r.target === t && r.feature_set === "post_draft")?.coverage80 ?? 0,
  }));
  return (
    <ChartFrame
      title="How often the 80% range contained the real five-season total"
      description={data.map((d) => `${d.stat}: ${fmtPct(d.pre)} pre-draft, ${fmtPct(d.post)} post-draft; nominal 80%`).join("; ")}
      columns={["Statistic", "Pre-draft", "Post-draft"]}
      rows={data.map((d) => [d.stat, fmtPct(d.pre), fmtPct(d.post)])}
    >
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }} barGap={2}>
            <CartesianGrid horizontal={false} stroke="var(--line)" strokeDasharray="2 4" />
            <XAxis type="number" domain={[0.5, 1]} tickFormatter={(v) => fmtPct(v)} tick={axis} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="stat" width={110} interval={0} tick={axis} axisLine={false} tickLine={false} />
            <ReferenceLine x={0.8} stroke="var(--chalk)" strokeDasharray="4 3" label={{ value: "80% nominal", fill: "var(--slate-text)", fontSize: 11, position: "top" }} />
            <Tooltip contentStyle={tip} cursor={{ fill: "rgb(233 229 219 / 0.05)" }} formatter={(v, n) => [fmtPct(Number(v)), n]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="pre" name="Pre-draft" fill="rgb(76 139 209 / 0.45)" stroke="var(--projection)" radius={[0, 3, 3, 0]} maxBarSize={10} isAnimationActive={false} />
            <Bar dataKey="post" name="Post-draft" fill="var(--projection)" radius={[0, 3, 3, 0]} maxBarSize={10} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

export function Methodology() {
  const q = usePerformance();
  return (
    <>
    <PhotoBanner
      photo={PHOTOS.practiceField}
      priority
      title="How the forecasts are built and tested"
      intro={
        <p>
          For recruiters and data scientists alike: the data, the models, how they were tested, and how far to trust them.
          Every number on this page comes from the backend&apos;s evaluation summary; draft classes are small, about ten
          quarterbacks a year, so every estimate is noisy.
        </p>
      }
    />
    <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6">
      {q.isPending ? (
        <div className="mt-10"><SlowLoading label="Loading evaluation results" lines={6} /></div>
      ) : q.isError ? (
        <ErrorState className="mt-10" error={q.error} onRetry={() => q.refetch()} />
      ) : (
        <Content s={q.data.sections} status={q.data.evaluation_status} limits={q.data.limitations} />
      )}
    </div>
    </>
  );
}

const PIPELINE = [
  ["Ingest", "CFBD college stats, games, recruiting and team context; nflverse draft board, NFL seasons, starts and Combine"],
  ["Link", "Each drafted quarterback matched to his college record, with identity checked across sources"],
  ["Features", "Final-season and career profiles, opponent-adjusted efficiency and team context; nothing from after the draft"],
  ["Train", "Regularised regression and ordinal models; the model family chosen inside earlier classes only"],
  ["Test", "Chronological backtests with a five-season label embargo; ranges calibrated on held-out errors"],
  ["Serve", "Versioned, hash-verified artifacts behind a read-only FastAPI service; this site renders its responses"],
] as const;

export function PipelineDiagram() {
  return (
    <figure aria-labelledby="pipe-cap">
      <figcaption id="pipe-cap" className="mb-4 text-sm font-medium text-chalk">From raw data to a served forecast</figcaption>
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {PIPELINE.map(([t, d], i) => (
          <li key={t} className="rounded-md border border-line bg-panel/60 p-4">
            <span className="num text-[13px] text-projection-text">Stage {i + 1}</span>
            <p className="mt-0.5 font-display text-[20px] font-semibold text-chalk">{t}</p>
            <p className="mt-1 text-[13px] text-muted-foreground">{d}</p>
          </li>
        ))}
      </ol>
    </figure>
  );
}

function Content({ s, status, limits }: { s: PerformanceSections; status: Record<string, string>; limits: string[] }) {
  const dp = s.draft_probability;
  const A = dp.models.find((m) => m.variant === "A_college")!;
  const B = dp.models.find((m) => m.variant === "B_college_invited")!;
  const C = dp.models.find((m) => m.variant === "C_college_invited_measures")!;
  const share = (B.auc - A.auc) / Math.max(C.auc - A.auc, 1e-9);
  const r3 = s.draft_round.three_groups;
  const sel3 = r3.find((r) => r.model === "selected")!;
  const freq3 = r3.find((r) => r.model === "freq")!;
  const pk = s.pick.errors;
  const pickModel = pk.find((p) => p.model.startsWith("ridge"))!;
  const pickBase = pk.find((p) => p.model === "historical_median")!;
  const pickCov = s.pick.coverage.find((c) => c.interval_method === s.pick.served_interval_method)!;
  const st = (fs: "pre_draft" | "post_draft", k: number) =>
    s.starter_probabilities.find((r) => r.feature_set === fs && r.threshold === k && r.model === "nested_logit_pava")!;
  return (
    <>
      <Block id="motivation" title="Why build this">
        <p>
          Teams spend first-round picks on quarterbacks with almost no public, testable forecast of how their careers will go.
          This project asks a narrower, answerable question: from what was knowable before the draft, how likely is a college
          quarterback to be drafted, to start, and to produce in his first five NFL seasons, and how wrong are those forecasts
          when checked against real careers?
        </p>
        <PipelineDiagram />
      </Block>

      <Block id="findings" title="What the evidence says">
        <ul className="list-disc space-y-2 pl-5">
          <li>College statistics carry real signal about who gets drafted (AUC {fmt3(A.auc)} on drafts the model had not seen).</li>
          <li>Combine participation, not 40 times or size, accounts for most of the post-Combine improvement.</li>
          <li>The actual draft slot sharply improves post-draft career forecasts, because teams give early picks the job.</li>
          <li>Before the draft, career production forecasts are only modestly better than historical averages.</li>
          <li>NFL efficiency rates (completion %, yards per attempt) could not be predicted reliably from this sample.</li>
          <li>Historical comparisons explain a profile; as a standalone predictor they are weak.</li>
        </ul>
      </Block>

      <Block id="data" title="Data">
        <p>
          College statistics, schedules, recruiting rankings and team context come from CollegeFootballData.com. Draft results, NFL
          statistics, starts and NFL Combine results come from nflverse. The models use only information available before each
          draft. Data licences for public display have not been reviewed yet.
        </p>
        <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
          <li>Career models: drafted quarterbacks from the 2005 to 2021 classes, whose five NFL seasons are complete.</li>
          <li>Draft probability: {dp.population}.</li>
          <li>Round and pick: drafted quarterbacks from 2005 to 2026 with a final college season in the data.</li>
          <li>Comparisons: earlier drafted quarterbacks with a complete college profile.</li>
        </ul>
        <p>
          Inputs describe the final college season and the career before it: completion rate, yards per attempt, touchdown and
          interception rates, yards per attempt adjusted for opponents, schedule strength, volume, rushing, team passing rate,
          team recruiting strength, age and height. Before the draft, the Combine adds invitation status and measurements; after
          the draft, the pick itself becomes an input.
        </p>
      </Block>

      <Block id="backtesting" title="Testing on the future">
        <p>
          Each forecast is tested the way it would have been used: on a draft class the model had not seen, trained only on
          classes whose results were already known. A five-season outcome is not known until five seasons later, so the 2018
          class is forecast from classes up to 2013.
        </p>
        <EmbargoTimeline />
        <p>
          Model choices (which model family, which interval method) were also made inside those earlier classes. The
          individual backtests on the <Link href="/backtesting" className="text-projection-text underline decoration-1 hover:decoration-2">backtesting page</Link>{" "}
          come from exactly this procedure.
        </p>
      </Block>

      <Block id="draft" title="Who gets drafted">
        <p>
          Tested on {dp.test_rows} final college seasons ({dp.test_drafted} drafted) in drafts 2014 to 2025. College statistics alone
          rank drafted above undrafted quarterbacks with an AUC of {fmt3(A.auc)}. Adding the Combine raises that to {fmt3(C.auc)}.
          Guessing the historical draft rate scores a Brier score of {fmt3(dp.baseline_brier_historical_rate)}; the model scores{" "}
          {fmt3(A.brier)} before the Combine and {fmt3(C.brier)} after it.
        </p>
        <DraftProbChart s={dp} />
        <p>
          <strong className="font-semibold text-chalk">The Combine finding.</strong> Adding only whether a player was invited
          to the Combine accounts for {fmtPct(Math.min(share, 1))} of the improvement. 40 times, height and weight add almost
          nothing on top, and in the project&apos;s Combine audit the measurements did not help at all without the invitation flag. An invitation is the NFL&apos;s own
          scouting verdict, published about two months before the draft, so the post-Combine number mostly tells you what scouts
          had already decided.
        </p>
        <Status text={status.draft_probability} />
      </Block>

      <Block id="round" title="Round and pick">
        <p>
          For drafted quarterbacks, the three-group round forecast (round 1, rounds 2 to 3, rounds 4 to 7) scores a ranked
          probability score of {fmt3(sel3.rps)} (95% interval {fmt3(sel3.rps_lo)} to {fmt3(sel3.rps_hi)}) against {fmt3(freq3.rps)}{" "}
          for historical round frequencies, on {sel3.n} quarterbacks. Single-round probabilities rest on one to three quarterbacks per
          round per year, so the site shows the three groups.
        </p>
        <p>
          The median miss on the overall pick is {Math.round(pickModel.median_abs_error_picks)} picks, against{" "}
          {Math.round(pickBase.median_abs_error_picks)} for always guessing the historical median pick. Pick ranges are wide, and
          the 80% range held the real pick {fmtPct(pickCov.coverage80)} of the time.
        </p>
        <Status text={status.draft_round_and_pick} />
      </Block>

      <Block id="starts" title="Starting chances">
        <p>
          Probabilities of 16, 32 and 64 or more starts in the first five seasons, tested on {st("pre_draft", 32).n} quarterbacks from
          the 2012 to 2021 classes. Before the draft, the 32-start model separates starters from non-starters with an AUC of{" "}
          {fmt3(st("pre_draft", 32).auc)}. Adding the draft slot raises it to {fmt3(st("post_draft", 32).auc)}.
        </p>
        <StarterChart s={s.starter_probabilities} />
        <p>
          <strong className="font-semibold text-chalk">Why the draft slot matters so much.</strong> NFL teams decide who plays,
          and they give their expensive early picks the job. The pick therefore carries information about opportunity that no
          college statistic has. The 64+ start forecast is the weakest: only {st("pre_draft", 64).n_pos} of{" "}
          {st("pre_draft", 64).n} test quarterbacks got there, and the pre-draft version is poorly calibrated (calibration slope{" "}
          {st("pre_draft", 64).calibration_slope?.toFixed(2)}, where 1 is ideal).
        </p>
        <Status text={status.starter_probabilities} />
      </Block>

      <Block id="production" title="Five-season production">
        <p>
          Ten totals over NFL seasons 1 to 5 (passing yards, touchdowns, interceptions, attempts, completions, rushing yards and
          touchdowns, rush attempts, games and starts). The chart shows how much smaller the average error is than always guessing
          the historical median, using the same fair procedure as the backtests.
        </p>
        <SkillChart s={s.production} />
        <p>
          Before the draft, the gain over the median is small and not statistically established. After the draft it is
          substantial for every statistic. Efficiency rates (completion %, yards per attempt) are not forecast: they could not be predicted from
          college data at this sample size.
        </p>
        <Status text={status.production} />
      </Block>

      <Block id="calibration" title="Calibration">
        <p>
          A forecast range is useful only if it is honest. On average the 80% ranges held the real total{" "}
          {fmtPct(s.interval_calibration.mean_coverage.pre_draft.coverage80)} of the time before the draft and{" "}
          {fmtPct(s.interval_calibration.mean_coverage.post_draft.coverage80)} after. Rushing ranges run a little narrow because
          recent quarterbacks run more than earlier ones.
        </p>
        <CoverageChart s={s.interval_calibration} />
        <Status text={status.intervals} />
      </Block>

      <Block id="comparisons" title="Comparisons">
        <p>
          The five most similar earlier quarterbacks are found by distance between standardized college profiles. They explain
          a profile; they do not forecast a career. In an exploratory test, the comps&apos; own starting rate predicted 32+ starts with
          an AUC of only {fmt3(s.comparisons.knn_outcome_rate_auc_32plus_starts)}.
        </p>
        <Status text={status.comparisons} />
      </Block>

      <Block id="bias" title="Bias and missing data">
        <ul className="list-disc space-y-2 pl-5">
          <li>
            <strong className="font-semibold text-chalk">Who is in the data.</strong> Career models see only drafted quarterbacks,
            so they describe careers conditional on being drafted. Undrafted players are not modelled.
          </li>
          <li>
            <strong className="font-semibold text-chalk">Eligibility.</strong> Declarations, transfers and remaining eligibility are
            not recorded, so draft probability is defined only for players whose college career verifiably ended after four or
            more years. Everyone else is marked not available rather than guessed.
          </li>
          <li>
            <strong className="font-semibold text-chalk">Missing values.</strong> Combine results exist only for invited players,
            so a missing 40 time is itself a sign of not being invited. That is why the participation flag dominates.
          </li>
          <li>
            <strong className="font-semibold text-chalk">Era.</strong> Passing efficiency and quarterback rushing have risen over
            time, so older classes are imperfect guides to newer ones; rushing ranges run narrow for this reason.
          </li>
        </ul>
      </Block>

      <Block id="reading-uncertainty" title="Reading the uncertainty">
        <p>
          Comparisons with a baseline come with a 95% interval from resampling whole draft classes. When that interval includes
          zero, the data cannot rule out that the model is no better than the baseline, even if its average error is lower.
          That is the case for most pre-draft career totals: the errors are a little smaller than the historical median&apos;s,
          but not reliably so. After the draft, for the model family the site uses, the intervals exclude zero for every statistic.
        </p>
        <p>
          The ranges on each report (50, 80 and 90%) were checked the same way: on later draft classes, the 80% range held the
          real five-season total about four times in five.
        </p>
      </Block>

      <Block id="limits" title="Limitations">
        <ul className="list-disc space-y-1.5 pl-5">
          {limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <p className="text-muted-foreground">
          Model versions:{" "}
          {Object.entries(s.model_versions)
            .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`)
            .join("; ")}
          . Code, evaluation reports and tests are on{" "}
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="text-projection-text underline decoration-1 hover:decoration-2">GitHub</a>.
        </p>
      </Block>
    </>
  );
}
