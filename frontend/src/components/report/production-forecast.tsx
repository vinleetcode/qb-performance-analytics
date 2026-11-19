"use client";
import { useId, useState } from "react";
import { AlertTriangle } from "lucide-react";
import type { StatForecast } from "@/lib/api/types";
import { fmtInt, fmtPct, statLabel } from "@/lib/format";
import {
  COVERAGE_HIGH,
  COVERAGE_LOW,
  HEADLINE_STATS,
  coverage80,
  intervalAt,
  intervalsNested,
  midSentence,
  readCoverage,
  unitFor,
} from "@/lib/forecast";
import { FieldRuler, rulerDomain } from "@/components/field-ruler";
import { InfoTip } from "@/components/info-tip";
import { cn } from "@/lib/utils";

const range = (s: StatForecast, level: number) => {
  const i = intervalAt(s, level);
  return i ? `${fmtInt(i.lower)} to ${fmtInt(i.upper)}` : "n/a";
};

/** One-line read of backtest coverage, always with an icon and words, never colour alone. */
export function CoverageFlag({ s, compact = false }: { s: StatForecast; compact?: boolean }) {
  const c = coverage80(s);
  const read = readCoverage(c);
  if (read === "narrow")
    return (
      <span className="inline-flex items-start gap-1.5 text-caution">
        <AlertTriangle className="mt-[3px] size-3.5 shrink-0" aria-hidden />
        <span>{compact ? "Runs narrow" : `Range ran narrow in backtests: ${fmtPct(c)} of real totals fell inside, against 80% intended`}</span>
      </span>
    );
  if (read === "wide") return <span className="text-muted-foreground">{compact ? "Wider than needed" : `Range ran wide in backtests (${fmtPct(c)})`}</span>;
  return null;
}

/* ------------------------------------------------------------------ headline strip */

export function HeadlineStats({ stats }: { stats: Map<string, StatForecast> }) {
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h3 className="text-[22px] text-chalk">Headline forecast</h3>
        <p className="text-[13px] text-muted-foreground">Totals over his first five NFL seasons, with the 80% range</p>
      </div>
      {/* hairline grid: a 1px gap over the rule colour draws the dividers, so cells never look like separate cards */}
      <ul className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line lg:grid-cols-5" aria-label="Headline forecast, NFL seasons 1 to 5">
        {HEADLINE_STATS.map((k, i) => {
          const s = stats.get(k);
          return (
            <li key={k} className={cn("flex min-w-0 flex-col gap-1 bg-panel p-4 sm:p-5", i === 4 && "col-span-2 lg:col-span-1")}>
              <span className="text-[14px] text-muted-foreground">{s?.label ?? statLabel(k)}</span>
              {s ? (
                <>
                  <span className="num text-[clamp(36px,4vw,46px)] leading-none font-semibold text-chalk">{fmtInt(s.expected)}</span>
                  <span className="text-[13px] text-muted-foreground"><span className="sr-only">expected </span>{unitFor(k, s.expected)}</span>
                  <span className="text-[13px] whitespace-nowrap text-chalk/85">
                    80%: <span className="num text-[14px]">{range(s, 0.8)}</span>
                  </span>
                  <FieldRuler
                    label={`${s.label}, NFL seasons 1 to 5`}
                    expected={s.expected}
                    intervals={s.intervals}
                    showScale={false}
                    showValue={false}
                    className="pt-1"
                    reference={s.upper_bound_beyond_training_max && s.training_max_5yr != null ? { value: s.training_max_5yr, label: "Historical maximum" } : null}
                  />
                  {readCoverage(coverage80(s)) === "narrow" && (
                    <span className="mt-auto pt-1 text-[12px] leading-snug">
                      <CoverageFlag s={s} compact />
                      <span className="block text-muted-foreground">{fmtPct(coverage80(s))} coverage in backtests</span>
                    </span>
                  )}
                </>
              ) : (
                <span className="mt-1 text-[14px] text-muted-foreground">Not available for this player</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}


/* ------------------------------------------------------------------ interval explorer */

function niceTicks(d0: number, d1: number, target = 5) {
  const span = d1 - d0 || 1;
  const raw = span / target;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = Math.ceil(d0 / step) * step; v <= d1 + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

export function IntervalExplorer({ stats }: { stats: Map<string, StatForecast> }) {
  const available = HEADLINE_STATS.filter((k) => stats.has(k));
  const [stat, setStat] = useState<string>(available[0] ?? "pass_yards");
  const name = useId();
  const s = stats.get(stat);
  if (!available.length) return null;
  return (
    <div className="rounded-lg border border-line bg-panel/40 p-4 sm:p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h3 className="text-[22px] text-chalk">How wide is the uncertainty?</h3>
          <p className="mt-1 text-[13px] text-muted-foreground">Pick a statistic to see its expected total and its 50% and 80% ranges on a common scale.</p>
        </div>
      </div>
      <fieldset className="mt-4">
        <legend className="sr-only">Statistic shown in the range chart</legend>
        <div className="flex flex-wrap gap-1.5">
          {available.map((k) => {
            const on = k === stat;
            return (
              <label
                key={k}
                className={cn(
                  "cursor-pointer rounded-md border px-3 py-1.5 text-[14px] select-none has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-projection-text",
                  on ? "border-projection bg-projection/15 text-chalk" : "border-line text-muted-foreground hover:border-chalk/40 hover:text-chalk",
                )}
              >
                <input type="radio" name={name} value={k} checked={on} onChange={() => setStat(k)} className="sr-only" />
                {stats.get(k)!.label}
              </label>
            );
          })}
        </div>
      </fieldset>
      {s && <IntervalChart s={s} />}
    </div>
  );
}

export function IntervalChart({ s }: { s: StatForecast }) {
  const a = intervalAt(s, 0.5);
  const b = intervalAt(s, 0.8);
  const ref = s.upper_bound_beyond_training_max && s.training_max_5yr != null ? s.training_max_5yr : null;
  const [d0, d1] = rulerDomain([s.expected, ref, b?.lower, b?.upper, a?.lower, a?.upper]);
  const pos = (v: number) => Math.max(0, Math.min(100, ((v - d0) / (d1 - d0 || 1)) * 100));
  const ticks = niceTicks(d0, d1);
  const unit = unitFor(s.stat);
  const nested = intervalsNested(s);
  const c = coverage80(s);
  const label =
    `Range chart for ${midSentence(s.label)} over NFL seasons 1 to 5: expected ${fmtInt(s.expected)} ${unit}` +
    (a ? `; 50% range ${fmtInt(a.lower)} to ${fmtInt(a.upper)}` : "") +
    (b ? `; 80% range ${fmtInt(b.lower)} to ${fmtInt(b.upper)}` : "") +
    (ref != null ? `; historical maximum ${fmtInt(ref)}` : "") +
    `. Scale ${fmtInt(d0)} to ${fmtInt(d1)} ${unit}.`;
  const expectedLeft = pos(s.expected);
  return (
    <div className="mt-6">
      <p className="flex flex-wrap items-baseline gap-x-2">
        <span className="num text-[40px] leading-none font-semibold text-chalk">{fmtInt(s.expected)}</span>
        <span className="text-[15px] text-muted-foreground">expected {unit}, NFL seasons 1 to 5</span>
      </p>
      <div role="img" aria-label={label} className="relative mt-5 px-1">
        {/* expected value label, kept inside the frame near the edges */}
        <div className="relative h-6">
          <span
            className={cn(
              "num absolute top-0 text-[13px] font-semibold whitespace-nowrap text-chalk",
              expectedLeft < 12 ? "translate-x-0" : expectedLeft > 88 ? "-translate-x-full" : "-translate-x-1/2",
            )}
            style={{ left: `${expectedLeft}%` }}
          >
            Expected {fmtInt(s.expected)}
          </span>
        </div>
        <div className="relative h-14">
          {/* gridlines at each tick */}
          {ticks.map((t) => (
            <div key={t} className="absolute inset-y-0 w-px bg-line/70" style={{ left: `${pos(t)}%` }} aria-hidden />
          ))}
          <div className="absolute inset-x-0 top-1/2 h-px bg-line" aria-hidden />
          {b && (
            <div
              className="absolute top-[8px] bottom-[8px] rounded-[3px]"
              style={{ left: `${pos(b.lower)}%`, width: `${Math.max(0.6, pos(b.upper) - pos(b.lower))}%`, background: "var(--projection-80)" }}
            />
          )}
          {a && (
            <div
              className="absolute top-[8px] bottom-[8px] rounded-[3px]"
              style={{ left: `${pos(a.lower)}%`, width: `${Math.max(0.6, pos(a.upper) - pos(a.lower))}%`, background: "var(--projection-50)" }}
            />
          )}
          {ref != null && (
            <div className="absolute -inset-y-1 border-l-2 border-dashed border-reference" style={{ left: `${pos(ref)}%` }} />
          )}
          {/* expected: a chalk bar with a 2px surface ring so it stays distinct over the bands */}
          <div
            className="absolute inset-y-0 w-[4px] -translate-x-1/2 rounded-full bg-chalk ring-2 ring-background"
            style={{ left: `${expectedLeft}%` }}
          />
        </div>
        <div className="relative mt-1 h-5 text-[12px] text-muted-foreground" aria-hidden>
          {ticks.map((t, i) => (
            <span
              key={t}
              className={cn("num absolute top-0", i === 0 ? "translate-x-0" : i === ticks.length - 1 && pos(t) > 95 ? "-translate-x-full" : "-translate-x-1/2")}
              style={{ left: `${pos(t)}%` }}
            >
              {fmtInt(t)}
            </span>
          ))}
        </div>
        <p className="mt-1 text-right text-[12px] text-muted-foreground" aria-hidden>
          {unit}
        </p>
      </div>
      <dl className="mt-4 grid gap-x-8 gap-y-3 text-[14px] sm:grid-cols-3">
        <div>
          <dt className="flex items-center gap-2 text-muted-foreground">
            <span className="inline-block h-3 w-5 rounded-[2px]" style={{ background: "var(--projection-50)" }} aria-hidden />
            50% range
          </dt>
          <dd className="num mt-0.5 text-[17px] text-chalk">{a ? `${fmtInt(a.lower)} to ${fmtInt(a.upper)}` : "n/a"} <span className="text-[13px] text-muted-foreground">{unit}</span></dd>
        </div>
        <div>
          <dt className="flex items-center gap-2 text-muted-foreground">
            <span className="inline-block h-3 w-5 rounded-[2px]" style={{ background: "var(--projection-80)" }} aria-hidden />
            80% range
          </dt>
          <dd className="num mt-0.5 text-[17px] text-chalk">{b ? `${fmtInt(b.lower)} to ${fmtInt(b.upper)}` : "n/a"} <span className="text-[13px] text-muted-foreground">{unit}</span></dd>
        </div>
        <div>
          <dt className="text-muted-foreground">80% range coverage in backtests</dt>
          <dd className="mt-0.5 text-chalk">
            <span className="num text-[17px]">{fmtPct(c)}</span>{" "}
            <span className="text-[13px] text-muted-foreground">
              {readCoverage(c) === "on_target" ? "close to the 80% target" : readCoverage(c) == null ? "not measured" : ""}
            </span>
          </dd>
        </div>
      </dl>
      {readCoverage(c) === "narrow" && (
        <p className="mt-3 rounded-md border-l-2 border-caution bg-background/40 px-3 py-2 text-[13px] text-chalk">
          <CoverageFlag s={s} />. Read this range as narrower than the uncertainty really is.
        </p>
      )}
      {ref != null && (
        <p className="mt-3 flex items-start gap-2 text-[13px] text-muted-foreground">
          <span className="mt-1 inline-block h-3 shrink-0 border-l-2 border-dashed border-reference" aria-hidden />
          The dashed line is the largest five-season total any training quarterback reached ({fmtInt(ref)}). The upper end of
          the range runs past it, so read that end as &ldquo;at least the historical maximum&rdquo;.
        </p>
      )}
      {!nested && (
        <p className="mt-3 text-[13px] text-caution" role="status">
          The 50% range reported for this statistic is not inside its 80% range. The numbers are shown exactly as the API
          returned them.
        </p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ plain-language guide */

export function ReadingGuide({ stats }: { stats: StatForecast[] }) {
  const narrow = stats.filter((s) => readCoverage(coverage80(s)) === "narrow");
  const list = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);
  return (
    <aside aria-labelledby="reading-h" className="rounded-lg border border-line p-4 sm:p-6">
      <h3 id="reading-h" className="text-[20px] text-chalk">How to read these numbers</h3>
      <ul className="mt-3 grid gap-x-10 gap-y-4 text-[14px] text-chalk/90 md:grid-cols-2">
        <li>
          <strong className="font-semibold text-chalk">Five-season totals.</strong> Every number adds up the first five NFL
          seasons. Nothing here is a single-season figure.
        </li>
        <li>
          <strong className="font-semibold text-chalk">Ranges show uncertainty, not guarantees.</strong> About half of similar
          careers land inside the 50% range and about eight in ten inside the 80% range. One in five lands outside it.
        </li>
        <li>
          <strong className="font-semibold text-chalk">Coverage is the track record.</strong> The model was replayed on 107
          earlier quarterbacks it had not been trained on, and we counted how often their real totals landed inside the 80%
          range. Close to 80% means the ranges are honest.
        </li>
        <li>
          <strong className="font-semibold text-chalk">Some outcomes are harder to predict.</strong>{" "}
          {narrow.length ? (
            <>
              In this view, the ranges for {list(narrow.map((s) => midSentence(s.label)))} caught fewer than{" "}
              {Math.round(COVERAGE_LOW * 100)}% of real totals in backtests, so treat them as too narrow.
            </>
          ) : (
            <>
              In this view every range caught between {Math.round(COVERAGE_LOW * 100)}% and {Math.round(COVERAGE_HIGH * 100)}%
              of real totals in backtests, close to the 80% target.
            </>
          )}
        </li>
      </ul>
    </aside>
  );
}

/* ------------------------------------------------------------------ full projection table */

const GROUPS: [string, string[]][] = [
  ["Passing", ["pass_yards", "pass_tds", "interceptions", "pass_attempts", "pass_completions"]],
  ["Rushing", ["rush_yards", "rush_tds", "rush_attempts"]],
  ["Playing time", ["games_started", "games_played"]],
];

const TIPS = {
  expected: "The model's central prediction for the total across his first five NFL seasons.",
  r50: "A narrower range, meant to contain about half of comparable outcomes.",
  r80: "A wider range, meant to contain about 80% of comparable outcomes.",
  cov: "The share of historical outcomes that actually fell inside the model's nominal 80% range, measured on later draft classes it had not seen.",
};

function HeaderTip({ children, tip, label, className }: { children: React.ReactNode; tip: string; label: string; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1", className)}>
      {children}
      <InfoTip label={label}>{tip}</InfoTip>
    </span>
  );
}

function grouped(stats: StatForecast[]) {
  const by = new Map(stats.map((s) => [s.stat, s]));
  const seen = new Set<string>();
  const out = GROUPS.map(([g, ks]) => {
    const rows = ks.map((k) => by.get(k)).filter((s): s is StatForecast => !!s);
    rows.forEach((s) => seen.add(s.stat));
    return [g, rows] as const;
  });
  const rest = stats.filter((s) => !seen.has(s.stat)); // anything new from the API is still shown
  if (rest.length) out.push(["Other", rest]);
  return out.filter(([, r]) => r.length);
}

export function ProjectionTable({ stats }: { stats: StatForecast[] }) {
  const groups = grouped(stats);
  const domains = new Map(stats.map((s) => [s.stat, rulerDomain([s.expected, ...s.intervals.flatMap((i) => [i.lower, i.upper])])]));
  return (
    <div>
      {/* tablet and desktop: a real table */}
      <div className="hidden overflow-hidden rounded-lg border border-line md:block">
        <table className="w-full text-left text-[14px]">
          <caption className="border-b border-line px-4 py-3 text-left">
            <span className="font-display text-[20px] font-semibold text-chalk">All ten projections with their ranges</span>
            <span className="ml-3 text-[13px] text-muted-foreground">Totals over NFL seasons 1 to 5</span>
          </caption>
          <thead className="bg-panel/60 text-[13px] whitespace-nowrap text-muted-foreground">
            <tr>
              <th scope="col" className="px-4 py-2.5 font-medium">Statistic</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                <HeaderTip tip={TIPS.expected} label="Expected" className="justify-end">Expected</HeaderTip>
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                <HeaderTip tip={TIPS.r50} label="50% range" className="justify-end">50% range</HeaderTip>
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                <HeaderTip tip={TIPS.r80} label="80% range" className="justify-end">80% range</HeaderTip>
              </th>
              <th scope="col" className="hidden w-[20%] px-2 py-2.5 font-medium lg:table-cell">
                <span className="sr-only">Range drawn to scale</span>
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                <HeaderTip tip={TIPS.cov} label="80% coverage in backtests" className="justify-end">80% coverage in backtests</HeaderTip>
              </th>
            </tr>
          </thead>
          {groups.map(([g, rows]) => (
            <tbody key={g} className="border-t border-line">
              {rows.map((s, i) => (
                <tr key={s.stat} className={cn("align-middle", i > 0 && "border-t border-line/50")}>
                  <th scope="row" className="px-4 py-2.5 font-normal whitespace-nowrap text-chalk">
                    {i === 0 && <span className="block text-[11px] font-medium tracking-wide text-muted-foreground">{g}</span>}
                    {s.label}
                  </th>
                  <td className="num px-4 py-2.5 text-right text-[17px] font-semibold text-chalk">{fmtInt(s.expected)}</td>
                  <td className="num px-4 py-2.5 text-right whitespace-nowrap text-chalk/90">{range(s, 0.5)}</td>
                  <td className="num px-4 py-2.5 text-right whitespace-nowrap text-chalk/90">{range(s, 0.8)}</td>
                  <td className="hidden px-2 py-2.5 lg:table-cell" aria-hidden>
                    <FieldRuler
                      label={`${s.label} range`}
                      expected={s.expected}
                      intervals={s.intervals.filter((x) => x.level < 0.85)}
                      domain={domains.get(s.stat)}
                      showScale={false}
                      showValue={false}
                    />
                  </td>
                  <td className="num px-4 py-2.5 text-right whitespace-nowrap">
                    <span className="text-chalk/90">{fmtPct(coverage80(s))}</span>
                    {readCoverage(coverage80(s)) && readCoverage(coverage80(s)) !== "on_target" && (
                      <span className="mt-0.5 block text-[12px]"><CoverageFlag s={s} compact /></span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>

      {/* phones: one card per statistic, the same numbers as the table */}
      <div className="md:hidden">
        <h3 className="font-display text-[20px] font-semibold text-chalk">All ten projections with their ranges</h3>
        <p className="text-[13px] text-muted-foreground">Totals over NFL seasons 1 to 5</p>
        <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-muted-foreground">
          <HeaderTip tip={TIPS.expected} label="Expected">Expected</HeaderTip>
          <HeaderTip tip={TIPS.r50} label="50% range">50% range</HeaderTip>
          <HeaderTip tip={TIPS.r80} label="80% range">80% range</HeaderTip>
          <HeaderTip tip={TIPS.cov} label="80% coverage in backtests">Coverage</HeaderTip>
        </p>
        {groups.map(([g, rows]) => (
          <section key={g} aria-label={g} className="mt-4">
            <h4 className="text-[13px] font-medium tracking-wide text-muted-foreground" style={{ fontFamily: "var(--font-body)" }}>{g}</h4>
            <ul className="mt-1.5 divide-y divide-line/60 rounded-lg border border-line">
              {rows.map((s) => (
                <li key={s.stat} className="px-3.5 py-3">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[15px] text-chalk">{s.label}</span>
                    <span className="num text-[22px] font-semibold text-chalk">{fmtInt(s.expected)}</span>
                  </div>
                  <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-[14px]">
                    <dt className="text-muted-foreground">50% range</dt>
                    <dd className="num text-right text-chalk/90">{range(s, 0.5)}</dd>
                    <dt className="text-muted-foreground">80% range</dt>
                    <dd className="num text-right text-chalk/90">{range(s, 0.8)}</dd>
                    <dt className="text-muted-foreground">Coverage in backtests</dt>
                    <dd className="num text-right text-chalk/90">{fmtPct(coverage80(s))}</dd>
                  </dl>
                  {readCoverage(coverage80(s)) === "narrow" && (
                    <p className="mt-1.5 text-[12px]"><CoverageFlag s={s} compact /></p>
                  )}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
