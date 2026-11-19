"use client";
import Image from "next/image";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { useBacktest, useComparisons, useForecast, usePerformance, useProspectSearch } from "@/lib/api/queries";
import type { ProspectSummary } from "@/lib/api/types";
import { fmt3, fmtInt, fmtPct, fmtProb } from "@/lib/format";
import { intervalAt } from "@/lib/forecast";
import { GITHUB_URL } from "@/lib/config";
import { CLASS_INFO, avgSkill } from "@/lib/backtest";
import { PHOTOS } from "@/lib/images";
import { FieldRuler, RulerLegend } from "@/components/field-ruler";
import { PhotoCredit } from "@/components/photo-banner";
import { ProspectSearch } from "@/components/prospect-search";
import { ErrorState, SlowLoading } from "@/components/status";

const LATEST_CLASS = 2026;
const SHOWCASE_BACKTEST = { id: "qb_226", name: "Josh Allen" };
const link = "text-projection-text underline decoration-1 hover:decoration-2";

function HeroPreview({ p }: { p: ProspectSummary }) {
  const f = useForecast(p.player_id, "pre_draft");
  const get = (k: string) => (f.data?.career.status === "available" ? f.data.career.five_year_totals.find((x) => x.stat === k) : undefined);
  const py = get("pass_yards");
  return (
    <div className="rounded-lg border border-line/80 bg-background/80 p-5 shadow-2xl shadow-black/50 backdrop-blur-md sm:p-6">
      <p className="text-[13px] text-muted-foreground">Live pre-draft forecast, first pick of the {p.draft_year} draft</p>
      <Link href={`/prospects/${p.player_id}`} className="mt-1 block font-display text-[32px] leading-tight font-semibold text-chalk hover:underline">
        {p.name}
      </Link>
      <p className="text-[14px] text-muted-foreground">{p.college}</p>
      <div className="mt-5" aria-live="polite">
        {f.isPending ? (
          <SlowLoading label="Loading live forecast" lines={3} />
        ) : f.isError ? (
          <ErrorState error={f.error} onRetry={() => f.refetch()} />
        ) : f.data.career.status === "available" ? (
          <div className="space-y-5">
            {py && (
              <div>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-[14px] text-chalk">{py.label}, NFL seasons 1 to 5</span>
                  <span className="num text-[30px] leading-none font-semibold text-chalk">{fmtInt(py.expected)}</span>
                </div>
                <FieldRuler label={`${py.label}, NFL seasons 1 to 5`} expected={py.expected} intervals={py.intervals} size="lg" showValue={false} />
              </div>
            )}
            <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-md border border-line bg-line">
              {(["pass_tds", "interceptions", "games_started"] as const).map((k) => {
                const s = get(k);
                const r = s ? intervalAt(s, 0.8) : undefined;
                return (
                  <div key={k} className="bg-panel/90 px-3 py-2.5">
                    <dt className="text-[12px] text-muted-foreground">{s?.label ?? k}</dt>
                    <dd className="num text-[24px] leading-tight font-semibold text-chalk">{s ? fmtInt(s.expected) : "n/a"}</dd>
                    <dd className="num text-[11px] text-muted-foreground">{r ? `80%: ${fmtInt(r.lower)} to ${fmtInt(r.upper)}` : ""}</dd>
                  </div>
                );
              })}
            </dl>
            <div className="flex flex-wrap items-center justify-between gap-3">
              {f.data.career.start_probabilities && (
                <p className="text-[14px] text-chalk/90">
                  Chance of 32+ starts:{" "}
                  <span className="num text-[18px] font-semibold">{fmtProb(f.data.career.start_probabilities.p_32_plus)}</span>
                </p>
              )}
              <Link href={`/prospects/${p.player_id}`} className={`text-[14px] ${link}`}>Open the full report</Link>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">{f.data.career.reason}</p>
        )}
      </div>
    </div>
  );
}

function Hero({ first }: { first?: ProspectSummary }) {
  const photo = PHOTOS.heroStadium;
  return (
    <section className="relative isolate overflow-hidden border-b border-line">
      <Image src={photo.src} alt={photo.alt} fill priority sizes="100vw" quality={72} className="-z-20 object-cover" style={{ objectPosition: photo.position }} />
      <div className="absolute inset-0 -z-10 bg-[linear-gradient(100deg,rgb(15_26_43/0.95)_0%,rgb(15_26_43/0.82)_48%,rgb(15_26_43/0.45)_100%)]" />
      <div className="absolute inset-x-0 bottom-0 -z-10 h-48 bg-gradient-to-t from-background to-transparent" />
      <div className="mx-auto grid max-w-6xl gap-12 px-4 pt-16 pb-20 sm:px-6 lg:grid-cols-[1.2fr_1fr] lg:items-end lg:pt-28 lg:pb-28">
        <div>
          <h1 className="text-[clamp(50px,8vw,104px)] leading-[0.9] text-chalk">From college production to NFL projection.</h1>
          <p className="measure mt-6 text-[18px] text-chalk/85">
            Data-driven quarterback forecasting, historical NFL comparisons and transparent model evaluation, from the last
            college snap to a fifth NFL season.
          </p>
          <ProspectSearch className="mt-8 max-w-xl" placeholder="Search a quarterback by name" />
          <div className="mt-5 flex flex-wrap gap-3">
            <Link href="/prospects" className="rounded-md bg-chalk px-5 py-2.5 text-[15px] font-medium text-background hover:bg-chalk/90">
              Explore prospects
            </Link>
            <Link href="/backtesting" className="rounded-md border border-chalk/40 px-5 py-2.5 text-[15px] font-medium text-chalk hover:bg-chalk/10">
              See historical accuracy
            </Link>
          </div>
        </div>
        <div>{first ? <HeroPreview p={first} /> : null}</div>
      </div>
      <PhotoCredit photo={photo} className="absolute right-3 bottom-2 sm:right-6" />
    </section>
  );
}

/** Three facts about the product, every number read from the API. Nothing is shown until it has loaded. */
function Credibility() {
  const all = useProspectSearch("", { limit: 1 });
  const perf = usePerformance();
  const prod = perf.data?.sections.production;
  const items: [string | null, string][] = [
    [all.data ? fmtInt(all.data.total) : null, "quarterbacks with a report, drafted or not"],
    [prod ? String(prod.test_qbs) : null, "real NFL careers the models were tested on, never trained on"],
    ["3", "ranges on every forecast (50, 80 and 90%), each checked against what actually happened"],
  ];
  return (
    <section aria-label="About the forecasts" className="border-b border-line bg-panel/50">
      <ul className="mx-auto grid max-w-6xl gap-px px-4 sm:grid-cols-3 sm:px-6">
        {items.map(([v, l]) => (
          <li key={l} className="flex items-baseline gap-3 py-3 first:pt-5 last:pb-5 sm:py-5 sm:border-l sm:border-line sm:px-6 sm:first:border-l-0 sm:first:pl-0">
            <span className="num min-w-[3ch] text-[30px] leading-none font-semibold whitespace-nowrap text-chalk">{v ?? <span className="inline-block h-6 w-14 animate-pulse rounded bg-raised align-middle" aria-label="Loading" />}</span>
            <span className="text-[14px] leading-snug text-muted-foreground">{l}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

const STEPS = [
  ["College data", "Every FBS passer-season since 2004: efficiency, volume, rushing, schedule strength and team context, plus recruiting and the NFL Combine."],
  ["Features", "Final-season and career profiles, adjusted for opponents, with missing values kept missing rather than guessed."],
  ["Models", "Draft probability, round and pick, starting chances and ten five-season totals, each with a calibrated range."],
  ["Tested on the future", "Every model is scored on draft classes it never saw, trained only on classes whose NFL results were already known."],
] as const;

function HowItWorks() {
  return (
    <section aria-labelledby="how-h" className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
      <h2 id="how-h" className="max-w-xl text-[clamp(34px,4.5vw,52px)] text-chalk">How the prediction engine works</h2>
      <ol className="mt-10 grid gap-x-8 gap-y-10 md:grid-cols-4">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="relative border-t-2 border-projection pt-5">
            <span className="num text-[15px] text-projection-text">Step {i + 1}</span>
            <h3 className="mt-1 text-[24px] text-chalk">{t}</h3>
            <p className="mt-2 text-[15px] text-muted-foreground">{d}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Featured({ cls, loading }: { cls: ProspectSummary[]; loading: boolean }) {
  return (
    <section aria-labelledby="class-h" className="border-y border-line bg-panel/40">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-20 sm:px-6 lg:grid-cols-[1fr_2fr]">
        <div>
          <h2 id="class-h" className="text-[clamp(34px,4.5vw,52px)] text-chalk">The {LATEST_CLASS} quarterback class</h2>
          <p className="mt-3 text-[15px] text-muted-foreground">
            Every quarterback drafted in {LATEST_CLASS}, in draft order. Open any of them for the full scouting report, before
            and after the draft.
          </p>
          <Link href="/prospects?drafted=1" className={`mt-5 inline-block text-[15px] ${link}`}>Browse every drafted quarterback</Link>
        </div>
        {loading ? (
          <SlowLoading label="Loading the draft class" lines={5} />
        ) : (
          <ol className="grid gap-x-8 sm:grid-cols-2">
            {cls.map((p) => (
              <li key={p.player_id} className="border-t border-line">
                <Link href={`/prospects/${p.player_id}`} className="group flex items-baseline justify-between gap-4 py-3.5">
                  <span>
                    <span className="font-display text-[22px] font-semibold text-chalk group-hover:underline">{p.name}</span>
                    <span className="ml-2 text-[13px] text-muted-foreground">{p.college}</span>
                  </span>
                  <span className="num shrink-0 text-[15px] text-muted-foreground">Pick {p.overall_pick}</span>
                </Link>
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

function Preview({ src, alt, w, h }: { src: string; alt: string; w: number; h: number }) {
  return (
    <figure className="overflow-hidden rounded-lg border border-line shadow-2xl shadow-black/40">
      <Image src={src} alt={alt} width={w} height={h} sizes="(min-width: 1024px) 55vw, 100vw" className="h-auto w-full" />
      <figcaption className="sr-only">{alt}</figcaption>
    </figure>
  );
}

function ProjectionsSection() {
  return (
    <section aria-labelledby="proj-h" className="mx-auto grid max-w-6xl items-center gap-12 px-4 py-20 sm:px-6 lg:grid-cols-[1fr_1.3fr]">
      <div>
        <h2 id="proj-h" className="text-[clamp(34px,4.5vw,52px)] text-chalk">Five seasons, ten statistics, honest ranges</h2>
        <p className="mt-4 text-[16px] text-chalk/85">
          Passing yards, touchdowns, interceptions, rushing and starts over a quarterback&apos;s first five NFL seasons. Each comes
          with 50, 80 and 90% ranges whose coverage was measured on later draft classes.
        </p>
        <p className="mt-3 text-[15px] text-muted-foreground">
          The expected value is the centre of a wide distribution, not a promise. Most careers land somewhere inside the
          bands; about one in five lands outside the 80% band.
        </p>
      </div>
      <Preview src="/previews/report-production.jpg" alt="Screenshot of a scouting report: five-season passing yards, touchdowns and interceptions drawn as nested 50, 80 and 90 percent ranges." w={1600} h={1222} />
    </section>
  );
}

function ComparisonsPreview({ p }: { p?: ProspectSummary }) {
  const c = useComparisons(p?.player_id ?? "qb_319");
  const comps = c.data?.status === "available" ? c.data.comparables : [];
  return (
    <section aria-labelledby="comps-h" className="border-y border-line bg-panel/40">
      <div className="mx-auto grid max-w-6xl gap-12 px-4 py-20 sm:px-6 lg:grid-cols-[1.3fr_1fr]">
        <div className="order-2 lg:order-1">
          {c.isPending ? (
            <SlowLoading label="Loading comparisons" lines={5} />
          ) : c.isError ? (
            <ErrorState error={c.error} onRetry={() => c.refetch()} />
          ) : (
            <ol className="space-y-0">
              {comps.map((x) => (
                <li key={x.name} className="grid grid-cols-[3rem_1fr_auto] items-center gap-4 border-t border-line py-4">
                  <span aria-hidden className="flex size-12 items-center justify-center rounded-full border border-line font-display text-[18px] font-semibold text-chalk">
                    {x.name.split(" ").map((w) => w[0]).slice(0, 2).join("")}
                  </span>
                  <span className="min-w-0">
                    <Link href={`/prospects/${x.player_id}`} className="font-display text-[22px] font-semibold text-chalk hover:underline">{x.name}</Link>
                    <span className="block truncate text-[13px] text-muted-foreground">Most alike on {x.most_similar_on}</span>
                  </span>
                  <span className="text-right text-[13px] text-muted-foreground">
                    {x.college}
                    <span className="num block">{x.draft_year} draft</span>
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>
        <div className="order-1 lg:order-2">
          <h2 id="comps-h" className="text-[clamp(34px,4.5vw,52px)] text-chalk">Who he looks like on paper</h2>
          <p className="mt-4 text-[16px] text-chalk/85">
            The five earlier drafted quarterbacks whose final college seasons look most like {p?.name ?? "this prospect"}&apos;s, with
            what they share, how they differ and how their NFL careers went.
          </p>
          <p className="mt-3 text-[15px] text-muted-foreground">
            Comparisons explain a profile. They are not a forecast: a comp&apos;s career says little about another player&apos;s.
          </p>
        </div>
      </div>
    </section>
  );
}

function BacktestPreview() {
  const b = useBacktest(SHOWCASE_BACKTEST.id);
  const v = b.data?.views.pre_draft;
  const pick = (k: string) => v?.statistics.find((s) => s.stat === k);
  return (
    <section aria-labelledby="bt-h" className="mx-auto grid max-w-6xl gap-12 px-4 py-20 sm:px-6 lg:grid-cols-[1fr_1.3fr]">
      <div>
        <h2 id="bt-h" className="text-[clamp(34px,4.5vw,52px)] text-chalk">Predicted versus what happened</h2>
        <p className="mt-4 text-[16px] text-chalk/85">
          What the models would have said before the 2018 draft about {SHOWCASE_BACKTEST.name}, trained only on classes whose
          results were known by then, against his first five NFL seasons.
        </p>
        <p className="mt-3 text-[15px] text-muted-foreground">
          His real career ran past all three ranges below: the models can be badly wrong about exceptional players. Every
          backtest is labelled as genuine, still open, or in-sample.
        </p>
        <Link href={`/backtesting?id=${SHOWCASE_BACKTEST.id}`} className={`mt-5 inline-block text-[15px] ${link}`}>Open the full backtest</Link>
      </div>
      <div className="rounded-lg border border-line bg-panel p-6">
        {b.isPending ? (
          <SlowLoading label="Loading backtest" lines={4} />
        ) : b.isError ? (
          <ErrorState error={b.error} onRetry={() => b.refetch()} />
        ) : (
          <div className="space-y-6">
            <p className="text-[14px] text-chalk">{CLASS_INFO[b.data.classification.status].title}, pre-draft view</p>
            {(["pass_yards", "pass_tds", "games_started"] as const).map((k) => {
              const s = pick(k);
              return s ? (
                <div key={k}>
                  <div className="flex items-baseline justify-between text-[14px]">
                    <span className="text-chalk">{k === "pass_yards" ? "Passing yards" : k === "pass_tds" ? "Passing TDs" : "Games started"}</span>
                    <span className="num text-muted-foreground">
                      predicted {fmtInt(s.predicted)}, actual <span className="text-observed-text">{fmtInt(s.actual)}</span>
                    </span>
                  </div>
                  <FieldRuler label={`${k}: prediction vs actual`} expected={s.predicted} intervals={[s.interval_80]} actual={s.actual} showScale={false} />
                </div>
              ) : null;
            })}
            <RulerLegend actual />
          </div>
        )}
      </div>
    </section>
  );
}

function EvaluationSection() {
  const q = usePerformance();
  const s = q.data?.sections;
  const A = s?.draft_probability.models.find((m) => m.variant === "A_college");
  const C = s?.draft_probability.models.find((m) => m.variant === "C_college_invited_measures");
  const stats = s
    ? [
        [fmt3(A?.auc), "AUC for who gets drafted, college stats alone (0.5 is chance)"],
        [fmt3(C?.auc), "AUC once the Combine is known; most of that gain is the invitation itself"],
        [fmtPct(avgSkill(s.production.errors, "pre_draft")), "smaller five-season errors than the historical median, before the draft"],
        [fmtPct(avgSkill(s.production.errors, "post_draft")), "smaller errors once the draft slot is known"],
      ]
    : [];
  return (
    <section aria-labelledby="eval-h" className="border-y border-line bg-panel/40">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <h2 id="eval-h" className="max-w-2xl text-[clamp(34px,4.5vw,52px)] text-chalk">How good the models are, and where they are not</h2>
        {q.isPending ? (
          <div className="mt-8"><SlowLoading label="Loading evaluation" lines={3} /></div>
        ) : q.isError ? (
          <ErrorState className="mt-8" error={q.error} onRetry={() => q.refetch()} />
        ) : (
          <>
            <dl className="mt-10 grid gap-x-8 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
              {stats.map(([v, l]) => (
                <div key={l} className="flex flex-col border-t border-line pt-4">
                  <dt className="order-2 text-[14px] text-muted-foreground">{l}</dt>
                  <dd className="num order-1 text-[44px] leading-none font-semibold text-chalk">{v}</dd>
                </div>
              ))}
            </dl>
            <p className="measure mt-8 text-[15px] text-chalk/85">
              Measured on draft classes the models had not seen ({s?.production.test_qbs} quarterbacks for the career models).
              Before the draft, career forecasts are only modestly better than historical averages. NFL efficiency rates could
              not be predicted at all.{" "}
              <Link href="/methodology" className={link}>Read the full evaluation and its limitations</Link>.
            </p>
          </>
        )}
      </div>
    </section>
  );
}

function CodeSection() {
  return (
    <section aria-labelledby="code-h" className="mx-auto grid max-w-6xl gap-10 px-4 py-20 sm:px-6 lg:grid-cols-[1fr_1fr]">
      <div>
        <h2 id="code-h" className="text-[clamp(34px,4.5vw,52px)] text-chalk">Open methods, open code</h2>
        <p className="mt-4 text-[16px] text-chalk/85">
          A Python data pipeline and scikit-learn models, served by a FastAPI backend with typed responses, and this Next.js
          site. Every number on these pages comes from that API.
        </p>
      </div>
      <ul className="divide-y divide-line border-y border-line">
        {[
          [GITHUB_URL, "Source code, tests and evaluation reports on GitHub", true],
          ["/methodology", "Methodology: data, chronological testing, calibration and limitations", false],
          ["/case-study", "Case study: the problem, architecture and findings", false],
        ].map(([href, label, ext]) => (
          <li key={String(href)}>
            <a href={String(href)} {...(ext ? { target: "_blank", rel: "noreferrer" } : {})} className="group flex items-center justify-between gap-4 py-4 text-[17px] text-chalk">
              <span className="group-hover:underline">{label}</span>
              <ArrowUpRight className="size-5 shrink-0 text-muted-foreground" aria-hidden />
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function Landing() {
  const latest = useProspectSearch("", { draftYear: LATEST_CLASS, draftedOnly: true, limit: 12 });
  const cls = [...(latest.data?.items ?? [])].sort((a, b) => (a.overall_pick ?? 999) - (b.overall_pick ?? 999));
  return (
    <div>
      <Hero first={cls[0]} />
      <Credibility />
      {latest.isError && (
        <div className="mx-auto max-w-6xl px-4 pt-10 sm:px-6"><ErrorState error={latest.error} onRetry={() => latest.refetch()} /></div>
      )}
      <HowItWorks />
      <Featured cls={cls} loading={latest.isPending} />
      <ProjectionsSection />
      <ComparisonsPreview p={cls[0]} />
      <BacktestPreview />
      <EvaluationSection />
      <CodeSection />
    </div>
  );
}
