"use client";
import Image from "next/image";
import Link from "next/link";
import { usePerformance } from "@/lib/api/queries";
import { fmt3, fmtPct } from "@/lib/format";
import { GITHUB_URL } from "@/lib/config";
import { avgSkill } from "@/lib/backtest";
import { PHOTOS } from "@/lib/images";
import { PhotoBanner } from "@/components/photo-banner";
import { ErrorState, SlowLoading } from "@/components/status";
import { PipelineDiagram } from "@/components/methodology/methodology";

const link = "text-projection-text underline decoration-1 hover:decoration-2";

function Part({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="grid gap-6 border-t border-line py-12 lg:grid-cols-[15rem_1fr]">
      <h2 id={`${id}-h`} className="text-[30px] text-chalk">{title}</h2>
      <div className="measure min-w-0 space-y-4 text-[16px] text-chalk/90">{children}</div>
    </section>
  );
}

const ARCH = [
  ["Data", "CollegeFootballData.com API and nflverse releases, cached locally; never called while serving"],
  ["Pipeline", "Python ingestion, identity linking and feature building into SQLite and Parquet"],
  ["Models", "scikit-learn models, versioned and SHA-256 pinned, with nested chronological backtests"],
  ["API", "FastAPI with Pydantic schemas, stable error codes, CORS allowlist, rate limiting"],
  ["Site", "Next.js 16, TypeScript, TanStack Query and Recharts, typed from the API's OpenAPI schema"],
] as const;

export function CaseStudy() {
  const q = usePerformance();
  const s = q.data?.sections;
  return (
    <>
      <PhotoBanner
        photo={PHOTOS.linePlay1927}
        priority
        grayscale
        title="Case study: forecasting quarterback careers"
        intro={<p>A complete applied machine-learning project, from raw college data to a tested, deployed-ready product.</p>}
      />
      <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6">
        <Part id="problem" title="The problem">
          <p>
            Quarterbacks are the most expensive and least predictable picks in the NFL Draft. Public projections are mostly
            opinions with no record of how often they are right. The goal was a forecasting system whose every number can be
            checked against what actually happened, and which says plainly when it cannot answer.
          </p>
        </Part>

        <Part id="data" title="The data">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>1,279 prospects: every drafted quarterback from 2000 to 2026, plus FBS quarterbacks with a final college season in the data.</li>
            <li>College seasons, games, recruiting and team context from CollegeFootballData.com.</li>
            <li>Draft results, NFL seasons, starts and Combine results from nflverse.</li>
            <li>Career models trained on 183 drafted quarterbacks (2005 to 2021 classes) whose five NFL seasons are complete.</li>
          </ul>
        </Part>

        <Part id="architecture" title="Architecture">
          <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
            {ARCH.map(([t, d], i) => (
              <li key={t} className="grid grid-cols-[2rem_1fr] gap-3 border-l-2 border-projection pl-4">
                <span className="num text-[15px] text-projection-text">{i + 1}</span>
                <span>
                  <span className="font-display text-[20px] font-semibold text-chalk">{t}</span>
                  <span className="block text-[14px] text-muted-foreground">{d}</span>
                </span>
              </li>
            ))}
          </ol>
        </Part>

        <Part id="method" title="Modelling">
          <p>
            Regularised logistic regression for draft probability; ordinal and multinomial models for draft rounds; ridge and
            quasi-Poisson regression for ten five-season totals, with empirical prediction intervals; nested logits for
            starting milestones; a nearest-neighbour engine for comparisons.
          </p>
          <p>
            The emphasis was on testing rather than model complexity. Every model was evaluated on draft classes it never saw,
            trained only on classes whose outcomes were known at the time, and model choices were made inside those earlier
            classes.
          </p>
          <PipelineDiagram />
        </Part>

        <Part id="results" title="Results">
          {q.isPending ? (
            <SlowLoading label="Loading results" lines={4} />
          ) : q.isError || !s ? (
            <ErrorState error={q.error} onRetry={() => q.refetch()} />
          ) : (
            <ul className="list-disc space-y-1.5 pl-5">
              <li>
                Draft probability: AUC {fmt3(s.draft_probability.models.find((m) => m.variant === "A_college")?.auc)} from college data,{" "}
                {fmt3(s.draft_probability.models.find((m) => m.variant === "C_college_invited_measures")?.auc)} with the Combine, on{" "}
                {s.draft_probability.test_rows} test seasons.
              </li>
              <li>
                Five-season totals: errors {fmtPct(avgSkill(s.production.errors, "pre_draft"))} smaller than the historical median before
                the draft and {fmtPct(avgSkill(s.production.errors, "post_draft"))} smaller after it, on {s.production.test_qbs} quarterbacks.
              </li>
              <li>
                Calibration: the 80% ranges held the real total {fmtPct(s.interval_calibration.mean_coverage.pre_draft.coverage80)} (pre-draft)
                and {fmtPct(s.interval_calibration.mean_coverage.post_draft.coverage80)} (post-draft) of the time.
              </li>
            </ul>
          )}
        </Part>

        <Part id="findings" title="What I learned">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>Most of the apparent Combine effect is the invitation, a scouting decision, not the measurements.</li>
            <li>The draft slot is the strongest single predictor of NFL opportunity, so post-draft forecasts are much better.</li>
            <li>Honest evaluation shrank several early results: a model-selection rule that looked good failed under nested testing.</li>
            <li>Saying &ldquo;not available&rdquo; is a feature. Draft probability is withheld wherever eligibility cannot be verified.</li>
          </ul>
        </Part>

        <Part id="limits" title="Limitations">
          <p>
            About ten quarterbacks per draft class makes every estimate noisy. Undrafted players, coaching, scheme fit and
            injuries are not modelled, and NFL efficiency rates could not be predicted. The licensing review for public display of
            the data is documented in the project reports, with its open questions.
          </p>
        </Part>

        <Part id="screens" title="The product">
          <div className="grid gap-6">
            {[
              ["/previews/report-production.jpg", "Scouting report with five-season ranges", 1600, 1222],
              ["/previews/backtest.jpg", "Backtest of Josh Rosen: predicted against actual", 1600, 1222],
              ["/previews/methodology.jpg", "Methodology page: draft-probability evaluation", 1600, 1111],
            ].map(([src, alt, w, h]) => (
              <figure key={String(src)} className="overflow-hidden rounded-lg border border-line">
                <Image src={String(src)} alt={String(alt)} width={Number(w)} height={Number(h)} sizes="(min-width: 1024px) 700px, 100vw" className="h-auto w-full" />
                <figcaption className="border-t border-line px-3 py-2 text-[13px] text-muted-foreground">{alt}</figcaption>
              </figure>
            ))}
          </div>
          <p className="flex flex-wrap gap-x-6 gap-y-2">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className={link}>Source code on GitHub</a>
            <Link href="/prospects" className={link}>Try the scouting reports</Link>
            <Link href="/methodology" className={link}>Full methodology</Link>
          </p>
        </Part>
      </div>
    </>
  );
}
