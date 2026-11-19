"use client";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { isPlayerId } from "@/lib/api/client";
import { useBacktest } from "@/lib/api/queries";
import { BACKTEST_EXAMPLES } from "@/lib/featured";
import { ProspectSearch } from "@/components/prospect-search";
import { ErrorState, SlowLoading, Unavailable } from "@/components/status";
import { ApiError } from "@/lib/api/client";
import { BacktestView } from "./backtest-view";
import { AggregateAccuracy } from "./aggregate-accuracy";
import { PhotoBanner } from "@/components/photo-banner";
import { EmbargoTimeline } from "@/components/methodology/methodology";
import { PHOTOS } from "@/lib/images";

export function BacktestExplorer() {
  const sp = useSearchParams();
  const router = useRouter();
  const path = usePathname();
  const raw = sp.get("id");
  const id = raw && isPlayerId(raw) ? raw : null;
  const q = useBacktest(id);
  const select = (pid: string) => router.replace(`${path}?id=${pid}`, { scroll: false });

  return (
    <>
    <PhotoBanner
      photo={PHOTOS.goalLine1922}
      priority
      grayscale
      title="Predictions against real careers"
      intro={
        <p>
          What the models would have said before a quarterback&apos;s draft, using only draft classes whose NFL results were
          already known, set against what then happened in his first five NFL seasons.
        </p>
      }
    />
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_1.2fr]">
        <ProspectSearch draftedOnly onSelect={(p) => select(p.player_id)} label="Choose a drafted quarterback" placeholder="Choose a drafted quarterback" />
        <div>
          <p className="text-[13px] text-muted-foreground">Or start with one of these (chosen for recognisability, not accuracy)</p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {BACKTEST_EXAMPLES.map((e) => (
              <li key={e.id}>
                <button
                  type="button"
                  onClick={() => select(e.id)}
                  aria-pressed={id === e.id}
                  className={`rounded border px-2.5 py-1 text-[13px] ${id === e.id ? "border-chalk text-chalk" : "border-line text-muted-foreground hover:text-chalk"}`}
                >
                  {e.name} <span className="num">{e.year}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <section aria-label="Selected backtest" className="mt-12 border-t border-line pt-10" aria-live="polite">
        {!id ? (
          <p className="text-muted-foreground">Choose a quarterback to see his historical forecast next to his real NFL career.</p>
        ) : q.isPending ? (
          <SlowLoading label="Loading backtest" lines={5} />
        ) : q.isError ? (
          q.error instanceof ApiError && q.error.code === "PREDICTION_UNSUPPORTED" ? (
            <Unavailable title="No historical backtest for this player" reason={q.error.message} />
          ) : (
            <ErrorState error={q.error} onRetry={() => q.refetch()} />
          )
        ) : (
          <BacktestView key={id} b={q.data} />
        )}
      </section>

      <section aria-labelledby="fair-h" className="mt-16 grid gap-8 border-t border-line pt-10 lg:grid-cols-[15rem_1fr]">
        <h2 id="fair-h" className="text-[30px] text-chalk">How a fair backtest works</h2>
        <div className="min-w-0 space-y-5 text-[15px] text-chalk/90">
          <p className="measure">
            A forecast is only honest if it was made without the answer. For each draft class the models are refit using only
            earlier classes whose five NFL seasons had finished by that draft, then asked about the new class. Nothing from the
            future leaks in: not the players&apos; careers, not later classes, not the choice of model.
          </p>
          <EmbargoTimeline />
          <p className="measure text-muted-foreground">
            Quarterbacks drafted before 2012 had too few earlier classes for this, so they are shown with the final model and
            labelled in-sample. Classes since 2022 are genuine forecasts, but their five seasons are not over yet.
          </p>
        </div>
      </section>

      <section aria-labelledby="agg-h" className="mt-16 border-t border-line pt-10">
        <h2 id="agg-h" className="text-[34px] text-chalk">Accuracy across all backtests</h2>
        <div className="mt-4">
          <AggregateAccuracy />
        </div>
      </section>
    </div>
    </>
  );
}
