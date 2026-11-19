"use client";
import { useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useForecast, useProspect } from "@/lib/api/queries";
import type { Mode, ProspectDetail } from "@/lib/api/types";
import { ErrorState, SlowLoading } from "@/components/status";
import { Button } from "@/components/ui/button";
import { CollegeSeasons, PlayerHeader } from "./player-header";
import { DraftSection } from "./draft-section";
import { ProductionSection, StartsSection } from "./career-sections";
import { ComparisonsSection } from "./comparisons-section";
import { UncertaintySection } from "./uncertainty-section";

export const postDraftSupported = (d: ProspectDetail) => d.prospect.kind === "drafted" && d.prospect.overall_pick != null;

/**
 * Mode lives in the URL (?mode=pre_draft|post_draft) and is only changed by the reader. If post-draft is requested
 * for a player without a recorded pick, the page says so instead of silently showing the pre-draft forecast.
 */
export function ModeSwitch({ mode, detail, onChange }: { mode: Mode; detail: ProspectDetail; onChange: (m: Mode) => void }) {
  const supported = postDraftSupported(detail);
  const opts: { value: Mode; label: string; help: string }[] = [
    { value: "pre_draft", label: "Pre-draft", help: "Only what was known before the draft: college data and the Combine." },
    {
      value: "post_draft",
      label: "Post-draft",
      help: supported ? "Adds his actual draft slot, which strongly shapes NFL opportunity." : "Needs a draft slot. Enter a hypothetical pick below.",
    },
  ];
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
      <div role="radiogroup" aria-label="Forecast mode" className="inline-flex rounded-md border border-line bg-panel p-1">
        {opts.map((o) => (
          <button
            key={o.value}
            role="radio"
            aria-checked={mode === o.value}
            onClick={() => onChange(o.value)}
            className={`rounded px-4 py-1.5 text-sm font-medium ${mode === o.value ? "bg-chalk text-background" : "text-muted-foreground hover:text-chalk"}`}
          >
            {o.label}
          </button>
        ))}
      </div>
      <p className="text-[13px] text-muted-foreground">{opts.find((o) => o.value === mode)?.help}</p>
    </div>
  );
}

function HypotheticalPick({ value, onApply }: { value?: number; onApply: (p: number | undefined) => void }) {
  const [text, setText] = useState(value ? String(value) : "");
  const n = Number(text);
  const valid = Number.isInteger(n) && n >= 1 && n <= 262;
  return (
    <form
      className="flex flex-wrap items-end gap-3 rounded-md border border-dashed border-line p-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (valid) onApply(n);
      }}
    >
      <div>
        <label htmlFor="assumed-pick" className="block text-[13px] text-muted-foreground">
          Hypothetical overall pick (1 to 262)
        </label>
        <input
          id="assumed-pick"
          inputMode="numeric"
          value={text}
          onChange={(e) => setText(e.target.value.replace(/[^0-9]/g, ""))}
          aria-invalid={text !== "" && !valid}
          className="num mt-1 h-9 w-28 rounded-md border border-line bg-panel px-3 text-chalk"
        />
      </div>
      <Button type="submit" variant="outline" disabled={!valid}>
        Forecast at this pick
      </Button>
      {value && (
        <Button type="button" variant="ghost" onClick={() => onApply(undefined)}>
          Clear
        </Button>
      )}
      <p className="w-full text-[12px] text-muted-foreground">
        A what-if: the career models use the pick as an input. The result is labelled hypothetical.
      </p>
    </form>
  );
}

function Forecast({ id, mode, pick, detail }: { id: string; mode: Mode; pick?: number; detail: ProspectDetail }) {
  const needsPick = mode === "post_draft" && !postDraftSupported(detail) && pick === undefined;
  const q = useForecast(id, mode, needsPick ? undefined : pick, !needsPick);
  if (needsPick)
    return (
      <p className="mt-10 rounded-md border border-dashed border-line p-5 text-sm text-muted-foreground" role="note">
        {detail.prospect.name} has no recorded NFL draft slot, so there is no real post-draft forecast. Enter a hypothetical pick
        above, or switch to the pre-draft view.
      </p>
    );
  if (q.isPending) return <div className="mt-10"><SlowLoading label="Loading forecast" lines={6} /></div>;
  if (q.isError) return <ErrorState className="mt-10" error={q.error} onRetry={() => q.refetch()} />;
  const f = q.data;
  const versions = (f.uncertainty.model_versions ?? {}) as Record<string, string>;
  return (
    <div aria-live="polite">
      <p className="pt-6 text-[13px] text-muted-foreground">
        {f.mode === "pre_draft" ? "Pre-draft forecast: the actual draft slot is never used." : "Post-draft forecast: uses the draft slot."}{" "}
        Models: {versions.career ?? "career model"}, {versions.draft_round_pick ?? "draft model"}, {versions.comparisons ?? "comparison engine"}.
      </p>
      <DraftSection f={f} />
      <StartsSection f={f} />
      <ProductionSection f={f} />
      <ComparisonsSection id={id} detail={detail} />
      <UncertaintySection f={f} />
    </div>
  );
}

export function ProspectReport({ id }: { id: string }) {
  const router = useRouter();
  const path = usePathname();
  const params = useSearchParams();
  const mode: Mode = params.get("mode") === "post_draft" ? "post_draft" : "pre_draft";
  const pickParam = Number(params.get("pick"));
  const pick = mode === "post_draft" && Number.isInteger(pickParam) && pickParam >= 1 && pickParam <= 262 ? pickParam : undefined;
  const detail = useProspect(id);

  const setParams = (m: Mode, p?: number) => {
    const sp = new URLSearchParams();
    if (m === "post_draft") sp.set("mode", m);
    if (m === "post_draft" && p) sp.set("pick", String(p));
    router.replace(sp.size ? `${path}?${sp}` : path, { scroll: false });
  };

  if (detail.isPending)
    return (
      <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
        <SlowLoading label="Loading prospect" lines={4} />
      </div>
    );
  if (detail.isError)
    return (
      <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
        <ErrorState error={detail.error} onRetry={() => detail.refetch()} />
      </div>
    );
  const d = detail.data;
  const supported = postDraftSupported(d);
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <PlayerHeader detail={d} mode={mode} />
      <div className="sticky top-0 z-20 -mx-4 border-b border-line bg-background/95 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6">
        <ModeSwitch mode={mode} detail={d} onChange={(m) => setParams(m, m === "post_draft" ? pick : undefined)} />
      </div>
      {mode === "post_draft" && !supported && <div className="mt-6"><HypotheticalPick value={pick} onApply={(p) => setParams("post_draft", p)} /></div>}
      <Forecast id={id} mode={mode} pick={supported ? undefined : pick} detail={d} />
      <details className="group border-b border-line py-8">
        <summary className="cursor-pointer text-[20px] font-display font-semibold text-chalk">College seasons</summary>
        <div className="mt-4">
          <CollegeSeasons detail={d} />
        </div>
      </details>
    </div>
  );
}
