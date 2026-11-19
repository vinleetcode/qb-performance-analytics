"use client";
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import type { Mode, ProspectDetail } from "@/lib/api/types";
import { fmt1, fmtInt, heightFt, NA } from "@/lib/format";
import Image from "next/image";
import { InfoTip } from "@/components/info-tip";
import { PhotoCredit } from "@/components/photo-banner";
import { PHOTOS } from "@/lib/images";

export const initials = (name: string) =>
  name.split(/\s+/).filter((w) => /^[A-Z]/.test(w)).map((w) => w[0]).slice(0, 2).join("");

type Rec = Record<string, unknown>;
export const num = (o: Rec | undefined, k: string): number | null => {
  const v = o?.[k];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
};

const POP_TEXT: Record<string, string> = {
  in_population: "In the draft-probability population (career verifiably ended after four or more college years)",
  final_season_not_verifiable_yet: "Final college season not verifiable yet: no later season exists in the data",
  no_evidence_of_four_college_years: "Fewer than four college years evidenced: eligibility unknown",
  three_or_fewer_years_since_high_school: "Three or fewer years since high school: eligibility unknown",
};

export function ActualDraft({ detail, mode }: { detail: ProspectDetail; mode: Mode }) {
  const [reveal, setReveal] = useState(false);
  const p = detail.prospect;
  const known = p.kind === "drafted" && p.overall_pick != null;
  const text = known
    ? `Drafted round ${p.draft_round}, pick ${p.overall_pick}`
    : p.drafted === false
      ? "Not drafted"
      : p.drafted === true
        ? "Drafted"
        : "Draft outcome not yet verifiable";
  if (mode === "post_draft" || !known) return <span>{text}</span>;
  return (
    <span className="inline-flex items-center gap-2">
      {reveal ? <span>{text}</span> : <span className="text-muted-foreground">Actual draft result hidden in the pre-draft view</span>}
      <button
        type="button"
        onClick={() => setReveal((r) => !r)}
        className="inline-flex items-center gap-1 rounded text-[13px] text-projection-text underline decoration-1 hover:decoration-2"
        aria-pressed={reveal}
      >
        {reveal ? <EyeOff className="size-3.5" aria-hidden /> : <Eye className="size-3.5" aria-hidden />}
        {reveal ? "Hide" : "Show"}
      </button>
    </span>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="min-w-0">
      <dt className="flex items-center gap-1 text-[13px] text-muted-foreground">
        {label}
        {hint && <InfoTip label={label}>{hint}</InfoTip>}
      </dt>
      <dd className="num text-[26px] leading-tight font-semibold text-chalk">{value}</dd>
    </div>
  );
}

export function PlayerHeader({ detail, mode }: { detail: ProspectDetail; mode: Mode }) {
  const p = detail.prospect;
  const fs = detail.final_season_statistics as Rec;
  const m = detail.athletic_measurements;
  const cov = detail.data_coverage as Rec;
  const pop = typeof cov.population_status === "string" ? cov.population_status : null;
  const h = m.combine_height_inches?.value ?? m.height_inches?.value;
  const w = m.combine_weight_lbs?.value ?? m.weight_lbs?.value;
  return (
    <header className="border-b border-line pb-8">
      <div className="relative isolate overflow-hidden rounded-lg border border-line">
        <Image
          src={PHOTOS.stadiumLights.src}
          alt=""
          fill
          priority
          sizes="(min-width: 1152px) 1152px, 100vw"
          quality={70}
          className="-z-20 object-cover"
          style={{ objectPosition: PHOTOS.stadiumLights.position }}
        />
        <div className="absolute inset-0 -z-10 bg-[linear-gradient(90deg,rgb(15_26_43/0.94)_0%,rgb(15_26_43/0.72)_50%,rgb(15_26_43/0.18)_100%)]" />
        <div className="flex items-end gap-6 px-5 pt-16 pb-7 sm:px-8 sm:pt-24">
          <span
            aria-hidden
            className="hidden size-24 shrink-0 items-center justify-center rounded-full border border-chalk/30 bg-background/50 font-display text-[40px] font-semibold text-chalk sm:flex"
          >
            {initials(p.name)}
          </span>
          <div className="min-w-0">
            <p className="text-[15px] text-chalk/80">
              {p.college ?? "College unknown"}, final college season {p.final_college_season ?? NA}, {p.draft_year} draft class
            </p>
            <h1 className="mt-1 text-[clamp(44px,8vw,88px)] leading-[0.95] text-chalk">{p.name}</h1>
            <p className="mt-3 text-[15px] text-chalk/90">
              <ActualDraft detail={detail} mode={mode} />
            </p>
          </div>
        </div>
        <PhotoCredit photo={PHOTOS.stadiumLights} className="absolute right-3 bottom-2" />
      </div>

      <dl className="mt-8 grid px-1 grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4 lg:grid-cols-7">
        <Stat label="Pass attempts" value={fmtInt(num(fs, "attempts"))} hint="Final college season, all games in the CFBD data." />
        <Stat label="Completion %" value={fmt1(num(fs, "completion_pct"))} />
        <Stat label="Yards / attempt" value={fmt1(num(fs, "yards_per_attempt"))} />
        <Stat label="TD %" value={fmt1(num(fs, "td_pct"))} />
        <Stat label="INT %" value={fmt1(num(fs, "int_pct"))} />
        <Stat
          label="Adj. yards / att."
          value={fmt1(num(fs, "opponent_adjusted_ypa"))}
          hint="Yards per attempt adjusted for the pass defenses faced."
        />
        <Stat label="Rush yds / game" value={fmt1(num(fs, "rush_yards_per_game_last3"))} hint="Average over the last three college seasons." />
      </dl>

      <div className="mt-6 flex flex-wrap gap-x-8 gap-y-2 text-[13px] text-muted-foreground">
        <span>
          Height {heightFt(h)}, weight {w != null ? `${Math.round(w)} lb` : NA}
          {detail.combine_participant === true ? " (NFL Combine)" : detail.combine_participant === false ? " (college roster; no Combine record)" : ""}
        </span>
        <span>
          Model inputs available: {String(cov.career_model_features_present ?? NA)} of {String(cov.career_model_features_total ?? NA)}
        </span>
        {pop && <span>{POP_TEXT[pop] ?? pop.replace(/_/g, " ")}</span>}
      </div>
    </header>
  );
}

export function CollegeSeasons({ detail }: { detail: ProspectDetail }) {
  const rows = detail.college_seasons as Rec[];
  if (!rows.length) return <p className="text-sm text-muted-foreground">No college passer-seasons in the data for this player.</p>;
  return (
    <div className="overflow-x-auto rounded-md border border-line">
      <table className="w-full min-w-[560px] text-left text-[13px]">
        <caption className="sr-only">College passing seasons</caption>
        <thead className="bg-panel text-muted-foreground">
          <tr>
            {["Season", "Team", "Att", "Cmp %", "Yards", "TD", "INT", "YPA", "Rush yds"].map((c) => (
              <th key={c} scope="col" className="px-3 py-2 font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={String(r.season) + String(r.team)} className="border-t border-line/60">
              <td className="num px-3 py-1.5">{String(r.season)}</td>
              <td className="px-3 py-1.5">{String(r.team ?? NA)}</td>
              <td className="num px-3 py-1.5">{fmtInt(num(r, "attempts"))}</td>
              <td className="num px-3 py-1.5">{fmt1(num(r, "completion_pct"))}</td>
              <td className="num px-3 py-1.5">{fmtInt(num(r, "passing_yards"))}</td>
              <td className="num px-3 py-1.5">{fmtInt(num(r, "passing_tds"))}</td>
              <td className="num px-3 py-1.5">{fmtInt(num(r, "interceptions"))}</td>
              <td className="num px-3 py-1.5">{fmt1(num(r, "yards_per_attempt"))}</td>
              <td className="num px-3 py-1.5">{fmtInt(num(r, "rushing_yards"))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
