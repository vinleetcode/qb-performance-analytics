"use client";
import { useState } from "react";
import type { DraftBlock, ForecastResponse } from "@/lib/api/types";
import { fmtPct, fmtProb, sentence } from "@/lib/format";
import { FieldRuler, RulerLegend } from "@/components/field-ruler";
import { InfoTip } from "@/components/info-tip";
import { Unavailable } from "@/components/status";
import { Section } from "./section";

const RAMP = ["var(--projection)", "rgb(76 139 209 / 0.6)", "rgb(76 139 209 / 0.35)"];

export function ProbBar({ label, p, color = "var(--projection)", sub }: { label: string; p: number; color?: string; sub?: string }) {
  return (
    <div className="grid grid-cols-[minmax(7rem,9rem)_1fr_3.5rem] items-center gap-3">
      <div className="text-[14px] text-chalk">
        {label}
        {sub && <div className="text-[12px] text-muted-foreground">{sub}</div>}
      </div>
      <div className="h-3 rounded-[2px] bg-panel" aria-hidden>
        <div className="h-full rounded-[2px]" style={{ width: `${Math.max(0.5, p * 100)}%`, background: color }} />
      </div>
      <div className="num text-right text-[17px] font-semibold text-chalk">{fmtProb(p)}</div>
    </div>
  );
}

function DraftProbability({ d }: { d: NonNullable<DraftBlock["probability"]> }) {
  if (d.status !== "available")
    return <Unavailable title="Probability of being drafted is not available for this player" reason={d.reason} />;
  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <div>
        <p className="flex items-center gap-1 text-[13px] text-muted-foreground">
          After the college season
          <InfoTip label="pre-Combine probability">
            College statistics, team context and recruiting only. This is the forecast before NFL Combine invitations go out.
          </InfoTip>
        </p>
        <p className="num text-[44px] leading-none font-semibold text-chalk">{fmtProb(d.pre_combine)}</p>
      </div>
      <div>
        <p className="flex items-center gap-1 text-[13px] text-muted-foreground">
          After the Combine
          <InfoTip label="post-Combine probability">
            Adds Combine participation and measurements. Most of the extra accuracy comes from participation itself, which is
            the NFL&apos;s own scouting decision, not from 40 times or size.
          </InfoTip>
        </p>
        {d.post_combine != null ? (
          <p className="num text-[44px] leading-none font-semibold text-chalk">{fmtProb(d.post_combine)}</p>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">{d.post_combine_note ?? "Not available"}</p>
        )}
      </div>
      <p className="text-[13px] text-muted-foreground sm:col-span-2">{sentence(`Population: ${d.population}`)} {sentence(d.backtest)}</p>
    </div>
  );
}

export function DraftSection({ f }: { f: ForecastResponse }) {
  const [detail, setDetail] = useState(false);
  const d = f.draft;
  const actualPick = f.mode === "post_draft" ? (f.player.actual_draft?.overall_pick ?? null) : null;
  return (
    <Section
      id="draft"
      title="Draft projection"
      intro="How likely the NFL was to draft him, and where. Round and pick forecasts assume he is drafted."
    >
      {d.status !== "available" ? (
        <Unavailable title="Draft projection not available" reason={d.reason} />
      ) : (
        <div className="space-y-10">
          {d.probability && <DraftProbability d={d.probability} />}
          {d.round && d.round.status === "available" && (
            <div>
              <h3 className="text-[20px] text-chalk">Round, if drafted</h3>
              <div className="mt-4 space-y-3" role="list" aria-label="Probability of each round group, if drafted">
                {Object.entries(d.round.groups).map(([k, p], i) => (
                  <div role="listitem" key={k}>
                    <ProbBar label={k} p={p} color={RAMP[i]} />
                  </div>
                ))}
              </div>
              <button
                type="button"
                onClick={() => setDetail((s) => !s)}
                aria-expanded={detail}
                className="mt-3 text-[13px] text-projection-text underline decoration-1 hover:decoration-2"
              >
                {detail ? "Hide single rounds" : "Show single rounds"}
              </button>
              {detail && (
                <div className="mt-3 space-y-2">
                  <p className="text-[13px] text-muted-foreground">{sentence(d.round.display_note)}</p>
                  {Object.entries(d.round.rounds).map(([r, p]) => (
                    <ProbBar key={r} label={`Round ${r}`} p={p} color="rgb(76 139 209 / 0.55)" />
                  ))}
                </div>
              )}
              <p className="mt-3 text-[13px] text-muted-foreground">
                Expected round {d.round.expected_round?.toFixed(1)}. In backtests these round-group forecasts scored a ranked
                probability score of {d.round.backtest_rps.groups?.toFixed(3)} (lower is better), better than historical round
                frequencies; see the methodology page.
              </p>
            </div>
          )}
          {d.pick && d.pick.status === "available" && d.pick.median != null && (
            <div>
              <h3 className="flex items-center gap-1 text-[20px] text-chalk">
                Overall pick, if drafted
                <InfoTip label="pick range">
                  Median forecast pick with 50/80/90% ranges. In backtests the 80% range contained the real pick{" "}
                  {fmtPct(d.pick.measured_coverage.coverage80)} of the time, a little under its nominal 80%.
                </InfoTip>
              </h3>
              <FieldRuler
                className="mt-2"
                size="lg"
                label="Overall pick, if drafted"
                expected={d.pick.median}
                intervals={d.pick.intervals}
                actual={actualPick}
                actualLabel="Actual pick"
                domain={[1, 262]}
                format={(v) => `#${Math.round(v)}`}
              />
              <div className="mt-2">
                <RulerLegend actual={actualPick != null} />
              </div>
            </div>
          )}
          {d.eligibility && (
            <p className="text-[13px] text-muted-foreground">
              Eligibility: {String(d.eligibility.classification ?? "unknown").replace(/_/g, " ")}.
            </p>
          )}
        </div>
      )}
    </Section>
  );
}
