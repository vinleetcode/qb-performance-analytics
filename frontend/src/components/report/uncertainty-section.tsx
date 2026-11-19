"use client";
import Link from "next/link";
import type { ForecastResponse } from "@/lib/api/types";
import { fmtPct } from "@/lib/format";
import { WarningList } from "@/components/status";
import { Section } from "./section";

type Rec = Record<string, unknown>;

export function UncertaintySection({ f }: { f: ForecastResponse }) {
  const u = f.uncertainty as Rec;
  const versions = (u.model_versions ?? {}) as Record<string, string>;
  const cal = ((u.calibration_status as Rec | undefined)?.five_year_intervals ?? {}) as Record<string, Record<string, number>>;
  const mine = cal[f.mode];
  const limits = (u.data_limitations ?? []) as string[];
  return (
    <Section id="uncertainty" title="How much to trust this" intro="Warnings for this forecast, how the ranges performed on past draft classes, and what the models cannot see.">
      <div className="space-y-8">
        {f.warnings.length ? <WarningList warnings={f.warnings} /> : <p className="text-sm text-muted-foreground">No forecast-specific warnings.</p>}
        {mine && (
          <p className="measure text-[14px] text-chalk/90">
            On draft classes 2012 to 2021, which the models had not been trained on when they forecast them, the {f.mode === "pre_draft" ? "pre-draft" : "post-draft"}{" "}
            ranges contained the real five-season total {fmtPct(mine.nominal_50)} of the time for the 50% range,{" "}
            {fmtPct(mine.nominal_80)} for the 80% range and {fmtPct(mine.nominal_90)} for the 90% range. Expect individual
            careers to land outside the 80% range about one time in five.
          </p>
        )}
        <div>
          <h3 className="text-[20px] text-chalk">What the models cannot see</h3>
          <ul className="measure mt-3 list-disc space-y-1.5 pl-5 text-[14px] text-muted-foreground">
            {limits.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
        <p className="text-[13px] text-muted-foreground">
          Model versions: {Object.entries(versions).map(([k, v]) => `${k.replace(/_/g, " ")} ${v.replace(/^qb_[a-z_]+ /, "")}`).join(", ")}.{" "}
          <Link href="/methodology" className="text-projection-text underline decoration-1 hover:decoration-2">
            How these were built and tested
          </Link>
        </p>
      </div>
    </Section>
  );
}
