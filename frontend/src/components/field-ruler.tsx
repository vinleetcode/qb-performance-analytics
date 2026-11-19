import { cn } from "@/lib/utils";
import { fmtInt } from "@/lib/format";
import type { Interval } from "@/lib/api/types";

/**
 * The field ruler: a forecast's uncertainty drawn as nested bands on a yard-line scale.
 * 90% band (faintest), 80% band, 50% band (strongest), and a chalk tick at the expected value. An observed
 * result is a separate amber flag; a reference value (e.g. a historical maximum) is a dashed neutral line.
 * Every number is also in the accessible label, so nothing depends on colour alone.
 */
export interface FieldRulerProps {
  label: string;
  expected?: number | null;
  intervals: Interval[];
  actual?: number | null;
  actualLabel?: string;
  reference?: { value: number; label: string } | null;
  domain?: [number, number];
  format?: (v: number) => string;
  unit?: string;
  size?: "md" | "lg";
  className?: string;
  showScale?: boolean;
  /** print the expected value above its tick (off where the number is already shown beside the ruler) */
  showValue?: boolean;
}

function niceMax(v: number) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  const step = m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10;
  return step * p;
}

export function rulerDomain(values: (number | null | undefined)[], floor = 0): [number, number] {
  const finite = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const lo = Math.min(floor, ...finite);
  const hi = niceMax(Math.max(...finite, 1) * 1.04);
  return [lo < 0 ? -niceMax(-lo) : lo, hi];
}

export function FieldRuler({
  label,
  expected,
  intervals,
  actual,
  actualLabel = "Actual",
  reference,
  domain,
  format = fmtInt,
  unit = "",
  size = "md",
  className,
  showScale = true,
  showValue = true,
}: FieldRulerProps) {
  const byLevel = new Map(intervals.map((i) => [Math.round(i.level * 100), i]));
  const [d0, d1] =
    domain ?? rulerDomain([expected, actual, reference?.value, ...intervals.flatMap((i) => [i.lower, i.upper])]);
  const pos = (v: number) => `${Math.max(0, Math.min(100, ((v - d0) / (d1 - d0 || 1)) * 100))}%`;
  const width = (a: number, b: number) => `${Math.max(0.6, ((Math.min(b, d1) - Math.max(a, d0)) / (d1 - d0 || 1)) * 100)}%`;
  const parts = [
    expected != null ? `expected ${format(expected)}${unit}` : null,
    ...[50, 80, 90].map((l) => {
      const i = byLevel.get(l);
      return i ? `${l}% range ${format(i.lower)} to ${format(i.upper)}` : null;
    }),
    actual != null ? `${actualLabel.toLowerCase()} ${format(actual)}${unit}` : null,
    reference ? `${reference.label} ${format(reference.value)}` : null,
  ].filter(Boolean);
  const h = size === "lg" ? "h-7" : "h-5";
  const i80 = byLevel.get(80);
  return (
    <div className={cn("w-full", className)}>
      <div role="img" aria-label={`${label}: ${parts.join("; ")}`} className={cn("relative pb-1", showValue ? "pt-5" : "pt-1.5")}>
        <div className={cn("relative w-full rounded-[2px] bg-panel ruler-track", h)}>
          {[90, 80, 50].map((l) => {
            const i = byLevel.get(l);
            if (!i) return null;
            return (
              <div
                key={l}
                className="absolute inset-y-0 rounded-[2px]"
                style={{
                  left: pos(i.lower),
                  width: width(i.lower, i.upper),
                  background: `var(--projection-${l})`,
                }}
              />
            );
          })}
          {reference && (
            <div
              className="absolute -inset-y-1 border-l border-dashed border-reference"
              style={{ left: pos(reference.value) }}
              title={`${reference.label}: ${format(reference.value)}`}
            />
          )}
          {expected != null && (
            <div className="absolute -inset-y-1 w-[3px] -translate-x-1/2 rounded-full bg-chalk" style={{ left: pos(expected) }} />
          )}
          {actual != null && (
            <div className="absolute -inset-y-2 -translate-x-1/2" style={{ left: pos(actual) }}>
              <div className="h-full w-[2px] bg-observed" />
              <div
                className="absolute -top-1 left-[2px] h-0 w-0 border-y-[5px] border-l-[8px] border-y-transparent border-l-observed"
                aria-hidden
              />
            </div>
          )}
        </div>
        {showValue && expected != null && (
          <span
            className="num absolute top-0 -translate-x-1/2 text-[13px] font-semibold text-chalk whitespace-nowrap"
            style={{ left: pos(expected) }}
          >
            {format(expected)}
          </span>
        )}
      </div>
      {showScale && (
        <div className="num flex justify-between text-[12px] text-muted-foreground" aria-hidden>
          <span>{i80 ? `80%: ${format(i80.lower)}` : format(d0)}</span>
          <span>{i80 ? `to ${format(i80.upper)}` : format(d1)}</span>
        </div>
      )}
    </div>
  );
}

export function RulerLegend({ actual = false, reference }: { actual?: boolean; reference?: string }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-muted-foreground" aria-label="Chart legend">
      <li className="flex items-center gap-1.5">
        <span className="inline-block h-3 w-[3px] rounded-full bg-chalk" aria-hidden /> Expected
      </li>
      {[50, 80, 90].map((l) => (
        <li key={l} className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-5 rounded-[2px]" style={{ background: `var(--projection-${l})` }} aria-hidden />
          {l}% range
        </li>
      ))}
      {actual && (
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-[2px] bg-observed" aria-hidden /> Actual
        </li>
      )}
      {reference && (
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 border-l border-dashed border-reference" aria-hidden /> {reference}
        </li>
      )}
    </ul>
  );
}
