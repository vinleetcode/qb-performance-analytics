import type { Interval, StatForecast } from "@/lib/api/types";

/** The five statistics the report leads with, in reading order. */
export const HEADLINE_STATS = ["pass_yards", "pass_tds", "interceptions", "rush_yards", "games_started"] as const;

/** Plain-language unit for each five-season total (used in the chart axis and the readouts). */
export const STAT_UNIT: Record<string, { one: string; many: string }> = {
  pass_yards: { one: "yard", many: "yards" },
  rush_yards: { one: "yard", many: "yards" },
  pass_tds: { one: "touchdown", many: "touchdowns" },
  rush_tds: { one: "touchdown", many: "touchdowns" },
  interceptions: { one: "interception", many: "interceptions" },
  games_started: { one: "start", many: "starts" },
  games_played: { one: "game", many: "games" },
  pass_attempts: { one: "attempt", many: "attempts" },
  pass_completions: { one: "completion", many: "completions" },
  rush_attempts: { one: "attempt", many: "attempts" },
};
export const unitFor = (stat: string, v?: number) => {
  const u = STAT_UNIT[stat];
  if (!u) return "";
  return v != null && Math.round(v) === 1 ? u.one : u.many;
};

export const intervalAt = (s: StatForecast, level: number): Interval | undefined =>
  s.intervals.find((i) => Math.abs(i.level - level) < 1e-9);

/**
 * True when the 50% range lies inside the 80% range, as the API promises. The UI checks this rather than assuming
 * it, and says so if it ever fails.
 */
export function intervalsNested(s: StatForecast): boolean {
  const a = intervalAt(s, 0.5);
  const b = intervalAt(s, 0.8);
  if (!a || !b) return true; // nothing to compare
  const eps = 1e-6 * Math.max(1, Math.abs(b.upper));
  return a.lower >= b.lower - eps && a.upper <= b.upper + eps && a.lower <= a.upper && b.lower <= b.upper;
}

/**
 * Backtest coverage of the nominal 80% range, read honestly:
 *  - below 75%: real careers fell outside the range noticeably more often than one in five, so the range runs narrow;
 *  - above 85%: the range was wider than it needed to be;
 *  - otherwise close to target.
 * Measured on 107 drafted quarterbacks from later draft classes, so a few points either way is within noise
 * (one standard error is about 4 points); the thresholds flag only gaps larger than that.
 */
export type CoverageRead = "narrow" | "wide" | "on_target";
export const COVERAGE_LOW = 0.75;
export const COVERAGE_HIGH = 0.85;
export function readCoverage(c80: number | null | undefined): CoverageRead | null {
  if (c80 == null || !Number.isFinite(c80)) return null;
  // compare the whole percentage the page shows, so a flag never contradicts the number beside it (74.8% shows as 75%)
  const pct = Math.round(c80 * 100);
  if (pct < Math.round(COVERAGE_LOW * 100)) return "narrow";
  if (pct > Math.round(COVERAGE_HIGH * 100)) return "wide";
  return "on_target";
}
export const coverage80 = (s: StatForecast) => s.measured_coverage?.c80 ?? null;

/** Lower-case a label for use mid-sentence without breaking abbreviations: "Passing TDs" -> "passing TDs". */
export const midSentence = (label: string) =>
  label
    .split(" ")
    .map((w) => ((w.match(/[A-Z]/g)?.length ?? 0) > 1 ? w : w.toLowerCase()))
    .join(" ");
