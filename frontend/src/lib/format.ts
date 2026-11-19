const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const ok = (v: number | null | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v);

export const NA = "n/a";
export const fmtInt = (v: number | null | undefined) => (ok(v) ? nf0.format(Math.round(v)) : NA);
export const fmt1 = (v: number | null | undefined) => (ok(v) ? nf1.format(v) : NA);
export const fmt3 = (v: number | null | undefined) => (ok(v) ? v.toFixed(3) : NA);
export const fmtPct = (p: number | null | undefined, digits = 0) => (ok(p) ? `${(100 * p).toFixed(digits)}%` : NA);
/** Probabilities near 0 or 1 are shown as bounds rather than with false precision. */
export const fmtProb = (p: number | null | undefined) => {
  if (!ok(p)) return NA;
  if (p >= 0.995) return ">99%";
  if (p <= 0.005) return "<1%";
  return `${Math.round(100 * p)}%`;
};
export const heightFt = (inches: number | null | undefined) => (ok(inches) ? `${Math.floor(inches / 12)}'${Math.round(inches % 12)}"` : NA);

export const STAT_LABEL: Record<string, string> = {
  games_played: "Games played",
  games_started: "Games started",
  pass_attempts: "Pass attempts",
  pass_completions: "Completions",
  pass_yards: "Passing yards",
  pass_tds: "Passing TDs",
  interceptions: "Interceptions",
  rush_attempts: "Rush attempts",
  rush_yards: "Rushing yards",
  rush_tds: "Rushing TDs",
};
export const statLabel = (s: string) => STAT_LABEL[s] ?? s.replace(/_/g, " ");
/** Backend notes are sentence fragments; show them as sentences. */
export const sentence = (s: string | null | undefined) => {
  const t = (s ?? "").trim();
  if (!t) return "";
  const c = t[0].toUpperCase() + t.slice(1);
  return /[.!?]$/.test(c) ? c : `${c}.`;
};
/** Backend reasons can contain internal codes (e.g. no_evidence_of_four_college_years); show them as words. */
export const humanize = (s: string | null | undefined) =>
  sentence((s ?? "").replace(/[a-z]+(?:_[a-z0-9]+)+/g, (m) => m.replace(/_/g, " ")));
export const MODE_LABEL = { pre_draft: "Pre-draft", post_draft: "Post-draft" } as const;
