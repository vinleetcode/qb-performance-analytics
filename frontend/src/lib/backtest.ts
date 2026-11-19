/**
 * Backtest labels and summary helpers shared by chart and non-chart pages. Kept free of chart imports so pages that
 * only need the labels (the homepage) do not download the charting library.
 */
import { CheckCircle2, Clock, FlaskConical } from "lucide-react";
import type { BacktestStatus, Mode, ProductionError } from "@/lib/api/types";

export const CLASS_INFO: Record<BacktestStatus, { title: string; short: string; icon: typeof CheckCircle2; tone: string }> = {
  valid_out_of_sample_backtest: {
    title: "Genuine out-of-sample backtest",
    short: "Out of sample",
    icon: CheckCircle2,
    tone: "border-projection text-projection-text",
  },
  out_of_sample_forecast_outcome_incomplete: {
    title: "Out-of-sample forecast, NFL window still open",
    short: "Window open",
    icon: Clock,
    tone: "border-reference text-chalk",
  },
  retrospective_illustration_in_sample: {
    title: "In-sample illustration, not a backtest",
    short: "In sample",
    icon: FlaskConical,
    tone: "border-caution text-caution",
  },
};

/** Mean skill of the selected model against the historical-median baseline for one prediction view. */
export const avgSkill = (errors: ProductionError[], view: Mode) => {
  const r = errors.filter((e) => e.view === view);
  return r.reduce((a, e) => a + (1 - e.mae_nested_selection / e.mae_historical_median), 0) / Math.max(r.length, 1);
};
