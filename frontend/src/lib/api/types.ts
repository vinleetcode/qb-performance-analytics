/**
 * Response types. Endpoint shapes come from the backend's OpenAPI schema (schema.d.ts, regenerated with
 * `npm run gen:api` while the API is running). The performance summary is typed `dict` on the backend, so its
 * sections are typed here by hand against the real /api/models/performance response.
 */
import type { components } from "./schema";

/**
 * FastAPI serializes every field of a response model (fields without a value are sent as null), but the OpenAPI
 * schema marks fields that have defaults as optional. `Wire<T>` restores the shape actually received: every key
 * present, nullability unchanged.
 */
type Wire<T> = T extends (infer U)[]
  ? Wire<U>[]
  : T extends object
    ? { [K in keyof T]-?: Wire<Exclude<T[K], undefined>> }
    : T;
type Raw = components["schemas"];
type S = { [K in keyof Raw]: Wire<Raw[K]> };
export type HealthResponse = S["HealthResponse"];
export type ProspectSummary = S["ProspectSummary"];
export type ProspectSearchResponse = S["ProspectSearchResponse"];
export type ProspectDetail = S["ProspectDetail"];
export type ForecastResponse = S["ForecastResponse"];
export type DraftBlock = S["DraftBlock"];
export type CareerBlock = S["CareerBlock"];
export type StatForecast = S["StatForecast"];
export type Interval = S["Interval"];
export type StartProbabilities = S["StartProbabilities"];
export type YearByYear = S["YearByYear"];
export type PredictionWarning = S["PredictionWarning"];
export type ComparisonsResponse = S["ComparisonsResponse"];
export type Comparable = S["Comparable"];
export type BacktestResponse = S["BacktestResponse"];
export type BacktestStat = S["BacktestStat"];
export type BacktestView = S["BacktestView"];
export type ModelsResponse = S["ModelsResponse"];
export type ErrorResponse = S["ErrorResponse"];
export type Mode = "pre_draft" | "post_draft";
export type BacktestStatus = BacktestResponse["classification"]["status"];

export interface DraftProbabilityVariant {
  variant: "A_college" | "B_college_invited" | "C_college_invited_measures";
  auc: number;
  auc_lo: number;
  auc_hi: number;
  brier: number;
  log_loss: number;
  calibration_slope: number;
}
export interface RoundMetric {
  model: string;
  n: number;
  rps: number;
  rps_lo: number;
  rps_hi: number;
  log_loss: number;
  top_class_accuracy: number;
}
export interface StarterMetric {
  feature_set: Mode;
  model: "base_rate" | "nested_logit_pava";
  threshold: 16 | 32 | 64;
  n: number;
  n_pos: number;
  auc: number;
  brier: number;
  log_loss: number;
  calibration_slope: number | null;
  calibration_intercept: number | null;
}
export interface ProductionError {
  view: Mode;
  statistic: string;
  mae_shipped_family: number;
  mae_nested_selection: number;
  mae_historical_median: number;
  shipped_family: string;
}
export interface CoverageRow {
  feature_set: Mode;
  target: string;
  coverage50: number;
  coverage80: number;
  coverage90: number;
}
export interface PerformanceSections {
  draft_probability: {
    population: string;
    test_rows: number;
    test_drafted: number;
    models: DraftProbabilityVariant[];
    baseline_brier_historical_rate: number;
    note: string;
  };
  draft_round: { seven_rounds: RoundMetric[]; three_groups: RoundMetric[]; note: string };
  pick: {
    errors: { model: string; mae_picks: number; median_abs_error_picks: number }[];
    coverage: { interval_method: string; coverage50: number; coverage80: number; coverage90: number }[];
    served_interval_method: string;
  };
  starter_probabilities: StarterMetric[];
  production: { test_qbs: number; test_classes: [number, number]; errors: ProductionError[] };
  interval_calibration: {
    served_method: Record<Mode, string>;
    coverage_by_statistic: CoverageRow[];
    mean_coverage: Record<Mode, { coverage50: number; coverage80: number; coverage90: number }>;
  };
  comparisons: { knn_outcome_rate_auc_32plus_starts: number | null; note: string };
  model_versions: Record<string, string>;
}
export interface PerformanceResponse {
  generated_from: string;
  evaluation_status: Record<string, string>;
  sections: PerformanceSections;
  limitations: string[];
}
