/**
 * TEST FIXTURES ONLY. Synthetic players ("Fixture QB ...") with made-up numbers, shaped exactly like the API's
 * responses (the TypeScript types are generated from the backend's OpenAPI schema, so a fixture that drifts from the
 * real contract fails to compile). Never imported by application code.
 */
import type {
  BacktestResponse,
  ComparisonsResponse,
  ForecastResponse,
  PerformanceResponse,
  ProspectDetail,
  ProspectSearchResponse,
  ProspectSummary,
  StatForecast,
} from "@/lib/api/types";

export const summary = (o: Partial<ProspectSummary> = {}): ProspectSummary => ({
  player_id: "qb_9001",
  name: "Fixture QB Alpha",
  college: "Fixture State",
  draft_year: 2019,
  final_college_season: 2018,
  drafted: true,
  draft_round: 1,
  overall_pick: 12,
  kind: "drafted",
  shares_name_with_other_prospects: false,
  ...o,
});

export const searchResponse: ProspectSearchResponse = {
  query: "fixture",
  draft_year: null,
  total: 3,
  limit: 8,
  offset: 0,
  items: [
    summary(),
    summary({ player_id: "qb_9002", name: "Fixture Twin", college: "North Fixture", draft_year: 2015, shares_name_with_other_prospects: true }),
    summary({
      player_id: "cfb_123456_2020",
      name: "Fixture Twin",
      college: "South Fixture",
      draft_year: 2021,
      final_college_season: 2020,
      drafted: false,
      draft_round: null,
      overall_pick: null,
      kind: "college_only",
      shares_name_with_other_prospects: true,
    }),
  ],
};

export const detail = (o: Partial<ProspectSummary> = {}): ProspectDetail => ({
  prospect: summary(o),
  identifiers: { qb_draft_prospect_id: 9001, cfbd_player_id: "1", gsis_id: null },
  draft_status: { label: "observed", drafted: true, round: 1, overall_pick: 12, note: null },
  final_season_statistics: { attempts: 400, completion_pct: 66.1, yards_per_attempt: 8.4, td_pct: 7.2, int_pct: 2.1, opponent_adjusted_ypa: 8.0, rush_yards_per_game_last3: 21.5 },
  college_seasons: [{ season: 2018, team: "Fixture State", attempts: 400, completion_pct: 66.1, passing_yards: 3360, passing_tds: 29, interceptions: 8, yards_per_attempt: 8.4, rushing_yards: 250 }],
  offensive_profile: { team_pass_rate: 0.5 },
  athletic_measurements: {
    height_inches: { value: 75, source: "college roster" },
    weight_lbs: { value: 220, source: "college roster" },
    combine_height_inches: { value: null, source: "NFL Combine" },
    combine_weight_lbs: { value: null, source: "NFL Combine" },
    combine_forty_seconds: { value: null, source: "NFL Combine" },
  },
  combine_participant: false,
  data_coverage: { career_model_features_present: 10, career_model_features_total: 11, population_status: "in_population" },
});

const stat = (k: string, label: string, e: number): StatForecast => ({
  stat: k,
  label,
  expected: e,
  intervals: [
    { level: 0.5, lower: e * 0.6, upper: e * 1.4 },
    { level: 0.8, lower: e * 0.3, upper: e * 1.9 },
    { level: 0.9, lower: e * 0.15, upper: e * 2.3 },
  ],
  model: "poisson",
  interval_method: "oof_scaled",
  measured_coverage: { c50: 0.5, c80: 0.79, c90: 0.87 },
  training_max_5yr: 21000,
  upper_bound_beyond_training_max: false,
});

const STATS: [string, string, number][] = [
  ["games_played", "Games played", 40],
  ["games_started", "Games started", 30],
  ["pass_attempts", "Pass attempts", 1000],
  ["pass_completions", "Completions", 620],
  ["pass_yards", "Passing yards", 7111],
  ["pass_tds", "Passing TDs", 44],
  ["interceptions", "Interceptions", 26],
  ["rush_attempts", "Rush attempts", 120],
  ["rush_yards", "Rushing yards", 500],
  ["rush_tds", "Rushing TDs", 6],
];

export const forecast = (mode: "pre_draft" | "post_draft" = "pre_draft", o: Partial<ForecastResponse> = {}): ForecastResponse => ({
  api_version: "1.0.0",
  schema_version: "1.0.0",
  mode,
  player: {
    player_id: "qb_9001",
    name: "Fixture QB Alpha",
    college: "Fixture State",
    draft_year: 2019,
    final_college_season: 2018,
    identifiers: { qb_draft_prospect_id: 9001, cfbd_player_id: "1", gsis_id: null },
    identity_verification: "fixture",
    actual_draft: mode === "post_draft" ? { drafted: true, round: 1, overall_pick: 12 } : null,
  },
  draft: {
    status: "available",
    reason: null,
    code: null,
    probability: { status: "unavailable", reason: "outside the verified prospect universe (no_evidence_of_four_college_years); P(drafted) is not extrapolated", code: "PREDICTION_UNSUPPORTED", pre_combine: null, post_combine: null, post_combine_note: null, population: null, model_version: null, backtest: null },
    round: {
      status: "available",
      reason: null,
      conditional_on: "being drafted",
      groups: { "Round 1": 0.6, "Rounds 2-3": 0.25, "Rounds 4-7": 0.15 },
      rounds: { "1": 0.6, "2": 0.15, "3": 0.1, "4": 0.06, "5": 0.04, "6": 0.03, "7": 0.02 },
      expected_round: 1.9,
      display_note: "show the three groups",
      model_version: "qb_draft v1.1.0",
      backtest_rps: { groups: 0.176, rounds: 0.176 },
    },
    pick: {
      status: "available",
      reason: null,
      conditional_on: "being drafted",
      median: 20,
      intervals: [
        { level: 0.5, lower: 10, upper: 40 },
        { level: 0.8, lower: 4, upper: 90 },
        { level: 0.9, lower: 2, upper: 130 },
      ],
      measured_coverage: { coverage50: 0.46, coverage80: 0.75, coverage90: 0.89 },
      model_version: "qb_draft v1.1.0",
    },
    eligibility: { classification: "eligibility_unknown" },
  },
  career: {
    status: "available",
    reason: null,
    code: null,
    hypothetical: false,
    assumptions: [],
    model_version: "qb_career_production v1.1.0",
    start_probabilities: {
      status: "available",
      p_16_plus: 0.71,
      p_32_plus: 0.52,
      p_64_plus: 0.18,
      p_any_nfl_game: 0.93,
      window: "NFL seasons 1-5 (regular season starts)",
      method: "fixture",
      nested: true,
      warnings: [],
    },
    five_year_totals: STATS.map(([k, l, e]) => stat(k, l, e)),
    year_by_year: { status: "unavailable", reason: "reference tables not in this checkout", method: null, interval_note: null, seasons: [] },
    reference_v2: { status: "available", reason: null, model_version: "qb_career_success v2.0.0", p_32_plus: 0.5 },
    efficiency_note: "efficiency rates are not forecast",
  },
  warnings: [{ code: "CALIBRATION_WARNING", message: "P(64+) pre-draft is poorly calibrated" }],
  unavailable_blocks: ["draft.probability"],
  uncertainty: {
    model_versions: { career: "qb_career_production v1.1.0" },
    calibration_status: { five_year_intervals: { pre_draft: { nominal_50: 0.5, nominal_80: 0.79, nominal_90: 0.87 }, post_draft: { nominal_50: 0.5, nominal_80: 0.77, nominal_90: 0.87 } } },
    data_limitations: ["Fixture limitation."],
  },
  ...o,
});

export const careerUnavailable = (): ForecastResponse =>
  forecast("post_draft", {
    career: {
      status: "unavailable",
      reason: "post_draft mode needs an actual overall pick",
      code: "PREDICTION_UNSUPPORTED",
      hypothetical: false,
      assumptions: [],
      model_version: null,
      start_probabilities: null,
      five_year_totals: [],
      year_by_year: null,
      reference_v2: null,
      efficiency_note: null,
    },
  });

export const comparisons: ComparisonsResponse = {
  player_id: "qb_9001",
  status: "available",
  reason: null,
  kind: "explanatory",
  model_version: "qb_similarity v1.0.0",
  note: "fixture",
  comparables: [1, 2, 3, 4, 5].map((i) => ({
    player_id: `qb_80${i}`,
    name: `Fixture Comp ${i}`,
    college: "Comp U",
    draft_year: 2010 + i,
    round: i,
    overall_pick: i * 20,
    distance: 2 + i / 10,
    closeness_index_heuristic: 99 - i,
    most_similar_on: "completion %",
    biggest_differences: "height lower (-1.0 SD)",
    nfl_outcome: { label: "observed", starts_first_5_seasons: i * 10, starts_to_date: null, window_complete: true },
  })),
};

const btStats = (complete: boolean) =>
  STATS.filter(([k]) => k !== "pass_completions").map(([k, , e]) => ({
    stat: k,
    predicted: e,
    interval_80: { level: 0.8, lower: e * 0.3, upper: e * 1.9 },
    actual: e * 1.5,
    actual_label: complete ? "observed (five seasons)" : "observed to date (2 seasons)",
    error: complete ? -e * 0.5 : null,
    inside_80: complete ? true : null,
    model: "poisson",
  }));

export const backtest = (status: BacktestResponse["classification"]["status"]): BacktestResponse => {
  const complete = status !== "out_of_sample_forecast_outcome_incomplete";
  const view = {
    source: "fixture",
    statistics: btStats(complete),
    start_probabilities: [16, 32, 64].map((t) => ({ threshold: t, predicted: 0.5, actual_event: complete ? t < 64 : null })),
  };
  return {
    player_id: "qb_9001",
    player: { name: "Fixture QB Alpha", college: "Fixture State", draft_year: 2016, round: 1, overall_pick: 12 },
    classification: {
      status,
      explanation:
        status === "retrospective_illustration_in_sample" ? "shown with the final model, which was TRAINED ON THIS QB" : "models trained only on earlier classes",
      in_sample: status === "retrospective_illustration_in_sample",
    },
    prediction_cutoff: "before the 2016 NFL Draft (late April 2016)",
    training_classes: status === "retrospective_illustration_in_sample" ? null : [2005, 2011],
    n_training_qbs: status === "retrospective_illustration_in_sample" ? null : 70,
    label_embargo_years: 5,
    model_version: "fixture",
    actual_window_complete: complete,
    observed_seasons: complete ? 5 : 2,
    horizons: { five_year: { status: "available" }, one_year: { status: "unavailable", reason: "x" }, three_year: { status: "unavailable", reason: "x" } },
    views: { pre_draft: view, post_draft: view },
  };
};

export const performance: PerformanceResponse = {
  generated_from: "fixture",
  evaluation_status: {
    draft_probability: "fixture draft status",
    draft_round_and_pick: "fixture round status",
    starter_probabilities: "fixture starter status",
    production: "fixture production status",
    intervals: "fixture interval status",
    comparisons: "fixture comparisons status",
  },
  limitations: ["Fixture limitation one."],
  sections: {
    draft_probability: {
      population: "fixture population",
      test_rows: 100,
      test_drafted: 25,
      baseline_brier_historical_rate: 0.18,
      note: "",
      models: [
        { variant: "A_college", auc: 0.8, auc_lo: 0.75, auc_hi: 0.85, brier: 0.12, log_loss: 0.4, calibration_slope: 0.9 },
        { variant: "B_college_invited", auc: 0.88, auc_lo: 0.84, auc_hi: 0.92, brier: 0.1, log_loss: 0.35, calibration_slope: 0.9 },
        { variant: "C_college_invited_measures", auc: 0.9, auc_lo: 0.86, auc_hi: 0.94, brier: 0.09, log_loss: 0.33, calibration_slope: 0.9 },
      ],
    },
    draft_round: {
      note: "",
      seven_rounds: [{ model: "selected", n: 100, rps: 0.17, rps_lo: 0.15, rps_hi: 0.19, log_loss: 1.7, top_class_accuracy: 0.3 }],
      three_groups: [
        { model: "selected", n: 100, rps: 0.17, rps_lo: 0.15, rps_hi: 0.19, log_loss: 0.9, top_class_accuracy: 0.6 },
        { model: "freq", n: 100, rps: 0.22, rps_lo: 0.2, rps_hi: 0.24, log_loss: 1.0, top_class_accuracy: 0.5 },
      ],
    },
    pick: {
      errors: [
        { model: "ridge_log_pick_nested", mae_picks: 70, median_abs_error_picks: 47 },
        { model: "historical_median", mae_picks: 74, median_abs_error_picks: 81 },
      ],
      coverage: [{ interval_method: "groupkfold", coverage50: 0.46, coverage80: 0.75, coverage90: 0.89 }],
      served_interval_method: "groupkfold",
    },
    starter_probabilities: ([16, 32, 64] as const).flatMap((t) =>
      (["pre_draft", "post_draft"] as const).flatMap((fs) => [
        { feature_set: fs, model: "base_rate" as const, threshold: t, n: 107, n_pos: 20, auc: 0.5, brier: 0.2, log_loss: 0.6, calibration_slope: null, calibration_intercept: null },
        { feature_set: fs, model: "nested_logit_pava" as const, threshold: t, n: 107, n_pos: 20, auc: fs === "pre_draft" ? 0.77 : 0.88, brier: 0.15, log_loss: 0.5, calibration_slope: 1, calibration_intercept: 0 },
      ]),
    ),
    production: {
      test_qbs: 107,
      test_classes: [2012, 2021],
      errors: (["pre_draft", "post_draft"] as const).flatMap((v) =>
        STATS.map(([k]) => ({ view: v, statistic: k, mae_shipped_family: 90, mae_nested_selection: v === "pre_draft" ? 98 : 74, mae_historical_median: 100, shipped_family: "poisson" })),
      ),
    },
    interval_calibration: {
      served_method: { pre_draft: "oof_scaled", post_draft: "oof_scaled" },
      coverage_by_statistic: STATS.flatMap(([k]) => (["pre_draft", "post_draft"] as const).map((fs) => ({ feature_set: fs, target: k, coverage50: 0.5, coverage80: 0.8, coverage90: 0.88 }))),
      mean_coverage: { pre_draft: { coverage50: 0.5, coverage80: 0.79, coverage90: 0.87 }, post_draft: { coverage50: 0.5, coverage80: 0.77, coverage90: 0.87 } },
    },
    comparisons: { knn_outcome_rate_auc_32plus_starts: 0.69, note: "" },
    model_versions: { career: "qb_career_production v1.1.0" },
  },
};
