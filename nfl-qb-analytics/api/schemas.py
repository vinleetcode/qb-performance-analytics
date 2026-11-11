"""Pydantic response models. Every block says whether it is available; unavailable blocks carry a
reason and a stable code and never carry numbers. Numbers are plain JSON floats (no NaN/inf)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Status = Literal['available', 'unavailable']
Mode = Literal['pre_draft', 'post_draft']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ErrorBody(Strict):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(Strict):
    error: ErrorBody


class Interval(Strict):
    level: float = Field(description='nominal coverage, e.g. 0.8')
    lower: float
    upper: float


class PredictionWarning(Strict):
    code: str
    message: str


class ModelInfo(Strict):
    name: str
    version: str
    available: bool
    verified: bool


# ------------------------------------------------------------------ health / models
class HealthResponse(Strict):
    status: Literal['ok', 'degraded']
    api_version: str
    data_source: str
    engine_loaded: bool
    bundle_version: Optional[str] = None
    models: list[ModelInfo]


class ModelsResponse(Strict):
    api_version: str
    models: list[dict[str, Any]]


class PerformanceResponse(Strict):
    model_config = ConfigDict(extra='allow')
    generated_from: str
    evaluation_status: dict[str, str]
    sections: dict[str, Any]
    limitations: list[str]


# ------------------------------------------------------------------ prospects
class Identifiers(Strict):
    qb_draft_prospect_id: Optional[int] = None
    cfbd_player_id: Optional[str] = None
    gsis_id: Optional[str] = None


class ProspectSummary(Strict):
    player_id: str
    name: str
    college: Optional[str] = None
    draft_year: int = Field(description='NFL draft the prospect entered (or would enter) after the final college season')
    final_college_season: Optional[int] = None
    drafted: Optional[bool] = Field(None, description='null when not yet verifiable')
    draft_round: Optional[int] = None
    overall_pick: Optional[int] = None
    kind: Literal['drafted', 'college_only']
    shares_name_with_other_prospects: bool = False


class ProspectSearchResponse(Strict):
    query: Optional[str]
    draft_year: Optional[int]
    total: int
    limit: int
    offset: int
    items: list[ProspectSummary]


class Measurement(Strict):
    value: Optional[float] = None
    source: str


class ProspectDetail(Strict):
    prospect: ProspectSummary
    identifiers: Identifiers
    draft_status: dict[str, Any]
    final_season_statistics: dict[str, Optional[float]]
    college_seasons: list[dict[str, Any]]
    offensive_profile: dict[str, Any]
    athletic_measurements: dict[str, Measurement]
    combine_participant: Optional[bool] = None
    data_coverage: dict[str, Any]


# ------------------------------------------------------------------ forecast
class DraftProbability(Strict):
    status: Status
    reason: Optional[str] = None
    code: Optional[str] = None
    pre_combine: Optional[float] = Field(None, ge=0, le=1)
    post_combine: Optional[float] = Field(None, ge=0, le=1)
    post_combine_note: Optional[str] = None
    population: Optional[str] = None
    model_version: Optional[str] = None
    backtest: Optional[str] = None


class RoundProjection(Strict):
    status: Status
    reason: Optional[str] = None
    conditional_on: Optional[str] = None
    groups: dict[str, float] = Field(default_factory=dict)
    rounds: dict[str, float] = Field(default_factory=dict)
    expected_round: Optional[float] = None
    display_note: Optional[str] = None
    model_version: Optional[str] = None
    backtest_rps: dict[str, float] = Field(default_factory=dict)


class PickProjection(Strict):
    status: Status
    reason: Optional[str] = None
    conditional_on: Optional[str] = None
    median: Optional[float] = None
    intervals: list[Interval] = Field(default_factory=list)
    measured_coverage: dict[str, float] = Field(default_factory=dict)
    model_version: Optional[str] = None


class DraftBlock(Strict):
    status: Status
    reason: Optional[str] = None
    code: Optional[str] = None
    probability: Optional[DraftProbability] = None
    round: Optional[RoundProjection] = None
    pick: Optional[PickProjection] = None
    eligibility: dict[str, Any] = Field(default_factory=dict)


class StartProbabilities(Strict):
    status: Status
    p_16_plus: float = Field(ge=0, le=1)
    p_32_plus: float = Field(ge=0, le=1)
    p_64_plus: float = Field(ge=0, le=1)
    p_any_nfl_game: Optional[float] = Field(None, ge=0, le=1)
    window: str
    method: str
    nested: bool
    warnings: list[str] = Field(default_factory=list)


class StatForecast(Strict):
    stat: str
    label: str
    expected: float
    intervals: list[Interval]
    model: str
    interval_method: str
    measured_coverage: Optional[dict[str, float]] = None
    training_max_5yr: Optional[float] = None
    upper_bound_beyond_training_max: Optional[bool] = None


class SeasonStat(Strict):
    expected: float
    interval_80: Interval


class SeasonForecast(Strict):
    season: int
    stats: dict[str, SeasonStat]


class YearByYear(Strict):
    status: Status
    reason: Optional[str] = None
    method: Optional[str] = None
    interval_note: Optional[str] = None
    seasons: list[SeasonForecast] = Field(default_factory=list)


class ReferenceProbability(Strict):
    status: Status
    reason: Optional[str] = None
    model_version: Optional[str] = None
    p_32_plus: Optional[float] = Field(None, ge=0, le=1)


class CareerBlock(Strict):
    status: Status
    reason: Optional[str] = None
    code: Optional[str] = None
    hypothetical: bool = False
    assumptions: list[str] = Field(default_factory=list)
    model_version: Optional[str] = None
    start_probabilities: Optional[StartProbabilities] = None
    five_year_totals: list[StatForecast] = Field(default_factory=list)
    year_by_year: Optional[YearByYear] = None
    reference_v2: Optional[ReferenceProbability] = None
    efficiency_note: Optional[str] = None


class ActualDraft(Strict):
    drafted: Optional[bool] = None
    round: Optional[int] = None
    overall_pick: Optional[int] = None


class ForecastPlayer(Strict):
    player_id: str
    name: str
    college: Optional[str] = None
    draft_year: int
    final_college_season: Optional[int] = None
    identifiers: Identifiers
    identity_verification: str
    actual_draft: Optional[ActualDraft] = Field(None, description='observed draft result; omitted in pre_draft mode')


class ForecastResponse(Strict):
    api_version: str
    schema_version: str
    mode: Mode
    player: ForecastPlayer
    draft: DraftBlock
    career: CareerBlock
    warnings: list[PredictionWarning]
    unavailable_blocks: list[str]
    uncertainty: dict[str, Any]


# ------------------------------------------------------------------ comparisons
class ComparableOutcome(Strict):
    label: Literal['observed']
    starts_first_5_seasons: Optional[int] = None
    starts_to_date: Optional[int] = None
    window_complete: bool


class Comparable(Strict):
    player_id: Optional[str] = None
    name: str
    college: Optional[str] = None
    draft_year: int
    round: Optional[int] = None
    overall_pick: Optional[int] = None
    distance: float = Field(description='nan-Euclidean distance on pool z-scores; smaller = more similar; NOT a probability')
    closeness_index_heuristic: float = Field(description='0-100 rank heuristic: share of pool pairs farther apart; NOT a probability')
    most_similar_on: str
    biggest_differences: str
    nfl_outcome: ComparableOutcome


class ComparisonsResponse(Strict):
    player_id: str
    status: Status
    reason: Optional[str] = None
    kind: Optional[str] = None
    model_version: Optional[str] = None
    note: Optional[str] = None
    comparables: list[Comparable] = Field(default_factory=list)


# ------------------------------------------------------------------ backtests
class BacktestStat(Strict):
    stat: str
    predicted: float
    interval_80: Interval
    actual: float
    actual_label: str
    error: Optional[float] = None
    inside_80: Optional[bool] = None
    model: str


class BacktestStartProbability(Strict):
    threshold: int
    predicted: float = Field(ge=0, le=1)
    actual_event: Optional[bool] = None


class BacktestView(Strict):
    source: str
    statistics: list[BacktestStat]
    start_probabilities: list[BacktestStartProbability]


class BacktestClassification(Strict):
    status: Literal['valid_out_of_sample_backtest', 'out_of_sample_forecast_outcome_incomplete',
                    'retrospective_illustration_in_sample']
    explanation: str
    in_sample: bool


class BacktestResponse(Strict):
    player_id: str
    player: dict[str, Any]
    classification: BacktestClassification
    prediction_cutoff: str
    training_classes: Optional[list[int]] = None
    n_training_qbs: Optional[int] = None
    label_embargo_years: int
    model_version: str
    actual_window_complete: bool
    observed_seasons: int
    horizons: dict[str, dict[str, str]]
    views: dict[str, BacktestView]
