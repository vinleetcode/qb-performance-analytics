# API endpoint reference (Phase 6A)

Base URL in development: `http://localhost:8000`. Every endpoint is a read-only `GET`. Interactive documentation is at
`/docs` and the machine-readable schema at `/openapi.json` (both can be hidden with `QB_API_DOCS_ENABLED=false`).

## Conventions

* **Player ids** are stable strings. `qb_<n>` is a drafted QB (n = `qb_draft_prospect_id`). `cfb_<cfbd id>_<final season>` is
  a college-only prospect, identified by the CFBD athlete id and his final college season. Names are not unique, so
  every request after search uses the id.
* **Status.** Every prediction block has `status: "available" | "unavailable"`. An unavailable block has a `reason` (and
  usually a `code`) and contains no numbers. The API never fills a gap with zeros or invented probabilities.
* **Labels.** Observed values (actual draft result, NFL outcomes) are separate from predicted values. `pre_draft`
  forecasts never show the actual pick.
* **Intervals** are lists of `{level, lower, upper}` objects with levels 0.5 / 0.8 / 0.9. Their measured backtest coverage
  is returned next to them.
* **Numbers** are plain JSON numbers. NaN and infinity become `null`.
* **Errors** always look like this:

```json
{"error": {"code": "PROSPECT_NOT_FOUND", "message": "No prospect with id qb_999999.", "details": {}}}
```

| Code | HTTP | When |
|---|---:|---|
| `PROSPECT_NOT_FOUND` | 404 | unknown player id, or `/resolve` found no match |
| `AMBIGUOUS_PLAYER` | 409 | `/resolve` matched several prospects; `details.candidates` lists them |
| `INVALID_PREDICTION_MODE` | 422 | `mode` is not `pre_draft` or `post_draft` |
| `INVALID_PARAMETER` | 422 (405 for non-GET) | malformed id, out-of-range or disallowed parameter |
| `PREDICTION_UNSUPPORTED` | 422 | e.g. a backtest for a QB with no NFL draft record |
| `INSUFFICIENT_DATA` | 422 | (inside blocks) too many model inputs missing |
| `MODEL_UNAVAILABLE` | 503 | serving bundle missing or failed SHA-256 verification |
| `RATE_LIMITED` | 429 | more than `RATE_LIMIT_PER_MINUTE` requests per minute from one address |
| `NOT_FOUND` | 404 | no such endpoint |
| `INTERNAL_ERROR` | 500 | unexpected failure (details stay in the server log) |

Within a successful forecast, unsupported parts are `status: "unavailable"` blocks with codes
(`PREDICTION_UNSUPPORTED`, `INSUFFICIENT_DATA`), so one missing piece never fails the whole page.

## `GET /api/health`

Cheap: never loads the forecasting engine. Returns `status` (`ok` / `degraded`), `api_version`, `data_source`,
`engine_loaded`, `bundle_version` and `models[]` (`name`, `version`, `available`, `verified`: SHA-256 against metadata,
computed once per process). No paths or configuration are returned.

## `GET /api/prospects`

| Parameter | Type | Notes |
|---|---|---|
| `q` | string, 1-80 chars | case- and accent-insensitive; every word must appear (`"mendo"`, `"jax dart"`) |
| `draft_year` | int 1990-2035 | optional filter |
| `drafted_only` | bool | default false |
| `limit` | int 1-200 | capped at `QB_API_MAX_PAGE_SIZE` (default 50) |
| `offset` | int >= 0 | pagination |

Response: `{query, draft_year, total, limit, offset, items[]}`. Each item has `player_id`, `name`, `college`,
`draft_year`, `final_college_season`, `drafted` (`null` = not yet verifiable), `draft_round`, `overall_pick`, `kind`
(`drafted` / `college_only`) and `shares_name_with_other_prospects`. Ordering: exact name, then name prefix, then other
matches; newest class first within each.

## `GET /api/prospects/resolve?name=&draft_year=`

Exact (normalized) name lookup for deep links. Returns one `ProspectSummary`, or `AMBIGUOUS_PLAYER` (409) with
`details.candidates`, or `PROSPECT_NOT_FOUND`.

## `GET /api/prospects/{player_id}`

* `prospect`: the summary above.
* `identifiers`: `qb_draft_prospect_id`, `cfbd_player_id`, `gsis_id`.
* `draft_status`: observed result, or a note when the final season is not verifiable yet.
* `final_season_statistics`: completion %, YPA, TD %, INT %, attempts per game, opponent-adjusted YPA, schedule
  strength, rushing yards per game (last three seasons), seasons with 100+ attempts, YPA change.
* `college_seasons[]`: every CFBD passer-season up to the final one.
* `offensive_profile`: team pass rate, plays per game, SP+ offense, team recruiting percentile, conference, Power-5
  flag, reported spread-family label (where researched).
* `athletic_measurements`: `{value, source}` for roster and Combine height/weight and Combine forty. Missing is `null`,
  never imputed.
* `combine_participant`.
* `data_coverage`: which model inputs are present or missing, and the draft-population status.

## `GET /api/prospects/{player_id}/forecast?mode=pre_draft|post_draft[&assumed_pick=]`

| Block | Contents |
|---|---|
| `player` | identity; `actual_draft` only in `post_draft` mode |
| `draft.probability` | `pre_combine` and `post_combine` P(drafted), only inside the verified population; otherwise `unavailable` with the population status |
| `draft.round` | `groups` (Round 1 / Rounds 2-3 / Rounds 4-7: show these), `rounds` 1-7, `expected_round`; conditional on being drafted |
| `draft.pick` | `median` and 50/80/90% `intervals`, with measured coverage |
| `draft.eligibility` | population status and classification (`eligible_career_ended` / `eligibility_unknown` / ...) |
| `career.start_probabilities` | `p_16_plus` >= `p_32_plus` >= `p_64_plus` (enforced and re-checked by the API), `p_any_nfl_game`, warnings |
| `career.five_year_totals[]` | games played/started, pass attempts, completions, yards, TDs, INTs, rush attempts, yards, TDs: `expected`, 50/80/90% `intervals`, model, interval method, measured coverage, training-maximum flag |
| `career.year_by_year` | seasons 1-5, each statistic with an 80% range; sums equal the five-year totals |
| `career.reference_v2` | P(32+ starts) from the v2 career-success model, for reference |
| `warnings[]` | `{code, message}`: `DRAFT_PROBABILITY_UNAVAILABLE`, `COMBINE_PARTICIPATION`, `CALIBRATION_WARNING`, `EXTRAPOLATION`, `HYPOTHETICAL_PICK`, `ASSUMPTION` |
| `unavailable_blocks[]` | e.g. `draft.probability`, `career` |
| `uncertainty` | model versions, measured calibration, training coverage, data limitations |

`post_draft` needs the actual pick. Otherwise pass `assumed_pick` (1-262), which marks the career block
`hypothetical: true`. In `pre_draft` mode `assumed_pick` is rejected. Forecasts are deterministic (fixed simulation
seed) and cached per (player, mode, pick, artifact version).

## `GET /api/prospects/{player_id}/comparisons`

Five nearest earlier drafted QBs by college profile: `player_id`, `name`, `college`, `draft_year`, `round`,
`overall_pick`, `distance` (smaller = more similar), `closeness_index_heuristic` (0-100 rank; **not a probability**),
`most_similar_on`, `biggest_differences`, `nfl_outcome` (observed starts in seasons 1-5 if complete, else to date).
`kind` is always `explanatory`.

## `GET /api/backtests/{player_id}`

Drafted QBs only (`PREDICTION_UNSUPPORTED` otherwise). `classification.status`:

* `valid_out_of_sample_backtest`: classes 2012-2021. Models were trained only on classes whose outcomes were known
  before that draft, and the QB's window is complete.
* `out_of_sample_forecast_outcome_incomplete`: classes 2022+. Out of sample, but actuals are to date and errors are `null`.
* `retrospective_illustration_in_sample`: classes before 2012. The final model was trained on this QB, so this is
  **not** a backtest (`in_sample: true`).

Also returned: `prediction_cutoff`, `training_classes`, `n_training_qbs`, `label_embargo_years`, `model_version`, and
`views.pre_draft|post_draft.statistics[]` (`predicted`, 80% interval, `actual`, `error`, `inside_80`) plus start
probabilities with the observed event. `horizons.one_year` and `horizons.three_year` are `unavailable`: only
five-season totals were forecast and backtested.

## `GET /api/models` and `GET /api/models/performance`

`/api/models`: served model versions, roles and integrity status. `/api/models/performance`: the curated
`api/static/model_performance.json`, built by `python -m api.performance` from existing evaluation outputs. It covers
draft probability (pre/post-Combine and the participation-only variant), round and pick metrics against historical
frequencies, start-probability Brier/AUC/calibration against base rates, production MAE (shipped family, nested
selection, historical median), interval coverage, an `evaluation_status` per section saying how far each result is from
an untouched test, and the limitations list.
