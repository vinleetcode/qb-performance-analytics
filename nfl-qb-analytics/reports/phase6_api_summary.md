# Phase 6A: FastAPI backend summary

A read-only FastAPI service now exposes the existing prediction system. It serves the same numbers as the Python
engine (verified to full floating-point equality), from tracked model files plus a 4.4 MB serving bundle. It needs no
database, no training data and no CFBD key at runtime. Nothing was deployed, committed or pushed, and no model was
retrained or changed.

Companion documents: [`api_endpoint_reference.md`](api_endpoint_reference.md) (every endpoint, parameter, schema
and error code) and [`deployment_readiness.md`](deployment_readiness.md) (artifacts, packaging, environment, checklist,
blockers).

## 1. Audit at the start

* Tests: `273 passed, 1 skipped` (the run overlapped with the first small source edits of this phase). The skip did
  not recur in the final run. It is consistent with one of the two pre-existing tests that query the live nflverse
  source and skip when it is unreachable; the final run did not confirm which test it was.
* `python -m src.modeling.artifacts --check`: 7 external data files and 11 model files, 0 problems.
* 29 model files hashed (`results/phase6/baseline_model_hashes.txt`). All are byte-identical at the end
  (`final_model_hashes.txt`, `diff` empty).
* Git: branch `main` at `eef9d98`. Phase 4-5.5 work still uncommitted. `../.claude/` was untracked and not ignored.

## 2. What was built

```text
nfl-qb-analytics/
├── api/
│   ├── main.py                 app factory: CORS allowlist, rate limiter, error handlers, request logging
│   ├── config.py               environment settings
│   ├── errors.py               stable error codes -> HTTP status
│   ├── schemas.py              Pydantic response models (extra fields forbidden)
│   ├── dependencies.py         one service per process
│   ├── bundle.py               serving bundle: build / verify / pack / install, BundleSource
│   ├── performance.py          builds api/static/model_performance.json from existing evaluation outputs
│   ├── static/model_performance.json   aggregate metrics only (tracked)
│   ├── routes/                 health, prospects, forecasts, comparisons, backtests, models
│   └── services/               prediction_service (the only code that touches the engine), model_service
├── requirements-api.txt  Dockerfile  .dockerignore
└── tests/test_api.py           30 tests
```

**No prediction logic was duplicated.** The service calls `prediction_api.Engine.forecast` (the function behind
`forecast_prospect`), `Engine.comps_block` and the `historical_backtest` outputs. Three small, behaviour-preserving changes
were made to the modeling code:

1. `prediction_api.Engine(source=None, combine_seasons=None, hist_max=None)`: the data source can be injected. The
   default is unchanged.
2. Comparison entries now also carry the comp's `qb_draft_prospect_id`, so the API can link to them.
3. `historical_backtest` fits each outer fold once per draft class (`lru_cache`) instead of once per QB. The output is
   identical; the existing recompute-vs-stored test still passes.

## 3. Endpoints (all GET)

| Endpoint | Notes |
|---|---|
| `/api/health` | `ok` / `degraded`, model integrity, bundle version; never loads the engine |
| `/api/prospects` | search: case/accent-insensitive, all words must match, ranked exact > prefix > other; `draft_year`, `drafted_only`, `limit` (capped at 50), `offset` |
| `/api/prospects/resolve` | exact name; `409 AMBIGUOUS_PLAYER` with candidates when names collide |
| `/api/prospects/{player_id}` | final-season statistics, every college season, offensive profile, measurements with source, Combine participation, data coverage |
| `/api/prospects/{player_id}/forecast?mode=` | draft probability (pre/post-Combine, verified population only), round groups and rounds, pick range, nested start probabilities, ten five-season statistics with 50/80/90% intervals and measured coverage, year-by-year paths, warnings, uncertainty |
| `/api/prospects/{player_id}/comparisons` | five explanatory comps with distances, closeness heuristic, similarities, differences, observed outcomes |
| `/api/backtests/{player_id}` | three classifications (valid out-of-sample / incomplete window / in-sample illustration); five-year view only |
| `/api/models`, `/api/models/performance` | versions and integrity; curated evaluation, baselines, evaluation status, limitations |

Player ids: `qb_<n>` for drafted QBs, `cfb_<cfbd id>_<final season>` for college-only QBs. The bundle has 1,279
prospects: 328 drafted (2000-2026 classes) and 951 college-only final seasons from the draft universe.

## 4. Statistical safeguards kept in the API

* **Pre/post-draft separation.** In `pre_draft` mode `player.actual_draft` is `null` and the pick is never used.
  `assumed_pick` is rejected in `pre_draft` and labelled `hypothetical` in `post_draft`.
* **Nesting.** P(64+) <= P(32+) <= P(16+) comes from the v1.1.0 projection and is re-checked in the service before a
  response is sent.
* **No fabricated values.** Unsupported parts are `status: unavailable` with a reason and a code: P(drafted) outside the
  verified population, `post_draft` without a pick, college-only backtests. Missing measurements are `null`.
* **Warnings** (`code`, `message`): `DRAFT_PROBABILITY_UNAVAILABLE`, `COMBINE_PARTICIPATION` (post-Combine probability
  reflects invitations), `CALIBRATION_WARNING` (P(64+) pre-draft), `EXTRAPOLATION` (80% upper bound beyond the training
  maximum; one consolidated warning), `HYPOTHETICAL_PICK`, `ASSUMPTION`.
* **Comparisons** are `kind: explanatory`. Distances and the closeness index are documented as not probabilities, and no
  field is called a probability.
* **Backtests** never label in-sample results as out-of-sample. One- and three-season views are `unavailable` because
  only five-season totals were forecast and backtested.
* **Performance endpoint.** Numbers are copied from existing evaluation outputs. Each section has an
  `evaluation_status` saying how far it is from an untouched test (for example: draft feature sets compared on the
  same test drafts; production selection rule revised once; pre-draft interval method is a reviewed override).

## 5. Model versions served

| Output | Model | Files |
|---|---|---|
| P(drafted), pre/post-Combine | qb_draft v1.0.0 | `prob_college_l2.joblib`, `prob_college_combine_l2.joblib` |
| Round groups, rounds, pick range | qb_draft v1.1.0 | `round_pick.joblib` |
| Start probabilities, five-season totals, year-by-year | qb_career_production v1.1.0 | `pre_draft.joblib`, `post_draft.joblib` + `reference_tables.npz` (bundle) |
| Comparisons | qb_similarity v1.0.0 | `scaler.joblib` + `pool.parquet` (bundle) |
| Reference P(32+) | qb_career_success v2.0.0 | `historical_selected.joblib` |

Not served: qb_career_success v1.0.0, qb_career_production v1.0.0 (gitignored, embeds player-level data).

## 6. Loading, caching and performance

* Startup imports only the app. `/api/health` hashes the served model files once per process and never loads the engine.
* The first data request loads the prospect index and verifies all 21 bundle and model files against the manifest
  (about 4 s). The first forecast loads the engine (about 6-10 s). Both happen once per process, under a lock.
* Forecasts are deterministic (fixed simulation seed) and cached (LRU, 512 entries). The cache key includes the bundle
  build and every model hash, so new artifacts never serve stale results. Measured warm latency: 0.2 s cached, 0.3-0.5 s
  uncached forecast, about 0.05 s comparisons and backtests (in process).
* Missing or tampered artifacts: `503 MODEL_UNAVAILABLE` and health `degraded`. Nothing is served from a bundle whose
  hashes do not match.

## 7. Security

* Every endpoint is a GET (`405` otherwise). There is no endpoint that runs code, reads files, loads models or queries a
  database, and the serving path contains no SQL.
* CORS: explicit allowlist from `FRONTEND_ORIGINS`; `*` is ignored; `allow_credentials=False`; GET only.
* Rate limit: 120 requests per minute per address on `/api/*` (health exempt), configurable.
* Validation: player-id pattern, bounded query lengths and page sizes, enumerated `mode`, pick range.
* Errors return `{"error": {"code", "message", "details"}}`. Unhandled exceptions return a generic `INTERNAL_ERROR`
  (tested with an exception message containing a path). Responses add `X-Request-ID` and `X-Content-Type-Options`.
* Secrets: the API never reads `CFBD_API_KEY` (shared modules load the key only when a CFBD request is made, and the
  API makes none). Tests assert that no response body, no bundle file and no committable file contains a configured
  secret, and that inference opens no non-loopback socket.
* Git: `.claude/` is now ignored by a new repository-root `.gitignore`. The project `.gitignore` adds `dist/`,
  `*.py[cod]`, `venv/`, `.streamlit/secrets.toml` and `.claude/`. Only code, reports, tracked models and the aggregate
  performance JSON would be added. No database, parquet, npz, `data/`, `results/`, `dist/` or `.env` file would be added
  (checked with `git ls-files --others --exclude-standard`).
* Bundle installation refuses archive members with absolute paths, `..`, links or unexpected prefixes (tested).

## 8. Tests

* **Final run: `304 passed` in 294 s.** That is the 274 Phase-5.5 tests plus 30 new API tests; none skipped or failed.
* `tests/test_api.py` covers: startup and docs; health without engine load and without paths; search, autocomplete and
  pagination bounds; duplicate names and 409 ambiguity; prospect details; pre/post-draft separation; start-probability
  ordering and interval nesting; unavailable statuses; API output equal to the direct Python engine (to 1e-12);
  determinism; error codes for invalid mode, pick, id and endpoint; 405 for POST; comparisons (explanatory, earlier
  classes only); all three backtest classifications and the unsupported case; the performance response; CORS allowlist
  and wildcard rejection; rate limiting; hidden internal errors; missing and tampered bundles (503); bundle and model
  verification; unsafe archive rejection; no network calls; no secrets; gitignore coverage.
* Pre-existing note: two older pipeline tests (`test_historical_pipeline_integration.py`, `test_prospect_pipeline.py`)
  query the live nflverse source when run and skip if it is unreachable. The API tests make no network requests.

## 9. Live verification

* `uvicorn api.main:app --port 8000` started from the repository; `/api/health` returned `ok` with all five models
  verified, and `/docs` and `/openapi.json` returned 200. Every endpoint returned the expected status for successful and
  unsupported requests (examples in `api_endpoint_reference.md`).
* **Clean-deployment run.** The exact Dockerfile file set (6.0 MB: code, five model directories, bundle, two model data
  files) was copied to an empty directory with no `.env`, no database and no results, and started with `CFBD_API_KEY`
  removed from the environment. All nine endpoint calls returned 200 and the server log had no errors. Forecasts (pre-
  and post-draft), a college-only forecast, comparisons and a backtest were byte-identical to the repository's
  responses.
* `python -m api.bundle pack` produced a 1.4 MB archive; `install` into an empty directory and `verify` reported no
  problems.

## 10. Limitations

* The prospect set is fixed at bundle build time: 1,279 prospects. New prospects need fresh data, then a bundle
  rebuild. The API cannot score a player who is not in the bundle (`PROSPECT_NOT_FOUND`).
* Prediction quality is unchanged from Phase 5.5. Pre-draft five-season totals are only modestly better than the
  historical median, rushing and pick intervals under-cover, and P(64+) pre-draft is poorly calibrated. The API surfaces
  these as warnings and in `/api/models/performance`.
* The cache and rate limiter are per process.
* FastAPI 0.115 / Starlette 0.45 emit deprecation warnings under Python 3.14 (third-party; harmless now).

## 11. Deployment blockers

1. The Docker image has not been built (Docker is not installed here). Its file set was verified with the clean-deployment run.
2. Python 3.14 plus the pinned scikit-learn/NumPy/SciPy versions are required to load the pickled models.
3. Data licences (CFBD, nflverse) for public display of player-level values need review before a public launch.
4. Use platform-level rate limiting if running several workers or instances.

## 12. Recommended frontend integration (Next.js)

* Call the API from Next.js server components or route handlers (`fetch` with `revalidate` of a few minutes). Results
  are deterministic, so they can also be pre-rendered for listed prospects.
* Search box: `GET /api/prospects?q=` (debounced, `limit` 10) -> navigate by `player_id`. Use `/resolve` only for
  name-based deep links, and show the candidate list on 409.
* Prospect page:
  * details from `/api/prospects/{id}`;
  * forecast with a pre/post-draft toggle;
  * comparisons and backtest as separate panels, so one slow or unavailable block does not hold up the page.
* Render every block from its `status`. Show `reason` text for unavailable blocks, show `warnings[]` near the numbers
  they qualify, and never show a number without its interval.
* Display rules:
  * show round **groups** rather than single rounds;
  * show the closeness index as a rank, never as a percentage chance;
  * label post-Combine draft probability as participation-driven;
  * label backtests with their classification badge;
  * hide error and inside-interval markers for incomplete windows.
* Generate TypeScript types from `/openapi.json` (for example with `openapi-typescript`) so the frontend tracks the
  schema.

## 13. Reproduction

```bash
cd nfl-qb-analytics
python -m pytest -q                              # 304 passed
python -m src.modeling.artifacts --check         # 7 data files, 11 model files, 0 problems
python -m api.bundle build                       # about 30 min on 16 cores (precomputes 328 backtests)
python -m api.bundle verify                      # 21 files, 0 problems
python -m api.performance                        # refresh the curated performance JSON
uvicorn api.main:app --reload --port 8000
curl "http://localhost:8000/api/prospects?q=mendoza"
curl "http://localhost:8000/api/prospects/qb_319/forecast?mode=pre_draft"
python -m api.bundle pack                        # dist/qb_serving_bundle_v1.tar.gz for deployment
```
