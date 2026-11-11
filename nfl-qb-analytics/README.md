# NFL Quarterback Analytics — Portfolio Project

An end-to-end, statistically grounded NFL player analytics starter. Designed for data science and applied statistics internship portfolios.

**Project goal (Phases 2-3):** a reproducible, leakage-audited CFB-to-NFL quarterback dataset for
predicting the NFL success of drafted college QBs from college performance, opponent-adjusted
efficiency, development trajectory, age, offensive environment, supporting cast, team talent and
the NFL Combine, with NFL career targets, survival records, prospect comparables and draft-value
structures. The phases build on each other, and none of them modifies the data of an earlier phase.

| Phase | What it adds | Entry point |
|---|---|---|
| MVP | 2000-2025 QB seasons, next-season YPA baseline model | `python -m src.pipeline`, `python -m src.model` |
| 1 | All-time NFL QB database (1999-2025 ingested; earlier gaps documented) | `python -m src.historical_pipeline` |
| 2 | NFL draft board + CFBD college stats + identity matching | `python -m src.prospect_pipeline` |
| 3 | Advanced college features, environment, combine, NFL targets, survival, similarity, draft-value prep | `python -m src.advanced_pipeline` |
| Model v1 | Career-success model (32+ starts in 5 seasons), basic vs context vs play-type comparison | `python -m src.modeling.train` |
| Model v2 (Phase 4) | Era normalization study + historical (2005-2021) training, label-availability embargo | `python -m src.modeling.train_v2` |
| Phase 5 | Draft projection, five-season stat forecasts, prospect comps, career trajectories, integrated forecast | `python -m src.modeling.forecast` |

Phase 3 coverage, validation and limitations: **[`reports/advanced_features_coverage.md`](reports/advanced_features_coverage.md)**.

## Scope (MVP)
1. Fetch **completed regular seasons 2000–2025** using `nflreadpy`.
2. Calculate passing efficiency (`YPA`, touchdown rate, interception rate) and within-season, attempt-weighted z-scores.
3. Construct quarterback-season rows and predict **next-season yards per attempt** using a regularized linear model.
4. Evaluate against mean and persistence baselines on a chronologically held-out **2024–2025** test set.

## Run
Requires Python 3.10+ and internet access for the data-fetch step.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline
python -m src.model
pytest -q
```

The data pipeline writes `data/qb_seasons.csv`. The model writes `results/metrics.json`.

## Interpretation / limitations
- `era_score` is a simple exploratory composite, **not** an established measure of individual QB talent. It equal-weights correlated metrics.
- Normalization includes all QBs with >=50 pass attempts per season, weighted by attempts; thresholds and weights should be sensitivity-tested. This is not official PFR index methodology.
- The first version is **not** a historical all-eras model: 2000–2025 is the modern initial scope.
- The model predicts next-season **YPA among QBs with >=100 next-season attempts**, and therefore does not predict who will remain active or be given playing time (selection bias).
- Results must be computed on a real machine with network access; **no results or performance claims are included in this starter**.
- No future season's data is used as predictor input. However, annual standardization uses full current-season statistics; this is appropriate for predicting the following season but not mid-season forecasting.

## Roadmap to a standout resume project
- Stage 2: SQL/DuckDB warehouse; automated data-quality checks; repeatable dataset refreshes.
- Stage 3: time-aware cross-validation; grouped player bootstrap intervals; calibration; additional model (gradient boosting) and contextual features.
- Stage 4: Streamlit dashboard with player comparison and clearly labeled uncertainty.
- Stage 5: report exact holdout metrics, baseline improvements (if any), error breakdowns and technical decisions in README.

## Data source
[nflreadpy docs](https://nflreadpy.nflverse.com/api/load_functions/) / nflverse. Respect upstream data licensing and attribution.

---

## Phase 1 update: all-time (1920-2025) historical dataset

A second, independent pipeline (`src/historical_pipeline.py`) extends the project toward a
full-history QB dataset, with its own database schema, qualification logic, and
era-adjustment framework. It does **not** modify the MVP pipeline/model above (only
`START` in `pipeline.py` moved from 2000 to 1999, since that is nflreadpy's actual floor).

### What was verified before writing any ingestion code
- Called `nflreadpy.load_player_stats(seasons=True)` directly: data starts at **1999**, not
  earlier. Also checked `load_schedules()` (1999+), `load_draft_picks()` (1980+, picks only,
  no stats), and `load_combine()` (2000+) -- nflreadpy has no pre-1999 passing statistics.
- Fetched Pro Football Reference's own data-use policy
  (https://www.sports-reference.com/data_use.html). It explicitly prohibits automated/bulk
  access without written permission, prohibits building a database that substitutes for
  their own data store, and states custom data requests require a **minimum $5,000**
  payment -- there is no free bulk/API path.
- **Result: 1920-1998 NFL/APFA and 1960-1969 AFL passing data could not be obtained through
  any legitimately licensed, free, bulk-accessible source found in this environment.** No
  scraping was attempted and no third-party re-hosted dataset of unverified license was
  substituted. This is recorded, with reasons, in `config/sources.json` and surfaced by the
  coverage report rather than silently assumed away.
- 1999-2025 nflverse data **was** ingested for real (see results below).

### What's in the database (`src/db/schema.py`, SQLite by default; set `DATABASE_URL` for Postgres)
`sources`, `leagues` (APFA/NFL/AFL kept distinct, with merger notes), `teams`,
`players` (canonical identity, keyed by source player id -- never by name, so same-name
players never collide), `player_seasons_raw` (as-sourced, nulls preserved,
`position_listed`/`is_qb_listed` kept separate so non-QB passers aren't miscast),
`player_season_team_stints` (per-team breakdown so trades don't double-count season
totals), `player_seasons_derived` (computed efficiency metrics, kept separate from raw),
`era_adjusted_stats` (z-score/percentile per metric per baseline method),
`season_calendar` (schedule-length reference metadata, not stats, used to scale
qualification thresholds), and `coverage_report`.

### Metrics (`src/metrics.py`) and era adjustment (`src/era_adjust.py`)
Passing: completion %, Y/A, TD%, INT%, AY/A, NFL passer rating, and ANY/A (ANY/A returns
null whenever sack data is missing rather than assuming zero sacks). Rushing (added
alongside passing, since nflverse's `player_stats` already carries carries/rushing
yards/TDs per player-season): yards/carry, rushing TD%, plus combined `total_yards`,
`total_tds`, and `yards_per_play` ((pass yards + rush yards) / (attempts + carries)) for
comparing dual-threat value. Z-scores/percentiles are computed two ways and stored side by
side for all 10 metrics: `season_league` (weighted, same season+league only -- AFL and NFL
are never pooled; passing metrics weighted by attempts, rushing metrics by carries,
`yards_per_play` by total plays) and `rolling_5yr` (trailing window, still split by
league), per the instruction to test an alternative rolling-era baseline rather than ship
only one fixed choice. Rushing-metric eligibility still gates on the *passing* qualification
threshold (keeps the comparison population "real" QB seasons rather than single-carry
trick plays) -- documented here as a deliberate simplification, not an oversight.

### Configurable qualification (`src/qualification.py`, `config/qualification_thresholds.json`, `config/season_calendar.json`)
Default rule is the NFL's own passing-title standard (14 attempts/scheduled game),
scaled by each season's game count -- e.g. 126 attempts for the 9-game 1982 strike season
vs. 238 for a 17-game season -- instead of one fixed number. All players are kept in the
dataset regardless of whether they qualify; the flag only controls leaderboard/baseline
inclusion. `season_calendar.json` itself is schedule-length metadata (a public structural
fact), not player statistics; bands before 1961 are marked `approximate_unverified`.

### Run it
```bash
python -m src.historical_pipeline      # ingest, compute, write DB + data/*.parquet + results/coverage_report.json
pytest -q                              # includes an integration test against the live source
```

### Actual results from the last real run (2026-10-28, against live nflreadpy)
- **27 seasons ingested** (1999-2025, NFL). **79 seasons not ingested** (1920-1998 NFL/APFA)
  **+ 10 seasons not ingested** (1960-1969 AFL) -- tracked as distinct rows in
  `coverage_report`, never fabricated.
- **820 unique players**, **2,745 player-season observations**, **2,799 team-stint rows**
  (multi-team seasons correctly split without double-counting -- e.g. Baker Mayfield 2022:
  206 CAR attempts + 129 LA attempts = 335, matching the single aggregated season total).
- Table row counts: `sources` 3, `leagues` 3, `teams` 36, `players` 820,
  `player_seasons_raw` 2,745, `player_season_team_stints` 2,799, `player_seasons_derived`
  2,745, `era_adjusted_stats` 54,900 (2,745 rows x 10 metrics [7 passing + 3 rushing/combined]
  x 2 baseline methods), `season_calendar` 116, `coverage_report` 116.
- Decade x league breakdown, missing-stat enumeration, and exact not-ingested ranges with
  reasons: see `results/coverage_report.json` (regenerated every run).
- Sanity check: 2024 passer-rating leaderboard computed by this pipeline is topped by Lamar
  Jackson (119.6), matching the real 2024 result -- used as a correctness check on the
  passer-rating formula, not a claim about model quality.
- Tests: **24/24 passed** (`pytest -q`), including live integration tests (passer-rating
  correctness, rushing-stat ingestion, idempotency) and a regression test that a
  non-default (e.g. test) run never overwrites the real project's Parquet exports.
- Data-source discrepancies/reconciliation: none to reconcile yet -- 1999-2025 has exactly
  one ingested source. This becomes relevant once a second historical source is added.

### Candidate source evaluated and rejected: the Kaggle/GitHub "nfl-player-stats" dump
On 2026-10-29, `zackthoutt/nfl-player-stats` (mirrored on Kaggle as
`zynicide/nfl-football-player-stats`) was evaluated as a possible fill for the 1920-1998
gap. Findings:
- **Provenance**: the repo's own README states, verbatim, "I decided to scrape
  `pro-football-reference.com` for all recorded NFL player data... I scraped every NFL
  player in their database going back to the 1940s" (scraped 12/1/17-12/4/17). This is the
  same publisher whose data_use.html (see above) explicitly prohibits automated bulk
  collection and building a substitute data store without a paid license.
- **License**: the GitHub repo is MIT-licensed (covers the *scraper code*, not the data),
  and the uploader separately marked the Kaggle mirror "CC0: Public Domain" (confirmed via
  the page's embedded schema.org metadata). A third party's CC0 self-declaration on data it
  scraped from a site whose terms forbid that scraping doesn't grant rights it doesn't hold,
  and isn't Sports Reference's own permission.
- **Decision: rejected, not ingested.** Using it would mean reconstructing the same
  restricted PFR bulk dataset at one remove -- exactly what this project's access policy
  rules out, whether the scraping is done directly or via a third-party mirror. It was
  **not** downloaded (no Kaggle API credentials were available in this environment, and the
  provenance finding means the payload wouldn't be ingested regardless), so schema/row-level
  completeness was assessed from the repo's documented schema and completeness notes only,
  not a hands-on audit. Recorded as `zackthoutt_pfr_scrape_kaggle` in `config/sources.json`
  with `status: not_ingested` and full reasoning; `tests/test_source_registry.py` asserts
  this decision so it can't silently flip. The coverage report logs its claimed 1940-2017
  range clipped to 1940-1998 (the part that's actually a gap -- its 1999-2017 overlap with
  the already-ingested nflverse range is not re-flagged as missing).
- Even on technical grounds alone, this candidate would not have been sufficient: it claims
  coverage only from "the 1940s" onward, so 1920-1939 (the APFA/earliest-NFL years) would
  have stayed a gap regardless.

### Known gaps and how to close them later
1920-1998 and 1960-1969 AFL are schema-ready but empty, by design, because no legitimate
bulk-access path was found (the Sports Reference $5,000+ custom-data minimum remains the
only concrete licensed path identified; the Kaggle candidate above was evaluated and
rejected on provenance grounds). If a licensed export is obtained later, add
`src/ingest_historical_pfr.py` targeting the same raw-row shape as `src/ingest_nflverse.py`,
register it as its own `source_id` in `config/sources.json`, and never merge its `league_id`
values into the existing NFL rows for the same years (the AFL in particular must stay its
own `league_id` through 1969). Other possible legitimate leads, not yet verified: official
NFL/team "gamebook" archives (PFR's own underlying primary source, but not available as a
bulk digital download), Wikipedia/Wikidata (CC BY-SA, but only season-leader tables, not
complete player-season logs -- a possible source for spot-checking, not bulk ingestion), and
commercial sports-data vendors (Sportradar, Stats Perform, Armchair Analysis) under a paid
license, none of which have been contacted or verified in this environment.

### Exploratory data analysis (`src/eda.py`)
Run after the pipeline (`python -m src.eda`); reads only the Parquet exports (so it also
smoke-tests that those exports are self-sufficient) and writes `results/eda_summary.json`
plus charts to `results/eda/`. Findings from the real run:
- **Raw mean passer rating among qualifying QBs rose from ~78 (1999) to ~94 (2025)** --
  the well-known "era inflation" in modern passing efficiency.
- **Mean season-relative passer-rating z-score stays within +/-0.2 of zero every single
  season**, by construction -- this is the concrete demonstration that era-adjustment is
  doing its job: a QB's z-score is comparable across 1999 and 2025 even though the raw
  rating numbers are not.
- **Not every retained passer is a QB**: e.g. a 2003 punter's fake-punt attempt (1 attempt,
  81 yards) and a 2001 punter's intercepted/negative attempt (-11 yards) are both present
  and flagged `is_qb_listed=False`, confirming the "don't assume every passer was an
  official QB" rule is actually doing something, not just declared.
- Missingness on the ingested range: sack data (`sacks_suffered`/`sack_yards_lost`,
  needed for ANY/A) is **0% missing** for 1999-2025 -- the gap is entirely the undetermined
  1920-1998/AFL seasons, not holes inside the ingested range.
- Rushing: mean rushing yards among qualifying QBs roughly **doubled from ~130
  (1999) to a 2022 peak of ~260**, visibly tracking the league's shift toward dual-threat
  quarterbacks (`mean_rushing_yards_by_season.png`).

### Explicitly out of scope for this phase
No frontend, no predictive model work beyond the pre-existing MVP `model.py`, and no
composite/aggregate QB rating -- per this phase's instructions, those come later.

---

## Phase 2: CFB-to-NFL draft prospect dataset

Extends the project toward its stated goal -- predicting NFL outcomes for drafted college
QBs -- by adding an NFL Draft board, CollegeFootballData (CFBD) college stats, and a
reproducible identity mapping between the two, **without touching any existing NFL table**.

### Security setup (do this once)
1. Get a free key at https://collegefootballdata.com/key
2. `cp .env.example .env` and replace `PASTE_YOUR_API_KEY_HERE` with the real key.
3. Never commit `.env` (it's gitignored; verified below) and never paste the key into chat,
   logs, or code.

`src/config.py` loads `.env` (via `python-dotenv`) and exposes `get_cfbd_api_key()` /
`cfbd_api_key_configured()`; a missing or still-placeholder key raises a clear error that
never includes the key value. `src/cfbd_client.py` is the only thing that ever sends the
key over the network (as an `Authorization: Bearer` header, from backend Python only) --
nothing in this repo has a frontend, so "never expose it to the frontend" is satisfied by
there not being one yet; this client is the seam that must stay server-side when one exists.

**Verified security checks (2026-10-30):**
- `git check-ignore .env` → exit 0 (ignored). `.gitignore` adds `.env`, `.env.*`,
  `!.env.example` while preserving every prior entry.
- `git ls-files | grep -i '\.env'` and a scan for credential/secret/key-shaped filenames →
  empty. Nothing credential-shaped is or was ever tracked.
- `git log --all --full-history -- "*.env"` → empty. No `.env` has ever entered git history,
  so there is nothing to purge and no key to rotate.
- `tests/test_config.py` and `tests/test_cfbd_client.py` assert the key never appears in an
  exception message or in an on-disk cache file.

### NFL Draft data (`src/ingest_draft.py`) -- works now, no key needed
Pulls `nflreadpy.load_draft_picks(seasons=2000..2026)` (all positions, kept as the
**original** record in `nfl_draft_picks_raw`) and `load_players()` for height/weight/
birthdate backfill by `gsis_id`. A **cleaned, QB-only** `qb_draft_prospects` table is built
separately, containing only pre-draft bio/identity fields -- draft-day info, not career
outcomes (see leakage section below). Schema was inspected empirically, not assumed:
`cfb_player_id` in nflverse's draft data turned out to be a sports-reference.com/cfb slug
(e.g. `"chad-pennington-1"`), **not** a CollegeFootballData id -- the two systems don't share
an id space, which is why matching (below) is name+school based rather than a direct join.

### College football data (`src/ingest_cfbd.py`) -- needs `CFBD_API_KEY`
Uses the real CFBD OpenAPI spec (fetched and inspected, not guessed) for three endpoints:
`/stats/player/season` (passing + rushing, long-format), `/roster` (bulk per-season
height/weight), `/ratings/sp` (team-season SP+ rating + strength of schedule). CFBD's own
`statType` values per category aren't documented in the spec, so `pivot_player_stats()`
pivots whatever comes back and reports any unrecognized `statType` strings in the coverage
report rather than silently dropping or guessing at them. Derived metrics (completion %,
Y/A, TD%, INT%, NCAA passer rating, Y/C) are computed in `src/metrics.py`, never taken
as a raw API field (CFBD doesn't appear to expose an efficiency-rating field directly).

**Rate-limit discipline**: CFBD's free tier is 1,000 calls/month. The full 2004-2025 backfill
is ~88 calls (22 seasons x 4 endpoint-calls), every response cached to disk
(`data/cfbd_cache/`, gitignored) so a rerun makes **zero** new calls, with a 1-second
throttle and bounded (3x) exponential-backoff retries on 429/5xx -- never an unbounded retry
loop, never multiple keys to dodge the limit.

### Matching (`src/match_prospects.py`)
Two independent links per prospect, both recorded in `college_nfl_player_map`, never guessed:
- **NFL side**: a direct, stable `gsis_id` join against the *existing* `players` table --
  no fuzzy logic needed or used, since nflverse's draft data already carries the same id.
- **CFB side**: name+school matching (CFBD's `playerId` can't be joined directly, per above).
  Name suffixes (Jr/Sr/II-V) and punctuation are normalized; school names are normalized
  (`"Ohio St."`→`"ohio state"`, `"Miami (FL)"`→`"miami"`); a `college` field listing multiple
  schools (nflverse already captures transfers this way, e.g. `"Wyoming; Reedley"`) is split
  and matched against either. An exact normalized name+school match within a 6-year
  draft-year window wins outright; otherwise a constrained fuzzy match (`difflib`,
  similarity >= 0.90) is accepted only if it's unambiguously the best candidate -- a tie
  within 0.05 of the next-best, or multiple exact matches, is recorded as `'ambiguous'`,
  never silently resolved. Unmatched (no candidate at all, or the season predates/exceeds
  CFBD coverage) is recorded as `'unmatched'`, not dropped.

### Database additions (new tables only -- see `src/db/schema.py`)
`cfb_players`, `cfb_player_seasons` (+ `cfb_player_seasons_derived`), `cfb_team_season_context`,
`nfl_draft_picks_raw` (original), `qb_draft_prospects` (cleaned), `college_nfl_player_map`.
All use the same idempotent delete-then-insert-per-batch pattern as the existing pipeline
(see `src/prospect_pipeline.py`), verified by `pytest`'s idempotent-rerun test. Existing NFL
tables (`players`, `player_seasons_raw`, etc.) are only ever *read* (for the gsis_id join),
confirmed by a dedicated test that seeds NFL data, runs the full prospect pipeline, and
asserts the NFL row counts and content are byte-for-byte unchanged afterward.

### Leakage-safe exports
`python -m src.prospect_pipeline` writes `data/qb_draft_prospects.parquet`,
`data/nfl_draft_picks_raw.parquet`, `data/cfb_player_seasons.parquet`,
`data/cfb_team_season_context.parquet`, `data/college_nfl_player_map.parquet`, and critically
**two separate** unified files:
- `prospects_pre_draft_features.parquet` -- bio, draft slot, matched college career stats,
  team context. No NFL career-outcome column ever enters this file; `export_parquet()`
  asserts this at write time (`NFL_OUTCOME_COLUMNS & set(pre_draft.columns) == set()`), and
  a test exercises that assertion.
- `prospects_nfl_outcomes.parquet` -- NFL career totals (games, career pass/rush stats,
  All-Pro/Pro Bowl counts, career AV) joined only by `qb_draft_prospect_id`, kept separate so
  it's structurally impossible to accidentally train a pre-draft model on a post-draft label
  by loading "the" prospect file.

### Actual results from the last real run (2026-10-30, live CFBD data)
A real `CFBD_API_KEY` was configured and validated (a live uncached call confirmed auth
before the full run). Full 2004-2025 backfill completed: **88 API calls total**, all cached
-- comfortably inside the 1,000/month free-tier limit, and a rerun makes zero new calls.
Seasons are fetched most-recent-first (not chronologically) so an interrupted run always
lands on the most immediately useful data first.
- **328 drafted QBs** (2000-2026 draft classes) from **6,901** total draft picks (all
  positions) in the raw board.
- **19,115 college players identified**, **40,672 college player-season records**, across
  **2,794** team-season context (SP+ rating/SOS) rows.
- **245 matched to an existing NFL player record by stable `gsis_id`**; **83 have zero
  tracked NFL passing stats** -- preserved as negative examples, not dropped (spot-checked:
  Tom Brady/pick 199/2000, Mahomes/pick 10/2017, Josh Allen/pick 7/2018 all linked correctly).
- **189 of 328 QB prospects matched to real college stats** (0 ambiguous -- every tie was
  either resolved unambiguously or left `unmatched`, never guessed). 139 unmatched; **130 of
  those are 2000-2009 draft classes whose college careers predate CFBD's 2004 coverage start**
  (structural, expected). **[Corrected in Phase 3:** that explanation is only right for the
  2000-2004 classes. The 2005-2009 misses were caused by CFBD's `/stats/player/season`
  being nearly empty for 2004-2008 seasons. Phase 3 recovers them from per-game box
  scores (see "Phase 3" below).**]** Spot-checked matches after fixing three real issues found during
  this run (below): Blake Bortles/UCF, David Fales/San José State, Jacoby Brissett &
  Ryan Finley/NC State, Matt Corral/Ole Miss -- all `exact_name_school`.
- Table row counts: `nfl_draft_picks_raw` 6,901, `qb_draft_prospects` 328, `cfb_players`
  19,115, `cfb_player_seasons` 40,672, `cfb_team_season_context` 2,794,
  `college_nfl_player_map` 328.
- `prospects_pre_draft_features.parquet`: 328 rows, 189 with real college career stats
  attached, **zero NFL-outcome columns** (verified by assertion + test).
- **Existing NFL data fully preserved** throughout every run: `players` 820,
  `player_seasons_raw` 2,745 -- identical to the pre-Phase-2 baseline.
- Tests: **67/67 passed**.

**Three real bugs found and fixed against live data** (all now covered by regression tests):
1. **Mid-season-transfer duplicate rows**: a handful of CFBD players appear under two teams
   in the same season (e.g. a grad transfer); the pivot now collapses these to one row per
   player-season (stats summed with `min_count=1` so an all-missing stat line stays null, not
   a fabricated 0), instead of violating the one-row-per-player-season DB constraint.
2. **Roster bio fan-out**: CFBD's `/roster` occasionally lists the same (season, team, name)
   twice; the merge now deduplicates first and uses `validate='many_to_one'` so a future
   regression fails loudly instead of silently duplicating rows.
3. **School-name/matching gaps**, confirmed against real data and now encoded as explicit,
   narrow fixes rather than loosened fuzzy thresholds: apostrophe stripping (`Hawai'i` vs
   `Hawaii`), diacritic stripping (`San José State` vs `San Jose St.`), and three confirmed
   whole-name aliases (`Mississippi`/`Ole Miss`, `North Carolina St.`/`NC State`,
   `Central Florida`/`UCF`). Combined effect: 180 -> 189 matched.
   (One dead end investigated and ruled out: a terminal-display artifact briefly looked like
   a UTF-8 corruption bug in "San José State"; checking the raw response and stored bytes
   directly showed the data was always correct UTF-8 -- it was this environment's console
   rendering, not the pipeline. The defensive `resp.encoding='utf-8'` pin was kept anyway as
   cheap insurance, just not reported as fixing a bug that didn't actually exist.)

### Commands to reproduce
```bash
pip install -r requirements.txt   # includes python-dotenv, requests
python -m src.prospect_pipeline   # idempotent; reuses data/cfbd_cache/ (zero new API calls if nothing changed)
pytest -q
```

### Known limitations / unfinished
- Remaining unmatched (9 of the 139, draft years 2010+): mostly nicknames CFBD and nflverse
  spell differently for the same person (`Cam`/`Cameron` Ward, `Nate`/`Nathan` Stanley) and a
  few players (Cam Newton 2011, Trey Lance 2021, John Skelton 2010) who simply don't appear
  in CFBD's `/stats/player/season` data for their draft-relevant seasons -- a genuine upstream
  coverage gap, not a matching bug. Nicknames were deliberately left unresolved: loosening the
  fuzzy-match threshold to catch them would risk false positives across the full 19,115-player
  pool, which the task explicitly warned against (identical-name collisions).
- "Class year" (Fr/So/Jr/Sr) was not found as a field in CFBD's `/roster` or `/stats` schemas
  -- left null rather than guessed. Strength-of-schedule context is team-season level
  (`/ratings/sp`), not computed per-opponent.
- `pivot_player_stats()` reports (not fabricates) 4 CFBD `statType`s it doesn't map:
  `passing:PCT`, `passing:YPA`, `rushing:LONG`, `rushing:YPC` -- the first three are CFBD's
  own pre-computed versions of stats this project already derives itself in `src/metrics.py`
  (intentionally not duplicated); `rushing:LONG` (longest rush) isn't tracked by this project.
- No frontend and no predictive model were built, per this phase's explicit instruction.

---

## Phase 3: advanced CFB-to-NFL features, outcomes and modelling preparation

Adds everything a later modelling phase needs: opponent-adjusted college performance,
development trajectories, age and experience, offensive environment and supporting cast,
team talent, NFL Combine athleticism, NFL career targets, survival records, historical comparables,
draft-value structures, and evaluation and explainability utilities. **No final predictive
model is trained in this phase, and no accuracy is claimed.** Coaching features are explicitly out of scope.

Full coverage, missingness, validation results and the list of incomplete features with reasons:
**[`reports/advanced_features_coverage.md`](reports/advanced_features_coverage.md)**. Charts are in `reports/figures/`.

### Run
```bash
python -m src.ingest_cfbd_advanced 750   # optional: warm the CFBD cache (hard per-run call cap)
python -m src.advanced_pipeline          # idempotent; 0 CFBD calls once cached
python -m src.eda_prospects              # descriptive charts -> reports/figures/
pytest -q --junitxml=results/junit.xml
python -m src.advanced_report            # renders reports/advanced_features_coverage.md from the run artifacts
```

### Architecture (new modules)
| Layer | Where | Notes |
|---|---|---|
| Raw / standardized sources | `src/ingest_cfbd_advanced.py` -> `cfb_games`, `cfb_game_passing`, `cfb_game_rushing`, `cfb_team_season_stats`, `cfb_recruits`, `cfb_receiving_seasons`, `cfbd_draft_picks`, `nfl_combine_measurements`, `cfb_offensive_scheme_labels` | Every endpoint checked against the live OpenAPI spec and probed before use; sources registered in `config/sources.json` |
| Derived team-season context | `src/features/environment.py` -> `cfb_team_season_environment`, `cfb_team_roster_talent` | Scheme profile, OL proxies, receiver context, talent (within-season percentiles) |
| Joined player-level | `src/features/opponent_adjust.py`, `development.py` -> `prospect_college_seasons`; `targets.py` -> `nfl_prospect_seasons` | One row per prospect x college season / NFL season |
| Model-ready features | `src/features/registry.py` -> `feature_definitions` (119 features, version `3.0.0`), `prospect_feature_values` (long form, each value dated) | Exports: `data/model_ready/features_<scope>.parquet` |
| Targets / outcomes | `nfl_career_outcomes`, `survival_records`, `draft_value_predictions` (empty until models exist), `prospect_similarity` | Exports: `nfl_targets`, `survival_records`, `draft_value_base`, `prospect_similarity` |
| Modelling prep | `src/modeling/`: `preprocess` (chronological splits, train-only preprocessing), `evaluation` (Brier, log loss, ROC-AUC, calibration, ECE, bootstrap / cluster-bootstrap CIs, conformal and bootstrap prediction intervals, incremental value vs a baseline), `explain` (permutation importance, optional SHAP, correlations, PDP, residuals), `similarity` (KNN comps, PCA, k-means), `survival` (Kaplan-Meier, lifelines / scikit-survival adapters, Cox readiness) | `lifelines`, `shap` and `scikit-survival` are optional and not required |

### Leakage controls
- **Model scopes** (`registry.py`): `pre_draft_college` < `pre_draft_full` (+ combine) < `pre_draft_scouting`
  (+ expert pre-draft grades) < `post_draft` (+ draft round / pick). Draft slot exists *only* in `post_draft`.
  Outcome columns can never be features (`assert_scope_safe`).
- **Every stored value carries `available_as_of`:**
  - college features: the date of the final college game (bowls included)
  - combine: March 15 of the draft year
  - draft slot: draft day
  - draft dates: `config/nfl_draft_dates.json`

  The audit found **0 of 26,885 pre-draft values knowable on or after draft day**.
- **Chronological splits by draft class** (`config/modeling_splits.json`): train 2005-2015, validation
  2016-2018, test 2019-2021. 2022-2026 is a censored holdout. Preprocessing refuses to fit on non-train rows.
- **Comparables** use only earlier draft classes, with scaling fit on that earlier pool.
- **Post-draft listings excluded:** NFL-roster heights and weights (a post-draft listing) are not used as pre-draft features.

### Headline results (2026-11-09 run; details in the report)
- 328 drafted QBs (2000-2026). Linked to college data: **252 of 260** in the 2005+ modelling universe (Phase 2: 189).
  - **Finding:** CFBD season stats are nearly empty for 2004-2008, so those seasons were rebuilt from box scores.
  - **Stable-ID check:** CFBD's draft-pick athlete id confirmed 185 name matches, with 0 conflicts.
- **Opponent adjustment:** leave-one-out opponent pass-defense baselines from 67,018 box-score lines; 678 of 679 prospect-seasons with 100+ attempts are adjusted.
- **Combine:** records for 293 QBs. Skipped drills stay NULL, with participation indicators.
- **Targets:** started 16+ in 5 seasons has 122 / 38 / 33 model-ready rows (train / validation / test). Window targets are NULL for unfinished windows.
- **Survival:** 4 event definitions. Career ends are *inferred* (2 absent seasons); nflverse has no usable official retirement flag (Brady is listed `ACT`).
- **Not populated, and documented why:**
  - college injury history
  - pro-day results
  - qualitative scheme labels (none documented, none invented)
  - verified college starts (a proxy is provided)
- **Preserved:** existing NFL tables are unchanged (verified before and after every run). Two consecutive runs gave identical row counts and byte-identical feature values.
- Tests: 133 passed, 0 failed, 13 skipped. The skips are the pre-existing live-network tests, skipped during a GitHub release-download outage (HTTP 504) on 2026-11-09.
- CFBD budget: about 644 of the free tier's 1,000 calls for November 2026. Every response is cached, and reruns make 0 calls.

### Notes on changes to earlier phases
- `src/prospect_pipeline.run_matching` now also consults CFBD's draft-pick athlete id (no-op when
  `cfbd_draft_picks` is empty). Three verified school aliases were added to `match_prospects.py`, which
  raised Phase-2 matches from 189 to 192.
- `src/cfbd_client.py` gained an optional hard per-run call budget (`max_network_calls`).
- `historical_sources.not_ingested_season_ranges` ignores Phase-3 entries tagged
  `coverage_scope: prospect_features`, so the Phase-1 NFL season-coverage report is unchanged.
- A pre-Phase-3 copy of the database is in `data/backups/` (gitignored).

---

## Offensive-scheme database (college offensive environment)

Historically grounded scheme features for testing whether a QB's college offensive environment
improves NFL career predictions. Coverage, provenance, citations and limits:
**[`reports/offensive_scheme_coverage.md`](reports/offensive_scheme_coverage.md)**.

Three evidence tiers are kept apart and never merged:

| Tier | Source | Status | Features |
|---|---|---|---|
| Verified | Manual review: a source that was read explicitly ties the label to that team **and** season (`config/offensive_scheme_labels.csv`, with URL, title, date and verbatim quote) | `verified` | `scheme_ver_*` |
| Reported | Wikipedia team-season infobox `off_scheme`, read through the official MediaWiki API and pinned to revision permalinks | `reported`, low confidence | `scheme_rep_*` |
| Quantitative | Box scores and CFBD play-by-play: standard-down and passing-down pass rates, tempo, efficiency | statistics, never labels | `scheme_q_*`, `env_*` |

- **Taxonomy** (`config/offensive_scheme_taxonomy.json`): offensive family, run concepts, passing concepts, formation and tempo. Labels overlap, and ambiguous source text stays unmapped.
- **Verified sample:** manual research used a seeded random sample of final college seasons stratified by draft class, so that "has a verified label" is not a proxy for draft position. 24 team-seasons were researched and 5 have verified labels. The rest of the log records insufficient, conflicting or unavailable sources.
- **Coverage:** 70.8% of the 2005+ modelling universe has at least one usable scheme feature. 61.9% has a reported final-season family; 42.3% has final-season play-by-play.
- **Play-by-play:** 2016-2025 (162 CFBD calls, 808 of 1,000 used in November 2026). 2004-2015 (about 200 calls) is deferred to the next quota so the free tier is not exceeded:
  `python -c "from src import ingest_cfbd_plays as P; from src.cfbd_client import CFBDClient; print(P.fetch_plays(P.SEASONS_LATER, CFBDClient(max_network_calls=220)))"`
- **Not used:** the suggested `deepmetricanalytics.com` scheme page is a 2026 preview with no per-team citations, so it covers none of these seasons.
- **Outputs:** `data/processed/college_offensive_schemes.parquet`, `qb_scheme_features.parquet`, `qb_scheme_seasons.parquet` and `offensive_scheme_labels_all.parquet`. In the database: `cfb_offensive_scheme_labels` and `cfb_team_season_schemes`. Feature version `3.1.0` in `prospect_feature_values`.
- **Run:** `python -m src.advanced_pipeline` (the scheme step is part of it), then `pytest --junitxml=results/junit.xml`, then `python -m src.scheme_report`.

## Career-success model v1 (32+ NFL starts in the first five seasons)

Results: **[`reports/model_evaluation.md`](reports/model_evaluation.md)**. Data audit: [`reports/model_training_readiness.md`](reports/model_training_readiness.md).
Feature dictionary: [`reports/model_feature_dictionary.csv`](reports/model_feature_dictionary.csv).

```bash
python -m src.modeling.build_dataset   # one row per drafted QB + separate target table (no network calls)
python -m src.modeling.train           # rolling-origin validation by draft class + final models
python -m src.modeling.evaluate        # metrics with bootstrap CIs, paired comparisons, figures, report
python -m src.modeling.predict         # score the censored 2022-2026 classes (not evaluable yet)
python -m pytest -q
```

* **Target:** regular-season starts from the nflverse schedule's starting-QB fields, NFL seasons `draft_year`
  through `draft_year + 4`. Incomplete windows are NA (censored), never failures. A QB with no NFL record counts
  as 0 starts only when the draft board confirms he never played.
* **Sample:** the 2017-2021 classes, whose final college season is inside the 2016-2025 play-by-play data and
  whose five-season windows are complete. That is 57 eligible QBs. The 53 with play-type features form the
  same-QB comparison cohort (19 successes).
* **Models (all sklearn Pipelines with train-only preprocessing):**
  * Base rate.
  * A: basic college stats (6 features).
  * B: + opponent and team context (11).
  * C: + play-type profile (16).
  * The A/B/C sets are each fit as L2, elastic-net and shallow gradient-boosted models.
  * A draft-pick reference model, which is not a college model.
  * Hyperparameters are pre-specified and the decision threshold is the training success rate.
* **Validation:** 2019, 2020 and 2021 are each predicted from the earlier classes only, giving 31 out-of-sample QBs.
* **Findings:**
  * College stats carry signal: model B reaches AUC 0.886 [0.759, 0.982].
  * Detailed play types gave **no detectable improvement** over model B.
  * Draft position is the strongest single predictor.
  * Intervals are wide at this sample size. See the report for limitations.
* **Artifacts:**
  * `models/qb_career_success/v1.0.0/*.joblib`, with `metadata.json` recording the training cutoff, dataset
    hashes, feature schema, parameters, seed and package versions.
  * The out-of-sample predictions and metric tables are written to `results/modeling/`, which is gitignored like
    the rest of `results/`.

## Phase 4: era normalization and historical training (model v2)

Reports:
* [`reports/college_era_analysis.md`](reports/college_era_analysis.md): how college passing changed from 2004 to 2025.
* [`reports/normalization_ablation.md`](reports/normalization_ablation.md): the normalization strategies compared.
* [`reports/historical_model_comparison.md`](reports/historical_model_comparison.md): v2 against v1.

```bash
python -m src.modeling.build_dataset_v2   # FBS reference population, season/conference normalization, targets
python -m src.modeling.era_report         # outcome-blind era analysis
python -m src.modeling.train_v2           # embargoed development selection, 2019-2021 holdout, thresholds
python -m src.modeling.evaluate_v2        # bootstrap intervals, figures, reports
python -m pytest -q
```

* **Normalization (`src/modeling/era.py`):** season z-scores, season percentiles and conference-season z-scores
  against every FBS passer-season with 100+ attempts (not just drafted QBs). The baselines are saved next to the model.
* **Protocol:**
  * The normalization strategy, feature level and model family were chosen on embargoed development folds only.
    Test classes are 2012-2018, and each trains only on classes whose five-season label was observable at that draft.
  * The 2019-2021 holdout is the same 31 QBs as v1.
  * Draft position is excluded from every pre-draft model.
* **Findings:**
  * Normalization did not help. Every normalized strategy was equal or slightly worse than raw.
  * Expanding training to 183 QBs from 2005-2021, with truncation-robust raw features, gave holdout Brier 0.121
    against 0.134 for v1. That is not statistically clear, but it meets the pre-registered preference rule.
  * v1's "history hurt" result came from its history pool (FCS finals, missing final seasons), not from era drift.
  * Play types still add nothing detectable.
* **Models:** `models/qb_career_success/v2.0.0/`. `v1.0.0` is unchanged.

## Phase 5: draft projection, career forecasts, comparisons and trajectories

Start with [`reports/phase5_summary.md`](reports/phase5_summary.md). Details are in:
* [`draft_model_evaluation.md`](reports/draft_model_evaluation.md)
* [`career_production_evaluation.md`](reports/career_production_evaluation.md)
* [`prospect_comparisons.md`](reports/prospect_comparisons.md)
* [`career_trajectory_evaluation.md`](reports/career_trajectory_evaluation.md)

```bash
python -m src.modeling.artifacts --check          # verify the gitignored data files the models depend on
python -m src.modeling.career_trajectory          # one row per drafted QB per NFL season; Kaplan-Meier analysis
python -m src.modeling.career_production          # five-season stat forecasts, embargoed backtest (~5 min)
python -m src.modeling.draft_predictor            # verified prospect universe; draft probability, round and pick models
python -m src.modeling.prospect_similarity        # historical comparison engine
python -m src.modeling.phase5_report              # summary report + integrated forecast examples
python -m src.modeling.forecast --name "Fernando Mendoza" --draft-year 2026
```

`src.modeling.forecast.forecast_prospect(...)` returns one dict per prospect:
* draft probability, where defined, plus round and pick ranges if drafted;
* 16/32/64+ start probabilities (v2 and Phase-5 models);
* five-season and year-by-year passing and rushing projections with 80% intervals;
* five historical comps.

Each output is tagged with how it was evaluated.

## Phase 5.5: model audit, calibration and the prediction interface

Start with [`reports/phase5_5_audit.md`](reports/phase5_5_audit.md). Details are in:
* [`draft_eligibility_audit.md`](reports/draft_eligibility_audit.md)
* [`combine_selection_bias.md`](reports/combine_selection_bias.md)
* [`calibration_improvements.md`](reports/calibration_improvements.md)
* [`model_reliability.md`](reports/model_reliability.md)
* [`frontend_readiness.md`](reports/frontend_readiness.md)

```bash
python -m src.modeling.artifacts --check            # every pinned model file and external data file
python -m src.modeling.draft_eligibility            # eligibility classes; population-definition effects
python -m src.modeling.combine_audit                # Combine participation vs measurement
python -m src.modeling.round_audit                  # nested round/pick validation; qb_draft v1.1.0
python -m src.modeling.career_production_v11        # nested production backtest; qb_career_production v1.1.0 (~15 min, 16 cores)
python -m src.modeling.similarity_audit
python -m src.modeling.trajectory_audit
python -m src.modeling.phase5_5_report
python -m src.modeling.prediction_api --name "Fernando Mendoza" --draft-year 2026 --mode pre_draft
python -m src.modeling.historical_backtest --name "Josh Allen" --draft-year 2018
```

`src.modeling.prediction_api.forecast_prospect(name, draft_year, mode='pre_draft' | 'post_draft')` is the interface
for the future dashboard. It returns a JSON-serialisable dict with player identity, draft projections, nested start
probabilities, five-season totals with 50/80/90% intervals, year-by-year paths and comps. Each block carries a
`status`; unsupported predictions return `unavailable` with a reason. `validate(f)` checks the schema.

`models/qb_career_production/v1.0.0/*.joblib` embed player-level career lines. They are kept locally, unchanged,
and are now gitignored; `v1.1.0` holds parameters only. Its simulation reference tables are a gitignored data file
listed in its metadata.

## Phase 6A: FastAPI backend

A read-only HTTP API around the existing prediction system, for the future Next.js frontend. Start with
[`reports/phase6_api_summary.md`](reports/phase6_api_summary.md). Endpoint details are in
[`api_endpoint_reference.md`](reports/api_endpoint_reference.md) and deployment in
[`deployment_readiness.md`](reports/deployment_readiness.md).

```bash
pip install -r requirements-api.txt        # inference-only dependencies (pinned to the training environment)
python -m api.bundle build                 # once, from the full local data: prediction-ready serving bundle
python -m api.bundle verify                # SHA-256 of the bundle and of every model file it uses
python -m api.performance                  # refresh api/static/model_performance.json (aggregates only; tracked)
uvicorn api.main:app --reload --port 8000  # http://localhost:8000/docs
python -m pytest -q tests/test_api.py
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status, model integrity (no engine load) |
| `GET /api/prospects?q=&draft_year=&limit=&offset=` | search / autocomplete |
| `GET /api/prospects/resolve?name=&draft_year=` | exact name, `409 AMBIGUOUS_PLAYER` when not unique |
| `GET /api/prospects/{player_id}` | verified college profile, measurements, data coverage |
| `GET /api/prospects/{player_id}/forecast?mode=pre_draft\|post_draft` | draft projection, nested start probabilities, five-season production with 50/80/90% intervals |
| `GET /api/prospects/{player_id}/comparisons` | five explanatory historical comps |
| `GET /api/backtests/{player_id}` | predicted vs actual, labelled out-of-sample / incomplete / in-sample |
| `GET /api/models`, `GET /api/models/performance` | versions, integrity, curated evaluation and limitations |

The API serves from tracked model files plus the **serving bundle** (`data/serving/v1/`, gitignored player-level
data). It needs no database, no training data and no CFBD key. Configuration (`FRONTEND_ORIGINS`,
`RATE_LIMIT_PER_MINUTE`, ...) is documented in `.env.example`.

### Data that is not in Git, and how to get it back

Trained models (`models/**`: joblib files and `metadata.json`) can be committed, except the v1.0.0 production
files above. Player-level data derived from CFBD or nflverse is never committed (see `.gitignore`). That covers the SQLite database, every parquet file, the
reference populations and the comparison pool. Each model's `metadata.json` lists the data files it needs under
`external_data_files`, with a repository-relative path under `data/`, a SHA-256 hash and the command that
regenerates it. `python -m src.modeling.artifacts --check` verifies them.

To rebuild from scratch:
1. Run the ingestion pipelines with your own `CFBD_API_KEY` in `.env`: `python -m src.historical_pipeline`, then
   `python -m src.prospect_pipeline`, then `python -m src.advanced_pipeline`.
2. Run the modeling commands above, in order.

A regenerated file whose hash differs from the pinned one is reported by `--check`. In that case, retrain into a new
model version rather than overwriting an existing one. The training scripts refuse to overwrite an existing version
directory.

