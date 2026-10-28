# NFL Quarterback Analytics — Portfolio Project

An end-to-end, statistically grounded NFL player analytics starter. Designed for data science and applied statistics internship portfolios.

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
  (structural, expected). Spot-checked matches after fixing three real issues found during
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
