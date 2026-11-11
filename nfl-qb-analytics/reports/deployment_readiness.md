# Deployment readiness (Phase 6A)

Nothing was deployed. This document says exactly what a deployment needs, what is in Git, and what is not.

## Runtime artifacts

The API needs three groups of files and nothing else: no SQLite database, no raw CFBD/nflverse downloads, no training
datasets, no `.env`, no CFBD key.

| Group | Files | In Git? | How it gets to the server |
|---|---|---|---|
| Code | `api/`, `src/`, `config/` | yes | repository |
| Model files | `models/qb_draft/v1.0.0`, `models/qb_draft/v1.1.0`, `models/qb_career_production/v1.1.0`, `models/qb_similarity/v1.0.0`, `models/qb_career_success/v2.0.0` (joblib + metadata.json) | yes (parameters only; checked by the embedded-data scan) | repository |
| Curated performance summary | `api/static/model_performance.json` (aggregate metrics only) | yes | repository |
| Serving bundle | `data/serving/v1/` (profiles, prospect index, college season lines, details, precomputed backtests, manifest) | **no**: player-level data derived from CFBD/nflverse | built locally, shipped with the deployment |
| Model data files | `data/model_artifacts/qb_career_production/v1.1.0/reference_tables.npz` (simulation donor paths), `data/model_artifacts/qb_similarity/v1.0.0/pool.parquet` (comparison pool) | **no**: player-level | inside the bundle archive |

Not needed and never deployed: `models/qb_career_production/v1.0.0/*.joblib` (gitignored; embed player-level career lines),
`models/qb_career_success/v1.0.0` (not served), `data/qb_history.db`, `results/`, `reports/`, `tests/`.

## Building and shipping the bundle

```bash
# on a machine with the full local data (the one that trained the models)
python -m api.bundle build        # data/serving/v1/ + manifest with SHA-256 of every file and every model file used
python -m api.bundle verify       # must report 0 problems
python -m api.bundle pack         # dist/qb_serving_bundle_v1.tar.gz  (gitignored)

# on the server, from nfl-qb-analytics/
python -m api.bundle install qb_serving_bundle_v1.tar.gz   # refuses absolute paths, '..', links, other prefixes
python -m api.bundle verify
```

The manifest pins the hash of every model file the bundle was built against. A bundle built against a different model
version fails `verify`, and the API answers `503 MODEL_UNAVAILABLE` rather than serving mismatched predictions.

**Licensing note (decision for you).** The bundle contains CFBD- and nflverse-derived player-level values (college
season lines, profiles, NFL outcomes of historical QBs). A public website displays these values by design. Before a
public launch, check the CollegeFootballData.com terms of use and the nflverse data licences for redistribution and
display. They were not reviewed in this phase. The repository itself never contains these files.

## Container

`Dockerfile` (inference-only image) and `.dockerignore` are in `nfl-qb-analytics/`. The image copies the code, the five
served model directories, the serving bundle and the two model data files, runs `python -m api.bundle verify` at build
time (the build fails on a mismatch), runs as a non-root user and defines a `HEALTHCHECK` on `/api/health`.

```bash
python -m api.bundle build && python -m api.bundle verify
docker build -t qb-api .
docker run -p 8000:8000 -e FRONTEND_ORIGINS=https://your-frontend.example qb-api
```

Docker is not installed on the development machine, so **the image has not been built**. The same file set was
checked instead by copying exactly the files the Dockerfile copies into an empty directory and running the API from
there (no database, no `.env`, no training data). See `phase6_api_summary.md`.

## Platform settings

| Setting | Value |
|---|---|
| Python | 3.14 (the artifacts were pickled under 3.14 with the versions pinned in `requirements-api.txt`) |
| Install | `pip install -r requirements-api.txt` |
| Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers` |
| Health check | `GET /api/health`, expect 200 and `"status": "ok"` (`degraded` = artifacts missing or failed verification) |
| Memory | about 1 GB per worker once the engine is loaded |
| Cold start | the first forecast loads the engine (a few seconds); health checks never load it |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FRONTEND_ORIGINS` | `http://localhost:3000` | comma-separated explicit CORS allowlist; `*` is ignored; credentials never allowed |
| `RATE_LIMIT_PER_MINUTE` | `120` | per client address, in process memory; `0` disables |
| `QB_API_BUNDLE_DIR` | `data/serving/v1` | bundle location relative to `nfl-qb-analytics/` |
| `QB_API_N_SIMS` | `2000` | simulation draws for year-by-year paths (fixed seed) |
| `QB_API_CACHE_SIZE` | `512` | in-memory forecast cache entries |
| `QB_API_MAX_PAGE_SIZE` | `50` | search page cap |
| `QB_API_DOCS_ENABLED` | `true` | serve `/docs` and `/openapi.json` |
| `LOG_LEVEL`, `APP_ENV` | `INFO`, `development` | logging |

`CFBD_API_KEY` is **not** used by the API and should not be set on the server.

## Deployment checklist

- [ ] `python -m pytest -q` passes locally
- [ ] `python -m src.modeling.artifacts --check` and `python -m api.bundle verify` report 0 problems
- [ ] `python -m api.performance` re-run if any evaluation output changed (commit the JSON)
- [ ] Data licences reviewed for public display (see above)
- [ ] `FRONTEND_ORIGINS` set to the production frontend origin(s) only
- [ ] `QB_API_DOCS_ENABLED` decided (public docs are harmless but optional)
- [ ] Platform-level rate limiting / WAF in front of the service (the built-in limiter is per process)
- [ ] Bundle archive uploaded through a private channel (not a public release asset)
- [ ] Health check configured on `/api/health`
- [ ] `CFBD_API_KEY` absent from the server environment

## Blockers and risks

1. **Docker build untested** (Docker not installed here). The file set it copies was verified with a clean-directory run.
2. **Pinned binary versions.** The pickled models need the pinned scikit-learn/NumPy/SciPy and Python 3.14. A host
   without Python 3.14 images needs the models re-exported (a retrain into a new version), which is out of scope here.
3. **Data licensing** for public display (above).
4. **Single-process state.** The forecast cache and rate limiter are in memory. With several workers or instances, each
   has its own; use platform rate limiting.
5. **Bundle refresh.** New data or model versions need a new bundle build (`build` -> `verify` -> `pack` -> `install`).
   The cache key includes the bundle and model hashes, and a restart clears the cache.
