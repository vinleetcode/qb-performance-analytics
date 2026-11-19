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


## Phases 6B-7 update (2026-11-11)

### Docker

**Not tested.** Docker is not installed on the development machine (`docker: command not found`), so the backend image
has still not been built or run. Its file set was verified in Phase 6A by a clean-directory run. On a machine with Docker:

```bash
cd nfl-qb-analytics
python -m api.bundle build && python -m api.bundle verify      # the bundle must be in the build context
docker build -t qb-api .                                       # the image runs `python -m api.bundle verify`
docker run --rm -p 8000:8000 -e FRONTEND_ORIGINS=http://localhost:3000 qb-api
curl localhost:8000/api/health                                 # expect "status":"ok", all models verified
docker run --rm qb-api python -c "import sklearn, sys; print(sys.version, sklearn.__version__)"   # 3.14 / 1.9.1
docker run --rm qb-api python -m src.modeling.artifacts --check
```

Then run the frontend's end-to-end suite against it: `E2E_API_URL=http://localhost:8000 E2E_BASE_URL=http://localhost:3000 npm run test:e2e`.

### Hosting plan (not deployed)

| Part | Platform | Notes |
|---|---|---|
| Frontend | Vercel, root directory `frontend` | `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_SITE_URL`, `NEXT_PUBLIC_GITHUB_URL`. Security headers (CSP restricted to the site and the API origin, frame denial, nosniff, referrer and permissions policies) come from `next.config.ts` |
| Backend | Any Docker host (`render.yaml` example included) | about 1 GB RAM per worker; health check `/api/health`; `FRONTEND_ORIGINS` set to the Vercel origin; `QB_API_DOCS_ENABLED=false`; platform rate limiting |
| Serving bundle | Private storage, pulled at image build | Never in Git, never in `frontend/public` (checked: `public/` holds only photos, screenshots and the licence JSON) |

**Cold start.** The first forecast after a backend start loads the models: about 6-10 s, and up to 27 s observed under
memory pressure. The site shows an explanatory loading note. Keep one warm instance, or call a forecast after each deploy.

**Error logging.** The backend logs every request with an id (`X-Request-ID`) and logs unhandled errors server-side only.
The frontend shows typed error states. A hosted error tracker is optional.

### Deployment blockers

1. **Data licensing.** The public JSON API could be used as a bulk mirror of CFBD-derived data, and some fields originate
   with Pro Football Reference. `public_data_licensing_review.md` marks the API exposure as blocked until mitigated.
2. **Docker image untested** (no Docker here).
3. **Python 3.14** with pinned scikit-learn/NumPy/SciPy is required to load the models.
4. **Origins.** `FRONTEND_ORIGINS` must be set to the production frontend origin.


### Phase 8 note (2026-11-11)

`api/main.py` now registers CORS as the outermost middleware, so rate-limited (429) responses carry CORS headers and
browsers see `RATE_LIMITED` rather than a network failure. Covered by `test_rate_limited_responses_keep_cors_headers`.
Nothing else in the API changed. The deployment blockers above are unchanged.


## Phase 9: free deployment, Vercel Hobby + Render Free (2026-11-19)

**Not live yet.** Everything below is prepared and verified locally. Creating the accounts, pushing to GitHub and
clicking deploy are owner actions; this environment has no Vercel, Render or GitHub credentials and no Docker.

### Architecture ($0)

| Part | Service | Plan | Cost |
|---|---|---|---|
| Next.js frontend (`frontend/`) | Vercel | Hobby (free) | $0 |
| FastAPI backend (`nfl-qb-analytics/Dockerfile`) | Render web service | Free (512 MB, 0.1 CPU, sleeps after 15 min idle) | $0 |
| Serving bundle archive (1.4 MB, player-level data) | private GitHub repository | Free | $0 |
| Domains | `*.vercel.app`, `*.onrender.com` | generated | $0 |

No database, no persistent disk, no paid add-ons, no custom domain. Render's 750 free instance hours per workspace per
month cover one always-on free service (a 31-day month is 744 h); do not run a second free service in the same
workspace. If Render's included build minutes run out and no payment method is on file, Render disables new builds
for the rest of the month instead of billing.

### How the bundle reaches Render without Git or a public URL

* Render builds the image from Git, and the bundle is gitignored. Render secret files are capped at 1 MB in total, too
  small for the 1.4 MB archive.
* So the archive sits in a **private** GitHub repository. The Docker build downloads it from the GitHub contents API
  with a fine-grained, read-only token.
* The token is a Render **secret file** (`qb_bundle_token`) mounted as a BuildKit secret for that one `RUN` step. It is
  never a build arg, and it is in no image layer or log.
* `api/fetch_bundle.py` refuses non-HTTPS URLs. It checks the archive against the SHA-256 pinned in Git
  (`deploy/serving_bundle.sha256`), then extracts it with the existing safe `api.bundle install`.
* `api.bundle verify` checks every file against the manifest and the tracked model hashes. Any problem fails the build
  instead of deploying an API without data.
* The bundle is never served. The API has no file endpoints, and `frontend/public` holds only photos and brand assets.

### Exact steps

**1. Commit and push** (yours to do: nothing has been committed). `frontend/` is not in Git at all yet, and Vercel can
only deploy what is on GitHub. The changed and new files are listed in `phase9_launch_summary.md`. `.env`,
`.env.local`, `data/`, `dist/` and `.next/` are gitignored; check `git status` before committing.

**2. Private bundle storage (GitHub, free)**
1. github.com/new: name `qb-serving-bundle`, visibility **Private**, create.
2. "uploading an existing file": add `nfl-qb-analytics/dist/qb_serving_bundle_v1.tar.gz`, then commit.
3. GitHub > Settings > Developer settings > Personal access tokens > **Fine-grained tokens** > Generate new token:
   - Repository access: *Only select repositories* > `qb-serving-bundle`
   - Permissions: Repository > **Contents: Read-only** (nothing else)
   - Expiration: your choice; the token is only needed when Render builds
4. Optional local check (PowerShell): `curl.exe -sSL -H "Authorization: Bearer <token>" -H "Accept: application/vnd.github.raw+json" https://api.github.com/repos/<you>/qb-serving-bundle/contents/qb_serving_bundle_v1.tar.gz -o b.tgz; certutil -hashfile b.tgz SHA256`.
   The hash must equal `deploy/serving_bundle.sha256`.

**3. Render (backend)**
1. render.com: sign up with GitHub. Do **not** add a payment method.
2. New > **Blueprint**, then pick the `qb-performance-analytics` repository. Render reads `render.yaml` from the repo root.
   Confirm that the service shows the **Free** instance type.
3. When prompted:
   - `FRONTEND_ORIGINS` = `https://qb-forecast-lab.vercel.app` (the name you will give the Vercel project; fix it in
     step 5 if Vercel assigns a different one)
   - `QB_BUNDLE_URL` = `https://api.github.com/repos/<you>/qb-serving-bundle/contents/qb_serving_bundle_v1.tar.gz`
4. Service > Environment > **Secret Files** > Add: filename `qb_bundle_token`, contents the token. Save.
5. Manual Deploy > Deploy latest commit. The first build will have failed without the token; that is expected.
6. When it is live, open `https://<service>.onrender.com/api/health`. Expect `"status":"ok"` and every model
   `"verified": true`.

   Dashboard alternative to the Blueprint: New > Web Service > the repository; Language **Docker**; Dockerfile path
   `nfl-qb-analytics/Dockerfile`; Docker build context `nfl-qb-analytics`; Instance type **Free**; health check path
   `/api/health`. Add the environment variables from `render.yaml` and the secret file above.

**4. Vercel (frontend)**
1. vercel.com/signup: choose **Hobby**, continue with GitHub.
2. Add New > Project > import `qb-performance-analytics`. Project name `qb-forecast-lab`. **Root Directory `frontend`**.
   Framework: Next.js (detected; `vercel.json` pins `npm ci` / `npm run build`).
3. Environment variables (Production and Preview):
   - `NEXT_PUBLIC_API_BASE_URL` = `https://<service>.onrender.com`. Required: the build fails without an `https://`
     value, so localhost can never ship.
   - `NEXT_PUBLIC_GITHUB_URL` = `https://github.com/vinleetcode/qb-performance-analytics`
   - Leave `NEXT_PUBLIC_SITE_URL` unset; the build uses Vercel's production domain.
4. Deploy. Leave Web Analytics, Speed Insights and other add-ons off.

**5. Connect them.** If the Vercel production URL differs from step 3, set Render's `FRONTEND_ORIGINS` to it exactly:
no trailing slash, and add more origins comma-separated. Saving redeploys the service.

**6. Staging, then production.** Every Vercel push to a non-production branch creates a Preview deployment. To use one
as staging against the same API, add its origin to `FRONTEND_ORIGINS` temporarily. Run the smoke test on the preview,
then merge to `main`.

**7. Smoke test** (from `nfl-qb-analytics/`, with the local bundle present, so it also proves live == local):

    python deploy/smoke_test.py --site https://qb-forecast-lab.vercel.app --api https://<service>.onrender.com

### Free-tier behaviour

* The backend sleeps after 15 minutes without traffic; waking takes about a minute (Render's figure), then the warm-up
  thread loads the engine.
* The site handles this openly:
  - the homepage hero, navigation, methodology text, credits and photos are static and never wait for the API
  - API-backed sections show skeletons and, after 2.5 s, a note that the free server is waking
  - requests time out after 75 s and retry with backoff (2, 4, 8, 15, 15 s) on unreachable/502/503/504 errors
  - after that, a friendly error with a retry button is shown
  - no forecast is ever invented or cached from a failure
* There is no keep-alive pinging or other artificial traffic: the free tier's sleep is respected as designed.
