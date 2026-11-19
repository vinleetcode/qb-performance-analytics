# Phase 9: free public launch ($0), summary

Date: 2026-11-19. Status: **ready to deploy, not deployed.** There is no public URL yet. Deploying needs account
sign-ups and a `git push`, which only the owner can do (see `deployment_readiness.md`, "Phase 9").

## Hosting ($0 total)

* **Frontend:** Vercel Hobby, root directory `frontend`, on a `*.vercel.app` domain.
* **Backend:** Render Free web service (Docker), defined by the root `render.yaml` with `plan: free`, on an
  `*.onrender.com` domain.
* **Serving bundle:** a private GitHub repository, fetched at build time with a read-only token held in a Render
  secret file.
* **Not used:** paid plans, payment details, databases, disks, custom domains, add-ons and keep-alive pings.

Vercel Hobby fit: the Hobby and fair-use pages read on 2026-11-19 confirm that the plan is free and that a Hobby
project is paused, not billed, at its limits. Fetching the Vercel Terms of Service was blocked in this session.
Vercel's published position is that Hobby is for personal, non-commercial use. A personal portfolio with no ads,
payments or client work fits that, but confirm it at vercel.com/legal/terms.

## What changed

| Area | Change | Files |
|---|---|---|
| Cold start | Profiles loaded once per process; fast C-pickle copy made at image build, hash-checked against the manifest; single-threaded BLAS; background warm-up at boot | `api/bundle.py`, `api/services/prediction_service.py`, `api/main.py`, `api/config.py`, `Dockerfile` |
| Bundle supply | `api/fetch_bundle.py`: local archive or private HTTPS URL plus BuildKit-secret token; pinned SHA-256; safe install; build fails on any problem | `api/fetch_bundle.py`, `deploy/serving_bundle.sha256`, `Dockerfile`, `.dockerignore` |
| Render | `$PORT` binding, proxy headers so the rate limiter sees real clients, docs off, page cap 20, free plan, health check | `Dockerfile`, `render.yaml` (repo root; replaces the untracked `nfl-qb-analytics/render.yaml` example) |
| Vercel | Build fails without an `https://` API URL; Open Graph base taken from Vercel's production domain; `vercel.json` | `frontend/next.config.ts`, `frontend/src/app/layout.tsx`, `frontend/vercel.json` |
| Free-tier UX | 75 s request timeout; retries that last through a wake-up (5 with backoff); honest "free server is waking" note; production-friendly error text; missing models never retried | `frontend/src/lib/api/client.ts`, `queries.ts`, `components/providers.tsx`, `components/status.tsx`, `components/prospect-search.tsx` |
| Bundle size | Recharts no longer on the homepage (246 KB initial JS instead of about 358 KB) | `frontend/src/lib/backtest.ts`, `components/landing.tsx`, `case-study.tsx`, `backtest/*` |
| Licensing | Sports Reference / PFR credit on `/credits`; licensing review updated | `frontend/src/app/credits/page.tsx`, `public_data_licensing_review.md` |
| Verification | Deployment tests; public smoke-test script | `tests/test_deploy.py`, `deploy/smoke_test.py`, `frontend/e2e/app.spec.ts`, `frontend/src/lib/api/client.test.ts`, `frontend/src/components/components.test.tsx` |

No model was retrained, and no prediction logic or model file changed.

## Test results (this session)

| Check | Result |
|---|---|
| Full Python suite (`python -m pytest -q`) | **321 passed** (includes 16 new deployment tests) |
| Bundle and model integrity (`python -m api.bundle verify`) | 21 files, 0 problems; archive members match the manifest; archive SHA-256 equals the pin |
| Fast profile path gives identical outputs (detail, pre/post-draft, hypothetical pick, comparisons; 3 prospects) | identical |
| Linux wheels for every pinned dependency on CPython 3.14 (`pip install --dry-run --platform manylinux...`) | all resolve (numpy 2.5.2, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, pyarrow 26.0.0, ...) |
| Container-equivalent run (only the files the Dockerfile copies, in a clean directory; fetch, verify, optimize; image env and CMD) | health ok in 2.8 s, engine warm in 6.8 s, first forecast 163 ms, docs 404, page cap 20, CORS allowlist enforced, per-client rate limit via `X-Forwarded-For` |
| Frontend unit tests (Vitest) | **48 passed** |
| TypeScript (`npm run typecheck`) | pass |
| ESLint | 0 problems |
| `next build`, production env | pass; the guard build without an API URL fails as intended |
| Playwright e2e against production build plus container-equivalent API (desktop and Pixel 7) | **45 passed, 2 skipped** (mobile-only tests in the desktop project). The first run hit the 120/min production rate limit from a single test machine (the test now fails loudly on 429); it passed when re-run |
| Smoke-test script against the local stack | all forecast, safety and page checks pass; live == local engine. Only HTTPS/HSTS/origin checks fail locally, as expected |
| Docker image build | **Not run: Docker is not installed here.** Covered by the clean-directory run above |
| Lighthouse / Core Web Vitals | **Not measured** (no Lighthouse here; needs the public URL) |

## Security review

| Item | Finding |
|---|---|
| CORS | Explicit allowlist from `FRONTEND_ORIGINS`; `*` ignored; GET only; no credentials. Verified: an allowed origin gets the header, others do not |
| HTTPS | Vercel and Render terminate TLS on their generated domains; the frontend build refuses a non-https API URL |
| Security headers | Frontend: CSP (`connect-src` limited to the site and the API origin), `X-Frame-Options: DENY`, `nosniff`, referrer and permissions policies, COOP. HSTS is added by Vercel on `vercel.app`. API: `nosniff`, `X-Request-ID` |
| Rate limiting | 120/min per client. Behind Render's proxy the client comes from `X-Forwarded-For` (verified: client A limited at request 121, client B unaffected). **Residual:** with `FORWARDED_ALLOW_IPS=*`, a scripted client can spoof the header to evade the per-client limit. Render Free has no edge rate limiting to add on top |
| Input validation | Player-id regex; typed, bounded query parameters; structured 4xx errors |
| Tracebacks | Hidden; unhandled errors return `INTERNAL_ERROR` and log server-side only (tested) |
| Secrets | API needs none; the CFBD key is absent from all 339 committable files and the frontend build (scanned); bundle token is a BuildKit secret, not a build arg |
| Files and archive | No file-serving endpoints; `/docs`, `/openapi.json`, bundle and `.env` paths return 404 on both API and site (smoke test); bundle archive only in a private repository |
| Deserialization | Only files from the hash-verified bundle and tracked models are unpickled; the fast-load copy is used only after both hash checks (a tampered copy is never unpickled, tested) |
| Raw dataset endpoints | None; search is paginated (max 20); per-player detail only |

## Blockers and owner decisions

1. **Deployment itself** needs you:
   - commit and push (`frontend/` is not in Git yet)
   - create the private bundle repository and token
   - create the Render service and the Vercel project
2. **Licensing decisions**, detailed in `public_data_licensing_review.md`:
   - whether the anti-mirroring mitigations are enough for publishing per-player college season lines (CFBD Sec. 5)
   - the unresolved question of Sports Reference's ML clause for data obtained through nflverse
3. **Docker image not built locally.** The first Render build is the first real build. If it fails, the build log
   names the step.
4. **Free-tier latency:**
   - the first visit after 15 idle minutes waits about a minute for Render to wake
   - engine warm-up at 0.1 CPU is estimated, not measured, at up to a minute or more
   - the site shows this honestly

## Recommended next steps

1. Deploy using the runbook, then run `python deploy/smoke_test.py --site ... --api ...` and paste the output into
   `production_performance.md`.
2. Run PageSpeed Insights on the homepage and a prospect report.
3. Add the live URL to `README.md`, `frontend/README.md` and the résumé.
4. Optional, still $0:
   - lazy-load the report's charts, which are below the fold
   - add `sitemap.ts` and `robots.ts`
