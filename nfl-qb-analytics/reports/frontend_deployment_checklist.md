# Frontend deployment checklist (Phase 6B)

Nothing was deployed. The plan is the Next.js frontend on Vercel and the FastAPI backend on a Python host (see
`deployment_readiness.md` for the backend).

## Architecture decision: browser calls the API directly

* **Read-only API.** The API is read-only, returns no secrets, and enforces an explicit CORS allowlist (GET only, no
  credentials).
* **Real client addresses.** With direct calls, the backend's per-address rate limiter sees each visitor's address. Behind
  a Vercel proxy, every visitor would arrive from Vercel's egress addresses and share one limit. The backend would also
  need `--proxy-headers` with trusted forwarders.
* **No extra hop.** Direct calls avoid a serverless hop and cold start in front of the backend's own first-forecast model
  load.
* **When to add a proxy instead.** Only if the API origin has to be private or authenticated. Then: a server-only base
  URL, a GET-only route handler with a path allowlist, and forwarded client addresses.

## Secrets and data

- [ ] No secret in any `NEXT_PUBLIC_*` variable (they are compiled into the browser bundle). The frontend needs none.
- [ ] `CFBD_API_KEY` exists only on the machine that builds the serving bundle, never on Vercel or the API host.
- [ ] The serving bundle, model files and database are never served by the frontend. The frontend contains none of them
      (`frontend/` holds code, generated types, test fixtures and screenshots only).
- [ ] **Data licences reviewed** (CFBD, nflverse) for public display of player-level values. **Outstanding blocker.**
- [ ] Decide whether `frontend/docs/screenshots/*.png` may be published. They show real players' model output and college
      statistics.

## Backend first

- [ ] Backend deployed and `GET /api/health` returns `"status": "ok"` with all models verified.
- [ ] `FRONTEND_ORIGINS` set to the exact Vercel production origin (and preview origins if wanted). Never `*`.
- [ ] Platform rate limiting in front of the API.
- [ ] `QB_API_DOCS_ENABLED` decided.

## Vercel project

- [ ] Root directory: `frontend`
- [ ] Framework preset: Next.js. Build `npm run build`, install `npm ci`, Node 20.9 or newer (developed on Node 24).
- [ ] Environment variables:
  - `NEXT_PUBLIC_API_BASE_URL=https://<api host>`
  - `NEXT_PUBLIC_GITHUB_URL=https://github.com/vinleetcode/qb-performance-analytics`
- [ ] Preview deployments need their origins in `FRONTEND_ORIGINS`, or a separate staging API.
- [ ] Fonts are fetched at build time by `next/font` (Google Fonts) and self-hosted in the build output. No runtime call
      to Google.

## Before going live

- [ ] `npm run lint && npm run typecheck && npm test` pass.
- [ ] `npm run build` passes.
- [ ] `E2E_BASE_URL=https://<preview url> E2E_API_URL=https://<api host> npm run test:e2e` passes against the preview.
- [ ] Cold-start check: the first forecast after an API restart shows the loading note and then the report (about 10 s).
- [ ] Check a phone (iOS Safari and Android Chrome) and a keyboard-only pass.
- [ ] 404 page and backend-outage message checked on the preview.
