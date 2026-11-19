# Frontend integration tests (Phase 6B)

All results below are from runs on 2026-11-10 on the development machine (Windows 11, Node 24.18, Python 3.14).

## Summary

| Suite | Command (from `frontend/` unless noted) | Result |
|---|---|---|
| TypeScript, strict | `npm run typecheck` | 0 errors |
| ESLint | `npm run lint` | 0 errors, 0 warnings |
| Unit and component (Vitest, Testing Library, axe-core) | `npm test` | **31 passed** |
| End-to-end, live API, production build (Playwright, Chromium) | `E2E_BASE_URL=http://localhost:3000 npm run test:e2e` | **25 passed** |
| Production build | `npm run build` | succeeded |
| Existing Python suite | `python -m pytest -q` (from `nfl-qb-analytics/`) | **304 passed** |

## What the unit and component tests cover (`src/**/*.test.ts(x)`)

They use synthetic fixtures (`src/test/fixtures`, players named "Fixture QB ..."). The fixtures are typed against the
generated API schema, so they cannot drift from the real contract without failing to compile. Application code never
imports them.

| Area | Tests |
|---|---|
| API client | parameter encoding; backend error bodies become typed `ApiError`s (`AMBIGUOUS_PLAYER` keeps its candidates); an unreachable backend becomes `BACKEND_UNREACHABLE`; `INVALID_PREDICTION_MODE`; only transient errors are retried, at most twice; player-id validation |
| Formatting | probabilities near 0 or 1 shown as bounds; backend codes and fragments turned into sentences |
| Prospect search | no query under two characters; autocomplete; two prospects sharing a name both flagged and told apart by college and year; keyboard selection returns the stable id; empty results explained |
| Field ruler | accessible label lists expected value, every interval and the actual result |
| Forecast rendering | unsupported P(drafted) shows the reason, never `0%`; nested starting probabilities with the five-season window; all ten projections with 50/80% ranges and measured coverage; missing year-by-year explained; unavailable career shows no table |
| Pre-/post-draft | actual pick hidden in the pre-draft view until revealed; shown post-draft; a college-only player in post-draft mode gets an explanation and **no forecast request is made** |
| Backtest labels | each of the three classifications gets its own title; in-sample results carry the "trained on this player" warning and a table caption saying they are not forecast accuracy; open windows show "Not yet known" |
| Model performance | methodology numbers come from the response (the Combine share is computed from the three AUCs); evaluation status and limitations shown |
| API errors | unreachable backend gives an alert with retry; a missing prospect gets no retry; a 503 is retried twice, then shown, with no forecast numbers |
| Accessibility | axe-core structural rules pass on the report sections, the backtest view and the open search combobox |

## What the end-to-end tests cover (`e2e/app.spec.ts`)

Real integration: the Next.js production build (`npm start`) talks to the running FastAPI backend and its real models,
under the backend's real CORS allowlist (`http://localhost:3000`).

1. The landing page shows the 2026 class from the API (Fernando Mendoza, pick 1) and a live forecast ruler.
2. Searching "josh allen" lists the Wyoming QB; choosing him opens `/prospects/qb_226`.
3. Pre-draft hides the pick. Switching to post-draft changes the URL, shows "Drafted round 1, pick 7", puts the actual
   pick on the ruler, and changes the passing-yards forecast.
4. The page's 32+ probability and passing-yards expectation equal the API's own values, and the API's probabilities
   are nested.
5. Unsupported outputs: Mendoza's P(drafted) explained. An undrafted QB in post-draft mode shows the explanation; a
   hypothetical pick produces a labelled hypothetical forecast.
6. Five comparisons, each linking to a `qb_<n>` report.
7. Backtest labels on real players: Josh Rosen (genuine out-of-sample), Matt Ryan (in-sample, warned), C.J. Stroud (window
   open, errors "Not yet known"), plus the aggregate panel over 107 QBs.
8. The methodology page shows the Combine finding and the draft-slot explanation.
9. Backend outage (all API requests aborted): alert shown, no production section rendered.
10. Keyboard: skip link first in tab order; search focusable.
11. axe-core in Chromium: no serious or critical violations on `/`, `/prospects`, a report, a backtest and the methodology page.
12. No horizontal overflow on five pages at a Pixel 7 viewport, and at desktop width.

## Problems the tests found, and the fixes

| Found by | Problem | Fix |
|---|---|---|
| browser console check | `usePathname()` in the header and awaited route params outside Suspense (Next 16 Cache Components) | nav wrapped in Suspense with a non-active fallback; params read inside the boundary |
| axe (real browser) | `aria-hidden-focus`: focusable Recharts surfaces inside an `aria-hidden` wrapper | wrapper no longer hidden; charts keep descriptions and data tables |
| axe (real browser) | `link-in-text-block`: inline links shown by colour only | inline links underlined |
| unit test | a shell-applied edit had put control characters into a regex, so backend codes were not humanized | regex rewritten; sources scanned for control characters (none left) |
| screenshots | the 80% and 90% bands were hard to tell apart, with the scale showing through | solid band steps; scale only on the empty track |
| screenshots | a negative rushing bound stretched a chart axis to -6,000 | whiskers drawn from zero; exact bound kept in the data table |
| screenshots | in post-draft mode the college-only v2 reference probability read like a contradiction | labelled as college data only, never the draft slot |

One run failed for an environmental reason, not a code reason: a stale production server from an earlier run was still
bound to port 3000. The rerun against a freshly started build passed 25 of 25.

## Visual inspection

* Every page was rendered in Chromium at 1440x1000 and 390x844 and the screenshots were reviewed, along with 2x crops of
  the ruler, production section and mobile header.
* README screenshots (`frontend/docs/screenshots/`) were generated by `scripts/readme-shots.mjs` from the production build
  against the live API.
* Not done: a manual screen-reader session, and testing on physical iOS or Android devices.

## Reproduce

```bash
# backend, from nfl-qb-analytics/
uvicorn api.main:app --port 8000
python -m pytest -q

# frontend, from frontend/
npm ci
npm run lint && npm run typecheck && npm test
npx playwright install chromium      # once
npm run build && npm start &
E2E_BASE_URL=http://localhost:3000 npm run test:e2e
```
