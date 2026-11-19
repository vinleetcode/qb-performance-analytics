# Frontend visual QA (Phases 6B-7)

Done 2026-11-11 in Chromium (Playwright) against the live FastAPI backend and its real models. Nothing below was judged
from code alone: each item was rendered, and the screenshots were reviewed.

## Widths and pages

| Width | Pages | How |
|---|---|---|
| 1440 x 900 / 1000 (desktop) | all pages | full-page and above-the-fold screenshots, reviewed |
| 820 x 1180 (tablet) | home, report, backtest, methodology | automated overflow check; backtest screenshot reviewed |
| 390 x 844 and Pixel 7 (phone) | all pages | full-page screenshots reviewed; automated overflow check on seven pages |

**Automated checks** (Playwright, `e2e/app.spec.ts`, 38 passed, 1 skipped by design):
* No horizontal overflow at phone, tablet and desktop widths.
* No broken images and no browser console errors on eight routes.
* No serious or critical axe-core violations on seven routes.
* The mobile menu opens, navigates and closes.
* Keyboard: skip link and search.

## Findings and fixes, this phase

| Found | Fix |
|---|---|
| Report header photo almost invisible under a heavy scrim, cropped to night sky at the wide aspect | Lighter scrim on the side without text; focal point moved to the floodlit field |
| Six headline statistics needed (passing yards, TDs, INTs, rushing yards, rushing TDs, games started) | Two rows of three headline rulers, then attempts, completions, rush attempts and games played |
| Navigation wrapped awkwardly on phones | Disclosure menu below 768 px (`aria-expanded`, Escape closes it, links close it) |
| Footer "Source code on GitHub" matched an end-to-end locator for photo "source" links | Test uses exact matching; no UI change |

From Phase 6B and still verified:
* solid interval bands;
* chart axes clamped at zero;
* underlined inline links;
* focusable charts no longer inside `aria-hidden`;
* the college-only reference probability labelled as such.

## What to look at

* **Homepage hero.** The floodlit night-stadium photo is behind a left-weighted scrim. The headline "From college
  production to NFL projection." is in Barlow Condensed at up to 104 px, with search, "Explore prospects" and "See
  historical accuracy". On the right is a live pre-draft forecast card (Fernando Mendoza, passing yards and TDs on the
  field ruler). The credit is bottom right.
* **Below the hero.** Seven sections in alternating layouts, not a repeated card grid:
  * a four-step sequence;
  * the 2026 class list;
  * a product screenshot beside text;
  * live comparisons with initials;
  * a live Josh Allen backtest on rulers;
  * evaluation figures from the API;
  * a link list.
* **Prospect report.** Contained photo header with an initials monogram (no player photography), statistics row, sticky
  pre/post-draft switch, a line naming the model versions, draft projection, starting chances, six headline rulers,
  season-by-season chart, comparison cards with a final-season dot-strip chart, uncertainty and limitations.
* **Backtesting.** The 1922 public-domain photo (grayscale) header, selector and examples, a classification banner, grouped
  bars, rulers with actual flags, the error table, "How a fair backtest works" with the embargo timeline, and aggregate
  accuracy.
* **Methodology.** Practice-field header, motivation and an original pipeline diagram, findings, data, testing timeline,
  draft (Combine finding), round and pick, starting chances, production, bias and missing data, reading the uncertainty,
  calibration, comparisons, limitations.
* **Case study and credits.** New pages. The case study uses the 1927 public-domain photo, real screenshots and live
  results from the API. Credits lists every photo, data source, font and icon licence.

## Not done

* No manual screen-reader session (NVDA or VoiceOver).
* No physical iOS or Android device, and no browser other than Chromium (Safari and Firefox untested).
* No Lighthouse run. Image optimization uses `next/image` (AVIF/WebP, responsive `sizes`, lazy below the fold,
  `priority` on page heroes), but load times were not profiled on a throttled network.


## Phase 8 (2026-11-11): forecast presentation, logo and brand

**Rendered and inspected.** The production build (`next start` on port 3100, because a separate server already held
3000) ran against the live API. Screenshots were captured with Playwright and reviewed at 1440, 1100, 820 and 390 px:
homepage, report production section, header logo. Logo concepts were rendered at 16, 32, 48, 64, 96, 180 and 200 px
on dark and light grounds.

**Found and fixed during review.**
* Headline cells wrapped the unit and range at five columns, so the unit moved under the number and the range was set on
  one line. Rulers sat at different heights across cells, so the coverage flag now takes the slack instead.
* "Runs narrow" appeared next to a coverage shown as "75%" (true value 74.8%) while the text said "fewer than 75%".
  Flags now compare the rounded percentage the page shows.
* The reading guide lower-cased "TDs" to "tds" (fixed with `midSentence`).
* Table headers and statistic names wrapped at desktop width; set to no-wrap, with a slimmer ruler column.
* The "50, 80, 90%" credibility figure wrapped onto three lines; replaced with "3" and the levels in the text.
* The search placeholder was cut off on phones; shortened.
* **Backend bug found by the end-to-end run:** rate-limited responses (429) had no CORS headers, because the
  rate-limit middleware sat outside the CORS middleware. Browsers then reported a rate limit as a network failure
  ("backend unreachable") instead of `RATE_LIMITED`. CORS is now the outermost middleware
  (`api/main.py`), with a regression test (`test_rate_limited_responses_keep_cors_headers`). No model or prediction
  code changed.

**Checks.**

| Check | Result |
|---|---|
| Lint, typecheck | clean |
| Vitest | 45 passed (10 new: headline strip, unavailable statistic, coverage flags, interval chart and stat switching, nesting warning, grouped table with header toggletips and axe, toggletip by click and keyboard, helpers) |
| Playwright against the production build and live API | 44 passed, 2 skipped by design (the two phone-only tests skip on the desktop project) |
| New end-to-end tests | every headline card, chart label and table row equals the API response for Josh Allen post-draft; 50% inside 80% for 20+ prospects in both modes (over 150 statistic checks); toggletips by hover and keyboard; on a Pixel 7, cards instead of the table and toggletips open on tap; no horizontal overflow on the report at phone width |
| axe (serious or critical) | none on 7 pages |
| Python tests | 305 passed |
| `artifacts --check`, `api.bundle verify` | 0 problems |
| Model files | 29, unchanged against `results/phase6/baseline_model_hashes.txt` |

The end-to-end run used `RATE_LIMIT_PER_MINUTE=600`: the suite sends a few hundred API requests a minute from one
address, more than the default 120.

**Not done.** No manual screen-reader session (NVDA, VoiceOver). No real Safari, Firefox or physical phone: touch was
tested with Playwright's Pixel 7 emulation.
