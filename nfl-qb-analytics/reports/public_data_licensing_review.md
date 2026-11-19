# Public data and image licensing review

Reviewed 2026-11-11 from the sources' current published terms, read on that date. This is a project review, **not legal
advice**. The decision to deploy publicly is the owner's.

## Sources and what their terms say

### CollegeFootballData.com: [terms](https://collegefootballdata.com/terms), effective 12 August 2026

| Question | What the terms say |
|---|---|
| Use in a website | Allowed: "use API Data in free or paid websites, applications, dashboards ..." (Sec. 4). |
| Showing statistics | Allowed: "display reasonable portions of factual API Data as part of a larger product" (Sec. 4). |
| Publishing model predictions | Allowed: "publish and commercialize independently created analyses, models, ... predictions, projections ..." (Sec. 4). Derived outputs must not reproduce "a substantial portion of the Raw Data". |
| Commercial use | "Commercial use is permitted." |
| Redistribution | Prohibited: no providing API data "as a standalone dataset or bulk download", no "raw feed, public database mirror, proxy, substitute API", and no giving others "direct or programmatic access to stored raw API responses" (Sec. 5). |
| Attribution | "Appreciated but not required"; recommended text "Data provided by CollegeFootballData.com." (Sec. 6). |
| API key | Must stay server-side, never in a public repository or browser bundle (Sec. 2 and 5). |
| Names, logos, likenesses | Not granted: no rights in team or conference marks, logos, or "player names or likenesses" (Sec. 8). |

### nflverse: [nflverse-data](https://github.com/nflverse/nflverse-data)

* The repository's data releases are licensed **CC BY 4.0** ([LICENSE.md](https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md)).
* CC BY 4.0 requires attribution and a licence notice, and does not license publicity or trademark rights (Sec. 2(b)).
* nflreadr states that "NFL data accessed by this package belong to their respective owners, and are governed by their
  terms of use."
* Several datasets originate with third parties: snap counts and advanced stats from Pro Football Reference, Next Gen
  Stats, and FTN charting. The `draft_picks` and `combine` releases are built by nflverse's PFR scraper workflows (an
  inference from the workflow names; the source terms are not stated on those pages).

### Other local sources

* **Wikipedia** (cached scheme descriptions used for offensive-scheme labels). The text is CC BY-SA, but the site displays
  only derived labels, not the text itself.
* **Photographs:** Wikimedia Commons, catalogued in `frontend/ASSET_LICENSES.md`.

## What the site does, category by category

| Category | What the site shows | Assessment |
|---|---|---|
| 1. Raw data redistribution | Nothing downloadable. The serving bundle is never in the frontend, `public/` or Git. | Compliant, but see the API point below |
| 2. Factual display | Per-player college season lines, final-season statistics, draft round and pick, NFL starts and totals for comparisons and backtests | Reasonable portions inside a larger product: allowed by CFBD Sec. 4. nflverse data needs CC BY attribution (added to the footer and `/credits`) |
| 3. Model predictions | Draft probabilities, round and pick ranges, start probabilities, five-season forecasts, comparisons | Independently generated derived outputs: allowed by CFBD Sec. 4. Not a reproduction of the raw data |
| 4. Photography | Six Wikimedia Commons photos: four CC BY, two public domain | Licensed; attribution shown on each image and on `/credits` |
| 5. Trademarks and publicity | Player names as factual identifiers; no player photos or likenesses; no league, team or school logos; non-affiliation statement in the footer | Names used descriptively; no endorsement implied. Not granted by any of the licences, so kept to factual use |

## Open issues

1. **The public API could act as a bulk mirror. Blocking issue for public deployment as currently configured.**
   * The FastAPI backend serves paginated search over all 1,279 prospects, and `/api/prospects/{id}` returns each player's
     college season lines. Anyone could script those endpoints and rebuild the CFBD-derived dataset. That comes close to
     the CFBD Sec. 5 prohibitions ("public database mirror", "programmatic access to stored raw API responses").
   * Mitigations before launch:
     * keep `QB_API_DOCS_ENABLED=false` in production;
     * apply strict platform rate limits on top of the per-process limiter;
     * cap pagination depth;
     * consider removing `college_seasons` from the public detail response, keeping the final-season profile only;
     * optionally put the API behind a Vercel proxy with bot protection.
   * Which mitigations are enough is the owner's call, possibly after asking CFBD.
2. **Third-party origin of some nflverse datasets. Unresolved.**
   * Draft picks, Combine measurements and career totals appear to originate from Pro Football Reference via nflverse.
   * Individual facts (a player's draft slot, his 40 time) are not copyrightable in the US, and the site shows only
     reasonable portions alongside its own models. But Sports Reference's terms were not reviewed.
   * Until they are, treat displaying Combine measurements and NFL career totals publicly as unresolved.
3. **CFBD attribution.** Not required, but added ("Data provided by CollegeFootballData.com") in the footer and on `/credits`.
4. **Screenshots in Git.** `frontend/docs/screenshots/` and `frontend/public/previews/` show real players' names,
   statistics and forecasts. They fall under category 2 or 3 and are fine wherever the site itself would be. Publish them
   only once issues 1 and 2 are settled.

## Status

| Material | Public deployment status |
|---|---|
| Model predictions, evaluation results, methodology, case study | Cleared by the reviewed terms, with attribution |
| Photographs and other assets | Cleared (see `ASSET_LICENSES.md`) |
| Factual CFBD data shown on pages | Cleared as reasonable display |
| Public JSON API exposing all college season lines | **Blocked** until issue 1 is mitigated |
| PFR-origin fields (Combine measurements, NFL career totals) | **Unresolved** (issue 2): review the Sports Reference terms first |

Nothing has been deployed, so nothing has yet been published under these terms.


## Update for the free public deployment (2026-11-19)

Re-read on 2026-11-19: CFBD terms (effective 12 Aug 2026), the nflverse-data licence, and the Sports Reference terms of
use (last updated 19 May 2023) and data-use page. This is not legal advice.

### What changed in the deployment

| Earlier open issue | Mitigation now in place |
|---|---|
| 1. Public API as a bulk mirror (CFBD Sec. 5: no "public database mirror ... substitute API") | `/docs` and `/openapi.json` are off in the image (`QB_API_DOCS_ENABLED=false`); search pages are capped at 20, the most the site requests; the per-client limit is 120 requests/minute and now sees real visitor addresses behind Render's proxy (`FORWARDED_ALLOW_IPS`), checked by test; no endpoint lists or downloads datasets; the bundle archive is in a private repository, never served |
| 2. PFR-origin fields (draft slots, Combine measurements) | Credit added on `/credits`; the site shows them as individual facts on player pages, not en masse |
| 3. CFBD attribution | Kept in the footer and on `/credits` ("Data provided by CollegeFootballData.com") |
| nflverse CC BY 4.0 | Attribution and licence link in the footer and on `/credits` |
| Photos | Six Wikimedia Commons photos (CC BY 3.0/4.0 and public domain), credited on each image and on `/credits`; `public/images/licenses.json` records the source, author and licence of each |
| CFBD key | Not used by the API or frontend; not in any committable file or the frontend build (scanned) |

### What the terms say, briefly

* **CFBD:**
  - Sec. 4 allows using API data "in free or paid websites" and displaying "reasonable portions of factual API Data as
    part of a larger product", and allows publishing independently created models and predictions.
  - Sec. 5 forbids bulk downloads, standalone datasets, a "public database mirror, proxy, substitute API", and giving
    others "programmatic access to stored raw API responses".
  - The terms define none of "reasonable", "substantial" or "raw". Per-player pages backed by a rate-limited JSON
    endpoint are a display inside a larger product. Whether the endpoint itself could be called a substitute API is
    not settled by the text.
* **nflverse:** CC BY 4.0. Display is allowed with attribution, the licence notice and a link. Publicity and trademark
  rights are not licensed.
* **Sports Reference:**
  - The data-use page says "facts cannot be copyrighted, so you are free to reuse facts found on this site". Its terms
    welcome using data from individual pages with credit.
  - It forbids scraping (this project never accessed SR sites) and building a database that substitutes for SR.
  - Since May 2023, item 11 of the terms prohibits using SR content for "training ... artificial intelligence models",
    including "machine learning methods used to predict".
  - The draft and Combine data reached this project through nflverse's CC BY 4.0 releases, not from SR. The terms do
    not say how they apply to data obtained through third parties.

### Status for launch

| Material | Status |
|---|---|
| Model predictions, evaluation results, methodology | Allowed by the reviewed terms (CFBD Sec. 4; nflverse CC BY with attribution) |
| Photos and brand | Cleared, attributed |
| Per-player college season lines and final-season profile | Allowed as reasonable display (CFBD Sec. 4), with the anti-mirroring mitigations above. **Owner decision:** whether those mitigations are enough. The conservative alternative is to drop `college_seasons` from the public detail response (a small API change; the report would lose its season table) |
| Draft slot and Combine measurements (displayed facts) | Individual facts with credit; SR's data-use page treats facts as freely reusable. Low residual risk |
| Models trained partly on nflverse draft/Combine fields that nflverse credits to PFR | **Unresolved, owner decision.** SR's item 11 bars ML use of SR content. Whether it reaches data obtained under nflverse's CC BY 4.0 licence is not addressed. This affects the trained models, not a dataset the site publishes. Asking SR (and nflverse) is the only way to settle it |
| Bulk data, dataset downloads, the serving bundle | Not published |

Nothing has been deployed, so nothing has yet been published under these terms.
