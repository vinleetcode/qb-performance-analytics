# Production performance (Phase 9, 2026-11-19)

All numbers below were measured in this session on the development machine:
* Windows 11, Python 3.14.6, Node 24.18
* the pinned API dependencies from `requirements-api.txt`
* the full Python test suite running in parallel during some measurements, so they are noisy upper bounds

**Nothing has been measured on Render or Vercel yet.** The site is not deployed. Render Free gives 0.1 CPU, so
CPU-bound steps there will be slower than here. Measure them with `deploy/smoke_test.py` after deployment.

## What was slow and what changed

Profiling the first forecast showed three costs:

| Cost | Cause | Change | Effect on output |
|---|---|---|---|
| `profiles.joblib` loaded **twice** (the GSIS lookup and the engine), about 10-19 s each here | Two independent `joblib.load` calls | `api.bundle.load_serving`: one shared, locked load per process | None (the same objects) |
| Each load slow | joblib's pure-Python unpickler over about 5,600 small arrays and 2,500 pandas Series | `python -m api.bundle optimize` (image build step, after `verify`) re-saves the same objects with the C pickle. They are used only if the copy's recorded source hash equals the manifest hash and its own hash matches; otherwise joblib is used | None: `test_fast_profiles_give_identical_outputs` compares detail, both forecast modes, a hypothetical pick and comparisons, byte for byte |
| Committed memory about 1.27 GB | Multi-threaded BLAS reserving per-thread buffers | `OMP/OPENBLAS/MKL_NUM_THREADS=1` in the image | None (the same arithmetic). Suits 0.1 CPU |
| Engine load on the first visitor's request | Lazy loading | `QB_API_WARMUP=true` in the image: a background thread loads the engine at boot. Requests that arrive meanwhile wait on the existing load lock | None |

Remaining unavoidable cost: importing scikit-learn and SciPy, which the pickled models need. That was about 10 s on
this Windows machine, where antivirus scanning inflates first imports.

## Measurements (local, API process, `RATE_LIMIT_PER_MINUTE=0`)

| Step | Before (Phase 8 code) | After |
|---|---|---|
| Process start to `/api/health` answering | 6.4 s | 2.7 s |
| First search | 2,167 ms | 574 ms |
| First prospect detail | 8,717 ms | 1,822 ms |
| First forecast, lazy engine load | 22,856 ms | 11,787 ms |
| First forecast after warm-up finished (container-equivalent run) | n/a | 163 ms (engine warm 6.8 s after start) |
| Repeat forecast (in-memory cache) | 37 ms | 12 ms |
| Uncached forecast, engine warm (average of 20 prospects) | 583 ms | 472 ms |
| Comparisons | 146 ms | 104 ms |
| Historical backtest, first / repeat | 132 / 12 ms | 95 / 25 ms |
| Model performance summary | 45 ms | 14 ms |
| Forecast response size | 15.5 KB | 15.5 KB (unchanged) |
| Working set after first forecast / end of run | 273 / 275 MB | 253 / 257 MB |
| Committed (private) memory, end of run | 1,266 MB | 263 MB |
| CPU seconds consumed by the first forecast | 20.9 s | 11.6 s |

Render Free allows 512 MB. The measured footprint of about 260 MB leaves headroom. Linux RSS will differ somewhat from
the Windows working set; check the Render metrics page after deploying.

Expected value of passing TDs for `qb_226`, pre-draft: 17.841732825648673 before and after. The container-equivalent
run and the live-vs-local comparison in the smoke test give the same value.

## Free-tier latency, honestly

* **Waking a sleeping service** takes about a minute, according to Render's documentation. Not measured.
* **Engine warm-up at 0.1 CPU:** about 6.8 s of wall time and about 11 s of CPU here. If the 0.1-CPU quota is enforced
  strictly, the warm-up after a wake could take a minute or more on Render. This is an estimate, not a measurement.
* **Uncached forecasts on a warm engine:** about 0.5 s of CPU here, so several seconds each at 0.1 CPU.
* **Caching:** forecasts are deterministic, so they are cached in process memory (512 entries) and, in browsers, for
  an hour by React Query. Successful API responses also carry `Cache-Control: public, max-age=300`.

## Frontend

Production build (`next build`, Next.js 16.4.6, Turbopack): all 13 routes are static or partially prerendered. The
prospect report streams its dynamic part.

Initial JavaScript per page (gzip, sum of the chunks referenced by the prerendered HTML):

| Page | Before | After |
|---|---|---|
| `/` (homepage) | about 358 KB (derived: today's 246 KB plus the 112 KB Recharts chunk it used to load) | **246 KB, no Recharts** |
| `/credits` | n/a | 226 KB |
| `/methodology` | n/a | 359 KB (charts on the page) |
| `/backtesting` | n/a | 378 KB (charts on the page) |
| `/prospects/[id]` | n/a | 371 KB (charts on the page) |

Fix: the homepage imported two plain values (`CLASS_INFO`, `avgSkill`) from chart modules, which pulled in Recharts.
They moved to `src/lib/backtest.ts`, and the chart modules re-export them.

Images are served through `next/image` (AVIF/WebP, 30-day cache), with six source photos. Vercel Hobby includes 5,000
image transformations a month, which is far more than six photos at a few widths need. Fonts are self-hosted by
`next/font`.

**Not measured:** Lighthouse scores and field Core Web Vitals (LCP, CLS, INP). There was no Lighthouse run in this
session, and field data needs the public URL. Run PageSpeed Insights on the Vercel URL after deployment.
