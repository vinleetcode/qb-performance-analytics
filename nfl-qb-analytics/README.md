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
