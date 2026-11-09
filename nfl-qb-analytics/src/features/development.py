"""College development trajectory and age/experience features (one row per prospect).

Design rules (from the Phase-3 brief):
  - Efficiency trends and volume trends are SEPARATE features: more attempts is
    more opportunity, not evidence of improvement.
  - Small samples: a season enters efficiency trends only with >= MIN_TREND_ATTEMPTS;
    slopes need >= 2 such seasons and are attempt-weighted least squares.
  - Transfers: seasons stay attached to the school played for; trends run across
    schools in chronological order and n_schools/transferred flag the move.
  - Left truncation: CFBD coverage starts in 2004, so a prospect whose first
    observed season is 2004 may have earlier, unobserved seasons ->
    career_possibly_truncated=1 and career totals are lower bounds.
  - Ages come only from a recorded birthdate (never from class year); missing
    birthdates stay missing.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import metrics

MIN_TREND_ATTEMPTS = 100      # season must have this many attempts to enter efficiency trends
PRODUCTIVE_MIN_ATTEMPTS = 150  # productive season: enough volume to evaluate ...
CFBD_FIRST_SEASON = 2004
PRIMARY_PASSER_SHARE = 0.5    # led team in attempts in >= half of team games -> primary-passer season (proxy)


def weighted_slope(x, y, w) -> float:
    """Attempt-weighted least-squares slope of y on x. NaN if < 2 usable points
    or no variation in x."""
    x, y, w = (np.asarray(a, dtype=float) for a in (x, y, w))
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(w)) & (w > 0)
    if m.sum() < 2:
        return np.nan
    x, y, w = x[m], y[m], w[m]
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    sxx = np.sum(w * (x - xm) ** 2)
    if sxx == 0:
        return np.nan
    return float(np.sum(w * (x - xm) * (y - ym)) / sxx)


def age_in_years(birthdate, on_date) -> float:
    if birthdate is None or on_date is None or pd.isna(birthdate) or pd.isna(on_date):
        return np.nan
    b = pd.Timestamp(birthdate)
    d = pd.Timestamp(on_date)
    return float((d - b).days / 365.2425)


def _career_rates(g: pd.DataFrame) -> dict:
    s = g[['completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions']].sum(min_count=1)
    att = s['attempts']
    return {
        'cmp_pct': float(metrics.completion_pct(s['completions'], att)),
        'ypa': float(metrics.yards_per_attempt(s['passing_yards'], att)),
        'td_pct': float(metrics.td_pct(s['passing_tds'], att)),
        'int_pct': float(metrics.int_pct(s['interceptions'], att)),
        'rating': float(metrics.ncaa_passer_rating(s['completions'], att, s['passing_yards'],
                                                   s['passing_tds'], s['interceptions'])),
    }


def prospect_development(seasons: pd.DataFrame, season_rating_medians: pd.Series) -> pd.DataFrame:
    """`seasons`: prospect_college_seasons-shaped frame (one row per prospect-season,
    pre-draft seasons only). `season_rating_medians`: index=season, value = median
    NCAA rating of all CFBD passers with >= PRODUCTIVE_MIN_ATTEMPTS that season
    (the bar a 'productive' season must clear)."""
    out = []
    for pid, g in seasons.sort_values('season').groupby('qb_draft_prospect_id'):
        g = g.copy()
        g['att0'] = g['attempts'].fillna(0)
        rec: dict = {'qb_draft_prospect_id': pid}
        q = g[g['att0'] >= MIN_TREND_ATTEMPTS]
        rec['n_trend_seasons'] = len(q)
        x = q['season'] - g['season'].min()
        for col, name in (('ncaa_passer_rating', 'rating'), ('yards_per_attempt', 'ypa'),
                          ('completion_pct', 'cmp_pct'), ('td_pct', 'td_pct'), ('int_pct', 'int_pct'),
                          ('opp_adj_ypa', 'opp_adj_ypa')):
            rec[f'dev_slope_{name}'] = weighted_slope(x, q[col], q['attempts'])
        # volume trajectory (explicitly NOT ability)
        rec['dev_slope_attempts'] = weighted_slope(g['season'] - g['season'].min(), g['att0'],
                                                   np.ones(len(g)))
        rec['dev_slope_rush_yards'] = weighted_slope(g['season'] - g['season'].min(),
                                                     g['rushing_yards'].fillna(0), np.ones(len(g)))

        final = g.iloc[-1]
        prior = g.iloc[:-1]
        prior_q = prior[prior['att0'] >= MIN_TREND_ATTEMPTS]
        if len(prior_q) and final['att0'] >= MIN_TREND_ATTEMPTS:
            prev = prior_q.iloc[-1]
            rec['dev_yoy_rating_final'] = final['ncaa_passer_rating'] - prev['ncaa_passer_rating']
            rec['dev_yoy_ypa_final'] = final['yards_per_attempt'] - prev['yards_per_attempt']
            rec['dev_yoy_cmp_pct_final'] = final['completion_pct'] - prev['completion_pct']
            rec['dev_yoy_td_pct_final'] = final['td_pct'] - prev['td_pct']
            rec['dev_yoy_int_pct_final'] = final['int_pct'] - prev['int_pct']
            pr = _career_rates(prior_q)
            rec['dev_final_vs_prior_rating'] = final['ncaa_passer_rating'] - pr['rating']
            rec['dev_final_vs_prior_ypa'] = final['yards_per_attempt'] - pr['ypa']
        rec['dev_final_vs_prior_attempts'] = (final['att0'] - prior['att0'].mean()) if len(prior) else np.nan
        if len(prior) and final.get('rushing_attempts') is not None:
            rec['dev_final_vs_prior_rush_yards'] = (final['rushing_yards'] if pd.notna(final['rushing_yards']) else 0) \
                - prior['rushing_yards'].fillna(0).mean()

        med = g['season'].map(season_rating_medians)
        productive = (g['att0'] >= PRODUCTIVE_MIN_ATTEMPTS) & (g['ncaa_passer_rating'] >= med)
        rec['n_productive_seasons'] = int(productive.sum())

        # age / experience
        rec['n_college_seasons_observed'] = int(g['season'].nunique())
        rec['career_pass_attempts'] = float(g['att0'].sum())
        rec['career_pass_yards'] = float(g['passing_yards'].fillna(0).sum())
        rec['career_pass_tds'] = float(g['passing_tds'].fillna(0).sum())
        rec['career_interceptions'] = float(g['interceptions'].fillna(0).sum())
        rec['career_rush_attempts'] = float(g['rushing_attempts'].fillna(0).sum())
        rec['career_rush_yards'] = float(g['rushing_yards'].fillna(0).sum())
        rec['career_rush_tds'] = float(g['rushing_tds'].fillna(0).sum())
        cr = _career_rates(g)
        rec.update({f'career_{k}': v for k, v in cr.items()})
        rec['career_games_with_pass_attempt'] = float(g['games_with_pass_attempt'].sum(min_count=1)) \
            if g['games_with_pass_attempt'].notna().any() else np.nan
        share = g['games_as_leading_passer'] / g['team_games_played']
        rec['seasons_as_primary_passer'] = int((share >= PRIMARY_PASSER_SHARE).sum()) \
            if g['games_as_leading_passer'].notna().any() else np.nan
        schools = [s for s in g['team'].dropna() for s in str(s).split('; ')]
        rec['n_schools'] = len(dict.fromkeys(schools))
        rec['transferred'] = int(rec['n_schools'] > 1)
        rec['career_possibly_truncated'] = int(g['season'].min() <= CFBD_FIRST_SEASON)
        rec['first_college_season'] = int(g['season'].min())
        rec['final_college_season'] = int(final['season'])
        out.append(rec)
    return pd.DataFrame(out)


def prospect_ages(prospects: pd.DataFrame, draft_dates: dict, final_seasons: pd.Series) -> pd.DataFrame:
    """age_at_draft from birthdate and the first day of the draft; age in final
    college season as of Sept 1 of that season (season-start reference). NaN
    whenever birthdate is missing -- never estimated from class year."""
    rows = []
    for _, p in prospects.iterrows():
        dd = draft_dates.get(str(int(p['draft_year'])))
        fs = final_seasons.get(p['qb_draft_prospect_id'])
        rows.append({
            'qb_draft_prospect_id': p['qb_draft_prospect_id'],
            'age_at_draft': age_in_years(p.get('birthdate'), dd),
            'age_final_college_season': age_in_years(p.get('birthdate'), date(int(fs), 9, 1)) if pd.notna(fs) else np.nan,
        })
    return pd.DataFrame(rows)
