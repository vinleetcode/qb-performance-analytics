"""NFL Combine athletic profile.

Combine missingness is informative and systematic (top prospects often skip
drills; some prospects are not invited), so:
  * a skipped drill stays NULL -- never imputed as a slow/poor result here;
  * explicit indicators (combine_record_found, combine_did_<drill>) are emitted
    so models can learn from participation separately from performance;
  * the whole block is its own feature group ('athletic_combine') that can be
    dropped or modelled separately.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.match_prospects import normalize_name

DRILLS = ['forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']


def parse_height(ht) -> float:
    """'6-4' -> 76.0; already-numeric inches pass through; anything else -> NaN."""
    if ht is None or (isinstance(ht, float) and np.isnan(ht)):
        return np.nan
    if isinstance(ht, (int, float)):
        return float(ht)
    m = re.fullmatch(r'\s*(\d)\s*-\s*(\d{1,2}(?:\.\d+)?)\s*', str(ht))
    if not m:
        return np.nan
    return int(m.group(1)) * 12 + float(m.group(2))


def standardize_combine(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={'season': 'combine_season', 'pfr_id': 'pfr_id', 'cfb_id': 'cfb_id_slug',
                             'pos': 'position', 'wt': 'weight_lbs', 'draft_ovr': 'draft_overall'}).copy()
    df['height_inches'] = df['ht'].map(parse_height)
    for c in DRILLS + ['weight_lbs']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    keep = ['combine_season', 'pfr_id', 'cfb_id_slug', 'player_name', 'position', 'school', 'height_inches',
            'weight_lbs'] + DRILLS + ['draft_year', 'draft_overall']
    return df[keep].drop_duplicates(['combine_season', 'player_name', 'school', 'position']).reset_index(drop=True)


def join_combine_to_prospects(prospects: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per prospect with combine fields + `combine_join_method`
    ('pfr_id' | 'cfb_slug' | 'name_year_qb' | None). A combine row is only
    accepted if its combine season equals the draft year (a player can attend
    only the combine before his draft)."""
    out = []
    by_pfr = {k: g for k, g in combine.dropna(subset=['pfr_id']).groupby('pfr_id')}
    by_slug = {k: g for k, g in combine.dropna(subset=['cfb_id_slug']).groupby('cfb_id_slug')}
    combine = combine.assign(_n=combine['player_name'].map(normalize_name))
    for _, p in prospects.iterrows():
        cand, method = None, None
        for key, idx, name in ((p.get('pfr_player_id'), by_pfr, 'pfr_id'),
                               (p.get('cfb_player_id_slug'), by_slug, 'cfb_slug')):
            if isinstance(key, str) and key in idx:
                c = idx[key]
                c = c[c['combine_season'] == p['draft_year']]
                if len(c) == 1:
                    cand, method = c.iloc[0], name
                    break
        if cand is None:
            c = combine[(combine['combine_season'] == p['draft_year']) & (combine['position'] == 'QB')
                        & (combine['_n'] == normalize_name(p['player_name']))]
            if len(c) == 1:
                cand, method = c.iloc[0], 'name_year_qb'
        rec = {'qb_draft_prospect_id': p['qb_draft_prospect_id'], 'combine_join_method': method,
               'combine_record_found': int(cand is not None)}
        for col in ['height_inches', 'weight_lbs'] + DRILLS:
            val = cand[col] if cand is not None else np.nan
            rec[f'combine_{col}'] = val
            if col in DRILLS:
                rec[f'combine_did_{col}'] = (int(pd.notna(val)) if cand is not None else np.nan)
        out.append(rec)
    return pd.DataFrame(out)
