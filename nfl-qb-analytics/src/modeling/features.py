"""Feature definitions for the NFL QB career-success model (MODEL_FEATURE_VERSION).

Three NESTED feature groups answer the project question "do detailed college
offensive play types improve prediction?":

  A  basic college production (box-score passing/rushing)
  B  A + opponent / team context and development (opponent-adjusted YPA,
     schedule strength, recruiting talent around the QB, age, height)
  C  B + play-type profile of the QB's FINAL college offense, computed here from
     CFBD play-by-play (data/cfbd_plays, 2016-2025 only)

Group sizes are deliberately small (6 / +5 / +5 = 16) because the eligible
sample is ~50 quarterbacks with ~19 positive outcomes; hundreds of features
would be fit noise. Every feature is known before the NFL draft:
  * college season stats -> after the final college season (incl. bowls, which
    are played before the late-April draft),
  * age at draft -> birthdate is known before the draft,
  * height -> combine measurement (known by mid-March) else college roster.
Draft slot (round/pick) is NOT a college feature; it is used only by a clearly
labelled REFERENCE model (`DRAFT_REFERENCE`), never mixed into A/B/C.

Play-type caveat: CFBD play rows carry no passer id in our slim cache, so the
play-type profile describes the QB's team offense in his final season (in which
the drafted QB was almost always the primary passer). CFBD play types do not
separate designed runs from scrambles and do not identify RPOs, play-action,
screens or formations, so none of those are claimed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.schemes import PASS_TYPES, RUSH_TYPES, is_garbage_time

MODEL_FEATURE_VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parents[2]
PLAYS_DIR = ROOT / 'data' / 'cfbd_plays'
MIN_PBP_PLAYS = 300          # same reliability floor as the scheme database
EXPLOSIVE_PASS_YARDS = 20
PBP_SEASONS = (2016, 2025)   # seasons present in data/cfbd_plays

GROUP_A_BASIC = ['final_completion_pct', 'final_yards_per_attempt', 'final_td_pct', 'final_int_pct',
                 'log_career_pass_attempts', 'career_rush_yards']
GROUP_B_CONTEXT = ['final_opp_adj_ypa', 'final_sos_pass_defense', 'talent_recruiting_4yr_pctile',
                   'age_at_draft', 'height_inches']
GROUP_C_PLAYTYPE = ['pt_standard_down_pass_rate', 'pt_passing_down_pass_rate', 'pt_explosive_pass_rate',
                    'pt_sack_rate', 'pt_plays_per_game']
DRAFT_REFERENCE = ['log_overall_pick']

FEATURE_SETS = {
    'A_basic': GROUP_A_BASIC,
    'B_context': GROUP_A_BASIC + GROUP_B_CONTEXT,
    'C_playtype': GROUP_A_BASIC + GROUP_B_CONTEXT + GROUP_C_PLAYTYPE,
    'draft_reference': DRAFT_REFERENCE,
}

# Columns that must never enter a college-only (A/B/C) model.
POST_DRAFT_COLUMNS = {'round', 'pick', 'overall_pick', 'log_overall_pick', 'nfl_team'}

# (feature, group, source, calculation, availability cutoff)
_DICT_ROWS = [
    ('final_completion_pct', 'A_basic', 'CFBD season stats; 2004-2008 rebuilt from CFBD box scores',
     'completions / attempts, final college season', 'end of final college season'),
    ('final_yards_per_attempt', 'A_basic', 'CFBD season stats / box scores', 'passing yards / attempts, final season',
     'end of final college season'),
    ('final_td_pct', 'A_basic', 'CFBD season stats / box scores', 'passing TDs / attempts, final season',
     'end of final college season'),
    ('final_int_pct', 'A_basic', 'CFBD season stats / box scores', 'interceptions / attempts, final season',
     'end of final college season'),
    ('log_career_pass_attempts', 'A_basic', 'CFBD season stats / box scores',
     'log(1 + total college pass attempts); stands in for career yards/TDs, which are >0.9 correlated with it',
     'end of final college season'),
    ('career_rush_yards', 'A_basic', 'CFBD season stats / box scores',
     'total college rushing yards (NCAA convention: sacks count as rushes)', 'end of final college season'),
    ('final_opp_adj_ypa', 'B_context', 'CFBD per-game box scores',
     'final-season YPA adjusted by each opponent\'s leave-one-out pass-defense YPA allowed',
     'end of final college season'),
    ('final_sos_pass_defense', 'B_context', 'CFBD per-game box scores',
     'mean leave-one-out pass-defense YPA allowed by final-season opponents (lower = tougher)',
     'end of final college season'),
    ('talent_recruiting_4yr_pctile', 'B_context', 'CFBD team recruiting rankings',
     'percentile of the final team\'s 4-year average recruiting points (supporting-cast talent proxy)',
     'end of final college season'),
    ('age_at_draft', 'B_context', 'nflverse draft picks birthdate', 'age in years on draft day',
     'before draft day (birthdate known)'),
    ('height_inches', 'B_context', 'nflverse combine; fallback CFBD roster height',
     'combine height if measured, else college roster height', 'combine (mid-March) / college roster'),
    ('pt_standard_down_pass_rate', 'C_playtype', 'CFBD /plays (2016-2025)',
     'final team-season share of dropbacks on standard downs (1st; 2nd&<=7; 3rd/4th&<=4), garbage time excluded',
     'end of final college season'),
    ('pt_passing_down_pass_rate', 'C_playtype', 'CFBD /plays (2016-2025)',
     'share of dropbacks on passing downs (all other downs), garbage time excluded', 'end of final college season'),
    ('pt_explosive_pass_rate', 'C_playtype', 'CFBD /plays (2016-2025)',
     f'share of dropbacks gaining >= {EXPLOSIVE_PASS_YARDS} yards (vertical/explosive passing tendency)',
     'end of final college season'),
    ('pt_sack_rate', 'C_playtype', 'CFBD /plays (2016-2025)', 'sacks / dropbacks, garbage time excluded',
     'end of final college season'),
    ('pt_plays_per_game', 'C_playtype', 'CFBD /plays (2016-2025)',
     'offensive scrimmage plays (pass + rush, downs 1-4) per game: tempo', 'end of final college season'),
    ('log_overall_pick', 'draft_reference', 'nflverse draft picks', 'log(overall pick number)',
     'draft day (REFERENCE model only; not a college feature)'),
]

# Candidate features considered and NOT selected, with the reason (kept in the dictionary).
NOT_SELECTED = {
    'career_pass_yards': 'collinear with log_career_pass_attempts (rho > 0.9)',
    'career_pass_tds': 'collinear with log_career_pass_attempts',
    'career_rush_tds': 'collinear with career_rush_yards',
    'final_qb_rush_share': 'overlaps career_rush_yards; kept out to hold group B at 5 features',
    'env_off_ppa': 'team efficiency; overlaps final_yards_per_attempt for the primary passer',
    'talent_composite_pctile': 'CFBD talent composite only exists from 2015; recruiting percentile covers all years',
    'dev_yoy_ypa_final': '12% missing in the eligible cohort (one-season starters); dimensionality',
    'n_college_seasons_observed': 'overlaps log_career_pass_attempts and age_at_draft',
    'combine_forty': '30% missing (QBs often skip drills); missingness is itself selection-driven',
    'combine_vertical': '28% missing', 'combine_broad_jump': '32% missing', 'combine_cone': '37% missing',
    'combine_shuttle': '37% missing', 'combine_bench': '100% missing for QBs',
    'final_sp_strength_of_schedule': '44% missing (CFBD SP+ SOS not published for every season)',
    'scheme_rep_final_spread_family': '35% missing; Wikipedia-reported labels are low confidence',
    'scheme_ver_final_spread_family': '96% missing (5 verified team-seasons)',
    'college_starts_verified': 'not available: CFBD does not publish starts; games_as_leading_passer is a proxy only',
    'injury_availability': 'not available from any ingested source (see reports/advanced_features_coverage.md)',
    'pro_day_measurements': 'not available from any ingested source',
}


def feature_dictionary(features: pd.DataFrame | None = None, cohort_mask=None) -> pd.DataFrame:
    rows = [{'feature': f, 'group': g, 'selected': True, 'source': s, 'calculation': c, 'availability_cutoff': a}
            for f, g, s, c, a in _DICT_ROWS]
    rows += [{'feature': f, 'group': 'not_selected', 'selected': False, 'source': '', 'calculation': reason,
              'availability_cutoff': ''} for f, reason in NOT_SELECTED.items()]
    d = pd.DataFrame(rows)
    d['feature_version'] = MODEL_FEATURE_VERSION
    if features is not None:
        sub = features if cohort_mask is None else features[cohort_mask]
        d['missing_rate_eligible'] = [float(sub[f].isna().mean()) if f in sub.columns else np.nan for f in d['feature']]
    return d


def load_plays(plays_dir: Path = PLAYS_DIR) -> pd.DataFrame:
    files = sorted(plays_dir.glob('plays_*.parquet'))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def team_play_type_profile(plays: pd.DataFrame, min_plays: int = MIN_PBP_PLAYS) -> pd.DataFrame:
    """Per (season, team) play-type profile. Includes postseason (bowls/CFP precede the draft).
    Rates use non-garbage-time plays; tempo uses all scrimmage plays. Teams below
    `min_plays` non-garbage plays (FCS teams seen only vs FBS opponents) are blanked."""
    cols = ['season', 'team', 'pt_n_plays', 'pt_games', 'pt_standard_down_pass_rate', 'pt_passing_down_pass_rate',
            'pt_explosive_pass_rate', 'pt_sack_rate', 'pt_plays_per_game', 'pt_reliable']
    if plays.empty:
        return pd.DataFrame(columns=cols)
    p = plays[plays['playType'].isin(PASS_TYPES | RUSH_TYPES) & plays['down'].between(1, 4)].copy()
    p['is_pass'] = p['playType'].isin(PASS_TYPES)
    p['garbage'] = is_garbage_time(p['period'], p['offenseScore'] - p['defenseScore'])
    d, dist = p['down'], p['distance']
    p['standard_down'] = (d == 1) | ((d == 2) & (dist <= 7)) | (d.isin([3, 4]) & (dist <= 4))
    p['explosive'] = p['is_pass'] & (p['yardsGained'] >= EXPLOSIVE_PASS_YARDS)
    p['sack'] = p['playType'] == 'Sack'
    tempo = (p.groupby(['season', 'offense'])
             .agg(all_plays=('is_pass', 'size'), pt_games=('gameId', 'nunique')).reset_index())
    ng = p[~p['garbage']]
    sd = ng[ng['standard_down']].groupby(['season', 'offense'])['is_pass'].mean().rename('pt_standard_down_pass_rate')
    pdn = ng[~ng['standard_down']].groupby(['season', 'offense'])['is_pass'].mean().rename('pt_passing_down_pass_rate')
    db = ng[ng['is_pass']].groupby(['season', 'offense']).agg(
        pt_explosive_pass_rate=('explosive', 'mean'), pt_sack_rate=('sack', 'mean'))
    n = ng.groupby(['season', 'offense']).size().rename('pt_n_plays')
    out = pd.concat([n, sd, pdn, db], axis=1).reset_index().merge(tempo, on=['season', 'offense'], how='left')
    out['pt_plays_per_game'] = out['all_plays'] / out['pt_games']
    out = out.rename(columns={'offense': 'team'})
    out['pt_reliable'] = out['pt_n_plays'] >= min_plays
    rate_cols = [c for c in GROUP_C_PLAYTYPE if c in out.columns]
    out.loc[~out['pt_reliable'], rate_cols] = np.nan
    out['season'] = out['season'].astype(int)
    return out.reindex(columns=cols)


def final_team_season(college_seasons: pd.DataFrame) -> pd.DataFrame:
    """The (team, season) of each prospect's final pre-draft college season."""
    f = college_seasons[college_seasons['is_final_college_season'] == True]  # noqa: E712
    f = f.sort_values(['qb_draft_prospect_id', 'attempts'], ascending=[True, False])
    out = f.drop_duplicates('qb_draft_prospect_id')[['qb_draft_prospect_id', 'team', 'season', 'conference']].rename(
        columns={'team': 'final_team', 'season': 'final_team_season', 'conference': 'final_conference'})
    out['final_fcs'] = out['final_conference'].isin(FCS_CONFERENCES)
    return out


# FCS conference names as they appear in CFBD data. Every QB in the 2017-2021 training
# cohort played his final season in FBS (CFBD play-by-play did not cover FCS-only games
# for those seasons), so FCS QBs are OUTSIDE the model's training support.
FCS_CONFERENCES = {'MVFC', 'Missouri Valley', 'Southland', 'CAA', 'Big Sky', 'OVC', 'Ohio Valley', 'SWAC', 'MEAC',
                   'Pioneer', 'Patriot', 'Ivy', 'Southern', 'Big South', 'Big South-OVC', 'NEC', 'UAC',
                   'United Athletic', 'FCS Independents'}


def add_derived_features(features: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    """Derived predictors on the Phase-3 pre-draft feature table (no target information)."""
    f = features.copy()
    f['log_career_pass_attempts'] = np.log1p(f['career_pass_attempts'])
    f['height_inches'] = f['combine_height_inches'].fillna(f['bio_college_height_inches'])
    f['height_source'] = np.where(f['combine_height_inches'].notna(), 'combine',
                                  np.where(f['bio_college_height_inches'].notna(), 'college_roster', None))
    pk = prospects[['id', 'round', 'pick']].rename(columns={'id': 'qb_draft_prospect_id', 'pick': 'overall_pick'})
    f = f.merge(pk, on='qb_draft_prospect_id', how='left')
    f['log_overall_pick'] = np.log(f['overall_pick'].astype(float))
    return f
