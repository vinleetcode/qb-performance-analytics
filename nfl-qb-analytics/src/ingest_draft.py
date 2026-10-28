"""NFL Draft ingestion via nflreadpy (no API key required -- same legitimate,
open nflverse source already used for NFL season stats).

Keeps two distinct outputs, per the project's instructions:
  - fetch_draft_raw(): the ORIGINAL draft board, every position, essentially
    as nflverse returns it (just renamed for our column conventions).
  - build_qb_prospects(): a CLEANED, QB-only, pre-draft-bio-only view, with
    height/weight/birthdate backfilled from load_players() by gsis_id.

NOTE on NFL-outcome leakage: nflverse's draft_picks rows already bundle career
NFL totals (games, pass_*, rush_*, allpro, probowls, seasons_started, car_av,
...) onto the same row as the draft-day bio fields. fetch_draft_raw() keeps
all of that (it's the honest "original" record), but build_qb_prospects()
deliberately does NOT carry those outcome columns forward -- it is meant to be
safe to use as a pre-draft model feature row. The raw table remains available
separately for anyone who wants the outcome fields for other purposes.
"""
from __future__ import annotations

import pandas as pd

SOURCE_ID = 'nflverse_draft_picks'
DEFAULT_SEASONS = list(range(2000, 2027))  # 2000-2026 inclusive, per the task's ask


def fetch_draft_raw(seasons: list[int] | None = None) -> pd.DataFrame:
    import nflreadpy as nfl
    seasons = seasons if seasons is not None else DEFAULT_SEASONS
    df = nfl.load_draft_picks(seasons=seasons).to_pandas()
    df = df.dropna(subset=['season', 'pick']).copy()

    rename = {
        'cfb_player_id': 'cfb_player_id_slug',
        'position': 'position_listed',
        'age': 'age_at_draft',
        'to': 'career_last_season',
        'car_av': 'career_av',
        'games': 'nfl_games_played_career',
        'pass_completions': 'career_pass_completions',
        'pass_attempts': 'career_pass_attempts',
        'pass_yards': 'career_pass_yards',
        'pass_tds': 'career_pass_tds',
        'pass_ints': 'career_pass_ints',
        'rush_atts': 'career_rush_attempts',
        'rush_yards': 'career_rush_yards',
        'rush_tds': 'career_rush_tds',
    }
    df = df.rename(columns=rename)
    df['source_id'] = SOURCE_ID
    df['hof'] = df['hof'].astype('boolean') if 'hof' in df.columns else None

    keep = ['source_id', 'season', 'round', 'pick', 'team', 'gsis_id', 'pfr_player_id',
            'cfb_player_id_slug', 'pfr_player_name', 'position_listed', 'college',
            'age_at_draft', 'hof', 'career_last_season', 'allpro', 'probowls',
            'seasons_started', 'career_av', 'nfl_games_played_career',
            'career_pass_completions', 'career_pass_attempts', 'career_pass_yards',
            'career_pass_tds', 'career_pass_ints', 'career_rush_attempts',
            'career_rush_yards', 'career_rush_tds']
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)


def fetch_bio_backfill(seasons: list[int] | None = None) -> pd.DataFrame:
    """gsis_id -> height/weight/birthdate/college_name, from nflreadpy's player
    roster table (covers far more players than the combine-only table)."""
    import nflreadpy as nfl
    seasons = seasons if seasons is not None else DEFAULT_SEASONS
    df = nfl.load_players().to_pandas()
    df = df[df['draft_year'].isin(seasons) & df['gsis_id'].notna()].copy()
    df = df.rename(columns={'height': 'height_inches', 'weight': 'weight_lbs', 'birth_date': 'birthdate'})
    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce').dt.date
    return df[['gsis_id', 'height_inches', 'weight_lbs', 'birthdate']].drop_duplicates('gsis_id')


def build_qb_prospects(raw_df: pd.DataFrame, bio_df: pd.DataFrame) -> pd.DataFrame:
    """QB-only, pre-draft-bio-only cleaned prospect rows. No NFL career-outcome
    columns -- see module docstring."""
    qb = raw_df[raw_df['position_listed'].eq('QB')].copy()
    qb = qb.merge(bio_df, on='gsis_id', how='left')
    qb = qb.rename(columns={'season': 'draft_year', 'team': 'nfl_team', 'pfr_player_name': 'player_name'})

    # A player_name is required for downstream matching; college is the other
    # half of the matching key, so a missing college also makes a row
    # unusable for CFB matching (still kept in the raw table, just not here).
    qb = qb.dropna(subset=['player_name', 'college']).copy()

    keep = ['source_id', 'draft_year', 'round', 'pick', 'nfl_team', 'player_name', 'gsis_id',
            'pfr_player_id', 'cfb_player_id_slug', 'college', 'position_listed',
            'height_inches', 'weight_lbs', 'birthdate']
    qb = qb[keep].drop_duplicates(subset=['draft_year', 'pick'])
    return qb.reset_index(drop=True)
