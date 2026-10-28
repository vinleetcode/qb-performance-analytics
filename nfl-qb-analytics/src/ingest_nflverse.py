"""Ingestion for the only source this pipeline can legitimately reach for bulk
historical passing data: nflverse, via the nflreadpy package (1999-2025).
See config/sources.json for the verified coverage and license notes, and
src/historical_sources.py for why 1920-1998 and the 1960-1969 AFL cannot be
filled in from here.
"""
from __future__ import annotations

import pandas as pd

SOURCE_ID = 'nflverse_player_stats'
LEAGUE_ID = 'NFL'
MIN_SEASON = 1999


def fetch_season_totals(seasons: list[int]) -> pd.DataFrame:
    """One row per player-season (already deduplicated across teams by nflverse
    for `summary_level='reg'`), including every player with at least one pass
    attempt -- not pre-filtered to position == 'QB', since some historical and
    even some modern passers are not officially rostered as quarterbacks."""
    import nflreadpy as nfl
    raw = nfl.load_player_stats(seasons=seasons, summary_level='reg').to_pandas()
    raw = raw[raw.season_type.eq('REG')].copy()
    df = raw[raw['attempts'].fillna(0) > 0].copy()
    # A handful of play-by-play rows have no resolvable player id/name (e.g. laterals,
    # miscredited plays); these can't be attached to a player identity, so drop them
    # rather than inventing a player record.
    df = df.dropna(subset=['player_id', 'player_display_name'])

    df = df.rename(columns={
        'player_id': 'source_player_id',
        'player_display_name': 'display_name',
        'position': 'position_listed',
        'recent_team': 'primary_team',
        'passing_interceptions': 'interceptions',
    })
    df['is_qb_listed'] = df['position_listed'].eq('QB')
    df['league_id'] = LEAGUE_ID
    df['season_type'] = 'REG'
    df['source_id'] = SOURCE_ID
    df['completeness_flag'] = 'complete'
    df['completeness_notes'] = None

    keep = ['source_player_id', 'display_name', 'position_listed', 'is_qb_listed',
            'season', 'league_id', 'season_type', 'source_id', 'primary_team', 'games',
            'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
            'sacks_suffered', 'sack_yards_lost', 'carries', 'rushing_yards', 'rushing_tds',
            'rushing_fumbles_lost', 'completeness_flag', 'completeness_notes']
    return df[keep].reset_index(drop=True)


def fetch_team_stints(seasons: list[int]) -> pd.DataFrame:
    """Per-player-season breakdown of games/attempts by team, built from
    week-level data, so multi-team seasons preserve team history without
    double-counting the season totals (which come from fetch_season_totals,
    already aggregated once by nflverse)."""
    import nflreadpy as nfl
    weekly = nfl.load_player_stats(seasons=seasons, summary_level='week').to_pandas()
    weekly = weekly[weekly.season_type.eq('REG')].copy()
    weekly = weekly[weekly['attempts'].fillna(0).ge(0) & weekly['player_id'].notna()]
    weekly = weekly[weekly.groupby(['player_id', 'season'])['attempts'].transform('sum') > 0]

    stints = (weekly.groupby(['player_id', 'season', 'team'], as_index=False)
              .agg(games=('week', 'nunique'), attempts=('attempts', 'sum')))
    stints = stints.rename(columns={'player_id': 'source_player_id', 'team': 'team_abbr'})
    stints['source_id'] = SOURCE_ID
    return stints.reset_index(drop=True)


def fetch_teams() -> pd.DataFrame:
    import nflreadpy as nfl
    teams = nfl.load_teams().to_pandas()
    out = teams[['team_abbr', 'team_name']].drop_duplicates().rename(
        columns={'team_abbr': 'abbr', 'team_name': 'name'})
    out['league_id'] = LEAGUE_ID
    return out.reset_index(drop=True)
