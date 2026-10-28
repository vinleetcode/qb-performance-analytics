"""CollegeFootballData.com ingestion. Requires CFBD_API_KEY (see src.config).

Endpoints used (confirmed against the live OpenAPI spec at
https://api.collegefootballdata.com/api-docs.json on 2026-10-30, not guessed):
  - GET /stats/player/season?year=&category=   -- long-format player-season stats
  - GET /roster?year=                           -- bulk per-season bio (height/weight)
  - GET /ratings/sp?year=                       -- team-season SP+ rating + strength of schedule

Call budget for the default 2004-2025 backfill: 22 seasons x (1 passing +
1 rushing + 1 roster + 1 sp) = 88 calls, each cached to disk by CFBDClient --
comfortably inside CFBD's published free-tier limit of 1,000 calls/month, and
a rerun makes zero new calls once cached.

/stats/player/season returns `stat` as a STRING and doesn't document the exact
set of `statType` values per category in the spec -- rather than assume a
fixed set, pivot_player_stats() pivots whatever statTypes actually come back
and maps the common ones we recognize; anything unrecognized is reported, not
silently dropped or guessed at.
"""
from __future__ import annotations

import pandas as pd

from src.cfbd_client import CFBDClient

SOURCE_ID = 'cfbd_player_season_stats'
DEFAULT_SEASONS = list(range(2004, 2026))  # 2004-2025 inclusive

# statType -> our canonical column name, per category. Keys are matched
# case-insensitively. Extend this if a run reports new "unrecognized" types.
PASSING_STAT_MAP = {
    'completions': 'completions', 'comp': 'completions', 'c': 'completions',
    'att': 'attempts', 'attempts': 'attempts',
    'yds': 'passing_yards', 'yards': 'passing_yards',
    'td': 'passing_tds', 'tds': 'passing_tds',
    'int': 'interceptions', 'ints': 'interceptions', 'interceptions': 'interceptions',
}
RUSHING_STAT_MAP = {
    'car': 'rushing_attempts', 'att': 'rushing_attempts', 'attempts': 'rushing_attempts',
    'yds': 'rushing_yards', 'yards': 'rushing_yards',
    'td': 'rushing_tds', 'tds': 'rushing_tds',
}
CATEGORY_MAPS = {'passing': PASSING_STAT_MAP, 'rushing': RUSHING_STAT_MAP}


def fetch_player_stats_long(seasons: list[int] | None, client: CFBDClient | None = None) -> pd.DataFrame:
    """Iterates seasons most-recent-first (not the order passed in) -- under a
    monthly call budget, current draft classes are the most immediately useful
    and this makes an interrupted/partial run still land on the most valuable
    data, picking up older seasons automatically on the next run via the cache."""
    client = client or CFBDClient()
    seasons = seasons if seasons is not None else DEFAULT_SEASONS
    frames = []
    for season in sorted(seasons, reverse=True):
        for category in CATEGORY_MAPS:
            rows = client.get('/stats/player/season', {'year': season, 'category': category})
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=['season', 'playerId', 'player', 'position', 'team',
                                      'conference', 'category', 'statType', 'stat'])
    return pd.concat(frames, ignore_index=True)


STAT_COLUMNS = sorted(set(PASSING_STAT_MAP.values()) | set(RUSHING_STAT_MAP.values()))


def pivot_player_stats(long_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns (wide_player_season_df, unrecognized_stat_types), one row per
    (season, playerId) -- even for a player who shows up under more than one
    team in the same season (a genuine CFBD occurrence, e.g. a mid-year/grad
    transfer or a split-season JUCO move). Multi-team seasons are combined
    (stats summed, schools joined "A; B", matching nflverse's own convention
    for the same situation) rather than left as duplicate rows, which would
    violate the one-row-per-player-season DB constraint."""
    if long_df.empty:
        return pd.DataFrame(), []

    df = long_df.copy()
    df['stat_numeric'] = pd.to_numeric(df['stat'], errors='coerce')

    unrecognized = set()
    def _target_col(row):
        stat_map = CATEGORY_MAPS.get(row['category'], {})
        key = str(row['statType']).strip().lower()
        if key not in stat_map:
            unrecognized.add(f"{row['category']}:{row['statType']}")
            return None
        return stat_map[key]  # both maps' values are already fully-qualified column names

    df['target_col'] = df.apply(_target_col, axis=1)
    recognized = df[df['target_col'].notna()].copy()

    identity_cols = ['season', 'playerId', 'player', 'position', 'team', 'conference']
    if recognized.empty:
        return pd.DataFrame(columns=identity_cols), sorted(unrecognized)

    per_team = recognized.pivot_table(index=identity_cols, columns='target_col', values='stat_numeric',
                                       aggfunc='first').reset_index()
    per_team.columns.name = None

    stat_cols_present = [c for c in STAT_COLUMNS if c in per_team.columns]
    wide = (per_team.groupby(['season', 'playerId'], as_index=False)
            .agg(**{
                'player': ('player', 'first'),
                'position': ('position', 'first'),
                'team': ('team', lambda s: '; '.join(dict.fromkeys(s))),  # dedupe, keep order
                'conference': ('conference', 'first'),
                # min_count=1: sum of an all-missing group must stay NaN, not
                # become a fabricated 0 (e.g. a pure rusher has no passing stats
                # at all, in any team-stint row, for that season).
                **{c: (c, lambda s: s.sum(min_count=1)) for c in stat_cols_present},
            }))
    return wide, sorted(unrecognized)


def fetch_roster_bio(seasons: list[int] | None, client: CFBDClient | None = None) -> pd.DataFrame:
    client = client or CFBDClient()
    seasons = seasons if seasons is not None else DEFAULT_SEASONS
    frames = []
    for season in sorted(seasons, reverse=True):
        rows = client.get('/roster', {'year': season})
        if rows:
            frame = pd.DataFrame(rows)
            frame['season'] = season
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=['season', 'team', 'first_name', 'last_name', 'height', 'weight'])
    df = pd.concat(frames, ignore_index=True)
    df['player_name'] = (df['firstName'].fillna('') + ' ' + df['lastName'].fillna('')).str.strip()
    return df[['season', 'team', 'player_name', 'height', 'weight', 'position']].rename(
        columns={'team': 'school', 'height': 'height_inches', 'weight': 'weight_lbs',
                 'position': 'roster_position'})


def fetch_team_ratings(seasons: list[int] | None, client: CFBDClient | None = None) -> pd.DataFrame:
    client = client or CFBDClient()
    seasons = seasons if seasons is not None else DEFAULT_SEASONS
    frames = []
    for season in sorted(seasons, reverse=True):
        rows = client.get('/ratings/sp', {'year': season})
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=['year', 'team', 'conference', 'rating', 'ranking', 'sos'])
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={'year': 'season'})[
        ['season', 'team', 'conference', 'rating', 'ranking', 'sos']
    ].rename(columns={'team': 'school', 'rating': 'sp_rating', 'ranking': 'sp_ranking',
                       'sos': 'strength_of_schedule'})
