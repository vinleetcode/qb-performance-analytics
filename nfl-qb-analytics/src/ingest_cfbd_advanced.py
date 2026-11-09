"""Phase-3 CollegeFootballData ingestion: game-level and team-level context.

Every endpoint and parameter here was checked against the live OpenAPI spec
(https://api.collegefootballdata.com/api-docs.json, fetched 2026-11-09) and
probed with a real (cached) call before being used:

  endpoint                     params                         coverage seen
  ---------------------------  -----------------------------  ------------------------------
  /games                       year, seasonType=both          2004+ (FBS+FCS schedule/results)
  /games/players               year, week, category=passing,  2004+ box scores, one call per
                               [seasonType=postseason]        week (week|team|id is REQUIRED)
  /stats/season                year                           2004+; NO `sacks` stat in 2004
  /stats/season/advanced       year                           2004+, but play-by-play is sparse
                                                              in early years (see coverage gate)
  /talent                      year                           247 Team Talent Composite
  /recruiting/teams            year                           team recruiting class points/rank
  /recruiting/players          year (HighSchool, default)     individual recruits, athleteId
  /stats/player/season         year, category=receiving       player receiving totals
  /player/returning            year                           returning-production shares
  /draft/picks                 year                           NFL picks with collegeAthleteId

Not used, and why:
  /passing/players/games  -- requires week|team|passerId AND returned 0 rows for
                             2005/2022 probes (newer charting-based data only).
  /games/{id}/preview/adjusted -- free tier reports adjustedMetrics: False.
  /coaches*               -- coaching is explicitly out of scope for this phase.

Call budget (free tier = 1,000/month): ~22 per season-level endpoint, ~17 per
season for weekly box scores (~375 total), ~610 for a full 2004-2025 backfill.
Every response is cached on disk by CFBDClient, so reruns cost zero calls, and
the client enforces a hard per-run ceiling (max_network_calls).
"""
from __future__ import annotations

import pandas as pd

from src.cfbd_client import CFBDClient

CFB_SEASONS = list(range(2004, 2026))       # college seasons with CFBD player stats
# /stats/player/season was found (2026-11-09) to be nearly empty for early seasons:
# of FBS passers with 100+ box-score attempts, it contains 0/132 (2004), 1/136
# (2005), 16/145 (2006), 53/145 (2007), 93/141 (2008) and ~all from 2009 on. Box
# scores are complete in every season, so for 2004-2008 rushing box scores are
# also fetched (passing box scores are fetched for every season anyway).
BOX_RUSHING_SEASONS = list(range(2004, 2009))


def id_str(value):
    """Canonical string form of a CFBD id: 2081046, 2081046.0, '2081046' -> '2081046';
    None/NaN -> None. (pandas turns int columns with gaps into floats, which once
    produced '2081046.0' keys that silently never matched.)"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    s = str(value).strip()
    if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
        s = s[:-2]
    return s or None
RECRUIT_YEARS = list(range(2000, 2026))     # classes that can be on a 2004+ roster
DRAFT_YEARS = list(range(2000, 2027))       # matches the nflverse draft board range

SOURCE_IDS = {
    'games': 'cfbd_games',
    'game_passing': 'cfbd_game_player_passing',
    'team_stats': 'cfbd_team_season_stats',
    'team_advanced': 'cfbd_team_season_stats',
    'talent': 'cfbd_team_talent',
    'recruiting_teams': 'cfbd_recruiting',
    'recruits': 'cfbd_recruiting',
    'receiving': 'cfbd_player_receiving',
    'returning': 'cfbd_returning_production',
    'draft_picks': 'cfbd_draft_picks',
    'game_rushing': 'cfbd_game_player_passing',  # same box-score endpoint, category=rushing
}


# --------------------------------------------------------------------------
# Raw fetchers (thin; caching + budget handled by CFBDClient)
# --------------------------------------------------------------------------

def fetch_games(seasons, client: CFBDClient) -> pd.DataFrame:
    frames = []
    for season in sorted(seasons, reverse=True):
        rows = client.get('/games', {'year': season, 'seasonType': 'both'})
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keep = ['id', 'season', 'week', 'seasonType', 'startDate', 'completed', 'neutralSite',
            'conferenceGame', 'homeId', 'homeTeam', 'homeConference', 'homeClassification',
            'homePoints', 'awayId', 'awayTeam', 'awayConference', 'awayClassification', 'awayPoints']
    return df[[c for c in keep if c in df.columns]]


def game_weeks(games: pd.DataFrame, season: int) -> list[tuple[str, int]]:
    """(seasonType, week) pairs that actually contain games involving an FBS
    team in `season` -- derived from the schedule, never a hard-coded week
    count (2020's COVID season and 12-vs-13-game years differ)."""
    g = games[(games['season'] == season)]
    fbs = g[(g['homeClassification'] == 'fbs') | (g['awayClassification'] == 'fbs')]
    pairs = fbs[['seasonType', 'week']].drop_duplicates()
    pairs = pairs[pairs['seasonType'].isin(['regular', 'postseason'])]
    return sorted((str(st), int(w)) for st, w in pairs.itertuples(index=False))


def fetch_game_passing(seasons, games: pd.DataFrame, client: CFBDClient) -> pd.DataFrame:
    """Per-game player passing box scores, one call per (season, week)."""
    return _fetch_box(seasons, games, client, 'passing', parse_game_passing, GAME_PASSING_COLUMNS)


def fetch_game_rushing(seasons, games: pd.DataFrame, client: CFBDClient) -> pd.DataFrame:
    """Per-game player rushing box scores (only BOX_RUSHING_SEASONS by default)."""
    return _fetch_box(seasons, games, client, 'rushing', parse_game_rushing, GAME_RUSHING_COLUMNS)


def _fetch_box(seasons, games, client, category, parser, columns) -> pd.DataFrame:
    frames = []
    for season in sorted(seasons, reverse=True):
        for season_type, week in game_weeks(games, season):
            params = {'year': season, 'week': week, 'category': category}
            if season_type != 'regular':
                params['seasonType'] = season_type
            parsed = parser(client.get('/games/players', params), season, week, season_type)
            if not parsed.empty:
                frames.append(parsed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


GAME_RUSHING_COLUMNS = ['game_id', 'season', 'week', 'season_type', 'team', 'opponent', 'cfbd_player_id',
                        'player_name', 'carries', 'rushing_yards', 'rushing_tds']


def parse_game_rushing(payload, season: int, week: int, season_type: str) -> pd.DataFrame:
    rows = []
    for game in payload or []:
        teams = game.get('teams') or []
        for i, team in enumerate(teams):
            opp = teams[1 - i] if len(teams) == 2 else {}
            per_athlete: dict[str, dict] = {}
            for cat in team.get('categories') or []:
                if cat.get('name') != 'rushing':
                    continue
                for typ in cat.get('types') or []:
                    for ath in typ.get('athletes') or []:
                        per_athlete.setdefault(str(ath.get('id')), {'name': ath.get('name')})[typ.get('name')] = ath.get('stat')
            for aid, rec in per_athlete.items():
                pid = _int_or_none(aid)
                if pid is None or pid <= 0:
                    continue
                rows.append({'game_id': game.get('id'), 'season': season, 'week': week, 'season_type': season_type,
                             'team': team.get('team'), 'opponent': opp.get('team'), 'cfbd_player_id': str(pid),
                             'player_name': rec.get('name'), 'carries': _int_or_none(rec.get('CAR')),
                             'rushing_yards': _int_or_none(rec.get('YDS')), 'rushing_tds': _int_or_none(rec.get('TD'))})
    return pd.DataFrame(rows, columns=GAME_RUSHING_COLUMNS)


GAME_PASSING_COLUMNS = ['game_id', 'season', 'week', 'season_type', 'team', 'conference', 'home_away',
                        'opponent', 'opponent_conference', 'cfbd_player_id', 'player_name',
                        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions']


def _int_or_none(text):
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError):
        return None


def parse_game_passing(payload, season: int, week: int, season_type: str) -> pd.DataFrame:
    """Flattens CFBD's nested /games/players passing payload into one row per
    (game, team, passer). "C/ATT" is split into completions/attempts; any stat
    that fails to parse stays null rather than becoming 0. Team pseudo-athletes
    (non-positive ids, e.g. aborted-snap "Team" rows) are kept -- they are real
    plays against the defense -- but carry cfbd_player_id=None so they can never
    be attributed to a quarterback."""
    rows = []
    for game in payload or []:
        teams = game.get('teams') or []
        for i, team in enumerate(teams):
            opp = teams[1 - i] if len(teams) == 2 else {}
            per_athlete: dict[str, dict] = {}
            for cat in team.get('categories') or []:
                if cat.get('name') != 'passing':
                    continue
                for typ in cat.get('types') or []:
                    tname = typ.get('name')
                    for ath in typ.get('athletes') or []:
                        key = str(ath.get('id'))
                        rec = per_athlete.setdefault(key, {'name': ath.get('name')})
                        rec[tname] = ath.get('stat')
            for aid, rec in per_athlete.items():
                comp = att = None
                ca = rec.get('C/ATT')
                if ca and '/' in str(ca):
                    c_txt, a_txt = str(ca).split('/', 1)
                    comp, att = _int_or_none(c_txt), _int_or_none(a_txt)
                pid = _int_or_none(aid)
                rows.append({
                    'game_id': game.get('id'), 'season': season, 'week': week, 'season_type': season_type,
                    'team': team.get('team'), 'conference': team.get('conference'),
                    'home_away': team.get('homeAway'), 'opponent': opp.get('team'),
                    'opponent_conference': opp.get('conference'),
                    'cfbd_player_id': str(pid) if pid is not None and pid > 0 else None,
                    'player_name': rec.get('name'),
                    'completions': comp, 'attempts': att,
                    'passing_yards': _int_or_none(rec.get('YDS')),
                    'passing_tds': _int_or_none(rec.get('TD')),
                    'interceptions': _int_or_none(rec.get('INT')),
                })
    return pd.DataFrame(rows, columns=GAME_PASSING_COLUMNS)


# statName -> column. Only stats that exist in /stats/season are mapped; any
# season that lacks one (e.g. no `sacks` in 2004) simply leaves it null.
TEAM_STAT_MAP = {
    'games': 'games',
    'passAttempts': 'pass_attempts', 'passCompletions': 'pass_completions',
    'netPassingYards': 'net_passing_yards', 'passingTDs': 'passing_tds',
    'interceptions': 'interceptions_thrown',
    'rushingAttempts': 'rushing_attempts', 'rushingYards': 'rushing_yards', 'rushingTDs': 'rushing_tds',
    'totalYards': 'total_yards', 'possessionTime': 'possession_seconds',
    'sacks': 'def_sacks', 'sacksOpponent': 'sacks_allowed',
    'tacklesForLoss': 'def_tfl', 'tacklesForLossOpponent': 'tfl_allowed',
    'passAttemptsOpponent': 'opp_pass_attempts', 'passCompletionsOpponent': 'opp_pass_completions',
    'netPassingYardsOpponent': 'opp_net_passing_yards', 'passingTDsOpponent': 'opp_passing_tds',
    'passesIntercepted': 'def_interceptions',
    'rushingAttemptsOpponent': 'opp_rushing_attempts', 'rushingYardsOpponent': 'opp_rushing_yards',
}


def fetch_team_season_stats(seasons, client: CFBDClient) -> tuple[pd.DataFrame, list[str]]:
    """Returns (wide team-season frame, sorted list of statNames seen but not mapped)."""
    frames = []
    for season in sorted(seasons, reverse=True):
        rows = client.get('/stats/season', {'year': season})
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(), []
    long = pd.concat(frames, ignore_index=True)
    unmapped = sorted(set(long['statName']) - set(TEAM_STAT_MAP))
    long = long[long['statName'].isin(TEAM_STAT_MAP)].copy()
    long['col'] = long['statName'].map(TEAM_STAT_MAP)
    long['value'] = pd.to_numeric(long['statValue'], errors='coerce')
    wide = long.pivot_table(index=['season', 'team'], columns='col', values='value', aggfunc='first').reset_index()
    conf = long.groupby(['season', 'team'])['conference'].first().reset_index()
    wide = wide.merge(conf, on=['season', 'team'], how='left')
    wide.columns.name = None
    for col in TEAM_STAT_MAP.values():
        if col not in wide.columns:
            wide[col] = pd.NA
    return wide, unmapped


ADVANCED_FIELDS = {
    # (side, path) -> column
    ('offense', 'plays'): 'off_plays',
    ('offense', 'ppa'): 'off_ppa',
    ('offense', 'successRate'): 'off_success_rate',
    ('offense', 'explosiveness'): 'off_explosiveness',
    ('offense', 'powerSuccess'): 'off_power_success',
    ('offense', 'stuffRate'): 'off_stuff_rate',
    ('offense', 'lineYards'): 'off_line_yards',
    ('offense', 'havoc.total'): 'off_havoc_allowed',
    ('offense', 'havoc.frontSeven'): 'off_front7_havoc_allowed',
    ('offense', 'passingPlays.rate'): 'off_passing_play_rate',
    ('offense', 'passingPlays.ppa'): 'off_passing_ppa',
    ('offense', 'passingPlays.successRate'): 'off_passing_success_rate',
    ('offense', 'rushingPlays.rate'): 'off_rushing_play_rate',
    ('offense', 'rushingPlays.ppa'): 'off_rushing_ppa',
    ('offense', 'passingDowns.rate'): 'off_passing_downs_rate',
    ('defense', 'plays'): 'def_plays',
    ('defense', 'ppa'): 'def_ppa',
    ('defense', 'successRate'): 'def_success_rate',
    ('defense', 'passingPlays.ppa'): 'def_passing_ppa',
    ('defense', 'passingPlays.successRate'): 'def_passing_success_rate',
    ('defense', 'havoc.total'): 'def_havoc',
}


def _dig(d, dotted):
    for part in dotted.split('.'):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def fetch_team_advanced(seasons, client: CFBDClient) -> pd.DataFrame:
    rows = []
    for season in sorted(seasons, reverse=True):
        for rec in client.get('/stats/season/advanced', {'year': season}) or []:
            row = {'season': rec.get('season'), 'team': rec.get('team')}
            for (side, path), col in ADVANCED_FIELDS.items():
                row[col] = _dig(rec.get(side) or {}, path)
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_talent(seasons, client: CFBDClient) -> pd.DataFrame:
    rows = []
    for season in sorted(seasons, reverse=True):
        for rec in client.get('/talent', {'year': season}) or []:
            rows.append({'season': rec.get('year'), 'team': rec.get('team'),
                         'talent_composite': pd.to_numeric(rec.get('talent'), errors='coerce')})
    return pd.DataFrame(rows, columns=['season', 'team', 'talent_composite'])


def fetch_recruiting_teams(years, client: CFBDClient) -> pd.DataFrame:
    rows = []
    for year in sorted(years, reverse=True):
        for rec in client.get('/recruiting/teams', {'year': year}) or []:
            rows.append({'class_year': rec.get('year'), 'team': rec.get('team'),
                         'recruiting_rank': rec.get('rank'),
                         'recruiting_points': pd.to_numeric(rec.get('points'), errors='coerce')})
    return pd.DataFrame(rows, columns=['class_year', 'team', 'recruiting_rank', 'recruiting_points'])


def fetch_recruits(years, client: CFBDClient) -> pd.DataFrame:
    rows = []
    for year in sorted(years, reverse=True):
        for rec in client.get('/recruiting/players', {'year': year}) or []:
            rows.append({
                'recruit_id': id_str(rec.get('id')), 'athlete_id': id_str(rec.get('athleteId')),
                'recruit_type': rec.get('recruitType'), 'class_year': rec.get('year'),
                'name': rec.get('name'), 'committed_to': rec.get('committedTo'),
                'position': rec.get('position'), 'stars': rec.get('stars'),
                'rating': rec.get('rating'), 'national_ranking': rec.get('ranking'),
            })
    return pd.DataFrame(rows)


def fetch_receiving(seasons, client: CFBDClient) -> pd.DataFrame:
    frames = []
    for season in sorted(seasons, reverse=True):
        rows = client.get('/stats/player/season', {'year': season, 'category': 'receiving'})
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    long['value'] = pd.to_numeric(long['stat'], errors='coerce')
    stat_map = {'REC': 'receptions', 'YDS': 'receiving_yards', 'TD': 'receiving_tds'}
    long = long[long['statType'].str.upper().isin(stat_map)].copy()
    long['col'] = long['statType'].str.upper().map(stat_map)
    wide = long.pivot_table(index=['season', 'playerId', 'player', 'position', 'team', 'conference'],
                            columns='col', values='value', aggfunc='first').reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={'playerId': 'cfbd_player_id', 'player': 'player_name'})
    wide['cfbd_player_id'] = wide['cfbd_player_id'].map(id_str)
    return wide


def fetch_returning(seasons, client: CFBDClient) -> pd.DataFrame:
    rows = []
    for season in sorted(seasons, reverse=True):
        for rec in client.get('/player/returning', {'year': season}) or []:
            rows.append({'season': rec.get('season'), 'team': rec.get('team'),
                         'returning_ppa_pct': rec.get('percentPPA'),
                         'returning_passing_ppa_pct': rec.get('percentPassingPPA'),
                         'returning_receiving_ppa_pct': rec.get('percentReceivingPPA'),
                         'returning_rushing_ppa_pct': rec.get('percentRushingPPA'),
                         'returning_usage': rec.get('usage'),
                         'returning_receiving_usage': rec.get('receivingUsage')})
    return pd.DataFrame(rows)


def fetch_draft_picks(years, client: CFBDClient) -> pd.DataFrame:
    rows = []
    for year in sorted(years, reverse=True):
        for rec in client.get('/draft/picks', {'year': year}) or []:
            rows.append({
                'year': rec.get('year'), 'overall': rec.get('overall'), 'round': rec.get('round'),
                'pick': rec.get('pick'), 'name': rec.get('name'), 'position': rec.get('position'),
                'college_athlete_id': id_str(rec.get('collegeAthleteId')), 'nfl_athlete_id': id_str(rec.get('nflAthleteId')),
                'college_team': rec.get('collegeTeam'), 'college_conference': rec.get('collegeConference'),
                'nfl_team': rec.get('nflTeam'), 'height': rec.get('height'), 'weight': rec.get('weight'),
                'pre_draft_ranking': rec.get('preDraftRanking'),
                'pre_draft_position_ranking': rec.get('preDraftPositionRanking'),
                'pre_draft_grade': rec.get('preDraftGrade'),
            })
    return pd.DataFrame(rows)


def prefetch_all(client: CFBDClient) -> dict:
    """Warms the on-disk cache for every Phase-3 endpoint, cheap season-level
    endpoints first, weekly box scores last (they're ~60% of the budget).
    Returns call accounting. Safe to interrupt and rerun."""
    games = fetch_games(CFB_SEASONS, client)
    fetch_team_season_stats(CFB_SEASONS, client)
    fetch_team_advanced(CFB_SEASONS, client)
    fetch_talent(CFB_SEASONS, client)
    fetch_recruiting_teams(RECRUIT_YEARS, client)
    fetch_recruits(RECRUIT_YEARS, client)
    fetch_receiving(CFB_SEASONS, client)
    fetch_returning(CFB_SEASONS, client)
    fetch_draft_picks(DRAFT_YEARS, client)
    fetch_game_passing(CFB_SEASONS, games, client)
    fetch_game_rushing(BOX_RUSHING_SEASONS, games, client)
    return {'network_calls': client.network_calls, 'cache_hits': client.cache_hits}


if __name__ == '__main__':  # python -m src.ingest_cfbd_advanced  (cache warm-up only)
    import json
    import sys
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 750
    c = CFBDClient(max_network_calls=budget)
    try:
        print(json.dumps(prefetch_all(c)))
    finally:
        print(json.dumps({'final_network_calls': c.network_calls, 'final_cache_hits': c.cache_hits}))
