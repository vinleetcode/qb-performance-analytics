"""Team-season offensive environment: scheme profile, offensive-line proxies,
receiver context, and roster talent.

Interpretation caveats (repeated in the feature registry):
  * NCAA statistics count sacks as rushing plays, so team rushing attempts/yards
    include sacks and pass_rate is understated for sack-prone teams.
  * Sack rate is NOT a pure offensive-line measure: QB pocket decisions, hold
    time and scheme (e.g. quick game) drive it too.
  * Team receiving yards are essentially the QB's own passing yards (circular);
    they are stored for completeness but flagged as volume, not receiver ability.
    The non-circular receiver measures are target concentration and the
    recruiting ratings of the top targets (assigned before they ever caught a
    pass from this QB).
  * Advanced play-by-play metrics (line yards, stuff rate, havoc, PPA) are only
    used where play-by-play coverage is near-complete (pbp_coverage_ratio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PBP_MIN_COVERAGE = 0.85
PBP_MAX_COVERAGE = 1.25
ADVANCED_COLUMNS = ['off_ppa', 'off_success_rate', 'off_explosiveness', 'off_power_success', 'off_stuff_rate',
                    'off_line_yards', 'off_havoc_allowed', 'off_front7_havoc_allowed', 'off_passing_play_rate',
                    'off_passing_ppa', 'off_passing_success_rate', 'off_rushing_play_rate', 'off_rushing_ppa',
                    'off_passing_downs_rate', 'def_ppa', 'def_success_rate', 'def_passing_ppa',
                    'def_passing_success_rate', 'def_havoc']
RECEIVER_POSITIONS = {'WR', 'TE', 'RB', 'ATH', 'FB', 'APB'}


def _div(a, b):
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    return np.where(b > 0, a / b, np.nan)


def gate_advanced(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Adds pbp_coverage_ratio/advanced_reliable and blanks advanced metrics for
    team-seasons whose play-by-play is too incomplete to trust (keeps raw box
    stats untouched)."""
    ts = team_stats.copy()
    for c in ADVANCED_COLUMNS + ['off_plays']:
        if c not in ts.columns:
            ts[c] = np.nan
    box_plays = pd.to_numeric(ts['pass_attempts'], errors='coerce') + pd.to_numeric(ts['rushing_attempts'], errors='coerce')
    ts['pbp_coverage_ratio'] = _div(ts.get('off_plays'), box_plays)
    ts['advanced_reliable'] = ((ts['pbp_coverage_ratio'] >= PBP_MIN_COVERAGE)
                               & (ts['pbp_coverage_ratio'] <= PBP_MAX_COVERAGE)).fillna(False)
    for c in ADVANCED_COLUMNS:
        if c in ts.columns:
            ts.loc[~ts['advanced_reliable'], c] = np.nan
    return ts


def schedule_pairs(games: pd.DataFrame) -> pd.DataFrame:
    """(season, team, opponent) for every completed game, both directions."""
    g = games[games['completed'].fillna(True).astype(bool)]
    a = g[['season', 'homeTeam', 'awayTeam']].rename(columns={'homeTeam': 'team', 'awayTeam': 'opponent'})
    b = g[['season', 'awayTeam', 'homeTeam']].rename(columns={'awayTeam': 'team', 'homeTeam': 'opponent'})
    return pd.concat([a, b], ignore_index=True)


def team_games_played(games: pd.DataFrame) -> pd.DataFrame:
    sp = schedule_pairs(games)
    return sp.groupby(['season', 'team']).size().rename('team_games_played').reset_index()


def season_end_dates(games: pd.DataFrame) -> pd.DataFrame:
    """Last game date per (season, team), incl. bowls -- the date a season's
    statistics become complete (feature availability timestamp)."""
    g = games.copy()
    g['date'] = pd.to_datetime(g['startDate'], utc=True, errors='coerce').dt.tz_convert(None).dt.normalize()
    a = g[['season', 'homeTeam', 'date']].rename(columns={'homeTeam': 'team'})
    b = g[['season', 'awayTeam', 'date']].rename(columns={'awayTeam': 'team'})
    return pd.concat([a, b]).groupby(['season', 'team'])['date'].max().rename('season_end_date').reset_index()


def fbs_teams(games: pd.DataFrame) -> pd.DataFrame:
    a = games.loc[games['homeClassification'].eq('fbs'), ['season', 'homeTeam']].rename(columns={'homeTeam': 'team'})
    b = games.loc[games['awayClassification'].eq('fbs'), ['season', 'awayTeam']].rename(columns={'awayTeam': 'team'})
    return pd.concat([a, b]).drop_duplicates().reset_index(drop=True)


def receiver_context(receiving: pd.DataFrame, recruits: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team): receiving volume (circular, flagged), concentration of
    receptions among top targets, and mean 247 composite rating of the top-3
    targets (by receptions) whose recruit record CFBD links by athleteId."""
    r = receiving.copy()
    r['receptions'] = pd.to_numeric(r['receptions'], errors='coerce')
    r['receiving_yards'] = pd.to_numeric(r['receiving_yards'], errors='coerce')
    rec_rating = (recruits.dropna(subset=['athlete_id', 'rating'])
                  .sort_values('rating', ascending=False)
                  .drop_duplicates('athlete_id')[['athlete_id', 'rating']]
                  .rename(columns={'athlete_id': 'cfbd_player_id', 'rating': 'recruit_rating'}))
    rec_rating['cfbd_player_id'] = rec_rating['cfbd_player_id'].astype(str)
    r['cfbd_player_id'] = r['cfbd_player_id'].astype(str)
    r = r.merge(rec_rating, on='cfbd_player_id', how='left')
    rows = []
    for (season, team), g in r.groupby(['season', 'team']):
        tot = g['receptions'].sum()
        top = g.sort_values('receptions', ascending=False)
        top3 = top.head(3)
        rows.append({
            'season': season, 'team': team,
            'team_receiving_yards': g['receiving_yards'].sum(min_count=1),
            'top_target_reception_share': (top['receptions'].iloc[0] / tot) if tot > 0 else np.nan,
            'top3_reception_share': (top3['receptions'].sum() / tot) if tot > 0 else np.nan,
            'top3_receivers_avg_recruit_rating': top3['recruit_rating'].mean(),
            'top3_receivers_rated': int(top3['recruit_rating'].notna().sum()),
        })
    return pd.DataFrame(rows)


def roster_talent(talent: pd.DataFrame, recruiting_teams: pd.DataFrame, returning: pd.DataFrame,
                  seasons: list[int]) -> pd.DataFrame:
    """Team-season roster talent with within-season percentiles. The 4-class
    recruiting average uses classes season-3..season, all signed before the
    season starts."""
    rt = recruiting_teams.copy()
    teams = set(rt['team']) | set(talent['team']) | set(returning.get('team', pd.Series(dtype=str)))
    base = pd.DataFrame([(s, t) for s in seasons for t in teams], columns=['season', 'team'])
    cur = rt.rename(columns={'class_year': 'season', 'recruiting_points': 'recruiting_points_class',
                             'recruiting_rank': 'recruiting_rank_class'})
    base = base.merge(cur, on=['season', 'team'], how='left')
    four = []
    for s in seasons:
        window = rt[rt['class_year'].between(s - 3, s)]
        agg = window.groupby('team')['recruiting_points'].agg(['mean', 'count']).reset_index()
        agg['season'] = s
        four.append(agg)
    four = pd.concat(four, ignore_index=True).rename(
        columns={'mean': 'recruiting_points_4yr_avg', 'count': 'recruiting_4yr_classes_found'})
    base = base.merge(four, on=['season', 'team'], how='left')
    base = base.merge(talent, on=['season', 'team'], how='left')
    if not returning.empty:
        base = base.merge(returning.drop(columns=[c for c in ('conference',) if c in returning.columns]),
                          on=['season', 'team'], how='left')
    base['talent_composite_pctile'] = base.groupby('season')['talent_composite'].rank(pct=True)
    base['recruiting_4yr_pctile'] = base.groupby('season')['recruiting_points_4yr_avg'].rank(pct=True)
    keep_any = ['talent_composite', 'recruiting_points_class', 'recruiting_points_4yr_avg', 'returning_ppa_pct']
    base = base.dropna(subset=[c for c in keep_any if c in base.columns], how='all')
    return base


def team_environment(team_stats: pd.DataFrame, games: pd.DataFrame, receivers: pd.DataFrame,
                     season_def: pd.DataFrame, sp_detail: pd.DataFrame) -> pd.DataFrame:
    """Combines box/advanced team stats, schedule-weighted opponent sack rates,
    receiver context, pass-defense table and SP+ detail into one team-season row."""
    ts = gate_advanced(team_stats)
    ts['plays'] = pd.to_numeric(ts['pass_attempts'], errors='coerce') + pd.to_numeric(ts['rushing_attempts'], errors='coerce')
    env = ts[['season', 'team']].copy()
    env['plays_per_game'] = _div(ts['plays'], ts['games'])
    env['pass_rate'] = _div(ts['pass_attempts'], ts['plays'])
    env['rush_rate'] = _div(ts['rushing_attempts'], ts['plays'])
    env['seconds_per_play'] = _div(ts['possession_seconds'], ts['plays'])
    env['off_ppa'] = ts['off_ppa']
    env['off_success_rate'] = ts['off_success_rate']
    env['sacks_allowed'] = pd.to_numeric(ts['sacks_allowed'], errors='coerce')
    env['sack_rate_allowed'] = _div(ts['sacks_allowed'], pd.to_numeric(ts['pass_attempts'], errors='coerce')
                                    + pd.to_numeric(ts['sacks_allowed'], errors='coerce'))
    env['team_rush_ypc'] = _div(ts['rushing_yards'], ts['rushing_attempts'])
    env['off_stuff_rate'] = ts['off_stuff_rate']
    env['off_line_yards'] = ts['off_line_yards']
    env['off_havoc_allowed'] = ts['off_havoc_allowed']

    # opponents' defensive sack rate, schedule-weighted (full season, includes the
    # game vs this team -- documented: no per-game sack data is ingested)
    def_rate = ts[['season', 'team']].copy()
    def_rate['def_sack_rate'] = _div(ts['def_sacks'], pd.to_numeric(ts['opp_pass_attempts'], errors='coerce')
                                     + pd.to_numeric(ts['def_sacks'], errors='coerce'))
    league = (ts.assign(_s=pd.to_numeric(ts['def_sacks'], errors='coerce'),
                        _a=pd.to_numeric(ts['opp_pass_attempts'], errors='coerce'))
              .groupby('season').apply(lambda d: d['_s'].sum(min_count=1) / (d['_a'].sum() + d['_s'].sum())
                                       if d['_s'].notna().any() else np.nan, include_groups=False)
              .rename('league_def_sack_rate').reset_index())
    sp = schedule_pairs(games).merge(def_rate.rename(columns={'team': 'opponent'}), on=['season', 'opponent'], how='inner')
    opp = sp.groupby(['season', 'team'])['def_sack_rate'].mean().rename('opp_def_sack_rate').reset_index()
    env = env.merge(opp, on=['season', 'team'], how='left').merge(league, on='season', how='left')
    env['opp_adj_sack_rate'] = env['sack_rate_allowed'] - (env['opp_def_sack_rate'] - env['league_def_sack_rate'])
    env = env.drop(columns=['league_def_sack_rate'])

    env = env.merge(receivers, on=['season', 'team'], how='left')
    sd = season_def.rename(columns={'defense': 'team', 'ypa': 'def_pass_ypa_allowed',
                                    'cmp_pct': 'def_pass_cmp_pct_allowed'})
    env = env.merge(sd[['season', 'team', 'def_pass_ypa_allowed', 'def_pass_cmp_pct_allowed', 'def_pass_rank']],
                    on=['season', 'team'], how='left')
    if sp_detail is not None and not sp_detail.empty:
        env = env.merge(sp_detail[['season', 'team', 'sp_pace', 'sp_run_rate', 'sp_offense_rating']],
                        on=['season', 'team'], how='left')
    return env
