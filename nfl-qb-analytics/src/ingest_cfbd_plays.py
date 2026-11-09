"""CFBD play-by-play (/plays) -> slim per-week Parquet, for down-specific pass rates.

Cost and storage (measured 2026-11-09): one call per (season, week) for all FBS games,
about 13 MB of JSON and about 19k plays per regular-season week. Raw JSON is NOT kept:
each response is reduced to the columns below, written to data/cfbd_plays/ (gitignored),
and the raw cache file is deleted. Reruns read the Parquet files and make zero calls.

Budget policy (agreed 2026-11-09): stay under the 1,000-call monthly free tier.
2016-2025 now (about 180 calls, hard cap 200 per run); 2004-2015 after the Dec 1 reset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import ingest_cfbd_advanced as A
from src.cfbd_client import CFBDClient

ROOT = Path(__file__).resolve().parents[1]
PLAYS_DIR = ROOT / 'data' / 'cfbd_plays'
SEASONS_NOW = list(range(2016, 2026))
SEASONS_LATER = list(range(2004, 2016))
KEEP = ['gameId', 'offense', 'defense', 'offenseScore', 'defenseScore', 'period', 'down', 'distance',
        'yardsToGoal', 'yardsGained', 'playType', 'ppa']


def week_path(season: int, season_type: str, week: int) -> Path:
    return PLAYS_DIR / f'plays_{season}_{season_type}_{week:02d}.parquet'


def fetch_week(client: CFBDClient, season: int, season_type: str, week: int) -> pd.DataFrame:
    path = week_path(season, season_type, week)
    if path.exists():
        return pd.read_parquet(path)
    params = {'year': season, 'week': week}
    if season_type != 'regular':
        params['seasonType'] = season_type
    rows = client.get('/plays', params)
    df = pd.DataFrame(rows)
    df = df.reindex(columns=KEEP)
    df['season'] = season
    df['season_type'] = season_type
    df['week'] = week
    PLAYS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    raw = client.cache_dir / client._cache_key('/plays', params)
    if raw.exists():
        raw.unlink()  # 13 MB per week; the slim Parquet is the cache
    return df


def planned_weeks(seasons, client: CFBDClient) -> list[tuple[int, str, int]]:
    games = A.fetch_games(seasons, client)  # cached
    return [(s, st, w) for s in sorted(seasons, reverse=True) for st, w in A.game_weeks(games, s)]


def fetch_plays(seasons, client: CFBDClient) -> dict:
    plan = planned_weeks(seasons, client)
    done = missing = 0
    for s, st, w in plan:
        if week_path(s, st, w).exists():
            done += 1
            continue
        fetch_week(client, s, st, w)
        missing += 1
    return {'weeks_planned': len(plan), 'already_cached': done, 'fetched_now': missing,
            'network_calls': client.network_calls}


def load_plays(seasons=None) -> pd.DataFrame:
    files = sorted(PLAYS_DIR.glob('plays_*.parquet')) if PLAYS_DIR.exists() else []
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP + ['season', 'season_type', 'week'])
    if seasons is not None:
        df = df[df['season'].isin(list(seasons))]
    return df


if __name__ == '__main__':
    import json
    import sys
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    c = CFBDClient(max_network_calls=cap)
    plan = planned_weeks(SEASONS_NOW, c)
    todo = sum(1 for s, st, w in plan if not week_path(s, st, w).exists())
    print(json.dumps({'weeks_planned': len(plan), 'to_fetch': todo, 'cap': cap}), flush=True)
    if todo > cap:
        raise SystemExit(f'{todo} calls needed exceeds cap {cap}; not starting')
    print(json.dumps(fetch_plays(SEASONS_NOW, c)))
