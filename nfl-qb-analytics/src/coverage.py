"""Builds the coverage report: exactly which seasons/leagues/players/stats are
available from each source, and which are not and why. Written to the
coverage_report DB table and results/coverage_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.db.schema import CoverageReport, PlayerSeasonRaw, Player
from src.historical_sources import not_ingested_season_ranges

ROOT = Path(__file__).resolve().parents[1]


def rebuild_coverage_table(session, ingested_source_id: str, ingested_seasons: list[int],
                            league_id: str = 'NFL') -> None:
    session.query(CoverageReport).delete(synchronize_session=False)

    rows_by_season = (
        session.query(PlayerSeasonRaw.season)
        .filter(PlayerSeasonRaw.source_id == ingested_source_id)
        .all()
    )
    season_counts: dict[int, int] = {}
    for (season,) in rows_by_season:
        season_counts[season] = season_counts.get(season, 0) + 1

    for season in ingested_seasons:
        session.add(CoverageReport(
            league_id=league_id, season=season, source_id=ingested_source_id,
            status='ingested', player_season_count=season_counts.get(season, 0),
            reason=None,
        ))

    already_ingested = set(ingested_seasons)
    for entry in not_ingested_season_ranges():
        implied_league = entry.get('implied_league_id', 'NFL')
        for season in range(entry['coverage_start_season'], entry['coverage_end_season'] + 1):
            # A rejected candidate's claimed range can overlap years another source
            # already legitimately ingested (e.g. the Kaggle PFR-scrape candidate claims
            # through 2017, well inside the nflverse-covered 1999-2025 span). Don't log a
            # not_ingested row for a season that's actually covered -- that would inflate
            # "not ingested" counts for years that aren't actually gaps.
            if season in already_ingested:
                continue
            session.add(CoverageReport(
                league_id=implied_league, season=season, source_id=entry['source_id'],
                status='not_ingested', player_season_count=0, reason=entry['status_reason'],
            ))
    session.commit()


def summarize(session) -> dict:
    rows = session.query(CoverageReport).all()
    df = pd.DataFrame([{
        'league_id': r.league_id, 'season': r.season, 'source_id': r.source_id,
        'status': r.status, 'player_season_count': r.player_season_count, 'reason': r.reason,
    } for r in rows])
    if df.empty:
        return {'by_decade_league': [], 'totals': {}, 'not_ingested_ranges': []}

    df['decade'] = (df['season'] // 10) * 10

    by_decade = (df.groupby(['decade', 'league_id', 'status'], as_index=False)
                 .agg(seasons=('season', 'nunique'), player_seasons=('player_season_count', 'sum'))
                 .sort_values(['decade', 'league_id', 'status']))

    ingested = df[df.status.eq('ingested')]
    not_ingested = df[df.status.eq('not_ingested')]

    player_count = session.query(Player).count()
    player_season_count = session.query(PlayerSeasonRaw).count()

    not_ingested_ranges = (
        not_ingested.groupby(['league_id', 'reason'], as_index=False)
        .agg(first_season=('season', 'min'), last_season=('season', 'max'), seasons=('season', 'nunique'))
        .to_dict(orient='records')
    )

    summary = {
        'totals': {
            'distinct_seasons_ingested': int(ingested['season'].nunique()),
            'distinct_seasons_not_ingested': int(not_ingested['season'].nunique()),
            'unique_players': int(player_count),
            'player_season_observations': int(player_season_count),
        },
        'by_decade_league': by_decade.to_dict(orient='records'),
        'not_ingested_ranges': not_ingested_ranges,
    }
    return summary


def write_report(session, dest: Path | None = None) -> dict:
    dest = dest or (ROOT / 'results' / 'coverage_report.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(session)
    dest.write_text(json.dumps(summary, indent=2, default=str))
    return summary
