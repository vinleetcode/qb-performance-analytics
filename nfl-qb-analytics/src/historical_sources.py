"""Registry of verified source availability (config/sources.json).

This module intentionally contains NO scraping or ingestion code for
1920-1998 or the 1960-1969 AFL: that bulk data lives exclusively behind
Pro Football Reference / Sports Reference, whose published data-use policy
(https://www.sports-reference.com/data_use.html, reviewed 2026-10-28)
prohibits automated/bulk access without a paid custom-data license
(minimum $5,000, per their own page) and prohibits building a replacement
data store from their content. No such license was available in this
environment, so per this project's instructions, that gap is documented
here and in the coverage report rather than filled with scraped or
third-party re-hosted data of unverified license.

A specific third-party candidate was evaluated on 2026-10-29 and rejected on
the same grounds: zackthoutt/nfl-player-stats (mirrored on Kaggle as
zynicide/nfl-football-player-stats). Its own README states the data was
scraped from pro-football-reference.com; a self-applied "CC0: Public Domain"
tag on the Kaggle mirror doesn't grant rights the uploader never held over
PFR's data, and doesn't substitute for PFR's own permission. Ingesting it would
mean reconstructing the same restricted PFR bulk dataset at one remove. See the
`zackthoutt_pfr_scrape_kaggle` entry in config/sources.json for the full
reasoning and tests/test_source_registry.py for the regression guard.

If a *licensed* export is obtained later (e.g. the $5,000+ Sports Reference
custom-data path, or a clean-provenance alternative), add an
`ingest_historical_pfr.py` module that targets the same raw schema used by
ingest_nflverse.py (source_player_id, display_name, position_listed, season,
league_id, season_type, primary_team, games, completions, attempts,
passing_yards, passing_tds, interceptions, sacks_suffered, sack_yards_lost,
carries, rushing_yards, rushing_tds, rushing_fumbles_lost, completeness_flag,
completeness_notes) and register it as its own `source_id` -- never merge it
into `nflverse_player_stats`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_source_registry() -> list[dict]:
    with open(ROOT / 'config' / 'sources.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['sources']


def not_ingested_season_ranges() -> list[dict]:
    """Returns the registry entries that could not be ingested, each with its
    season range, league implication, and reason -- used to build the
    coverage report's explicit 'missing coverage' section."""
    return [s for s in load_source_registry() if s['status'] == 'not_ingested']
