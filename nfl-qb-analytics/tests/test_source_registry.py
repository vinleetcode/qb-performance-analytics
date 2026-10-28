"""Guards the source-registry decisions in config/sources.json -- in particular,
that the Kaggle/GitHub "nfl-player-stats" candidate (zackthoutt/nfl-player-stats,
mirrored as zynicide/nfl-football-player-stats on Kaggle) stays rejected. It was
evaluated on 2026-10-29: the repo's own README says the data was scraped from
pro-football-reference.com, the same publisher whose data_use.html prohibits
automated bulk collection and building a substitute data store. A third party's
self-applied CC0 tag on the Kaggle mirror doesn't grant rights the uploader
doesn't hold, so this project does not ingest it. This test exists so a future
change can't silently flip that decision without deliberately editing this file.
"""
from src.historical_sources import load_source_registry
from src.ingest_nflverse import SOURCE_ID as NFLVERSE_SOURCE_ID

REQUIRED_FIELDS = ['source_id', 'name', 'url', 'access_method', 'coverage_start_season',
                   'coverage_end_season', 'status']


def test_every_source_has_required_fields():
    for entry in load_source_registry():
        for field in REQUIRED_FIELDS:
            assert entry.get(field) not in (None, ''), f'{entry.get("source_id")} missing {field}'
        if entry['status'] == 'not_ingested':
            assert entry.get('status_reason'), f'{entry["source_id"]} has no status_reason'


KNOWN_LEGITIMATE_INGESTED_SOURCES = {
    NFLVERSE_SOURCE_ID, 'nflverse_draft_picks', 'cfbd_player_season_stats',
}


def test_only_known_legitimate_sources_are_marked_ingested():
    ingested = {e['source_id'] for e in load_source_registry() if e['status'] == 'ingested'}
    # A guard, not a whitelist to copy-paste past: any new 'ingested' source_id must be
    # added here deliberately (and should have gone through the same access/license
    # review as the others), not just appear because someone flipped a status field.
    assert ingested == KNOWN_LEGITIMATE_INGESTED_SOURCES
    # The rejected Kaggle scrape must never end up in this set.
    assert 'zackthoutt_pfr_scrape_kaggle' not in ingested


def test_kaggle_pfr_scrape_candidate_is_rejected_with_documented_provenance():
    entries = {e['source_id']: e for e in load_source_registry()}
    candidate = entries['zackthoutt_pfr_scrape_kaggle']
    assert candidate['status'] == 'not_ingested'
    assert candidate['access_method'] == 'rejected_after_provenance_review'
    # The rejection must be traceable to the actual provenance finding, not a vague note.
    assert 'pro-football-reference.com' in candidate['license_notes']
    assert 'CC0' in candidate['license_notes']
    assert candidate['status_reason']
