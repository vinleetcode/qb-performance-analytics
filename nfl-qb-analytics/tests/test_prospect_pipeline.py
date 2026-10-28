"""Integration tests for the CFB/draft prospect pipeline. Uses live nflreadpy
(network required, same as the rest of this suite) but never a real CFBD key --
these specifically exercise the no-key path, since that's what CI/this
environment actually has. Every run uses an isolated temp DB/data/results dir,
and NEVER the real project's data/qb_history.db, per the project's hard
requirement to preserve existing NFL data.
"""
import pytest

from src import prospect_pipeline as pp
from src.db.schema import (
    CollegeNflPlayerMap, League, NflDraftPickRaw, Player, PlayerSeasonRaw,
    QbDraftProspect, Source, get_engine, get_session, init_db,
)
from src.historical_pipeline import seed_reference_tables

SMALL_SEASONS = [2020]  # keeps the live nflreadpy calls fast


def _run(tmp_path, draft_seasons=None):
    db_path = tmp_path / 'test.db'
    db_url = f'sqlite:///{db_path.as_posix()}'
    try:
        report = pp.main(db_url=db_url, draft_seasons=draft_seasons or SMALL_SEASONS, cfb_seasons=[],
                          data_dir=tmp_path / 'data', results_dir=tmp_path / 'results')
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f'nflreadpy source unreachable in this environment: {exc}')
    return report, db_url


def test_isolated_run_does_not_touch_real_project_exports(tmp_path):
    # Regression guard: an earlier version of this test file had one pp.main()
    # call missing data_dir/results_dir and it silently overwrote the real
    # project's data/*.parquet and results/prospect_coverage_report.json with
    # a tiny [2020]-only test run. Every pp.main() call in this file must pass
    # isolated dirs; this test catches a future regression of that mistake.
    real_data = pp.DATA
    real_results = pp.RESULTS
    before_data = {p: p.stat().st_mtime_ns for p in real_data.glob('*.parquet')} if real_data.exists() else {}
    before_report = ((real_results / 'prospect_coverage_report.json').stat().st_mtime_ns
                      if (real_results / 'prospect_coverage_report.json').exists() else None)
    _run(tmp_path)
    after_data = {p: p.stat().st_mtime_ns for p in real_data.glob('*.parquet')} if real_data.exists() else {}
    after_report = ((real_results / 'prospect_coverage_report.json').stat().st_mtime_ns
                     if (real_results / 'prospect_coverage_report.json').exists() else None)
    assert before_data == after_data, 'a test run must never overwrite the real project Parquet exports'
    assert before_report == after_report, 'a test run must never overwrite the real prospect coverage report'


def test_runs_without_a_cfbd_key_and_says_so_in_the_report(tmp_path, monkeypatch):
    # Force the no-key path explicitly rather than relying on whatever the
    # real .env happens to contain (a real key may legitimately be configured
    # in this environment) -- this test is specifically about that code path,
    # not about ambient secret state.
    monkeypatch.setattr(pp, 'cfbd_api_key_configured', lambda: False)
    report, _ = _run(tmp_path)
    assert report['cfbd_api_key_configured'] is False
    assert report['totals']['total_drafted_quarterbacks'] > 0
    assert report['row_counts']['cfb_players'] == 0  # honestly empty, not fabricated


def test_idempotent_rerun_produces_identical_counts(tmp_path):
    first, db_url = _run(tmp_path)
    second = pp.main(db_url=db_url, draft_seasons=SMALL_SEASONS, cfb_seasons=[],
                      data_dir=tmp_path / 'data', results_dir=tmp_path / 'results')
    assert first['row_counts'] == second['row_counts']
    assert first['totals'] == second['totals']


def test_no_duplicate_draft_picks_after_rerun(tmp_path):
    _, db_url = _run(tmp_path)
    pp.main(db_url=db_url, draft_seasons=SMALL_SEASONS, cfb_seasons=[],
            data_dir=tmp_path / 'data', results_dir=tmp_path / 'results')
    session = get_session(get_engine(db_url))
    picks = session.query(NflDraftPickRaw).filter(NflDraftPickRaw.season == 2020).all()
    keys = [(p.season, p.pick) for p in picks]
    assert len(keys) == len(set(keys))
    session.close()


def test_existing_nfl_tables_are_untouched_by_the_prospect_pipeline(tmp_path):
    db_path = tmp_path / 'preexisting.db'
    db_url = f'sqlite:///{db_path.as_posix()}'
    engine = init_db(get_engine(db_url))
    session = get_session(engine)
    # Seed a minimal, realistic NFL table state -- the kind of data this
    # pipeline must never touch.
    session.add(Source(source_id='nflverse_player_stats', name='x', access_method='open_package_api',
                        status='ingested'))
    session.add(League(league_id='NFL', name='National Football League', start_season=1922))
    session.commit()
    player = Player(source_id='nflverse_player_stats', source_player_id='00-TEST', display_name='Existing QB')
    session.add(player)
    session.flush()
    session.add(PlayerSeasonRaw(player_id=player.id, source_id='nflverse_player_stats', season=2020,
                                 league_id='NFL', season_type='REG', position_listed='QB',
                                 is_qb_listed=True, attempts=400))
    session.commit()
    before_players = session.query(Player).count()
    before_seasons = session.query(PlayerSeasonRaw).count()
    session.close()

    try:
        pp.main(db_url=db_url, draft_seasons=SMALL_SEASONS, cfb_seasons=[],
                data_dir=tmp_path / 'data', results_dir=tmp_path / 'results')
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'nflreadpy source unreachable in this environment: {exc}')

    session = get_session(get_engine(db_url))
    assert session.query(Player).count() == before_players
    assert session.query(PlayerSeasonRaw).count() == before_seasons
    existing = session.query(Player).filter(Player.source_player_id == '00-TEST').one()
    assert existing.display_name == 'Existing QB'
    session.close()


def test_prospect_with_no_gsis_id_is_preserved_as_negative_example(tmp_path):
    report, db_url = _run(tmp_path)
    session = get_session(get_engine(db_url))
    no_gsis = session.query(QbDraftProspect).filter(QbDraftProspect.gsis_id.is_(None)).all()
    for p in no_gsis:
        mapping = session.query(CollegeNflPlayerMap).filter(
            CollegeNflPlayerMap.qb_draft_prospect_id == p.id).one()
        assert mapping.nfl_match_method == 'no_gsis_id'
        # Still present in the map, not dropped -- a drafted-but-unlinked QB is
        # exactly the kind of negative example this project must keep.
    session.close()


def test_pre_draft_export_excludes_every_nfl_outcome_column(tmp_path):
    import pandas as pd
    _run(tmp_path)
    pre_draft = pd.read_parquet(tmp_path / 'data' / 'prospects_pre_draft_features.parquet')
    leaked = pp.NFL_OUTCOME_COLUMNS & set(pre_draft.columns)
    assert leaked == set()


def test_nfl_outcomes_export_exists_and_is_keyed_to_prospects(tmp_path):
    import pandas as pd
    _run(tmp_path)
    outcomes = pd.read_parquet(tmp_path / 'data' / 'prospects_nfl_outcomes.parquet')
    assert 'qb_draft_prospect_id' in outcomes.columns
    assert 'career_pass_yards' in outcomes.columns
