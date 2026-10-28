"""Integration test against the real nflreadpy source. Needs network access;
skips (rather than fails) if the source is unreachable, since that is an
environment condition, not a code defect."""
import pytest

from src import historical_pipeline as hp


def _run_two_seasons(tmp_path):
    db_path = tmp_path / 'qb_history_test.db'
    db_url = f'sqlite:///{db_path.as_posix()}'
    # data_dir/results_dir MUST be isolated from the project's real data/ and
    # results/ directories -- otherwise this 2-season test run would clobber
    # the full 1999-2025 Parquet exports and coverage report (this happened
    # once during development; see the data_dir/results_dir params on main()).
    try:
        return hp.main(db_url=db_url, seasons=[2023, 2024],
                        data_dir=tmp_path / 'data', results_dir=tmp_path / 'results'), db_url
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f'nflreadpy source unreachable in this environment: {exc}')


def test_isolated_run_does_not_touch_real_project_exports(tmp_path):
    real_data = hp.DATA
    before = {p: p.stat().st_mtime_ns for p in real_data.glob('*.parquet')} if real_data.exists() else {}
    _run_two_seasons(tmp_path)
    after = {p: p.stat().st_mtime_ns for p in real_data.glob('*.parquet')} if real_data.exists() else {}
    assert before == after, 'a test run must never overwrite the real project Parquet exports'


def test_pipeline_ingests_only_requested_seasons(tmp_path):
    summary, _ = _run_two_seasons(tmp_path)
    assert summary['totals']['distinct_seasons_ingested'] == 2
    assert summary['totals']['player_season_observations'] > 0
    assert summary['totals']['unique_players'] > 0


def test_pipeline_is_idempotent_across_repeated_runs(tmp_path):
    first, db_url = _run_two_seasons(tmp_path)
    second = hp.main(db_url=db_url, seasons=[2023, 2024],
                      data_dir=tmp_path / 'data', results_dir=tmp_path / 'results')
    assert first['totals'] == second['totals']


def test_known_leader_matches_real_2024_passer_rating_order(tmp_path):
    from src.db.schema import Player, PlayerSeasonDerived, PlayerSeasonRaw, get_engine, get_session

    _, db_url = _run_two_seasons(tmp_path)
    session = get_session(get_engine(db_url))
    top = (session.query(Player.display_name, PlayerSeasonDerived.passer_rating)
           .join(PlayerSeasonRaw, PlayerSeasonRaw.player_id == Player.id)
           .join(PlayerSeasonDerived, PlayerSeasonDerived.player_season_id == PlayerSeasonRaw.id)
           .filter(PlayerSeasonRaw.season == 2024, PlayerSeasonDerived.qualifies_leaderboard.is_(True))
           .order_by(PlayerSeasonDerived.passer_rating.desc())
           .first())
    session.close()
    # Lamar Jackson led the NFL in passer rating in the real 2024 season.
    assert top is not None
    assert top[0] == 'Lamar Jackson'


def test_rushing_stats_are_ingested_for_a_known_dual_threat_qb(tmp_path):
    from src.db.schema import Player, PlayerSeasonDerived, PlayerSeasonRaw, get_engine, get_session

    _, db_url = _run_two_seasons(tmp_path)
    session = get_session(get_engine(db_url))
    row = (session.query(PlayerSeasonRaw, PlayerSeasonDerived)
           .join(Player, Player.id == PlayerSeasonRaw.player_id)
           .join(PlayerSeasonDerived, PlayerSeasonDerived.player_season_id == PlayerSeasonRaw.id)
           .filter(Player.display_name == 'Lamar Jackson', PlayerSeasonRaw.season == 2024)
           .first())
    session.close()
    assert row is not None
    raw, derived = row
    assert raw.carries is not None and raw.carries > 0
    assert raw.rushing_yards is not None and raw.rushing_yards > 500  # known: a dual-threat season
    assert derived.yards_per_carry is not None
    assert derived.total_yards == raw.passing_yards + raw.rushing_yards
