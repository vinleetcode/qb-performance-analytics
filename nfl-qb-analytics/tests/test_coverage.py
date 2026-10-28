from src import coverage
from src.db.schema import Player, PlayerSeasonRaw, get_engine, get_session, init_db
from src.historical_sources import not_ingested_season_ranges


def make_session():
    engine = init_db(get_engine('sqlite:///:memory:'))
    return get_session(engine)


def test_not_ingested_ranges_cover_the_known_gap():
    entries = not_ingested_season_ranges()
    ranges = {(e['coverage_start_season'], e['coverage_end_season']): e for e in entries}
    assert (1920, 1998) in ranges
    assert (1960, 1969) in ranges
    # Each gap entry must carry an explicit reason -- never a silent omission.
    for e in entries:
        assert e['status_reason']


def test_coverage_report_never_fabricates_rows_for_ungathered_seasons():
    session = make_session()
    player = Player(source_id='nflverse_player_stats', source_player_id='00-1', display_name='Test QB')
    session.add(player)
    session.flush()
    session.add(PlayerSeasonRaw(
        player_id=player.id, source_id='nflverse_player_stats', season=2023, league_id='NFL',
        season_type='REG', position_listed='QB', is_qb_listed=True, attempts=400,
    ))
    session.commit()

    # Use the real pipeline's full ingested range (not just [2023]) so the
    # rejected Kaggle candidate's claimed 1940-2017 range is actually clipped
    # against it, the way it is in a real run.
    ingested_seasons = list(range(1999, 2026))
    coverage.rebuild_coverage_table(session, 'nflverse_player_stats', ingested_seasons)
    summary = coverage.summarize(session)

    assert summary['totals']['distinct_seasons_ingested'] == len(ingested_seasons)
    # Distinct *season numbers* not ingested is still 79 (1920-1998): the AFL gap
    # (1960-1969) and the rejected Kaggle candidate's claimed 1940-2017 range both
    # fall inside -- or get clipped against the already-ingested 1999-2025 years
    # out of -- that same 1920-1998 span, so neither adds new distinct numbers here
    # even though each is tracked as its own row (see the three separate
    # not_ingested_ranges entries asserted below).
    assert summary['totals']['distinct_seasons_not_ingested'] == 79

    not_ingested = {(r['league_id'], r['first_season'], r['last_season']) for r in summary['not_ingested_ranges']}
    assert ('NFL', 1920, 1998) in not_ingested
    assert ('AFL', 1960, 1969) in not_ingested
    # The rejected Kaggle candidate's clipped range stops at 1998, never reaching
    # into the real, already-ingested 1999-2025 years it claimed to cover.
    assert ('NFL', 1940, 1998) in not_ingested

    by_decade = {(r['decade'], r['league_id'], r['status']): r for r in summary['by_decade_league']}
    assert by_decade[(1920, 'NFL', 'not_ingested')]['player_seasons'] == 0
    assert by_decade[(2020, 'NFL', 'ingested')]['player_seasons'] == 1


def test_afl_and_nfl_gap_rows_are_never_merged():
    session = make_session()
    coverage.rebuild_coverage_table(session, 'nflverse_player_stats', [])
    summary = coverage.summarize(session)
    leagues_in_1960s_gap = {r['league_id'] for r in summary['not_ingested_ranges']
                             if r['first_season'] <= 1969 <= r['last_season'] or
                             (r['first_season'] == 1960)}
    assert 'AFL' in leagues_in_1960s_gap
    assert 'NFL' in leagues_in_1960s_gap
