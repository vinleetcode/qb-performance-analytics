import pandas as pd

from src.ingest_cfbd import pivot_player_stats


def fixture_long():
    return pd.DataFrame([
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='passing', statType='COMP', stat='20'),
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='passing', statType='ATT', stat='30'),
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='passing', statType='YDS', stat='300'),
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='passing', statType='TD', stat='3'),
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='passing', statType='INT', stat='1'),
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='rushing', statType='CAR', stat='10'),
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='rushing', statType='YDS', stat='40'),
        # An unrecognized statType must be reported, not silently dropped or guessed.
        dict(season=2022, playerId='111', player='Test QB', position='QB', team='Georgia',
             conference='SEC', category='passing', statType='QBR', stat='88.5'),
    ])


def test_pivot_builds_wide_row_with_expected_columns():
    wide, unrecognized = pivot_player_stats(fixture_long())
    assert len(wide) == 1
    row = wide.iloc[0]
    assert row['completions'] == 20
    assert row['attempts'] == 30
    assert row['passing_yards'] == 300
    assert row['passing_tds'] == 3
    assert row['interceptions'] == 1
    assert row['rushing_attempts'] == 10
    assert row['rushing_yards'] == 40


def test_unrecognized_stat_types_are_reported_not_silently_dropped():
    _, unrecognized = pivot_player_stats(fixture_long())
    assert 'passing:QBR' in unrecognized


def test_empty_input_returns_empty_output_not_an_error():
    wide, unrecognized = pivot_player_stats(pd.DataFrame(
        columns=['season', 'playerId', 'player', 'position', 'team', 'conference', 'category',
                 'statType', 'stat']))
    assert wide.empty
    assert unrecognized == []


def test_mid_season_transfer_collapses_to_one_row_not_a_duplicate():
    # A real occurrence in CFBD data: the same playerId appears under two
    # different teams within the same season (e.g. a grad/mid-year transfer).
    # This must collapse to exactly one row per (season, playerId), with
    # stats summed and schools joined -- never two rows that would violate
    # the DB's one-row-per-player-season uniqueness constraint.
    long_df = pd.DataFrame([
        dict(season=2021, playerId='222', player='Transfer QB', position='QB', team='Tulane',
             conference='AAC', category='passing', statType='ATT', stat='50'),
        dict(season=2021, playerId='222', player='Transfer QB', position='QB', team='Tulane',
             conference='AAC', category='passing', statType='YDS', stat='400'),
        dict(season=2021, playerId='222', player='Transfer QB', position='QB', team='Houston',
             conference='AAC', category='passing', statType='ATT', stat='30'),
        dict(season=2021, playerId='222', player='Transfer QB', position='QB', team='Houston',
             conference='AAC', category='passing', statType='YDS', stat='250'),
    ])
    wide, _ = pivot_player_stats(long_df)
    assert len(wide) == 1
    row = wide.iloc[0]
    assert row['attempts'] == 80
    assert row['passing_yards'] == 650
    assert set(row['team'].split('; ')) == {'Tulane', 'Houston'}


def test_sum_of_all_missing_group_stays_null_not_fabricated_zero():
    # A pure rusher with passing attempts recorded as a rounding/charting
    # artifact in one stint and genuinely absent in another must not have
    # rushing stats fabricated as 0 when collapsing team-stints.
    long_df = pd.DataFrame([
        dict(season=2021, playerId='333', player='Runner', position='RB', team='Army',
             conference='Ind', category='rushing', statType='CAR', stat='15'),
        dict(season=2021, playerId='333', player='Runner', position='RB', team='Army',
             conference='Ind', category='rushing', statType='YDS', stat='90'),
        dict(season=2021, playerId='333', player='Runner', position='RB', team='Navy',
             conference='AAC', category='rushing', statType='CAR', stat='5'),
        dict(season=2021, playerId='333', player='Runner', position='RB', team='Navy',
             conference='AAC', category='rushing', statType='YDS', stat='20'),
    ])
    wide, _ = pivot_player_stats(long_df)
    row = wide.iloc[0]
    assert row['rushing_attempts'] == 20
    assert row['rushing_yards'] == 110
    assert 'attempts' not in wide.columns or pd.isna(row.get('attempts'))


def test_missing_statline_stays_null_not_zero():
    # This player has passing attempts but no rushing stats at all -- rushing
    # fields must come back as missing (NaN), never fabricated as 0.
    long_df = fixture_long()
    pass_only = long_df[long_df.category.eq('passing') & long_df.statType.ne('QBR')]
    wide, _ = pivot_player_stats(pass_only)
    row = wide.iloc[0]
    assert 'rushing_attempts' not in wide.columns or pd.isna(row.get('rushing_attempts'))
