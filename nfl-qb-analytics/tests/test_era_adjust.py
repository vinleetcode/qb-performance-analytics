import numpy as np
import pandas as pd

from src.era_adjust import rolling_era_baseline, season_league_baseline


def fixture():
    # Two leagues' worth of rows across a few seasons so grouping/isolation can be checked.
    return pd.DataFrame([
        dict(player_season_id=1, season=2020, league_id='NFL', attempts=300, qualifies_leaderboard=True, ypa=8.0),
        dict(player_season_id=2, season=2020, league_id='NFL', attempts=300, qualifies_leaderboard=True, ypa=6.0),
        dict(player_season_id=3, season=2020, league_id='NFL', attempts=10, qualifies_leaderboard=False, ypa=20.0),
        dict(player_season_id=4, season=2021, league_id='NFL', attempts=300, qualifies_leaderboard=True, ypa=7.0),
        dict(player_season_id=5, season=2020, league_id='AFL', attempts=300, qualifies_leaderboard=True, ypa=3.0),
    ])


def test_season_league_baseline_excludes_unqualified_rows_from_mean():
    df = fixture()
    scored = season_league_baseline(df, 'ypa', weight_col='attempts')
    nfl_2020 = scored[(scored.season == 2020) & (scored.league_id == 'NFL')].set_index('player_season_id')
    # Mean should be based only on the two qualifying rows (8.0, 6.0) -> 7.0, not the
    # 20.0 outlier from the unqualified low-volume row.
    assert np.isclose(nfl_2020.loc[1, 'baseline_mean'], 7.0)
    # The unqualified row is still scored against that baseline (never dropped from output).
    assert nfl_2020.loc[3, 'baseline_mean'] == nfl_2020.loc[1, 'baseline_mean']
    assert not np.isnan(nfl_2020.loc[3, 'z_score'])


def test_season_league_baseline_never_pools_afl_with_nfl():
    df = fixture()
    scored = season_league_baseline(df, 'ypa', weight_col='attempts')
    nfl_mean = scored[(scored.season == 2020) & (scored.league_id == 'NFL')]['baseline_mean'].iloc[0]
    afl_mean = scored[(scored.season == 2020) & (scored.league_id == 'AFL')]['baseline_mean'].iloc[0]
    assert afl_mean != nfl_mean
    assert np.isclose(afl_mean, 3.0)


def test_rolling_era_baseline_pools_the_trailing_window():
    df = fixture()
    scored = rolling_era_baseline(df, 'ypa', weight_col='attempts', window_years=5, group_cols=('league_id',))
    row_2021 = scored[scored.player_season_id == 4].iloc[0]
    # window_years=5 with half=2 centered on 2021 includes seasons 2019-2023; within
    # this fixture that pulls in the two qualifying 2020 NFL rows (8.0, 6.0) plus the
    # 2021 row itself (7.0) -> weighted mean 7.0.
    assert np.isclose(row_2021['baseline_mean'], 7.0)
