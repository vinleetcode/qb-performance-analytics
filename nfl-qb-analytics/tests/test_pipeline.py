import pandas as pd
from src.pipeline import prepare, add_era_scores
from src.model import make_examples


def fixture():
    return pd.DataFrame([
      dict(player_id='a', player_display_name='A', position='QB', season=2020, attempts=200, passing_yards=1400, passing_tds=10, passing_interceptions=5),
      dict(player_id='b', player_display_name='B', position='QB', season=2020, attempts=200, passing_yards=1200, passing_tds=8, passing_interceptions=8),
      dict(player_id='a', player_display_name='A', position='QB', season=2021, attempts=250, passing_yards=2000, passing_tds=15, passing_interceptions=4),
      dict(player_id='b', player_display_name='B', position='QB', season=2021, attempts=150, passing_yards=900, passing_tds=7, passing_interceptions=10),
    ])


def test_era_score_prefers_stronger_qb():
    frame = add_era_scores(prepare(fixture()))
    for year in (2020, 2021):
        q = frame[frame.season.eq(year)].set_index('player_id')
        assert q.loc['a', 'era_score'] > q.loc['b', 'era_score']


def test_next_year_targets_do_not_leak_future():
    frame = add_era_scores(prepare(fixture()))
    pairs = make_examples(frame)
    assert len(pairs) == 2
    assert (pairs.target_season == pairs.season + 1).all()
