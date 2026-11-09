"""Validation of the REAL Phase-3 dataset (data/qb_history.db), read-only.
Skipped automatically if the Phase-3 pipeline has not been run in this checkout.

These are data-quality assertions about the built dataset (identity, draft
assignment, transfers, duplicates, joins, combine missingness, outcome windows,
censoring, availability/leakage, preservation of existing NFL data) -- not
claims about model accuracy.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data' / 'qb_history.db'
DRAFT_DATES = json.loads((ROOT / 'config' / 'nfl_draft_dates.json').read_text(encoding='utf-8'))['draft_first_day']


def _conn():
    if not DB.exists():
        pytest.skip('real database not built')
    c = sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True)
    try:
        n = c.execute('select count(*) from prospect_feature_values').fetchone()[0]
    except sqlite3.OperationalError:
        c.close()
        pytest.skip('Phase-3 tables not built yet')
    if n == 0:
        c.close()
        pytest.skip('Phase-3 pipeline not run yet')
    return c


@pytest.fixture(scope='module')
def con():
    c = _conn()
    yield c
    c.close()


def q(con, sql):
    return pd.read_sql_query(sql, con)


# --- preservation of existing data -----------------------------------------------

def test_existing_nfl_statistics_preserved(con):
    # Phase-1 baseline (README): 820 players, 2,745 player-seasons, 54,900 era stats.
    assert q(con, 'select count(*) n from players')['n'][0] == 820
    assert q(con, 'select count(*) n from player_seasons_raw')['n'][0] == 2745
    assert q(con, 'select count(*) n from era_adjusted_stats')['n'][0] == 54900
    assert q(con, 'select count(*) n from cfb_player_seasons')['n'][0] == 40672


# --- identity / draft assignment ----------------------------------------------------

def test_every_prospect_has_exactly_one_identity_row(con):
    d = q(con, 'select p.id, count(m.id) n from qb_draft_prospects p left join college_nfl_player_map m '
               'on m.qb_draft_prospect_id = p.id group by p.id')
    assert (d['n'] == 1).all()


def test_draft_assignments_consistent_between_tables(con):
    d = q(con, 'select p.draft_year, p.pick, p.player_name, r.season, r.pick rpick, r.pfr_player_name '
               'from qb_draft_prospects p join nfl_draft_picks_raw r on r.id = p.draft_pick_raw_id')
    assert (d['draft_year'] == d['season']).all() and (d['pick'] == d['rpick']).all()
    assert (d['player_name'] == d['pfr_player_name']).all()
    assert not d.duplicated(['draft_year', 'pick']).any()


def test_cfbd_draft_record_agrees_on_who_was_picked(con):
    d = q(con, "select p.player_name, c.name from qb_draft_prospects p join cfbd_draft_picks c "
               "on c.year = p.draft_year and c.overall = p.pick")
    from src.match_prospects import _plausibly_same_person
    agree = d.apply(lambda r: _plausibly_same_person(r['player_name'], r['name']), axis=1)
    assert len(d) > 250 and agree.mean() > 0.95


def test_matched_college_school_overlaps_draft_college(con):
    from src.match_prospects import normalize_school, split_schools
    d = q(con, "select p.college, s.team from prospect_college_seasons s join qb_draft_prospects p "
               "on p.id = s.qb_draft_prospect_id where s.is_final_college_season = 1")
    ok = d.apply(lambda r: bool(set(split_schools(r['college'])) & {normalize_school(t) for t in str(r['team']).split('; ')}),
                 axis=1)
    assert ok.mean() > 0.9


@pytest.mark.parametrize('name,year,school', [('Joe Burrow', 2020, 'LSU'), ('Patrick Mahomes', 2017, 'Texas Tech'),
                                              ('Josh Allen', 2018, 'Wyoming'), ('Baker Mayfield', 2018, 'Oklahoma')])
def test_identity_spot_checks(con, name, year, school):
    d = q(con, f"select s.team from prospect_college_seasons s join qb_draft_prospects p on p.id = s.qb_draft_prospect_id "
               f"where p.player_name = '{name}' and p.draft_year = {year} and s.is_final_college_season = 1")
    assert d['team'].tolist() == [school]


def test_transfer_is_tracked_across_schools(con):
    # Baker Mayfield: Texas Tech (2013) -> Oklahoma (2015-2017)
    d = q(con, "select s.season, s.team from prospect_college_seasons s join qb_draft_prospects p "
               "on p.id = s.qb_draft_prospect_id where p.player_name = 'Baker Mayfield' order by s.season")
    assert d['team'].iloc[0] == 'Texas Tech' and d['team'].iloc[-1] == 'Oklahoma'
    v = q(con, "select v.value from prospect_feature_values v join qb_draft_prospects p on p.id = v.qb_draft_prospect_id "
               "where p.player_name = 'Baker Mayfield' and v.feature_name = 'transferred'")
    assert v['value'].tolist() == [1.0]


# --- duplicates / joins ---------------------------------------------------------------

def test_no_duplicate_player_season_observations(con):
    assert q(con, 'select count(*) n from (select qb_draft_prospect_id, season, count(*) c from prospect_college_seasons '
                  'group by 1, 2 having c > 1)')['n'][0] == 0
    assert q(con, 'select count(*) n from (select player_id, season, count(*) c from cfb_player_seasons '
                  'group by 1, 2 having c > 1)')['n'][0] == 0
    assert q(con, 'select count(*) n from (select qb_draft_prospect_id, nfl_season, count(*) c from nfl_prospect_seasons '
                  'group by 1, 2 having c > 1)')['n'][0] == 0


def test_offensive_profile_tables_unique_per_team_season(con):
    for t in ('cfb_team_season_environment', 'cfb_team_roster_talent', 'cfb_team_season_stats'):
        assert q(con, f'select count(*) n from (select season, team, count(*) c from {t} group by 1, 2 having c > 1)')['n'][0] == 0


def test_college_seasons_strictly_precede_draft(con):
    d = q(con, 'select s.season, p.draft_year from prospect_college_seasons s join qb_draft_prospects p '
               'on p.id = s.qb_draft_prospect_id')
    assert (d['season'] < d['draft_year']).all()


def test_box_scores_reconcile_with_season_totals(con):
    r = q(con, 'select box_season_attempt_ratio r from prospect_college_seasons where attempts >= 100')['r'].dropna()
    assert 0.9 <= r.median() <= 1.05


def test_rates_within_valid_ranges(con):
    d = q(con, 'select completion_pct, td_pct, int_pct, participation_rate, opp_adj_attempt_coverage '
               'from prospect_college_seasons where attempts >= 50')
    assert d['completion_pct'].between(0, 100).all()
    assert d['td_pct'].between(0, 30).all() and d['int_pct'].between(0, 30).all()
    assert d['participation_rate'].dropna().between(0, 1.0001).all()
    assert d['opp_adj_attempt_coverage'].dropna().between(0, 1.0001).all()


# --- combine / missing features ---------------------------------------------------------

def test_missing_combine_drills_are_absent_not_imputed(con):
    d = q(con, "select a.qb_draft_prospect_id from prospect_feature_values a "
               "where a.feature_name = 'combine_did_forty' and a.value = 0 and exists (select 1 from prospect_feature_values b "
               "where b.qb_draft_prospect_id = a.qb_draft_prospect_id and b.feature_name = 'combine_forty')")
    assert d.empty


def test_no_feature_value_stored_as_placeholder_for_missing(con):
    d = q(con, 'select count(*) n from prospect_feature_values where value is null')
    assert d['n'][0] == 0


def test_pre_coverage_classes_have_no_college_features(con):
    d = q(con, "select count(*) n from prospect_feature_values v join qb_draft_prospects p on p.id = v.qb_draft_prospect_id "
               "where p.draft_year <= 2004 and v.feature_name like 'final_%'")
    assert d['n'][0] == 0  # 2000-2004 classes' final college seasons (<= 2003) predate CFBD's 2004 start


# --- availability / leakage -------------------------------------------------------------

def test_every_pre_draft_feature_known_before_draft_day(con):
    d = q(con, "select v.available_as_of, p.draft_year, f.model_scopes from prospect_feature_values v "
               "join qb_draft_prospects p on p.id = v.qb_draft_prospect_id "
               "join feature_definitions f on f.feature_name = v.feature_name and f.feature_version = v.feature_version")
    d['draft_date'] = pd.to_datetime(d['draft_year'].astype(str).map(DRAFT_DATES))
    pre = d[d['model_scopes'].str.startswith('pre_draft')]
    assert (pd.to_datetime(pre['available_as_of']) < pre['draft_date']).all()
    post = d[~d['model_scopes'].str.startswith('pre_draft')]
    assert (pd.to_datetime(post['available_as_of']) >= post['draft_date']).all()


def test_registry_rows_match_code(con):
    from src.features import registry
    d = q(con, 'select feature_name, model_scopes from feature_definitions')
    assert set(d['feature_name']) == set(registry.FEATURES)
    assert d.set_index('feature_name').loc['draft_overall_pick', 'model_scopes'] == 'post_draft'


def test_age_at_draft_consistent_with_nflverse(con):
    d = q(con, "select v.value age, r.age_at_draft nfl_age from prospect_feature_values v "
               "join qb_draft_prospects p on p.id = v.qb_draft_prospect_id join nfl_draft_picks_raw r on r.id = p.draft_pick_raw_id "
               "where v.feature_name = 'age_at_draft' and r.age_at_draft is not null")
    assert len(d) > 200
    assert ((np.floor(d['age']) - d['nfl_age']).abs() <= 1).mean() > 0.97


# --- outcome windows / censoring ------------------------------------------------------------

def test_every_drafted_qb_has_an_outcome_row(con):
    assert q(con, 'select count(*) n from qb_draft_prospects')['n'][0] == q(con, 'select count(*) n from nfl_career_outcomes')['n'][0]


def test_five_year_targets_null_for_incomplete_windows(con):
    d = q(con, 'select p.draft_year, o.* from nfl_career_outcomes o join qb_draft_prospects p on p.id = o.qb_draft_prospect_id')
    inc = d[d['draft_year'] > 2021]
    assert not inc['window_5yr_complete'].astype(bool).any()
    assert inc['games_started_5yr'].isna().all() and inc['pass_yards_5yr'].isna().all()
    assert not (inc['started_32plus_5yr'] == 0).any()   # never a False label for an unfinished window
    comp = d[(d['draft_year'] <= 2021) & (d['career_end_status'] != 'unlinked_nfl_record')]
    assert comp['started_16plus_5yr'].notna().all()


def test_known_career_spot_checks(con):
    d = q(con, "select p.player_name, o.* from nfl_career_outcomes o join qb_draft_prospects p "
               "on p.id = o.qb_draft_prospect_id where (p.player_name = 'Patrick Mahomes' and p.draft_year = 2017) "
               "or (p.player_name = 'Tom Brady' and p.draft_year = 2000)").set_index('player_name')
    assert d.loc['Patrick Mahomes', 'games_started_5yr'] == 63      # 1 + 16 + 14 + 15 + 17
    assert d.loc['Patrick Mahomes', 'started_32plus_5yr'] == 1 and d.loc['Patrick Mahomes', 'started_64plus_5yr'] == 0
    assert d.loc['Tom Brady', 'started_any_game'] == 1
    assert d.loc['Tom Brady', 'career_end_status'] == 'inferred_inactive'   # last season 2022


def test_survival_censoring_consistent_with_outcomes(con):
    d = q(con, "select s.*, o.last_appearance_season, o.career_end_status from survival_records s "
               "join nfl_career_outcomes o on o.qb_draft_prospect_id = s.qb_draft_prospect_id "
               "where s.event_name = 'final_appearance_inferred'")
    recent = d[d['last_appearance_season'] >= 2024]
    assert not recent['event_observed'].astype(bool).any()          # recent players are censored
    assert (d['event_observed'].astype(bool) == (d['career_end_status'] == 'inferred_inactive')).all()
    assert (d['duration_seasons'] > 0).all()
