"""Unit tests for Phase-3 feature engineering, targets, survival, leakage guards
and modelling-preparation utilities. Pure synthetic data -- no network, no DB."""
import numpy as np
import pandas as pd
import pytest

from src import ingest_cfbd_advanced as A
from src.features import athletic, development, environment, opponent_adjust, registry, targets
from src.match_prospects import MatchResult, reconcile_with_cfbd_draft_id
from src.modeling import evaluation, preprocess, similarity, survival


# ---------------------------------------------------------------------------
# player identity matching
# ---------------------------------------------------------------------------

def test_cfbd_id_confirms_existing_name_match():
    r = reconcile_with_cfbd_draft_id('Joe Burrow', MatchResult(7, 'exact_name_school', 1.0, None), '3915511', 7, 'Joe Burrow')
    assert r.cfb_player_id == 7 and r.method == 'exact_name_school'
    assert 'confirmed by cfbd_draft_athlete_id' in r.notes


def test_cfbd_id_bridges_nickname_the_name_matcher_missed():
    r = reconcile_with_cfbd_draft_id('Cam Ward', MatchResult(None, 'unmatched', None, 'x'), '999', 42, 'Cameron Ward')
    assert r.cfb_player_id == 42 and r.method == 'cfbd_draft_athlete_id'


def test_cfbd_id_rejected_when_names_clearly_differ():
    r = reconcile_with_cfbd_draft_id('Cam Ward', MatchResult(None, 'unmatched', None, None), '999', 42, 'Bob Smith')
    assert r.cfb_player_id is None and 'rejected' in r.notes


def test_conflicting_name_match_and_cfbd_id_is_ambiguous_not_guessed():
    r = reconcile_with_cfbd_draft_id('John Doe', MatchResult(5, 'fuzzy_name_school', 0.93, None), '1', 6, 'John Doe')
    assert r.cfb_player_id is None and r.method == 'ambiguous'


def test_cfbd_id_without_stat_rows_keeps_name_result():
    base = MatchResult(None, 'unmatched', None, None)
    r = reconcile_with_cfbd_draft_id('Cam Newton', base, '123', None, None)
    assert r.cfb_player_id is None and 'no CFBD season-stat rows' in r.notes


def test_no_stable_id_is_a_no_op():
    base = MatchResult(3, 'exact_name_school', 1.0, None)
    assert reconcile_with_cfbd_draft_id('X Y', base, None, None, None) is base


# ---------------------------------------------------------------------------
# box-score parsing / invalid statistical calculations
# ---------------------------------------------------------------------------

def _payload():
    return [{'id': 1, 'teams': [
        {'team': 'A', 'conference': 'C1', 'homeAway': 'home', 'points': 21, 'categories': [{'name': 'passing', 'types': [
            {'name': 'C/ATT', 'athletes': [{'id': '10', 'name': 'QB A', 'stat': '15/20'}, {'id': '-1', 'name': 'Team', 'stat': '0/1'}]},
            {'name': 'YDS', 'athletes': [{'id': '10', 'name': 'QB A', 'stat': '200'}, {'id': '-1', 'name': 'Team', 'stat': '0'}]},
            {'name': 'TD', 'athletes': [{'id': '10', 'name': 'QB A', 'stat': '2'}]},
            {'name': 'INT', 'athletes': [{'id': '10', 'name': 'QB A', 'stat': '--'}]}]}]},
        {'team': 'B', 'conference': 'C1', 'homeAway': 'away', 'points': 7, 'categories': []}]}]


def test_parse_game_passing_splits_comp_att_and_keeps_bad_stats_null():
    df = A.parse_game_passing(_payload(), 2019, 1, 'regular')
    qb = df[df['cfbd_player_id'] == '10'].iloc[0]
    assert (qb['completions'], qb['attempts'], qb['passing_yards'], qb['passing_tds']) == (15, 20, 200, 2)
    assert pd.isna(qb['interceptions'])  # '--' is not silently 0
    assert qb['opponent'] == 'B'
    team_row = df[df['player_name'] == 'Team'].iloc[0]
    assert pd.isna(team_row['cfbd_player_id'])  # never attributable to a QB


def test_cfbd_call_budget_blocks_network_before_request(tmp_path, monkeypatch):
    from src.cfbd_client import CFBDBudgetExceeded, CFBDClient
    c = CFBDClient(cache_dir=tmp_path, api_key='dummy-not-real', max_network_calls=0)
    monkeypatch.setattr(c._session, 'get', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network used')))
    with pytest.raises(CFBDBudgetExceeded) as exc:
        c.get('/games', {'year': 2019})
    assert 'dummy-not-real' not in str(exc.value)
    (tmp_path / c._cache_key('/games', {'year': 2019})).write_text('[]')
    assert c.get('/games', {'year': 2019}) == [] and c.cache_hits == 1  # cache reads are free


def test_game_weeks_come_from_schedule_not_hardcoded():
    games = pd.DataFrame({'season': [2020] * 3, 'seasonType': ['regular', 'regular', 'postseason'], 'week': [1, 17, 1],
                          'homeClassification': ['fbs', 'fbs', 'fbs'], 'awayClassification': ['fcs', 'fbs', 'fbs']})
    assert A.game_weeks(games, 2020) == [('postseason', 1), ('regular', 1), ('regular', 17)]


def test_weighted_slope_known_values():
    assert development.weighted_slope([0, 1, 2], [100, 110, 120], [1, 1, 1]) == pytest.approx(10.0)
    assert np.isnan(development.weighted_slope([0], [1], [1]))           # 1 point
    assert np.isnan(development.weighted_slope([1, 1], [1, 2], [1, 1]))  # no x variation


def test_parse_height():
    assert athletic.parse_height('6-4') == 76
    assert athletic.parse_height('5-11') == 71
    assert np.isnan(athletic.parse_height(None)) and np.isnan(athletic.parse_height('tall'))


def _league():
    """4 FBS teams, round robin over 3 'weeks' + repeat => each defense has 6 games.
    Defense D allows exactly 10.0 YPA in every game; others allow 6.0."""
    rows, gid = [], 0
    teams = ['A', 'B', 'C', 'D']
    for rep in range(2):
        for i, a in enumerate(teams):
            for b in teams[i + 1:]:
                gid += 1
                for off, de in ((a, b), (b, a)):
                    ypa = 10.0 if de == 'D' else 6.0
                    rows.append({'game_id': gid, 'season': 2019, 'week': gid, 'season_type': 'regular', 'team': off,
                                 'conference': 'X', 'home_away': 'home', 'opponent': de, 'opponent_conference': 'X',
                                 'cfbd_player_id': f'qb_{off}', 'player_name': off, 'completions': 12, 'attempts': 20,
                                 'passing_yards': int(20 * ypa), 'passing_tds': 1, 'interceptions': 1})
    gp = pd.DataFrame(rows)
    fbs = pd.DataFrame({'season': [2019] * 4, 'team': teams})
    return gp, fbs


def test_leave_one_out_excludes_the_game_itself():
    gp, _ = _league()
    tg = opponent_adjust.team_game_totals(gp)
    loo = opponent_adjust.defense_allowed_loo(tg)
    row = loo[(loo['game_id'] == 1) & (loo['defense'] == 'B')].iloc[0]
    season = tg[tg['defense'] == 'B']
    assert row['loo_attempts'] == season['attempts'].sum() - 20
    assert row['loo_games'] == season['game_id'].nunique() - 1


def test_opponent_adjustment_arithmetic():
    gp, fbs = _league()
    qg = opponent_adjust.qb_game_adjustments(gp, fbs)
    # league mean allowed YPA = (3 defenses * 6.0 + 1 * 10.0) / 4 = 7.0 (equal attempts)
    vs_d = qg[(qg['team'] == 'A') & (qg['opponent'] == 'D')].iloc[0]
    assert vs_d['league_ypa'] == pytest.approx(7.0)
    assert vs_d['opp_ypa'] == pytest.approx(10.0)          # LOO baseline of D
    assert vs_d['adj_ypa'] == pytest.approx(10.0 - (10.0 - 7.0))  # 7.0: torching a bad defense is discounted
    vs_b = qg[(qg['team'] == 'A') & (qg['opponent'] == 'B')].iloc[0]
    assert vs_b['adj_ypa'] == pytest.approx(6.0 - (6.0 - 7.0))     # 7.0
    # D's pass defense is the worst -> ranked last
    sdef = opponent_adjust.season_defense_table(opponent_adjust.team_game_totals(gp), fbs)
    assert sdef.set_index('defense').loc['D', 'def_pass_rank'] == 4


def test_opponent_without_enough_other_games_gets_no_baseline():
    gp, fbs = _league()
    extra = gp.iloc[[0]].copy()
    extra[['game_id', 'opponent']] = [999, 'FCS Team']
    qg = opponent_adjust.qb_game_adjustments(pd.concat([gp, extra]), fbs)
    row = qg[qg['game_id'] == 999].iloc[0]
    assert not row['has_opp_baseline'] and np.isnan(row['adj_ypa'])


def test_qb_season_aggregation_coverage_and_leading_passer():
    gp, fbs = _league()
    backup = gp[gp['team'] == 'A'].head(2).copy()
    backup[['cfbd_player_id', 'attempts', 'completions', 'passing_yards']] = ['qb_A2', 3, 1, 10]
    gp2 = pd.concat([gp, backup])
    qg = opponent_adjust.qb_game_adjustments(gp2, fbs)
    s = opponent_adjust.qb_season_from_games(qg, opponent_adjust.team_game_totals(gp2)).set_index('cfbd_player_id')
    assert s.loc['qb_A', 'games_with_pass_attempt'] == 6
    assert s.loc['qb_A', 'games_as_leading_passer'] == 6
    assert s.loc['qb_A2', 'games_as_leading_passer'] == 0
    assert s.loc['qb_A', 'opp_adj_attempt_coverage'] == pytest.approx(1.0)


def test_conference_relative_z_scores_ignore_small_samples():
    df = pd.DataFrame({'season': 2019, 'conference': 'X', 'cfbd_player_id': list('abcd'),
                       'attempts': [300, 300, 300, 20], 'ncaa_passer_rating': [120.0, 140.0, 160.0, 300.0],
                       'yards_per_attempt': [6.0, 7.0, 8.0, 15.0]})
    z = opponent_adjust.conference_relative(df).set_index('cfbd_player_id')
    assert z.loc['b', 'conf_rel_rating_z'] == pytest.approx(0.0)
    assert z.loc['c', 'conf_rel_rating_z'] > 0
    assert np.isnan(z.loc['d', 'conf_rel_rating_z'])   # 20 attempts: excluded, not compared


# ---------------------------------------------------------------------------
# development / transfers / missing features / ages
# ---------------------------------------------------------------------------

def _seasons(rows):
    cols = ['qb_draft_prospect_id', 'season', 'team', 'attempts', 'completions', 'passing_yards', 'passing_tds',
            'interceptions', 'rushing_attempts', 'rushing_yards', 'rushing_tds', 'completion_pct', 'yards_per_attempt',
            'td_pct', 'int_pct', 'ncaa_passer_rating', 'opp_adj_ypa', 'games_with_pass_attempt',
            'games_as_leading_passer', 'team_games_played']
    return pd.DataFrame(rows, columns=cols)


def test_development_separates_volume_from_efficiency_and_handles_transfer():
    s = _seasons([
        (1, 2016, 'School A', 120, 70, 840, 6, 4, 30, 100, 1, 58.3, 7.0, 5.0, 3.3, 130.0, 7.1, 8, 6, 12),
        (1, 2017, 'School A', 300, 180, 2400, 20, 8, 50, 150, 2, 60.0, 8.0, 6.7, 2.7, 145.0, 8.0, 12, 12, 12),
        (1, 2018, 'School B', 400, 260, 3600, 32, 6, 60, 250, 4, 65.0, 9.0, 8.0, 1.5, 165.0, 8.9, 13, 13, 13),
    ])
    med = pd.Series({2016: 135.0, 2017: 135.0, 2018: 135.0})
    d = development.prospect_development(s, med).iloc[0]
    assert d['dev_slope_rating'] > 0 and d['dev_slope_attempts'] > 0
    assert d['dev_yoy_rating_final'] == pytest.approx(20.0)
    assert d['n_schools'] == 2 and d['transferred'] == 1
    assert d['n_productive_seasons'] == 2  # 2016 too few attempts AND below median
    assert d['seasons_as_primary_passer'] == 3
    assert d['career_possibly_truncated'] == 0


def test_small_samples_do_not_produce_trends():
    s = _seasons([
        (2, 2017, 'A', 40, 20, 300, 2, 1, 5, 10, 0, 50.0, 7.5, 5.0, 2.5, 130.0, np.nan, 3, 0, 12),
        (2, 2018, 'A', 350, 220, 3000, 25, 7, 40, 120, 2, 62.9, 8.6, 7.1, 2.0, 155.0, 8.4, 12, 12, 12),
    ])
    d = development.prospect_development(s, pd.Series({2017: 135.0, 2018: 135.0})).iloc[0]
    assert np.isnan(d['dev_slope_rating'])      # only one season >= 100 attempts
    assert pd.isna(d.get('dev_yoy_rating_final'))


def test_2004_first_season_flags_possible_truncation():
    s = _seasons([(3, 2004, 'A', 350, 220, 3000, 25, 7, 40, 120, 2, 62.9, 8.6, 7.1, 2.0, 155.0, 8.4, 12, 12, 12)])
    assert development.prospect_development(s, pd.Series({2004: 135.0})).iloc[0]['career_possibly_truncated'] == 1


def test_missing_birthdate_gives_missing_age_not_a_guess():
    p = pd.DataFrame({'qb_draft_prospect_id': [1, 2], 'draft_year': [2018, 2018],
                      'birthdate': [pd.Timestamp('1996-05-29').date(), None]})
    ages = development.prospect_ages(p, {'2018': '2018-04-26'}, pd.Series({1: 2017, 2: 2017})).set_index('qb_draft_prospect_id')
    assert ages.loc[1, 'age_at_draft'] == pytest.approx(21.91, abs=0.01)
    assert np.isnan(ages.loc[2, 'age_at_draft']) and np.isnan(ages.loc[2, 'age_final_college_season'])


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

def test_combine_skipped_drills_stay_null_and_absence_is_not_poor_performance():
    combine = athletic.standardize_combine(pd.DataFrame({
        'season': [2018, 2018, 2017], 'pfr_id': ['AllJo00', None, 'OldGu00'], 'cfb_id': [None, None, None],
        'player_name': ['Josh Allen', 'Someone Else', 'Old Guy'], 'pos': ['QB', 'QB', 'QB'],
        'school': ['Wyoming', 'X', 'Y'], 'ht': ['6-5', '6-2', '6-1'], 'wt': [237, 220, 210],
        'forty': [4.75, None, 4.9], 'bench': [None, None, None], 'vertical': [33.5, None, 30],
        'broad_jump': [119, None, 110], 'cone': [6.9, None, 7.1], 'shuttle': [4.4, None, 4.3],
        'draft_year': [2018, None, 2017], 'draft_ovr': [7, None, 100]}))
    prospects = pd.DataFrame({'qb_draft_prospect_id': [1, 2, 3], 'draft_year': [2018, 2018, 2018],
                              'pfr_player_id': ['AllJo00', None, 'OldGu00'], 'cfb_player_id_slug': [None, None, None],
                              'player_name': ['Josh Allen', 'No Combine', 'Old Guy']})
    j = athletic.join_combine_to_prospects(prospects, combine).set_index('qb_draft_prospect_id')
    assert j.loc[1, 'combine_height_inches'] == 77 and j.loc[1, 'combine_did_forty'] == 1
    assert np.isnan(j.loc[1, 'combine_bench']) and j.loc[1, 'combine_did_bench'] == 0
    assert j.loc[2, 'combine_record_found'] == 0 and np.isnan(j.loc[2, 'combine_forty'])
    assert np.isnan(j.loc[2, 'combine_did_forty'])  # unknown participation, not "did not run"
    # a combine row from a different year (2017) is never attached to a 2018 draftee
    assert j.loc[3, 'combine_record_found'] == 0


# ---------------------------------------------------------------------------
# environment joins
# ---------------------------------------------------------------------------

def test_receiver_context_concentration_and_ratings():
    recv = pd.DataFrame({'season': 2019, 'team': 'A', 'cfbd_player_id': ['1', '2', '3', '4'],
                         'receptions': [50, 30, 15, 5], 'receiving_yards': [700, 400, 150, 50]})
    recruits = pd.DataFrame({'athlete_id': ['1', '2', None], 'rating': [0.95, 0.85, 0.99]})
    r = environment.receiver_context(recv, recruits).iloc[0]
    assert r['top_target_reception_share'] == pytest.approx(0.5)
    assert r['top3_reception_share'] == pytest.approx(0.95)
    assert r['top3_receivers_avg_recruit_rating'] == pytest.approx(0.90) and r['top3_receivers_rated'] == 2


def test_advanced_metrics_blanked_when_play_by_play_incomplete():
    ts = pd.DataFrame({'season': [2004, 2019], 'team': ['A', 'A'], 'pass_attempts': [400, 400],
                       'rushing_attempts': [450, 450], 'off_plays': [158, 840], 'off_line_yards': [2.9, 3.1]})
    g = environment.gate_advanced(ts)
    assert not g.loc[0, 'advanced_reliable'] and np.isnan(g.loc[0, 'off_line_yards'])
    assert g.loc[1, 'advanced_reliable'] and g.loc[1, 'off_line_yards'] == 3.1


def test_team_environment_has_one_row_per_team_season():
    teams = ['A', 'B', 'C']
    ts = pd.DataFrame({'season': 2019, 'team': teams, 'games': 12, 'pass_attempts': 400, 'rushing_attempts': 450,
                       'possession_seconds': 12 * 1800, 'sacks_allowed': [20, 30, 40], 'def_sacks': [25, 30, 35],
                       'opp_pass_attempts': 380, 'rushing_yards': 2000, 'off_plays': 860})
    for c in A.TEAM_STAT_MAP.values():
        if c not in ts.columns:
            ts[c] = np.nan
    games = pd.DataFrame({'season': 2019, 'homeTeam': ['A', 'B', 'A'], 'awayTeam': ['B', 'C', 'C'], 'completed': True})
    sdef = pd.DataFrame({'season': 2019, 'defense': teams, 'ypa': [6, 7, 8], 'cmp_pct': [55, 60, 65],
                         'def_pass_rank': [1, 2, 3]})
    env = environment.team_environment(ts, games, pd.DataFrame(columns=['season', 'team']), sdef, None)
    assert len(env) == 3 and not env.duplicated(['season', 'team']).any()
    a = env.set_index('team').loc['A']
    assert a['sack_rate_allowed'] == pytest.approx(20 / 420)
    assert a['plays_per_game'] == pytest.approx(850 / 12)


# ---------------------------------------------------------------------------
# NFL outcome windows and survival censoring
# ---------------------------------------------------------------------------

def _nfl_fixture():
    prospects = pd.DataFrame({'qb_draft_prospect_id': [1, 2, 3, 4, 5],
                              'draft_year': [2015, 2023, 2015, 2015, 2024],
                              'gsis_id': ['g1', 'g2', 'g3', None, 'g5']})
    weekly = pd.DataFrame([
        # p1: starter 2016-2018, last appearance 2019 (>= 2 seasons before 2025 -> inferred end)
        *[{'gsis_id': 'g1', 'season': s, 'week': w, 'attempts': 30, 'completions': 20, 'passing_yards': 250,
           'passing_tds': 2, 'passing_interceptions': 1, 'sacks_suffered': 2, 'sack_yards_lost': 12}
          for s in (2016, 2017, 2018, 2019) for w in range(1, 17)],
        # p2: 2023 draftee, 20 starts by 2024 (window incomplete but 16+ already achieved)
        *[{'gsis_id': 'g2', 'season': s, 'week': w, 'attempts': 25, 'completions': 15, 'passing_yards': 180,
           'passing_tds': 1, 'passing_interceptions': 1, 'sacks_suffered': 1, 'sack_yards_lost': 5}
          for s, n in ((2023, 4), (2024, 16)) for w in range(1, n + 1)],
    ])
    starts = pd.DataFrame([{'gsis_id': 'g1', 'season': s, 'game_type': 'REG', 'games_started': 16} for s in (2016, 2017, 2018)]
                          + [{'gsis_id': 'g2', 'season': 2023, 'game_type': 'REG', 'games_started': 4},
                             {'gsis_id': 'g2', 'season': 2024, 'game_type': 'REG', 'games_started': 16}])
    status = pd.DataFrame({'gsis_id': ['g1'], 'status': ['RET']})
    pfr = pd.DataFrame({'qb_draft_prospect_id': [4], 'pfr_last_season': [None], 'pfr_games': [None]})
    seasons = targets.build_prospect_seasons(prospects, targets.weekly_to_season_totals(weekly), starts, obs_end=2025)
    out = targets.build_career_outcomes(prospects, seasons, status, pfr, obs_end=2025).set_index('qb_draft_prospect_id')
    return prospects, seasons, out


def test_prospect_season_grid_has_explicit_zero_seasons():
    _, seasons, _ = _nfl_fixture()
    p3 = seasons[seasons['qb_draft_prospect_id'] == 3]
    assert len(p3) == 11 and p3['games_with_stats'].sum() == 0   # 2015..2025, never played


def test_five_year_window_targets():
    _, _, out = _nfl_fixture()
    p1 = out.loc[1]
    assert p1['window_5yr_complete'] and p1['games_started_5yr'] == 48
    assert p1['started_32plus_5yr'] and not p1['started_64plus_5yr']
    assert p1['any_a_5yr'] is not None and not np.isnan(p1['any_a_5yr'])


def test_incomplete_window_is_censored_not_failure():
    _, _, out = _nfl_fixture()
    p2 = out.loc[2]
    assert not p2['window_5yr_complete']
    assert p2['started_16plus_5yr'] is True           # already achieved -> known positive
    assert pd.isna(p2['started_32plus_5yr'])          # not reached YET -> NULL, never False
    assert pd.isna(p2['games_started_5yr']) and pd.isna(p2['pass_yards_5yr'])


def test_drafted_qbs_who_never_played_are_preserved():
    _, _, out = _nfl_fixture()
    assert out.loc[3, 'career_end_status'] == 'never_appeared_window_complete'
    assert out.loc[3, 'started_1plus_5yr'] is False or out.loc[3, 'started_1plus_5yr'] == False  # noqa: E712
    assert out.loc[4, 'career_end_status'] == 'never_appeared_window_complete'  # no gsis id, PFR agrees
    assert out.loc[5, 'career_end_status'] == 'never_appeared_censored'
    assert pd.isna(out.loc[5, 'started_any_game'])


def test_survival_censoring_rules():
    prospects, seasons, out = _nfl_fixture()
    s = targets.build_survival_records(out.reset_index(), seasons, prospects, obs_end=2025)
    get = lambda pid, ev: s[(s['qb_draft_prospect_id'] == pid) & (s['event_name'] == ev)].iloc[0]  # noqa: E731
    fa = get(1, 'final_appearance_inferred')
    assert fa['event_observed'] and fa['duration_seasons'] == 5      # 2015 draft, last app 2019
    assert get(1, 'career_end_official')['event_observed']           # status RET
    p2 = get(2, 'final_appearance_inferred')
    assert not p2['event_observed'] and p2['duration_seasons'] == 3  # active in 2024 -> censored at 3 seasons
    fs = get(3, 'first_nfl_start')
    assert not fs['event_observed'] and fs['duration_seasons'] == 11
    assert s[(s['qb_draft_prospect_id'] == 3) & (s['event_name'] == 'final_appearance_inferred')].empty


def test_most_recent_season_is_not_treated_as_retirement():
    prospects = pd.DataFrame({'qb_draft_prospect_id': [1], 'draft_year': [2018], 'gsis_id': ['g']})
    weekly = pd.DataFrame([{'gsis_id': 'g', 'season': 2024, 'week': 1, 'attempts': 1, 'completions': 1,
                            'passing_yards': 5, 'passing_tds': 0, 'passing_interceptions': 0, 'sacks_suffered': 0,
                            'sack_yards_lost': 0}])
    starts = pd.DataFrame(columns=['gsis_id', 'season', 'game_type', 'games_started'])
    seasons = targets.build_prospect_seasons(prospects, targets.weekly_to_season_totals(weekly), starts, obs_end=2025)
    out = targets.build_career_outcomes(prospects, seasons, pd.DataFrame(columns=['gsis_id', 'status']),
                                        pd.DataFrame(columns=['qb_draft_prospect_id', 'pfr_last_season', 'pfr_games']),
                                        obs_end=2025)
    assert out.iloc[0]['career_end_status'] == 'active_or_censored'
    assert pd.isna(out.iloc[0].get('seasons_to_final_appearance'))


def test_kaplan_meier_matches_hand_computation():
    km = survival.kaplan_meier([1, 2, 2, 3, 4], [True, True, False, True, False]).set_index('time')
    assert km.loc[1, 'survival'] == pytest.approx(0.8)
    assert km.loc[2, 'survival'] == pytest.approx(0.8 * 3 / 4)
    assert km.loc[3, 'survival'] == pytest.approx(0.6 * 1 / 2)
    assert km.loc[4, 'survival'] == pytest.approx(0.3)   # censored -> no drop


# ---------------------------------------------------------------------------
# leakage / availability / splits
# ---------------------------------------------------------------------------

def test_draft_position_only_in_post_draft_scope():
    for scope in ('pre_draft_college', 'pre_draft_full', 'pre_draft_scouting'):
        feats = registry.features_for_scope(scope)
        assert 'draft_round' not in feats and 'draft_overall_pick' not in feats
    assert 'draft_overall_pick' in registry.features_for_scope('post_draft')


def test_combine_and_scouting_excluded_from_college_only_scope():
    college = set(registry.features_for_scope('pre_draft_college'))
    assert not any(f.startswith(('combine_', 'scout_')) for f in college)


def test_outcome_columns_rejected_as_features():
    with pytest.raises(ValueError):
        registry.assert_scope_safe(['final_ypa_does_not_exist'], 'pre_draft_college')
    with pytest.raises(ValueError):
        registry.assert_scope_safe(['games_started_5yr'], 'post_draft')
    with pytest.raises(ValueError):
        registry.assert_scope_safe(['draft_overall_pick'], 'pre_draft_full')


def test_every_registered_feature_has_availability_rule_and_source():
    for f in registry.FEATURES.values():
        assert f.availability_rule and f.source_ids and f.scopes and f.version == registry.FEATURE_VERSION


def test_leakage_audit_flags_value_known_after_draft():
    from src import advanced_pipeline as ap
    feats = pd.DataFrame({'qb_draft_prospect_id': [1], 'draft_year': [2018],
                          'college_available_as_of': [pd.Timestamp('2018-06-01')],  # after the 2018-04-26 draft
                          'final_attempts': [300.0], 'draft_overall_pick': [7.0]})
    feats = feats.reindex(columns=list(feats.columns) + [c for c in registry.FEATURES if c not in feats.columns])
    dates = ap.load_draft_dates()
    lv = ap.long_feature_values(feats, dates)
    audit = ap.leakage_audit(lv, feats, dates)
    assert audit['violations'] == 1
    assert audit['post_draft_values'] == 1


def test_chronological_split_assignment():
    assert preprocess.assign_split(2003) == 'pre_coverage'
    assert preprocess.assign_split(2010) == 'train'
    assert preprocess.assign_split(2017) == 'validation'
    assert preprocess.assign_split(2020) == 'test'
    assert preprocess.assign_split(2024) == 'censored_holdout'


def test_preprocessor_refuses_to_fit_on_test_rows_and_uses_train_stats_only():
    X = pd.DataFrame({'a': [1.0, 2.0, np.nan, 100.0], 'b': [0.0, 1.0, 1.0, 50.0]})
    splits = pd.Series(['train', 'train', 'train', 'test'])
    with pytest.raises(ValueError):
        preprocess.TrainOnlyPreprocessor().fit(X, splits)
    pp = preprocess.TrainOnlyPreprocessor().fit(X.iloc[:3], splits.iloc[:3])
    assert pp.medians_['a'] == 1.5          # test row's 100.0 never influenced the fit
    out = pp.transform(X)
    assert 'a__missing' in out.columns and out.loc[2, 'a__missing'] == 1.0


def test_model_matrix_excludes_censored_targets_and_out_of_scope_columns():
    feats = pd.DataFrame({'qb_draft_prospect_id': [1, 2, 3], 'draft_year': [2010, 2019, 2023],
                          'final_attempts': [300.0, 350.0, 400.0], 'draft_overall_pick': [1.0, 2.0, 3.0]})
    tg = pd.DataFrame({'qb_draft_prospect_id': [1, 2, 3], 'started_16plus_5yr': [True, None, False]})
    X, y, meta = preprocess.build_model_matrix(feats, tg, 'pre_draft_college', 'started_16plus_5yr')
    assert list(meta['qb_draft_prospect_id']) == [1]          # NULL target and holdout class excluded
    assert 'draft_overall_pick' not in X.columns


def test_similarity_uses_only_earlier_classes_and_never_outcomes():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({'qb_draft_prospect_id': range(n), 'draft_year': np.repeat(range(2006, 2018), 5)})
    for f in similarity.DEFAULT_SIMILARITY_FEATURES:
        df[f] = rng.normal(size=n)
    comps = similarity.historical_comps(df, k=3, min_pool=10)
    yr = df.set_index('qb_draft_prospect_id')['draft_year']
    assert (comps['comp_qb_draft_prospect_id'].map(yr) < comps['qb_draft_prospect_id'].map(yr)).all()
    with pytest.raises(ValueError):
        similarity.check_features(['games_started_5yr'])


def test_nan_euclidean_ignores_missing_and_requires_overlap():
    a = np.array([0.0, 0.0, np.nan, 0.0])
    B = np.array([[1.0, 0.0, 5.0, 0.0], [np.nan, np.nan, np.nan, 1.0]])
    d, n = similarity.nan_euclidean(a, B, min_overlap_frac=0.7)
    assert n.tolist() == [3, 1]
    assert d[0] == pytest.approx(np.sqrt(1.0 * 4 / 3)) and np.isnan(d[1])


# ---------------------------------------------------------------------------
# evaluation utilities (behaviour only; no model accuracy is claimed)
# ---------------------------------------------------------------------------

def test_classification_metrics_and_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    s = evaluation.classification_summary(y, p)
    assert s['roc_auc'] == 1.0 and s['brier'] == pytest.approx(np.mean((p - y) ** 2))
    assert np.isnan(evaluation.roc_auc([1, 1], [0.2, 0.3]))
    t = evaluation.calibration_table(y, p, n_bins=2)
    assert t['n'].sum() == 4


def test_bootstrap_ci_brackets_estimate_and_cluster_option():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.uniform(0, 0.4, 200), 0, 1)
    ci = evaluation.bootstrap_ci(evaluation.brier_score, y, p, n_boot=300)
    assert ci['lower'] <= ci['estimate'] <= ci['upper']
    ci_g = evaluation.bootstrap_ci(evaluation.brier_score, y, p, n_boot=300, groups=np.repeat(range(20), 10))
    assert ci_g['n_boot_valid'] > 0


def test_conformal_interval_width():
    out = evaluation.conformal_interval(residuals_calibration=[1, -2, 3, -4, 5, -6, 7, -8, 9], predictions=[10.0], alpha=0.2)
    assert out.loc[0, 'upper'] - out.loc[0, 'lower'] == pytest.approx(2 * 9)


# ---------------------------------------------------------------------------
# box-score season reconstruction (CFBD season stats are ~empty for 2004-2008)
# ---------------------------------------------------------------------------

def test_box_season_totals_rushing_observed_vs_unobserved():
    from src import advanced_pipeline as ap
    gp = pd.DataFrame({'game_id': [1, 2, 3], 'season': [2005, 2005, 2010], 'week': [1, 2, 1], 'team': ['A', 'A', 'B'],
                       'conference': 'X', 'cfbd_player_id': ['7', '7', '8'], 'player_name': ['Q', 'Q', 'R'],
                       'completions': [10, 12, 5], 'attempts': [20, 22, 9], 'passing_yards': [150, 170, 60],
                       'passing_tds': [1, 2, 0], 'interceptions': [0, 1, 1]})
    gr = pd.DataFrame({'game_id': [1], 'season': [2005], 'cfbd_player_id': ['99'], 'carries': [12],
                       'rushing_yards': [80], 'rushing_tds': [1]})
    b = ap.box_season_totals(gp, gr).set_index(['cfbd_player_id', 'season'])
    assert b.loc[('7', 2005), 'attempts'] == 42 and b.loc[('7', 2005), 'completions'] == 22
    assert b.loc[('7', 2005), 'rushing_attempts'] == 0        # rushing box scores exist for 2005: zero carries observed
    assert np.isnan(b.loc[('8', 2010), 'rushing_attempts'])   # no rushing box scores for 2010: unobserved, NOT 0


def test_season_stats_preferred_over_box_scores():
    from src import advanced_pipeline as ap
    cfb_all = pd.DataFrame({'cfbd_player_id': ['7'], 'player_id': [1], 'season': [2009], 'school': ['A'],
                            'conference': ['X'], 'height_inches': [75], 'weight_lbs': [220], 'completions': [200],
                            'attempts': [300], 'passing_yards': [2500], 'passing_tds': [20], 'interceptions': [8],
                            'rushing_attempts': [40], 'rushing_yards': [100], 'rushing_tds': [2]})
    box = pd.DataFrame({'cfbd_player_id': ['7', '7'], 'season': [2008, 2009], 'team': ['A', 'A'], 'conference': 'X',
                        'player_name': 'Q', 'completions': [50, 199], 'attempts': [90, 299], 'passing_yards': [600, 2490],
                        'passing_tds': [4, 20], 'interceptions': [3, 8], 'rushing_attempts': [10, np.nan],
                        'rushing_yards': [30, np.nan], 'rushing_tds': [0, np.nan]})
    out = ap.all_passer_seasons(cfb_all, box).set_index('season')
    assert out.loc[2009, 'season_stats_source'] == 'cfbd_season_stats' and out.loc[2009, 'attempts'] == 300
    assert out.loc[2008, 'season_stats_source'] == 'box_score_aggregate' and out.loc[2008, 'cfb_player_id'] == 1


def test_team_box_coverage_flags_fcs_teams():
    from src import advanced_pipeline as ap
    games = pd.DataFrame({'season': 2007, 'homeTeam': ['Delaware'] * 4 + ['Big U'], 'awayTeam': ['F1', 'F2', 'F3', 'Big U', 'X'],
                          'completed': True})
    box = pd.DataFrame({'season': [2007, 2007], 'team': ['Delaware', 'Big U'], 'game_id': [4, 4]})
    cov = ap.team_box_coverage(games, box).set_index('team')['team_box_score_coverage']
    assert cov['Delaware'] == pytest.approx(0.25)   # 1 of 4 games box-scored -> per-game features blanked
