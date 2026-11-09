"""End-to-end Phase-3 pipeline test on a tiny synthetic league, with a fake CFBD
client and injected NFL inputs -- no network, no API key, never the real DB.

Covers: database idempotency, existing-NFL-table preservation, identity bridge
via CFBD draft picks, offensive-profile joins without fan-out, pre-draft
availability (zero leakage violations), and window targets.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from src import advanced_pipeline as ap
from src import metrics
from src.db.schema import (
    CfbPlayer, CfbPlayerSeason, CfbPlayerSeasonDerived, CollegeNflPlayerMap, League, NflCareerOutcome,
    NflDraftPickRaw, Player, PlayerSeasonRaw, ProspectCollegeSeason, ProspectFeatureValue, QbDraftProspect,
    Source, SurvivalRecord, get_engine, get_session, init_db,
)
from src.historical_pipeline import seed_reference_tables

TEAMS = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta']
QB_IDS = {'Alpha': '11', 'Beta': '22'}
PAIRS = list(itertools.combinations(TEAMS, 2))  # 15 games, 3 per week over 5 weeks


def _week_of(i):
    return i // 3 + 1


class FakeClient:
    max_network_calls = 0

    def __init__(self):
        self.network_calls = 0
        self.cache_hits = 0

    def get(self, path, params=None):
        self.cache_hits += 1
        params = params or {}
        if path == '/games':
            return [{'id': 1000 + i, 'season': 2019, 'week': _week_of(i), 'seasonType': 'regular',
                     'startDate': f'2019-09-{7 * _week_of(i):02d}T19:00:00.000Z', 'completed': True,
                     'neutralSite': False, 'conferenceGame': True, 'homeTeam': h, 'homeConference': 'Test',
                     'homeClassification': 'fbs', 'homePoints': 21, 'awayTeam': a, 'awayConference': 'Test',
                     'awayClassification': 'fbs', 'awayPoints': 14} for i, (h, a) in enumerate(PAIRS)]
        if path == '/games/players':
            out = []
            for i, (h, a) in enumerate(PAIRS):
                if _week_of(i) != params['week'] or params.get('seasonType', 'regular') != 'regular':
                    continue
                teams = []
                for t in (h, a):
                    pid = QB_IDS.get(t, f'9{TEAMS.index(t)}')
                    att = 30 if t == 'Alpha' else 25
                    yds = 300 if t == 'Alpha' else 175 + 5 * TEAMS.index(t)
                    teams.append({'team': t, 'conference': 'Test', 'homeAway': 'home' if t == h else 'away',
                                  'categories': [{'name': 'passing', 'types': [
                                      {'name': 'C/ATT', 'athletes': [{'id': pid, 'name': f'{t} Passer', 'stat': f'{att // 2 + 3}/{att}'}]},
                                      {'name': 'YDS', 'athletes': [{'id': pid, 'name': f'{t} Passer', 'stat': str(yds)}]},
                                      {'name': 'TD', 'athletes': [{'id': pid, 'name': f'{t} Passer', 'stat': '2'}]},
                                      {'name': 'INT', 'athletes': [{'id': pid, 'name': f'{t} Passer', 'stat': '1'}]}]}]})
                out.append({'id': 1000 + i, 'teams': teams})
            return out
        if path == '/stats/season':
            rows = []
            for t in TEAMS:
                for name, v in (('games', 5), ('passAttempts', 140), ('passCompletions', 80), ('netPassingYards', 1100),
                                ('passingTDs', 10), ('interceptions', 5), ('rushingAttempts', 160), ('rushingYards', 700),
                                ('rushingTDs', 8), ('possessionTime', 9000), ('sacks', 10), ('sacksOpponent', 8 + TEAMS.index(t)),
                                ('passAttemptsOpponent', 135), ('rushingAttemptsOpponent', 150)):
                    rows.append({'season': 2019, 'team': t, 'conference': 'Test', 'statName': name, 'statValue': v})
            return rows
        if path == '/talent':
            return [{'year': 2019, 'team': t, 'talent': 600 + 10 * i} for i, t in enumerate(TEAMS)]
        if path == '/recruiting/teams':
            return [{'year': params['year'], 'rank': i + 1, 'team': t, 'points': 200 - 10 * i} for i, t in enumerate(TEAMS)]
        if path == '/recruiting/players':
            if params['year'] != 2016:
                return []
            return [{'id': 'r1', 'athleteId': '501', 'recruitType': 'HighSchool', 'year': 2016, 'ranking': 50,
                     'name': 'Alpha WR', 'committedTo': 'Alpha', 'position': 'WR', 'stars': 4, 'rating': 0.95}]
        if path == '/stats/player/season' and params.get('category') == 'receiving':
            return [{'season': 2019, 'playerId': pid, 'player': n, 'position': 'WR', 'team': 'Alpha', 'conference': 'Test',
                     'category': 'receiving', 'statType': st, 'stat': str(v)}
                    for pid, n, rec in (('501', 'Alpha WR', 40), ('502', 'Alpha WR2', 20))
                    for st, v in (('REC', rec), ('YDS', rec * 12), ('TD', 3))]
        if path == '/player/returning':
            return [{'season': 2019, 'team': t, 'conference': 'Test', 'percentPPA': 0.6, 'percentPassingPPA': 0.7,
                     'percentReceivingPPA': 0.5, 'percentRushingPPA': 0.4, 'usage': 0.55, 'receivingUsage': 0.5}
                    for t in TEAMS]
        if path == '/draft/picks':
            return [{'year': 2020, 'overall': 5, 'round': 1, 'pick': 5, 'name': 'Alpha Passer', 'position': 'Quarterback',
                     'collegeAthleteId': 11, 'nflAthleteId': 1, 'collegeTeam': 'Alpha', 'nflTeam': 'X',
                     'preDraftRanking': 3, 'preDraftPositionRanking': 1, 'preDraftGrade': 92},
                    {'year': 2020, 'overall': 100, 'round': 4, 'pick': 4, 'name': 'Beta Passer', 'position': 'Quarterback',
                     'collegeAthleteId': 22, 'nflAthleteId': 2, 'collegeTeam': 'Beta', 'nflTeam': 'Y',
                     'preDraftRanking': 150, 'preDraftPositionRanking': 9, 'preDraftGrade': 70}]
        if path == '/ratings/sp':
            return [{'year': 2019, 'team': t, 'rating': 5.0 - i, 'ranking': i + 1, 'sos': 0.5,
                     'offense': {'rating': 30.0, 'pace': 1.0, 'runRate': 0.5}, 'defense': {'rating': 20.0 + i}}
                    for i, t in enumerate(TEAMS)]
        return []


def _seed(db_url):
    engine = init_db(get_engine(db_url))
    s = get_session(engine)
    seed_reference_tables(s)
    if not s.get(League, 'NFL'):
        s.add(League(league_id='NFL', name='National Football League', start_season=1922))
    s.commit()
    nfl_player = Player(source_id='nflverse_player_stats', source_player_id='00-EXIST', display_name='Existing QB')
    s.add(nfl_player)
    s.flush()
    s.add(PlayerSeasonRaw(player_id=nfl_player.id, source_id='nflverse_player_stats', season=2020, league_id='NFL',
                          season_type='REG', position_listed='QB', is_qb_listed=True, attempts=400))
    for pick, name, school, gsis in ((5, 'Alpha Passer', 'Alpha', 'G-ALPHA'), (100, 'Beta Passer', 'Beta', 'G-BETA')):
        raw = NflDraftPickRaw(source_id='nflverse_draft_picks', season=2020, round=1 if pick == 5 else 4, pick=pick,
                              team='X', gsis_id=gsis, pfr_player_name=name, position_listed='QB', college=school)
        s.add(raw)
        s.flush()
        s.add(QbDraftProspect(draft_pick_raw_id=raw.id, source_id='nflverse_draft_picks', draft_year=2020,
                              round=raw.round, pick=pick, nfl_team='X', player_name=name, gsis_id=gsis, college=school,
                              position_listed='QB', birthdate=pd.Timestamp('1997-03-01').date()))
        cp = CfbPlayer(source_id='cfbd_player_season_stats', cfbd_player_id=QB_IDS[school], display_name=name)
        s.add(cp)
        s.flush()
        # Beta's QB is spelled differently on the CFBD side -> only the stable id bridge can link him
        if school == 'Beta':
            cp.display_name = 'Bartholomew Passer'
        att = 150 if school == 'Alpha' else 125
        cs = CfbPlayerSeason(player_id=cp.id, source_id='cfbd_player_season_stats', season=2019, school=school,
                             conference='Test', completions=att // 2 + 15, attempts=att, passing_yards=1500 if school == 'Alpha' else 900,
                             passing_tds=10, interceptions=5, rushing_attempts=30, rushing_yards=100, rushing_tds=1,
                             height_inches=75, weight_lbs=220)
        s.add(cs)
        s.flush()
        s.add(CfbPlayerSeasonDerived(
            cfb_player_season_id=cs.id, completion_pct=float(metrics.completion_pct(cs.completions, cs.attempts)),
            yards_per_attempt=float(metrics.yards_per_attempt(cs.passing_yards, cs.attempts)),
            td_pct=float(metrics.td_pct(cs.passing_tds, cs.attempts)), int_pct=float(metrics.int_pct(cs.interceptions, cs.attempts)),
            ncaa_passer_rating=float(metrics.ncaa_passer_rating(cs.completions, cs.attempts, cs.passing_yards,
                                                                cs.passing_tds, cs.interceptions))))
    s.commit()
    s.close()


def _nfl_inputs():
    weekly = pd.DataFrame([{'gsis_id': 'G-ALPHA', 'season': y, 'week': w, 'attempts': 30, 'completions': 20,
                            'passing_yards': 240, 'passing_tds': 2, 'passing_interceptions': 1, 'sacks_suffered': 2,
                            'sack_yards_lost': 10} for y in range(2020, 2026) for w in range(1, 18)])
    starts = pd.DataFrame([{'gsis_id': 'G-ALPHA', 'season': y, 'game_type': 'REG', 'games_started': 17}
                           for y in range(2020, 2026)])
    from src.features.targets import weekly_to_season_totals
    return {'season_totals': weekly_to_season_totals(weekly), 'starts': starts, 'roster_status': pd.DataFrame({'gsis_id': ['G-ALPHA'], 'status': ['ACT']})}


def _combine():
    return pd.DataFrame({'season': [2020], 'draft_year': [2020.0], 'draft_team': ['X'], 'draft_round': [1.0],
                         'draft_ovr': [5.0], 'pfr_id': [None], 'cfb_id': [None], 'player_name': ['Alpha Passer'],
                         'pos': ['QB'], 'school': ['Alpha'], 'ht': ['6-3'], 'wt': [220.0], 'forty': [4.8],
                         'bench': [None], 'vertical': [31.0], 'broad_jump': [110.0], 'cone': [None], 'shuttle': [4.3]})


def _run(tmp_path, db_url):
    return ap.main(db_url=db_url, cfb_seasons=[2019], recruit_years=[2016, 2017, 2018, 2019], draft_years=[2020],
                   nfl_seasons=list(range(2020, 2026)), combine_seasons=[2020], data_dir=tmp_path / 'data',
                   results_dir=tmp_path / 'results', client=FakeClient(), nfl_inputs=_nfl_inputs(),
                   combine_raw=_combine(), fetch_wikipedia=False, plays=pd.DataFrame())


@pytest.fixture()
def built(tmp_path):
    db_url = f'sqlite:///{(tmp_path / "t.db").as_posix()}'
    _seed(db_url)
    first = _run(tmp_path, db_url)
    return tmp_path, db_url, first


def _table(db_url, model, order):
    s = get_session(get_engine(db_url))
    cols = [c.name for c in model.__table__.columns if c.name != 'id']
    df = pd.DataFrame(s.execute(model.__table__.select()).all(), columns=[c.name for c in model.__table__.columns])[cols]
    s.close()
    return df.sort_values(order).reset_index(drop=True)


def test_database_idempotency(built):
    tmp_path, db_url, first = built
    fv1 = _table(db_url, ProspectFeatureValue, ['qb_draft_prospect_id', 'feature_name'])
    second = _run(tmp_path, db_url)
    assert first['row_counts'] == second['row_counts']
    fv2 = _table(db_url, ProspectFeatureValue, ['qb_draft_prospect_id', 'feature_name'])
    pd.testing.assert_frame_equal(fv1, fv2)


def test_existing_nfl_tables_untouched(built):
    _, db_url, report = built
    s = get_session(get_engine(db_url))
    assert s.query(Player).count() == 1 and s.query(PlayerSeasonRaw).count() == 1
    assert s.query(Player).one().display_name == 'Existing QB'
    s.close()
    assert report['existing_table_counts_before']['players'] == report['existing_table_counts_after']['players']


def test_identity_bridge_and_join_without_fanout(built):
    _, db_url, report = built
    m = _table(db_url, CollegeNflPlayerMap, ['qb_draft_prospect_id'])
    assert (m['match_status'] == 'matched').all()
    assert set(m['cfb_match_method']) == {'exact_name_school', 'cfbd_draft_athlete_id'}
    pcs = _table(db_url, ProspectCollegeSeason, ['qb_draft_prospect_id'])
    assert len(pcs) == 2 and not pcs.duplicated(['qb_draft_prospect_id', 'season']).any()
    assert pcs['games_with_pass_attempt'].tolist() == [5, 5]
    assert pcs['opp_adj_ypa'].notna().all()


def test_no_leakage_and_model_ready_exports(built):
    tmp_path, _, report = built
    assert report['leakage_audit']['violations'] == 0
    pre = pd.read_parquet(tmp_path / 'data' / 'model_ready' / 'features_pre_draft_college.parquet')
    assert 'draft_overall_pick' not in pre.columns and not any(c.startswith('combine_') for c in pre.columns)
    post = pd.read_parquet(tmp_path / 'data' / 'model_ready' / 'features_post_draft.parquet')
    assert 'draft_overall_pick' in post.columns
    alpha = pre[pre['player_name'] == 'Alpha Passer'].iloc[0]
    assert alpha['env_pass_rate'] == pytest.approx(140 / 300)
    assert alpha['wr_top3_avg_recruit_rating'] == pytest.approx(0.95)


def test_targets_and_survival_for_synthetic_careers(built):
    _, db_url, _ = built
    o = _table(db_url, NflCareerOutcome, ['qb_draft_prospect_id'])
    assert o['window_5yr_complete'].all()                 # 2020 + 4 <= 2025
    alpha, beta = o.iloc[0], o.iloc[1]
    assert alpha['games_started_5yr'] == 85 and alpha['started_64plus_5yr']
    assert beta['career_end_status'] == 'never_appeared_window_complete' and beta['started_1plus_5yr'] == False  # noqa: E712
    sv = _table(db_url, SurvivalRecord, ['qb_draft_prospect_id', 'event_name'])
    fa = sv[(sv['event_name'] == 'final_appearance_inferred')].iloc[0]
    assert not fa['event_observed']                        # active through 2025 -> censored
