"""Offensive-scheme dataset tests: label mapping, provenance/verification rules,
team-season joins, transfers, duplicate annotations, play-by-play calculations,
missing statistics, leakage, and checks on the real built dataset (skipped if absent)."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import schemes as S
from src.ingest_wikipedia_schemes import FIELD_RE, strip_wikitext

TAX = S.load_taxonomy()
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data' / 'qb_history.db'


def _label(**kw):
    base = {c: None for c in S.LABEL_COLUMNS}
    base.update({'team': 'A', 'season': 2018, 'scheme_dimension': 'offensive_family', 'scheme_label': 'spread',
                 'verification_status': 'verified', 'confidence': 'high', 'source_type': 'newspaper',
                 'source_url': 'https://example.org/a', 'source_title': 'T', 'publication_date': '2018-11-01',
                 'evidence_summary': '"ran the spread"', 'annotation_method': 'manual_source_review_v1'})
    base.update(kw)
    return base


# --- mapping / unsupported classifications ------------------------------------------

def test_raw_text_mapping_overlapping_and_unmapped():
    assert set(S.map_raw_scheme('Pro spread', TAX)) == {('offensive_family', 'spread'), ('offensive_family', 'pro_style')}
    assert S.map_raw_scheme('  AIR RAID ', TAX) == [('offensive_family', 'air_raid')]
    assert S.map_raw_scheme('Pistol', TAX) == [('formation_tendency', 'pistol')]  # formation, not a family
    assert S.map_raw_scheme('Gulf Coast', TAX) == []   # ambiguous -> stays unknown
    assert S.map_raw_scheme(None, TAX) == []


def test_every_mapped_label_is_in_the_taxonomy():
    for pairs in TAX['raw_text_map'].values():
        for dim, lab in pairs:
            assert lab in TAX['dimensions'][dim]


def test_validation_rejects_labels_outside_taxonomy():
    with pytest.raises(ValueError):
        S.validate_labels(pd.DataFrame([_label(scheme_label='wishbone_magic')]), TAX)


def test_validation_rejects_verified_label_without_provenance():
    for col in ('source_url', 'source_title', 'publication_date', 'evidence_summary'):
        with pytest.raises(ValueError):
            S.validate_labels(pd.DataFrame([_label(**{col: ''})]), TAX)


def test_only_manual_review_can_verify_and_statistics_never_label():
    with pytest.raises(ValueError):
        S.validate_labels(pd.DataFrame([_label(annotation_method='wikipedia_infobox_automated_v1')]), TAX)
    with pytest.raises(ValueError):
        S.validate_labels(pd.DataFrame([_label(verification_status='reported', annotation_method='kmeans_cluster_v1')]), TAX)


def test_duplicate_annotations_rejected():
    with pytest.raises(ValueError):
        S.validate_labels(pd.DataFrame([_label(), _label()]), TAX)
    S.validate_labels(pd.DataFrame([_label(), _label(source_url='https://example.org/b')]), TAX)  # 2 sources OK


def test_wikipedia_rows_are_reported_low_confidence_never_verified():
    ib = pd.DataFrame([{'team': 'A', 'season': 2018, 'exists': True, 'page_title': '2018 A Owls football team',
                        'revid': 123, 'rev_timestamp': '2026-01-01T00:00:00Z', 'off_scheme': 'Spread option',
                        'off_scheme_has_citation': False},
                       {'team': 'B', 'season': 2018, 'exists': True, 'page_title': '2018 B Bees football team',
                        'revid': 124, 'rev_timestamp': '2026-01-01T00:00:00Z', 'off_scheme': 'Run and gun',
                        'off_scheme_has_citation': False}])
    w = S.wikipedia_labels(ib, TAX)
    assert len(w) == 1 and w.iloc[0]['scheme_label'] == 'spread_option'
    assert (w['verification_status'] == 'reported').all() and (w['confidence'] == 'low').all()
    assert w.iloc[0]['source_url'].endswith('oldid=123')


def test_infobox_parsing_empty_field_does_not_capture_next_line():
    txt = "{{Infobox\n| off_scheme =\n| def_coach = Someone\n}}"
    m = FIELD_RE['off_scheme'].search(txt)
    assert m.group(1).strip() == ''
    assert strip_wikitext('[[Run and shoot offense|Run and shoot]]<ref>x</ref>') == 'Run and shoot'


def test_manual_label_file_is_valid_and_conflicts_are_not_labels():
    manual = S.load_manual_labels()
    S.validate_labels(manual, TAX)
    ver = manual[manual['verification_status'] == 'verified']
    assert len(ver) >= 1 and ver['evidence_summary'].str.len().gt(10).all()
    sch = S.team_season_schemes(manual[['team', 'season']], manual, pd.DataFrame(
        columns=['team', 'season', 'exists', 'off_scheme']), TAX)
    ok = sch.set_index(['team', 'season'])
    assert ok.loc[('Oklahoma', 2017), 'manual_research_status'] == 'conflicting'
    assert pd.isna(ok.loc[('Oklahoma', 2017), 'verified_offensive_family'])   # conflicting -> no label


# --- joins / transfers / research status ----------------------------------------------

def _labels():
    return pd.DataFrame([_label(team='A', season=2016, scheme_label='air_raid'),
                         _label(team='B', season=2018, scheme_label='pro_style', source_url='https://example.org/c')])


def test_team_season_join_is_exact_and_status_distinguishes_unresearched():
    ts = pd.DataFrame({'team': ['A', 'A', 'B'], 'season': [2016, 2017, 2018]})
    sch = S.team_season_schemes(ts, _labels(), pd.DataFrame(columns=['team', 'season', 'exists', 'off_scheme']), TAX)
    s = sch.set_index(['team', 'season'])
    assert s.loc[('A', 2016), 'verified_offensive_family'] == 'air_raid'
    assert pd.isna(s.loc[('A', 2017), 'verified_offensive_family'])    # never carried forward to 2017
    assert s.loc[('A', 2017), 'manual_research_status'] == 'not_researched'


def test_transfer_uses_each_schools_offense():
    ts = pd.DataFrame({'team': ['A', 'B'], 'season': [2016, 2018]})
    lab = _labels()
    lab['verification_status'] = 'reported'
    lab['annotation_method'] = 'wikipedia_infobox_automated_v1'
    sch = S.team_season_schemes(ts, lab, pd.DataFrame(columns=['team', 'season', 'exists', 'off_scheme']), TAX)
    pcs = pd.DataFrame({'qb_draft_prospect_id': [1, 1], 'season': [2016, 2018], 'team': ['A', 'B'],
                        'attempts': [100.0, 300.0], 'is_final_college_season': [False, True],
                        'seasons_before_draft': [3, 1]})
    seasons = S.qb_scheme_seasons(pcs, sch, pd.DataFrame(), None)
    f = S.qb_scheme_features(seasons, pd.Series({1: 2019})).iloc[0]
    assert f['scheme_rep_final_pro_style'] == 1 and f['scheme_rep_final_air_raid'] == 0
    assert f['scheme_rep_n_distinct_families'] == 2
    assert f['scheme_rep_career_spread_family_share'] == pytest.approx(100 / 400)
    assert f['final_observed_season_gap'] == 0


def test_unknown_scheme_stays_missing_not_zero():
    ts = pd.DataFrame({'team': ['Z'], 'season': [2010]})
    sch = S.team_season_schemes(ts, _labels(), pd.DataFrame(columns=['team', 'season', 'exists', 'off_scheme']), TAX)
    pcs = pd.DataFrame({'qb_draft_prospect_id': [7], 'season': [2010], 'team': ['Z'], 'attempts': [300.0],
                        'is_final_college_season': [True], 'seasons_before_draft': [1]})
    f = S.qb_scheme_features(S.qb_scheme_seasons(pcs, sch, pd.DataFrame(), None), pd.Series({7: 2011})).iloc[0]
    assert f['scheme_rep_final_known'] == 0 and np.isnan(f['scheme_rep_final_spread'])
    assert np.isnan(f['scheme_q_final_standard_down_pass_rate'])   # no play-by-play -> NaN


# --- play-by-play calculations ---------------------------------------------------------

def _plays(n_extra=400):
    rows = [  # standard downs: 1st&10 pass, 2nd&5 rush, 3rd&3 pass ; passing downs: 2nd&9 pass, 3rd&8 rush
        dict(down=1, distance=10, playType='Pass Reception'), dict(down=2, distance=5, playType='Rush'),
        dict(down=3, distance=3, playType='Sack'), dict(down=2, distance=9, playType='Pass Incompletion'),
        dict(down=3, distance=8, playType='Rush'),
        dict(down=1, distance=10, playType='Penalty'),                      # not a pass/rush play
        dict(down=1, distance=10, playType='Rush', period=4, offenseScore=50, defenseScore=0),  # garbage time
    ]
    rows += [dict(down=1, distance=10, playType='Rush')] * n_extra
    df = pd.DataFrame(rows)
    df['period'] = df.get('period').fillna(1) if 'period' in df else 1
    df['offenseScore'] = df['offenseScore'].fillna(0) if 'offenseScore' in df else 0
    df['defenseScore'] = df['defenseScore'].fillna(0) if 'defenseScore' in df else 0
    df['season'], df['offense'], df['gameId'] = 2019, 'A', 1
    return df


def test_down_specific_pass_rates():
    p = S.play_profile(_plays(n_extra=0).assign(gameId=1))
    # with only 5 counted plays the team is below MIN_PBP_PLAYS -> blanked
    assert p.iloc[0]['pbp_plays_nongarbage'] == 5 and np.isnan(p.iloc[0]['standard_down_pass_rate'])
    p = S.play_profile(_plays())
    r = p.iloc[0]
    assert r['pbp_plays_nongarbage'] == 405           # penalty and garbage-time play excluded
    assert r['passing_down_pass_rate'] == pytest.approx(1 / 2)
    assert r['standard_down_pass_rate'] == pytest.approx(2 / 403)   # sacks count as pass plays


def test_garbage_time_definition():
    g = S.is_garbage_time([1, 2, 3, 4, 4], [50, 39, 29, 23, 22])
    assert g.tolist() == [False, True, True, True, False]


# --- real built dataset (skipped if not built) --------------------------------------------

@pytest.fixture(scope='module')
def con():
    if not DB.exists():
        pytest.skip('database not built')
    c = sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True)
    try:
        n = c.execute('select count(*) from cfb_team_season_schemes').fetchone()[0]
    except sqlite3.OperationalError:
        c.close()
        pytest.skip('scheme tables not built')
    if n == 0:
        c.close()
        pytest.skip('scheme tables empty')
    yield c
    c.close()


def test_real_labels_provenance_and_uniqueness(con):
    lab = pd.read_sql('select * from cfb_offensive_scheme_labels', con)
    key = ['team', 'season', 'scheme_dimension', 'scheme_label', 'source_url']
    assert not lab.duplicated(key).any()
    ver = lab[lab['verification_status'] == 'verified']
    assert (ver['annotation_method'] == 'manual_source_review_v1').all()
    wiki = lab[lab['annotation_method'] == 'wikipedia_infobox_automated_v1']
    assert (wiki['verification_status'] == 'reported').all() and (wiki['confidence'] == 'low').all()
    assert wiki['source_url'].str.contains(r'index\.php\?oldid=\d+$').all()
    assert all(str(s) in t and t.endswith('football team') for s, t in zip(wiki['season'], wiki['source_title']))


def test_real_team_seasons_match_prospect_seasons(con):
    ts = pd.read_sql('select team, season from cfb_team_season_schemes', con)
    pcs = pd.read_sql('select distinct team, season from prospect_college_seasons', con)
    assert not ts.duplicated().any()
    assert set(map(tuple, ts.values)) == set(map(tuple, pcs.values))


def test_real_scheme_features_are_pre_draft_only():
    p = ROOT / 'data' / 'processed' / 'qb_scheme_seasons.parquet'
    if not p.exists():
        pytest.skip('processed scheme outputs not built')
    s = pd.read_parquet(p)
    assert (s['seasons_before_draft'] >= 1).all()
    f = pd.read_parquet(ROOT / 'data' / 'processed' / 'qb_scheme_features.parquet')
    from src.features.registry import is_forbidden
    assert not any(is_forbidden(c) for c in f.columns if c.startswith('scheme_'))


def test_profile_already_on_team_seasons_is_not_merged_twice():
    ts = pd.DataFrame({'team': ['A'], 'season': [2019]})
    sch = S.team_season_schemes(ts, _labels(), pd.DataFrame(columns=['team', 'season', 'exists', 'off_scheme']), TAX)
    prof = S.play_profile(_plays())
    sch = sch.merge(prof, on=['season', 'team'], how='left')
    pcs = pd.DataFrame({'qb_draft_prospect_id': [1], 'season': [2019], 'team': ['A'], 'attempts': [300.0],
                        'is_final_college_season': [True], 'seasons_before_draft': [1]})
    seasons = S.qb_scheme_seasons(pcs, sch, prof, None)
    assert not any(c.endswith(('_x', '_y')) for c in seasons.columns)
    f = S.qb_scheme_features(seasons, pd.Series({1: 2020})).iloc[0]
    assert f['scheme_q_final_passing_down_pass_rate'] == pytest.approx(0.5)


def test_reported_status_distinguishes_non_family_labels_from_unmapped_text():
    ib = pd.DataFrame([
        {'team': t, 'season': 2018, 'exists': True, 'page_title': f'2018 {t} X football team', 'revid': i,
         'rev_timestamp': '2026-01-01T00:00:00Z', 'off_scheme': txt, 'off_scheme_has_citation': False}
        for i, (t, txt) in enumerate([('A', 'Spread'), ('B', 'Pistol'), ('C', 'Gulf Coast'), ('D', None)])])
    lab = S.wikipedia_labels(ib, TAX)
    sch = S.team_season_schemes(ib[['team', 'season']], lab, ib, TAX).set_index('team')['reported_label_status']
    assert sch.to_dict() == {'A': 'labelled_family', 'B': 'labelled_other_dimension_only',
                             'C': 'unmapped_text', 'D': 'page_without_scheme_field'}
