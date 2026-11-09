"""College offensive-scheme features: documented labels kept apart from measured tendencies.

Three evidence tiers, never merged into one "scheme" column:
  1. VERIFIED (manual)  config/offensive_scheme_labels.csv rows with verification_status='verified':
                        a source was read and explicitly ties the label to that team AND season.
  2. REPORTED (low)     Wikipedia team-season infobox `off_scheme`: season-specific but
                        community-edited, almost always uncited, usually written after the
                        fact. Status 'reported', confidence 'low'. Never called verified.
  3. QUANTITATIVE       measured tendencies from box scores / play-by-play. These are
                        statistics, not schemes; nothing here ever produces a label.

Research status per team-season distinguishes "verified", "researched, nothing reliable
found", and "never researched" -- an unresearched season is not evidence of anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / 'config' / 'offensive_scheme_taxonomy.json'
MANUAL_LABELS_PATH = ROOT / 'config' / 'offensive_scheme_labels.csv'

LABEL_COLUMNS = ['team', 'season', 'scheme_dimension', 'scheme_label', 'verification_status', 'confidence',
                 'source_type', 'source_url', 'source_title', 'publication_date', 'evidence_summary',
                 'annotation_method', 'raw_text', 'notes']
VERIFIED = 'verified'
REPORTED = 'reported'
PASS_TYPES = {'Pass Reception', 'Pass Incompletion', 'Passing Touchdown', 'Sack', 'Interception',
              'Pass Interception Return', 'Interception Return Touchdown', 'Pass Completion', 'Pass'}
RUSH_TYPES = {'Rush', 'Rushing Touchdown'}


def load_taxonomy(path: Path | None = None) -> dict:
    return json.loads((path or TAXONOMY_PATH).read_text(encoding='utf-8'))


def map_raw_scheme(text: str | None, taxonomy: dict) -> list[tuple[str, str]]:
    """Raw infobox text -> [(dimension, label)]. Unlisted text maps to [] (stays unknown)."""
    if not text:
        return []
    key = ' '.join(str(text).lower().split())
    return [tuple(x) for x in taxonomy['raw_text_map'].get(key, [])]


def validate_labels(labels: pd.DataFrame, taxonomy: dict) -> None:
    """Raises on the failure modes the brief calls out: labels outside the taxonomy,
    'verified' rows without full provenance, duplicated annotations, and any label
    produced by a statistical method."""
    dims = taxonomy['dimensions']
    real = labels[labels['scheme_label'] != 'unknown']
    bad = [(d, l) for d, l in zip(real['scheme_dimension'], real['scheme_label']) if l not in dims.get(d, [])]
    if bad:
        raise ValueError(f'labels outside the taxonomy: {bad[:5]}')
    ver = labels[labels['verification_status'] == VERIFIED]
    for col in ('source_url', 'source_title', 'publication_date', 'evidence_summary', 'confidence', 'annotation_method'):
        missing = ver[ver[col].fillna('').astype(str).str.strip() == '']
        if len(missing):
            raise ValueError(f'verified labels missing {col}: {missing[["team", "season"]].values.tolist()}')
    if (ver['annotation_method'] != 'manual_source_review_v1').any():
        raise ValueError('only manual source review can mark a label verified')
    if labels['annotation_method'].fillna('').str.contains('cluster|statistic|model', case=False).any():
        raise ValueError('statistically derived rows can never be scheme labels')
    key = ['team', 'season', 'scheme_dimension', 'scheme_label', 'source_url']
    dup = labels[labels.duplicated(key, keep=False)]
    if len(dup):
        raise ValueError(f'duplicated scheme annotations: {dup[key].values.tolist()[:5]}')


def load_manual_labels(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or MANUAL_LABELS_PATH, dtype=str, keep_default_na=False)
    df['season'] = df['season'].astype(int)
    df['raw_text'] = None
    return df.reindex(columns=LABEL_COLUMNS).replace({'': None})


def wikipedia_labels(infoboxes: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    """One 'reported' row per (team-season, mapped label) from the infobox field."""
    from src.ingest_wikipedia_schemes import permalink
    rows = []
    for _, r in infoboxes[infoboxes['exists'].fillna(False).astype(bool)].iterrows():
        mapped = map_raw_scheme(r.get('off_scheme'), taxonomy)
        for dim, lab in mapped:
            rows.append({'team': r['team'], 'season': int(r['season']), 'scheme_dimension': dim, 'scheme_label': lab,
                         'verification_status': REPORTED, 'confidence': 'low', 'source_type': 'wikipedia_infobox',
                         'source_url': permalink(int(r['revid'])), 'source_title': r['page_title'],
                         'publication_date': str(r['rev_timestamp'])[:10],
                         'evidence_summary': f"Infobox off_scheme = '{r['off_scheme']}'"
                                             + (' (has inline citation)' if r.get('off_scheme_has_citation') else ' (uncited)'),
                         'annotation_method': 'wikipedia_infobox_automated_v1', 'raw_text': r['off_scheme'],
                         'notes': 'publication_date is the date of the revision read, not of the original edit'})
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def team_season_schemes(team_seasons: pd.DataFrame, labels: pd.DataFrame, infoboxes: pd.DataFrame,
                        taxonomy: dict) -> pd.DataFrame:
    """One row per relevant team-season: research status and labels per tier."""
    out = team_seasons[['team', 'season']].drop_duplicates().copy()
    manual = labels[labels['annotation_method'] == 'manual_source_review_v1']
    researched = set(zip(manual['team'], manual['season']))
    ver = labels[labels['verification_status'] == VERIFIED]
    rep = labels[labels['verification_status'] == REPORTED]
    conflicting = set(zip(manual.loc[manual['verification_status'] == 'conflicting', 'team'],
                          manual.loc[manual['verification_status'] == 'conflicting', 'season']))
    spread_family = set(taxonomy['spread_family'])

    def collect(df, team, season, dim):
        sub = df[(df['team'] == team) & (df['season'] == season) & (df['scheme_dimension'] == dim)]
        return '|'.join(sorted(set(sub['scheme_label']))) or None

    rows = []
    for t, s in zip(out['team'], out['season']):
        rec = {'team': t, 'season': int(s)}
        vfam = collect(ver, t, s, 'offensive_family')
        rec['manual_research_status'] = ('verified' if (t, s) in set(zip(ver['team'], ver['season']))
                                         else 'conflicting' if (t, s) in conflicting
                                         else 'researched_no_verified_label' if (t, s) in researched
                                         else 'not_researched')
        for dim in taxonomy['dimensions']:
            rec[f'verified_{dim}'] = collect(ver, t, s, dim)
            rec[f'reported_{dim}'] = collect(rep, t, s, dim)
        rec['verified_spread_family'] = (None if not vfam else int(any(x in spread_family for x in vfam.split('|'))))
        rfam = rec['reported_offensive_family']
        rec['reported_spread_family'] = None if not rfam else int(any(x in spread_family for x in rfam.split('|')))
        rows.append(rec)
    res = pd.DataFrame(rows)
    ib = infoboxes[['team', 'season', 'exists', 'off_scheme']].rename(
        columns={'exists': 'wikipedia_page_found', 'off_scheme': 'wikipedia_off_scheme_raw'})
    res = res.merge(ib, on=['team', 'season'], how='left')
    other_dims = [f'reported_{d}' for d in taxonomy['dimensions'] if d != 'offensive_family']
    res['reported_label_status'] = np.select(
        [res['reported_offensive_family'].notna(),
         res[other_dims].notna().any(axis=1),
         res['wikipedia_off_scheme_raw'].notna(),
         res['wikipedia_page_found'].fillna(False).astype(bool)],
        ['labelled_family', 'labelled_other_dimension_only', 'unmapped_text', 'page_without_scheme_field'],
        default='no_page')
    return res


def is_garbage_time(period, margin) -> np.ndarray:
    """CFBD / Connelly convention: margin > 38 in Q2, > 28 in Q3, > 22 in Q4."""
    p = np.asarray(period, dtype=float)
    m = np.abs(np.asarray(margin, dtype=float))
    return ((p == 2) & (m > 38)) | ((p == 3) & (m > 28)) | ((p >= 4) & (m > 22))


def play_profile(plays: pd.DataFrame) -> pd.DataFrame:
    """Per (season, offense team) down-specific pass rates from play-by-play.
    Pass = dropback outcomes incl. sacks; rush = designed rushes AND scrambles (CFBD
    play types do not separate scrambles or designed QB runs, and do not identify RPOs,
    play-action or formations -- so none of those are computed). Garbage time excluded.
    Standard downs: 1st; 2nd & <=7; 3rd/4th & <=4. Passing downs: the rest."""
    if plays.empty:
        return pd.DataFrame(columns=['season', 'team'])
    p = plays[plays['playType'].isin(PASS_TYPES | RUSH_TYPES)].copy()
    p = p[p['down'].between(1, 4)]
    p['is_pass'] = p['playType'].isin(PASS_TYPES)
    p = p[~is_garbage_time(p['period'], p['offenseScore'] - p['defenseScore'])]
    d, dist = p['down'], p['distance']
    p['standard_down'] = (d == 1) | ((d == 2) & (dist <= 7)) | (d.isin([3, 4]) & (dist <= 4))
    rows = []
    for (season, team), g in p.groupby(['season', 'offense']):
        sd, pdn = g[g['standard_down']], g[~g['standard_down']]
        early = g[g['down'].isin([1, 2])]
        rows.append({'season': int(season), 'team': team, 'pbp_plays_nongarbage': int(len(g)),
                     'pbp_games': int(g['gameId'].nunique()),
                     'pass_rate_nongarbage': g['is_pass'].mean(),
                     'standard_down_pass_rate': sd['is_pass'].mean() if len(sd) else np.nan,
                     'passing_down_pass_rate': pdn['is_pass'].mean() if len(pdn) else np.nan,
                     'early_down_pass_rate': early['is_pass'].mean() if len(early) else np.nan,
                     'standard_down_share': len(sd) / len(g),
                     'plays_per_game_nongarbage': len(g) / max(g['gameId'].nunique(), 1)})
    out = pd.DataFrame(rows)
    # FCS teams appear only in their few games vs FBS opponents -> rates from a handful
    # of plays are flagged unreliable and blanked (never reported from tiny samples).
    out['pbp_reliable'] = out['pbp_plays_nongarbage'] >= MIN_PBP_PLAYS
    rate_cols = ['pass_rate_nongarbage', 'standard_down_pass_rate', 'passing_down_pass_rate', 'early_down_pass_rate',
                 'standard_down_share', 'plays_per_game_nongarbage']
    out.loc[~out['pbp_reliable'], rate_cols] = np.nan
    return out


MIN_PBP_PLAYS = 300


def qb_scheme_seasons(pcs: pd.DataFrame, schemes: pd.DataFrame, profile: pd.DataFrame,
                      environment: pd.DataFrame) -> pd.DataFrame:
    """Per (prospect, pre-draft college season): the team-season's labels (each tier)
    and quantitative profile. Joined on the EXACT (team, season) the QB played for, so
    transfers pick up each school's offense and a label never leaks across seasons."""
    cols = ['qb_draft_prospect_id', 'season', 'team', 'attempts', 'is_final_college_season', 'seasons_before_draft']
    s = pcs[cols].merge(schemes, on=['team', 'season'], how='left')
    if not profile.empty:
        # the team-season table may already carry the profile; never merge it twice
        # (a second merge would suffix the columns _x/_y and silently blank the features)
        new = [c for c in profile.columns if c not in s.columns or c in ('season', 'team')]
        if len(new) > 2:
            s = s.merge(profile[new], on=['season', 'team'], how='left')
    env_cols = ['season', 'team', 'pass_rate', 'plays_per_game', 'seconds_per_play', 'sp_pace', 'off_ppa',
                'off_success_rate']
    if environment is not None and not environment.empty:
        s = s.merge(environment[[c for c in env_cols if c in environment.columns]], on=['season', 'team'], how='left')
    return s


def _share(g: pd.DataFrame, flag_col: str) -> float:
    k = g[g[flag_col].notna()]
    w = k['attempts'].fillna(0)
    return float(np.average(k[flag_col].astype(float), weights=w)) if w.sum() > 0 else np.nan


def _flag(labels: str | None, label: str):
    if labels is None or (isinstance(labels, float) and np.isnan(labels)):
        return np.nan
    return float(label in str(labels).split('|'))


def qb_scheme_features(seasons: pd.DataFrame, draft_years: pd.Series) -> pd.DataFrame:
    """Aggregated pre-draft scheme features per prospect (registry group names:
    scheme_verified / scheme_reported / scheme_quantitative)."""
    out = []
    for pid, g in seasons.sort_values('season').groupby('qb_draft_prospect_id'):
        fin = g[g['is_final_college_season'].astype(bool)].iloc[-1] if g['is_final_college_season'].any() else g.iloc[-1]
        rec = {'qb_draft_prospect_id': pid}
        # A. verified (manual) -- sparse by design
        rec['scheme_ver_final_known'] = float(pd.notna(fin['verified_offensive_family']))
        rec['scheme_ver_final_spread_family'] = (np.nan if pd.isna(fin['verified_spread_family'])
                                                 else float(fin['verified_spread_family']))
        # A'. reported (Wikipedia infobox, low confidence)
        rec['scheme_rep_final_known'] = float(pd.notna(fin['reported_offensive_family']))
        for lab in ('spread', 'air_raid', 'pro_style', 'spread_option', 'multiple_hybrid'):
            rec[f'scheme_rep_final_{lab}'] = _flag(fin['reported_offensive_family'], lab)
        rec['scheme_rep_final_spread_family'] = (np.nan if pd.isna(fin['reported_spread_family'])
                                                 else float(fin['reported_spread_family']))
        att = g['attempts'].fillna(0)
        known = g['reported_offensive_family'].notna()
        rec['scheme_rep_career_label_coverage'] = float(att[known].sum() / att.sum()) if att.sum() > 0 else np.nan
        rec['scheme_rep_career_spread_family_share'] = _share(g, 'reported_spread_family')
        g = g.assign(_pro=g['reported_offensive_family'].map(lambda x: _flag(x, 'pro_style')))
        rec['scheme_rep_career_pro_style_share'] = _share(g, '_pro')
        fams = {f for x in g['reported_offensive_family'].dropna() for f in str(x).split('|')}
        rec['scheme_rep_n_distinct_families'] = float(len(fams)) if fams else np.nan
        # B. quantitative (statistics, not schemes)
        rec['scheme_q_final_standard_down_pass_rate'] = fin.get('standard_down_pass_rate', np.nan)
        rec['scheme_q_final_passing_down_pass_rate'] = fin.get('passing_down_pass_rate', np.nan)
        rec['scheme_q_final_early_down_pass_rate'] = fin.get('early_down_pass_rate', np.nan)
        rec['scheme_q_final_pass_rate_nongarbage'] = fin.get('pass_rate_nongarbage', np.nan)
        if 'standard_down_pass_rate' in g:
            k = g[g['standard_down_pass_rate'].notna()]
            w = k['attempts'].fillna(0)
            rec['scheme_q_career_standard_down_pass_rate'] = (float(np.average(k['standard_down_pass_rate'], weights=w))
                                                              if w.sum() > 0 else np.nan)
        rec['final_observed_season_gap'] = float(draft_years[pid] - 1 - fin['season'])
        rec['scheme_n_schools'] = float(g['team'].nunique())
        out.append(rec)
    return pd.DataFrame(out)
