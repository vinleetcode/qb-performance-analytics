"""Phase-4 dataset: v1 dataset + era-normalized college features + multi-threshold
targets + historical (2005-2021) eligibility and label-availability bookkeeping.

Run:  python -m src.modeling.build_dataset_v2

Reads only local data; makes NO network calls; never writes to the database (it is
opened read-only) or to data/model_ready/. v1 files and models are not touched.

Writes:
  data/modeling/college_reference_population_v2.0.0.parquet   (all-passer reference population)
  data/modeling/college_season_baselines_v2.0.0.parquet       (season & conference-season baselines)
  data/modeling/qb_success_features_v2.0.0.parquet            (predictors only)
  data/modeling/qb_success_targets_v2.0.0.parquet             (targets + eligibility, no predictors)
  data/modeling/dataset_manifest_v2.json
  reports/model_feature_dictionary_v2.csv

FEATURE STRATEGIES (each at two levels: basic = 6 features, context = 11):
  raw_v1      v1 definitions exactly (career totals; truncated for 2005-2008 classes, whose
              earlier college seasons predate CFBD coverage -- kept for continuity)
  raw         truncation-robust definitions in raw units:
                final-season completion %, YPA, TD %, INT %, attempts per game,
                rushing yards per game (games-weighted mean over the last <=3 pre-draft
                seasons), final-season opponent-adjusted YPA and schedule strength
  season_z    the same features as season z-scores vs the FBS reference population
  season_pct  the same features as season percentiles
  conf_z      the same features as conference-season z-scores (season fallback)
  combo       season_z where the reference population drifted >= DRIFT_MIN SD between
              2005-09 and 2021-25, raw elsewhere (decided from the reference population only)
  Context adds three era-stable features to every strategy: team recruiting percentile
  (already a within-season percentile), age at draft, height.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.targets import OBS_END_SEASON
from src.modeling import build_dataset as BD
from src.modeling import era
from src.modeling import features as F

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'data' / 'modeling'
REPORTS = ROOT / 'reports'
DATASET_VERSION = '2.0.0'
FEATURES_FILE = OUT_DIR / f'qb_success_features_v{DATASET_VERSION}.parquet'
TARGETS_FILE = OUT_DIR / f'qb_success_targets_v{DATASET_VERSION}.parquet'
BASELINES_FILE = OUT_DIR / f'college_season_baselines_v{DATASET_VERSION}.parquet'

HISTORICAL_DRAFT_YEARS = (2005, 2021)
HOLDOUT_YEARS = (2019, 2021)
THRESHOLDS = (16, 24, 28, 32, 36, 40, 64)
LABEL_EMBARGO_YEARS = BD.WINDOW            # class d's 5-season label is observable at the draft of d+5
RUSH_WINDOW_SEASONS = 3
DRIFT_MIN = 0.30
EARLY_ERA, LATE_ERA = (2005, 2009), (2021, 2025)

# short feature name -> era stat
STAT_OF = {'cmp': 'completion_pct', 'ypa': 'yards_per_attempt', 'td': 'td_pct', 'int': 'int_pct',
           'att_pg': 'attempts_per_game', 'rush_pg': 'rush_yards_per_game', 'opp_ypa': 'opp_adj_ypa',
           'sos': 'sos_pass_defense'}
BASIC_SHORT = ['cmp', 'ypa', 'td', 'int', 'att_pg', 'rush_pg']
CONTEXT_SHORT = BASIC_SHORT + ['opp_ypa', 'sos']
SHARED_CONTEXT = ['talent_recruiting_4yr_pctile', 'age_at_draft', 'height_inches']
STRATEGIES = ('raw_v1', 'raw', 'season_z', 'season_pct', 'conf_z', 'combo')


def drift_summary(ref: pd.DataFrame) -> pd.DataFrame:
    """Standardized shift of the reference population between the early and late era,
    per stat (no outcome data involved)."""
    r = ref[ref['in_reference']]
    rows = []
    for short, stat in STAT_OF.items():
        e = r.loc[r['season'].between(*EARLY_ERA), stat].dropna()
        lt = r.loc[r['season'].between(*LATE_ERA), stat].dropna()
        pooled = np.sqrt(((len(e) - 1) * e.var() + (len(lt) - 1) * lt.var()) / (len(e) + len(lt) - 2))
        yearly = r.groupby('season')[stat].mean()
        rows.append({'feature': short, 'stat': stat, 'early_mean': e.mean(), 'late_mean': lt.mean(),
                     'early_sd': e.std(), 'late_sd': lt.std(), 'std_drift': (lt.mean() - e.mean()) / pooled,
                     'sd_ratio_late_early': lt.std() / e.std(),
                     'trend_spearman': yearly.reset_index().corr(method='spearman').iloc[0, 1]})
    return pd.DataFrame(rows)


def combo_choice(drift: pd.DataFrame) -> dict:
    d = drift.set_index('feature')['std_drift'].abs()
    return {s: ('season_z' if d[s] >= DRIFT_MIN else 'raw') for s in CONTEXT_SHORT}


def feature_sets(combo: dict) -> dict:
    """(strategy, level) -> feature list."""
    out = {('raw_v1', 'basic'): list(F.GROUP_A_BASIC), ('raw_v1', 'context'): list(F.FEATURE_SETS['B_context'])}
    for s in ('raw', 'season_z', 'season_pct', 'conf_z'):
        out[(s, 'basic')] = [f'{x}__{s}' for x in BASIC_SHORT]
        out[(s, 'context')] = [f'{x}__{s}' for x in CONTEXT_SHORT] + SHARED_CONTEXT
    out[('combo', 'basic')] = [f'{x}__{combo[x]}' for x in BASIC_SHORT]
    out[('combo', 'context')] = [f'{x}__{combo[x]}' for x in CONTEXT_SHORT] + SHARED_CONTEXT
    return out


def prepare_seasons(pcs: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    s = pcs.copy()
    g = s['games_with_pass_attempt']
    s['attempts_per_game'] = np.where(g > 0, s['attempts'] / g, np.nan)
    s['rush_yards_per_game'] = np.where(g > 0, s['rushing_yards'] / g, np.nan)
    s = s.merge(ref[['cfbd_player_id', 'season', 'is_fbs']], on=['cfbd_player_id', 'season'], how='left')
    return era.normalize_frame(s, ref)


def prospect_features(seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per prospect: final-season normalized values + windowed rushing."""
    s = seasons.sort_values(['qb_draft_prospect_id', 'season', 'attempts'], ascending=[True, True, False])
    fin = s[s['is_final_college_season'] == True].drop_duplicates('qb_draft_prospect_id')  # noqa: E712
    out = fin[['qb_draft_prospect_id', 'season', 'team', 'conference', 'attempts', 'is_fbs']].rename(
        columns={'season': 'v2_final_season', 'team': 'v2_final_team', 'conference': 'v2_final_conference',
                 'attempts': 'v2_final_attempts', 'is_fbs': 'v2_final_is_fbs'})
    for short in CONTEXT_SHORT:
        if short == 'rush_pg':
            continue
        stat = STAT_OF[short]
        out[f'{short}__raw'] = fin[stat].to_numpy()
        for m in era.METHODS:
            out[f'{short}__{m}'] = fin[f'{stat}__{m}'].to_numpy()
    out[f'{STAT_OF["sos"]}_conf_fallback'] = fin['sos_pass_defense__conf_fallback'].to_numpy()
    out['cmp_conf_fallback'] = fin['completion_pct__conf_fallback'].to_numpy()
    # rushing: games-weighted mean over the last RUSH_WINDOW_SEASONS pre-draft seasons
    w = s[s['seasons_before_draft'] <= RUSH_WINDOW_SEASONS].copy()
    w = w[w['games_with_pass_attempt'].fillna(0) > 0]
    agg = {}
    for col, name in [('rush_yards_per_game', 'rush_pg__raw')] + [(f'rush_yards_per_game__{m}', f'rush_pg__{m}')
                                                                    for m in era.METHODS]:
        v = w[[col, 'games_with_pass_attempt', 'qb_draft_prospect_id']].dropna()
        agg[name] = v.groupby('qb_draft_prospect_id').apply(
            lambda g, c=col: np.average(g[c], weights=g['games_with_pass_attempt']), include_groups=False)
    out = out.merge(pd.DataFrame(agg), left_on='qb_draft_prospect_id', right_index=True, how='left')
    out['rush_pg_n_seasons'] = out['qb_draft_prospect_id'].map(w.groupby('qb_draft_prospect_id')['season'].nunique())
    return out


def threshold_targets(t: pd.DataFrame) -> pd.DataFrame:
    """success_5yr_{k}: 1 if >= k starts in seasons 1-5, 0 only with a complete window."""
    t = t.copy()
    s_final = t['nfl_starts_first_5_seasons']           # NaN unless window complete and record verified
    s_to_date = t['nfl_starts_first_5_seasons_to_date']
    for k in THRESHOLDS:
        y = pd.Series(pd.NA, index=t.index, dtype='Int64')
        y[s_to_date >= k] = 1
        y[s_final.notna() & (s_final < k)] = 0
        t[f'success_5yr_{k}'] = y
    assert (t['success_5yr_32'].fillna(-1) == t['career_success'].fillna(-1)).all()
    return t


def label_observable_at(draft_year_of_test: int, class_year: int) -> bool:
    """Is class `class_year`'s five-season label observable by the draft of `draft_year_of_test`?
    Window = seasons class_year..class_year+4; season class_year+4 ends (Jan) before the April
    draft of class_year+5."""
    return class_year + LABEL_EMBARGO_YEARS <= draft_year_of_test


def label_availability_table(t: pd.DataFrame, test_years) -> pd.DataFrame:
    rows = []
    h = t[t['v2_historical']]
    for ty in test_years:
        obs = h[[label_observable_at(ty, d) for d in h['draft_year']]]
        retro = h[h['draft_year'] < ty]
        rows.append({'test_draft_year': ty, 'latest_class_with_observable_label': ty - LABEL_EMBARGO_YEARS,
                     'embargoed_train_n': len(obs), 'embargoed_train_pos': int((obs['career_success'] == 1).sum()),
                     'retrospective_train_n': len(retro), 'retrospective_train_pos': int((retro['career_success'] == 1).sum()),
                     'modern_only_retrospective_n': int(((t['comparison_cohort']) & (t['draft_year'] < ty)).sum()),
                     'classes_unobservable_but_used_retrospectively': f'{max(ty - LABEL_EMBARGO_YEARS + 1, 2005)}-{ty - 1}'})
    return pd.DataFrame(rows)


def dictionary(combo: dict, feats: pd.DataFrame, mask) -> pd.DataFrame:
    desc = {'cmp': 'final-season completion %', 'ypa': 'final-season yards per attempt', 'td': 'final-season TD %',
            'int': 'final-season INT % (normalized scores: higher = FEWER interceptions)',
            'att_pg': 'final-season pass attempts per game with a pass attempt',
            'rush_pg': f'rushing yards per game, games-weighted mean over the last <= {RUSH_WINDOW_SEASONS} pre-draft seasons',
            'opp_ypa': 'final-season opponent-adjusted YPA (leave-one-out opponent pass defense)',
            'sos': 'final-season mean opponent pass-defense YPA allowed vs league (normalized: higher = TOUGHER)'}
    meth = {'raw': 'raw units', 'season_z': f'z vs season FBS reference (>= {era.REF_MIN_ATTEMPTS} att), clip +/-{era.Z_CLIP}',
            'season_pct': 'mid-rank percentile vs season FBS reference (0-1)',
            'conf_z': f'z vs (season, conference) reference; season fallback if < {era.CONF_MIN_GROUP} passers'}
    rows = []
    sub = feats[mask]
    for short in CONTEXT_SHORT:
        for m in ('raw',) + era.METHODS:
            col = f'{short}__{m}'
            rows.append({'feature': col, 'base_stat': desc[short], 'method': meth[m],
                         'in_combo': combo[short] == m, 'availability_cutoff': 'end of final college season',
                         'missing_rate_historical': float(sub[col].isna().mean())})
    for c in SHARED_CONTEXT:
        rows.append({'feature': c, 'base_stat': 'era-stable context (see v1 dictionary)', 'method': 'as v1',
                     'in_combo': True, 'availability_cutoff': 'before draft day',
                     'missing_rate_historical': float(sub[c].isna().mean())})
    d = pd.DataFrame(rows)
    d['normalization_version'] = era.NORMALIZATION_VERSION
    return d


def build(rebuild_reference: bool = False):
    f1, t1, meta1 = BD.build()                          # v1 features/targets, recomputed (same code, same data)
    ref = era.load_or_build_reference(rebuild=rebuild_reference)
    pcs = pd.read_parquet(BD.MODEL_READY / 'prospect_college_seasons.parquet')
    seasons = prepare_seasons(pcs, ref)
    pf = prospect_features(seasons)
    feats = f1.merge(pf, on='qb_draft_prospect_id', how='left')
    leak = [c for c in feats.columns if c.startswith(('started_', 'games_started', 'nfl_', 'career_success', 'success_'))]
    assert not leak, leak

    t = threshold_targets(t1)
    fm = feats.set_index('qb_draft_prospect_id')
    get = lambda c: t['qb_draft_prospect_id'].map(fm[c])  # noqa: E731
    final_ok = (get('v2_final_season') == t['draft_year'] - 1)
    t['v2_identity_linked'] = get('identity_source').notna()
    t['v2_final_season_is_pre_draft_season'] = final_ok.fillna(False).astype(bool)
    t['v2_final_fbs'] = get('v2_final_is_fbs').fillna(False).astype(bool)
    t['v2_final_attempts_ok'] = (get('v2_final_attempts') >= era.PROSPECT_MIN_ATTEMPTS).fillna(False).astype(bool)
    base = t['v2_identity_linked'] & t['v2_final_season_is_pre_draft_season'] & t['v2_final_fbs'] & t['v2_final_attempts_ok']
    t['v2_feature_eligible'] = base
    t['v2_historical'] = (base & t['draft_year'].between(*HISTORICAL_DRAFT_YEARS) & t['window_5yr_complete']
                          & t['career_success'].notna())
    t['v2_common_holdout'] = t['v2_historical'] & t['comparison_cohort'] & t['draft_year'].between(*HOLDOUT_YEARS)
    t['v2_scoring_2022_2026'] = base & t['draft_year'].between(2022, 2026)

    drift = drift_summary(ref)
    combo = combo_choice(drift)
    baselines = pd.concat([era.group_baselines(ref[ref['in_reference']], st, ('season',)).assign(level='season')
                           for st in era.NORM_STATS] +
                          [era.group_baselines(ref[ref['in_reference']].dropna(subset=['conference']), st,
                                               ('season', 'conference')).assign(level='conference_season')
                           for st in era.NORM_STATS], ignore_index=True)
    return feats, t, {'v1_meta': meta1, 'drift': drift, 'combo': combo, 'baselines': baselines, 'ref': ref,
                      'seasons': seasons}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(rebuild_reference: bool = False) -> dict:
    feats, t, aux = build(rebuild_reference)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_FILE, index=False)
    t.drop(columns=['gsis_id']).to_parquet(TARGETS_FILE, index=False)
    aux['baselines'].to_parquet(BASELINES_FILE, index=False)
    combo = aux['combo']
    sets = feature_sets(combo)
    manifest = {'dataset_version': DATASET_VERSION, 'normalization_version': era.NORMALIZATION_VERSION,
                'observation_end_season': OBS_END_SEASON,
                'reference_population_rule': {'fbs_team_season': True, 'min_attempts': era.REF_MIN_ATTEMPTS,
                                              'min_games_with_pass_attempt': era.REF_MIN_GAMES, 'weighting': 'unweighted',
                                              'conference_min_group': era.CONF_MIN_GROUP, 'z_clip': era.Z_CLIP,
                                              'prospect_min_attempts': era.PROSPECT_MIN_ATTEMPTS},
                'combo_choice': combo, 'drift_min_sd': DRIFT_MIN,
                'feature_sets': {f'{k[0]}|{k[1]}': v for k, v in sets.items()},
                'files': {p.name: sha256(p) for p in (FEATURES_FILE, TARGETS_FILE, BASELINES_FILE, era.REF_FILE)}}
    (OUT_DIR / f'dataset_manifest_v{DATASET_VERSION}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    hist_mask = feats['qb_draft_prospect_id'].map(t.set_index('qb_draft_prospect_id')['v2_historical']).fillna(False).astype(bool).values
    dictionary(combo, feats, hist_mask).to_csv(REPORTS / 'model_feature_dictionary_v2.csv', index=False)
    h = t[t['v2_historical']]
    print(f'historical eligible {len(h)} ({int((h.career_success == 1).sum())}+) classes {h.draft_year.nunique()} | '
          f'common holdout {int(t.v2_common_holdout.sum())} | combo {combo}')
    return {'targets': t, 'features': feats, **aux}


if __name__ == '__main__':
    main()
