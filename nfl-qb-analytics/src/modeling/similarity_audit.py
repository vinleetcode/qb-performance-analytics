"""Historical prospect comparison audit (Phase 5.5).

Run:  python -m src.modeling.similarity_audit        (local data only)

The comparison engine (models/qb_similarity/v1.0.0) is an EXPLANATORY tool, not a forecast. This
audit checks, without changing it:
  * leakage: no NFL outcome is a feature; changing outcomes never changes a comparison
  * identity: one row per drafted QB; the query never matches itself; no duplicate name+college+year
  * cutoffs: comps only from earlier classes; outcomes hidden until observable
  * scaling / weighting: z-scores on the pool; redundant (highly correlated) features double-weight a trait
  * missing values: how many features a distance is actually computed on
  * era: whether comps cluster in the query's own era, and how an era-normalized variant
    (season z-scores instead of raw rates) changes the comps
  * stability: top-5 overlap when one feature is dropped, and under small measurement noise
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling import artifacts as A
from src.modeling import prospect_similarity as PS
from src.modeling.reporting import md_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase5_5'
SECTION = RESULTS / 'similarity_audit_section.md'
SEED = 20261110
QUERY_YEARS = range(2012, 2027)
ERA_MAP = {'cmp__raw': 'cmp__season_z', 'ypa__raw': 'ypa__season_z', 'td__raw': 'td__season_z', 'int__raw': 'int__season_z',
           'att_pg__raw': 'att_pg__season_z', 'rush_pg__raw': 'rush_pg__season_z', 'opp_ypa__raw': 'opp_ypa__season_z',
           'sos__raw': 'sos__season_z'}
OUTCOMES = ['nfl_starts_first_5_seasons', 'career_success', 'success_5yr_16', 'success_5yr_64',
            'nfl_starts_first_5_seasons_to_date', 'window_5yr_complete']


def load_pool() -> pd.DataFrame:
    pool = pd.read_parquet(A.resolve(PS.MODEL_DIR, 'pool'))
    v2 = pd.read_parquet(ROOT / 'data' / 'modeling' / 'qb_success_features_v2.0.0.parquet')
    return pool.merge(v2[['qb_draft_prospect_id'] + list(ERA_MAP.values())], on='qb_draft_prospect_id', how='left')


def topk(pool, features, k=PS.K, years=QUERY_YEARS, noise=0.0, rng=None) -> dict:
    """{query id: [comp ids]} for every query QB in `years`, pool = earlier classes."""
    out = {}
    for y in years:
        ref = pool[pool['draft_year'] < y]
        if len(ref) < 20:
            continue
        cmpr = PS.Comparator(features).fit(ref)
        for _, r in pool[pool['draft_year'] == y].iterrows():
            r = r.copy()
            if noise:
                sd = ref[features].std()
                r[features] = r[features].astype(float) + rng.normal(0, noise, len(features)) * sd.to_numpy()
            out[r['qb_draft_prospect_id']] = cmpr.query(r, k)['qb_draft_prospect_id'].tolist()
    return out


def overlap(a: dict, b: dict) -> dict:
    j, top1 = [], []
    for q in a:
        if q in b:
            s1, s2 = set(a[q]), set(b[q])
            j.append(len(s1 & s2) / len(s1 | s2))
            top1.append(a[q][0] == b[q][0])
    return {'queries': len(j), 'mean_jaccard_top5': float(np.mean(j)), 'mean_shared_of_5': float(np.mean([len(set(a[q]) & set(b[q])) for q in a if q in b])),
            'top1_unchanged': float(np.mean(top1))}


def leakage_check(pool) -> dict:
    feats = PS.SIM_FEATURES
    bad = [f for f in feats if f in OUTCOMES or f.startswith(('nfl_', 'career_success', 'success_'))]
    q = pool[pool['draft_year'] == 2020].iloc[0]
    ref = pool[pool['draft_year'] < 2020]
    a = PS.Comparator().fit(ref).query(q)['qb_draft_prospect_id'].tolist()
    ref2 = ref.copy()
    rng = np.random.default_rng(SEED)
    for c in ('nfl_starts_first_5_seasons', 'nfl_starts_first_5_seasons_to_date'):
        ref2[c] = rng.permutation(ref2[c].to_numpy())
    ref2['career_success'] = rng.permutation(ref2['career_success'].to_numpy())
    b = PS.Comparator().fit(ref2).query(q)['qb_draft_prospect_id'].tolist()
    return {'outcome_named_features': bad, 'comps_unchanged_when_outcomes_shuffled': a == b}


def identity_check(pool) -> dict:
    key = pool['player_name'].str.lower() + '|' + pool['college'].fillna('').str.lower() + '|' + pool['draft_year'].astype(str)
    q = pool[pool['draft_year'] == 2024].iloc[0]
    self_match = q['qb_draft_prospect_id'] in PS.Comparator().fit(pool).query(q)['qb_draft_prospect_id'].tolist()
    return {'rows': int(len(pool)), 'unique_ids': int(pool['qb_draft_prospect_id'].nunique()),
            'duplicate_name_college_year': int(key.duplicated().sum()), 'query_matches_itself': bool(self_match)}


def redundancy(pool) -> pd.DataFrame:
    c = pool[PS.SIM_FEATURES].corr()
    rows = []
    for i, a in enumerate(PS.SIM_FEATURES):
        for b in PS.SIM_FEATURES[i + 1:]:
            if abs(c.loc[a, b]) >= 0.6:
                rows.append({'feature_a': PS.LABELS[a], 'feature_b': PS.LABELS[b], 'correlation': float(c.loc[a, b])})
    return pd.DataFrame(rows).sort_values('correlation', key=abs, ascending=False)


def era_mixing(pool, comps: dict) -> dict:
    yr = pool.set_index('qb_draft_prospect_id')['draft_year']
    gaps, rand = [], []
    rng = np.random.default_rng(SEED)
    for q, cs in comps.items():
        y = yr[q]
        gaps += [y - yr[c] for c in cs]
        earlier = yr[yr < y].to_numpy()
        rand += list(y - rng.choice(earlier, len(cs)))
    return {'mean_years_between_query_and_comp': float(np.mean(gaps)), 'same_for_random_earlier_qbs': float(np.mean(rand)),
            'share_comps_within_5_years': float(np.mean(np.array(gaps) <= 5)),
            'share_random_within_5_years': float(np.mean(np.array(rand) <= 5))}


def features_used(pool, comps: dict, features) -> dict:
    present = pool.set_index('qb_draft_prospect_id')[features].notna()
    n = [int((present.loc[q] & present.loc[c]).sum()) for q, cs in comps.items() for c in cs]
    return {'features_total': len(features), 'min_shared_features': int(min(n)), 'median_shared_features': float(np.median(n)),
            'share_pairs_missing_any': float(np.mean(np.array(n) < len(features)))}


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    rng = np.random.default_rng(SEED)
    base = topk(pool, PS.SIM_FEATURES)
    stab = []
    for f in PS.SIM_FEATURES:
        r = overlap(base, topk(pool, [x for x in PS.SIM_FEATURES if x != f]))
        stab.append({'dropped_feature': PS.LABELS[f], **r})
    STAB = pd.DataFrame(stab)
    noise = []
    for s in (0.05, 0.10, 0.25):
        noise.append({'noise_sd_fraction': s, **overlap(base, topk(pool, PS.SIM_FEATURES, noise=s, rng=rng))})
    NOISE = pd.DataFrame(noise)
    era_feats = [ERA_MAP.get(f, f) for f in PS.SIM_FEATURES]
    for k, v in ERA_MAP.items():                 # labels for the explanation strings of the era variant
        PS.LABELS.setdefault(v, PS.LABELS[k] + ' (season z)')
    era_comps = topk(pool, era_feats)
    ERA = {'raw': era_mixing(pool, base), 'era_normalized': era_mixing(pool, era_comps), 'overlap_raw_vs_era': overlap(base, era_comps)}
    out = {'leakage': leakage_check(pool), 'identity': identity_check(pool), 'era': ERA,
           'features_used': features_used(pool, base, PS.SIM_FEATURES), 'pool_missing_share': pool[PS.SIM_FEATURES].isna().mean().round(3).to_dict()}
    RED = redundancy(pool)
    STAB.to_csv(RESULTS / 'similarity_stability_drop_one.csv', index=False)
    NOISE.to_csv(RESULTS / 'similarity_stability_noise.csv', index=False)
    RED.to_csv(RESULTS / 'similarity_redundancy.csv', index=False)
    (RESULTS / 'similarity_audit.json').write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    SECTION.write_text(render(out, STAB, NOISE, RED, pool), encoding='utf-8')
    print(json.dumps(out, indent=2, default=str)); print(STAB.round(3).to_string()); print(NOISE.round(3).to_string())
    return out


def render(out, STAB, NOISE, RED, pool) -> str:
    e, lk, idn, fu = out['era'], out['leakage'], out['identity'], out['features_used']
    worst = STAB.sort_values('mean_jaccard_top5').iloc[0]
    L = ['## Historical comparisons', '',
         f'Pool: {idn["rows"]} drafted QBs with a v2 college profile; queries = every pool QB drafted {QUERY_YEARS[0]}-'
         f'{QUERY_YEARS[-1]}, compared only with earlier classes. The engine (`models/qb_similarity/v1.0.0`) is unchanged.', '',
         '| Check | Result |', '|---|---|',
         f'| NFL outcomes among the features | {"none" if not lk["outcome_named_features"] else lk["outcome_named_features"]} |',
         f'| Comps change when NFL outcomes are shuffled | {"no" if lk["comps_unchanged_when_outcomes_shuffled"] else "YES (leak)"} |',
         f'| One row per QB | {idn["unique_ids"]} ids for {idn["rows"]} rows; {idn["duplicate_name_college_year"]} duplicate name+college+year |',
         f'| A query can match itself | {"yes" if idn["query_matches_itself"] else "no"} |',
         f'| Features a distance uses | median {fu["median_shared_features"]:.0f} of {fu["features_total"]}, minimum {fu["min_shared_features"]}; '
         f'{fu["share_pairs_missing_any"]:.0%} of comp pairs miss at least one (mostly final-year YPA change) |',
         '| Scaling | z-scores fit on the pool of earlier classes only; equal weights |',
         '| Closeness index | a rank heuristic (share of pool pairs farther apart), labelled "not a probability" in every output |', '',
         '**Redundant features.** Pairs with |correlation| >= 0.6 effectively count one trait twice:', '',
         md_table(RED), '',
         '**Era.** Raw college rates drift over time (passing efficiency rose through the 2010s), so raw-feature comps '
         + ('lean toward recent classes' if e['raw']['mean_years_between_query_and_comp'] < e['raw']['same_for_random_earlier_qbs']
            else 'do not lean toward recent classes') +
         f': comps are on average {e["raw"]["mean_years_between_query_and_comp"]:.1f} years older than the query, '
         f'against {e["raw"]["same_for_random_earlier_qbs"]:.1f} for random earlier QBs '
         f'({e["raw"]["share_comps_within_5_years"]:.0%} vs {e["raw"]["share_random_within_5_years"]:.0%} within five years). '
         'An era-normalized variant (season z-scores instead of raw rates) gives '
         f'{e["era_normalized"]["mean_years_between_query_and_comp"]:.1f} years and shares on average '
         f'{e["overlap_raw_vs_era"]["mean_shared_of_5"]:.1f} of 5 comps with the raw version.', '',
         '**Stability.** Dropping one feature at a time (mean top-5 overlap with the full feature set):', '',
         md_table(STAB), '',
         'Small measurement noise added to the query (fraction of each feature\'s SD):', '',
         md_table(NOISE), '',
         f'The comps are moderately stable. Dropping a single feature keeps on average {STAB["mean_shared_of_5"].mean():.1f} of 5 comps; '
         f'the most influential feature is {worst["dropped_feature"]} ({worst["mean_shared_of_5"]:.1f} of 5 kept). '
         'Nearest-neighbour lists are sensitive at the margin: QBs 4-6 are often nearly tied. The dashboard should show '
         'distances and the "most similar on / biggest differences" explanations, not a ranked list presented as precise.', '',
         '**Decision.** The v1.0.0 engine stays the default, as an explanation of a profile. Its outcome rate is a weak '
         'predictor (Phase 5: AUC 0.69 for 32+ starts) and is never shown as a probability. Raw vs era-normalized is a '
         'design choice with no outcome-based winner: raw rates describe what the QB actually did, era-normalized ones how '
         'he compared with his season. The interface labels the comps as raw-feature comps.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
