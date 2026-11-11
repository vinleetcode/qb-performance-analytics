"""Integrated QB prospect forecast (Phase 5).

    from src.modeling.forecast import forecast_prospect
    f = forecast_prospect(name='Fernando Mendoza', draft_year=2026)        # a drafted QB (post-draft view available)
    f = forecast_prospect(cfbd_player_id='4870906', final_season=2024)     # any college passer-season in the data

    python -m src.modeling.forecast --name "Fernando Mendoza" --draft-year 2026

Returns one dict with every component, each tagged with its evaluation status:
  draft            P(drafted) only inside the verified prospect universe (see draft_model_evaluation.md);
                   round probabilities and pick range "if drafted" for everyone
  starter          P(16+/32+/64+ starts in seasons 1-5): v2 reference model (32+) and the Phase-5 logistic models
  five_year        expected totals and 80% intervals from the structured simulation (pre-draft; post-draft
                   too when the actual pick is known)
  year_by_year     seasons 1-5 means and 80% intervals
  comps            five nearest historically drafted college profiles (outcomes shown only where complete)
All inputs are pre-draft college/combine data, plus the actual pick in the post-draft view only.
Uses saved artifacts; loads player-level reference data from the gitignored data/ directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling import artifacts as A
from src.modeling import career_production as CP
from src.modeling import draft_predictor as DP
from src.modeling import prospect_similarity as PS
from src.modeling import train_v2 as T2

ROOT = Path(__file__).resolve().parents[2]
STATUS = {
    'draft_probability': 'backtested (test drafts 2014-2025); defined only for the verified prospect universe',
    'draft_round_pick': 'backtested (test drafts 2012-2026); proportional-odds round model beats historical round frequencies '
                        '(log loss, RPS) but was added after a first ordinal model failed; pick ranges wide (80% coverage 0.77)',
    'starter_probabilities': 'backtested with a five-season label embargo (test classes 2012-2021); 2019-2021 reused, exploratory',
    'five_year_totals': 'backtested (test classes 2012-2021); 80% intervals measured at 84-93% coverage (conservative)',
    'year_by_year': 'backtested for starts, attempts, passing and rushing yards; other yearly stats not separately evaluated',
    'comps': 'descriptive; comp outcome rates are not calibrated probabilities',
}
V2_MAP = {'cmp__raw': 'completion_pct', 'ypa__raw': 'yards_per_attempt', 'td__raw': 'td_pct', 'int__raw': 'int_pct',
          'att_pg__raw': 'attempts_per_game', 'rush_pg__raw': 'rush_ypg_last3', 'opp_ypa__raw': 'opp_adj_ypa',
          'sos__raw': 'sos_pass_defense', 'talent_recruiting_4yr_pctile': 'team_recruiting_pctile',
          'height_inches': 'height_inches', 'dev_yoy_ypa_final': 'ypa_change_vs_prior',
          'seasons_as_primary_passer': 'n_seasons_100att'}


class Engine:
    def __init__(self):
        self.v2 = joblib.load(T2.MODEL_DIR / 'historical_selected.joblib')
        self._prod = None      # v1.0.0 production files are local-only (gitignored); loaded on first use
        self.draft_prob = {k: joblib.load(DP.MODEL_DIR / f'prob_{k}.joblib') for k in ('college_l2', 'college_combine_l2')}
        self.round_pick = joblib.load(DP.MODEL_DIR / 'round_pick.joblib')
        self.pool = pd.read_parquet(A.resolve(PS.MODEL_DIR, 'pool'))
        self.universe = pd.read_parquet(A.resolve(DP.MODEL_DIR, 'draft_universe'))
        self.v2_data = T2.load()
        self._src = None

    @property
    def prod(self):
        if self._prod is None:
            self._prod = {fs: joblib.load(CP.MODEL_DIR / f'{fs}.joblib') for fs in ('pre_draft', 'post_draft')}
        return self._prod

    def src(self):
        if self._src is None:
            self._src = DP.load_sources()
        return self._src

    # ---------------------------------------------------------------- profile assembly
    def draft_features(self, cfbd_player_id: str, season: int, draft_year: int, name: str | None, birthdate=None) -> pd.Series:
        u = self.universe[(self.universe['cfbd_player_id'] == cfbd_player_id) & (self.universe['season'] == season)]
        if len(u):
            row = u.iloc[0].copy()
        else:
            src = self.src()
            rows = pd.DataFrame({'cfbd_player_id': [cfbd_player_id], 'season': [season], 'draft_year': [draft_year]})
            f = DP.passer_features(src['ref'], rows, src)
            f = DP.attach_combine(f.assign(name=f['name'].fillna(name or '')), src['combine'])
            row = f.iloc[0].copy()
            row['in_population'] = False
            row['population_status'] = 'not_in_verified_universe'
        bd = pd.to_datetime(birthdate, errors='coerce') if birthdate is not None else pd.NaT
        row['age_at_draft'] = (pd.Timestamp(f'{draft_year}-04-25') - bd).days / 365.25 if pd.notna(bd) else np.nan
        return row

    def profile(self, qb_draft_prospect_id=None, name=None, draft_year=None, cfbd_player_id=None, final_season=None) -> dict:
        d = self.v2_data
        if qb_draft_prospect_id is None and name is not None:
            m = d[(d['player_name'] == name) & ((d['draft_year'] == draft_year) if draft_year else True)]
            if len(m):
                qb_draft_prospect_id = m.iloc[0]['qb_draft_prospect_id']
        if qb_draft_prospect_id is not None:
            v = d[d['qb_draft_prospect_id'] == qb_draft_prospect_id].iloc[0]
            src = self.src()
            ident = src['identity'][src['identity']['qb_draft_prospect_id'] == qb_draft_prospect_id]
            cid = ident['cfbd_player_id'].iloc[0] if len(ident) else None
            season = int(v['v2_final_season']) if pd.notna(v.get('v2_final_season')) else int(v['draft_year']) - 1
            dfe = self.draft_features(cid, season, int(v['draft_year']), v['player_name'],
                                      ident['birthdate'].iloc[0] if len(ident) else None) if cid else pd.Series(dtype=float)
            return {'v2': v, 'draft': dfe, 'name': v['player_name'], 'college': v['college'], 'draft_year': int(v['draft_year']),
                    'actual_pick': v.get('overall_pick'), 'actual_round': v.get('round'), 'drafted': True}
        if cfbd_player_id is None or final_season is None:
            raise ValueError('pass qb_draft_prospect_id, name (+draft_year) of a drafted QB, or cfbd_player_id + final_season')
        dy = int(final_season) + 1
        dfe = self.draft_features(str(cfbd_player_id), int(final_season), dy, name)
        v = pd.Series({k: dfe.get(src_col, np.nan) for k, src_col in V2_MAP.items()}, dtype=object)
        v['age_at_draft'] = dfe.get('age_at_draft', np.nan)
        v['env_pass_rate'] = np.nan
        v['draft_year'] = dy
        v['player_name'] = dfe.get('name', name)
        v['qb_draft_prospect_id'] = None
        return {'v2': v, 'draft': dfe, 'name': v['player_name'], 'college': dfe.get('team'), 'draft_year': dy,
                'actual_pick': None, 'actual_round': None, 'drafted': None}

    # ---------------------------------------------------------------- components
    def draft_block(self, p: dict) -> dict:
        f = p['draft']
        if f is None or len(f) == 0:
            return {'available': False, 'reason': 'no college passer-season found for this QB'}
        X = f.to_frame().T
        rp = self.round_pick
        Xr = X.reindex(columns=rp['features']).astype(float)
        probs = rp['prop_odds'].predict_proba(Xr)[0] if 'prop_odds' in rp else rp['ordinal'].predict_proba(Xr)[0]
        pk = rp['pick'].predict(Xr).iloc[0]
        out = {'round_probabilities_if_drafted': {int(r): float(probs[r - 1]) for r in DP.ROUNDS},
               'expected_round_if_drafted': float(probs @ DP.ROUNDS),
               'pick_if_drafted': {'median': float(pk['median_pick']), 'interval_50': [float(pk['q25']), float(pk['q75'])],
                                   'interval_80': [float(pk['q10']), float(pk['q90'])]},
               'in_verified_universe': bool(f.get('in_population', False)),
               'population_status': f.get('population_status')}
        if out['in_verified_universe']:
            pc = float(self.draft_prob['college_l2']['pipeline'].predict_proba(X.reindex(columns=self.draft_prob['college_l2']['features']).astype(float))[:, 1][0])
            pcc = float(self.draft_prob['college_combine_l2']['pipeline'].predict_proba(
                X.reindex(columns=self.draft_prob['college_combine_l2']['features']).astype(float))[:, 1][0])
            out.update({'p_drafted_college_only': pc, 'p_drafted_with_combine': pcc,
                        'unconditional_round_probabilities': {r: pcc * q for r, q in out['round_probabilities_if_drafted'].items()}})
        else:
            out['p_drafted'] = None
            out['p_drafted_note'] = ('not defined: this QB is outside the verified prospect universe '
                                     f'({out["population_status"]}); only the "if drafted" projection applies')
        return out

    def career_block(self, p: dict, n_sims: int) -> dict:
        v = p['v2']
        res = {}
        if p['drafted'] is not True:
            res['note'] = ('career models are trained only on DRAFTED QBs: these projections assume this QB is drafted '
                           '(and the post-draft view needs an actual pick)')
        Xv = v.to_frame().T
        res['p32_v2_reference'] = float(self.v2['pipeline'].predict_proba(Xv.reindex(columns=self.v2['features']).astype(float))[:, 1][0])
        for fs, art in self.prod.items():
            if fs == 'post_draft' and (p['actual_pick'] is None or pd.isna(p['actual_pick'])):
                continue
            row = Xv.copy()
            if fs == 'post_draft':
                row['log_overall_pick'] = np.log(float(p['actual_pick']))
            X = row.reindex(columns=art['features']).astype(float)
            sm = art['structured']
            sims = sm.simulate(X, n_sims=n_sims, seed=CP.SEED)
            s = sm.summarize(sims)
            totals = {t: {'expected': float(s['mean'][0, i]), 'median': float(s['median'][0, i]),
                          'interval_80': [float(s['lo'][0, i]), float(s['hi'][0, i])]} for i, t in enumerate(CP.TARGETS)}
            tot = sims[0].sum(axis=1)
            att = tot[:, CP.TARGETS.index('pass_attempts')]
            with np.errstate(invalid='ignore', divide='ignore'):
                eff = {'completion_pct': 100 * tot[:, CP.TARGETS.index('pass_completions')] / att,
                       'yards_per_attempt': tot[:, CP.TARGETS.index('pass_yards')] / att}
            ok = att >= 300
            totals['efficiency_if_300_plus_attempts'] = {k: (float(np.median(x[ok])) if ok.sum() >= 50 else None) for k, x in eff.items()}
            years = [{'season': y + 1, **{t: {'expected': float(s['year_mean'][0, y, i]),
                                               'interval_80': [float(s['year_lo'][0, y, i]), float(s['year_hi'][0, y, i])]}
                                          for i, t in enumerate(CP.TARGETS)}} for y in range(CP.YEARS)]
            thr = {f'p_{k}plus_starts_logit': float(CP._prob(art['threshold_logits'][k], X)[0]) for k in CP.THRESHOLDS}
            thr.update({f'p_{k}plus_starts_simulation': float(s['p_starts'][k][0]) for k in CP.THRESHOLDS})
            res[fs] = {'five_year': totals, 'year_by_year': years, 'starter': thr,
                       'p_any_nfl_game': float(CP._prob(sm.m_appear, X)[0])}
        return res

    def comps_block(self, p: dict) -> list[dict]:
        v = p['v2'].copy()
        pool = self.pool[self.pool['draft_year'] < p['draft_year']]
        if pool.empty:
            return []
        c = PS.Comparator().fit(pool).query(v, PS.K)
        keep = ['player_name', 'college', 'draft_year', 'round', 'overall_pick', 'distance', 'closeness_index_heuristic',
                'most_similar_on', 'biggest_differences_comp_vs_query', 'nfl_starts_first_5_seasons', 'window_5yr_complete']
        return c[keep].to_dict('records')

    def forecast(self, n_sims: int = 2000, **key) -> dict:
        p = self.profile(**key)
        return {'prospect': {'name': p['name'], 'college': p['college'], 'draft_year': p['draft_year'],
                             'actual_round': None if p['actual_round'] is None or pd.isna(p['actual_round']) else int(p['actual_round']),
                             'actual_pick': None if p['actual_pick'] is None or pd.isna(p['actual_pick']) else int(p['actual_pick'])},
                'draft': self.draft_block(p), 'career': self.career_block(p, n_sims), 'comps': self.comps_block(p),
                'evaluation_status': STATUS}


_ENGINE = None


def forecast_prospect(n_sims: int = 2000, **key) -> dict:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE.forecast(n_sims=n_sims, **key)


def summarize(f: dict) -> str:
    p, dft, car = f['prospect'], f['draft'], f['career']
    L = [f'{p["name"]} ({p["college"]}), draft class {p["draft_year"]}'
         + (f', actual pick {p["actual_pick"]} (round {p["actual_round"]})' if p['actual_pick'] else '')]
    if dft.get('round_probabilities_if_drafted'):
        rp = dft['round_probabilities_if_drafted']
        L.append(f'  Draft: P(drafted) = ' + (f'{dft["p_drafted_with_combine"]:.2f} (college+combine), {dft["p_drafted_college_only"]:.2f} (college only)'
                                              if dft.get('in_verified_universe') else 'not defined (' + str(dft.get('population_status')) + ')'))
        L.append('  If drafted: ' + ', '.join(f'R{r} {q:.2f}' for r, q in rp.items()) +
                 f'; expected round {dft["expected_round_if_drafted"]:.1f}; pick ~{dft["pick_if_drafted"]["median"]:.0f} '
                 f'(80%: {dft["pick_if_drafted"]["interval_80"][0]:.0f}-{dft["pick_if_drafted"]["interval_80"][1]:.0f})')
    L.append(f'  P(32+ starts, v2 reference) = {car["p32_v2_reference"]:.2f}')
    for fs in ('pre_draft', 'post_draft'):
        if fs not in car:
            continue
        c = car[fs]
        t = c['five_year']
        L.append(f'  [{fs}] P(any game) {c["p_any_nfl_game"]:.2f}; P(16/32/64+ starts) ' +
                 '/'.join(f'{c["starter"][f"p_{k}plus_starts_logit"]:.2f}' for k in CP.THRESHOLDS) +
                 f'; 5-yr starts {t["games_started"]["expected"]:.0f} (80%: {t["games_started"]["interval_80"][0]:.0f}-{t["games_started"]["interval_80"][1]:.0f}), '
                 f'pass yds {t["pass_yards"]["expected"]:.0f} ({t["pass_yards"]["interval_80"][0]:.0f}-{t["pass_yards"]["interval_80"][1]:.0f}), '
                 f'pass TD {t["pass_tds"]["expected"]:.0f}, rush yds {t["rush_yards"]["expected"]:.0f}')
    L.append('  Comps: ' + '; '.join(f'{c["player_name"]} ({c["draft_year"]}, d={c["distance"]:.2f})' for c in f['comps']))
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name')
    ap.add_argument('--draft-year', type=int)
    ap.add_argument('--cfbd-id')
    ap.add_argument('--final-season', type=int)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    key = {'name': a.name, 'draft_year': a.draft_year} if a.name else {'cfbd_player_id': a.cfbd_id, 'final_season': a.final_season}
    f = forecast_prospect(**key)
    print(json.dumps(f, indent=2, default=str) if a.json else summarize(f))
