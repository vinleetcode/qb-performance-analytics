"""Build the one-row-per-drafted-QB modeling dataset and the separate target table.

Run:  python -m src.modeling.build_dataset

Reads only existing, already-ingested local data (makes NO network calls and
writes nothing to the database or to data/model_ready/):
  data/model_ready/features_pre_draft_full.parquet   pre-draft college/combine features
  data/model_ready/prospect_college_seasons.parquet  final college team-season
  data/model_ready/nfl_prospect_seasons.parquet      NFL (prospect x season) grid with starts
  data/model_ready/nfl_targets.parquet                Phase-3 outcome table (cross-check)
  data/qb_draft_prospects.parquet                     draft slot (reference model only)
  data/cfbd_plays/*.parquet                           CFBD play-by-play 2016-2025

Writes:
  data/modeling/qb_success_features_v{V}.parquet     predictors (no outcome columns)
  data/modeling/qb_success_targets_v{V}.parquet      targets + eligibility flags (no predictors)
  data/modeling/dataset_manifest.json                file hashes = dataset version
  reports/model_feature_dictionary.csv
  reports/model_training_readiness.md, results/model_training_readiness.json

TARGET. nfl_starts_first_5_seasons = regular-season games in which nflverse's
schedule lists the QB as his team's starting QB (home_qb_id/away_qb_id), summed
over NFL seasons draft_year .. draft_year+4. career_success = starts >= 32.
  * window complete (draft_year + 4 <= 2025): 1 or 0.
  * window incomplete: 1 only if 32 starts are ALREADY reached, else NA (censored).
  * QB with no linked NFL stat record: 0 only when the draft board's own career
    record confirms he never played (no games, no last season); otherwise NA.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.targets import OBS_END_SEASON
from src.modeling import features as F

ROOT = Path(__file__).resolve().parents[2]
MODEL_READY = ROOT / 'data' / 'model_ready'
OUT_DIR = ROOT / 'data' / 'modeling'
REPORTS = ROOT / 'reports'
RESULTS = ROOT / 'results'
DATASET_VERSION = F.MODEL_FEATURE_VERSION

SUCCESS_STARTS = 32
WINDOW = 5
PRIMARY_DRAFT_YEARS = (2017, 2021)   # final college season 2016-2020 (inside play-by-play coverage), window complete
SUPP_HORIZON = 3                     # supplementary task: >= 16 starts in the first 3 seasons
SUPP_STARTS = 16
SUPP_DRAFT_YEARS = (2017, 2023)
HISTORY_DRAFT_YEARS = (2005, 2016)   # pre-play-by-play classes: supplementary sample-size check only

FEATURES_FILE = OUT_DIR / f'qb_success_features_v{DATASET_VERSION}.parquet'
TARGETS_FILE = OUT_DIR / f'qb_success_targets_v{DATASET_VERSION}.parquet'


def window_starts(nfl_seasons: pd.DataFrame, horizon: int) -> pd.Series:
    w = nfl_seasons[nfl_seasons['season_index'] <= horizon]
    return w.groupby('qb_draft_prospect_id')['games_started'].sum()


def build_targets(prospects: pd.DataFrame, nfl_seasons: pd.DataFrame, outcomes: pd.DataFrame,
                  obs_end: int = OBS_END_SEASON) -> pd.DataFrame:
    """One row per drafted QB. `prospects`: qb_draft_prospect_id, draft_year, gsis_id.
    `outcomes`: Phase-3 nfl_targets (career_end_status tells verified never-played apart
    from unlinkable records)."""
    t = prospects[['qb_draft_prospect_id', 'draft_year', 'gsis_id']].copy()
    t['draft_year'] = t['draft_year'].astype(int)
    status = outcomes.set_index('qb_draft_prospect_id')['career_end_status']
    t['career_end_status'] = t['qb_draft_prospect_id'].map(status)
    in_grid = set(nfl_seasons['qb_draft_prospect_id'])
    never_verified = t['career_end_status'].isin(['never_appeared_window_complete', 'never_appeared_censored'])
    for horizon, thresh, prefix in ((WINDOW, SUCCESS_STARTS, '5'), (SUPP_HORIZON, SUPP_STARTS, '3')):
        starts = window_starts(nfl_seasons, horizon)
        complete = t['draft_year'] + horizon - 1 <= obs_end
        linked = t['qb_draft_prospect_id'].isin(in_grid)
        s = t['qb_draft_prospect_id'].map(starts).astype(float)
        s = s.where(linked, np.where(never_verified, 0.0, np.nan))
        observed = linked | never_verified
        col_s = f'nfl_starts_first_{horizon}_seasons'
        t[col_s] = s.where(observed & complete)              # only report a window total when it is final
        t[f'nfl_starts_first_{horizon}_seasons_to_date'] = s.where(observed)
        reached = observed & (s >= thresh)
        y = pd.Series(pd.NA, index=t.index, dtype='Int64')
        y[reached] = 1
        y[observed & complete & ~reached] = 0
        name = 'career_success' if prefix == '5' else f'success_{horizon}yr_{thresh}starts'
        t[name] = y
        t[f'window_{horizon}yr_complete'] = complete
        t[f'{name}_status'] = np.select(
            [t['draft_year'] > obs_end, ~observed, complete, reached],
            ['not_yet_observed', 'unverified_nfl_record', 'observed_complete_window', 'achieved_in_partial_window'],
            'censored_incomplete_window')
    t['target_source'] = np.where(t['qb_draft_prospect_id'].isin(in_grid), 'nflverse_schedule_starting_qb',
                                  np.where(never_verified, 'never_appeared_per_draft_board_record', 'unlinked'))
    return t


def assign_eligibility(feat: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    t = tgt.copy()
    m = feat.set_index('qb_draft_prospect_id')
    has_box = t['qb_draft_prospect_id'].map(m['final_completion_pct'].notna()).fillna(False).astype(bool)
    has_pt = t['qb_draft_prospect_id'].map(m[F.GROUP_C_PLAYTYPE].notna().all(axis=1)).fillna(False).astype(bool)
    dy = t['draft_year']
    t['has_college_box_features'] = has_box
    t['has_playtype_features'] = has_pt
    t['eligible_primary'] = dy.between(*PRIMARY_DRAFT_YEARS) & t['window_5yr_complete'] & t['career_success'].notna()
    # Same-QB comparison cohort: every model (A/B/C/advanced/reference) is fit and scored on exactly these rows.
    t['comparison_cohort'] = t['eligible_primary'] & has_box & has_pt
    s = f'success_{SUPP_HORIZON}yr_{SUPP_STARTS}starts'
    t['eligible_supp_3yr'] = dy.between(*SUPP_DRAFT_YEARS) & t['window_3yr_complete'] & t[s].notna() & has_box & has_pt
    t['eligible_history'] = dy.between(*HISTORY_DRAFT_YEARS) & t['window_5yr_complete'] & t['career_success'].notna() & has_box
    return t


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(model_ready: Path = MODEL_READY, prospects_path: Path = ROOT / 'data' / 'qb_draft_prospects.parquet',
          plays: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feats = pd.read_parquet(model_ready / 'features_pre_draft_full.parquet')
    pcs = pd.read_parquet(model_ready / 'prospect_college_seasons.parquet')
    nfl = pd.read_parquet(model_ready / 'nfl_prospect_seasons.parquet')
    outcomes = pd.read_parquet(model_ready / 'nfl_targets.parquet')
    prospects = pd.read_parquet(prospects_path)

    f = F.add_derived_features(feats, prospects)
    plays = F.load_plays() if plays is None else plays
    profile = F.team_play_type_profile(plays)
    fts = F.final_team_season(pcs)
    f = f.merge(fts, on='qb_draft_prospect_id', how='left')
    f = f.merge(profile.rename(columns={'team': 'final_team', 'season': 'final_team_season'}),
                on=['final_team', 'final_team_season'], how='left')
    leak = [c for c in f.columns if c.startswith(('started_', 'games_started', 'nfl_', 'career_success'))]
    assert not leak, f'outcome columns in predictor table: {leak}'

    p = prospects.rename(columns={'id': 'qb_draft_prospect_id'})
    tgt = build_targets(p, nfl, outcomes)
    # cross-check against the Phase-3 outcome table built by the advanced pipeline
    chk = tgt.merge(outcomes[['qb_draft_prospect_id', 'games_started_5yr', 'started_32plus_5yr']],
                    on='qb_draft_prospect_id', how='left')
    both = chk['nfl_starts_first_5_seasons'].notna() & chk['games_started_5yr'].notna()
    mismatch = int((chk.loc[both, 'nfl_starts_first_5_seasons'] != chk.loc[both, 'games_started_5yr']).sum())
    lab = chk['career_success'].notna() & chk['started_32plus_5yr'].notna()
    label_mismatch = int((chk.loc[lab, 'career_success'].astype(bool) != chk.loc[lab, 'started_32plus_5yr'].astype(bool)).sum())
    tgt = assign_eligibility(f, tgt)
    meta = {'starts_crosscheck_rows': int(both.sum()), 'starts_crosscheck_mismatches': mismatch,
            'label_crosscheck_rows': int(lab.sum()), 'label_crosscheck_mismatches': label_mismatch,
            'n_team_seasons_profiled': int(len(profile)), 'n_team_seasons_reliable': int(profile['pt_reliable'].sum()),
            'plays_rows': int(len(plays))}
    return f, tgt, meta


def readiness(f: pd.DataFrame, t: pd.DataFrame, meta: dict) -> dict:
    m = f.merge(t, on=['qb_draft_prospect_id', 'draft_year'], how='inner')
    el = m[m['eligible_primary']]
    coh = m[m['comparison_cohort']]
    by_class = (m[m['draft_year'].between(2005, 2026)].groupby('draft_year')
                .agg(drafted=('qb_draft_prospect_id', 'size'),
                     nfl_record_linked=('target_source', lambda s: (s == 'nflverse_schedule_starting_qb').sum()),
                     college_identity_linked=('identity_source', lambda s: s.notna().sum()),
                     playtype_features=('has_playtype_features', 'sum'),
                     window_5yr_complete=('window_5yr_complete', 'sum'),
                     success=('career_success', lambda s: int((s == 1).sum())),
                     failure=('career_success', lambda s: int((s == 0).sum())),
                     target_na=('career_success', lambda s: int(s.isna().sum())),
                     comparison_cohort=('comparison_cohort', 'sum'))
                .reset_index())
    cand = F.FEATURE_SETS['C_playtype']
    complete_cases = int(coh[cand].notna().all(axis=1).sum())
    excluded = el[~el['comparison_cohort']][['draft_year', 'player_name', 'college', 'career_success']].copy()
    excluded['reason'] = np.where(~el.loc[~el['comparison_cohort'], 'has_college_box_features'],
                                  'no college stats in CFBD (FCS)', 'no play-by-play profile (FCS team)')
    id_methods = el['identity_source'].fillna('none').value_counts().to_dict()
    near = el[(el['nfl_starts_first_5_seasons'] - SUCCESS_STARTS).abs() <= 3][
        ['draft_year', 'player_name', 'nfl_starts_first_5_seasons', 'career_success']]
    return {
        'dataset_version': DATASET_VERSION,
        'observation_end_season': OBS_END_SEASON,
        'total_drafted_qbs_all_years': int(len(m)),
        'total_drafted_qbs_2005_2026': int(m['draft_year'].between(2005, 2026).sum()),
        'drafted_qbs_primary_years': int(m['draft_year'].between(*PRIMARY_DRAFT_YEARS).sum()),
        'with_playtype_features_primary_years': int(m.loc[m['draft_year'].between(*PRIMARY_DRAFT_YEARS), 'has_playtype_features'].sum()),
        'with_playtype_features_all_years': int(m['has_playtype_features'].sum()),
        'nfl_record_linked_primary_years': int((m.loc[m['draft_year'].between(*PRIMARY_DRAFT_YEARS), 'target_source'] != 'unlinked').sum()),
        'college_identity_linked_primary_years': int(m.loc[m['draft_year'].between(*PRIMARY_DRAFT_YEARS), 'identity_source'].notna().sum()),
        'college_identity_methods_eligible': id_methods,
        'eligible_primary': int(len(el)), 'eligible_positive': int((el['career_success'] == 1).sum()),
        'eligible_negative': int((el['career_success'] == 0).sum()),
        'comparison_cohort': int(len(coh)), 'cohort_positive': int((coh['career_success'] == 1).sum()),
        'cohort_negative': int((coh['career_success'] == 0).sum()),
        'cohort_complete_cases_all_16_features': complete_cases,
        'n_candidate_features_selected': len(cand),
        'eligible_draft_classes': list(range(PRIMARY_DRAFT_YEARS[0], PRIMARY_DRAFT_YEARS[1] + 1)),
        'excluded_from_cohort': excluded.to_dict('records'),
        'by_class': by_class.to_dict('records'),
        'cohort_by_class': coh.groupby('draft_year')['career_success'].agg(
            n='size', pos=lambda s: int((s == 1).sum())).reset_index().to_dict('records'),
        'feature_missing_cohort': {c: float(coh[c].isna().mean()) for c in cand + F.DRAFT_REFERENCE},
        'feature_missing_eligible': {c: float(el[c].isna().mean()) for c in cand + F.DRAFT_REFERENCE},
        'supp_3yr_eligible': int(m['eligible_supp_3yr'].sum()),
        'supp_3yr_positive': int((m.loc[m['eligible_supp_3yr'], f'success_{SUPP_HORIZON}yr_{SUPP_STARTS}starts'] == 1).sum()),
        'history_eligible': int(m['eligible_history'].sum()),
        'history_positive': int((m.loc[m['eligible_history'], 'career_success'] == 1).sum()),
        'near_threshold': near.to_dict('records'),
        'censored_classes_2022_2026': {
            'n': int(m['draft_year'].between(2022, 2026).sum()),
            'already_successful': int(((m['draft_year'].between(2022, 2026)) & (m['career_success'] == 1)).sum()),
            'censored_na': int(((m['draft_year'].between(2022, 2026)) & m['career_success'].isna()).sum())},
        **meta,
    }


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    return f'{100 * v:.0f}%' if pct else str(v)


def render_readiness(r: dict) -> str:
    L = ['# Model training readiness: NFL QB career success', '',
         f'Generated by `python -m src.modeling.build_dataset` (dataset version {r["dataset_version"]}). '
         f'NFL observation ends with the {r["observation_end_season"]} regular season. No network calls; '
         'built only from data already in `data/`.', '',
         '## Target', '',
         '`career_success = int(nfl_starts_first_5_seasons >= 32)`. Starts are regular-season games in which the '
         "nflverse schedule names the QB as his team's starting quarterback (`home_qb_id` / `away_qb_id`), "
         'summed over NFL seasons `draft_year` through `draft_year + 4`. The schedule file lists a starting QB for '
         'every regular-season game from 2014 through 2025 (256 or 272 games a season; 271 in 2022, when one game was '
         'cancelled), so starts come from real game records. Games played and pass attempts are not used as a proxy.', '',
         '* A QB whose five-season window is incomplete gets a value only if he has already reached 32 starts. '
         'Otherwise the target is NA (censored), never 0.',
         '* A drafted QB with no NFL stat record counts as 0 starts only when the draft board\'s own career fields '
         'confirm he never played. Otherwise the target is NA.',
         f'* Cross-check against the Phase-3 outcome table (`nfl_targets`): {r["starts_crosscheck_mismatches"]} start-total '
         f'mismatches in {r["starts_crosscheck_rows"]} rows, and {r["label_crosscheck_mismatches"]} label mismatches in '
         f'{r["label_crosscheck_rows"]} rows.', '',
         '## Headline counts', '',
         '| Quantity | Count |', '|---|---|',
         f'| Drafted QBs in the database (all draft years) | {r["total_drafted_qbs_all_years"]} |',
         f'| Drafted QBs, 2005-2026 classes | {r["total_drafted_qbs_2005_2026"]} |',
         f'| Drafted QBs, 2017-2021 classes (final college season inside 2016-2025 play-by-play) | {r["drafted_qbs_primary_years"]} |',
         f'| ... with play-by-play play-type features | {r["with_playtype_features_primary_years"]} |',
         f'| ... with a linked NFL record or verified never-played | {r["nfl_record_linked_primary_years"]} |',
         f'| ... with a linked college record | {r["college_identity_linked_primary_years"]} |',
         f'| ... with a complete five-season window and known target (**eligible**) | {r["eligible_primary"]} |',
         f'| Eligible: success (32+ starts) / not | {r["eligible_positive"]} / {r["eligible_negative"]} |',
         f'| **Same-QB comparison cohort** (eligible, with box-score and play-type features) | **{r["comparison_cohort"]}** |',
         f'| Cohort: success / not | {r["cohort_positive"]} / {r["cohort_negative"]} |',
         f'| Cohort complete cases on all {r["n_candidate_features_selected"]} model-C features | {r["cohort_complete_cases_all_16_features"]} |',
         f'| Drafted QBs with play-type features, any class | {r["with_playtype_features_all_years"]} |', '',
         'NFL identity comes from the nflverse draft record itself (`gsis_id`), so every linked QB is the drafted '
         'player by construction. College identity methods in the eligible set: ' +
         ', '.join(f'`{k}` {v}' for k, v in r['college_identity_methods_eligible'].items()) + '.', '',
         '## Eligible draft classes', '',
         'The 2017-2021 classes are eligible: their final college season (2016-2020) is inside the play-by-play data, '
         'and their five-season NFL windows (through 2025 at the latest) are complete. 2022-2026 classes have '
         'incomplete windows. Of their ' + str(r['censored_classes_2022_2026']['n']) + ' QBs, ' +
         str(r['censored_classes_2022_2026']['already_successful']) + ' have already reached 32 starts. The other ' +
         str(r['censored_classes_2022_2026']['censored_na']) + ' are censored (NA), not failures. These classes are '
         'scored by `predict.py` but cannot be evaluated yet.', '',
         'Training support is FBS only. Every cohort QB played his final college season in FBS, because CFBD '
         'play-by-play for 2016-2020 does not cover FCS-only games. By the 2024-2025 seasons CFBD play-by-play does '
         'include FCS teams, so a few recent FCS QBs (North Dakota St., Montana St.) have play-type profiles. '
         '`predict.py` does not score them by default: their rates against FCS defenses are outside anything the '
         'model was trained on.', '',
         '"NFL linked" counts QBs with an NFL season grid row. The 2026 class has none because no NFL season has been '
         'played since that draft. 2020 shows 12 of 13 because Cole McDonald is a verified never-played (0 starts), not '
         'a linked record.', '',
         '| Draft class | Drafted | NFL linked | College linked | Play-type | 5-yr complete | Success | Not | NA | Cohort |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for b in r['by_class']:
        if b['draft_year'] >= 2014:
            L.append(f'| {b["draft_year"]} | {b["drafted"]} | {b["nfl_record_linked"]} | {b["college_identity_linked"]} | '
                     f'{b["playtype_features"]} | {b["window_5yr_complete"]} | {b["success"]} | {b["failure"]} | '
                     f'{b["target_na"]} | {b["comparison_cohort"]} |')
    L += ['', 'Eligible QBs left out of the comparison cohort (all are non-successes, so every success in the eligible set stays in the cohort):', '',
          '| Class | QB | College | Reason |', '|---|---|---|---|']
    for e in r['excluded_from_cohort']:
        L.append(f'| {e["draft_year"]} | {e["player_name"]} | {e["college"]} | {e["reason"]} |')
    L += ['', '## Feature missingness (selected features)', '',
          '| Feature | Missing, eligible (n=' + str(r['eligible_primary']) + ') | Missing, cohort (n=' + str(r['comparison_cohort']) + ') |',
          '|---|---|---|']
    for c, v in r['feature_missing_eligible'].items():
        L.append(f'| `{c}` | {_fmt(v, True)} | {_fmt(r["feature_missing_cohort"][c], True)} |')
    L += ['', 'Each feature\'s source, calculation, availability cutoff and missingness are in '
          '`reports/model_feature_dictionary.csv`. The dictionary also lists the candidate features that were not '
          'selected, with the reason. Not available from any ingested source: verified college starts (CFBD does not publish them), '
          'injury/availability history and pro-day measurements.', '',
          '## Statistical feasibility', '',
          f'* Sample: {r["comparison_cohort"]} QBs, {r["cohort_positive"]} positive and {r["cohort_negative"]} negative '
          f'(about {r["cohort_positive"] / max(r["comparison_cohort"], 1):.0%} positive), across 5 draft classes.',
          f'* Candidate features: {r["n_candidate_features_selected"]} in the largest model (6 basic, 5 context, '
          f'5 play-type). That is about {r["cohort_positive"] / r["n_candidate_features_selected"]:.1f} positive events per feature, far below '
          'the usual 10-20 for unpenalised logistic regression. Only strongly regularised models are defensible. '
          'Gradient boosting is run only as a labelled exploratory comparison, and XGBoost is not justified at this size.',
          '* Chronological holdout: rolling-origin by draft class. Classes 2017-2018 are the first training window, '
          'then 2019, 2020 and 2021 are each predicted from the classes before them. Cohort QBs per class:', '']
    for c in r['cohort_by_class']:
        L.append(f'  * {c["draft_year"]}: {c["n"]} QBs, {c["pos"]} positive, {c["n"] - c["pos"]} negative')
    L += ['', 'Every class, including each of the three holdout classes, contains both outcome classes. Pooled, the out-of-sample set is the 2019-2021 '
          'cohort QBs. With roughly 30 holdout QBs, a ROC-AUC confidence interval is about +/-0.15 wide, so only large '
          'differences between models are detectable.',
          f'* Supplementary tasks to reduce small-sample risk: (1) a shorter horizon, 16+ starts in the first 3 seasons, '
          f'for classes 2017-2023 ({r["supp_3yr_eligible"]} QBs, {r["supp_3yr_positive"]} positive); (2) adding the '
          f'2005-2016 classes ({r["history_eligible"]} QBs, {r["history_positive"]} positive, no play-type data) to '
          'training for the A/B feature sets, to show how much sample size alone matters.', '',
          '## QBs near the 32-start threshold', '',
          'Labels within 3 starts of the cut-off depend on single games. They are listed so the threshold\'s effect is '
          'visible:', '', '| Class | QB | Starts, seasons 1-5 | Success |', '|---|---|---|---|']
    for n in r['near_threshold']:
        L.append(f'| {n["draft_year"]} | {n["player_name"]} | {int(n["nfl_starts_first_5_seasons"])} | {int(n["career_success"])} |')
    L += ['', f'Play-by-play profile: {r["n_team_seasons_reliable"]} of {r["n_team_seasons_profiled"]} team-seasons '
          f'have at least {F.MIN_PBP_PLAYS} non-garbage plays ({r["plays_rows"]:,} play rows, 2016-2025). Team-seasons below '
          'that (FCS teams seen only against FBS opponents) are left blank.', '']
    return '\n'.join(L)


def main() -> dict:
    f, t, meta = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f.to_parquet(FEATURES_FILE, index=False)
    t.drop(columns=['gsis_id']).to_parquet(TARGETS_FILE, index=False)
    manifest = {'dataset_version': DATASET_VERSION, 'observation_end_season': OBS_END_SEASON,
                'files': {p.name: sha256(p) for p in (FEATURES_FILE, TARGETS_FILE)}}
    (OUT_DIR / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    r = readiness(f, t, meta)
    REPORTS.mkdir(exist_ok=True)
    el = t.set_index('qb_draft_prospect_id')['eligible_primary']
    F.feature_dictionary(f, f['qb_draft_prospect_id'].map(el).fillna(False).astype(bool).values).to_csv(
        REPORTS / 'model_feature_dictionary.csv', index=False)
    (REPORTS / 'model_training_readiness.md').write_text(render_readiness(r), encoding='utf-8')
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'model_training_readiness.json').write_text(json.dumps(r, indent=2, default=str), encoding='utf-8')
    print(f'eligible {r["eligible_primary"]} ({r["eligible_positive"]}+/{r["eligible_negative"]}-) | cohort '
          f'{r["comparison_cohort"]} ({r["cohort_positive"]}+/{r["cohort_negative"]}-) | starts mismatches '
          f'{r["starts_crosscheck_mismatches"]} | label mismatches {r["label_crosscheck_mismatches"]}')
    return r


if __name__ == '__main__':
    main()
