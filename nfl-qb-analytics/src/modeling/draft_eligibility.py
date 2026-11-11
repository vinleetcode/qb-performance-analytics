"""NFL Draft eligibility audit (Phase 5.5).

Run:  python -m src.modeling.draft_eligibility        (local data only; no API calls)

The Phase-5 draft-probability population ("verified prospect universe", 556 final seasons) is a
HEURISTIC, not verified draft eligibility. This module classifies EVERY FBS passer-season
(>= 100 attempts) by what the local data shows happened next, and keeps an explicit
`eligibility_unknown` class wherever the data cannot say:

  returned_same_school       appears again the next season at the same school (not a final season)
  transferred                next appearance is at a different school (FBS or the partial non-FBS coverage)
  returned_after_gap         no appearance the next season, but a later one (injury/redshirt/transfer)
  drafted_early_entrant      no later college record, drafted in the next draft, <= 3 years since the
                             high-school class (an underclassman who declared)
  drafted_next_draft         no later college record, drafted in the next draft, >= 4 years (or unknown)
  career_ended_4plus_years   no later college record, not drafted, >= 4 college years evidenced. This is
                             the Phase-5 "undrafted" label. It mixes exhausted eligibility, early
                             declarations (redshirt juniors), quitting, and transfers to levels CFBD does
                             not cover.
  eligibility_unknown_underclass_exit   no later record, <= 3 years since HS, not drafted: undrafted
                             early entrant, transfer down, injury or quitting. These can't be told apart.
  eligibility_unknown_no_class_evidence no later record, no recruiting link and < 4 observed seasons
  final_season_not_verifiable_yet       2025 seasons (no later season in the data yet)

What CANNOT be identified from local data (documented gap): draft declarations (the NFL's early-
entrant list), transfer-portal entries, remaining eligibility (redshirts, the 2020 COVID year,
medical hardships), class year (cfb_player_seasons.class_year is empty for every row), and
non-FBS seasons before 2022 (CFBD covers FCS teams mostly in games against FBS opponents until 2022).

The report also refits the Phase-5 college-only draft model under alternative population
definitions with the same chronological backtest, and compares the probabilities on the rows
that every definition shares. That measures how much the definition itself moves P(drafted).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ingest_cfbd_advanced import id_str
from src.modeling import backtesting as BT
from src.modeling import draft_predictor as DP
from src.modeling import uncertainty as U
from src.modeling.reporting import md_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase5_5'
REPORT = ROOT / 'reports' / 'draft_eligibility_audit.md'
SEED = DP.SEED
UNKNOWN = ('eligibility_unknown_underclass_exit', 'eligibility_unknown_no_class_evidence', 'final_season_not_verifiable_yet')

# credible external sources for the gap (NOT fetched: no external requests are authorised for this phase)
DECLARATION_SOURCES = [
    ('NFL Football Operations: annual list of early-entry (underclassmen) players granted special eligibility',
     'official list released each January; names, schools, positions'),
    ('NCAA transfer portal records (via CFBD /player/portal, available from 2021)',
     'transfer entries with origin, destination and eligibility year; would separate transfers from exits. '
     'A CFBD request, so it needs approval and counts against the monthly quota'),
    ('CFBD /roster (team rosters with class year, 2004+)',
     'class year and redshirt status per season; would replace the years-since-high-school heuristic. '
     'A CFBD request, so it needs approval'),
    ('School athletics roster pages and official conference releases',
     'class year and medical hardship / COVID-year status; manual, per player'),
]


def _ro():
    return sqlite3.connect(f'file:{DP.DB.as_posix()}?mode=ro', uri=True)


def load_presence_with_school() -> pd.DataFrame:
    """Every (athlete, season, school) the local CFBD data has, with the school's classification."""
    con = _ro()
    try:
        rows = pd.concat([
            pd.read_sql('select p.cfbd_player_id, s.season, s.school as team from cfb_player_seasons s '
                        'join cfb_players p on p.id = s.player_id', con),
            pd.read_sql('select distinct cfbd_player_id, season, team from cfb_game_passing where cfbd_player_id is not null', con),
            pd.read_sql('select distinct cfbd_player_id, season, team from cfb_game_rushing where cfbd_player_id is not null', con),
        ])
        g = pd.read_sql('select season, home_team, home_classification, away_team, away_classification from cfb_games', con)
    finally:
        con.close()
    rows['cfbd_player_id'] = rows['cfbd_player_id'].map(id_str)
    rows = rows.dropna(subset=['cfbd_player_id']).drop_duplicates()
    cls = pd.concat([g[['season', 'home_team', 'home_classification']].set_axis(['season', 'team', 'classification'], axis=1),
                     g[['season', 'away_team', 'away_classification']].set_axis(['season', 'team', 'classification'], axis=1)])
    cls = cls.dropna().drop_duplicates(['season', 'team'])
    return rows.merge(cls, on=['season', 'team'], how='left')


def classify(ref: pd.DataFrame, presence: pd.DataFrame, recruits: pd.DataFrame, drafted: pd.DataFrame,
             min_attempts: int = DP.MIN_FINAL_ATTEMPTS, last_verifiable: int | None = None) -> pd.DataFrame:
    """One row per FBS passer-season with >= min_attempts, labelled by what happened next.

    drafted: (cfbd_player_id, draft_year) for every player drafted at any position."""
    cand = ref[(ref['attempts'] >= min_attempts) & ref['is_fbs']][['cfbd_player_id', 'season', 'team_first', 'attempts']].copy()
    cand = cand.sort_values('attempts', ascending=False).drop_duplicates(['cfbd_player_id', 'season'])
    data_max = int(presence['season'].max())
    last_verifiable = data_max - 1 if last_verifiable is None else last_verifiable
    pres = presence.copy()
    by = {k: g.sort_values('season') for k, g in pres.groupby('cfbd_player_id')}
    rec = (recruits.sort_values('rating', ascending=False).drop_duplicates('athlete_id')
           .set_index('athlete_id')['class_year'])
    first_draft = drafted.groupby('cfbd_player_id')['draft_year'].min()
    out = []
    for r in cand.itertuples(index=False):
        g = by.get(r.cfbd_player_id, pd.DataFrame(columns=['season', 'team', 'classification']))
        upto = g[g['season'] <= r.season]
        later = g[g['season'] > r.season]
        nxt = later[later['season'] == r.season + 1]
        hs = rec.get(r.cfbd_player_id, np.nan)
        yrs = r.season - hs + 1 if pd.notna(hs) else np.nan
        n_seasons = int(upto['season'].nunique()) if len(upto) else 1
        dy = first_draft.get(r.cfbd_player_id, np.nan)
        drafted_next = bool(pd.notna(dy) and dy == r.season + 1)
        if len(nxt):
            same = r.team_first in set(nxt['team'])
            status = 'returned_same_school' if same else 'transferred'
            dest = None if same else nxt['team'].iloc[0]
            dest_cls = None if same else nxt['classification'].iloc[0]
        elif len(later):
            status, dest = 'returned_after_gap', later['team'].iloc[0]
            dest_cls = later['classification'].iloc[0]
        else:
            dest = dest_cls = None
            if r.season > last_verifiable:
                status = 'final_season_not_verifiable_yet'
            elif drafted_next:
                status = 'drafted_early_entrant' if (pd.notna(yrs) and yrs <= 3) else 'drafted_next_draft'
            elif pd.notna(dy) and dy > r.season + 1:
                status = 'drafted_later_draft'
            elif (pd.notna(yrs) and yrs >= DP.MIN_YEARS_SINCE_HS) or n_seasons >= DP.MIN_YEARS_SINCE_HS:
                status = 'career_ended_4plus_years'
            elif pd.notna(yrs):
                status = 'eligibility_unknown_underclass_exit'
            else:
                status = 'eligibility_unknown_no_class_evidence'
        out.append({'cfbd_player_id': r.cfbd_player_id, 'season': int(r.season), 'team': r.team_first, 'attempts': r.attempts,
                    'hs_class_year': hs, 'years_since_hs': yrs, 'n_seasons_observed': n_seasons, 'next_status': status,
                    'destination_team': dest, 'destination_classification': dest_cls,
                    'drafted_next_draft': drafted_next, 'first_draft_year': dy,
                    'draft_eligible_by_hs_rule': bool(pd.notna(yrs) and yrs >= 3) if pd.notna(yrs) else None})
    c = pd.DataFrame(out)
    c['final_season'] = ~c['next_status'].isin(['returned_same_school', 'transferred', 'returned_after_gap'])
    c['eligibility_class'] = np.select(
        [c['next_status'].isin(UNKNOWN), c['next_status'].isin(['returned_same_school', 'transferred', 'returned_after_gap']),
         c['next_status'].str.startswith('drafted'), c['next_status'] == 'career_ended_4plus_years'],
        ['eligibility_unknown', 'stayed_in_college', 'entered_and_drafted', 'eligible_career_ended_undrafted'],
        'eligibility_unknown')
    return c


def drafted_ids(src: dict) -> pd.DataFrame:
    ident = src['identity'][['cfbd_player_id', 'draft_year']]
    picks = src['picks'].rename(columns={'college_athlete_id': 'cfbd_player_id', 'year': 'draft_year'})[['cfbd_player_id', 'draft_year']]
    return pd.concat([ident, picks]).dropna().drop_duplicates()


# ------------------------------------------------------------------ population definitions and their effect
def population_definitions(u: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks over the Phase-5 universe (final seasons only)."""
    verifiable = u['population_status'] != 'final_season_not_verifiable_yet'
    known_later = ~u['drafted_other_year'].astype(bool)
    return {
        'phase5_current (>=4 yrs: recruit class or observed seasons)': u['in_population'].astype(bool),
        'recruit_link_only (>=4 yrs since HS class)': u['in_population'].astype(bool) & (u['eligibility_basis'] == 'recruit_class_year'),
        'likely_exhausted (>=5 yrs since HS or >=5 seasons)': u['in_population'].astype(bool) & (
            (u['years_since_hs'] >= 5) | (u['n_seasons_present'] >= 5)),
        'naive_all_final_seasons (missing = undrafted)': verifiable & known_later,
    }


def population_effect(u: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the college-only logistic draft model per population definition (same backtest); compare
    predictions on the rows that every definition shares."""
    from src.modeling.evaluate import fast_auc, fast_brier
    defs = population_definitions(u)
    preds, summ = {}, []
    for name, m in defs.items():
        pop = u[m].reset_index(drop=True)

        def fp(train, test):
            mdl = DP.pipe(LogisticRegression(C=0.5, max_iter=5000)).fit(train[DP.COLLEGE_FEATURES], train['drafted'])
            return pd.DataFrame({'key': (test['cfbd_player_id'] + '_' + test['season'].astype(str)).to_numpy(),
                                 'y': test['drafted'].to_numpy(), 'p': mdl.predict_proba(test[DP.COLLEGE_FEATURES])[:, 1]})
        b = BT.run(pop, 'draft_year', DP.PROB_TEST_YEARS, BT.LAG_DRAFT, fp, min_train=50)
        preds[name] = b
        summ.append({'definition': name, 'rows': int(len(pop)), 'drafted': int(pop['drafted'].sum()),
                     'draft_rate': float(pop['drafted'].mean()), 'backtest_rows': int(len(b)),
                     'auc_own_population': fast_auc(b['y'], b['p']) if b['y'].nunique() == 2 else np.nan,
                     'brier_own_population': fast_brier(b['y'], b['p'])})
    common = set.intersection(*[set(b['key']) for b in preds.values()])
    rows = []
    for name, b in preds.items():
        c = b[b['key'].isin(common)]
        rows.append({'definition': name, 'common_rows': len(c), 'observed_draft_rate': float(c['y'].mean()),
                     'mean_predicted': float(c['p'].mean()), 'calibration_in_the_large': float(c['y'].mean() - c['p'].mean()),
                     'auc_common': fast_auc(c['y'], c['p']) if c['y'].nunique() == 2 else np.nan,
                     'brier_common': fast_brier(c['y'], c['p'])})
    return pd.DataFrame(summ), pd.DataFrame(rows)


def drafted_entry_profile(c: pd.DataFrame, src: dict) -> pd.DataFrame:
    """Among drafted QBs whose final college season is classified, how many were early entrants, by round."""
    ident = src['identity'][['cfbd_player_id', 'draft_year', 'round']]
    f = c[c['next_status'].isin(['drafted_early_entrant', 'drafted_next_draft'])].merge(
        ident, left_on=['cfbd_player_id', 'first_draft_year'], right_on=['cfbd_player_id', 'draft_year'], how='inner')
    f['entry'] = np.where(f['next_status'] == 'drafted_early_entrant', 'early entrant (<= 3 yrs since HS)',
                          np.where(f['years_since_hs'].notna(), '>= 4 yrs since HS', 'class year unknown'))
    f['round_group'] = pd.cut(f['round'], [0, 1, 3, 7], labels=['Round 1', 'Rounds 2-3', 'Rounds 4-7'])
    return f.groupby(['round_group', 'entry'], observed=False).size().unstack(fill_value=0)


def fcs_exit_rate(c: pd.DataFrame) -> pd.DataFrame:
    """Share of 4+-year 'career ended' style exits that actually continued at a non-FBS school, measurable
    only where non-FBS coverage is broad (next season 2022+). Estimates how often the pre-2022
    'career_ended_4plus_years' label hides a transfer down."""
    down = (c['next_status'].isin(['transferred', 'returned_after_gap'])
            & (c['destination_classification'].fillna('') != 'fbs'))
    exits = c['final_season'] | down          # the season ended the FBS career, one way or another
    rows = []
    for lo, hi, lab in ((2005, 2020, 'next season 2006-2021 (sparse non-FBS coverage)'),
                        (2021, 2024, 'next season 2022-2025 (broad non-FBS coverage)')):
        m = c['season'].between(lo, hi) & exits & (c['next_status'] != 'final_season_not_verifiable_yet')
        rows.append({'period': lab, 'fbs_career_exits': int(m.sum()), 'continued_at_non_fbs_school': int((m & down).sum()),
                     'share': float((m & down).sum() / max(m.sum(), 1))})
    return pd.DataFrame(rows)


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    src = DP.load_sources()
    pres = load_presence_with_school()
    c = classify(src['ref'], pres, src['recruits'], drafted_ids(src))
    c.to_parquet(RESULTS / 'eligibility_classification.parquet', index=False)
    u = pd.read_parquet(DP.DATA_FILE)
    S, E = population_effect(u)
    S.to_csv(RESULTS / 'eligibility_population_definitions.csv', index=False)
    E.to_csv(RESULTS / 'eligibility_population_effect.csv', index=False)
    prof = drafted_entry_profile(c, src)
    fcs = fcs_exit_rate(c)
    counts = c.groupby(['next_status']).agg(rows=('season', 'size')).reset_index()
    by_period = (c[c['final_season']].assign(period=pd.cut(c['season'], [2003, 2010, 2016, 2021, 2025],
                                                          labels=['2004-2010', '2011-2016', '2017-2021', '2022-2025']))
                 .groupby(['period', 'eligibility_class'], observed=False).size().unstack(fill_value=0))
    # agreement with the Phase-5 universe on final seasons
    m = u.merge(c[['cfbd_player_id', 'season', 'next_status', 'eligibility_class']], on=['cfbd_player_id', 'season'], how='left')
    agree = m.groupby(['population_status', 'next_status'], dropna=False).size().reset_index(name='rows')
    summary = {'fbs_passer_seasons': int(len(c)), 'final_seasons': int(c['final_season'].sum()),
               'next_status_counts': counts.to_dict('records'),
               'eligibility_class_counts': c['eligibility_class'].value_counts().to_dict(),
               'phase5_universe_rows': int(len(u)), 'phase5_population': int(u['in_population'].sum())}
    (RESULTS / 'eligibility_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    REPORT.write_text(render(summary, c, S, E, prof, fcs, by_period, agree, u), encoding='utf-8')
    print(counts.to_string()); print(E.round(3).to_string())
    return summary


def _md(df: pd.DataFrame, floatfmt: str = '.3f') -> str:
    return md_table(df, floatfmt)


def interpret(S: pd.DataFrame, E: pd.DataFrame) -> list[str]:
    e = E.set_index('definition')
    cur, naive, exh = e.index[0], e.index[3], e.index[2]
    out = [f'* On the {int(e.loc[cur, "common_rows"])} shared test rows (observed draft rate {e.loc[cur, "observed_draft_rate"]:.1%}), '
           f'mean predicted P(drafted) ranges from {e["mean_predicted"].min():.1%} to {e["mean_predicted"].max():.1%} across '
           f'definitions. The current rule is the least calibrated here ({e.loc[cur, "mean_predicted"]:.1%} predicted): it '
           'includes redshirt-junior and fourth-year exits, a group with a higher draft rate than the fifth-year QBs who '
           'dominate the shared rows.',
           f'* The **naive** definition (missing future record = undrafted, eligible exit) has the most rows '
           f'({int(S.iloc[3]["rows"]):,}) and the highest own-population AUC ({S.iloc[3]["auc_own_population"]:.3f}). '
           'That AUC is flattering: it adds many short-career and unknown-eligibility QBs who are easy to rank as '
           f'undrafted. On shared rows its AUC is {e.loc[naive, "auc_common"]:.3f}, against {e.loc[cur, "auc_common"]:.3f} for the current rule. '
           'Its labels are also wrong for undrafted early declarants it cannot identify.',
           f'* The **likely-exhausted** definition is cleaner but has only {int(S.iloc[2]["rows"])} rows '
           f'({int(S.iloc[2]["drafted"])} drafted).']
    return out


def render(summary, c, S, E, prof, fcs, by_period, agree, u) -> str:
    n = c['next_status'].value_counts()
    fin = c[c['final_season']]
    L = ['# Draft eligibility audit (Phase 5.5)', '',
         'Generated by `python -m src.modeling.draft_eligibility` from local data only (no API calls, no scraping).', '',
         '## Summary', '',
         f'* The local data has **{len(c):,} FBS passer-seasons with at least {DP.MIN_FINAL_ATTEMPTS} attempts**. '
         f'{int((~c["final_season"]).sum()):,} were followed by another college season; '
         f'{len(fin):,} were the QB\'s last appearance in CFBD data.',
         '* **Draft eligibility is not observed anywhere in the local data.** There is no declaration list, no transfer-portal '
         'table and no class year (`cfb_player_seasons.class_year` is empty for all 40,672 rows). Eligibility is inferred '
         'from the high-school recruiting class and the number of observed seasons.',
         f'* **{int(c["eligibility_class"].eq("eligibility_unknown").sum()):,} passer-seasons stay `eligibility_unknown`.** '
         'They are excluded from the draft-probability population, and the forecast interface returns no P(drafted) for them.',
         '* The Phase-5 population (556 final seasons) is "FBS QBs whose college career verifiably ended after at least '
         'four years". Its draft probability is not P(drafted) for every college QB, and should not be extended to '
         'underclassmen or players still in college.', '',
         '## What happened after each FBS passer-season', '',
         '| Next status | Seasons | Meaning |', '|---|---:|---|']
    meaning = {
        'returned_same_school': 'played again for the same school the next season',
        'transferred': 'next appearance at a different school',
        'returned_after_gap': 'skipped a season, then appeared again',
        'drafted_early_entrant': 'last season, drafted next spring, <= 3 years since HS (declared early)',
        'drafted_next_draft': 'last season, drafted next spring, >= 4 years since HS (or class unknown)',
        'drafted_later_draft': 'last season, drafted in a later draft',
        'career_ended_4plus_years': 'last season, not drafted, >= 4 college years evidenced (Phase-5 "undrafted")',
        'eligibility_unknown_underclass_exit': 'last season, not drafted, <= 3 years since HS: undrafted declarant, transfer down, injury or quit',
        'eligibility_unknown_no_class_evidence': 'last season, not drafted, no recruit link and < 4 observed seasons',
        'final_season_not_verifiable_yet': '2025 season: no later season in the data yet',
    }
    for k, v in meaning.items():
        L.append(f'| `{k}` | {int(n.get(k, 0)):,} | {v} |')
    L += ['', 'Final seasons by era and eligibility class:', '', by_period.reset_index().pipe(_md), '',
          '## Questions the data can and cannot answer', '',
          '| Question | Identifiable from local data? | How / why not |', '|---|---|---|',
          f'| Draft-eligible underclassmen | Partly | 3+ years since the HS class marks eligibility, but only where the recruit link exists '
          f'({int(c["hs_class_year"].notna().sum()):,} of {len(c):,} passer-seasons). Redshirt status is unknown. |',
          '| Exhausted eligibility | No | No class year, redshirt, medical-hardship or COVID-year (2020) data. 5+ years since HS is a proxy only. |',
          f'| Transferred | Yes, between covered schools | {int(n.get("transferred", 0)):,} passer-seasons were followed by a different school. '
          'Transfers to non-FBS schools before 2022 are mostly invisible (see below). |',
          f'| Declared early | Only if drafted | {int(n.get("drafted_early_entrant", 0)):,} drafted early entrants. Undrafted early declarants '
          'can\'t be separated from players who quit or transferred down. |',
          f'| Returned to college | Yes | {int(n.get("returned_same_school", 0) + n.get("returned_after_gap", 0)):,} passer-seasons (same school, or after a gap). |',
          '| Never entered the draft | No | Not entering and going undrafted look identical without the declaration list. |', '',
          '### Hidden transfers to non-FBS schools', '',
          'CFBD covers non-FBS teams broadly only from 2022. Among FBS passer-seasons that ended an FBS career (a final '
          'season, or a move to a non-FBS school), the share that continued at a non-FBS school shows how many "career '
          'ended" labels before 2022 may really be transfers down:', '',
          _md(fcs), '',
          f'The difference between the two periods ({fcs["share"].iloc[1] - fcs["share"].iloc[0]:+.1%} of exits) is a rough estimate of '
          'transfers down that the older data cannot see. Those QBs sit in the "undrafted, career ended" group. They were '
          'not drafted next spring either way, so the label is right, but they were not really prospects leaving football.', '',
          '### Who the four-year rule leaves out', '',
          'Drafted QBs by round and how long they had been out of high school:', '',
          prof.reset_index().pipe(_md), '',
          'Early entrants are concentrated in round 1. A population restricted to QBs with at least four college years '
          'therefore leaves out many of the strongest prospects. The draft-probability model never sees them, and its '
          'P(drafted) should not be applied to a true junior.', '',
          '## How the definition changes the population and the probabilities', '',
          'The college-only logistic draft model was refit under four definitions, with the same chronological backtest '
          f'(test drafts {DP.PROB_TEST_YEARS[0]}-{DP.PROB_TEST_YEARS[-1]}, one-year label lag):', '',
          _md(S), '',
          'All four, scored on the rows they share (the strictest population\'s test rows):', '',
          _md(E), '',
          'Reading this:', '',
          *interpret(S, E),
          '* Because every definition is a heuristic, P(drafted) carries definition uncertainty on top of sampling '
          'uncertainty. The calibration-in-the-large column (observed minus mean predicted) is the size of that effect on the '
          'shared rows. Own-population AUCs are not comparable across definitions: a broader population contains more easy '
          'negatives.', '',
          '## Agreement with the Phase-5 universe', '',
          'Phase-5 `population_status` against this audit\'s `next_status` for the same final seasons:', '',
          _md(agree), '',
          '## Decisions', '',
          '* The Phase-5 population and model (`models/qb_draft/v1.0.0/`) are unchanged. They answer a narrow, defensible '
          'question: P(drafted | an FBS QB\'s college career verifiably ended after at least four years).',
          '* `eligibility_unknown` is kept as an explicit class, and the forecast interface returns '
          '`status: unavailable` for P(drafted) outside the verified population. It never extrapolates.',
          '* Round and pick projections are conditional on being drafted. They are available for anyone with a college '
          'profile, and are labelled "if drafted".', '',
          '## Data needed to close the gap (not fetched)', '',
          'No external requests were made. Credible sources, for a later approved phase:', '',
          '| Source | What it adds |', '|---|---|']
    L += [f'| {a} | {b} |' for a, b in DECLARATION_SOURCES]
    L += ['']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
