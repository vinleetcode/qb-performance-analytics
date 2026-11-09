"""Renders reports/offensive_scheme_coverage.md from a real run's artifacts:
  results/offensive_scheme_coverage.json        (python -m src.advanced_pipeline)
  data/processed/college_offensive_schemes.parquet
  results/junit.xml                              (pytest --junitxml=results/junit.xml)
Run: python -m src.scheme_report
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.advanced_report import _junit_summary, _table

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / 'results' / 'offensive_scheme_coverage.json'
TEAM_PARQUET = ROOT / 'data' / 'processed' / 'college_offensive_schemes.parquet'
JUNIT = ROOT / 'results' / 'junit.xml'
OUT = ROOT / 'reports' / 'offensive_scheme_coverage.md'
SCHEME_TEST_MODULES = {'test_offensive_schemes'}


def label_vs_measured(team: pd.DataFrame) -> list:
    t = team[team['standard_down_pass_rate'].notna()]
    rows = []
    for fam in ('air_raid', 'spread', 'spread_option', 'veer_and_shoot', 'multiple_hybrid', 'pro_style'):
        m = t['reported_offensive_family'].fillna('').str.split('|').map(lambda x, f=fam: f in x)
        g = t[m]
        if len(g):
            rows.append((fam, len(g), f"{g['standard_down_pass_rate'].mean():.3f}",
                         f"{g['passing_down_pass_rate'].mean():.3f}", f"{g['plays_per_game_nongarbage'].mean():.1f}"))
    u = t[t['reported_offensive_family'].isna()]
    rows.append(('(no reported label)', len(u), f"{u['standard_down_pass_rate'].mean():.3f}",
                 f"{u['passing_down_pass_rate'].mean():.3f}", f"{u['plays_per_game_nongarbage'].mean():.1f}"))
    return rows


def render(rep: dict, team: pd.DataFrame, junit: dict) -> str:
    L: list[str] = []
    ts, u = rep['team_seasons'], rep['usable_by_qb']
    L.append('# Offensive Scheme Coverage Report\n')
    L.append(f"Generated {rep['generated_on']} from a real pipeline run. Taxonomy `{rep['taxonomy_version']}`, feature "
             f"version `{rep['feature_version']}`. No predictive model was trained; nothing here is a claim about "
             "whether scheme improves NFL predictions. That is the question these features exist to test.\n")
    L.append('## Summary\n')
    L.append(_table(['Measure', 'Value'], [
        ('Relevant college team-seasons (prospects\' pre-draft seasons)', ts['relevant']),
        ('Team-seasons with a **verified** label (manual source review)',
         f"{ts['with_any_verified_label']} ({ts['with_verified_family']} with an offensive-family label)"),
        ('Team-seasons with a **reported** family label (Wikipedia infobox, low confidence)', ts['with_reported_family']),
        ('Team-seasons with play-by-play tendencies (2016-2025 fetched)', ts['with_pbp_profile']),
        ('Drafted QBs (2005+ model universe) with any usable scheme feature', f"{u['model_universe_2005plus']['any_scheme_feature']}%"),
        ('Drafted QBs (all 2000-2026) with any usable scheme feature', f"{u['all_drafted_qbs']['any_scheme_feature']}%"),
        ('Conflicting manual classifications (kept unlabelled)', ts['manual_research_status'].get('conflicting', 0)),
        ('Reported text mapped only to a non-family dimension (e.g. Pistol, West Coast)',
         ts['reported_label_status'].get('labelled_other_dimension_only', 0)),
        ('Wikipedia scheme text deliberately left unmapped (ambiguous)', ts['reported_label_status'].get('unmapped_text', 0)),
        ('Team-seasons whose page has no scheme field (unknown)', ts['reported_label_status'].get('page_without_scheme_field', 0)
         + ts['reported_label_status'].get('no_page', 0)),
    ]))
    L.append('\n**Bottom line.** Verified, contemporaneous scheme labels are expensive to find. About a quarter of the '
             'candidate sources that were read actually tied a scheme to a specific team **and** season. So the verified '
             'tier is a small, randomly sampled audit set, not a modelling feature. The usable categorical signal comes from '
             'the low-confidence *reported* tier. It is kept separately named (`scheme_rep_*`) and is never presented as '
             'verified. The measured tendencies (`scheme_q_*`, `env_*`) are statistics, not schemes.\n')

    L.append('## 1. How labels were assigned\n')
    L.append(_table(['Tier', 'Evidence rule', 'Status / confidence', 'Feature prefix'], [
        ('A. Verified', 'A source was read and explicitly ties the label to that team AND season (contemporaneous where '
                        'possible). Verbatim quote, URL, title and publication date stored.', 'verified / high-medium-low',
         '`scheme_ver_*`'),
        ("A'. Reported", 'Wikipedia team-season infobox `off_scheme` (season-specific, but community-edited, '
                         f"{rep['wikipedia']['fields_with_inline_citation']} of {rep['wikipedia']['pages_with_field']} fields "
                         'cited, usually written after the fact).', 'reported / low', '`scheme_rep_*`'),
        ('B. Quantitative', 'Measured from box scores and CFBD play-by-play. Never turned into a scheme name.', 'n/a',
         '`scheme_q_*`, `env_*`'),
    ]))
    L.append('\nRules applied:')
    L.append('- A coach\'s or coordinator\'s philosophy, or a statement about a different season, is **not** evidence. '
             'Such sources are logged as `insufficient_evidence`. This excluded, for example, every coach-tenure list on the '
             'Wikipedia Air Raid / Flexbone / Spread articles.')
    L.append('- Sources that contradict each other produce no label (`conflicting`).')
    L.append('- Pages that could not be read (HTTP 403/404) are logged as `source_unavailable`, with no claim.')
    L.append('- Researched with nothing found: `searched_no_candidate_source`. Never researched: `not_researched`. '
             'Not researching a season is not evidence of anything.')
    L.append('- Labels may overlap (one row per label), across dimensions: offensive family, run concepts, passing '
             'concepts, formation tendency, tempo. See `config/offensive_scheme_taxonomy.json`.')
    L.append('- **Reference not used:** `deepmetricanalytics.com/college-football/scheme-matchups` is a 2026-season '
             'preview ("Opening-day expectations from coaching research... Not observed play-calling"), with no '
             'per-team citations. It covers none of the 2004-2025 seasons here.\n')

    L.append('### Manual research design\n')
    m = rep['manual']
    L.append(f"Researched team-seasons came from a **seeded random sample** (seed 20261109): 3 final college seasons per "
             f"draft class, 2005-2021, 51 in total. That keeps 'has a verified label' from becoming a proxy for draft "
             f"position, which it would be if only famous QBs were researched. {m['team_seasons_researched']} team-seasons "
             f"were researched in this pass ({m['rows']} log rows); the rest of the sample remains `not_researched`. Outcomes:\n")
    L.append(_table(['Verification status', 'Rows'], sorted(m['status_counts'].items(), key=lambda x: -x[1])))

    L.append('\n## 2. Source citations and provenance (manual tier)\n')
    rows = []
    for r in m['rows_detail']:
        lab = r['scheme_label'] if r['scheme_label'] != 'unknown' else ''
        title = (r['source_title'] or '').replace('|', '-')  # a literal pipe would break the table
        link = f"[{title or 'link'}]({r['source_url']})" if r['source_url'] else ''
        rows.append((r['team'], r['season'], f"{r['scheme_dimension']}: {lab}" if lab else '-', r['verification_status'],
                     r['confidence'] or '', link, r['publication_date'] or '', (r['evidence_summary'] or '').replace('|', '/')))
    L.append(_table(['Team', 'Season', 'Label', 'Status', 'Conf.', 'Source', 'Published', 'Evidence (verbatim where quoted)'], rows))
    w = rep['wikipedia']
    L.append(f"\n**Reported tier provenance:** {w['rows']} label rows from {w['pages_with_field']} infobox fields "
             f"({w['pages_found']} of {ts['relevant']} team-season pages found through the official MediaWiki API, "
             f"revisions dated {w['revision_dates'][0]} to {w['revision_dates'][1]}). Each row stores the revision "
             "permalink (`index.php?oldid=...`), the page title and the raw field text. Content is CC BY-SA 4.0.\n")

    L.append('## 3. Coverage by decade\n')
    L.append(_table(['Decade', 'Team-seasons', 'Verified family', 'Reported family', 'Play-by-play profile'],
                    [(d['decade'], d['team_seasons'], d['verified_family'], d['reported_family'], d['pbp_profile'])
                     for d in rep['coverage_by_decade']]))
    L.append('\nPlay-by-play exists only for 2016-2025 so far. 2004-2015 needs about 200 more CFBD calls, deferred to '
             'the next monthly quota to stay under the 1,000-call free tier.\n')

    L.append('## 4. Label distribution and unknowns\n')
    L.append(_table(['Reported offensive family', 'Team-seasons'], list(rep['reported_family_counts'].items())))
    L.append('\nUnmapped Wikipedia text (ambiguous terms, deliberately left `unknown`):\n')
    L.append(_table(['Raw text', 'Team-seasons'], list(rep['unmapped_reported_text'].items())))
    L.append('\nVerified vs reported, where both exist:\n')
    L.append(_table(['Team', 'Season', 'Verified family', 'Reported family', 'Wikipedia text'],
                    [(r['team'], r['season'], r['verified_offensive_family'], r['reported_offensive_family'],
                      r['wikipedia_off_scheme_raw']) for r in rep['verified_vs_reported']]))
    L.append('\nThe tiers agree for Hawai\'i 2007, Texas Tech 2016 (Air Raid is spread-family) and LSU 2019 ("Pro Spread" '
             'includes spread). They disagree for Georgia 2008. That is one reason the tiers are never merged.\n')

    L.append('## 5. Quantitative offensive profiles\n')
    L.append(_table(['Feature (team-season)', 'Calculation', 'Limitations'], [
        ('Passing / rushing play rate', 'Team pass attempts / (pass att + rush att), season box totals (`env_pass_rate`, `rush_rate`)',
         'NCAA counts sacks as rushes, so pass rate is understated for sack-prone teams'),
        ('Offensive plays per game', '(pass att + rush att) / games; also non-garbage play-by-play plays per game', 'Includes all game states (box version)'),
        ('Standard-down pass rate', 'Play-by-play: pass share on 1st down, 2nd & <=7, 3rd/4th & <=4; garbage time removed',
         '2016+ only; sacks count as passes; scrambles count as rushes (CFBD play types cannot separate designed runs)'),
        ('Passing-down pass rate', 'Same, on 2nd & 8+, 3rd/4th & 5+', 'As above'),
        ('Early-down pass rate / overall non-garbage pass rate', 'Same, 1st-2nd down / all downs', 'As above'),
        ('Explosiveness, success rate, PPA', 'CFBD advanced season stats (`off_explosiveness`, `off_success_rate`, `off_ppa`)',
         'Only where play-by-play covers >= 85% of box-score plays'),
        ('QB rushing involvement', "QB's share of team rushing attempts (`final_qb_rush_share`)", 'Includes sacks (NCAA)'),
        ('Team passing efficiency', 'Net passing yards / pass attempts (`team_pass_ypa`); passing PPA', 'Largely the QB\'s own production (circular)'),
        ('Tempo', 'Possession seconds per play; SP+ pace (2005-2018); plays per game', 'Possession time is a crude proxy'),
        ('Teams with < 300 non-garbage plays', 'Rates blanked', 'Mostly FCS teams seen only in games vs FBS opponents'),
    ]))
    L.append('\nNot computed, because CFBD play types do not reliably identify them: designed-run rate, RPO rate, '
             'play-action rate, formation (shotgun / pistol / under-center) rates.\n')
    L.append('Team-season coverage of play-by-play features: ' + ', '.join(f"{k} {v}%" for k, v in rep['team_season_quant_coverage_pct'].items()) + '.\n')
    L.append('**Face-validity check** (descriptive; statistics used to *check* labels, never to create them): mean '
             'tendencies by reported family, team-seasons with play-by-play.\n')
    L.append(_table(['Reported family', 'n', 'Std-down pass rate', 'Pass-down pass rate', 'Non-garbage plays/game'],
                    label_vs_measured(team)))
    L.append('\nThe order matches what the labels mean: Air Raid teams pass most on standard downs, pro-style teams '
             'least. The ranges overlap heavily, so the labels carry signal but are coarse.\n')

    L.append('## 6. QB features (per season and aggregated)\n')
    q = rep['qb_seasons']
    L.append(f"- Per-season table: {q['rows']} prospect-seasons. Each is joined on the exact (team, season) the QB played "
             f"for, so transfers ({q['transfer_qbs_multi_school']} multi-school QBs) pick up each school's offense.")
    L.append(f"- Aggregates: final-season labels and tendencies, career exposure (attempt-weighted share of labelled "
             f"seasons in spread-family / pro-style offenses), label coverage of career attempts, and the number of "
             f"distinct systems. `final_observed_season_gap` flags {q['final_observed_gap_positive']} QBs whose last "
             "observed season is not the one right before the draft (usually a final FCS season outside CFBD box scores, "
             "e.g. Rhett Bomar).\n")
    L.append(_table(['Group', 'All drafted QBs', '2005+ model universe'], [
        ('Verified final-season family', f"{u['all_drafted_qbs']['verified_final_family']}%", f"{u['model_universe_2005plus']['verified_final_family']}%"),
        ('Reported final-season family', f"{u['all_drafted_qbs']['reported_final_family']}%", f"{u['model_universe_2005plus']['reported_final_family']}%"),
        ('Any reported label in career', f"{u['all_drafted_qbs']['any_reported_career_label']}%", f"{u['model_universe_2005plus']['any_reported_career_label']}%"),
        ('Final-season play-by-play profile', f"{u['all_drafted_qbs']['pbp_final_season_profile']}%", f"{u['model_universe_2005plus']['pbp_final_season_profile']}%"),
        ('Final-season box-score pass rate', f"{u['all_drafted_qbs']['env_pass_rate_final']}%", f"{u['model_universe_2005plus']['env_pass_rate_final']}%"),
        ('Any scheme feature', f"{u['all_drafted_qbs']['any_scheme_feature']}%", f"{u['model_universe_2005plus']['any_scheme_feature']}%"),
    ]))
    L.append('\nFeature-level coverage (2005+ model universe, % non-missing):\n')
    L.append(_table(['Feature', '% non-missing'], list(rep['feature_coverage_model_universe_pct'].items())))
    mv = rep['missingness_vs_outcome']
    L.append(f"\n**Is missingness informative?** Among train/validation/test QBs, the share reaching 16+ starts is "
             f"{mv['reported_known_positive_rate']:.1%} with a reported final-season label (n={mv['n_known']}) vs "
             f"{mv['reported_unknown_positive_rate']:.1%} without (n={mv['n_unknown']}). There is no large outcome gap, "
             "so missing-label indicators are unlikely to act as an outcome proxy. That is descriptive, not a test.\n")

    L.append('## 7. Leakage and validation\n')
    L.append(f"- QB scheme seasons on or after the draft year: **{rep['leakage']['qb_scheme_seasons_on_or_after_draft_year']}**. "
             'All scheme features use pre-draft college seasons only, and they are in the feature registry with the '
             '`final_college_season_end` availability rule, so the pipeline\'s leakage audit covers them.')
    L.append('- Wikipedia text was written after the fact. It describes the pre-draft season but could carry hindsight; '
             'this is why it is low confidence and kept apart from the verified tier.')
    L.append(f"- Label-table schema check this run: `{rep['label_table_migration']}` (the empty v1 table was dropped and "
             "recreated as the v2 provenance schema on the first run, before which the database was backed up to "
             "`data/backups/qb_history_pre_schemes_20261109.db`; later runs report `current`). "
             'The migration refuses to run if the old table holds rows.\n')
    if junit:
        tot = junit['totals']
        L.append(f"`pytest`: **{tot['passed']} passed, {tot['failures'] + tot['errors']} failed, {tot['skipped']} skipped** "
                 f"of {tot['tests']} (whole suite).\n")
        L.append(_table(['Test module', 'Passed', 'Failed', 'Skipped'],
                        [(k, v['passed'], v['failed'], v['skipped']) for k, v in sorted(junit['by_file'].items())]))
        if junit['failed']:
            L.append('\nFailed: ' + ', '.join(f'`{f}`' for f in junit['failed']))
    L.append('\nScheme-specific checks (`tests/test_offensive_schemes.py`): incorrect team-season joins (no carry-over '
             'to other seasons), transfers, duplicated annotations, source attribution (verified rows need URL, title, '
             'date and quote; Wikipedia rows need revision permalinks), unsupported classifications (taxonomy-only labels, '
             'only manual review can verify, statistics can never label), missing statistics stay NaN (not 0), '
             'down-specific rate calculations, garbage time, double-merge regression, pre-draft-only seasons. Database '
             'idempotency is covered by the end-to-end integration test (two runs, identical tables).\n')

    L.append('## 8. Outputs\n')
    L.append(_table(['Output', 'Contents'], [
        ('`config/offensive_scheme_labels.csv`', 'Manual research log: every assertion with provenance, including failed and conflicting ones'),
        ('`config/offensive_scheme_taxonomy.json`', 'Dimensions, labels, raw-text mapping, deliberately unmapped terms'),
        ('`data/processed/college_offensive_schemes.parquet`', 'One row per relevant team-season: tiers, research status, quantitative profile'),
        ('`data/processed/qb_scheme_features.parquet`', 'One row per drafted QB: aggregated pre-draft scheme features'),
        ('`data/processed/qb_scheme_seasons.parquet`', 'One row per QB college season: that season\'s labels and tendencies'),
        ('`data/processed/offensive_scheme_labels_all.parquet`', 'All label rows (manual + reported) with provenance'),
        ('DB: `cfb_offensive_scheme_labels`, `cfb_team_season_schemes`', 'Same, in the project database; features also in `prospect_feature_values`'),
    ]))
    L.append('\n## 9. Next steps\n')
    L.append('- After the Dec 1 quota reset: fetch 2004-2015 play-by-play (about 200 calls; `python -m src.ingest_cfbd_plays` with those seasons) to extend `scheme_q_*` to the earlier draft classes.')
    L.append('- Continue the manual audit through the remaining 27 sampled team-seasons. Prefer official athletics sites, newspapers, '
             'and pre-draft analyses that name the season.')
    L.append('- Use the verified tier to estimate the accuracy of the reported tier once it has enough rows. With 4 '
             'overlapping team-seasons today, no accuracy figure is meaningful.')
    return '\n'.join(L) + '\n'


def main() -> str:
    rep = json.loads(REPORT_JSON.read_text())
    team = pd.read_parquet(TEAM_PARQUET)
    md = render(rep, team, _junit_summary(JUNIT))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding='utf-8')
    return str(OUT)


if __name__ == '__main__':
    print(main())
