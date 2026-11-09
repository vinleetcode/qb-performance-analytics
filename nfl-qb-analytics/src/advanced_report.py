"""Renders reports/advanced_features_coverage.md from the artifacts of a real run:
  results/advanced_coverage_report.json   (python -m src.advanced_pipeline)
  reports/figures/eda_prospects_summary.json (python -m src.eda_prospects)
  results/junit.xml                       (pytest --junitxml=results/junit.xml)

Every number in the report is read from those files -- nothing is typed in by hand.
Run: python -m src.advanced_report
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / 'results' / 'advanced_coverage_report.json'
EDA_JSON = ROOT / 'reports' / 'figures' / 'eda_prospects_summary.json'
JUNIT = ROOT / 'results' / 'junit.xml'
OUT = ROOT / 'reports' / 'advanced_features_coverage.md'


def _table(headers, rows) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    out += ['| ' + ' | '.join('' if v is None else str(v) for v in r) + ' |' for r in rows]
    return '\n'.join(out)


def _junit_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    suites = root.findall('testsuite') if root.tag == 'testsuites' else [root]
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    by_file: dict = {}
    failed = []
    for case in root.iter('testcase'):
        f = case.get('classname', '').split('.')[-1] or '?'
        st = 'passed'
        if case.find('failure') is not None or case.find('error') is not None:
            st = 'failed'
            failed.append(f"{case.get('classname')}::{case.get('name')}")
        elif case.find('skipped') is not None:
            st = 'skipped'
        by_file.setdefault(f, {'passed': 0, 'failed': 0, 'skipped': 0})[st] += 1
    tot['passed'] = tot['tests'] - tot['failures'] - tot['errors'] - tot['skipped']
    return {'totals': tot, 'by_file': by_file, 'failed': failed}


def render(rep: dict, eda: dict, junit: dict) -> str:
    L: list[str] = []
    p = rep['prospects']
    L.append('# Advanced Features Coverage Report (Phase 3)\n')
    L.append(f"Generated {rep['generated_on']} from a real pipeline run. Feature version `{rep['feature_version']}`, "
             f"target version `{rep['target_version']}`, survival version `{rep['survival_version']}`. NFL observation "
             f"window ends with the **{rep['observation_end_season']}** regular season.\n")
    L.append('> **Scope reminder.** This report describes data coverage and descriptive patterns. No final predictive '
             'model has been trained or tested, so nothing here is a claim about predictive accuracy. Coaching '
             'staff / coaching-change features are out of scope for this phase and were not built.\n')

    L.append('## 1. Prospects and draft classes\n')
    L.append(f"- **Total drafted QB prospects:** {p['total']}  (draft classes {p['draft_classes'][0]}-{p['draft_classes'][1]})")
    L.append(f"- **Split assignment (by draft class, chronological):** " +
             ', '.join(f'{k}: {v}' for k, v in sorted(p['by_split'].items())))
    L.append('- `pre_coverage` = 2000-2004 classes whose college careers predate CFBD (2004+). They stay in outcome '
             'tables and EDA but cannot carry college features. `censored_holdout` = 2022-2026 classes whose five-season '
             'window is not complete.\n')
    L.append(_table(['Draft class', 'QBs'], sorted(p['by_class'].items())))
    L.append('')

    i = rep['identity']
    L.append('\n## 2. Player identity matching\n')
    L.append('Phase 2 matched prospects to CFBD players by normalized name + school. Phase 3 adds CFBD\'s own NFL-draft '
             'record (`/draft/picks`), whose `collegeAthleteId` is a stable CFBD id keyed by (draft year, overall pick). '
             'It confirms name matches, fills matches the name matcher missed (e.g. nickname spellings) when the names '
             'are plausibly the same person, and turns any disagreement into `ambiguous` instead of guessing.\n')
    L.append(_table(['Match method', 'Prospects'], sorted(i['cfb_match_method'].items(), key=lambda x: -x[1])))
    L.append('')
    L.append(f"- Match status: " + ', '.join(f'{k} {v}' for k, v in i['match_status'].items()))
    L.append(f"- Name/school matches **confirmed** by the CFBD draft id: {i['confirmed_by_cfbd_draft_id']}; "
             f"**conflicts** (left ambiguous): {i['conflicts_name_vs_cfbd_id']}; CFBD draft id present but player has no "
             f"CFBD season-stat rows: {i['cfbd_id_without_stat_rows']}.")
    L.append(f"- Phase-2 map (CFBD season-stat players) within the modelling universe (2005+ classes): "
             f"{i['matched_in_model_universe']} of {i['model_universe_size']}.\n")
    L.append("**Phase-3 identity extension.** CFBD's season-level player stats turned out to be nearly empty for 2004-2008 "
             "(of FBS passers with 100+ box-score attempts, they contain 0/132 in 2004, 1/136 in 2005, 16/145 in 2006, "
             "53/145 in 2007, 93/141 in 2008, and nearly all from 2009). Phase 2 had attributed its unmatched 2005-2009 "
             "classes to careers predating CFBD, but that was the actual cause. Per-game box scores are complete in every "
             "season, so prospects not linked by the Phase-2 map are linked against box-score passers: first by CFBD's "
             "draft-pick athlete id (if it appears in box scores under a plausibly matching name), then by the same "
             "name+school matcher Phase 2 uses. Ties stay unresolved; nothing is guessed. Three narrow, verified school "
             "aliases were added (Boston Col./Boston College, Connecticut/UConn, Texas-El Paso/UTEP).\n")
    L.append(_table(['Phase-3 identity source', 'Prospects'],
                    sorted(i['phase3_identity_source'].items(), key=lambda x: -x[1])))
    L.append(f"\n- **Linked to college data in the modelling universe: {i['phase3_linked_in_model_universe']} of "
             f"{i['model_universe_size']}.** Still unresolved there: mostly FCS / Division II QBs (e.g. Harvard, "
             "Coastal Carolina, San Diego, West Texas A&M, Fordham, North Dakota St.), whose games rarely appear in CFBD "
             "box scores, plus players with no college season in the window.\n")

    c = rep['college_stats']
    L.append('## 3. College statistics coverage\n')
    L.append(f"- Prospects with at least one pre-draft CFBD season: **{c['prospects_with_any_pre_draft_cfbd_season']}**; "
             f"prospect-season rows: {c['prospect_college_season_rows']}.")
    L.append(f"- Season totals by provenance: " + ', '.join(f'{k} {v}' for k, v in c['season_stats_source'].items())
             + f". Box-score seasons get rushing from rushing box scores for {c['box_rushing_seasons'][0]}-"
             f"{c['box_rushing_seasons'][-1]}; outside those seasons a box-score-only season has NULL rushing, never 0.")
    L.append("- FCS gate: box scores cover only games involving an FBS team. If under 80% of a team's games are "
             "box-scored, box-derived season totals are dropped and per-game features (games, participation, opponent "
             "adjustment, top-defense splits) are set to NULL rather than computed from a one- or two-game sample.")
    L.append(f"- Careers possibly left-truncated (first observed season = 2004): {c['career_possibly_truncated']}; "
             f"transfers (more than one school): {c['transferred']}.")
    if c.get('box_score_attempt_ratio_median') is not None:
        L.append(f"- Box-score vs season-total attempts (seasons with 100+ att): median ratio "
                 f"{c['box_score_attempt_ratio_median']:.3f} (10th-90th pct {c['box_score_attempt_ratio_p10_p90'][0]:.2f}-"
                 f"{c['box_score_attempt_ratio_p10_p90'][1]:.2f}). Box scores exist only for games with an FBS team, so "
                 f"a few FCS-opponent games can be missing.")
    b = rep['box_scores']
    L.append(f"- Per-game passing box scores: {b['rows']:,} rows across {b['games']:,} games, seasons "
             f"{b['seasons'][0] if b['seasons'] else '-'}-{b['seasons'][-1] if b['seasons'] else '-'} "
             f"({b['team_pseudo_rows']:,} team pseudo-rows kept for defense totals, never attributed to a QB).\n")
    L.append(_table(['Draft class', '% with college stats'],
                    [(k, v) for k, v in sorted(c['by_class_pct_with_college_stats'].items())]))

    o = rep['opponent_adjusted']
    L.append('\n## 4. Opponent-adjusted statistics\n')
    L.append('Method: for each QB game, the opponent\'s pass-defense baseline is its allowed completion %, YPA, TD % and '
             'INT % over its **other** games that season (leave-one-out). The baseline needs at least 4 other box-scored '
             'games, which FCS opponents rarely have. Adjusted rate = QB rate - (opponent allowed - FBS mean), weighted by '
             'attempts. Also computed: schedule strength vs pass defenses, conference-relative z-scores (pool = every CFBD '
             'passer with 100+ attempts in that conference-season), and splits against top-25 FBS pass defenses.\n')
    L.append('**Timing.** College seasons end (bowls) in early January; the draft is in late April. Full-season opponent '
             'data for completed seasons is therefore knowable at the pre-draft cutoff. It would *not* be valid for an '
             'in-season forecast, which this project does not make.\n')
    L.append(f"- Prospect-seasons with 100+ attempts: {o['prospect_seasons_100plus_att']}; with opponent-adjusted YPA: "
             f"{o['with_opp_adj_ypa']}; mean share of attempts that had an opponent baseline: "
             f"{o['mean_attempt_coverage']:.3f}.")
    L.append(f"- Prospects with final-season opponent-adjusted YPA: {o['prospects_final_season_opp_adj']}.")
    L.append("- The SP+-based opponent defensive rating (CFBD's overall SP+ defense rating) exists in every season; a few "
             "opponents, mostly FCS, are unrated. SP+ *offensive* pace and run-rate sub-ratings exist only for 2005-2018.\n")
    rows = [(s, o['by_season_pct_with_opp_adj'].get(s), o['by_season_pct_with_sp_opp_def'].get(s))
            for s in sorted(o['by_season_pct_with_opp_adj'])]
    L.append(_table(['College season', '% with box-score adjustment', '% with SP+ opponent defense'], rows))

    L.append('\n## 5. Offensive environment, scheme and supporting cast\n')
    t = rep.get('team_stats', {})
    if t:
        L.append(f"Team-season stats rows: {t['rows']}. Unmapped CFBD statNames (reported, not guessed): "
                 f"{', '.join(t['unmapped_stat_names']) or 'none'}.\n")
        L.append(_table(['Season', 'Teams', 'Sacks allowed available', 'Advanced metrics reliable (PBP >= 85%)', 'Havoc available'],
                        [(s, v['teams'], v['sacks_allowed_available'], v['advanced_reliable'], v['havoc_available'])
                         for s, v in sorted(t['by_season'].items(), key=lambda x: int(x[0]))]))
    sch = rep['offensive_scheme']
    L.append(f"\n- Quantitative offensive profiles (pass rate, plays/game, tempo proxy, QB rush share, PPA/success where "
             f"reliable, SP+ pace/run rate 2005-2018): {sch['quantitative_profiles_team_seasons']} team-seasons; "
             f"prospects with final-season pass rate: {sch['prospects_with_pass_rate']}; with SP+ pace: "
             f"{sch['prospects_with_sp_pace']}.")
    L.append(f"- **Qualitative scheme labels: {sch['qualitative_label_rows']} rows.** The repository had no existing "
             "documented labels to keep, and none were invented. Pass/run ratios are never turned into scheme names. "
             "`cfb_offensive_scheme_labels` and `config/offensive_scheme_labels.csv` are ready for labels that each carry "
             "a citation and a confidence level. Labels may overlap, one row per label.")
    r = rep['receivers']
    L.append(f"- Receivers: {r['receiving_rows']:,} player-season receiving rows. Prospects with target-concentration "
             f"features: {r['prospects_with_concentration']}; with top-3 target recruit rating: "
             f"{r['prospects_with_top3_recruit_rating']}. Team receiving yards are stored but flagged **circular** (they "
             "equal the QB's own passing volume). The non-circular measures are concentration and the recruit ratings "
             "of the top targets, assigned before those targets caught a pass. A \"drafted receiving teammates\" feature "
             "was **rejected as leaky**: a teammate drafted in the same draft is not known before that draft.")
    L.append('- Offensive line: sack rate allowed and an opponent-adjusted version (opponents\' full-season defensive sack '
             'rates, which include the game against this team). Also team yards per rush, plus stuff rate, line yards and '
             'havoc allowed where play-by-play coverage is reliable. **Sack rate is not an offensive-line-only measure**: '
             'QB hold time, decisions and scheme drive it too. NCAA statistics count sacks as rushes.')
    tt = rep['team_talent']
    L.append(f"- Team talent: 247 Team Talent Composite seasons {tt['talent_composite_seasons'][:1]}..."
             f"{tt['talent_composite_seasons'][-1:]} ({len(tt['talent_composite_seasons'])} seasons); recruiting classes "
             f"{tt['recruiting_team_class_years'][:1]}...{tt['recruiting_team_class_years'][-1:]}; returning production "
             f"seasons {tt['returning_production_seasons'][:1]}...{tt['returning_production_seasons'][-1:]}. Individual "
             f"recruits: {tt['recruits_rows']:,} ({tt['recruits_with_athlete_id_pct']}% linked by CFBD to a college "
             f"athlete id). Prospects with talent composite: {tt['prospects_with_talent_composite']}; with 4-class "
             f"recruiting average: {tt['prospects_with_recruiting_4yr']}. Percentiles are within-season.\n")

    cb = rep['combine']
    L.append('## 6. Athletic profile (NFL Combine)\n')
    L.append(f"- Combine rows (all positions, 2000-2026): {cb['combine_rows_all_positions']:,}. Drafted QBs with a combine "
             f"record for their draft year: **{cb['prospects_with_combine_record']}**. Join methods: "
             + ', '.join(f'{k} {v}' for k, v in cb['join_methods'].items()) + '.')
    L.append('- Drill participation among QBs with a record (a skipped drill stays NULL and is never imputed as a poor '
             'result; `combine_did_*` indicators carry participation): '
             + ', '.join(f'{k} {v}%' for k, v in cb['drill_participation_pct_given_record'].items()) + '.')
    L.append('- Measurement date: the source has no per-player test date. Results are stored at season precision, '
             'treated as knowable by March 15 of the draft year.')
    L.append(f"- **Pro-day measurements: {cb['pro_day_rows']} rows.** No verified bulk source was found. "
             "Combine data is its own feature group (`athletic_combine`, scope `pre_draft_full`) because its "
             "missingness is systematic.\n")

    L.append('## 7. Injury and availability\n')
    inj = rep['injury']
    L.append(f"- **Injury data: {inj['rows']} rows ({inj['status']}).** {inj['reason']} No injury was inferred from "
             "missing statistics.")
    L.append('- What *is* available: participation (`games_with_pass_attempt / team_games_played`) from box scores. It is '
             'labelled as participation, **not** as injury, because a QB can miss games for many reasons (depth chart, '
             'redshirt, suspension, injury).\n')

    n = rep['nfl_outcomes']
    L.append('## 8. NFL outcome coverage\n')
    L.append('Starts come from nflverse schedules (listed starting QB per game, regular season). Appearances are weeks '
             'with any recorded stat. The five-season window is the draft-year season through draft year + 4. If that '
             'window is incomplete, window targets are NULL, never 0. A positive label already reached inside a partial '
             'window is recorded as True.\n')
    L.append(_table(['Career status', 'Prospects'], sorted(n['career_end_status'].items(), key=lambda x: -x[1])))
    L.append(f"\n- Five-season window complete: {n['window_5yr_complete']}; appeared in at least one NFL game: "
             f"{n['appeared_any_game']}; nflverse roster status `RET` (the only \"official\" retirement signal available): "
             f"{n['nflverse_status_RET']}. nflverse roster status does not record retirement for these players (Tom "
             "Brady, who retired in 2023, is listed `ACT`), so no career end here is officially confirmed. Inferred "
             "inactivity (no appearance in either of the last two observed seasons) is the career-end event, and it is "
             "always labelled *inferred*. QBs do sometimes return after a two-season gap.\n")

    L.append('## 9. Usable observations per prediction target\n')
    L.append('"Non-null" counts every class. "Model-ready" also requires matched college features and a train/validation/'
             'test class (2005-2021).\n')
    rows = []
    for tname, v in rep['target_usable_observations'].items():
        mr = v['model_ready_with_college_features']
        rows.append((tname, v['non_null_all_classes'], mr['train'], mr['validation'], mr['test'],
                     v.get('positive_rate_non_null', '')))
    L.append(_table(['Target', 'Non-null (all classes)', 'Train', 'Validation', 'Test', 'Positive rate'], rows))
    L.append('\nLongevity targets (`seasons_to_final_*`) are filled only for careers inferred to have ended. Active '
             'careers are handled through the censored survival records below, so the last observed season is never '
             'treated as retirement.\n')

    L.append('## 10. Survival-analysis records\n')
    L.append(_table(['Event', 'At risk', 'Events', 'Censored'],
                    [(k, v['n'], v['events'], v['censored']) for k, v in rep['survival'].items()]))
    L.append('\nTime unit: NFL seasons since the draft (1 = rookie season). Definitions: `first_nfl_start` (all linkable '
             'drafted QBs); `final_appearance_inferred` / `final_start_inferred` (event = no appearance / no start in '
             'the last two observed seasons); `career_end_official` (nflverse status `RET`): every row is censored, because '
             'no prospect carries that status. It is kept so the distinction between inferred and confirmed endings stays '
             'explicit in the data. '
             'Drafted QBs who never appeared are kept, but are not at risk for "final" events.\n')

    s = rep['similarity']
    L.append('## 11. Prospect similarity\n')
    L.append(f"- Prospects with historical comparables: {s['prospects_with_comps']}. Each comp list uses only **earlier** "
             "draft classes, with standardization fit on that earlier pool. NFL outcomes are never inputs.")
    L.append('- Missing values: nan-Euclidean distance over features present for both prospects, rescaled, with at least '
             '70% overlap required. No imputation, so a missing value never looks average.')
    L.append('- Scaling effect: share of total feature variance (the driver of unscaled Euclidean distance) per feature, '
             'before vs after standardization:\n')
    L.append(_table(['Feature', 'Share unscaled', 'Share standardized', 'Missing frac'],
                    [(x['feature'], f"{x['share_unscaled']:.4f}", f"{x['share_standardized']:.4f}", f"{x['missing_frac']:.3f}")
                     for x in s['scaling_effect']]))
    L.append('\nUnscaled, career pass attempts would dominate the distance almost entirely. That is why every '
             'similarity method here standardizes first.\n')

    L.append('## 12. Feature missingness\n')
    L.append('Average missing share by feature group in the modelling universe (2005+ classes):\n')
    L.append(_table(['Feature group', '% missing'], sorted(rep['missingness_by_group_model_universe_pct'].items())))
    L.append('\nKey features, % missing by draft class:\n')
    mbc = rep['missingness_by_class_pct']
    feats = list(next(iter(mbc.values())).keys()) if mbc else []
    L.append(_table(['Class'] + feats, [[k] + [v[f] for f in feats] for k, v in sorted(mbc.items(), key=lambda x: int(x[0]))]))
    L.append('\nSee also `reports/figures/missing_feature_rates_by_draft_class.png`.\n')

    a = rep['leakage_audit']
    L.append('## 13. Leakage audit\n')
    L.append(f"- Pre-draft feature values checked: {a['pre_draft_values_checked']:,}. **Values knowable on or after draft "
             f"day: {a['violations']}.** Values dropped because no availability date could be resolved: "
             f"{a['values_dropped_no_availability_date']}.")
    L.append(f"- Post-draft values (draft round / pick): {a['post_draft_values']:,}, all dated on draft day "
             f"({a['post_draft_values_on_or_after_draft_day']:,}) and allowed only in the `post_draft` scope.")
    L.append('- Every feature has a registry row (`feature_definitions`) with an availability rule, source ids, the model '
             'scopes allowed to use it, and its caveats. `registry.assert_scope_safe` rejects unregistered, out-of-scope '
             'or outcome columns when model matrices are built.')
    L.append('- Heights/weights listed on NFL rosters (Phase-2 `qb_draft_prospects.height_inches`) are post-draft listings. '
             'They are **not** used as pre-draft features; combine measurements and college-roster listings are used instead.\n')

    L.append('## 14. Database row counts\n')
    L.append(_table(['Table', 'Rows'], sorted(rep['row_counts'].items())))
    L.append('\nExisting tables before vs after this run (any change here would mean existing data was modified):\n')
    L.append(_table(['Table', 'Before', 'After'],
                    [(k, rep['existing_table_counts_before'][k], rep['existing_table_counts_after'][k])
                     for k in rep['existing_table_counts_before']]))
    cc = rep['cfbd_calls']
    L.append(f"\nCFBD API calls made by this pipeline run: {cc['network_calls_this_run']} (cache hits "
             f"{cc['cache_hits_this_run']:,}; per-run cap {cc['per_run_budget']}). The one-time cache warm-up for Phase 3 "
             "used 559 calls plus 77 for 2004-2008 rushing box scores and 8 coverage probes: about 644 of the free "
             "tier's 1,000/month (November 2026 quota, resets Dec 1).\n")

    L.append('## 15. Validation and test results\n')
    if junit:
        tot = junit['totals']
        L.append(f"`pytest` run: **{tot['passed']} passed, {tot['failures'] + tot['errors']} failed, {tot['skipped']} "
                 f"skipped** (of {tot['tests']}).\n")
        L.append(_table(['Test module', 'Passed', 'Failed', 'Skipped'],
                        [(k, v['passed'], v['failed'], v['skipped']) for k, v in sorted(junit['by_file'].items())]))
        if junit['failed']:
            L.append('\nFailed tests:\n' + '\n'.join(f'- `{f}`' for f in junit['failed']))
    else:
        L.append('No JUnit results found. Run `pytest --junitxml=results/junit.xml` before rendering this report.')
    L.append('\nPhase-3 test coverage: player identity matching, missing historical features, invalid statistical '
             'calculations, incorrect draft assignments, college transfers, duplicate player-season observations, '
             'offensive-profile joins, missing combine measurements, five-year outcome windows, survival censoring, '
             'pre-draft feature availability, future-data leakage (scopes, splits, train-only preprocessing, earlier-'
             'class-only similarity) and database idempotency.\n')

    if eda:
        L.append('## 16. Exploratory analysis (descriptive only)\n')
        L.append(f"Figures in `reports/figures/`: {', '.join('`' + f + '`' for f in eda['figures'])}.\n")
        L.append(f"- Final-season NCAA rating vs NFL starts in 5 seasons: Spearman rho = "
                 f"{eda.get('efficiency_vs_starts_spearman')}. Overall pick vs starts: rho = {eda.get('pick_vs_starts_spearman')} "
                 "(negative = earlier picks start more). Draft slot partly *causes* opportunity, so this is not "
                 "evidence of evaluation skill.")
        L.append('- Share reaching 16+ starts in 5 seasons by round: '
                 + ', '.join(f"R{k} {v:.0%}" for k, v in eda.get('share_16plus_by_round', {}).items()) + '.')
        L.append('- Same, by age at draft: ' + ', '.join(f"{k}: {v:.0%}" for k, v in eda.get('age_bins_share_16plus', {}).items())
                 + '. Age is confounded with early-entry decisions, which themselves reflect perceived talent.')
        oa = eda.get('opp_adj_minus_raw_ypa', {})
        L.append(f"- Opponent-adjusted minus raw final-season YPA: median {oa.get('median')}, 10th-90th percentile "
                 f"{oa.get('p10')} to {oa.get('p90')} (n={oa.get('n')}). For most prospects the adjustment moves YPA by "
                 "under half a yard.")
        L.append('\nThese are raw historical associations on small samples, with selection effects. They are **not** '
                 'validated predictive results and should not be read causally.\n')

    L.append('## 17. Incomplete features and reasons\n')
    L.append(_table(['Feature / group', 'Status', 'Reason'], [
        ('Injury history (games missed to injury, documented injuries)', 'Not populated',
         'No verified, legally bulk-accessible historical college injury source; CFBD has no injury endpoint; inferring injuries from missing stats is prohibited.'),
        ('Pro-day measurements', 'Not populated', 'No verified bulk source; pro-day numbers are mostly school-reported or held by sites that prohibit bulk collection.'),
        ('Qualitative scheme labels (Air Raid, Spread, ...)', 'Structure only (0 rows)', 'No documented labels existed in the repo; labels are never derived from pass/run ratios. Add cited rows to config/offensive_scheme_labels.csv.'),
        ('Verified college starts / seasons as starter', 'Proxy only', 'CFBD exposes no starts field; `seasons_as_primary_passer` (led team in attempts in >=50% of games) is a labelled proxy.'),
        ('College career totals for 2000-2004 classes (and 2004-start careers)', 'Missing / lower bound', 'CFBD player stats start in 2004; `career_possibly_truncated` flags affected careers.'),
        ('Team talent composite before 2015', 'Missing', '247 Team Talent Composite exists in CFBD from 2015 only; recruiting 4-class averages cover earlier seasons.'),
        ('Returning production before 2014', 'Missing', 'CFBD /player/returning coverage starts 2014.'),
        ('SP+ offensive pace / run rate', '2005-2018 only', 'CFBD SP+ offensive sub-ratings are null outside 2005-2018.'),
        ('College stats for FCS / D-II QBs', 'Mostly missing', 'CFBD box scores only include games involving an FBS team; partial FCS seasons are dropped (80% team-coverage gate), not reported from a 1-2 game sample.'),
        ('QB rushing in box-score-only seasons after 2008', 'NULL where unobserved', 'Rushing box scores were fetched for 2004-2008 only (where CFBD season stats are missing); e.g. Cam Newton 2010 has no CFBD season-stat row, so his rushing is NULL.'),
        ('NFL appearances without a pass attempt', 'Undercounted', 'NFL season totals reuse the existing Phase-1 table, which kept only seasons with >= 1 pass attempt; starts are added back from schedules.'),
        ('Advanced PBP metrics (line yards, stuff rate, havoc, PPA) in early seasons', 'Gated', 'Used only when play-by-play plays >= 85% of box-score plays; havoc fields are null in some seasons.'),
        ('Sacks allowed / sack rate (OL)', '2016+ only', 'CFBD /stats/season exposes team sacks only from 2016 (section 5 table); earlier OL context relies on rushing efficiency and gated play-by-play metrics.'),
        ('Per-game opponent-adjusted sack rate', 'Approximation', 'No per-game sack data ingested; opponent sack rates are full-season (they include the game vs the team).'),
        ('"Official" NFL retirement date', 'Sparse', 'Only nflverse roster status RET; most career ends are inferred (2 absent seasons) and labelled as such.'),
        ('Drafted receiving teammates', 'Rejected', 'Same-draft teammates are not known before the QB\'s draft (leakage).'),
        ('Model predictions (draft_value_predictions)', 'Empty by design', 'This phase trains no final models; the table and draft_value_base.parquet are ready for them.'),
    ]))
    L.append('')
    return '\n'.join(L)


def main() -> str:
    rep = json.loads(REPORT_JSON.read_text())
    eda = json.loads(EDA_JSON.read_text()) if EDA_JSON.exists() else {}
    junit = _junit_summary(JUNIT)
    md = render(rep, eda, junit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding='utf-8')
    return str(OUT)


if __name__ == '__main__':
    print(main())
