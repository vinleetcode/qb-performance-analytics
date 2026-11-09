# Advanced Features Coverage Report (Phase 3)

Generated 2026-11-09 from a real pipeline run. Feature version `3.1.0`, target version `1.0.0`, survival version `1.0.0`. NFL observation window ends with the **2025** regular season.

> **Scope reminder.** This report describes data coverage and descriptive patterns. No final predictive model has been trained or tested, so nothing here is a claim about predictive accuracy. Coaching staff / coaching-change features are out of scope for this phase and were not built.

## 1. Prospects and draft classes

- **Total drafted QB prospects:** 328  (draft classes 2000-2026)
- **Split assignment (by draft class, chronological):** censored_holdout: 58, pre_coverage: 68, test: 34, train: 130, validation: 38
- `pre_coverage` = 2000-2004 classes whose college careers predate CFBD (2004+). They stay in outcome tables and EDA but cannot carry college features. `censored_holdout` = 2022-2026 classes whose five-season window is not complete.

| Draft class | QBs |
|---|---|
| 2000 | 12 |
| 2001 | 11 |
| 2002 | 15 |
| 2003 | 13 |
| 2004 | 17 |
| 2005 | 14 |
| 2006 | 13 |
| 2007 | 11 |
| 2008 | 13 |
| 2009 | 11 |
| 2010 | 13 |
| 2011 | 12 |
| 2012 | 11 |
| 2013 | 11 |
| 2014 | 14 |
| 2015 | 7 |
| 2016 | 15 |
| 2017 | 10 |
| 2018 | 13 |
| 2019 | 11 |
| 2020 | 13 |
| 2021 | 10 |
| 2022 | 9 |
| 2023 | 14 |
| 2024 | 11 |
| 2025 | 14 |
| 2026 | 10 |


## 2. Player identity matching

Phase 2 matched prospects to CFBD players by normalized name + school. Phase 3 adds CFBD's own NFL-draft record (`/draft/picks`), whose `collegeAthleteId` is a stable CFBD id keyed by (draft year, overall pick). It confirms name matches, fills matches the name matcher missed (e.g. nickname spellings) when the names are plausibly the same person, and turns any disagreement into `ambiguous` instead of guessing.

| Match method | Prospects |
|---|---|
| exact_name_school | 188 |
| unmatched | 136 |
| cfbd_draft_athlete_id | 3 |
| fuzzy_name_school | 1 |

- Match status: matched 192, unmatched 136
- Name/school matches **confirmed** by the CFBD draft id: 185; **conflicts** (left ambiguous): 0; CFBD draft id present but player has no CFBD season-stat rows: 103.
- Phase-2 map (CFBD season-stat players) within the modelling universe (2005+ classes): 192 of 260.

**Phase-3 identity extension.** CFBD's season-level player stats turned out to be nearly empty for 2004-2008 (of FBS passers with 100+ box-score attempts, they contain 0/132 in 2004, 1/136 in 2005, 16/145 in 2006, 53/145 in 2007, 93/141 in 2008, and nearly all from 2009). Phase 2 had attributed its unmatched 2005-2009 classes to careers predating CFBD, but that was the actual cause. Per-game box scores are complete in every season, so prospects not linked by the Phase-2 map are linked against box-score passers: first by CFBD's draft-pick athlete id (if it appears in box scores under a plausibly matching name), then by the same name+school matcher Phase 2 uses. Ties stay unresolved; nothing is guessed. Three narrow, verified school aliases were added (Boston Col./Boston College, Connecticut/UConn, Texas-El Paso/UTEP).

| Phase-3 identity source | Prospects |
|---|---|
| phase2_map:exact_name_school | 188 |
| unresolved | 76 |
| box_score_name_school:exact_name_school | 46 |
| cfbd_draft_athlete_id_box_scores | 14 |
| phase2_map:cfbd_draft_athlete_id | 3 |
| phase2_map:fuzzy_name_school | 1 |

- **Linked to college data in the modelling universe: 252 of 260.** Still unresolved there: mostly FCS / Division II QBs (e.g. Harvard, Coastal Carolina, San Diego, West Texas A&M, Fordham, North Dakota St.), whose games rarely appear in CFBD box scores, plus players with no college season in the window.

## 3. College statistics coverage

- Prospects with at least one pre-draft CFBD season: **251**; prospect-season rows: 867.
- Season totals by provenance: cfbd_season_stats 722, box_score_aggregate 145. Box-score seasons get rushing from rushing box scores for 2004-2008; outside those seasons a box-score-only season has NULL rushing, never 0.
- FCS gate: box scores cover only games involving an FBS team. If under 80% of a team's games are box-scored, box-derived season totals are dropped and per-game features (games, participation, opponent adjustment, top-defense splits) are set to NULL rather than computed from a one- or two-game sample.
- Careers possibly left-truncated (first observed season = 2004): 42; transfers (more than one school): 56.
- Box-score vs season-total attempts (seasons with 100+ att): median ratio 1.000 (10th-90th pct 1.00-1.00). Box scores exist only for games with an FBS team, so a few FCS-opponent games can be missing.
- Per-game passing box scores: 67,018 rows across 21,093 games, seasons 2004-2025 (1,195 team pseudo-rows kept for defense totals, never attributed to a QB).

| Draft class | % with college stats |
|---|---|
| 2000 | 0.0 |
| 2001 | 0.0 |
| 2002 | 0.0 |
| 2003 | 0.0 |
| 2004 | 0.0 |
| 2005 | 85.7 |
| 2006 | 84.6 |
| 2007 | 90.9 |
| 2008 | 92.3 |
| 2009 | 90.9 |
| 2010 | 92.3 |
| 2011 | 100.0 |
| 2012 | 100.0 |
| 2013 | 100.0 |
| 2014 | 100.0 |
| 2015 | 100.0 |
| 2016 | 100.0 |
| 2017 | 100.0 |
| 2018 | 100.0 |
| 2019 | 100.0 |
| 2020 | 100.0 |
| 2021 | 90.0 |
| 2022 | 100.0 |
| 2023 | 100.0 |
| 2024 | 100.0 |
| 2025 | 100.0 |
| 2026 | 100.0 |

## 4. Opponent-adjusted statistics

Method: for each QB game, the opponent's pass-defense baseline is its allowed completion %, YPA, TD % and INT % over its **other** games that season (leave-one-out). The baseline needs at least 4 other box-scored games, which FCS opponents rarely have. Adjusted rate = QB rate - (opponent allowed - FBS mean), weighted by attempts. Also computed: schedule strength vs pass defenses, conference-relative z-scores (pool = every CFBD passer with 100+ attempts in that conference-season), and splits against top-25 FBS pass defenses.

**Timing.** College seasons end (bowls) in early January; the draft is in late April. Full-season opponent data for completed seasons is therefore knowable at the pre-draft cutoff. It would *not* be valid for an in-season forecast, which this project does not make.

- Prospect-seasons with 100+ attempts: 679; with opponent-adjusted YPA: 678; mean share of attempts that had an opponent baseline: 0.959.
- Prospects with final-season opponent-adjusted YPA: 245.
- The SP+-based opponent defensive rating (CFBD's overall SP+ defense rating) exists in every season; a few opponents, mostly FCS, are unrated. SP+ *offensive* pace and run-rate sub-ratings exist only for 2005-2018.

| College season | % with box-score adjustment | % with SP+ opponent defense |
|---|---|---|
| 2004 | 100.0 | 100.0 |
| 2005 | 100.0 | 100.0 |
| 2006 | 100.0 | 100.0 |
| 2007 | 100.0 | 100.0 |
| 2008 | 100.0 | 100.0 |
| 2009 | 100.0 | 100.0 |
| 2010 | 100.0 | 100.0 |
| 2011 | 100.0 | 100.0 |
| 2012 | 100.0 | 100.0 |
| 2013 | 100.0 | 100.0 |
| 2014 | 100.0 | 100.0 |
| 2015 | 100.0 | 100.0 |
| 2016 | 100.0 | 100.0 |
| 2017 | 100.0 | 100.0 |
| 2018 | 100.0 | 100.0 |
| 2019 | 100.0 | 100.0 |
| 2020 | 97.1 | 97.1 |
| 2021 | 100.0 | 100.0 |
| 2022 | 100.0 | 97.5 |
| 2023 | 100.0 | 93.8 |
| 2024 | 100.0 | 100.0 |
| 2025 | 100.0 | 90.0 |

## 5. Offensive environment, scheme and supporting cast

Team-season stats rows: 2771. Unmapped CFBD statNames (reported, not guessed): firstDowns, firstDownsOpponent, fourthDownConversions, fourthDownConversionsOpponent, fourthDowns, fourthDownsOpponent, fumblesLost, fumblesLostOpponent, fumblesRecovered, fumblesRecoveredOpponent, interceptionTDs, interceptionTDsOpponent, interceptionYards, interceptionYardsOpponent, interceptionsOpponent, kickReturnTDs, kickReturnTDsOpponent, kickReturnYards, kickReturnYardsOpponent, kickReturns, kickReturnsOpponent, passesInterceptedOpponent, penalties, penaltiesOpponent, penaltyYards, penaltyYardsOpponent, possessionTimeOpponent, puntReturnTDs, puntReturnTDsOpponent, puntReturnYards, puntReturnYardsOpponent, puntReturns, puntReturnsOpponent, rushingTDsOpponent, thirdDownConversions, thirdDownConversionsOpponent, thirdDowns, thirdDownsOpponent, totalYardsOpponent, turnovers, turnoversOpponent.

| Season | Teams | Sacks allowed available | Advanced metrics reliable (PBP >= 85%) | Havoc available |
|---|---|---|---|---|
| 2004 | 119 | 0 | 64 | 63 |
| 2005 | 119 | 0 | 81 | 81 |
| 2006 | 119 | 0 | 86 | 86 |
| 2007 | 119 | 0 | 80 | 79 |
| 2008 | 120 | 0 | 107 | 105 |
| 2009 | 120 | 0 | 111 | 110 |
| 2010 | 120 | 0 | 111 | 111 |
| 2011 | 120 | 0 | 116 | 116 |
| 2012 | 124 | 0 | 119 | 118 |
| 2013 | 125 | 0 | 125 | 123 |
| 2014 | 128 | 0 | 126 | 126 |
| 2015 | 128 | 0 | 128 | 127 |
| 2016 | 128 | 128 | 125 | 125 |
| 2017 | 130 | 130 | 129 | 129 |
| 2018 | 130 | 130 | 130 | 130 |
| 2019 | 130 | 130 | 130 | 130 |
| 2020 | 128 | 127 | 127 | 127 |
| 2021 | 130 | 130 | 130 | 130 |
| 2022 | 131 | 131 | 131 | 131 |
| 2023 | 133 | 133 | 133 | 133 |
| 2024 | 134 | 134 | 134 | 134 |
| 2025 | 136 | 136 | 136 | 136 |

- Quantitative offensive profiles (pass rate, plays/game, tempo proxy, QB rush share, PPA/success where reliable, SP+ pace/run rate 2005-2018): 2771 team-seasons; prospects with final-season pass rate: 241; with SP+ pace: 153.
- **Qualitative scheme labels: 7 rows.** The repository had no existing documented labels to keep, and none were invented. Pass/run ratios are never turned into scheme names. `cfb_offensive_scheme_labels` and `config/offensive_scheme_labels.csv` are ready for labels that each carry a citation and a confidence level. Labels may overlap, one row per label.
- Receivers: 49,676 player-season receiving rows. Prospects with target-concentration features: 218; with top-3 target recruit rating: 191. Team receiving yards are stored but flagged **circular** (they equal the QB's own passing volume). The non-circular measures are concentration and the recruit ratings of the top targets, assigned before those targets caught a pass. A "drafted receiving teammates" feature was **rejected as leaky**: a teammate drafted in the same draft is not known before that draft.
- Offensive line: sack rate allowed and an opponent-adjusted version (opponents' full-season defensive sack rates, which include the game against this team). Also team yards per rush, plus stuff rate, line yards and havoc allowed where play-by-play coverage is reliable. **Sack rate is not an offensive-line-only measure**: QB hold time, decisions and scheme drive it too. NCAA statistics count sacks as rushes.
- Team talent: 247 Team Talent Composite seasons [2015]...[2025] (11 seasons); recruiting classes [2000]...[2025]; returning production seasons [2014]...[2025]. Individual recruits: 77,275 (48.1% linked by CFBD to a college athlete id). Prospects with talent composite: 125; with 4-class recruiting average: 251. Percentiles are within-season.

## 6. Athletic profile (NFL Combine)

- Combine rows (all positions, 2000-2026): 8,968. Drafted QBs with a combine record for their draft year: **293**. Join methods: pfr_id 279, cfb_slug 10, name_year_qb 4.
- Drill participation among QBs with a record (a skipped drill stays NULL and is never imputed as a poor result; `combine_did_*` indicators carry participation): forty 81.9%, bench 2.7%, vertical 70.0%, broad_jump 70.3%, cone 64.2%, shuttle 64.8%.
- Measurement date: the source has no per-player test date. Results are stored at season precision, treated as knowable by March 15 of the draft year.
- **Pro-day measurements: 0 rows.** No verified bulk source was found. Combine data is its own feature group (`athletic_combine`, scope `pre_draft_full`) because its missingness is systematic.

## 7. Injury and availability

- **Injury data: 0 rows (not_ingested).** No verified, legally bulk-accessible historical college injury source (see config/sources.json: college_injury_availability). No injury was inferred from missing statistics.
- What *is* available: participation (`games_with_pass_attempt / team_games_played`) from box scores. It is labelled as participation, **not** as injury, because a QB can miss games for many reasons (depth chart, redshirt, suspension, injury).

## 8. NFL outcome coverage

Starts come from nflverse schedules (listed starting QB per game, regular season). Appearances are weeks with any recorded stat. The five-season window is the draft-year season through draft year + 4. If that window is incomplete, window targets are NULL, never 0. A positive label already reached inside a partial window is recorded as True.

| Career status | Prospects |
|---|---|
| inferred_inactive | 166 |
| active_or_censored | 79 |
| never_appeared_window_complete | 60 |
| never_appeared_censored | 13 |
| not_yet_observed | 10 |

- Five-season window complete: 270; appeared in at least one NFL game: 245; nflverse roster status `RET` (the only "official" retirement signal available): 0. nflverse roster status does not record retirement for these players (Tom Brady, who retired in 2023, is listed `ACT`), so no career end here is officially confirmed. Inferred inactivity (no appearance in either of the last two observed seasons) is the career-end event, and it is always labelled *inferred*. QBs do sometimes return after a two-season gap.

## 9. Usable observations per prediction target

"Non-null" counts every class. "Model-ready" also requires matched college features and a train/validation/test class (2005-2021).

| Target | Non-null (all classes) | Train | Validation | Test | Positive rate |
|---|---|---|---|---|---|
| started_any_game | 303 | 122 | 38 | 33 | 0.71 |
| started_1plus_5yr | 303 | 122 | 38 | 33 | 0.693 |
| started_16plus_5yr | 282 | 122 | 38 | 33 | 0.383 |
| started_32plus_5yr | 275 | 122 | 38 | 33 | 0.265 |
| started_64plus_5yr | 270 | 122 | 38 | 33 | 0.093 |
| games_started_5yr | 270 | 122 | 38 | 33 |  |
| pass_yards_5yr | 270 | 122 | 38 | 33 |  |
| pass_tds_5yr | 270 | 122 | 38 | 33 |  |
| any_a_5yr | 120 | 56 | 18 | 15 |  |
| passer_rating_5yr | 120 | 56 | 18 | 15 |  |
| seasons_to_final_appearance | 166 | 78 | 16 | 13 |  |
| seasons_to_final_start | 138 | 66 | 11 | 11 |  |
| career_span_seasons | 166 | 78 | 16 | 13 |  |

Longevity targets (`seasons_to_final_*`) are filled only for careers inferred to have ended. Active careers are handled through the censored survival records below, so the last observed season is never treated as retirement.

## 10. Survival-analysis records

| Event | At risk | Events | Censored |
|---|---|---|---|
| career_end_official | 245 | 0 | 245 |
| final_appearance_inferred | 245 | 166 | 79 |
| final_start_inferred | 215 | 144 | 71 |
| first_nfl_start | 318 | 215 | 103 |

Time unit: NFL seasons since the draft (1 = rookie season). Definitions: `first_nfl_start` (all linkable drafted QBs); `final_appearance_inferred` / `final_start_inferred` (event = no appearance / no start in the last two observed seasons); `career_end_official` (nflverse status `RET`): every row is censored, because no prospect carries that status. It is kept so the distinction between inferred and confirmed endings stays explicit in the data. Drafted QBs who never appeared are kept, but are not at risk for "final" events.

## 11. Prospect similarity

- Prospects with historical comparables: 222. Each comp list uses only **earlier** draft classes, with standardization fit on that earlier pool. NFL outcomes are never inputs.
- Missing values: nan-Euclidean distance over features present for both prospects, rescaled, with at least 70% overlap required. No imputation, so a missing value never looks average.
- Scaling effect: share of total feature variance (the driver of unscaled Euclidean distance) per feature, before vs after standardization:

| Feature | Share unscaled | Share standardized | Missing frac |
|---|---|---|---|
| career_pass_attempts | 0.9987 | 0.0769 | 0.035 |
| dev_slope_rating | 0.0010 | 0.0770 | 0.169 |
| final_completion_pct | 0.0002 | 0.0769 | 0.035 |
| final_td_pct | 0.0000 | 0.0769 | 0.035 |
| final_opp_adj_ypa | 0.0000 | 0.0769 | 0.058 |
| final_yards_per_attempt | 0.0000 | 0.0769 | 0.035 |
| final_int_pct | 0.0000 | 0.0769 | 0.035 |
| n_college_seasons_observed | 0.0000 | 0.0769 | 0.035 |
| age_at_draft | 0.0000 | 0.0769 | 0.035 |
| final_conf_rel_rating_z | 0.0000 | 0.0769 | 0.088 |
| final_sos_pass_defense | 0.0000 | 0.0769 | 0.058 |
| talent_recruiting_4yr_pctile | 0.0000 | 0.0769 | 0.035 |
| final_qb_rush_share | 0.0000 | 0.0769 | 0.081 |

Unscaled, career pass attempts would dominate the distance almost entirely. That is why every similarity method here standardizes first.

## 12. Feature missingness

Average missing share by feature group in the modelling universe (2005+ classes):

| Feature group | % missing |
|---|---|
| age_experience | 4.2 |
| athletic_combine | 23.1 |
| college_production | 3.6 |
| college_roster_bio | 45.2 |
| development | 14.7 |
| draft_position | 0.0 |
| env_offensive_line | 27.1 |
| env_receivers | 25.2 |
| env_scheme | 16.0 |
| opponent_adjusted | 8.1 |
| scheme_quantitative | 57.5 |
| scheme_reported | 28.3 |
| scheme_verified | 51.0 |
| scouting | 6.2 |
| team_talent | 35.4 |

Key features, % missing by draft class:

| Class | final_attempts | final_opp_adj_ypa | final_conf_rel_rating_z | dev_slope_rating | age_at_draft | env_pass_rate | ol_sack_rate_allowed | wr_top3_avg_recruit_rating | talent_composite | talent_recruiting_points_4yr_avg | combine_forty | combine_height_inches | scout_pre_draft_grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 100 | 100 | 100 | 100 | 17 | 100 | 100 | 100 | 100 | 100 | 8 | 8 | 100 |
| 2001 | 100 | 100 | 100 | 100 | 18 | 100 | 100 | 100 | 100 | 100 | 9 | 9 | 100 |
| 2002 | 100 | 100 | 100 | 100 | 27 | 100 | 100 | 100 | 100 | 100 | 33 | 33 | 100 |
| 2003 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 | 100 | 100 | 31 | 31 | 100 |
| 2004 | 100 | 100 | 100 | 100 | 24 | 100 | 100 | 100 | 100 | 100 | 18 | 18 | 0 |
| 2005 | 14 | 14 | 21 | 100 | 29 | 14 | 100 | 100 | 100 | 14 | 14 | 14 | 7 |
| 2006 | 15 | 15 | 15 | 31 | 15 | 15 | 100 | 92 | 100 | 15 | 0 | 0 | 0 |
| 2007 | 9 | 9 | 9 | 9 | 0 | 9 | 100 | 45 | 100 | 9 | 9 | 9 | 0 |
| 2008 | 8 | 8 | 15 | 31 | 8 | 8 | 100 | 46 | 100 | 8 | 0 | 0 | 0 |
| 2009 | 9 | 9 | 18 | 18 | 0 | 9 | 100 | 18 | 100 | 9 | 18 | 18 | 0 |
| 2010 | 8 | 8 | 15 | 15 | 0 | 8 | 100 | 31 | 100 | 8 | 15 | 15 | 0 |
| 2011 | 0 | 0 | 0 | 8 | 0 | 0 | 100 | 42 | 100 | 0 | 0 | 0 | 0 |
| 2012 | 0 | 0 | 9 | 9 | 0 | 0 | 100 | 9 | 100 | 0 | 0 | 0 | 0 |
| 2013 | 0 | 0 | 9 | 9 | 0 | 9 | 100 | 9 | 100 | 0 | 9 | 9 | 9 |
| 2014 | 0 | 7 | 7 | 7 | 7 | 7 | 100 | 14 | 100 | 0 | 21 | 7 | 0 |
| 2015 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 100 | 0 | 14 | 14 | 0 |
| 2016 | 0 | 7 | 7 | 13 | 0 | 7 | 100 | 13 | 7 | 0 | 7 | 7 | 0 |
| 2017 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 10 | 0 | 0 | 30 | 10 | 0 |
| 2018 | 0 | 8 | 8 | 8 | 0 | 8 | 8 | 38 | 0 | 0 | 23 | 8 | 0 |
| 2019 | 0 | 9 | 9 | 18 | 0 | 9 | 9 | 9 | 0 | 0 | 18 | 0 | 0 |
| 2020 | 0 | 8 | 8 | 15 | 8 | 8 | 8 | 23 | 0 | 0 | 38 | 15 | 15 |
| 2021 | 10 | 10 | 20 | 10 | 0 | 10 | 10 | 10 | 10 | 10 | 40 | 10 | 0 |
| 2022 | 0 | 11 | 11 | 11 | 0 | 11 | 11 | 11 | 0 | 0 | 44 | 11 | 11 |
| 2023 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 64 | 7 | 7 |
| 2024 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 91 | 0 | 0 |
| 2025 | 0 | 0 | 7 | 0 | 0 | 14 | 14 | 14 | 14 | 0 | 93 | 14 | 7 |
| 2026 | 0 | 0 | 0 | 20 | 0 | 10 | 10 | 10 | 10 | 0 | 80 | 10 | 0 |

See also `reports/figures/missing_feature_rates_by_draft_class.png`.

## 13. Leakage audit

- Pre-draft feature values checked: 29,993. **Values knowable on or after draft day: 0.** Values dropped because no availability date could be resolved: 0.
- Post-draft values (draft round / pick): 656, all dated on draft day (656) and allowed only in the `post_draft` scope.
- Every feature has a registry row (`feature_definitions`) with an availability rule, source ids, the model scopes allowed to use it, and its caveats. `registry.assert_scope_safe` rejects unregistered, out-of-scope or outcome columns when model matrices are built.
- Heights/weights listed on NFL rosters (Phase-2 `qb_draft_prospects.height_inches`) are post-draft listings. They are **not** used as pre-draft features; combine measurements and college-roster listings are used instead.

## 14. Database row counts

| Table | Rows |
|---|---|
| cfb_game_passing | 67018 |
| cfb_game_rushing | 38356 |
| cfb_games | 42376 |
| cfb_offensive_scheme_labels | 608 |
| cfb_player_seasons | 40672 |
| cfb_player_seasons_derived | 40672 |
| cfb_players | 19115 |
| cfb_receiving_seasons | 49676 |
| cfb_recruits | 77275 |
| cfb_team_roster_talent | 4424 |
| cfb_team_season_context | 2794 |
| cfb_team_season_environment | 2771 |
| cfb_team_season_schemes | 770 |
| cfb_team_season_stats | 2771 |
| cfbd_draft_picks | 6901 |
| college_nfl_player_map | 328 |
| coverage_report | 175 |
| draft_value_predictions | 0 |
| era_adjusted_stats | 54900 |
| feature_definitions | 138 |
| leagues | 3 |
| nfl_career_outcomes | 328 |
| nfl_combine_measurements | 8968 |
| nfl_draft_picks_raw | 6901 |
| nfl_prospect_seasons | 3951 |
| player_season_team_stints | 2799 |
| player_seasons_derived | 2745 |
| player_seasons_raw | 2745 |
| players | 820 |
| prospect_college_seasons | 867 |
| prospect_feature_values | 30649 |
| prospect_similarity | 1110 |
| qb_draft_prospects | 328 |
| season_calendar | 116 |
| sources | 21 |
| survival_records | 1023 |
| teams | 36 |

Existing tables before vs after this run (any change here would mean existing data was modified):

| Table | Before | After |
|---|---|---|
| players | 820 | 820 |
| player_seasons_raw | 2745 | 2745 |
| player_seasons_derived | 2745 | 2745 |
| era_adjusted_stats | 54900 | 54900 |
| nfl_draft_picks_raw | 6901 | 6901 |
| qb_draft_prospects | 328 | 328 |
| cfb_players | 19115 | 19115 |
| cfb_player_seasons | 40672 | 40672 |
| cfb_team_season_context | 2794 | 2794 |

CFBD API calls made by this pipeline run: 0 (cache hits 665; per-run cap 750). The one-time cache warm-up for Phase 3 used 559 calls plus 77 for 2004-2008 rushing box scores and 8 coverage probes: about 644 of the free tier's 1,000/month (November 2026 quota, resets Dec 1).

## 15. Validation and test results

`pytest` run: **165 passed, 0 failed, 0 skipped** (of 165).

| Test module | Passed | Failed | Skipped |
|---|---|---|---|
| test_advanced_data_validation | 25 | 0 | 0 |
| test_advanced_features | 47 | 0 | 0 |
| test_advanced_pipeline_integration | 5 | 0 | 0 |
| test_cfbd_client | 7 | 0 | 0 |
| test_config | 6 | 0 | 0 |
| test_coverage | 3 | 0 | 0 |
| test_era_adjust | 3 | 0 | 0 |
| test_historical_pipeline_integration | 5 | 0 | 0 |
| test_ingest_cfbd_pivot | 6 | 0 | 0 |
| test_match_prospects | 14 | 0 | 0 |
| test_metrics | 7 | 0 | 0 |
| test_offensive_schemes | 19 | 0 | 0 |
| test_pipeline | 2 | 0 | 0 |
| test_prospect_pipeline | 8 | 0 | 0 |
| test_qualification | 4 | 0 | 0 |
| test_source_registry | 4 | 0 | 0 |

Phase-3 test coverage: player identity matching, missing historical features, invalid statistical calculations, incorrect draft assignments, college transfers, duplicate player-season observations, offensive-profile joins, missing combine measurements, five-year outcome windows, survival censoring, pre-draft feature availability, future-data leakage (scopes, splits, train-only preprocessing, earlier-class-only similarity) and database idempotency.

## 16. Exploratory analysis (descriptive only)

Figures in `reports/figures/`: `college_efficiency_vs_nfl_success.png`, `age_at_draft_vs_nfl_outcomes.png`, `college_development_trends.png`, `opponent_adjusted_performance.png`, `offensive_system_distributions.png`, `supporting_cast_quality.png`, `combine_distributions.png`, `draft_position_vs_nfl_production.png`, `missing_feature_rates_by_draft_class.png`, `nfl_success_by_draft_round.png`.

- Final-season NCAA rating vs NFL starts in 5 seasons: Spearman rho = 0.276. Overall pick vs starts: rho = -0.706 (negative = earlier picks start more). Draft slot partly *causes* opportunity, so this is not evidence of evaluation skill.
- Share reaching 16+ starts in 5 seasons by round: R1 91%, R2 56%, R3 36%, R4 16%, R5 3%, R6 11%, R7 7%.
- Same, by age at draft: < 22: 64%, 22-23: 38%, 23-24: 31%, 24+: 19%. Age is confounded with early-entry decisions, which themselves reflect perceived talent.
- Opponent-adjusted minus raw final-season YPA: median 0.096, 10th-90th percentile -0.383 to 0.491 (n=245). For most prospects the adjustment moves YPA by under half a yard.

These are raw historical associations on small samples, with selection effects. They are **not** validated predictive results and should not be read causally.

## 17. Incomplete features and reasons

| Feature / group | Status | Reason |
|---|---|---|
| Injury history (games missed to injury, documented injuries) | Not populated | No verified, legally bulk-accessible historical college injury source; CFBD has no injury endpoint; inferring injuries from missing stats is prohibited. |
| Pro-day measurements | Not populated | No verified bulk source; pro-day numbers are mostly school-reported or held by sites that prohibit bulk collection. |
| Qualitative scheme labels (Air Raid, Spread, ...) | Structure only (0 rows) | No documented labels existed in the repo; labels are never derived from pass/run ratios. Add cited rows to config/offensive_scheme_labels.csv. |
| Verified college starts / seasons as starter | Proxy only | CFBD exposes no starts field; `seasons_as_primary_passer` (led team in attempts in >=50% of games) is a labelled proxy. |
| College career totals for 2000-2004 classes (and 2004-start careers) | Missing / lower bound | CFBD player stats start in 2004; `career_possibly_truncated` flags affected careers. |
| Team talent composite before 2015 | Missing | 247 Team Talent Composite exists in CFBD from 2015 only; recruiting 4-class averages cover earlier seasons. |
| Returning production before 2014 | Missing | CFBD /player/returning coverage starts 2014. |
| SP+ offensive pace / run rate | 2005-2018 only | CFBD SP+ offensive sub-ratings are null outside 2005-2018. |
| College stats for FCS / D-II QBs | Mostly missing | CFBD box scores only include games involving an FBS team; partial FCS seasons are dropped (80% team-coverage gate), not reported from a 1-2 game sample. |
| QB rushing in box-score-only seasons after 2008 | NULL where unobserved | Rushing box scores were fetched for 2004-2008 only (where CFBD season stats are missing); e.g. Cam Newton 2010 has no CFBD season-stat row, so his rushing is NULL. |
| NFL appearances without a pass attempt | Undercounted | NFL season totals reuse the existing Phase-1 table, which kept only seasons with >= 1 pass attempt; starts are added back from schedules. |
| Advanced PBP metrics (line yards, stuff rate, havoc, PPA) in early seasons | Gated | Used only when play-by-play plays >= 85% of box-score plays; havoc fields are null in some seasons. |
| Sacks allowed / sack rate (OL) | 2016+ only | CFBD /stats/season exposes team sacks only from 2016 (section 5 table); earlier OL context relies on rushing efficiency and gated play-by-play metrics. |
| Per-game opponent-adjusted sack rate | Approximation | No per-game sack data ingested; opponent sack rates are full-season (they include the game vs the team). |
| "Official" NFL retirement date | Sparse | Only nflverse roster status RET; most career ends are inferred (2 absent seasons) and labelled as such. |
| Drafted receiving teammates | Rejected | Same-draft teammates are not known before the QB's draft (leakage). |
| Model predictions (draft_value_predictions) | Empty by design | This phase trains no final models; the table and draft_value_base.parquet are ready for them. |
