# Offensive Scheme Coverage Report

Generated 2026-11-09 from a real pipeline run. Taxonomy `1.0.0`, feature version `3.1.0`. No predictive model was trained; nothing here is a claim about whether scheme improves NFL predictions. That is the question these features exist to test.

## Summary

| Measure | Value |
|---|---|
| Relevant college team-seasons (prospects' pre-draft seasons) | 770 |
| Team-seasons with a **verified** label (manual source review) | 5 (4 with an offensive-family label) |
| Team-seasons with a **reported** family label (Wikipedia infobox, low confidence) | 493 |
| Team-seasons with play-by-play tendencies (2016-2025 fetched) | 318 |
| Drafted QBs (2005+ model universe) with any usable scheme feature | 70.8% |
| Drafted QBs (all 2000-2026) with any usable scheme feature | 56.1% |
| Conflicting manual classifications (kept unlabelled) | 1 |
| Reported text mapped only to a non-family dimension (e.g. Pistol, West Coast) | 21 |
| Wikipedia scheme text deliberately left unmapped (ambiguous) | 4 |
| Team-seasons whose page has no scheme field (unknown) | 252 |

**Bottom line.** Verified, contemporaneous scheme labels are expensive to find. About a quarter of the candidate sources that were read actually tied a scheme to a specific team **and** season. So the verified tier is a small, randomly sampled audit set, not a modelling feature. The usable categorical signal comes from the low-confidence *reported* tier. It is kept separately named (`scheme_rep_*`) and is never presented as verified. The measured tendencies (`scheme_q_*`, `env_*`) are statistics, not schemes.

## 1. How labels were assigned

| Tier | Evidence rule | Status / confidence | Feature prefix |
|---|---|---|---|
| A. Verified | A source was read and explicitly ties the label to that team AND season (contemporaneous where possible). Verbatim quote, URL, title and publication date stored. | verified / high-medium-low | `scheme_ver_*` |
| A'. Reported | Wikipedia team-season infobox `off_scheme` (season-specific, but community-edited, 6 of 518 fields cited, usually written after the fact). | reported / low | `scheme_rep_*` |
| B. Quantitative | Measured from box scores and CFBD play-by-play. Never turned into a scheme name. | n/a | `scheme_q_*`, `env_*` |

Rules applied:
- A coach's or coordinator's philosophy, or a statement about a different season, is **not** evidence. Such sources are logged as `insufficient_evidence`. This excluded, for example, every coach-tenure list on the Wikipedia Air Raid / Flexbone / Spread articles.
- Sources that contradict each other produce no label (`conflicting`).
- Pages that could not be read (HTTP 403/404) are logged as `source_unavailable`, with no claim.
- Researched with nothing found: `searched_no_candidate_source`. Never researched: `not_researched`. Not researching a season is not evidence of anything.
- Labels may overlap (one row per label), across dimensions: offensive family, run concepts, passing concepts, formation tendency, tempo. See `config/offensive_scheme_taxonomy.json`.
- **Reference not used:** `deepmetricanalytics.com/college-football/scheme-matchups` is a 2026-season preview ("Opening-day expectations from coaching research... Not observed play-calling"), with no per-team citations. It covers none of the 2004-2025 seasons here.

### Manual research design

Researched team-seasons came from a **seeded random sample** (seed 20261109): 3 final college seasons per draft class, 2005-2021, 51 in total. That keeps 'has a verified label' from becoming a proxy for draft position, which it would be if only famous QBs were researched. 24 team-seasons were researched in this pass (28 log rows); the rest of the sample remains `not_researched`. Outcomes:

| Verification status | Rows |
|---|---|
| insufficient_evidence | 8 |
| verified | 7 |
| source_unavailable | 6 |
| searched_no_candidate_source | 4 |
| conflicting | 2 |
| not_supported | 1 |

## 2. Source citations and provenance (manual tier)

| Team | Season | Label | Status | Conf. | Source | Published | Evidence (verbatim where quoted) |
|---|---|---|---|---|---|---|---|
| Hawai'i | 2007 | offensive_family: run_and_shoot | verified | high | [Colt has taken UH for joy ride (The Honolulu Advertiser)](https://archives-new.honoluluadvertiser.com/article/2007/Dec/30/sp/hawaii712300302.html) | 2007-12-30 | "Brennan has thrown for 13,997 yards in June Jones' run-and-shoot offense, including 4,174 this season." |
| Georgia | 2008 | offensive_family: multiple_hybrid | verified | low | [Georgia Bulldogs Football: An In-Depth Preview of the 2008-2009 Season (Bleacher Report)](https://bleacherreport.com/articles/38140-georgia-bulldogs-football-an-in-depth-preview-of-the-2008-2009-season) | 2008-07-15 | "The offense will feature the "semi" pro spread offense of Mike Bobo." / "...originally a pro style system." |
| Wisconsin | 2011 | run_concepts: zone_read | verified | medium | [Deceptively simple Badgers offense enjoying unprecedented success (Sports Illustrated)](https://www.si.com/more-sports/2011/09/29/wisconsin-offense) | 2011-09-29 | "...Chryst, who added a shotgun zone-read component to the offense to better suit Wilson." |
| Texas Tech | 2016 | offensive_family: spread | verified | medium | [2017 NFL draft prospect countdown, No. 26: Patrick Mahomes, QB, Texas Tech (Sports Illustrated)](https://www.si.com/nfl/2017/03/27/patrick-mahomes-nfl-draft-scouting-report) | 2017-03-27 | "...within Texas Tech's spread system." (article covers the 2016 season) |
| LSU | 2019 | offensive_family: spread | verified | high | ["Everybody Eats" - Tigers Feast in Fast Spread Attack (LSU Athletics)](https://lsusports.net/news/2019/09/01/211810591) | 2019-08-31 | "Saturday night, the spread was spectacular." |
| LSU | 2019 | offensive_family: spread | verified | high | ['Out of the Dark Ages': How LSU's Lethal New Spread Attack Was Built (Sports Illustrated)](https://www.si.com/college/2019/09/17/lsu-spread-offense-joe-burrow-ed-orgeron) | 2019-09-17 | "We're going to the spread, baby!" (OC Steve Ensminger, January 2019, before the 2019 season) |
| LSU | 2019 | formation_tendency: shotgun_heavy | verified | high | ['Out of the Dark Ages': How LSU's Lethal New Spread Attack Was Built (Sports Illustrated)](https://www.si.com/college/2019/09/17/lsu-spread-offense-joe-burrow-ed-orgeron) | 2019-09-17 | "...128 of their 134 snaps in their first two games were out of the shotgun, a 95.5% clip." |
| Oklahoma | 2017 | offensive_family: air_raid | conflicting |  | [Baker Mayfield Is NFL Draft's QB1, And It Isn't Even Close (Bleacher Report)](https://bleacherreport.com/articles/2772192-baker-mayfield-is-nfl-drafts-qb1-and-it-isnt-even-close) | 2018-04-25 | "...Lincoln Riley employs an Air Raid variation that schemes receivers open." |
| Oklahoma | 2017 | offensive_family: air_raid | conflicting |  | [First Look: The least SEC fans should know about Oklahoma, a finely tuned offensive machine (Saturday Down South)](https://www.saturdaydownsouth.com/georgia-football/first-look-least-sec-fans-know-oklahoma-finely-tuned-offensive-machine) | 2017-12-06 | "...it would be a serious mistake to lump Oklahoma's offense in with the kind of pass-happy, Air Raid-style attacks..." |
| BYU | 2006 | offensive_family: spread | insufficient_evidence |  | [2006 Season Preview (BYU Athletics)](https://byucougars.com/news/2006/03/20/2006-season-preview) | 2006-03-20 | "...experienced players who are well suited to be effective in the spread offense." |
| Florida State | 2010 | - | insufficient_evidence |  | [Ponder could be the gateway to the West (Sportspress Northwest)](https://www.sportspressnw.com/2110084/2011/ponder-could-be-the-gateway-to-the-west) | 2011-01-27 | Describes Jimbo Fisher's 'multiple-read' system and contrasts it with the spread; names no scheme for 2010. |
| Stanford | 2011 | offensive_family: pro_style | insufficient_evidence |  | [2012 NFL Draft Scouting Report: Andrew Luck (WalterFootball)](https://walterfootball.com/scoutingreport2012aluck.php) |  | "Mastered a pro-style offense" (player-strength bullet) |
| Wisconsin | 2011 | offensive_family: pro_style | insufficient_evidence |  | [Wilson could help make 2011 a special year for the Badgers (The Daily Cardinal)](https://www.dailycardinal.com/article/2011/06/wilson-could-help-make-2011-a-special-year-for-the-badgers) | 2011-06-27 | "...similar to the pro-style system Wilson ran at N.C. State." |
| Oregon | 2014 | offensive_family: spread | insufficient_evidence |  | [How hard will it be for Marcus Mariota to transition from spread to pros? (Sports Illustrated)](https://www.si.com/nfl/2015/02/27/2015-nfl-draft-marcus-mariota-spread-offense) | 2015-02-27 | "...quarterback from a spread offense" (career-level; 2014 not named) |
| Oregon | 2019 | - | insufficient_evidence |  | [2020 NFL Draft scouting report: Oregon quarterback Justin Herbert (NFL Mocks)](https://nflmocks.com/2020/01/05/2020-nfl-draft-scouting-report-herbert) | 2020-01-05 | "Quick throws and screens at Oregon were run in abundance this last year" (no scheme named) |
| Alabama | 2020 | run_concepts: rpo | insufficient_evidence |  | [Mac Jones: 2021 NFL Draft Prospect Profile (RotoBaller)](https://www.rotoballer.com/mac-jones-2021-nfl-draft-prospect-profile/845107) | 2021-03-15 | 'Offense based around the run game and RPOs' (paraphrase-level; verbatim wording and season not confirmed) |
| Georgia | 2019 | - | insufficient_evidence |  | [2020 NFL Draft Scouting Report: Jake Fromm (WalterFootball)](https://walterfootball.com/scoutingreport2020JFromm.php) |  | Play-action language refers to his 2017-2018 seasons, not 2019. |
| NC State | 2018 | - | not_supported |  | [Finley, Ryan, QB, NC State (CBS Sports)](https://www.cbssports.com/nfl/draft/news/finley-ryan-qb-nc-state) | 2019-03-28 | No sentence describes NC State's offensive scheme. |
| Missouri | 2005 | - | source_unavailable |  | [link](https://www.semissourian.com/sports/pinkel--smith-eager-to-regain-respect-1116583) |  | HTTP 403 when fetched; content not read so no claim is recorded. |
| Bowling Green | 2005 | - | source_unavailable |  | [link](https://vault.si.com/vault/2005/08/15/23-bowling-green) |  | HTTP 404 when fetched. |
| USC | 2005 | - | source_unavailable |  | [link](https://vault.si.com/vault/2005/08/15/1-usc) |  | HTTP 404 when fetched. |
| Boston College | 2007 | - | source_unavailable |  | [link](https://vault.si.com/vault/2007/10/01/special-delivery) |  | HTTP 404 when fetched. |
| Central Michigan | 2009 | - | source_unavailable |  | [link](https://vault.si.com/vault/2009/10/05/total-package) |  | HTTP 404 when fetched. |
| USC | 2017 | - | source_unavailable |  | [link](https://elevenwarriors.com/ohio-state-football/2017/12/89605/scouting-report-usc-boasts-future-nfl-quarterback-inconsistent-run-defense) |  | HTTP 403 when fetched. |
| Georgia | 2004 | - | searched_no_candidate_source |  |  |  | Web search found no contemporaneous source naming the 2004 scheme. |
| Oregon State | 2004 | - | searched_no_candidate_source |  |  |  | Web search found only general preseason descriptions; no scheme named. |
| Michigan | 2007 | - | searched_no_candidate_source |  |  |  | Web search found player-level 'pro-style hurler' descriptions only. |
| Duke | 2018 | - | searched_no_candidate_source |  |  |  | Web search found player-level 'pro-style concepts' descriptions only. |

**Reported tier provenance:** 580 label rows from 518 infobox fields (770 of 770 team-season pages found through the official MediaWiki API, revisions dated 2025-09-15 to 2026-11-08). Each row stores the revision permalink (`index.php?oldid=...`), the page title and the raw field text. Content is CC BY-SA 4.0.

## 3. Coverage by decade

| Decade | Team-seasons | Verified family | Reported family | Play-by-play profile |
|---|---|---|---|---|
| 2000s | 220 | 2 | 103 | 0 |
| 2010s | 372 | 2 | 236 | 146 |
| 2020s | 178 | 0 | 154 | 172 |

Play-by-play exists only for 2016-2025 so far. 2004-2015 needs about 200 more CFBD calls, deferred to the next monthly quota to stay under the 1,000-call free tier.

## 4. Label distribution and unknowns

| Reported offensive family | Team-seasons |
|---|---|
| spread | 171 |
| pro_style | 145 |
| multiple_hybrid | 120 |
| air_raid | 44 |
| spread_option | 30 |
| veer_and_shoot | 22 |
| run_and_shoot | 6 |

Unmapped Wikipedia text (ambiguous terms, deliberately left `unknown`):

| Raw text | Team-seasons |
|---|---|
| Gulf Coast | 2 |
| Play Action/Read | 1 |
| Run and gun | 1 |

Verified vs reported, where both exist:

| Team | Season | Verified family | Reported family | Wikipedia text |
|---|---|---|---|---|
| Texas Tech | 2016 | spread | air_raid | Air raid |
| LSU | 2019 | spread | pro_style|spread | Pro Spread |
| Hawai'i | 2007 | run_and_shoot | run_and_shoot | Run and shoot |
| Georgia | 2008 | multiple_hybrid | pro_style | Pro-style |

The tiers agree for Hawai'i 2007, Texas Tech 2016 (Air Raid is spread-family) and LSU 2019 ("Pro Spread" includes spread). They disagree for Georgia 2008. That is one reason the tiers are never merged.

## 5. Quantitative offensive profiles

| Feature (team-season) | Calculation | Limitations |
|---|---|---|
| Passing / rushing play rate | Team pass attempts / (pass att + rush att), season box totals (`env_pass_rate`, `rush_rate`) | NCAA counts sacks as rushes, so pass rate is understated for sack-prone teams |
| Offensive plays per game | (pass att + rush att) / games; also non-garbage play-by-play plays per game | Includes all game states (box version) |
| Standard-down pass rate | Play-by-play: pass share on 1st down, 2nd & <=7, 3rd/4th & <=4; garbage time removed | 2016+ only; sacks count as passes; scrambles count as rushes (CFBD play types cannot separate designed runs) |
| Passing-down pass rate | Same, on 2nd & 8+, 3rd/4th & 5+ | As above |
| Early-down pass rate / overall non-garbage pass rate | Same, 1st-2nd down / all downs | As above |
| Explosiveness, success rate, PPA | CFBD advanced season stats (`off_explosiveness`, `off_success_rate`, `off_ppa`) | Only where play-by-play covers >= 85% of box-score plays |
| QB rushing involvement | QB's share of team rushing attempts (`final_qb_rush_share`) | Includes sacks (NCAA) |
| Team passing efficiency | Net passing yards / pass attempts (`team_pass_ypa`); passing PPA | Largely the QB's own production (circular) |
| Tempo | Possession seconds per play; SP+ pace (2005-2018); plays per game | Possession time is a crude proxy |
| Teams with < 300 non-garbage plays | Rates blanked | Mostly FCS teams seen only in games vs FBS opponents |

Not computed, because CFBD play types do not reliably identify them: designed-run rate, RPO rate, play-action rate, formation (shotgun / pistol / under-center) rates.

Team-season coverage of play-by-play features: pass_rate_nongarbage 41.3%, standard_down_pass_rate 41.3%, passing_down_pass_rate 41.3%, early_down_pass_rate 41.3%, plays_per_game_nongarbage 41.3%.

**Face-validity check** (descriptive; statistics used to *check* labels, never to create them): mean tendencies by reported family, team-seasons with play-by-play.

| Reported family | n | Std-down pass rate | Pass-down pass rate | Non-garbage plays/game |
|---|---|---|---|---|
| air_raid | 27 | 0.485 | 0.711 | 69.2 |
| spread | 114 | 0.447 | 0.678 | 66.3 |
| spread_option | 21 | 0.427 | 0.656 | 65.5 |
| veer_and_shoot | 15 | 0.423 | 0.641 | 67.7 |
| multiple_hybrid | 52 | 0.436 | 0.669 | 64.6 |
| pro_style | 63 | 0.396 | 0.662 | 63.6 |
| (no reported label) | 61 | 0.444 | 0.683 | 68.1 |

The order matches what the labels mean: Air Raid teams pass most on standard downs, pro-style teams least. The ranges overlap heavily, so the labels carry signal but are coarse.

## 6. QB features (per season and aggregated)

- Per-season table: 867 prospect-seasons. Each is joined on the exact (team, season) the QB played for, so transfers (56 multi-school QBs) pick up each school's offense.
- Aggregates: final-season labels and tendencies, career exposure (attempt-weighted share of labelled seasons in spread-family / pro-style offenses), label coverage of career attempts, and the number of distinct systems. `final_observed_season_gap` flags 6 QBs whose last observed season is not the one right before the draft (usually a final FCS season outside CFBD box scores, e.g. Rhett Bomar).

| Group | All drafted QBs | 2005+ model universe |
|---|---|---|
| Verified final-season family | 1.2% | 1.5% |
| Reported final-season family | 49.1% | 61.9% |
| Any reported label in career | 59.1% | 74.6% |
| Final-season play-by-play profile | 33.5% | 42.3% |
| Final-season box-score pass rate | 73.5% | 92.7% |
| Any scheme feature | 56.1% | 70.8% |

Feature-level coverage (2005+ model universe, % non-missing):

| Feature | % non-missing |
|---|---|
| scheme_ver_final_known | 96.5 |
| scheme_ver_final_spread_family | 1.5 |
| scheme_rep_final_known | 96.5 |
| scheme_rep_final_spread | 61.9 |
| scheme_rep_final_air_raid | 61.9 |
| scheme_rep_final_pro_style | 61.9 |
| scheme_rep_final_spread_option | 61.9 |
| scheme_rep_final_multiple_hybrid | 61.9 |
| scheme_rep_final_spread_family | 61.9 |
| scheme_rep_career_label_coverage | 96.5 |
| scheme_rep_career_spread_family_share | 74.6 |
| scheme_rep_career_pro_style_share | 74.6 |
| scheme_rep_n_distinct_families | 74.6 |
| scheme_q_final_standard_down_pass_rate | 42.3 |
| scheme_q_final_passing_down_pass_rate | 42.3 |
| scheme_q_final_early_down_pass_rate | 42.3 |
| scheme_q_final_pass_rate_nongarbage | 42.3 |
| scheme_q_career_standard_down_pass_rate | 43.1 |

**Is missingness informative?** Among train/validation/test QBs, the share reaching 16+ starts is 36.4% with a reported final-season label (n=110) vs 38.0% without (n=92). There is no large outcome gap, so missing-label indicators are unlikely to act as an outcome proxy. That is descriptive, not a test.

## 7. Leakage and validation

- QB scheme seasons on or after the draft year: **0**. All scheme features use pre-draft college seasons only, and they are in the feature registry with the `final_college_season_end` availability rule, so the pipeline's leakage audit covers them.
- Wikipedia text was written after the fact. It describes the pre-draft season but could carry hindsight; this is why it is low confidence and kept apart from the verified tier.
- Label-table schema check this run: `current` (the empty v1 table was dropped and recreated as the v2 provenance schema on the first run, before which the database was backed up to `data/backups/qb_history_pre_schemes_20261109.db`; later runs report `current`). The migration refuses to run if the old table holds rows.

`pytest`: **165 passed, 0 failed, 0 skipped** of 165 (whole suite).

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

Scheme-specific checks (`tests/test_offensive_schemes.py`): incorrect team-season joins (no carry-over to other seasons), transfers, duplicated annotations, source attribution (verified rows need URL, title, date and quote; Wikipedia rows need revision permalinks), unsupported classifications (taxonomy-only labels, only manual review can verify, statistics can never label), missing statistics stay NaN (not 0), down-specific rate calculations, garbage time, double-merge regression, pre-draft-only seasons. Database idempotency is covered by the end-to-end integration test (two runs, identical tables).

## 8. Outputs

| Output | Contents |
|---|---|
| `config/offensive_scheme_labels.csv` | Manual research log: every assertion with provenance, including failed and conflicting ones |
| `config/offensive_scheme_taxonomy.json` | Dimensions, labels, raw-text mapping, deliberately unmapped terms |
| `data/processed/college_offensive_schemes.parquet` | One row per relevant team-season: tiers, research status, quantitative profile |
| `data/processed/qb_scheme_features.parquet` | One row per drafted QB: aggregated pre-draft scheme features |
| `data/processed/qb_scheme_seasons.parquet` | One row per QB college season: that season's labels and tendencies |
| `data/processed/offensive_scheme_labels_all.parquet` | All label rows (manual + reported) with provenance |
| DB: `cfb_offensive_scheme_labels`, `cfb_team_season_schemes` | Same, in the project database; features also in `prospect_feature_values` |

## 9. Next steps

- After the Dec 1 quota reset: fetch 2004-2015 play-by-play (about 200 calls; `python -m src.ingest_cfbd_plays` with those seasons) to extend `scheme_q_*` to the earlier draft classes.
- Continue the manual audit through the remaining 27 sampled team-seasons. Prefer official athletics sites, newspapers, and pre-draft analyses that name the season.
- Use the verified tier to estimate the accuracy of the reported tier once it has enough rows. With 4 overlapping team-seasons today, no accuracy figure is meaningful.
