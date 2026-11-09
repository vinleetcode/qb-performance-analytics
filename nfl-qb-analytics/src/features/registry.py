"""Versioned feature registry -- the single source of truth for which prospect
features exist, where they come from, when they become knowable, and which
model families may use them.

Model scopes (nested):
  pre_draft_college  : college performance, opponent adjustment, development,
                       age/experience, offensive environment, team talent,
                       college-roster bio. No combine, no scouting, no draft slot.
  pre_draft_full     : + NFL Combine block (systematically missing -> own group).
  pre_draft_scouting : + expert pre-draft rankings/grades (pre-draft, but they
                       encode market expectations much like draft slot does).
  post_draft         : + draft round / overall pick. ONLY here.
Targets/outcomes are never features (see FORBIDDEN_FEATURE_PREFIXES).

Bump FEATURE_VERSION whenever a definition or formula changes; the version is
stored with every feature value and with the registry rows in the database.
"""
from __future__ import annotations

from dataclasses import dataclass

FEATURE_VERSION = '3.1.0'  # 3.1.0: offensive-scheme groups (verified / reported / quantitative)

SCOPES = ['pre_draft_college', 'pre_draft_full', 'pre_draft_scouting', 'post_draft']

AVAIL_SEASON_END = 'final_college_season_end'   # last game date of the final college season
AVAIL_PRE_DRAFT = 'before_draft_day'            # known before draft day (birthdate-derived; published rankings)
AVAIL_COMBINE = 'combine_known_by'              # March 15 of draft year (config/nfl_draft_dates.json)
AVAIL_DRAFT_DAY = 'draft_day'                   # first day of the draft -> post-draft only

GROUP_SCOPE = {
    'college_production': 'pre_draft_college',
    'opponent_adjusted': 'pre_draft_college',
    'development': 'pre_draft_college',
    'age_experience': 'pre_draft_college',
    'env_scheme': 'pre_draft_college',
    'env_offensive_line': 'pre_draft_college',
    'env_receivers': 'pre_draft_college',
    'team_talent': 'pre_draft_college',
    'college_roster_bio': 'pre_draft_college',
    'scheme_verified': 'pre_draft_college',
    'scheme_reported': 'pre_draft_college',
    'scheme_quantitative': 'pre_draft_college',
    'athletic_combine': 'pre_draft_full',
    'scouting': 'pre_draft_scouting',
    'draft_position': 'post_draft',
}

GROUP_SOURCES = {
    'college_production': 'cfbd_player_season_stats,cfbd_game_player_passing',
    'opponent_adjusted': 'cfbd_game_player_passing,cfbd_games,cfbd_player_season_stats',
    'development': 'cfbd_player_season_stats,cfbd_game_player_passing',
    'age_experience': 'nflverse_draft_picks,cfbd_player_season_stats,cfbd_game_player_passing,cfbd_games',
    'env_scheme': 'cfbd_team_season_stats,cfbd_player_season_stats',
    'env_offensive_line': 'cfbd_team_season_stats,cfbd_games',
    'env_receivers': 'cfbd_player_receiving,cfbd_recruiting,cfbd_returning_production',
    'team_talent': 'cfbd_team_talent,cfbd_recruiting,cfbd_returning_production',
    'college_roster_bio': 'cfbd_player_season_stats',
    'scheme_verified': 'manual_scheme_research',
    'scheme_reported': 'wikipedia_team_season_infobox',
    'scheme_quantitative': 'cfbd_plays',
    'athletic_combine': 'nflverse_combine',
    'scouting': 'cfbd_draft_picks',
    'draft_position': 'nflverse_draft_picks',
}

GROUP_AVAILABILITY = {g: AVAIL_SEASON_END for g in GROUP_SCOPE}
GROUP_AVAILABILITY.update({'athletic_combine': AVAIL_COMBINE, 'scouting': AVAIL_PRE_DRAFT,
                           'draft_position': AVAIL_DRAFT_DAY})

FORBIDDEN_FEATURE_PREFIXES = ('started_', 'games_started_', 'pass_yards_5yr', 'pass_tds_5yr', 'pass_attempts_5yr',
                              'any_a_5yr', 'passer_rating_5yr', 'seasons_to_final', 'career_span', 'career_end',
                              'appeared_any', 'nfl_', 'career_av', 'allpro', 'probowls', 'hof')

_SC = 'Season totals; sacks count as NCAA rushing plays.'
_OA = 'Leave-one-out opponent pass-defense baseline (opponent needs >=4 other box-scored games); FCS opponents usually lack a baseline -> see *_attempt_coverage.'
_SAMPLE = 'NaN when fewer than 2 seasons with >=100 attempts.'

# (name, group, description, caveats)
_SPECS = [
    # --- final-season production -----------------------------------------
    ('final_attempts', 'college_production', 'Pass attempts in final pre-draft college season.', 'Volume/opportunity, not ability.'),
    ('final_completion_pct', 'college_production', 'Completion % in final season.', None),
    ('final_yards_per_attempt', 'college_production', 'Yards per attempt in final season.', None),
    ('final_td_pct', 'college_production', 'Passing TDs per 100 attempts, final season.', None),
    ('final_int_pct', 'college_production', 'Interceptions per 100 attempts, final season.', None),
    ('final_ncaa_passer_rating', 'college_production', 'NCAA passer efficiency rating, final season.', None),
    ('final_rushing_attempts', 'college_production', 'QB rushing attempts, final season.', _SC),
    ('final_rushing_yards', 'college_production', 'QB rushing yards, final season (net of sacks).', _SC),
    ('final_rushing_tds', 'college_production', 'QB rushing TDs, final season.', None),
    ('final_qb_rush_share', 'env_scheme', "QB's share of team rushing attempts, final season (QB rushing involvement).", _SC),
    # --- opponent adjustment (final season) --------------------------------
    ('final_opp_adj_ypa', 'opponent_adjusted', 'Opponent-adjusted YPA, final season.', _OA),
    ('final_opp_adj_completion_pct', 'opponent_adjusted', 'Opponent-adjusted completion %, final season.', _OA),
    ('final_opp_adj_td_pct', 'opponent_adjusted', 'Opponent-adjusted TD %, final season.', _OA),
    ('final_opp_adj_int_pct', 'opponent_adjusted', 'Opponent-adjusted INT %, final season.', _OA),
    ('final_opp_adj_attempt_coverage', 'opponent_adjusted', 'Share of final-season box-score attempts that had an opponent baseline.', None),
    ('final_sos_pass_defense', 'opponent_adjusted', 'Attempt-weighted opponent YPA allowed minus FBS mean (negative = tougher pass defenses), final season.', _OA),
    ('final_sp_strength_of_schedule', 'opponent_adjusted', "SP+ strength of schedule of the QB's final-season team (CFBD /ratings/sp).", 'Team-level, final ratings.'),
    ('final_opp_def_sp_rating_avg', 'opponent_adjusted', 'Attempt-weighted SP+ defensive rating of opponents faced, final season.', 'SP+ overall defensive rating; a few opponents (mostly FCS) are unrated.'),
    ('final_conf_rel_rating_z', 'opponent_adjusted', 'Final-season NCAA rating z-score vs QBs (>=100 att) in same conference-season.', 'Pool = all CFBD passers, attempt-weighted.'),
    ('final_conf_rel_ypa_z', 'opponent_adjusted', 'Final-season YPA z-score vs same conference-season pool.', None),
    ('final_conf_rel_rating_pctile', 'opponent_adjusted', 'Final-season rating percentile within conference-season pool.', None),
    ('final_vs_top_def_attempts', 'opponent_adjusted', 'Final-season attempts vs top-25 FBS pass defenses (leave-one-out YPA allowed).', None),
    ('final_vs_top_def_ypa', 'opponent_adjusted', 'Final-season YPA vs top-25 pass defenses.', 'Small samples; check attempts.'),
    ('final_vs_top_def_rating', 'opponent_adjusted', 'Final-season NCAA rating vs top-25 pass defenses.', 'Small samples; check attempts.'),
    # --- opponent adjustment (career) ---------------------------------------
    ('career_opp_adj_ypa', 'opponent_adjusted', 'Career attempt-weighted opponent-adjusted YPA.', _OA),
    ('career_opp_adj_completion_pct', 'opponent_adjusted', 'Career opponent-adjusted completion %.', _OA),
    ('career_opp_adj_td_pct', 'opponent_adjusted', 'Career opponent-adjusted TD %.', _OA),
    ('career_opp_adj_int_pct', 'opponent_adjusted', 'Career opponent-adjusted INT %.', _OA),
    ('career_sos_pass_defense', 'opponent_adjusted', 'Career attempt-weighted opponent pass-defense strength.', _OA),
    ('career_vs_top_def_attempts', 'opponent_adjusted', 'Career attempts vs top-25 pass defenses.', None),
    ('career_vs_top_def_ypa', 'opponent_adjusted', 'Career pooled YPA vs top-25 pass defenses.', None),
    ('career_vs_top_def_rating', 'opponent_adjusted', 'Career pooled NCAA rating vs top-25 pass defenses.', None),
    # --- development --------------------------------------------------------
    ('n_trend_seasons', 'development', 'Seasons with >=100 attempts (eligible for efficiency trends).', None),
    ('dev_slope_rating', 'development', 'Attempt-weighted slope of NCAA rating per season.', _SAMPLE),
    ('dev_slope_ypa', 'development', 'Attempt-weighted slope of YPA per season.', _SAMPLE),
    ('dev_slope_cmp_pct', 'development', 'Attempt-weighted slope of completion % per season.', _SAMPLE),
    ('dev_slope_td_pct', 'development', 'Attempt-weighted slope of TD % per season.', _SAMPLE),
    ('dev_slope_int_pct', 'development', 'Attempt-weighted slope of INT % per season.', _SAMPLE),
    ('dev_slope_opp_adj_ypa', 'development', 'Attempt-weighted slope of opponent-adjusted YPA per season.', _SAMPLE),
    ('dev_slope_attempts', 'development', 'Slope of pass attempts per season (VOLUME trend, not ability).', 'Opportunity, not improvement.'),
    ('dev_slope_rush_yards', 'development', 'Slope of rushing yards per season (volume).', _SC),
    ('dev_yoy_rating_final', 'development', 'Final-season NCAA rating minus previous qualifying season.', 'Needs both seasons >=100 att.'),
    ('dev_yoy_ypa_final', 'development', 'Final-season YPA minus previous qualifying season.', None),
    ('dev_yoy_cmp_pct_final', 'development', 'Final-season completion % minus previous qualifying season.', None),
    ('dev_yoy_td_pct_final', 'development', 'Final-season TD % minus previous qualifying season.', None),
    ('dev_yoy_int_pct_final', 'development', 'Final-season INT % minus previous qualifying season.', None),
    ('dev_final_vs_prior_rating', 'development', 'Final-season rating minus pooled rating of earlier qualifying seasons.', None),
    ('dev_final_vs_prior_ypa', 'development', 'Final-season YPA minus pooled YPA of earlier qualifying seasons.', None),
    ('dev_final_vs_prior_attempts', 'development', 'Final-season attempts minus mean of earlier seasons (volume).', 'Opportunity, not improvement.'),
    ('dev_final_vs_prior_rush_yards', 'development', 'Final-season rush yards minus mean of earlier seasons.', None),
    ('n_productive_seasons', 'development', 'Seasons with >=150 attempts AND NCAA rating >= that season\'s median among CFBD passers with >=150 attempts.', None),
    # --- age & experience ---------------------------------------------------
    ('age_at_draft', 'age_experience', 'Age in years on the first day of the draft (recorded birthdate only).', 'Missing birthdate -> NaN; never estimated from class year.'),
    ('age_final_college_season', 'age_experience', 'Age on Sept 1 of the final college season.', 'Missing birthdate -> NaN.'),
    ('n_college_seasons_observed', 'age_experience', 'College seasons with a CFBD stat record.', 'Lower bound when career_possibly_truncated=1.'),
    ('career_pass_attempts', 'age_experience', 'Career college pass attempts.', 'Lower bound when truncated.'),
    ('career_pass_yards', 'college_production', 'Career college passing yards.', 'Lower bound when truncated.'),
    ('career_pass_tds', 'college_production', 'Career college passing TDs.', None),
    ('career_interceptions', 'college_production', 'Career college interceptions.', None),
    ('career_rush_attempts', 'college_production', 'Career college rushing attempts.', _SC),
    ('career_rush_yards', 'college_production', 'Career college rushing yards.', _SC),
    ('career_rush_tds', 'college_production', 'Career college rushing TDs.', None),
    ('career_cmp_pct', 'college_production', 'Career completion %.', None),
    ('career_ypa', 'college_production', 'Career YPA.', None),
    ('career_td_pct', 'college_production', 'Career TD %.', None),
    ('career_int_pct', 'college_production', 'Career INT %.', None),
    ('career_rating', 'college_production', 'Career NCAA passer rating.', None),
    ('career_games_with_pass_attempt', 'age_experience', 'College games with >=1 pass attempt in CFBD box scores.', 'Rush-only appearances not counted.'),
    ('seasons_as_primary_passer', 'age_experience', 'Seasons leading the team in attempts in >=50% of team games (PROXY for starter seasons).', 'Not verified starts; CFBD has no starts field.'),
    ('final_participation_rate', 'age_experience', 'Final season: games with a pass attempt / team games played.', 'Participation, NOT an injury measure.'),
    ('n_schools', 'age_experience', 'Distinct schools across observed college seasons.', None),
    ('transferred', 'age_experience', '1 if more than one school.', None),
    ('career_possibly_truncated', 'age_experience', '1 if first observed season is 2004 (CFBD floor) -> earlier seasons may be missing.', None),
    # --- environment: scheme ------------------------------------------------
    ('env_plays_per_game', 'env_scheme', 'Team offensive plays (pass att + rush att) per game, final season.', _SC),
    ('env_pass_rate', 'env_scheme', 'Team pass attempts / plays, final season.', 'Understated for sack-prone teams (sacks = rushes).'),
    ('env_seconds_per_play', 'env_scheme', 'Team possession seconds per offensive play (tempo proxy), final season.', 'Crude tempo proxy.'),
    ('env_sp_pace', 'env_scheme', 'SP+ offensive pace, final season.', 'CFBD SP+ sub-ratings only 2005-2018.'),
    ('env_sp_run_rate', 'env_scheme', 'SP+ run rate, final season.', 'Only 2005-2018.'),
    ('env_off_ppa', 'env_scheme', 'Team offensive PPA per play, final season.', 'Only where play-by-play coverage >= 85%.'),
    ('env_off_success_rate', 'env_scheme', 'Team offensive success rate, final season.', 'Only where play-by-play coverage >= 85%.'),
    ('env_sp_offense_rating', 'env_scheme', 'SP+ offensive rating (opponent-adjusted), final season.', 'Includes the QB\'s own contribution.'),
    # --- environment: offensive line ---------------------------------------
    ('ol_sack_rate_allowed', 'env_offensive_line', 'Team sacks allowed / (pass att + sacks), final season.', 'CFBD team sack stats exist from 2016 only. Reflects QB decisions and scheme, not only pass protection.'),
    ('ol_opp_adj_sack_rate', 'env_offensive_line', 'Sack rate minus (schedule-weighted opponent defensive sack rate - FBS mean).', 'From 2016 only; opponent rates include the game vs this team.'),
    ('ol_team_rush_ypc', 'env_offensive_line', 'Team yards per rush, final season.', _SC),
    ('ol_off_stuff_rate', 'env_offensive_line', 'Share of runs stopped at/behind line, final season.', 'Only where play-by-play coverage >= 85%.'),
    ('ol_off_line_yards', 'env_offensive_line', 'Offensive line yards per rush, final season.', 'Only where play-by-play coverage >= 85%.'),
    ('ol_off_havoc_allowed', 'env_offensive_line', 'Havoc plays allowed rate, final season.', 'Only where play-by-play coverage >= 85%; CFBD havoc is null in some years.'),
    # --- environment: receivers ----------------------------------------------
    ('wr_team_receiving_yards', 'env_receivers', 'Team receiving yards, final season.', 'CIRCULAR: equals team passing volume; not receiver ability.'),
    ('wr_top_target_reception_share', 'env_receivers', "Top target's share of team receptions, final season (concentration).", None),
    ('wr_top3_reception_share', 'env_receivers', 'Top-3 targets share of team receptions, final season.', None),
    ('wr_top3_avg_recruit_rating', 'env_receivers', 'Mean 247 composite recruit rating of top-3 targets (by receptions), final season.', 'Only recruits CFBD links by athleteId; transfers/JUCO often unrated.'),
    ('wr_returning_receiving_ppa_pct', 'env_receivers', 'Share of prior-season receiving PPA returning (known before the season).', 'CFBD coverage starts 2014.'),
    # --- team talent ---------------------------------------------------------
    ('talent_composite', 'team_talent', '247 Team Talent Composite, final season.', 'Available 2015+ only.'),
    ('talent_composite_pctile', 'team_talent', 'Within-season percentile of talent composite.', 'Available 2015+ only.'),
    ('talent_recruiting_points_4yr_avg', 'team_talent', 'Mean recruiting class points, classes season-3..season.', None),
    ('talent_recruiting_4yr_pctile', 'team_talent', 'Within-season percentile of 4-class recruiting average.', None),
    ('talent_returning_ppa_pct', 'team_talent', 'Share of total prior-season PPA returning (roster experience).', 'CFBD coverage starts 2014.'),
    ('talent_returning_usage', 'team_talent', 'Share of prior-season usage returning.', 'CFBD coverage starts 2014.'),
    # --- college roster bio -----------------------------------------------------
    ('bio_college_height_inches', 'college_roster_bio', 'Height listed on final-season CFBD roster.', 'School-listed, unverified.'),
    ('bio_college_weight_lbs', 'college_roster_bio', 'Weight listed on final-season CFBD roster.', 'School-listed, unverified.'),
    ('final_observed_season_gap', 'age_experience', 'Draft year - 1 - last observed college season (0 = final observed season is the one right before the draft).', 'Positive values usually mean later seasons were at FCS schools outside CFBD box-score coverage.'),
    # --- offensive scheme: A. verified (manual source review; sparse by design) ---
    ('scheme_ver_final_known', 'scheme_verified', '1 if the final college season has a manually verified offensive-family label.', 'Research covered a random, draft-class-stratified sample only; 0 mostly means not researched.'),
    ('scheme_ver_final_spread_family', 'scheme_verified', '1 if the verified final-season family is spread-family (spread, air raid, spread option, veer & shoot, run & shoot).', 'Very sparse.'),
    # --- offensive scheme: A'. reported (Wikipedia infobox; LOW confidence, not verified) ---
    ('scheme_rep_final_known', 'scheme_reported', '1 if the final season has a reported (Wikipedia infobox) family label.', 'Missingness is about Wikipedia editing, not the offense.'),
    ('scheme_rep_final_spread', 'scheme_reported', 'Final-season reported family includes spread.', 'Low confidence; uncited infobox text.'),
    ('scheme_rep_final_air_raid', 'scheme_reported', 'Final-season reported family includes air raid.', 'Low confidence.'),
    ('scheme_rep_final_pro_style', 'scheme_reported', 'Final-season reported family includes pro-style.', 'Low confidence.'),
    ('scheme_rep_final_spread_option', 'scheme_reported', 'Final-season reported family includes spread option.', 'Low confidence.'),
    ('scheme_rep_final_multiple_hybrid', 'scheme_reported', 'Final-season reported family includes multiple/hybrid.', 'Low confidence.'),
    ('scheme_rep_final_spread_family', 'scheme_reported', 'Final-season reported family is spread-family.', 'Low confidence.'),
    ('scheme_rep_career_label_coverage', 'scheme_reported', 'Share of career attempts thrown in seasons with a reported family label.', None),
    ('scheme_rep_career_spread_family_share', 'scheme_reported', 'Attempt-weighted share of labelled career seasons in spread-family offenses (exposure).', 'Computed over labelled seasons only.'),
    ('scheme_rep_career_pro_style_share', 'scheme_reported', 'Attempt-weighted share of labelled career seasons with pro-style in the family.', 'Computed over labelled seasons only.'),
    ('scheme_rep_n_distinct_families', 'scheme_reported', 'Distinct reported families across the college career (transfers / coordinator changes).', None),
    # --- offensive scheme: B. quantitative (statistics, NOT schemes) ---
    ('scheme_q_final_standard_down_pass_rate', 'scheme_quantitative', 'Final-season team pass rate on standard downs, non-garbage (CFBD play-by-play).', 'Play-by-play fetched for 2016-2025 so far; earlier seasons NaN until fetched. Sacks count as passes; scrambles count as rushes.'),
    ('scheme_q_final_passing_down_pass_rate', 'scheme_quantitative', 'Final-season team pass rate on passing downs, non-garbage.', 'As above.'),
    ('scheme_q_final_early_down_pass_rate', 'scheme_quantitative', 'Final-season team pass rate on 1st/2nd down, non-garbage.', 'As above.'),
    ('scheme_q_final_pass_rate_nongarbage', 'scheme_quantitative', 'Final-season team pass rate, all downs, non-garbage.', 'As above.'),
    ('scheme_q_career_standard_down_pass_rate', 'scheme_quantitative', 'Attempt-weighted career standard-down pass rate over seasons with play-by-play.', 'As above.'),
    # --- combine --------------------------------------------------------------
    ('combine_record_found', 'athletic_combine', '1 if a combine record exists for the draft year.', 'Absence = not invited OR not recorded; not a poor result.'),
    ('combine_height_inches', 'athletic_combine', 'Combine-measured height (inches).', None),
    ('combine_weight_lbs', 'athletic_combine', 'Combine-measured weight (lbs).', None),
    ('combine_forty', 'athletic_combine', '40-yard dash (s).', 'Skipped drill = NaN, never imputed as slow.'),
    ('combine_bench', 'athletic_combine', 'Bench press reps (225 lb).', 'QBs rarely bench -> mostly NaN.'),
    ('combine_vertical', 'athletic_combine', 'Vertical jump (in).', None),
    ('combine_broad_jump', 'athletic_combine', 'Broad jump (in).', None),
    ('combine_cone', 'athletic_combine', 'Three-cone drill (s).', None),
    ('combine_shuttle', 'athletic_combine', '20-yard shuttle (s).', None),
    ('combine_did_forty', 'athletic_combine', '1 if 40 recorded (given a combine record).', None),
    ('combine_did_bench', 'athletic_combine', '1 if bench recorded.', None),
    ('combine_did_vertical', 'athletic_combine', '1 if vertical recorded.', None),
    ('combine_did_broad_jump', 'athletic_combine', '1 if broad jump recorded.', None),
    ('combine_did_cone', 'athletic_combine', '1 if three-cone recorded.', None),
    ('combine_did_shuttle', 'athletic_combine', '1 if shuttle recorded.', None),
    # --- scouting ---------------------------------------------------------------
    ('scout_pre_draft_ranking', 'scouting', 'CFBD pre-draft overall ranking.', 'Expert evaluation; encodes market expectation.'),
    ('scout_pre_draft_position_ranking', 'scouting', 'CFBD pre-draft position ranking.', None),
    ('scout_pre_draft_grade', 'scouting', 'CFBD pre-draft grade.', None),
    # --- draft slot (post-draft ONLY) ----------------------------------------------
    ('draft_round', 'draft_position', 'NFL draft round.', 'Post-draft information; never in pre-draft models.'),
    ('draft_overall_pick', 'draft_position', 'NFL draft overall pick number.', 'Post-draft information; never in pre-draft models.'),
]


@dataclass(frozen=True)
class FeatureDef:
    name: str
    group: str
    description: str
    caveats: str | None
    version: str = FEATURE_VERSION

    @property
    def min_scope(self) -> str:
        return GROUP_SCOPE[self.group]

    @property
    def scopes(self) -> list[str]:
        return SCOPES[SCOPES.index(self.min_scope):]

    @property
    def availability_rule(self) -> str:
        return GROUP_AVAILABILITY[self.group]

    @property
    def source_ids(self) -> str:
        return GROUP_SOURCES[self.group]


FEATURES: dict[str, FeatureDef] = {n: FeatureDef(n, g, d, c) for n, g, d, c in _SPECS}
assert len(FEATURES) == len(_SPECS), 'duplicate feature name in registry'


def features_for_scope(scope: str) -> list[str]:
    if scope not in SCOPES:
        raise ValueError(f'unknown scope {scope!r}; expected one of {SCOPES}')
    return [n for n, f in FEATURES.items() if scope in f.scopes]


def is_forbidden(name: str) -> bool:
    return name.startswith(FORBIDDEN_FEATURE_PREFIXES)


def assert_scope_safe(columns, scope: str) -> None:
    """Raises if any column is unregistered, outside `scope`, or an outcome."""
    allowed = set(features_for_scope(scope))
    bad_outcome = [c for c in columns if is_forbidden(c) and c not in allowed]
    if bad_outcome:
        raise ValueError(f'outcome/target columns cannot be features: {bad_outcome}')
    unknown = [c for c in columns if c not in FEATURES]
    if unknown:
        raise ValueError(f'unregistered feature columns: {unknown}')
    outside = [c for c in columns if c not in allowed]
    if outside:
        raise ValueError(f'columns not allowed in scope {scope!r}: {outside}')


def registry_records() -> list[dict]:
    return [{'feature_name': f.name, 'feature_version': f.version, 'feature_group': f.group,
             'description': f.description, 'source_ids': f.source_ids, 'availability_rule': f.availability_rule,
             'model_scopes': ','.join(f.scopes), 'caveats': f.caveats} for f in FEATURES.values()]
