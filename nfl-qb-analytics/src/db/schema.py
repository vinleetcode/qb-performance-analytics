"""SQLAlchemy schema for the all-time (1920-2025) QB dataset.

Designed to run unchanged against SQLite (default, used in this environment)
or PostgreSQL (set DATABASE_URL, e.g. postgresql+psycopg2://user:pass@host/db).
No dialect-specific SQL is used; idempotent loads are done with a
delete-then-insert-per-batch pattern in src/historical_pipeline.py rather than
dialect-specific upserts, so the same code path works on both engines.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String,
    UniqueConstraint, create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = ROOT / 'data' / 'qb_history.db'


class Base(DeclarativeBase):
    pass


class Source(Base):
    """Registry of data sources this pipeline knows about (see config/sources.json)."""
    __tablename__ = 'sources'
    source_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    url = Column(String)
    access_method = Column(String, nullable=False)
    license_notes = Column(String)
    coverage_start_season = Column(Integer)
    coverage_end_season = Column(Integer)
    status = Column(String, nullable=False)  # ingested | not_ingested | partial
    status_reason = Column(String)
    verified_on = Column(String)


class League(Base):
    __tablename__ = 'leagues'
    league_id = Column(String, primary_key=True)  # 'APFA' | 'NFL' | 'AFL'
    name = Column(String, nullable=False)
    start_season = Column(Integer, nullable=False)
    end_season = Column(Integer)  # null = still active / merged forward under same id
    notes = Column(String)


class Team(Base):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True, autoincrement=True)
    league_id = Column(String, ForeignKey('leagues.league_id'), nullable=False)
    abbr = Column(String, nullable=False)
    name = Column(String)
    __table_args__ = (UniqueConstraint('league_id', 'abbr', name='uq_team_league_abbr'),)


class Player(Base):
    """Canonical player identity. For now keyed 1:1 to the nflverse gsis id, with a
    nullable pfr_player_id reserved for future cross-source identity resolution once a
    legitimate historical (PFR) source is ingested. Distinct humans with identical
    display names always get distinct rows because the key is the source player id,
    never the name.
    """
    __tablename__ = 'players'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    source_player_id = Column(String, nullable=False)
    pfr_player_id = Column(String)  # reserved for future cross-source linking
    display_name = Column(String, nullable=False)
    birthdate = Column(Date)
    __table_args__ = (UniqueConstraint('source_id', 'source_player_id', name='uq_player_source_id'),)

    seasons = relationship('PlayerSeasonRaw', back_populates='player')


class PlayerSeasonRaw(Base):
    """Raw, as-sourced player-season passing record. Kept separate from derived stats."""
    __tablename__ = 'player_seasons_raw'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey('players.id'), nullable=False)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    season = Column(Integer, nullable=False)
    league_id = Column(String, ForeignKey('leagues.league_id'), nullable=False)
    season_type = Column(String, nullable=False, default='REG')
    position_listed = Column(String)  # raw position string from the source; not forced to 'QB'
    is_qb_listed = Column(Boolean, nullable=False, default=False)
    primary_team = Column(String)  # most recent team that season; full history in PlayerSeasonTeamStint
    games = Column(Integer)
    completions = Column(Integer)
    attempts = Column(Integer)
    passing_yards = Column(Integer)
    passing_tds = Column(Integer)
    interceptions = Column(Integer)
    sacks_suffered = Column(Float)
    sack_yards_lost = Column(Float)
    carries = Column(Integer)
    rushing_yards = Column(Integer)
    rushing_tds = Column(Integer)
    rushing_fumbles_lost = Column(Integer)
    completeness_flag = Column(String, nullable=False, default='complete')  # complete | partial | unofficial
    completeness_notes = Column(String)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint('player_id', 'season', 'season_type', 'source_id', name='uq_player_season_source'),
    )

    player = relationship('Player', back_populates='seasons')
    team_stints = relationship('PlayerSeasonTeamStint', back_populates='player_season', cascade='all, delete-orphan')
    derived = relationship('PlayerSeasonDerived', back_populates='player_season', uselist=False, cascade='all, delete-orphan')
    era_stats = relationship('EraAdjustedStat', back_populates='player_season', cascade='all, delete-orphan')


class PlayerSeasonTeamStint(Base):
    """Team-by-team breakdown within a season, so multi-team seasons are never
    double-counted in totals but team history is still preserved."""
    __tablename__ = 'player_season_team_stints'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_season_id = Column(Integer, ForeignKey('player_seasons_raw.id'), nullable=False)
    team_abbr = Column(String, nullable=False)
    games = Column(Integer)
    attempts = Column(Integer)
    __table_args__ = (UniqueConstraint('player_season_id', 'team_abbr', name='uq_stint_player_season_team'),)

    player_season = relationship('PlayerSeasonRaw', back_populates='team_stints')


class PlayerSeasonDerived(Base):
    """Standardized efficiency metrics computed from the raw record. 1:1 with PlayerSeasonRaw."""
    __tablename__ = 'player_seasons_derived'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_season_id = Column(Integer, ForeignKey('player_seasons_raw.id'), nullable=False, unique=True)
    completion_pct = Column(Float)
    yards_per_attempt = Column(Float)
    td_pct = Column(Float)
    int_pct = Column(Float)
    adjusted_yards_per_attempt = Column(Float)  # AY/A (no sacks)
    adjusted_net_yards_per_attempt = Column(Float)  # ANY/A (requires sacks)
    passer_rating = Column(Float)
    yards_per_carry = Column(Float)
    rushing_td_pct = Column(Float)  # rushing TDs per carry, as a percentage
    total_yards = Column(Float)  # passing + rushing
    total_tds = Column(Float)  # passing + rushing
    yards_per_play = Column(Float)  # total_yards / (attempts + carries)
    qualifies_leaderboard = Column(Boolean, nullable=False, default=False)
    qualification_threshold_attempts = Column(Integer)

    player_season = relationship('PlayerSeasonRaw', back_populates='derived')


class EraAdjustedStat(Base):
    """Season/league-relative (or rolling-era) z-score and percentile for one metric."""
    __tablename__ = 'era_adjusted_stats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_season_id = Column(Integer, ForeignKey('player_seasons_raw.id'), nullable=False)
    metric = Column(String, nullable=False)  # e.g. 'yards_per_attempt', 'passer_rating'
    baseline_method = Column(String, nullable=False)  # 'season_league' | 'rolling_5yr' | ...
    baseline_group_size = Column(Integer)
    baseline_mean = Column(Float)
    baseline_sd = Column(Float)
    z_score = Column(Float)
    percentile = Column(Float)

    player_season = relationship('PlayerSeasonRaw', back_populates='era_stats')

    __table_args__ = (
        UniqueConstraint('player_season_id', 'metric', 'baseline_method', name='uq_era_stat'),
    )


class SeasonCalendar(Base):
    """Reference metadata (schedule length), not player statistics. See config/season_calendar.json."""
    __tablename__ = 'season_calendar'
    id = Column(Integer, primary_key=True, autoincrement=True)
    league_id = Column(String, ForeignKey('leagues.league_id'), nullable=False)
    season = Column(Integer, nullable=False)
    games_scheduled = Column(Integer)
    confidence = Column(String, nullable=False)
    notes = Column(String)
    __table_args__ = (UniqueConstraint('league_id', 'season', name='uq_calendar_league_season'),)


class CoverageReport(Base):
    """Persisted snapshot of what was actually ingested, by league/season/source."""
    __tablename__ = 'coverage_report'
    id = Column(Integer, primary_key=True, autoincrement=True)
    league_id = Column(String, nullable=False)
    season = Column(Integer, nullable=False)
    source_id = Column(String, nullable=False)
    status = Column(String, nullable=False)  # ingested | not_ingested | partial
    player_season_count = Column(Integer, nullable=False, default=0)
    reason = Column(String)
    generated_at = Column(DateTime, server_default=func.now())
    __table_args__ = (UniqueConstraint('league_id', 'season', 'source_id', name='uq_coverage_league_season_source'),)


class CfbPlayer(Base):
    """Canonical college-football player identity, keyed by CFBD's own playerId
    (stable within CFBD's data, confirmed via /stats/player/season -- NOT the
    same id space as nflverse draft_picks.cfb_player_id, which is a
    sports-reference.com/cfb slug; see CollegeNflPlayerMap for how the two are
    reconciled by name/school matching instead)."""
    __tablename__ = 'cfb_players'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    cfbd_player_id = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint('source_id', 'cfbd_player_id', name='uq_cfb_player_source_id'),)

    seasons = relationship('CfbPlayerSeason', back_populates='player')


class CfbPlayerSeason(Base):
    """Raw-ish college player-season passing/rushing record, aggregated from
    CFBD's long-format /stats/player/season rows. Kept separate from any
    derived/standardized metrics, same pattern as PlayerSeasonRaw/PlayerSeasonDerived."""
    __tablename__ = 'cfb_player_seasons'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey('cfb_players.id'), nullable=False)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    season = Column(Integer, nullable=False)
    season_type = Column(String, nullable=False, default='regular')
    school = Column(String)
    conference = Column(String)
    position_listed = Column(String)
    completions = Column(Integer)
    attempts = Column(Integer)
    passing_yards = Column(Integer)
    passing_tds = Column(Integer)
    interceptions = Column(Integer)
    rushing_attempts = Column(Integer)
    rushing_yards = Column(Integer)
    rushing_tds = Column(Integer)
    height_inches = Column(Float)  # bio backfill, from /roster; null if unavailable
    weight_lbs = Column(Integer)
    class_year = Column(String)  # left null when CFBD doesn't expose it -- never fabricated
    completeness_flag = Column(String, nullable=False, default='complete')
    completeness_notes = Column(String)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint('player_id', 'season', 'season_type', 'source_id', name='uq_cfb_player_season_source'),
    )

    player = relationship('CfbPlayer', back_populates='seasons')
    derived = relationship('CfbPlayerSeasonDerived', back_populates='cfb_season', uselist=False,
                            cascade='all, delete-orphan')


class CfbPlayerSeasonDerived(Base):
    """Standardized college efficiency metrics, derived from CfbPlayerSeason."""
    __tablename__ = 'cfb_player_seasons_derived'
    id = Column(Integer, primary_key=True, autoincrement=True)
    cfb_player_season_id = Column(Integer, ForeignKey('cfb_player_seasons.id'), nullable=False, unique=True)
    completion_pct = Column(Float)
    yards_per_attempt = Column(Float)
    td_pct = Column(Float)
    int_pct = Column(Float)
    ncaa_passer_rating = Column(Float)
    yards_per_carry = Column(Float)

    cfb_season = relationship('CfbPlayerSeason', back_populates='derived')


class CfbTeamSeasonContext(Base):
    """Team-season context (SP+ rating/ranking, strength of schedule) from CFBD's
    /ratings/sp. Team-level, not player-level -- joined to player-seasons by
    (season, school), never duplicated onto every player row in storage."""
    __tablename__ = 'cfb_team_season_context'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    season = Column(Integer, nullable=False)
    school = Column(String, nullable=False)
    conference = Column(String)
    sp_rating = Column(Float)
    sp_ranking = Column(Integer)
    strength_of_schedule = Column(Float)
    __table_args__ = (UniqueConstraint('season', 'school', 'source_id', name='uq_team_season_context'),)


class NflDraftPickRaw(Base):
    """Original (unmodified beyond column renaming) NFL draft-pick record from
    nflverse load_draft_picks, ALL positions -- kept separate from the cleaned
    QB-only prospect table. Includes nflverse's own bundled NFL career totals
    (games/pass_*/rush_*/allpro/etc.) -- these are NFL OUTCOME fields baked into
    the draft record; downstream pre-draft feature exports must exclude them
    (see prospect_pipeline.export_parquet's leakage-safety split)."""
    __tablename__ = 'nfl_draft_picks_raw'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    season = Column(Integer, nullable=False)
    round = Column(Integer)
    pick = Column(Integer, nullable=False)
    team = Column(String)
    gsis_id = Column(String)
    pfr_player_id = Column(String)
    cfb_player_id_slug = Column(String)  # sports-reference.com/cfb slug, NOT a CFBD id
    pfr_player_name = Column(String)
    position_listed = Column(String)
    college = Column(String)
    age_at_draft = Column(Float)
    hof = Column(Boolean)
    career_last_season = Column(Integer)  # nflverse's "to" column
    allpro = Column(Integer)
    probowls = Column(Integer)
    seasons_started = Column(Integer)
    career_av = Column(Float)
    nfl_games_played_career = Column(Integer)  # nflverse's "games" column (NFL outcome)
    career_pass_completions = Column(Integer)
    career_pass_attempts = Column(Integer)
    career_pass_yards = Column(Integer)
    career_pass_tds = Column(Integer)
    career_pass_ints = Column(Integer)
    career_rush_attempts = Column(Integer)
    career_rush_yards = Column(Integer)
    career_rush_tds = Column(Integer)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint('season', 'pick', 'source_id', name='uq_draft_pick_season_pick'),)


class QbDraftProspect(Base):
    """Cleaned, QB-only view of a draft pick -- pre-draft bio/identity fields
    only (no NFL career-outcome columns), suitable as a leak-free feature row
    once joined to college stats. 1:1 with NflDraftPickRaw."""
    __tablename__ = 'qb_draft_prospects'
    id = Column(Integer, primary_key=True, autoincrement=True)
    draft_pick_raw_id = Column(Integer, ForeignKey('nfl_draft_picks_raw.id'), nullable=False, unique=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    draft_year = Column(Integer, nullable=False)
    round = Column(Integer)
    pick = Column(Integer, nullable=False)
    nfl_team = Column(String)
    player_name = Column(String, nullable=False)
    gsis_id = Column(String)  # null for picks nflverse never linked to a gsis id
    pfr_player_id = Column(String)
    cfb_player_id_slug = Column(String)
    college = Column(String)
    position_listed = Column(String)
    height_inches = Column(Float)  # backfilled from nflreadpy load_players(), when available
    weight_lbs = Column(Integer)
    birthdate = Column(Date)

    __table_args__ = (UniqueConstraint('draft_pick_raw_id', name='uq_qb_prospect_draft_pick'),)


class CollegeNflPlayerMap(Base):
    """Reproducible identity mapping: QB draft prospect -> existing NFL `players`
    row (stable gsis_id join) and -> CfbPlayer (name/school matched, since the
    two systems don't share an id space). One row per prospect. Ambiguous or
    failed matches are recorded explicitly, never silently guessed."""
    __tablename__ = 'college_nfl_player_map'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False, unique=True)
    nfl_player_id = Column(Integer, ForeignKey('players.id'))  # null if no NFL passing-stat row exists
    cfb_player_id = Column(Integer, ForeignKey('cfb_players.id'))  # null if no CFB match found
    nfl_match_method = Column(String)  # 'stable_gsis_id' | 'no_gsis_id' | 'no_nfl_passing_record'
    cfb_match_method = Column(String)  # 'exact_name_school' | 'fuzzy_name_school' | 'ambiguous' | 'unmatched'
    cfb_match_confidence = Column(Float)  # 0-1 similarity score; null for exact/stable matches
    match_status = Column(String, nullable=False)  # 'matched' | 'ambiguous' | 'unmatched'
    notes = Column(String)

    __table_args__ = (UniqueConstraint('qb_draft_prospect_id', name='uq_map_prospect'),)


# ==========================================================================
# Phase 3 -- advanced college context, athletic profile, NFL outcomes and
# modelling-preparation tables. NEW tables only; nothing above is altered.
#
# Layering (see README "Phase 3"):
#   raw/standardized source rows : cfb_games, cfb_game_passing, cfb_team_season_stats,
#                                  cfb_recruits, cfb_receiving_seasons, cfbd_draft_picks,
#                                  nfl_combine_measurements, cfb_offensive_scheme_labels
#   derived team-season context  : cfb_team_roster_talent, cfb_team_season_environment
#   joined player-level          : prospect_college_seasons, nfl_prospect_seasons
#   model-ready features         : feature_definitions, prospect_feature_values,
#                                  prospect_similarity
#   outcome / target tables      : nfl_career_outcomes, survival_records,
#                                  draft_value_predictions
# ==========================================================================


class CfbGame(Base):
    """One row per CFBD game (schedule + final score). Used for team schedules,
    games-scheduled denominators, and season end dates (feature availability)."""
    __tablename__ = 'cfb_games'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    game_id = Column(Integer, nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer)
    season_type = Column(String)
    start_date = Column(String)  # ISO-8601 UTC as returned by CFBD
    completed = Column(Boolean)
    neutral_site = Column(Boolean)
    conference_game = Column(Boolean)
    home_team = Column(String)
    home_conference = Column(String)
    home_classification = Column(String)
    home_points = Column(Integer)
    away_team = Column(String)
    away_conference = Column(String)
    away_classification = Column(String)
    away_points = Column(Integer)
    __table_args__ = (UniqueConstraint('game_id', 'source_id', name='uq_cfb_game'),)


class CfbGamePassing(Base):
    """Per-game passing box-score line (raw). `athlete_key` is the CFBD athlete id,
    or 'TEAM:<name>' for team pseudo-rows, which count against the defense but are
    never attributed to a quarterback (cfbd_player_id stays null)."""
    __tablename__ = 'cfb_game_passing'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    game_id = Column(Integer, nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer)
    season_type = Column(String)
    team = Column(String, nullable=False)
    conference = Column(String)
    home_away = Column(String)
    opponent = Column(String)
    opponent_conference = Column(String)
    athlete_key = Column(String, nullable=False)
    cfbd_player_id = Column(String)
    player_name = Column(String)
    completions = Column(Integer)
    attempts = Column(Integer)
    passing_yards = Column(Integer)
    passing_tds = Column(Integer)
    interceptions = Column(Integer)
    __table_args__ = (UniqueConstraint('game_id', 'team', 'athlete_key', 'source_id', name='uq_cfb_game_passing'),)


class CfbGameRushing(Base):
    """Per-game rushing box-score line (raw), fetched only for seasons where
    CFBD's season-level player stats are incomplete (2004-2008), so QB rushing
    can be reconstructed there instead of being left silently null."""
    __tablename__ = 'cfb_game_rushing'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    game_id = Column(Integer, nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer)
    season_type = Column(String)
    team = Column(String, nullable=False)
    opponent = Column(String)
    cfbd_player_id = Column(String, nullable=False)
    player_name = Column(String)
    carries = Column(Integer)
    rushing_yards = Column(Integer)
    rushing_tds = Column(Integer)
    __table_args__ = (UniqueConstraint('game_id', 'team', 'cfbd_player_id', 'source_id', name='uq_cfb_game_rushing'),)


class CfbTeamSeasonStats(Base):
    """Team-season box totals (/stats/season) + advanced play-by-play metrics
    (/stats/season/advanced). Advanced columns are only trustworthy when
    play-by-play coverage is near-complete -- see `pbp_coverage_ratio` and
    `advanced_reliable` (gate applied in src/features/environment.py)."""
    __tablename__ = 'cfb_team_season_stats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    season = Column(Integer, nullable=False)
    team = Column(String, nullable=False)
    conference = Column(String)
    games = Column(Integer)
    pass_attempts = Column(Float)
    pass_completions = Column(Float)
    net_passing_yards = Column(Float)
    passing_tds = Column(Float)
    interceptions_thrown = Column(Float)
    rushing_attempts = Column(Float)
    rushing_yards = Column(Float)
    rushing_tds = Column(Float)
    total_yards = Column(Float)
    possession_seconds = Column(Float)
    def_sacks = Column(Float)
    sacks_allowed = Column(Float)
    def_tfl = Column(Float)
    tfl_allowed = Column(Float)
    opp_pass_attempts = Column(Float)
    opp_pass_completions = Column(Float)
    opp_net_passing_yards = Column(Float)
    opp_passing_tds = Column(Float)
    def_interceptions = Column(Float)
    opp_rushing_attempts = Column(Float)
    opp_rushing_yards = Column(Float)
    off_plays = Column(Float)
    off_ppa = Column(Float)
    off_success_rate = Column(Float)
    off_explosiveness = Column(Float)
    off_power_success = Column(Float)
    off_stuff_rate = Column(Float)
    off_line_yards = Column(Float)
    off_havoc_allowed = Column(Float)
    off_front7_havoc_allowed = Column(Float)
    off_passing_play_rate = Column(Float)
    off_passing_ppa = Column(Float)
    off_passing_success_rate = Column(Float)
    off_rushing_play_rate = Column(Float)
    off_rushing_ppa = Column(Float)
    off_passing_downs_rate = Column(Float)
    def_plays = Column(Float)
    def_ppa = Column(Float)
    def_success_rate = Column(Float)
    def_passing_ppa = Column(Float)
    def_passing_success_rate = Column(Float)
    def_havoc = Column(Float)
    pbp_coverage_ratio = Column(Float)  # advanced off_plays / box-score (pass att + rush att)
    advanced_reliable = Column(Boolean, nullable=False, default=False)
    __table_args__ = (UniqueConstraint('season', 'team', 'source_id', name='uq_cfb_team_season_stats'),)


class CfbRecruit(Base):
    """Individual high-school recruit (247Sports composite via CFBD). `athlete_id`
    is CFBD's college athlete id when CFBD has linked it (same id space as
    cfb_players.cfbd_player_id), otherwise null -- never guessed by name."""
    __tablename__ = 'cfb_recruits'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    recruit_id = Column(String, nullable=False)
    athlete_id = Column(String)
    recruit_type = Column(String)
    class_year = Column(Integer, nullable=False)
    name = Column(String)
    committed_to = Column(String)
    position = Column(String)
    stars = Column(Integer)
    rating = Column(Float)
    national_ranking = Column(Integer)
    __table_args__ = (UniqueConstraint('recruit_id', 'source_id', name='uq_cfb_recruit'),)


class CfbReceivingSeason(Base):
    """Player-season receiving totals (raw), used ONLY to describe how a team's
    receiving production is distributed (concentration) and to identify top
    targets for exogenous talent lookups -- never as a stand-in for QB ability."""
    __tablename__ = 'cfb_receiving_seasons'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    season = Column(Integer, nullable=False)
    cfbd_player_id = Column(String, nullable=False)
    player_name = Column(String)
    position = Column(String)
    team = Column(String, nullable=False)
    conference = Column(String)
    receptions = Column(Float)
    receiving_yards = Column(Float)
    receiving_tds = Column(Float)
    __table_args__ = (UniqueConstraint('season', 'cfbd_player_id', 'team', 'source_id', name='uq_cfb_receiving'),)


class CfbdDraftPick(Base):
    """CFBD's own NFL-draft record (raw). Its `college_athlete_id` is a stable
    CFBD id, so (year, overall) gives an id-based bridge from the nflverse draft
    board to cfb_players that does not depend on name spelling. Pre-draft
    rankings/grades are expert evaluations: legitimately pre-draft, but kept out
    of the college-only feature set (see feature registry group 'scouting')."""
    __tablename__ = 'cfbd_draft_picks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    year = Column(Integer, nullable=False)
    overall = Column(Integer, nullable=False)
    round = Column(Integer)
    pick = Column(Integer)
    name = Column(String)
    position = Column(String)
    college_athlete_id = Column(String)
    nfl_athlete_id = Column(String)
    college_team = Column(String)
    college_conference = Column(String)
    nfl_team = Column(String)
    height = Column(Float)
    weight = Column(Float)
    pre_draft_ranking = Column(Integer)
    pre_draft_position_ranking = Column(Integer)
    pre_draft_grade = Column(Float)
    __table_args__ = (UniqueConstraint('year', 'overall', 'source_id', name='uq_cfbd_draft_pick'),)


class NflCombineMeasurement(Base):
    """NFL Scouting Combine results (nflverse load_combine, sourced from PFR's
    combine pages and redistributed by nflverse). A null drill = not recorded
    (did not participate / not measured) -- never imputed as a poor result.
    No exact per-player test date exists in the source, so the date is stored
    at season precision with the combine's documented window."""
    __tablename__ = 'nfl_combine_measurements'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey('sources.source_id'), nullable=False)
    combine_season = Column(Integer, nullable=False)
    pfr_id = Column(String)
    cfb_id_slug = Column(String)
    player_name = Column(String, nullable=False)
    position = Column(String)
    school = Column(String)
    height_inches = Column(Float)
    weight_lbs = Column(Float)
    forty = Column(Float)
    bench = Column(Float)
    vertical = Column(Float)
    broad_jump = Column(Float)
    cone = Column(Float)
    shuttle = Column(Float)
    draft_year = Column(Integer)
    draft_overall = Column(Integer)
    measurement_event = Column(String, nullable=False, default='nfl_scouting_combine')
    measurement_date_precision = Column(String, nullable=False, default='season')
    measurement_window = Column(String)
    __table_args__ = (UniqueConstraint('combine_season', 'player_name', 'school', 'position', 'source_id',
                                       name='uq_combine_row'),)


class CfbOffensiveSchemeLabel(Base):
    """One row per (team, season, dimension, label, source) scheme annotation, with full
    provenance. verification_status: verified (manual review; the source explicitly ties
    the label to that team AND season) | reported (Wikipedia infobox, low confidence) |
    conflicting | insufficient_evidence | not_supported | source_unavailable |
    searched_no_candidate_source. Labels may overlap; nothing is derived from statistics.
    (v2 schema, 2026-11-09; the v1 table was empty and is migrated by advanced_pipeline.)"""
    __tablename__ = 'cfb_offensive_scheme_labels'
    id = Column(Integer, primary_key=True, autoincrement=True)
    team = Column(String, nullable=False)
    season = Column(Integer, nullable=False)
    scheme_dimension = Column(String, nullable=False)
    scheme_label = Column(String, nullable=False)
    verification_status = Column(String, nullable=False)
    confidence = Column(String)
    source_type = Column(String)
    source_url = Column(String)
    source_title = Column(String)
    publication_date = Column(String)
    evidence_summary = Column(String)
    annotation_method = Column(String, nullable=False)
    raw_text = Column(String)
    notes = Column(String)
    __table_args__ = (UniqueConstraint('team', 'season', 'scheme_dimension', 'scheme_label', 'source_url',
                                       name='uq_scheme_label_v2'),)


class CfbTeamSeasonScheme(Base):
    """Per relevant team-season: research status and labels by evidence tier, plus
    play-by-play tendencies (statistics, never labels). Pipe-separated label lists."""
    __tablename__ = 'cfb_team_season_schemes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    team = Column(String, nullable=False)
    season = Column(Integer, nullable=False)
    taxonomy_version = Column(String, nullable=False)
    manual_research_status = Column(String, nullable=False)
    reported_label_status = Column(String, nullable=False)
    verified_offensive_family = Column(String)
    verified_run_concepts = Column(String)
    verified_passing_concepts = Column(String)
    verified_formation_tendency = Column(String)
    verified_tempo = Column(String)
    verified_spread_family = Column(Integer)
    reported_offensive_family = Column(String)
    reported_run_concepts = Column(String)
    reported_passing_concepts = Column(String)
    reported_formation_tendency = Column(String)
    reported_tempo = Column(String)
    reported_spread_family = Column(Integer)
    wikipedia_off_scheme_raw = Column(String)
    pbp_plays_nongarbage = Column(Integer)
    pbp_reliable = Column(Boolean)
    pass_rate_nongarbage = Column(Float)
    standard_down_pass_rate = Column(Float)
    passing_down_pass_rate = Column(Float)
    early_down_pass_rate = Column(Float)
    standard_down_share = Column(Float)
    plays_per_game_nongarbage = Column(Float)
    __table_args__ = (UniqueConstraint('team', 'season', name='uq_team_season_scheme'),)


class CfbTeamRosterTalent(Base):
    """Team-season roster talent: 247 Team Talent Composite (2015+), recruiting
    class points (current + trailing 4-class average), and returning production
    (known before the season). Percentiles are within-season, so ratings from
    different years are comparable."""
    __tablename__ = 'cfb_team_roster_talent'
    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(Integer, nullable=False)
    team = Column(String, nullable=False)
    feature_version = Column(String, nullable=False)
    talent_composite = Column(Float)
    talent_composite_pctile = Column(Float)
    recruiting_points_class = Column(Float)
    recruiting_rank_class = Column(Integer)
    recruiting_points_4yr_avg = Column(Float)
    recruiting_4yr_classes_found = Column(Integer)
    recruiting_4yr_pctile = Column(Float)
    returning_ppa_pct = Column(Float)
    returning_passing_ppa_pct = Column(Float)
    returning_receiving_ppa_pct = Column(Float)
    returning_rushing_ppa_pct = Column(Float)
    returning_usage = Column(Float)
    returning_receiving_usage = Column(Float)
    __table_args__ = (UniqueConstraint('season', 'team', name='uq_team_roster_talent'),)


class CfbTeamSeasonEnvironment(Base):
    """Derived team-season offensive environment: quantitative scheme profile,
    offensive-line proxies and receiver-corps context. Each block is a PROXY:
    sack rate reflects QB decisions and scheme as well as pass protection, and
    team receiving volume is mostly the QB's own production (circular) -- the
    non-circular receiver measures are concentration and exogenous recruiting
    ratings."""
    __tablename__ = 'cfb_team_season_environment'
    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(Integer, nullable=False)
    team = Column(String, nullable=False)
    feature_version = Column(String, nullable=False)
    # scheme (quantitative)
    plays_per_game = Column(Float)
    pass_rate = Column(Float)
    rush_rate = Column(Float)
    seconds_per_play = Column(Float)
    sp_pace = Column(Float)
    sp_run_rate = Column(Float)
    off_ppa = Column(Float)
    off_success_rate = Column(Float)
    sp_offense_rating = Column(Float)
    # offensive line proxies
    sacks_allowed = Column(Float)
    sack_rate_allowed = Column(Float)
    opp_def_sack_rate = Column(Float)
    opp_adj_sack_rate = Column(Float)
    team_rush_ypc = Column(Float)
    off_stuff_rate = Column(Float)
    off_line_yards = Column(Float)
    off_havoc_allowed = Column(Float)
    # receivers
    team_receiving_yards = Column(Float)
    top_target_reception_share = Column(Float)
    top3_reception_share = Column(Float)
    top3_receivers_avg_recruit_rating = Column(Float)
    top3_receivers_rated = Column(Integer)
    # pass defense (this team's defense; used as opponent context)
    def_pass_ypa_allowed = Column(Float)
    def_pass_cmp_pct_allowed = Column(Float)
    def_pass_rank = Column(Integer)
    __table_args__ = (UniqueConstraint('season', 'team', name='uq_team_season_environment'),)


class ProspectCollegeSeason(Base):
    """Joined QB-season table: one row per (drafted QB prospect, college season)
    observed in CFBD. Season totals come from cfb_player_seasons where CFBD has
    them, otherwise they are aggregated from box scores (season_stats_source);
    per-game measures (games, opponent adjustment, top-defense splits) always
    come from box scores. Transfers keep each season attached to the school
    played for that season. identity_source records HOW the prospect was linked
    to this CFBD athlete id (Phase-2 map, CFBD draft id, or box-score name match)."""
    __tablename__ = 'prospect_college_seasons'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False)
    cfb_player_id = Column(Integer, ForeignKey('cfb_players.id'))  # null when linked only via box scores
    cfbd_player_id = Column(String, nullable=False)
    identity_source = Column(String, nullable=False)
    season_stats_source = Column(String, nullable=False)  # cfbd_season_stats | box_score_aggregate
    season = Column(Integer, nullable=False)
    feature_version = Column(String, nullable=False)
    team = Column(String)
    conference = Column(String)
    seasons_before_draft = Column(Integer)  # draft_year - season (1 = final pre-draft season)
    is_final_college_season = Column(Boolean)
    available_as_of = Column(Date)  # last game date of that season (data-driven)
    attempts = Column(Float)
    completions = Column(Float)
    passing_yards = Column(Float)
    passing_tds = Column(Float)
    interceptions = Column(Float)
    rushing_attempts = Column(Float)
    rushing_yards = Column(Float)
    rushing_tds = Column(Float)
    completion_pct = Column(Float)
    yards_per_attempt = Column(Float)
    td_pct = Column(Float)
    int_pct = Column(Float)
    ncaa_passer_rating = Column(Float)
    qb_rush_share = Column(Float)
    box_attempts = Column(Float)
    box_season_attempt_ratio = Column(Float)
    team_box_score_coverage = Column(Float)  # share of team's games with box scores; < 0.8 -> per-game fields NULL
    games_with_pass_attempt = Column(Integer)
    games_as_leading_passer = Column(Integer)
    team_games_played = Column(Integer)
    participation_rate = Column(Float)
    opp_adj_ypa = Column(Float)
    opp_adj_completion_pct = Column(Float)
    opp_adj_td_pct = Column(Float)
    opp_adj_int_pct = Column(Float)
    opp_adj_attempt_coverage = Column(Float)
    sos_pass_defense = Column(Float)  # attempt-weighted opponent YPA allowed minus FBS mean (neg = tougher)
    sp_strength_of_schedule = Column(Float)
    opp_def_sp_rating_avg = Column(Float)
    conf_rel_rating_z = Column(Float)
    conf_rel_ypa_z = Column(Float)
    conf_rel_rating_pctile = Column(Float)
    vs_top_def_games = Column(Integer)
    vs_top_def_attempts = Column(Float)
    vs_top_def_ypa = Column(Float)
    vs_top_def_completion_pct = Column(Float)
    vs_top_def_rating = Column(Float)
    __table_args__ = (UniqueConstraint('qb_draft_prospect_id', 'season', name='uq_prospect_college_season'),)


class FeatureDefinition(Base):
    """Versioned feature registry (source of truth: src/features/registry.py).
    `availability_rule` states when the value becomes knowable; `model_scopes`
    lists the model families allowed to use it (e.g. draft slot is post_draft
    only; outcomes are target only)."""
    __tablename__ = 'feature_definitions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, nullable=False)
    feature_version = Column(String, nullable=False)
    feature_group = Column(String, nullable=False)
    description = Column(String, nullable=False)
    source_ids = Column(String)
    availability_rule = Column(String, nullable=False)
    model_scopes = Column(String, nullable=False)
    caveats = Column(String)
    __table_args__ = (UniqueConstraint('feature_name', 'feature_version', name='uq_feature_def'),)


class ProspectFeatureValue(Base):
    """Model-ready pre-draft feature values in long form (one row per prospect x
    feature x version). Missing values are simply absent rows -- never stored as
    0 -- and every row carries the date it became knowable."""
    __tablename__ = 'prospect_feature_values'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False)
    feature_name = Column(String, nullable=False)
    feature_version = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    available_as_of = Column(Date, nullable=False)
    __table_args__ = (UniqueConstraint('qb_draft_prospect_id', 'feature_name', 'feature_version',
                                       name='uq_prospect_feature'),)


class ProspectSimilarity(Base):
    """Nearest historical comparables, computed from standardized PRE-DRAFT
    features only, with the reference pool restricted to earlier draft classes
    (so a comp list never uses information from the prospect's own or later
    classes). NFL outcomes are never inputs."""
    __tablename__ = 'prospect_similarity'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False)
    comp_qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False)
    rank = Column(Integer, nullable=False)
    distance = Column(Float, nullable=False)
    n_features_compared = Column(Integer, nullable=False)
    method = Column(String, nullable=False)
    feature_version = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint('qb_draft_prospect_id', 'rank', 'method', 'feature_version',
                                       name='uq_prospect_similarity'),)


class NflProspectSeason(Base):
    """One row per (drafted QB, NFL season) from the draft year through the last
    completed season, INCLUDING seasons with zero appearances (explicit zeros,
    because nflverse 1999+ is complete for regular-season appearances)."""
    __tablename__ = 'nfl_prospect_seasons'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False)
    nfl_season = Column(Integer, nullable=False)
    season_index = Column(Integer, nullable=False)  # 1 = rookie season
    games_with_stats = Column(Integer, nullable=False, default=0)
    games_started = Column(Integer, nullable=False, default=0)  # REG, from nflverse schedules starting-QB ids
    postseason_starts = Column(Integer, nullable=False, default=0)
    pass_attempts = Column(Float)
    pass_completions = Column(Float)
    passing_yards = Column(Float)
    passing_tds = Column(Float)
    interceptions = Column(Float)
    sacks = Column(Float)
    sack_yards = Column(Float)
    __table_args__ = (UniqueConstraint('qb_draft_prospect_id', 'nfl_season', name='uq_nfl_prospect_season'),)


class NflCareerOutcome(Base):
    """Target table. Window-based targets are NULL (not 0) when the window is not
    fully observed, so incomplete careers are never labelled failures."""
    __tablename__ = 'nfl_career_outcomes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False, unique=True)
    target_version = Column(String, nullable=False)
    observation_end_season = Column(Integer, nullable=False)
    seasons_observed = Column(Integer, nullable=False)
    window_5yr_complete = Column(Boolean, nullable=False)
    appeared_any_game = Column(Boolean, nullable=False)
    started_any_game = Column(Boolean)  # NULL when 0 starts observed AND window incomplete
    started_1plus_5yr = Column(Boolean)
    started_16plus_5yr = Column(Boolean)
    started_32plus_5yr = Column(Boolean)
    started_64plus_5yr = Column(Boolean)
    games_started_5yr = Column(Float)
    pass_yards_5yr = Column(Float)
    pass_tds_5yr = Column(Float)
    pass_attempts_5yr = Column(Float)
    any_a_5yr = Column(Float)  # NULL unless window complete AND attempts >= threshold
    passer_rating_5yr = Column(Float)
    efficiency_min_attempts = Column(Integer, nullable=False)
    games_started_observed = Column(Integer, nullable=False)
    first_appearance_season = Column(Integer)
    last_appearance_season = Column(Integer)
    last_start_season = Column(Integer)
    seasons_to_final_appearance = Column(Integer)
    seasons_to_final_start = Column(Integer)
    career_span_seasons = Column(Integer)
    career_end_status = Column(String, nullable=False)  # active_or_censored | inferred_inactive | never_appeared_*
    nflverse_roster_status = Column(String)
    pfr_last_season = Column(Integer)


class SurvivalRecord(Base):
    """Survival-analysis rows (one per prospect x event definition). Time unit =
    NFL seasons since the draft (1 = rookie season). event_observed=False means
    right-censored at `duration_seasons`."""
    __tablename__ = 'survival_records'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False)
    event_name = Column(String, nullable=False)
    definition_version = Column(String, nullable=False)
    duration_seasons = Column(Float, nullable=False)
    event_observed = Column(Boolean, nullable=False)
    censor_reason = Column(String)
    event_season = Column(Integer)
    last_observed_season = Column(Integer, nullable=False)
    officially_confirmed = Column(Boolean)
    at_risk_population = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint('qb_draft_prospect_id', 'event_name', 'definition_version',
                                       name='uq_survival_record'),)


class DraftValuePrediction(Base):
    """Model predictions stored next to the draft decision, for draft-value and
    incremental-value analysis. Empty until models are trained (this phase
    deliberately trains none). `model_scope` must be 'pre_draft_college_only',
    'pre_draft_full' or 'post_draft'; only post_draft models may use draft slot."""
    __tablename__ = 'draft_value_predictions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    qb_draft_prospect_id = Column(Integer, ForeignKey('qb_draft_prospects.id'), nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    model_scope = Column(String, nullable=False)
    target_name = Column(String, nullable=False)
    split = Column(String)  # train | validation | test | censored_holdout
    predicted_value = Column(Float)
    prediction_lower = Column(Float)
    prediction_upper = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (UniqueConstraint('qb_draft_prospect_id', 'model_name', 'model_version', 'target_name',
                                       name='uq_draft_value_prediction'),)


def get_engine(db_url: str | None = None):
    """Returns a SQLAlchemy engine. Defaults to the project's SQLite file; set
    DATABASE_URL (e.g. postgresql+psycopg2://user:pass@host:5432/dbname) to target Postgres."""
    url = db_url or os.environ.get('DATABASE_URL') or f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connect_args = {'check_same_thread': False} if url.startswith('sqlite') else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine=None):
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None) -> Session:
    engine = engine or get_engine()
    return Session(engine)
