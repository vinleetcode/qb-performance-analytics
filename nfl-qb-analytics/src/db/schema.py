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
