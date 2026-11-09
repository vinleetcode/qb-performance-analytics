"""Reproducible identity matching: QB draft prospect -> NFL player (stable
gsis_id join, never fuzzy) and -> college-football player (name+school
matching, since nflverse's draft-record cfb_player_id is a sports-reference.com
slug, not a CollegeFootballData id -- the two systems don't share an id space).

Matching never silently guesses: a tie between multiple plausible CFB
candidates is recorded as 'ambiguous', not resolved by picking the first or
highest-scoring one.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
SCHOOL_ALIASES = {
    'st.': 'state', 'st': 'state',
    'univ.': 'university', 'univ': 'university',
}
# Whole-name aliases for schools nflverse and CFBD spell differently enough
# that token-level rules can't bridge them -- confirmed against real CFBD data
# (2026-10-30), not guessed. Left side: normalize_school's output for
# nflverse's spelling. Right side: normalize_school's output for CFBD's own
# spelling (which is what CFB player-season rows are actually keyed by).
WHOLE_SCHOOL_ALIASES = {
    'mississippi': 'ole miss',  # nflverse: "Mississippi"; CFBD: "Ole Miss"
    'north carolina state': 'nc state',  # nflverse: "North Carolina St."; CFBD: "NC State"
    'central florida': 'ucf',  # nflverse: "Central Florida"; CFBD: "UCF"
    # Added 2026-11-09 after checking CFBD's own team names in /games (Phase 3):
    'boston col': 'boston college',  # nflverse: "Boston Col."; CFBD: "Boston College"
    'connecticut': 'uconn',  # nflverse: "Connecticut"; CFBD: "UConn"
    'texas-el paso': 'utep',  # nflverse: "Texas-El Paso"; CFBD: "UTEP"
}


def _strip_diacritics(text: str) -> str:
    """"José" -> "jose", "Peña" -> "pena". Applied to both names and schools so
    an accented spelling on one side (CFBD was confirmed 2026-10-30 to spell
    "San José State" with the accent) matches an unaccented one on the other."""
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in decomposed if not unicodedata.combining(c))
# College transfers often show up in nflverse's `college` field as a
# semicolon/comma-separated list (e.g. "Wyoming; Reedley", "Miami; Georgia").
SCHOOL_SPLIT_RE = re.compile(r'[;,/]| and ')

FUZZY_ACCEPT_THRESHOLD = 0.90
AMBIGUITY_MARGIN = 0.05
SEASON_WINDOW_YEARS = 6  # draft_year - 6 .. draft_year - 1


def normalize_name(name: str) -> str:
    if not name:
        return ''
    cleaned = re.sub(r"[.']", '', _strip_diacritics(name.lower())).strip()
    tokens = [t for t in cleaned.split() if t not in SUFFIXES]
    return ' '.join(tokens)


def normalize_school(school: str) -> str:
    if not school:
        return ''
    cleaned = re.sub(r'\([^)]*\)', '', _strip_diacritics(school.lower()))  # drop "(FL)"-style parentheticals
    cleaned = re.sub(r"[.']", '', cleaned).strip()  # periods AND apostrophes (e.g. "Hawai'i" -> "hawaii")
    tokens = [SCHOOL_ALIASES.get(t, t) for t in cleaned.split()]
    tokens = [t for t in tokens if t not in ('the', 'university', 'of')]
    normalized = ' '.join(tokens)
    return WHOLE_SCHOOL_ALIASES.get(normalized, normalized)


def split_schools(college_field: str) -> list[str]:
    """nflverse's `college` can list multiple schools for a transfer; CFBD data
    is matched against each candidate school independently."""
    if not college_field:
        return []
    parts = SCHOOL_SPLIT_RE.split(college_field)
    return [normalize_school(p) for p in parts if p.strip()]


@dataclass
class CfbCandidate:
    cfb_player_id: int
    name: str
    schools: set  # normalized school names across all of this identity's seasons
    seasons: set  # int season years


@dataclass
class MatchResult:
    cfb_player_id: int | None
    method: str  # 'exact_name_school' | 'fuzzy_name_school' | 'ambiguous' | 'unmatched'
    confidence: float | None
    notes: str | None


def match_prospect_to_cfb(prospect_name: str, college_field: str, draft_year: int,
                           candidates: list[CfbCandidate]) -> MatchResult:
    target_name = normalize_name(prospect_name)
    target_schools = set(split_schools(college_field))
    if not target_name or not candidates:
        return MatchResult(None, 'unmatched', None, 'no CFB candidate data available')

    window = range(draft_year - SEASON_WINDOW_YEARS, draft_year)

    def school_overlaps(cand: CfbCandidate) -> bool:
        return bool(cand.schools & target_schools) if target_schools else True

    def season_overlaps(cand: CfbCandidate) -> bool:
        return bool(cand.seasons & set(window))

    in_window = [c for c in candidates if season_overlaps(c)]

    exact = [c for c in in_window if normalize_name_matches(c.name, target_name) and school_overlaps(c)]
    if len(exact) == 1:
        return MatchResult(exact[0].cfb_player_id, 'exact_name_school', 1.0, None)
    if len(exact) > 1:
        return MatchResult(None, 'ambiguous', 1.0,
                            f'{len(exact)} exact name+school candidates in {draft_year - SEASON_WINDOW_YEARS}-{draft_year - 1}')

    # Fall back to fuzzy name matching, still constrained to a school overlap.
    scored = []
    for c in in_window:
        if not school_overlaps(c):
            continue
        score = SequenceMatcher(None, target_name, normalize_name(c.name)).ratio()
        if score >= FUZZY_ACCEPT_THRESHOLD:
            scored.append((score, c))
    if not scored:
        return MatchResult(None, 'unmatched', None, f'no candidate >= {FUZZY_ACCEPT_THRESHOLD} similarity')

    scored.sort(key=lambda sc: sc[0], reverse=True)
    best_score, best = scored[0]
    if len(scored) > 1 and (best_score - scored[1][0]) < AMBIGUITY_MARGIN:
        return MatchResult(None, 'ambiguous', best_score,
                            f'top-2 fuzzy candidates within {AMBIGUITY_MARGIN} of each other '
                            f'({best_score:.3f} vs {scored[1][0]:.3f})')
    return MatchResult(best.cfb_player_id, 'fuzzy_name_school', best_score, None)


def normalize_name_matches(a: str, b: str) -> bool:
    return normalize_name(a) == b


STABLE_ID_NAME_MIN_SIMILARITY = 0.6
MATCHED_METHODS = ('exact_name_school', 'fuzzy_name_school', 'cfbd_draft_athlete_id')


def _plausibly_same_person(prospect_name: str, cfb_name: str | None) -> bool:
    """Sanity gate for an id-based bridge: the CFBD record at (year, overall pick)
    must plausibly be the same person -- same last name, or overall similarity
    >= STABLE_ID_NAME_MIN_SIMILARITY (lets 'Cam'/'Cameron' through, blocks a
    pick-numbering mismatch that points at a different player)."""
    a, b = normalize_name(prospect_name), normalize_name(cfb_name or '')
    if not a or not b:
        return False
    if a.split()[-1] == b.split()[-1]:
        return True
    return SequenceMatcher(None, a, b).ratio() >= STABLE_ID_NAME_MIN_SIMILARITY


def reconcile_with_cfbd_draft_id(prospect_name: str, name_result: MatchResult, stable_athlete_id: str | None,
                                 stable_cfb_player_id: int | None, stable_display_name: str | None) -> MatchResult:
    """Combines the name/school match with CFBD's own draft-pick collegeAthleteId
    for the same (draft year, overall pick):
      - agreement            -> keep the name match, note the confirmation;
      - name match failed    -> accept the id bridge (method 'cfbd_draft_athlete_id')
                                if the names are plausibly the same person;
      - disagreement         -> 'ambiguous' (never silently pick one side).
    """
    if not stable_athlete_id:
        return name_result
    if stable_cfb_player_id is None:
        note = f'cfbd_draft_athlete_id={stable_athlete_id} has no CFBD season-stat rows'
        return MatchResult(name_result.cfb_player_id, name_result.method, name_result.confidence,
                           '; '.join(n for n in (name_result.notes, note) if n))
    plausible = _plausibly_same_person(prospect_name, stable_display_name)
    if name_result.cfb_player_id == stable_cfb_player_id:
        return MatchResult(name_result.cfb_player_id, name_result.method, name_result.confidence,
                           '; '.join(n for n in (name_result.notes, 'confirmed by cfbd_draft_athlete_id') if n))
    if name_result.cfb_player_id is None:
        if plausible:
            return MatchResult(stable_cfb_player_id, 'cfbd_draft_athlete_id', None,
                               f'id bridge via CFBD /draft/picks (year, overall); name/school result was '
                               f'{name_result.method}')
        return MatchResult(None, name_result.method, name_result.confidence,
                           '; '.join(n for n in (name_result.notes,
                                                 f'cfbd draft id candidate {stable_athlete_id} rejected: name mismatch '
                                                 f'({stable_display_name!r})') if n))
    return MatchResult(None, 'ambiguous', None,
                       f'conflict: name/school match cfb_player_id={name_result.cfb_player_id} vs CFBD draft '
                       f'athlete id -> cfb_player_id={stable_cfb_player_id}')
