from src.match_prospects import (
    CfbCandidate, match_prospect_to_cfb, normalize_name, normalize_school, split_schools,
)


def test_normalize_name_strips_suffixes_and_punctuation():
    assert normalize_name('Robert Griffin III') == 'robert griffin'
    assert normalize_name("D'Wayne O'Neil Jr.") == "dwayne oneil"
    assert normalize_name('Matt  Ryan') == 'matt ryan'


def test_normalize_school_handles_state_abbreviation_and_parentheticals():
    assert normalize_school('Ohio St.') == 'ohio state'
    assert normalize_school('Miami (FL)') == 'miami'
    assert normalize_school('The University of Georgia') == 'georgia'


def test_normalize_school_strips_apostrophes():
    # Confirmed real mismatch: CFBD spells it "Hawai'i", nflverse's draft data
    # (and most other sources) spell it "Hawaii".
    assert normalize_school("Hawai'i") == normalize_school('Hawaii') == 'hawaii'


def test_normalize_school_resolves_confirmed_naming_convention_differences():
    # Both confirmed against real CFBD data on 2026-10-30 (see WHOLE_SCHOOL_ALIASES).
    assert normalize_school('Mississippi') == normalize_school('Ole Miss') == 'ole miss'
    assert normalize_school('North Carolina St.') == normalize_school('NC State') == 'nc state'
    assert normalize_school('Central Florida') == normalize_school('UCF') == 'ucf'


def test_normalize_strips_diacritics_on_names_and_schools():
    # CFBD was confirmed (2026-10-30) to spell it "San José State"; nflverse's
    # draft college field spells it ASCII-only as "San Jose St.".
    assert normalize_school('San José State') == normalize_school('San Jose St.') == 'san jose state'
    assert normalize_name('José Ramírez') == normalize_name('Jose Ramirez') == 'jose ramirez'


def test_split_schools_handles_transfer_list():
    assert split_schools('Wyoming; Reedley') == ['wyoming', 'reedley']
    assert split_schools('Miami; Georgia') == ['miami', 'georgia']
    assert split_schools('Texas St.') == ['texas state']


def test_exact_match_when_unique_name_and_school():
    candidates = [CfbCandidate(cfb_player_id=1, name='Joe Burrow', schools={'lsu'}, seasons={2019})]
    result = match_prospect_to_cfb('Joe Burrow', 'LSU', 2020, candidates)
    assert result.method == 'exact_name_school'
    assert result.cfb_player_id == 1


def test_transfer_matches_via_either_school():
    candidates = [CfbCandidate(cfb_player_id=5, name='Cam Newton', schools={'florida', 'auburn'},
                                seasons={2008, 2009, 2010})]
    result = match_prospect_to_cfb('Cam Newton', 'Florida; Auburn', 2011, candidates)
    assert result.method == 'exact_name_school'
    assert result.cfb_player_id == 5


def test_identical_names_at_different_schools_do_not_collide():
    candidates = [
        CfbCandidate(cfb_player_id=1, name='Mike Williams', schools={'clemson'}, seasons={2015}),
        CfbCandidate(cfb_player_id=2, name='Mike Williams', schools={'usc'}, seasons={2015}),
    ]
    result = match_prospect_to_cfb('Mike Williams', 'USC', 2016, candidates)
    assert result.cfb_player_id == 2
    assert result.method == 'exact_name_school'


def test_same_name_same_school_is_flagged_ambiguous_not_guessed():
    # Two genuinely different seasons/people who happen to share name+school
    # inside the matching window must not be silently resolved.
    candidates = [
        CfbCandidate(cfb_player_id=1, name='John Smith', schools={'alabama'}, seasons={2018}),
        CfbCandidate(cfb_player_id=2, name='John Smith', schools={'alabama'}, seasons={2019}),
    ]
    result = match_prospect_to_cfb('John Smith', 'Alabama', 2020, candidates)
    assert result.method == 'ambiguous'
    assert result.cfb_player_id is None


def test_fuzzy_match_accepts_close_misspelling():
    candidates = [CfbCandidate(cfb_player_id=9, name='Trevor Lawrence', schools={'clemson'}, seasons={2020})]
    result = match_prospect_to_cfb('Trevor Lawrance', 'Clemson', 2021, candidates)  # typo'd source name
    assert result.method == 'fuzzy_name_school'
    assert result.cfb_player_id == 9
    assert result.confidence >= 0.9


def test_no_candidate_outside_season_window_is_unmatched():
    candidates = [CfbCandidate(cfb_player_id=1, name='Old Timer', schools={'notre dame'}, seasons={1990})]
    result = match_prospect_to_cfb('Old Timer', 'Notre Dame', 2024, candidates)
    assert result.method == 'unmatched'
    assert result.cfb_player_id is None


def test_no_candidates_at_all_is_unmatched_not_an_error():
    result = match_prospect_to_cfb('Nobody Here', 'Nowhere State', 2024, [])
    assert result.method == 'unmatched'
    assert result.cfb_player_id is None
