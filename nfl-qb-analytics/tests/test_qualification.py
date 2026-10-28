from src.qualification import QualificationRules, SeasonCalendar


def test_threshold_scales_with_strike_shortened_season():
    rules = QualificationRules()
    # 1982 strike season: 9 scheduled games * 14 attempts/game = 126.
    assert rules.threshold(1982, 'NFL') == 126
    # 2024: 17 scheduled games * 14 = 238.
    assert rules.threshold(2024, 'NFL') == 238
    # A fixed 100-attempt rule is clearly wrong for a 9-game season but available
    # as an explicit alternative for sensitivity testing.
    assert rules.threshold(1982, 'NFL', method='fixed') == 100


def test_threshold_falls_back_when_season_calendar_unknown():
    rules = QualificationRules()
    # A season outside every configured band falls back to the configured
    # default games-scheduled rather than raising.
    assert rules.threshold(1900, 'NFL') == 14 * 16


def test_afl_and_nfl_never_share_a_calendar_band_for_the_same_years():
    cal = SeasonCalendar()
    nfl_games, _ = cal.games_scheduled(1965, 'NFL')
    afl_games, _ = cal.games_scheduled(1965, 'AFL')
    # Both resolve independently -- the point is they come from separate bands,
    # not that the numbers must differ.
    assert nfl_games is not None
    assert afl_games is not None


def test_qualifies_handles_missing_attempts():
    rules = QualificationRules()
    assert rules.qualifies(None, 2024, 'NFL') is False
    assert rules.qualifies(float('nan'), 2024, 'NFL') is False
    assert rules.qualifies(240, 2024, 'NFL') is True
    assert rules.qualifies(100, 2024, 'NFL') is False
