"""Era-aware qualification thresholds.

A fixed attempt count (e.g. 100) is not comparable across a 9-game 1982 season
and a 17-game 2024 season. The default rule instead scales to each season's
scheduled game count using the NFL's own passing-title rule (14 attempts per
scheduled game), configured in config/qualification_thresholds.json and
config/season_calendar.json. All players are retained in the dataset regardless
of whether they qualify -- this only controls a boolean flag used for leaderboards
and for choosing who counts toward a season's baseline distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config'


def _load_json(name: str) -> dict:
    with open(CONFIG / name, 'r', encoding='utf-8') as f:
        return json.load(f)


class SeasonCalendar:
    def __init__(self, data: dict | None = None):
        self._data = data or _load_json('season_calendar.json')

    def games_scheduled(self, season: int, league_id: str) -> tuple[int | None, str]:
        """Returns (games_scheduled, confidence). games_scheduled is None if unknown."""
        bands_key = 'afl_bands' if league_id == 'AFL' else 'nfl_apfa_bands'
        for band in self._data.get(bands_key, []):
            if band['start'] <= season <= band['end']:
                return band.get('games_scheduled'), band.get('confidence', 'unknown')
        return None, 'unknown'


class QualificationRules:
    def __init__(self, config: dict | None = None, calendar: SeasonCalendar | None = None):
        self._config = config or _load_json('qualification_thresholds.json')
        self._calendar = calendar or SeasonCalendar()

    def threshold(self, season: int, league_id: str, method: str | None = None) -> int:
        method = method or self._config['default_method']
        methods = self._config['methods']
        if method == 'fixed':
            return int(methods['fixed']['attempts'])
        if method == 'per_game_rate':
            per_game = methods['per_game_rate']['attempts_per_game']
            games, _confidence = self._calendar.games_scheduled(season, league_id)
            if games is None:
                games = self._config['fallback_games_scheduled_if_unknown']
            return round(per_game * games)
        raise ValueError(f'Unknown qualification method: {method}')

    def qualifies(self, attempts, season: int, league_id: str, method: str | None = None) -> bool:
        if attempts is None:
            return False
        try:
            if attempts != attempts:  # NaN check without importing numpy here
                return False
        except TypeError:
            pass
        return attempts >= self.threshold(season, league_id, method)
