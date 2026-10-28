"""NFL quarterback season pipeline. Run: python -m src.pipeline"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
START, END = 1999, 2025  # nflreadpy's earliest season is 1999; excludes incomplete 2026.
REQUIRED = ['player_id', 'player_display_name', 'position', 'season', 'attempts',
            'passing_yards', 'passing_tds', 'passing_interceptions']


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing source fields: {missing}')
    df = frame.copy()
    df = df[df.position.eq('QB')].copy()
    df = df.dropna(subset=['player_id', 'season', 'attempts'])
    for col in ['attempts', 'passing_yards', 'passing_tds', 'passing_interceptions']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['attempts', 'passing_yards', 'passing_tds', 'passing_interceptions'])
    df = df[df.attempts >= 50].copy()  # Low-volume QBs excluded from first baseline.
    df['ypa'] = df.passing_yards / df.attempts
    df['td_rate'] = df.passing_tds / df.attempts
    df['int_rate'] = df.passing_interceptions / df.attempts
    df = df.sort_values(['player_id', 'season']).reset_index(drop=True)
    return df


def add_era_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute within-season weighted z-scores. These are exploratory, NOT talent estimates."""
    out = df.copy()
    for metric, sign in [('ypa', 1), ('td_rate', 1), ('int_rate', -1)]:
        for season, group in out.groupby('season'):
            weights = group.attempts.to_numpy(dtype=float)
            vals = group[metric].to_numpy(dtype=float)
            avg = np.average(vals, weights=weights)
            sd = np.sqrt(np.average((vals - avg) ** 2, weights=weights))
            out.loc[group.index, f'{metric}_z'] = sign * (vals - avg) / sd if sd > 0 else 0.0
    out['era_score'] = out[['ypa_z', 'td_rate_z', 'int_rate_z']].mean(axis=1)
    return out


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    seasons = list(range(START, END + 1))
    import nflreadpy as nfl
    raw = nfl.load_player_stats(seasons=seasons, summary_level='reg').to_pandas()
    prepared = add_era_scores(prepare(raw))
    prepared.to_csv(DATA / 'qb_seasons.csv', index=False)
    print(f'Saved {len(prepared)} quarterback-seasons to data/qb_seasons.csv')
    print(prepared.nlargest(10, 'era_score')[['player_display_name', 'season', 'attempts', 'era_score']].to_string(index=False))


if __name__ == '__main__':
    main()
