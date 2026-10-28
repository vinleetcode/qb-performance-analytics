"""Exploratory data analysis on the ingested 1999-2025 all-time QB dataset.

Reads only the Parquet exports written by src/historical_pipeline.py (so it
doubles as a smoke test that those exports are self-sufficient), and writes
summary stats + charts to results/eda/. Run after `python -m src.historical_pipeline`.

Per this phase's scope, this is descriptive only -- no predictive modeling and
no composite rating is built here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'eda'

# Validated categorical/sequential palette (see dataviz skill references/palette.md).
BLUE = '#2a78d6'
ORANGE = '#eb6834'
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
AXIS = '#c3c2b7'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'text.color': INK_PRIMARY, 'axes.labelcolor': INK_SECONDARY,
    'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
    'axes.edgecolor': AXIS, 'grid.color': GRIDLINE,
    'font.family': 'sans-serif', 'axes.grid': True, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_dataset() -> pd.DataFrame:
    raw = pd.read_parquet(DATA / 'player_seasons_raw.parquet')
    derived = pd.read_parquet(DATA / 'player_seasons_derived.parquet')
    era = pd.read_parquet(DATA / 'era_adjusted_stats.parquet')
    df = raw.merge(derived, left_on='id', right_on='player_season_id', suffixes=('', '_derived'))
    era_pr = era[(era.metric == 'passer_rating') & (era.baseline_method == 'season_league')][
        ['player_season_id', 'z_score', 'percentile']
    ].rename(columns={'z_score': 'passer_rating_z', 'percentile': 'passer_rating_percentile'})
    df = df.merge(era_pr, left_on='id', right_on='player_season_id', how='left', suffixes=('', '_era'))
    return df


def _line_chart(x, y, title, ylabel, path, color=BLUE, subtitle=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color=color, linewidth=2)
    ax.axhline(0, color=AXIS, linewidth=1) if y.min() < 0 < y.max() else None
    ax.set_title(title, color=INK_PRIMARY, loc='left', fontsize=14, pad=28 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY,
                 fontsize=10, va='bottom')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Season')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(df: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    qualifying = df[df.qualifies_leaderboard]

    by_season = qualifying.groupby('season').agg(
        mean_passer_rating=('passer_rating', 'mean'),
        mean_ypa=('yards_per_attempt', 'mean'),
        mean_passer_rating_z=('passer_rating_z', 'mean'),
        n_qualifying=('id', 'count'),
    ).reset_index()
    all_by_season = df.groupby('season').agg(n_total=('id', 'count')).reset_index()
    by_season = by_season.merge(all_by_season, on='season')

    _line_chart(by_season.season, by_season.mean_passer_rating,
                'Mean passer rating, qualifying QBs, by season (raw -- not era-adjusted)',
                'Passer rating', OUT / 'mean_passer_rating_by_season.png', color=BLUE)
    _line_chart(by_season.season, by_season.mean_ypa,
                'Mean yards per attempt, qualifying QBs, by season (raw)',
                'Yards / attempt', OUT / 'mean_ypa_by_season.png', color=BLUE)

    rushing_by_season = qualifying.groupby('season').agg(
        mean_rushing_yards=('rushing_yards', 'mean'),
    ).reset_index()
    _line_chart(rushing_by_season.season, rushing_by_season.mean_rushing_yards,
                'Mean rushing yards, qualifying QBs, by season',
                'Rushing yards', OUT / 'mean_rushing_yards_by_season.png', color=ORANGE,
                subtitle='Rise of the dual-threat QB: rushing production among qualifying passers over time')
    _line_chart(by_season.season, by_season.mean_passer_rating_z,
                'Mean season-relative passer-rating z-score, by season',
                'Z-score', OUT / 'zscore_passer_rating_by_season.png', color=ORANGE,
                subtitle='Near zero every year by construction -- this is what enables cross-era comparison')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(by_season.season, by_season.n_qualifying, color=BLUE, width=0.8, label='Qualifying')
    ax.bar(by_season.season, by_season.n_total - by_season.n_qualifying, bottom=by_season.n_qualifying,
           color=GRIDLINE, width=0.8, label='Below threshold (retained, not on leaderboard)')
    ax.set_title('Passers per season: qualifying vs. retained-but-below-threshold', loc='left', fontsize=13)
    ax.set_ylabel('Player-seasons')
    ax.set_xlabel('Season')
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT / 'qualifying_counts_by_season.png', dpi=150)
    plt.close(fig)

    decade = df.copy()
    decade['decade'] = (decade.season // 10) * 10
    missingness = {
        col: float(df[col].isna().mean() * 100)
        for col in ['completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
                    'sacks_suffered', 'sack_yards_lost', 'adjusted_net_yards_per_attempt',
                    'carries', 'rushing_yards', 'yards_per_carry']
    }
    non_qb_passers = int((~df.is_qb_listed).sum())

    summary = {
        'rows_analyzed': int(len(df)),
        'season_range': [int(df.season.min()), int(df.season.max())],
        'missingness_pct_by_column': missingness,
        'non_qb_listed_passers_retained': non_qb_passers,
        'decade_summary': {
            str(dec): grp[['passer_rating', 'yards_per_attempt', 'completion_pct']].describe().round(2).to_dict()
            for dec, grp in decade.groupby('decade')
        },
        'charts': [p.name for p in sorted(OUT.glob('*.png'))],
    }
    (ROOT / 'results' / 'eda_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> dict:
    df = load_dataset()
    return build(df)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, default=str)[:2000])
