"""Phase-3 exploratory data analysis -> reports/figures/*.png (+ eda_prospects_summary.json).

Run after the advanced pipeline: python -m src.eda_prospects

EVERYTHING HERE IS DESCRIPTIVE. Charts show raw associations in historical
data (with small samples and selection effects -- e.g. teams give early picks
more starts regardless of talent). None of it is a validated predictive result;
no model has been trained or tested in this phase.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.features import registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_READY = ROOT / 'data' / 'model_ready'
FIG_DIR = ROOT / 'reports' / 'figures'

# reference palette (light mode) -- see the dataviz method: <=3 categorical hues, fixed order
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
DESCRIPTIVE = 'Descriptive association only - not a validated predictive result.'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.axisbelow': True,
})


def _finish(fig, name: str, note: str, out_dir: Path) -> str:
    fig.text(0.01, 0.01, f'{note}  {DESCRIPTIVE}', fontsize=8, color=INK_2, ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out_dir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _bar(ax, x, heights, labels_n, color=S1):
    bars = ax.bar(x, heights, color=color, width=0.7, edgecolor=SURFACE, linewidth=2)
    for b, h, n in zip(bars, heights, labels_n):
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f'{h:.0%}\nn={n}', ha='center', va='bottom',
                    fontsize=8, color=INK_2)
    return bars


def load(model_ready: Path):
    feats = pd.read_parquet(model_ready / 'features_post_draft.parquet')  # widest scope: every registered feature
    tg = pd.read_parquet(model_ready / 'nfl_targets.parquet').drop(columns=['draft_year', 'split'])
    pcs = pd.read_parquet(model_ready / 'prospect_college_seasons.parquet')
    return feats.merge(tg, on='qb_draft_prospect_id', how='left'), pcs


def main(model_ready: Path | None = None, out_dir: Path | None = None) -> dict:
    model_ready = model_ready or MODEL_READY
    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df, pcs = load(model_ready)
    summary: dict = {'note': DESCRIPTIVE, 'figures': []}
    complete = df[df['window_5yr_complete'].astype('boolean').fillna(False)]
    cc = complete[complete['final_attempts'].notna()]
    cls = f'draft classes {int(cc.draft_year.min())}-{int(cc.draft_year.max())}' if len(cc) else ''

    # 1. college efficiency vs NFL success
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    d = cc.dropna(subset=['final_ncaa_passer_rating', 'games_started_5yr'])
    ax.scatter(d['final_ncaa_passer_rating'], d['games_started_5yr'], s=36, color=S1, alpha=0.75,
               edgecolor=SURFACE, linewidth=1)
    bins = pd.qcut(d['final_ncaa_passer_rating'], 5, duplicates='drop')
    m = d.groupby(bins, observed=True).agg(x=('final_ncaa_passer_rating', 'median'), y=('games_started_5yr', 'mean'))
    ax.plot(m['x'], m['y'], color=S2, linewidth=2, marker='o', markersize=8, label='quintile mean')
    ax.set_xlabel('Final-season NCAA passer rating')
    ax.set_ylabel('NFL games started, first 5 seasons')
    ax.set_title('College passing efficiency vs NFL starts')
    ax.legend(frameon=False, loc='upper left')
    rho = d[['final_ncaa_passer_rating', 'games_started_5yr']].corr(method='spearman').iloc[0, 1]
    summary['efficiency_vs_starts_spearman'] = round(float(rho), 3)
    summary['figures'].append(_finish(fig, 'college_efficiency_vs_nfl_success.png', f'n={len(d)}, {cls}; Spearman rho={rho:.2f}.', out_dir))

    # 2. age at draft vs outcomes
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    a = complete.dropna(subset=['age_at_draft', 'started_16plus_5yr']).copy()
    a['age_bin'] = pd.cut(a['age_at_draft'], [0, 22, 23, 24, 99], labels=['< 22', '22-23', '23-24', '24+'], right=False)
    g = a.groupby('age_bin', observed=False)['started_16plus_5yr'].agg(['mean', 'size'])
    _bar(ax, g.index.astype(str), g['mean'].astype(float), g['size'])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel('Age on first day of the draft')
    ax.set_ylabel('Share with 16+ starts in 5 seasons')
    ax.set_title('Age at draft vs reaching 16 NFL starts')
    summary['age_bins_share_16plus'] = {str(k): round(float(v), 3) for k, v in g['mean'].items() if pd.notna(v)}
    summary['figures'].append(_finish(fig, 'age_at_draft_vs_nfl_outcomes.png',
                                      f'n={len(a)} drafted QBs with recorded birthdates, window-complete classes.', out_dir))

    # 3. college development trends
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    p = pcs[pcs['attempts'] >= 100].merge(complete[['qb_draft_prospect_id', 'started_16plus_5yr']], on='qb_draft_prospect_id')
    p = p[p['seasons_before_draft'].between(1, 4)]
    for val, color, lab in ((True, S1, '16+ NFL starts in 5 seasons'), (False, S2, 'fewer than 16 starts')):
        s = p[p['started_16plus_5yr'] == val].groupby('seasons_before_draft')['ncaa_passer_rating'].agg(['mean', 'size'])
        s = s.sort_index(ascending=False)
        x = [f'{int(i)} before' if i > 1 else 'final' for i in s.index]
        ax.plot(x, s['mean'], color=color, linewidth=2, marker='o', markersize=8, label=lab)
        ax.annotate(lab, (x[-1], s['mean'].iloc[-1]), xytext=(8, 0), textcoords='offset points', color=INK_2,
                    fontsize=8, va='center')
    ax.set_xlabel('College season relative to draft (seasons with 100+ attempts)')
    ax.set_ylabel('Mean NCAA passer rating')
    ax.set_title('College efficiency by season, grouped by later NFL outcome')
    ax.legend(frameon=False, loc='upper left')
    summary['figures'].append(_finish(fig, 'college_development_trends.png',
                                      f'{p.qb_draft_prospect_id.nunique()} QBs; groups use outcomes, so this is retrospective.', out_dir))

    # 4. opponent-adjusted performance
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    o = df.dropna(subset=['final_yards_per_attempt', 'final_opp_adj_ypa'])
    ax.scatter(o['final_yards_per_attempt'], o['final_opp_adj_ypa'], s=34, color=S1, alpha=0.75, edgecolor=SURFACE, linewidth=1)
    lim = [min(o['final_yards_per_attempt'].min(), o['final_opp_adj_ypa'].min()) - 0.3,
           max(o['final_yards_per_attempt'].max(), o['final_opp_adj_ypa'].max()) + 0.3]
    ax.plot(lim, lim, linestyle='--', color=INK_2, linewidth=1)
    ax.annotate('no adjustment (y = x)', (lim[1], lim[1]), xytext=(-4, -12), textcoords='offset points',
                ha='right', fontsize=8, color=INK_2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Final-season raw yards per attempt')
    ax.set_ylabel('Opponent-adjusted yards per attempt')
    ax.set_title('Raw vs opponent-adjusted YPA')
    diff = (o['final_opp_adj_ypa'] - o['final_yards_per_attempt'])
    summary['opp_adj_minus_raw_ypa'] = {'median': round(float(diff.median()), 3), 'p10': round(float(diff.quantile(0.1)), 3),
                                        'p90': round(float(diff.quantile(0.9)), 3), 'n': int(len(o))}
    summary['figures'].append(_finish(fig, 'opponent_adjusted_performance.png',
                                      f'n={len(o)} drafted QBs; points below the line faced weaker pass defenses.', out_dir))

    # 5. offensive-system distributions (quantitative profile only; no labels invented)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, col, lab in zip(axes, ['env_pass_rate', 'env_plays_per_game', 'final_qb_rush_share'],
                            ['Team pass rate', 'Team plays per game', 'QB share of team rushes']):
        v = df[col].dropna()
        ax.hist(v, bins=18, color=S1, edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel('Drafted QBs')
        ax.text(0.98, 0.95, f'n={len(v)}', transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK_2)
    fig.suptitle("Offensive environment of each QB's final college season", fontweight='bold')
    summary['figures'].append(_finish(fig, 'offensive_system_distributions.png',
                                      'Sacks count as rushes in NCAA stats (pass rate understated).', out_dir))

    # 6. supporting cast
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, col, lab in zip(axes, ['wr_top3_avg_recruit_rating', 'talent_recruiting_4yr_pctile', 'ol_sack_rate_allowed'],
                            ['Top-3 targets: avg recruit rating', 'Roster recruiting (4-yr) percentile',
                             'Team sack rate allowed']):
        v = df[col].dropna()
        ax.hist(v, bins=18, color=S1, edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel('Drafted QBs')
        ax.text(0.98, 0.95, f'n={len(v)}', transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK_2)
    fig.suptitle('Supporting cast in the final college season', fontweight='bold')
    summary['figures'].append(_finish(fig, 'supporting_cast_quality.png',
                                      'Sack rate also reflects QB decisions and scheme.', out_dir))

    # 7. combine distributions
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    rec = df[df['combine_record_found'] == 1]
    for ax, col, lab in zip(axes, ['combine_forty', 'combine_vertical', 'combine_broad_jump', 'combine_cone'],
                            ['40-yard dash (s)', 'Vertical (in)', 'Broad jump (in)', 'Three-cone (s)']):
        v = rec[col].dropna()
        ax.hist(v, bins=15, color=S1, edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(lab, fontsize=11)
        ax.text(0.98, 0.95, f'ran: {len(v)}/{len(rec)}', transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK_2)
    axes[0].set_ylabel('Drafted QBs')
    fig.suptitle('NFL Combine results among drafted QBs (skipped drills not imputed)', fontweight='bold')
    summary['figures'].append(_finish(fig, 'combine_distributions.png',
                                      f'{len(rec)} of {len(df)} drafted QBs have a combine record.', out_dir))

    # 8. draft position vs production
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    dd = complete.dropna(subset=['draft_overall_pick', 'games_started_5yr'])
    ax.scatter(dd['draft_overall_pick'], dd['games_started_5yr'], s=34, color=S1, alpha=0.75, edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel('Overall draft pick')
    ax.set_ylabel('NFL games started, first 5 seasons')
    ax.set_title('Draft position vs NFL starts')
    rho2 = dd[['draft_overall_pick', 'games_started_5yr']].corr(method='spearman').iloc[0, 1]
    summary['pick_vs_starts_spearman'] = round(float(rho2), 3)
    summary['figures'].append(_finish(fig, 'draft_position_vs_nfl_production.png',
                                      f'n={len(dd)}, classes {int(dd.draft_year.min())}-{int(dd.draft_year.max())}; '
                                      f'Spearman rho={rho2:.2f}. Early picks also get more opportunity.', out_dir))

    # 9. missing-feature rates by class (heatmap, sequential blue)
    groups = sorted({f.group for f in registry.FEATURES.values()})
    mat = []
    for gname in groups:
        cols = [n for n, f in registry.FEATURES.items() if f.group == gname and n in df.columns]
        mat.append(df.groupby('draft_year')[cols].apply(lambda x: x.isna().mean().mean()))
    mat = pd.DataFrame(mat, index=groups)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ)
    im = ax.imshow(mat.to_numpy(), aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(mat.shape[1]), [str(c) for c in mat.columns], rotation=90, fontsize=8)
    ax.set_yticks(range(len(groups)), groups, fontsize=9)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label('Share of feature values missing')
    ax.set_title('Missing-feature rate by feature group and draft class')
    summary['missing_rate_by_group_class'] = mat.round(3).to_dict()
    summary['figures'].append(_finish(fig, 'missing_feature_rates_by_draft_class.png',
                                      'CFBD college coverage starts 2004; talent composite 2015+; SP+ detail 2005-2018.', out_dir))

    # 10. NFL success rates by draft round
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    r = complete.dropna(subset=['draft_round', 'started_16plus_5yr'])
    g = r.groupby('draft_round')['started_16plus_5yr'].agg(['mean', 'size'])
    _bar(ax, [str(int(i)) for i in g.index], g['mean'].astype(float), g['size'])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel('Draft round')
    ax.set_ylabel('Share with 16+ starts in 5 seasons')
    ax.set_title('NFL success rate by draft round')
    summary['share_16plus_by_round'] = {int(k): round(float(v), 3) for k, v in g['mean'].items()}
    summary['figures'].append(_finish(fig, 'nfl_success_by_draft_round.png',
                                      f'n={len(r)} drafted QBs, classes {int(r.draft_year.min())}-{int(r.draft_year.max())} (all window-complete).', out_dir))

    (out_dir / 'eda_prospects_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, default=str)[:3000])
