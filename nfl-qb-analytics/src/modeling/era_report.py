"""College passing era analysis -> reports/college_era_analysis.md + figures.

Run:  python -m src.modeling.era_report      (after build_dataset_v2)

Descriptive only, and outcome-blind: nothing here looks at NFL results. It describes
how the FBS reference population's passing statistics moved from 2004 to 2025 and
whether the shift is large enough to justify season-relative normalization.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sst  # noqa: E402

from src.modeling import build_dataset_v2 as B2  # noqa: E402
from src.modeling import era  # noqa: E402
from src.modeling.evaluate import GRID, INK, INK_2, SURFACE, save, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports' / 'college_era_analysis.md'
RESULTS = ROOT / 'results' / 'modeling_v2'
BLUE_100, BLUE_250, BLUE_500 = '#cde2fb', '#86b6ef', '#256abf'
LABELS = {'completion_pct': 'Completion %', 'yards_per_attempt': 'Yards per attempt', 'td_pct': 'TD %',
          'int_pct': 'INT %', 'attempts_per_game': 'Pass attempts per game', 'rush_yards_per_game': 'QB rushing yards per game',
          'opp_adj_ypa': 'Opponent-adjusted YPA', 'sos_pass_defense': 'Opp. pass-D YPA vs league', 'pass_rate': 'Team pass rate'}


def yearly_table(ref: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    r = ref[ref['in_reference']]
    rows = []
    for stat in list(era.NORM_STATS) + ['pass_rate']:
        src = teams if stat == 'pass_rate' else r
        for s, g in src.groupby('season'):
            v = g[stat].dropna()
            rows.append({'stat': stat, 'season': int(s), 'n': len(v), 'mean': v.mean(), 'sd': v.std(),
                         'p10': v.quantile(.1), 'p25': v.quantile(.25), 'p50': v.median(), 'p75': v.quantile(.75),
                         'p90': v.quantile(.9)})
    return pd.DataFrame(rows)


def shift_tests(ref: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    r = ref[ref['in_reference']]
    rows = []
    for stat in list(era.NORM_STATS) + ['pass_rate']:
        src = teams if stat == 'pass_rate' else r
        groups = [g[stat].dropna().to_numpy() for _, g in src.groupby('season')]
        kw = sst.kruskal(*groups)
        e = src.loc[src['season'].between(*B2.EARLY_ERA), stat].dropna()
        lt = src.loc[src['season'].between(*B2.LATE_ERA), stat].dropna()
        pooled = np.sqrt(((len(e) - 1) * e.var() + (len(lt) - 1) * lt.var()) / (len(e) + len(lt) - 2))
        yearly = src.groupby('season')[stat].mean()
        rho, p_rho = sst.spearmanr(yearly.index, yearly.values)
        # share of total variance explained by season (eta^2 from one-way ANOVA)
        grand = src[stat].dropna()
        ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
        ss_total = ((grand - grand.mean()) ** 2).sum()
        rows.append({'stat': stat, 'early_mean': e.mean(), 'late_mean': lt.mean(), 'std_drift': (lt.mean() - e.mean()) / pooled,
                     'sd_ratio_late_early': lt.std() / e.std(), 'trend_rho': rho, 'trend_p': p_rho,
                     'kruskal_p': kw.pvalue, 'eta_sq_season': ss_between / ss_total})
    return pd.DataFrame(rows)


def fig_distributions(y: pd.DataFrame, ref: pd.DataFrame):
    stats = list(era.NORM_STATS) + ['pass_rate']
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), facecolor=SURFACE)
    for ax, stat in zip(axes.ravel(), stats):
        style(ax)
        g = y[y['stat'] == stat]
        ax.fill_between(g['season'], g['p10'], g['p90'], color=BLUE_100, linewidth=0, label='10th-90th pct')
        ax.fill_between(g['season'], g['p25'], g['p75'], color=BLUE_250, linewidth=0, label='25th-75th pct')
        ax.plot(g['season'], g['p50'], color=BLUE_500, linewidth=2, label='Median')
        ax.plot(g['season'], g['mean'], color=INK, linewidth=1.2, linestyle='--', label='Mean')
        ax.set_title(LABELS[stat], color=INK, fontsize=10, loc='left')
    axes.ravel()[0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle('FBS reference passers (>= 100 attempts) by season; team pass rate = FBS teams', color=INK,
                 fontsize=11, x=0.01, ha='left')
    save(fig, 'college_era_distributions.png')


def fig_sd_and_counts(y: pd.DataFrame, ref: pd.DataFrame):
    from matplotlib.colors import LinearSegmentedColormap
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    stats = list(era.NORM_STATS)
    seasons = sorted(y['season'].unique())
    M = np.full((len(stats), len(seasons)), np.nan)
    for i, stat in enumerate(stats):
        g = y[y['stat'] == stat].set_index('season')
        base = g.loc[B2.EARLY_ERA[0]:B2.EARLY_ERA[1], 'sd'].mean()
        M[i] = (g['sd'] / base).reindex(seasons).to_numpy()
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#eb6834'])
    im = ax.imshow(M, cmap=cmap, vmin=0.75, vmax=1.25, aspect='auto')
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([LABELS[s] for s in stats], fontsize=8, color=INK)
    ax.set_xticks(range(0, len(seasons), 3))
    ax.set_xticklabels([seasons[i] for i in range(0, len(seasons), 3)], fontsize=8, color=INK_2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('SD / 2005-09 average SD', fontsize=8)
    ax.set_title('Season spread relative to 2005-2009 (1.0 = unchanged)', color=INK, fontsize=10, loc='left')
    ax = axes[1]
    style(ax)
    r = ref[ref['in_reference']]
    c = r.groupby(['season', 'season_stats_source']).size().unstack(fill_value=0)
    bottom = np.zeros(len(c))
    for col, color, lab in (('cfbd_season_stats', BLUE_500, 'CFBD season stats'),
                            ('box_score_aggregate', BLUE_250, 'Summed from box scores')):
        if col in c.columns:
            ax.bar(c.index, c[col], bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1, label=lab, width=0.8)
            bottom += c[col].to_numpy()
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title('Reference passers per season, by statistics source', color=INK, fontsize=10, loc='left')
    save(fig, 'college_era_sd_and_counts.png')


def fig_raw_vs_normalized(feats: pd.DataFrame, t: pd.DataFrame):
    h = feats.merge(t[['qb_draft_prospect_id', 'v2_feature_eligible']], on='qb_draft_prospect_id')
    h = h[h['v2_feature_eligible'] & h['draft_year'].between(2005, 2026)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for row, short in enumerate(('cmp', 'ypa')):
        for col, m in enumerate(('raw', 'season_z')):
            ax = axes[row, col]
            style(ax)
            v = h[f'{short}__{m}']
            ax.plot(h['draft_year'] + np.random.default_rng(0).uniform(-0.25, 0.25, len(h)), v, 'o', markersize=4,
                    color=BLUE_250, alpha=0.8)
            mean = h.groupby('draft_year')[f'{short}__{m}'].mean()
            ax.plot(mean.index, mean.values, color=BLUE_500, linewidth=2)
            unit = {'cmp': 'Completion %', 'ypa': 'Yards per attempt'}[short]
            ax.set_title(f'{unit}: {"raw" if m == "raw" else "season z-score"}', color=INK, fontsize=10, loc='left')
            if m == 'season_z':
                ax.axhline(0, color=INK_2, linewidth=1)
    fig.suptitle('Drafted QBs, final college season (dots) and draft-class mean (line)', color=INK, fontsize=11,
                 x=0.01, ha='left')
    save(fig, 'college_era_raw_vs_normalized.png')


def drafted_relative(feats: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    h = feats.merge(t[['qb_draft_prospect_id', 'v2_feature_eligible']], on='qb_draft_prospect_id')
    h = h[h['v2_feature_eligible'] & h['draft_year'].between(2005, 2025)]
    h['era'] = pd.cut(h['draft_year'], [2004, 2010, 2015, 2020, 2025], labels=['2005-10', '2011-15', '2016-20', '2021-25'])
    return h.groupby('era', observed=True)[['cmp__raw', 'cmp__season_z', 'ypa__raw', 'ypa__season_z',
                                            'rush_pg__raw', 'rush_pg__season_z']].mean()


def render(y, tests, drifted, ref, combo, rho_season=(float('nan'),) * 2, rho_conf=(float('nan'),) * 2) -> str:
    T = tests.set_index('stat')
    r = ref[ref['in_reference']]
    n = r.groupby('season').size()
    L = ['# College QB statistics across eras (2004-2025)', '',
         'Generated by `python -m src.modeling.era_report`. Outcome-blind: no NFL result is used anywhere in this '
         'analysis. The reference population is every passer-season at an FBS team with at least '
         f'{era.REF_MIN_ATTEMPTS} attempts and {era.REF_MIN_GAMES} games with a pass attempt. It covers '
         f'{len(r):,} passer-seasons, {n.min()}-{n.max()} per season, built from CFBD season stats plus box-score '
         'season totals where CFBD season stats are missing (mostly 2004-2008).', '',
         '## Did the distributions shift?', '',
         '| Statistic | 2005-09 mean | 2021-25 mean | Shift (pooled SDs) | SD ratio late/early | Trend of yearly means (Spearman, p) | Variance explained by season |',
         '|---|---|---|---|---|---|---|']
    for stat in list(era.NORM_STATS) + ['pass_rate']:
        x = T.loc[stat]
        L.append(f'| {LABELS[stat]} | {x["early_mean"]:.3f} | {x["late_mean"]:.3f} | {x["std_drift"]:+.2f} | '
                 f'{x["sd_ratio_late_early"]:.2f} | {x["trend_rho"]:+.2f} (p={x["trend_p"]:.3f}) | {100 * x["eta_sq_season"]:.1f}% |')
    L += ['', 'All Kruskal-Wallis tests across seasons are listed in `results/modeling_v2/era_shift_tests.csv`. '
          'With 3,000+ passer-seasons even a trivial shift is "significant", so the effect sizes matter more than the '
          'p-values.', '',
          '![Distributions](figures/college_era_distributions.png)', '',
          '![SD and counts](figures/college_era_sd_and_counts.png)', '',
          '## Reading', '',
          f'* **Completion % and YPA drifted up.** Completion % moved {T.loc["completion_pct", "std_drift"]:+.2f} SD and YPA '
          f'{T.loc["yards_per_attempt", "std_drift"]:+.2f} SD from 2005-09 to 2021-25, with a monotone yearly trend '
          f'(Spearman {T.loc["completion_pct", "trend_rho"]:+.2f} and {T.loc["yards_per_attempt", "trend_rho"]:+.2f}). '
          'A 62% passer in 2025 is close to average; in 2005 he was well above it.',
          f'* **Interceptions fell** ({T.loc["int_pct", "std_drift"]:+.2f} SD), and opponent-adjusted YPA rose with raw YPA '
          f'({T.loc["opp_adj_ypa", "std_drift"]:+.2f} SD). The adjustment only removes opponent strength relative to '
          'the same season, not the season level.',
          f'* **QB rushing rose in level but varies a lot** ({T.loc["rush_yards_per_game", "std_drift"]:+.2f} SD). Attempts per game '
          f'({T.loc["attempts_per_game", "std_drift"]:+.2f} SD), TD % ({T.loc["td_pct", "std_drift"]:+.2f} SD) and schedule strength '
          f'({T.loc["sos_pass_defense", "std_drift"]:+.2f} SD) barely moved.',
          f'* **Season explains a small share of variance:** at most {100 * T["eta_sq_season"].max():.1f}% for any statistic. '
          'Most variation is between QBs within a season. So normalization is a modest correction, not a transformation. '
          'Among the historical-eligible drafted QBs, the rank correlation between raw values and season z-scores is '
          f'|rho| {rho_season[0]:.2f}-{rho_season[1]:.2f}; for conference z-scores it is {rho_conf[0]:.2f}-{rho_conf[1]:.2f}.',
          f'* **Teams did not pass more.** The FBS team pass rate is flat ({T.loc["pass_rate", "std_drift"]:+.2f} SD), so the '
          'efficiency gains are not a by-product of passing more often.',
          '* **The spread was stable.** SD ratios are near 1, so a z-score mainly re-centres each season rather than '
          're-scaling it.', '',
          '## Not every difference is an era effect', '',
          '* **Rule changes.** Attempts per game dip in 2006, when the NCAA started the clock on kickoffs and changes of '
          'possession (fewer plays). The rule was reverted in 2007. That is a real one-season environment change, '
          'and normalizing removes it, which is the intent.',
          '* **2020.** The COVID season had shortened, uneven schedules, opt-outs and the fewest reference passers '
          f'({int(n.loc[2020])}). It is a noisier baseline.',
          '* **Composition.** FBS grew from 119 to 136 teams. New members change who is in the reference population.',
          '* **Source change.** 2004-2008 seasons are mostly summed from box scores, 2009+ come from CFBD season stats '
          '(see the counts chart). There is no visible break in the series at 2008/2009, which suggests the two '
          'sources agree. But box-score coverage of games against FCS opponents can differ slightly.',
          '* **Genuine change.** Spread and RPO concepts and better QB training may have produced genuinely more '
          'accurate passers. Season normalization cannot separate "the league got easier" from "QBs got better". It '
          'measures a QB against his contemporaries, which is what scouts compare at draft time.', '',
          '## Decision', '',
          'The shift in completion %, YPA and INT % (about 0.35-0.55 SD over 20 years) is large enough that raw rates from '
          '2005 and 2021 are not interchangeable. It is small next to between-QB variation. So normalization is '
          'justified as a correction, but **it is not expected to transform predictive performance**. '
          'That is the question `reports/normalization_ablation.md` tests.', '',
          f'The `combo` strategy normalizes only features whose reference population shifted at least {B2.DRIFT_MIN} '
          'SD. It was chosen here, before any model was fit: ' +
          ', '.join(f'`{k}` -> {v}' for k, v in combo.items()) + '.', '',
          '## Drafted QBs, raw vs normalized', '',
          '![Raw vs normalized](figures/college_era_raw_vs_normalized.png)', '',
          '| Draft era | Completion % raw | Completion % z | YPA raw | YPA z | Rush yds/game raw | Rush yds/game z |',
          '|---|---|---|---|---|---|---|']
    for e, x in drifted.iterrows():
        L.append(f'| {e} | {x["cmp__raw"]:.1f} | {x["cmp__season_z"]:+.2f} | {x["ypa__raw"]:.2f} | {x["ypa__season_z"]:+.2f} | '
                 f'{x["rush_pg__raw"]:.1f} | {x["rush_pg__season_z"]:+.2f} |')
    L += ['', 'Drafted QBs sit about 1 SD above the reference population in every era. Their raw completion % climbs '
          'with the league, but their season z-scores do not. Normalization removes the league-level drift while '
          'keeping each QB\'s standing among his peers.', '',
          '## Method details', '',
          f'* z = (x - season mean) / season SD, clipped to +/-{era.Z_CLIP}. Season percentile = mid-rank empirical '
          'percentile among the season\'s reference passers. Conference z uses the (season, conference) group, the '
          f'team\'s conference in that season; groups with fewer than {era.CONF_MIN_GROUP} passers fall back to the '
          'season baseline.',
          '* INT % and schedule strength are sign-flipped, so higher always means better (fewer interceptions, tougher schedule).',
          f'* Prospect seasons with fewer than {era.PROSPECT_MIN_ATTEMPTS} attempts get no rate score (NaN). A baseline '
          'with fewer than 2 passers or zero SD gives NaN, never 0 or infinity. Raw values are kept next to every score.',
          '* Baselines are unweighted (each qualifying passer counts once). Attempt weighting and rolling multi-season '
          'windows are implemented and tested in `src/modeling/era.py`, but not used.',
          '* A season-s baseline uses only season-s passers. All of those seasons ended before the next NFL draft, so '
          'normalized values are available at draft time. No NFL information enters the baselines.', '']
    return '\n'.join(L)


def main() -> dict:
    ref = era.load_or_build_reference()
    tables = era.load_tables()
    teams = era.team_pass_rates(tables['team_stats'], era.fbs_team_seasons(tables['games']))
    feats = pd.read_parquet(B2.FEATURES_FILE)
    t = pd.read_parquet(B2.TARGETS_FILE)
    manifest = json.loads((B2.OUT_DIR / f'dataset_manifest_v{B2.DATASET_VERSION}.json').read_text(encoding='utf-8'))
    y = yearly_table(ref, teams)
    tests = shift_tests(ref, teams)
    RESULTS.mkdir(parents=True, exist_ok=True)
    y.to_csv(RESULTS / 'era_yearly_distributions.csv', index=False)
    tests.to_csv(RESULTS / 'era_shift_tests.csv', index=False)
    fig_distributions(y, ref)
    fig_sd_and_counts(y, ref)
    fig_raw_vs_normalized(feats, t)
    drifted = drafted_relative(feats, t)
    hh = feats.merge(t[['qb_draft_prospect_id', 'v2_historical']], on='qb_draft_prospect_id')
    hh = hh[hh['v2_historical']]
    rs = [abs(hh[f'{k}__raw'].corr(hh[f'{k}__season_z'], method='spearman')) for k in B2.CONTEXT_SHORT]
    rc = [abs(hh[f'{k}__raw'].corr(hh[f'{k}__conf_z'], method='spearman')) for k in B2.CONTEXT_SHORT]
    REPORT.write_text(render(y, tests, drifted, ref, manifest['combo_choice'], (min(rs), max(rs)), (min(rc), max(rc))),
                      encoding='utf-8')
    print(tests[['stat', 'std_drift', 'sd_ratio_late_early', 'trend_rho', 'eta_sq_season']].round(3).to_string())
    return {'tests': tests}


if __name__ == '__main__':
    main()
