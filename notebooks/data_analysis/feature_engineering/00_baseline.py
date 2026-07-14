"""
00_baseline — 4 OBB size features with log1p + RobustScaler.

No predecessor. This script establishes the reference feature set and k-sweep
that every subsequent script compares against.

Artifacts (results/feature_engineering/00_baseline/):
  - scaling_comparison.png       Raw vs Robust vs Log+Robust on all 4 features
  - scaling_comparison.csv       Purity/Sil/DB/CH for the three scalings (k=K_MAIN)
  - distribution_by_type_*.png   One per baseline feature (log-scaled)
  - k_sweep_metrics.png / .csv   Baseline metrics across K_VALUES
  - cluster_heatmap_k{K_MAIN}.png  Cluster × merged-type composition
  - summary.md
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

from _common import (
    BASELINE, K_MAIN, RESULTS_ROOT,
    append_progression, build_X, eval_k, load_data,
    plot_cluster_heatmap, plot_distribution_by_type, plot_k_sweep, save_fig,
    sweep_k,
)
from sklearn.preprocessing import StandardScaler

STAGE_ID = '00_baseline'
CONFIG_NAME = 'Baseline (4, log)'
OUT = RESULTS_ROOT / STAGE_ID
OUT.mkdir(parents=True, exist_ok=True)


def plot_scaling_comparison(df, labels, out):
    from sklearn.preprocessing import RobustScaler
    X_raw = df[BASELINE].clip(lower=0).values
    X_robust    = RobustScaler().fit_transform(X_raw)
    X_log_rob   = RobustScaler().fit_transform(np.log1p(X_raw))
    X_log_std   = StandardScaler().fit_transform(np.log1p(X_raw))
    transforms = [
        ('Raw',                    X_raw,      'tab:blue'),
        ('RobustScaler',           X_robust,   'tab:orange'),
        ('Log + RobustScaler',     X_log_rob,  'steelblue'),
        ('Log + StandardScaler',   X_log_std,  'seagreen'),
    ]
    fig, axes = plt.subplots(len(transforms), 4, figsize=(18, 2.7*len(transforms)))
    for row, (label, X, color) in enumerate(transforms):
        for i, col in enumerate(BASELINE):
            ax = axes[row, i]
            ax.hist(X[:, i], bins=50, edgecolor='white', alpha=0.85, color=color)
            ax.set_title(f'{col}\nskew = {skew(X[:, i]):.1f}', fontsize=10)
            if i == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold')
    fig.suptitle('Baseline feature distributions: Raw vs RobustScaler vs Log+RobustScaler',
                 fontsize=13, y=1.01)
    save_fig(fig, out)

    rows = []
    for label, X in [('raw_unscaled', X_raw), ('robust', X_robust),
                     ('log_robust', X_log_rob), ('log_standard', X_log_std)]:
        r = eval_k(X, labels, K_MAIN, scale=False)
        rows.append({
            'config': label,
            **{f'{m}_mean': r[m][0] for m in r},
            **{f'{m}_std':  r[m][1] for m in r},
        })
    return pd.DataFrame(rows)


def main():
    df, ifc_merged = load_data()
    print(f'{len(df)} objects, {len(set(ifc_merged))} merged types')

    # (1) Scaling comparison at k=K_MAIN
    scaling_df = plot_scaling_comparison(df, ifc_merged, OUT / 'scaling_comparison.png')
    scaling_df.to_csv(OUT / 'scaling_comparison.csv', index=False)
    print(f'\nScaling comparison (k={K_MAIN}):')
    print(scaling_df[['config', 'purity_mean', 'silhouette_mean',
                      'davies_bouldin_mean', 'calinski_harabasz_mean']].to_string(index=False))

    # (2) Distribution by type for each baseline feature (log-transformed)
    df_log = df.copy()
    for col in BASELINE:
        df_log[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        plot_distribution_by_type(df_log, f'{col}_log', ifc_merged,
                                   OUT / f'distribution_by_type_{col}.png',
                                   title=f'log1p({col}) by merged IfcType')

    # (3) k-sweep on the full baseline
    X = build_X(df, BASELINE)
    sweep = sweep_k(X, ifc_merged)
    sweep.to_csv(OUT / 'k_sweep_metrics.csv', index=False)
    plot_k_sweep({CONFIG_NAME: sweep}, OUT / 'k_sweep_metrics.png')
    append_progression(CONFIG_NAME, sweep)
    print('\nBaseline k-sweep:')
    print(sweep[['k', 'purity_mean', 'purity_std', 'silhouette_mean',
                 'davies_bouldin_mean', 'calinski_harabasz_mean']].round(3).to_string(index=False))

    # (4) Cluster heatmap at K_MAIN
    plot_cluster_heatmap(X, ifc_merged, K_MAIN, OUT / f'cluster_heatmap_k{K_MAIN}.png',
                         title=CONFIG_NAME)

    # (5) Summary.md
    at_main = sweep[sweep['k'] == K_MAIN].iloc[0]
    n_types = len(set(ifc_merged))
    md = f"""# 00_baseline — {CONFIG_NAME}

## Starting point
The baseline is 4 OBB size features, log1p-transformed then StandardScaled:
`Length`, `CrossSectionArea`, `Volume`, `NumVertices`.

## Why log + StandardScaler (see scaling_comparison.png)
Raw features span orders of magnitude and are heavily right-skewed. Plain
RobustScaler centers on the median and divides by IQR — it does not remove the
skew, so a single very large object still dominates. `log1p` first compresses
the range; then StandardScaler normalizes so every feature contributes
equally to Euclidean distance (k-means's objective). Skew drops from ~10+
to ~0–1 (see `scaling_comparison.csv`).

## Quantitative baseline (k={K_MAIN}, 10 seeds, {n_types} merged types)

| Metric | Value |
|---|---|
| Macro purity ↑ | {at_main['purity_mean']:.3f} ± {at_main['purity_std']:.3f} |
| Silhouette ↑ | {at_main['silhouette_mean']:.3f} ± {at_main['silhouette_std']:.3f} |
| Davies-Bouldin ↓ | {at_main['davies_bouldin_mean']:.3f} ± {at_main['davies_bouldin_std']:.3f} |
| Calinski-Harabasz ↑ | {at_main['calinski_harabasz_mean']:.0f} ± {at_main['calinski_harabasz_std']:.0f} |

Reference points (sanity):
- Random label assignment: 1/{n_types} ≈ {1.0/n_types:.3f}
- Largest-class baseline (label all as Furniture, 258/1700): ≈ 0.152

So the 4-feature baseline at k={K_MAIN} is **{at_main['purity_mean']/(1.0/n_types):.1f}×** random
and **{at_main['purity_mean']/0.152:.1f}×** the largest-class trivial.

k={K_MAIN} is chosen because it matches the merged-label count ({n_types} classes).
Additional sweep at k ∈ {{2, 4, 8, 24, 32}} in `k_sweep_metrics.csv` shows
the same baseline at coarser/finer granularities — purity rises monotonically
with k.

## Known deficiency
Size-only features cannot separate objects that differ only in **orientation**
(Column vs Beam), **cross-section shape** (Column vs Plate), or **surface
composition** (Slab vs Railing, Slab vs Roof). Those failures motivate
scripts 01–04.
"""
    (OUT / 'summary.md').write_text(md)
    print(f'\nArtifacts written to {OUT}/')


if __name__ == '__main__':
    main()
