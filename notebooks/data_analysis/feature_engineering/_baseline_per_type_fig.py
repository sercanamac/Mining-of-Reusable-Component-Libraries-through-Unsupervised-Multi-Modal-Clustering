"""
_baseline_per_type_fig.py — render the baseline per-type purity bar chart at
k=K_MAIN (=17). Uses the `prev` column of 01_orientation/per_type_delta_k17.csv,
which is the baseline state's per-type purity at the headline operating point.

Writes:
  results/feature_engineering/baseline_per_type_k{K_MAIN}.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import K_MAIN, RESULTS_ROOT, save_fig

SRC = RESULTS_ROOT / '01_orientation' / f'per_type_delta_k{K_MAIN}.csv'
OUT = RESULTS_ROOT / f'baseline_per_type_k{K_MAIN}.png'


def main():
    # Principled reference points (not vibes).
    LARGEST_CLASS = 258.0 / 1700.0          # Furniture / total
    PROG = pd.read_csv(RESULTS_ROOT / 'progression_summary.csv')
    macro_baseline = float(
        PROG[(PROG['config'] == 'Baseline (4, log)') & (PROG['k'] == K_MAIN)].iloc[0]['purity_mean']
    )

    df = pd.read_csv(SRC).rename(columns={'prev': 'baseline_purity'})
    df = df[['type', 'baseline_purity']].sort_values('baseline_purity', ascending=True)
    n_zero = int((df['baseline_purity'] == 0).sum())
    n_below_macro = int((df['baseline_purity'] < macro_baseline).sum())

    colors = [
        '#d62728' if v == 0
        else '#f4a582' if v < macro_baseline
        else '#92c5de'
        for v in df['baseline_purity']
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(df['type'], df['baseline_purity'], color=colors, edgecolor='white')
    for bar, v in zip(bars, df['baseline_purity']):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f'{v:.2f}',
                va='center', fontsize=8, alpha=0.85)

    ax.axvline(LARGEST_CLASS, color='#888', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.text(LARGEST_CLASS, -0.7, f' largest-class\n {LARGEST_CLASS:.3f}',
            va='top', fontsize=8, alpha=0.7)
    ax.axvline(macro_baseline, color='black', linestyle='--', linewidth=1.0, alpha=0.75)
    ax.text(macro_baseline, -0.7, f' macro baseline\n {macro_baseline:.3f}',
            va='top', fontsize=8, alpha=0.85)

    ax.set_xlabel(f'Per-type purity (k={K_MAIN} k-means, 10 seeds, baseline 4 features)')
    ax.set_xlim(0, 1.05)
    ax.set_title(
        f'Baseline failure modes at k={K_MAIN}: '
        f'{n_zero}/{len(df)} types win zero clusters; '
        f'{n_below_macro}/{len(df)} below baseline macro ({macro_baseline:.2f})',
        fontsize=11,
    )

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color='#d62728', label=f'zero ({n_zero} types)'),
        plt.Rectangle((0, 0), 1, 1, color='#f4a582',
                      label=f'< macro ({n_below_macro - n_zero} non-zero types)'),
        plt.Rectangle((0, 0), 1, 1, color='#92c5de',
                      label=f'≥ macro ({len(df) - n_below_macro} types)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
    save_fig(fig, OUT)
    print(f'baseline_per_type_k{K_MAIN}.png → {OUT}')
    print(f'  macro baseline   : {macro_baseline:.3f}')
    print(f'  largest-class    : {LARGEST_CLASS:.3f}')
    print(f'  zero-purity types: {df[df.baseline_purity == 0]["type"].tolist()}')
    print(f'  below-macro types: {df[df.baseline_purity < macro_baseline]["type"].tolist()}')


if __name__ == '__main__':
    main()
