"""P5e — Per-type normalized confusion heatmap at the winning config (G22).

Refits the winning HDBSCAN on the canonical fusion and emits a
(17 merged types × k_effective clusters) confusion matrix. Rows are row-
normalized (each type's mass sums to 1). Columns are ordered by the
dominant type for readability.

Output:
    results/midterm/mining/confusion.csv
    results/midterm/mining/confusion.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import RESULTS_ROOT
from _winners import cluster_with_canonical_winner

MINING_DIR = RESULTS_ROOT.parent / 'midterm' / 'mining'
MINING_DIR.mkdir(parents=True, exist_ok=True)


def main():
    labels, X, y, gids, info = cluster_with_canonical_winner(algo='hdbscan')
    print(f'[P5e] using winner: {info["subset"]}/{info["recipe"]} '
          f'macro={info["macro_purity"]:.4f}')
    n = len(y)
    types = sorted(set(y.tolist()))
    clusters = sorted(set(labels.tolist()))

    # Raw counts
    mat = np.zeros((len(types), len(clusters)), dtype=np.int64)
    for i in range(n):
        r = types.index(y[i])
        c = clusters.index(int(labels[i]))
        mat[r, c] += 1

    # Row-normalized
    row_sum = mat.sum(axis=1, keepdims=True).clip(min=1)
    norm = mat / row_sum

    # Reorder columns by dominant type
    dominant = np.argmax(mat, axis=0)      # (n_clusters,) type row per cluster
    col_order = np.argsort(dominant, kind='stable')
    mat_ord = mat[:, col_order]
    norm_ord = norm[:, col_order]
    clusters_ord = [clusters[i] for i in col_order]

    df = pd.DataFrame(norm_ord, index=types,
                       columns=[f'c{c}' for c in clusters_ord])
    df.to_csv(MINING_DIR / 'confusion.csv')
    print(f'saved CSV → {MINING_DIR / "confusion.csv"}  shape={df.shape}')

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(12, 0.18 * len(clusters)), 6), dpi=200)
    im = ax.imshow(norm_ord, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=9)
    ax.set_xticks(range(len(clusters_ord)))
    ax.set_xticklabels([str(c) for c in clusters_ord], fontsize=6, rotation=90)
    ax.set_xlabel(f'HDBSCAN cluster id (ordered by dominant type, k={len(clusters)})')
    ax.set_ylabel('IFC merged type')
    ax.set_title('Per-type confusion on winning fusion '
                 f'(row-normalized; n={n})')
    plt.colorbar(im, ax=ax, label='fraction of type mass')
    fig.tight_layout()
    fig.savefig(MINING_DIR / 'confusion.png')
    plt.close(fig)
    print(f'saved PNG → {MINING_DIR / "confusion.png"}')

    # Summary: per-type purity = max over clusters of row-normalized count
    tp_best = norm_ord.max(axis=1)
    summary = pd.DataFrame({
        'type': types,
        'support': [int(row_sum[i, 0]) for i in range(len(types))],
        'best_cluster_frac': tp_best,
    }).sort_values('best_cluster_frac')
    print('\n[P5e] worst 6 types by per-type purity:')
    print(summary.head(6).to_string(index=False))


if __name__ == '__main__':
    main()
