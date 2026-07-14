"""P5c — UMAP 2-D projection of the winning fusion embedding.

Produces two figures:
    results/midterm/mining/umap_by_type.png     — colored by merged IFC type
    results/midterm/mining/umap_by_cluster.png  — colored by HDBSCAN cluster

Also persists the 2-D coordinates so downstream scripts (catalog, duplicates)
can reuse them.
    results/midterm/mining/umap_coords.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))

from _common import RESULTS_ROOT
from _winners import cluster_with_canonical_winner

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'mining'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fig(coords, labels, title, out_path):
    uniq = sorted(set(labels.tolist()))
    cmap = plt.get_cmap('tab20', max(20, len(uniq)))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i, lab in enumerate(uniq):
        m = labels == lab
        short = str(lab).replace('Ifc', '')
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=[cmap(i % cmap.N)], s=8, alpha=0.75, label=short,
                   edgecolors='none')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.tick_params(labelsize=8)
    if len(uniq) <= 25:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'saved → {out_path}')


def main():
    try:
        import umap
    except ImportError:
        print('umap-learn is required. Install with: pip install umap-learn')
        sys.exit(1)

    clusters, X, y, gids, info = cluster_with_canonical_winner(algo='hdbscan')
    print(f'[P5c] using winner: {info["subset"]}/{info["recipe"]} '
          f'macro={info["macro_purity"]:.4f}; HDBSCAN → {len(set(clusters))} clusters')

    # Fit 2-D UMAP on the winning fusion matrix (metric cosine when winner is
    # UMAP-based, euclidean otherwise — cheap to just always use cosine).
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.05, metric='euclidean',
        random_state=42, n_jobs=1,
    )
    print('[P5c] fitting 2-D UMAP for visualization...')
    coords = reducer.fit_transform(X)
    print(f'  coords shape = {coords.shape}')

    pd.DataFrame({
        'GlobalId': gids, 'u1': coords[:, 0], 'u2': coords[:, 1],
        'type': y, 'cluster': clusters,
    }).to_parquet(OUT_DIR / 'umap_coords.parquet')
    print(f'saved → {OUT_DIR / "umap_coords.parquet"}')

    _fig(coords, y, 'Winning fusion — UMAP colored by IFC type',
         OUT_DIR / 'umap_by_type.png')
    _fig(coords, clusters.astype(str), 'Winning fusion — UMAP colored by cluster',
         OUT_DIR / 'umap_by_cluster.png')


if __name__ == '__main__':
    main()
