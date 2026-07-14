"""P5d — Cluster composition report.

Writes:
    results/midterm/mining/composition.md     — summary of cluster sizes
    results/midterm/mining/cluster_sizes.png  — histogram
    results/midterm/mining/cluster_type_mix.csv — rows=cluster, cols=type, counts
"""
from __future__ import annotations
import sys
from collections import Counter
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


def main():
    clusters, X, y, gids, info = cluster_with_canonical_winner(algo='hdbscan')
    print(f'[P5d] using winner: {info["subset"]}/{info["recipe"]} '
          f'macro={info["macro_purity"]:.4f}')
    cluster_ids = sorted(set(clusters.tolist()))

    # Size histogram
    sizes = np.array([(clusters == c).sum() for c in cluster_ids])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(sizes)), sorted(sizes, reverse=True), color='steelblue',
           edgecolor='white')
    ax.set_xlabel('cluster rank (large→small)')
    ax.set_ylabel('members')
    ax.set_title(f'Cluster size distribution — {len(sizes)} clusters, '
                 f'total {int(sizes.sum())} objects')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'cluster_sizes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Type mix matrix
    types = sorted(set(y.tolist()))
    rows = []
    for c in cluster_ids:
        m = clusters == c
        cnt = Counter(y[m].tolist())
        row = {'cluster_id': int(c), 'size': int(m.sum())}
        for t in types:
            row[str(t).replace('Ifc', '')] = int(cnt.get(t, 0))
        top_t, top_n = cnt.most_common(1)[0]
        row['dominant_type'] = str(top_t)
        row['purity'] = top_n / m.sum()
        rows.append(row)
    mix = pd.DataFrame(rows).sort_values('size', ascending=False)
    mix.to_csv(OUT_DIR / 'cluster_type_mix.csv', index=False)

    # Composition md summary
    n_sparse = int((sizes <= 5).sum())
    n_dense = int((sizes >= 50).sum())
    median = int(np.median(sizes))
    md = [
        '# Library composition — winning fusion clustering',
        '',
        f'- Total clusters: **{len(sizes)}**',
        f'- Total objects: **{int(sizes.sum())}**',
        f'- Median cluster size: **{median}**',
        f'- Smallest / largest: **{int(sizes.min())} / {int(sizes.max())}**',
        f'- Sparse clusters (≤5 members): **{n_sparse}**',
        f'- Dense clusters (≥50 members): **{n_dense}**',
        '',
        '## Dominant type breakdown',
        '',
        '| Dominant type | #clusters | total members |',
        '|---|---:|---:|',
    ]
    dom = mix.groupby('dominant_type').agg(
        n_clusters=('cluster_id', 'count'),
        total_members=('size', 'sum'),
    ).sort_values('total_members', ascending=False)
    for t, row in dom.iterrows():
        md.append(f'| {str(t).replace("Ifc","")} | {row.n_clusters} | {row.total_members} |')
    md.append('')
    md.append('## Top 10 largest clusters')
    md.append('')
    md.append('| Cluster | Size | Dominant type | Purity |')
    md.append('|---|---:|---|---:|')
    for _, r in mix.head(10).iterrows():
        md.append(f'| {r.cluster_id:03d} | {r["size"]} | '
                  f'{str(r.dominant_type).replace("Ifc","")} | {r.purity:.2%} |')
    (OUT_DIR / 'composition.md').write_text('\n'.join(md))

    print(f'[P5d] {len(sizes)} clusters, size median={median}, '
          f'sparse={n_sparse}, dense={n_dense}')
    print(f'saved → {OUT_DIR / "cluster_sizes.png"}')
    print(f'saved → {OUT_DIR / "cluster_type_mix.csv"}')
    print(f'saved → {OUT_DIR / "composition.md"}')


if __name__ == '__main__':
    main()
