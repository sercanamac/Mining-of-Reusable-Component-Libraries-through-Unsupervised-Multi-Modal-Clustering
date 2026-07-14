"""Regenerate the F5_varbal k=126 trimodal winner partition and refresh the
stale downstream mining artifacts (which were left over from the old F1a k=102
midterm run).

Winner: geo+text+visual, SigLIP coloured, v7 text, F5_varbal (variance-balanced
concat -> UMAP-4), HDBSCAN leaf, mcs=5, ms=3, force-assign.
Reproduces from the cached UMAP-4 reducer; deterministic (random_state=42).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent / 'feature_engineering'))

from _common import (BASELINE, build_X, cumulative_bounded_up_to, load_data,
                     per_type_purity, compute_purity)
from _loaders import load_text, load_visual
from _sweep_runner import _force_assign

MIDTERM = _EXP.parent / 'results' / 'midterm'
CACHE = MIDTERM / 'cache' / 'reducers'
WINNER_UMAP4 = CACHE / ('fusion_geo_text_visual_siglip_colored_F5_vb'
                        '__65e4dcd408f0__umap_4__8a4ac8e713.npy')


def build_partition():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))
    Xt, ti = load_text(gids, 'v7', 'single')
    Xv, vi = load_visual(gids, 'siglip', 'colored')
    common = np.intersect1d(ti, vi)
    y = y_full[common]
    gids_b = gids[common]

    emb = np.load(WINNER_UMAP4)
    Xs = StandardScaler().fit_transform(emb).astype(np.float64)
    lab_raw = HDBSCAN(min_cluster_size=5, min_samples=3,
                      cluster_selection_method='leaf', n_jobs=-1).fit_predict(Xs)
    lab = _force_assign(lab_raw, Xs)

    k = len(np.unique(lab))
    macro = float(np.mean(list(per_type_purity(lab, y).values())))
    overall = float(compute_purity(lab, y))
    print(f'[winner126] objects={len(common)} k={k} '
          f'overall={overall:.4f} macro={macro:.4f}')
    return gids_b, y, lab, emb, df.loc[df['GlobalId'].isin(set(gids_b))]


if __name__ == '__main__':
    gids_b, y, lab, emb, _ = build_partition()
    # persist per-object partition for figure builders
    out = pd.DataFrame({'GlobalId': gids_b, 'type': y, 'cluster': lab})
    out.to_parquet(MIDTERM / 'mining' / 'winner126_partition.parquet')
    print('wrote winner126_partition.parquet')
