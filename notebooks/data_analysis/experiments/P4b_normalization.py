"""P4b — Normalization ablation on the winning fusion config.

Tests {StandardScaler, RobustScaler, log1p+StandardScaler} as the per-block
scaler before concat, then runs k-means k=32 and the winning HDBSCAN config.

Output:
    results/midterm/stability/normalization.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))

from _common import (
    BASELINE, RESULTS_ROOT, build_X, compute_purity,
    cumulative_bounded_up_to, load_data, per_type_purity,
)
from _loaders import load_text, load_visual
from _winners import (
    CANONICAL_TEXT_AGG, CANONICAL_TEXT_VERSION,
    CANONICAL_VISUAL_ENCODER, CANONICAL_VISUAL_VARIANT,
    PCA_DIM_TEXT, PCA_DIM_VISUAL, get_canonical_winner,
)

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'stability'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _scale(X, scheme):
    if scheme == 'standard':
        return StandardScaler().fit_transform(X)
    if scheme == 'robust':
        return RobustScaler().fit_transform(X)
    if scheme == 'log_standard':
        # log1p only valid for non-negative; shift if needed
        mn = X.min(axis=0)
        shift = np.where(mn < 0, -mn + 1e-6, 0.0)
        return StandardScaler().fit_transform(np.log1p(X + shift))
    raise ValueError(scheme)


def _force_assign(lab, X):
    lab = lab.copy()
    non_noise = lab != -1
    if non_noise.sum() == 0:
        return lab
    ids = sorted(set(lab[non_noise].tolist()))
    C = np.stack([X[lab == c].mean(0) for c in ids])
    ni = np.where(~non_noise)[0]
    if ni.size:
        d = np.linalg.norm(X[ni][:, None, :] - C[None, :, :], axis=2)
        lab[ni] = np.array(ids)[d.argmin(1)]
    return lab


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt_raw, txt_idx = load_text(gids, CANONICAL_TEXT_VERSION, CANONICAL_TEXT_AGG)
    X_vis_raw, vis_idx = load_visual(
        gids, CANONICAL_VISUAL_ENCODER, CANONICAL_VISUAL_VARIANT,
    )
    common = np.intersect1d(txt_idx, vis_idx)

    def _align(X_raw, idx_arr):
        return X_raw[np.searchsorted(idx_arr, common)]

    geo_b = geo[common]
    txt_b = PCA(n_components=PCA_DIM_TEXT, random_state=42).fit_transform(
        _align(X_txt_raw, txt_idx),
    )
    vis_b = PCA(n_components=PCA_DIM_VISUAL, random_state=42).fit_transform(
        _align(X_vis_raw, vis_idx),
    )
    y = y_full[common]

    # Use the globally winning HDBSCAN hps, but keep the F2-style block-wise
    # concat below — this phase is a normalization ablation on the classic
    # recipe, not a full re-fusion.
    winner = get_canonical_winner(algo='hdbscan')
    hps = winner['hps']
    print(f'[P4b] winner from fusion.csv: {winner["subset"]}/{winner["recipe"]} '
          f'hps={hps} (applied here to F2-style geo+text+visual)')

    rows = []
    for scheme in ['standard', 'robust', 'log_standard']:
        blocks = [_scale(b, scheme) for b in (geo_b, txt_b, vis_b)]
        X = np.hstack(blocks).astype(np.float64)
        print(f'\n[P4b] scheme={scheme}  X shape={X.shape}')

        for seed in [42, 43, 44]:
            lab = MiniBatchKMeans(
                n_clusters=32, batch_size=1024, n_init='auto',
                random_state=seed,
            ).fit_predict(X)
            pur = compute_purity(lab, y)
            tp = per_type_purity(lab, y)
            macro = float(np.mean(list(tp.values())))
            rows.append(dict(scheme=scheme, algo='kmeans_k32', seed=seed,
                             purity=pur, macro_purity=macro, k=32))
            print(f'  kmeans seed={seed}  pur={pur:.4f}  macro={macro:.4f}')

        lab_raw = HDBSCAN(
            min_cluster_size=int(hps['mcs']),
            min_samples=int(hps['ms']),
            cluster_selection_method=hps['method'],
            n_jobs=-1,
        ).fit_predict(X)
        lab = _force_assign(lab_raw, X) if hps.get('force', True) else lab_raw
        pur = compute_purity(lab, y)
        tp = per_type_purity(lab, y)
        macro = float(np.mean(list(tp.values())))
        k_eff = len(set(lab.tolist())) - (1 if -1 in lab else 0)
        rows.append(dict(
            scheme=scheme, algo='hdbscan_winner', seed=-1,
            purity=pur, macro_purity=macro, k=k_eff,
        ))
        print(f'  hdbscan        pur={pur:.4f}  macro={macro:.4f}  k={k_eff}')

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / 'normalization.csv', index=False)

    summary = (out.groupby(['scheme', 'algo'])[['purity', 'macro_purity', 'k']]
                .agg(['mean', 'std']).round(4))
    print('\n[P4b] normalization × algo summary:')
    print(summary.to_string())
    print(f'\nsaved → {OUT_DIR / "normalization.csv"}')


if __name__ == '__main__':
    main()
