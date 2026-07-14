"""Smoke test: run each fusion strategy once with a single HDBSCAN config on
one encoder. Confirms each strategy returns a valid feature matrix and
HDBSCAN produces non-trivial clustering.
"""
import os, tempfile, sys
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'feature_engineering'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.cluster import HDBSCAN
import P2c_fusion_full as P
from _common import BASELINE, build_X, cumulative_bounded_up_to, load_data, compute_purity, per_type_purity
from _loaders import load_text, load_visual


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt_raw, txt_idx = load_text(gids, 'v7', 'single')
    X_vis_raw, vis_idx = load_visual(gids, 'siglip', 'colored')
    common = np.intersect1d(txt_idx, vis_idx)
    geo_b = geo[common]
    txt_b = X_txt_raw[np.searchsorted(txt_idx, common)]
    vis_b = X_vis_raw[np.searchsorted(vis_idx, common)]
    y = y_full[common]
    n = len(common)
    print(f'n={n} | geo={geo_b.shape[1]} text={txt_b.shape[1]} vis={vis_b.shape[1]}')

    subsets = [
        ('geo+text', {'geo': geo_b, 'text': txt_b}),
        ('geo+visual', {'geo': geo_b, 'visual': vis_b}),
        ('geo+text+visual', {'geo': geo_b, 'text': txt_b, 'visual': vis_b}),
    ]

    # Quick HDBSCAN config (the one we know is good)
    def fit_check(label, X):
        h = HDBSCAN(min_cluster_size=5, min_samples=3,
                    cluster_selection_method='leaf')
        lab = h.fit_predict(X)
        # force-assign noise
        non_noise = lab != -1
        if non_noise.sum() > 0 and (-1 in lab):
            cids = sorted(set(lab[non_noise].tolist()))
            cents = np.stack([X[lab == c].mean(0) for c in cids])
            ni = np.where(~non_noise)[0]
            d = np.linalg.norm(X[ni][:, None, :] - cents[None, :, :], axis=2)
            lab[ni] = np.array(cids)[d.argmin(1)]
        k = len(set(lab.tolist())) - (1 if -1 in lab else 0)
        pur = float(compute_purity(lab, y))
        tp = per_type_purity(lab, y)
        macro = float(np.mean(list(tp.values()))) if tp else 0.0
        n_unique = len(set(lab.tolist()))
        # sanity assertions
        assert np.all(np.isfinite(X)), f'{label}: non-finite in X'
        assert n_unique >= 2, f'{label}: only {n_unique} cluster(s)'
        print(f'  {label:35s}  d={X.shape[1]:4d}  k={k:3d}  pur={pur:.3f}  macro={macro:.3f}')

    for subset_label, blocks in subsets:
        print(f'\n=== subset: {subset_label} ===')
        # F1a (PCA-32)
        X, _ = P.build_F1a(blocks, 32)
        fit_check('F1a pca_32', X)
        # F1a_umap (UMAP-8)
        X, _ = P.build_F1a_umap(blocks, 8, f'smoke_{subset_label}')
        fit_check('F1a_umap umap_8', X)
        # F1b (PCA-on-concat, k=32)
        X, _ = P.build_F1b(blocks, 32)
        fit_check('F1b pca_32_on_concat', X)
        # F2 raw
        X, _ = P.build_F2(blocks)
        fit_check('F2 raw_concat', X)
        # F3 equal
        X, _ = P.build_F3(blocks, weights=P.WEIGHT_SCHEMES['equal'])
        fit_check('F3 var_balanced', X)
        # F3w geo-heavy
        X, _ = P.build_F3(blocks, weights=P.WEIGHT_SCHEMES['geo_heavy'])
        fit_check('F3w geo_heavy', X)
        # F3w embed-heavy
        X, _ = P.build_F3(blocks, weights=P.WEIGHT_SCHEMES['embed_heavy'])
        fit_check('F3w embed_heavy', X)
        # F5 (UMAP-on-concat, k=8)
        X, _ = P.build_F5(blocks, 8, f'smoke_{subset_label}_F5')
        fit_check('F5 umap_8_on_concat', X)
        # F6 L2-norm
        X, _ = P.build_F6(blocks)
        fit_check('F6 l2_norm_concat', X)
        # F7 CCA-8
        try:
            X, meta = P.build_F7(blocks, 8)
            fit_check(f'F7 cca_8 (eff={meta["cca_k_effective"]})', X)
        except Exception as e:
            print(f'  F7 cca_8                            FAILED: {e}')

    print('\nSMOKE TEST OK')


if __name__ == '__main__':
    main()
