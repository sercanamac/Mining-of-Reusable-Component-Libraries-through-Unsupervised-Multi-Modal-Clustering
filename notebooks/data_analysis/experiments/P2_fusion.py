"""P2 — Multi-modal fusion ablation (post-addendum §12.4).

Fusion recipes evaluated per subset:
  F1a. Per-block PCA then concat with raw geo. k ∈ {8,16,32,64,128}.
       (geo is raw 9-d; text/visual reduced to k then std-scaled.)
  F2.  Raw concat + std (legacy baseline).
  F3.  Variance-balanced concat + std (legacy).
  F5.  UMAP on std-concat(geo, raw_text, raw_visual). k ∈ {4,8,16,64}.

Subsets evaluated: {geo+text, geo+visual, geo+text+visual}.

Cross-algorithm comparison runs all 4 algos only at F2, F3, F1a-pca32
(three anchor configs per subset). Every other recipe variant runs
HDBSCAN-only to keep the CSV tractable.

HDBSCAN grid: HDBSCAN_GRID_FACTORIAL (60 combos).

Output:
    results/midterm/fusion/fusion.csv
    results/midterm/fusion/fusion_per_type.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual
from _reducers import (PCA_DIMS, UMAP_DIMS, UMAP_PARAMS_GEOFUS, fit_reducer,
                        ReducerConfig, reducer_row_meta)
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep, save_sweep

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'fusion'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_VERSION = 'v7'
TEXT_AGG = 'single'
VISUAL_ENCODER = 'siglip'
VISUAL_VARIANT = 'colorless'   # intentionally colorless: shows system works without color; more generalizable across BIM software

ALGOS_ALL = ['kmeans', 'hdbscan', 'gmm', 'bisecting']
ALGOS_HDB_ONLY = ['hdbscan']


def _weighted(block: np.ndarray) -> np.ndarray:
    s = StandardScaler().fit_transform(block)
    total_var = s.var(axis=0, ddof=0).sum()
    return s / np.sqrt(total_var) if total_var > 1e-9 else s


def _std(X):
    return StandardScaler().fit_transform(X).astype(np.float32)


def _pca(X, k):
    p = PCA(n_components=k, random_state=42).fit(X)
    return p.transform(X).astype(np.float32), float(p.explained_variance_ratio_.sum())


def _build_F1a(blocks: dict, k: int) -> tuple[np.ndarray, dict]:
    """Per-block PCA of text/visual to k, then std, concat with raw geo (std)."""
    parts, pca_meta = [], {}
    if 'geo' in blocks:
        parts.append(_std(blocks['geo']))
    for name in ('text', 'visual'):
        if name in blocks:
            Xk, var = _pca(blocks[name], k)
            parts.append(_std(Xk))
            pca_meta[f'{name}_var_ret'] = round(var, 4)
    X = np.concatenate(parts, axis=1)
    return X, pca_meta


def _build_F5(blocks: dict, k: int, modality_name: str) -> tuple[np.ndarray, dict]:
    """std-concat(blocks) then UMAP to k."""
    parts = [_std(blocks[n]) for n in blocks]
    concat = np.concatenate(parts, axis=1)
    cfg = ReducerConfig(f'umap_{k}', 'umap', k, dict(UMAP_PARAMS_GEOFUS))
    Xr, meta = fit_reducer(cfg, concat, modality_name=modality_name)
    return Xr, {'umap_fit_s': round(meta['fit_seconds'], 2),
                'umap_cache_hit': meta.get('cache_hit', False)}


def _build_F2(blocks: dict) -> np.ndarray:
    return np.concatenate([_std(b) for b in blocks.values()], axis=1)


def _build_F3(blocks: dict) -> np.ndarray:
    return np.concatenate([_weighted(b) for b in blocks.values()], axis=1)


def _sweep(name, X, y, extra, all_algos):
    return run_sweep(
        name=name, X=X, y=y,
        algorithms=ALGOS_ALL if all_algos else ALGOS_HDB_ONLY,
        scale=True,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        extra_meta=extra,
        verbose=False,
    )


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt_raw, txt_idx = load_text(gids, TEXT_VERSION, TEXT_AGG)
    X_vis_raw, vis_idx = load_visual(gids, VISUAL_ENCODER, VISUAL_VARIANT)
    common = np.intersect1d(txt_idx, vis_idx)
    print(f'[P2] aligned intersection: {len(common)} / {len(gids)} objects')

    def _align(X_raw, idx_arr):
        pos = np.searchsorted(idx_arr, common)
        return X_raw[pos]

    geo_b = geo[common]
    txt_b = _align(X_txt_raw, txt_idx)
    vis_b = _align(X_vis_raw, vis_idx)
    y = y_full[common]
    print(f'  block dims: geo={geo_b.shape[1]} text={txt_b.shape[1]} '
          f'visual={vis_b.shape[1]}')

    # Three subsets per plan §12.4
    SUBSETS = [
        ('geo+text', {'geo': geo_b, 'text': txt_b}),
        ('geo+visual', {'geo': geo_b, 'visual': vis_b}),
        ('geo+text+visual', {'geo': geo_b, 'text': txt_b, 'visual': vis_b}),
    ]

    all_summary, all_per_type = [], []

    for subset_label, blocks in SUBSETS:
        print(f'\n[P2] === subset: {subset_label} ===')

        # F2: raw concat + std ─ ALL ALGOS
        X = _build_F2(blocks)
        extra = {'subset': subset_label, 'recipe': 'F2',
                 'variant': 'raw_concat', 'out_dim': int(X.shape[1]),
                 'text_version': TEXT_VERSION,
                 'visual_encoder': f'{VISUAL_ENCODER}_{VISUAL_VARIANT}'}
        print(f'  F2             d={X.shape[1]:5d} ALL')
        s, pt = _sweep(f'{subset_label}__F2', X, y, extra, all_algos=True)
        all_summary.append(s); all_per_type.append(pt)

        # F3: variance-balanced + std ─ ALL ALGOS
        X = _build_F3(blocks)
        extra = {'subset': subset_label, 'recipe': 'F3',
                 'variant': 'var_balanced', 'out_dim': int(X.shape[1]),
                 'text_version': TEXT_VERSION,
                 'visual_encoder': f'{VISUAL_ENCODER}_{VISUAL_VARIANT}'}
        print(f'  F3             d={X.shape[1]:5d} ALL')
        s, pt = _sweep(f'{subset_label}__F3', X, y, extra, all_algos=True)
        all_summary.append(s); all_per_type.append(pt)

        # F1a: per-block PCA → concat with geo
        for k in PCA_DIMS:
            X, pca_meta = _build_F1a(blocks, k)
            all_algos = (k == 32)
            extra = {'subset': subset_label, 'recipe': 'F1a',
                     'variant': f'pca_{k}_then_concat_geo',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]),
                     'text_version': TEXT_VERSION,
                     'visual_encoder': f'{VISUAL_ENCODER}_{VISUAL_VARIANT}',
                     **pca_meta}
            tag = 'ALL' if all_algos else 'HDB'
            print(f'  F1a pca_{k:<3d}    d={X.shape[1]:5d} {tag}')
            s, pt = _sweep(f'{subset_label}__F1a_pca{k}', X, y, extra, all_algos=all_algos)
            all_summary.append(s); all_per_type.append(pt)

        # F5: UMAP on full std-concat
        mod_name = f'fusion_{subset_label.replace("+", "_")}'
        for k in UMAP_DIMS:
            X, umap_meta = _build_F5(blocks, k, mod_name)
            extra = {'subset': subset_label, 'recipe': 'F5',
                     'variant': f'umap_{k}_on_concat',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]),
                     'text_version': TEXT_VERSION,
                     'visual_encoder': f'{VISUAL_ENCODER}_{VISUAL_VARIANT}',
                     **umap_meta}
            cache_tag = '(cache)' if umap_meta.get('umap_cache_hit') else ''
            print(f'  F5 umap_{k:<3d}   d={X.shape[1]:5d} HDB  umap_fit={umap_meta["umap_fit_s"]:.1f}s {cache_tag}')
            s, pt = _sweep(f'{subset_label}__F5_umap{k}', X, y, extra, all_algos=False)
            all_summary.append(s); all_per_type.append(pt)

    # Unique run_ids across cells
    offset = 0
    for s, pt in zip(all_summary, all_per_type):
        s['run_id'] = s['run_id'].to_numpy() + offset
        pt['run_id'] = pt['run_id'].to_numpy() + offset
        offset = int(s['run_id'].max()) + 1
    summary = pd.concat(all_summary, ignore_index=True)
    per_type = pd.concat(all_per_type, ignore_index=True)

    save_sweep(
        summary, per_type,
        summary_csv=OUT_DIR / 'fusion.csv',
        per_type_csv=OUT_DIR / 'fusion_per_type.csv',
    )

    key = ['subset', 'recipe', 'variant', 'algo']
    idx = summary.groupby(key)['macro_purity'].idxmax()
    best = summary.loc[idx, key + ['purity', 'macro_purity', 'k', 'noise_frac']]
    print('\n[P2] best per (subset × recipe × variant × algo) by macro purity:')
    print(best.sort_values(key).to_string(index=False))


if __name__ == '__main__':
    main()
