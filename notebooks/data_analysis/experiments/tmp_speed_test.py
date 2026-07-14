"""Quick per-(modality, reducer) timing for one HDBSCAN fit.

Feeds the §12 cost budget with measured rather than guessed numbers.
Prints a table at the end.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent / 'feature_engineering'))

from _common import load_data, build_X
from _loaders import load_text, load_visual

HDBSCAN_KW = dict(min_cluster_size=5, min_samples=3,
                  cluster_selection_method='eom', core_dist_n_jobs=1)

PCA_DIMS = [8, 16, 32, 64, 128]
UMAP_DIMS = [4, 8, 16, 64]
UMAP_KW = dict(n_neighbors=30, min_dist=0.0, random_state=42)


def time_reducer(kind, n_components, X, metric='cosine'):
    t0 = time.perf_counter()
    if kind == 'identity':
        Xr = X
    elif kind == 'pca':
        Xr = PCA(n_components=n_components, random_state=42).fit_transform(X)
    elif kind == 'umap':
        import umap
        Xr = umap.UMAP(n_components=n_components, metric=metric, **UMAP_KW).fit_transform(X)
    else:
        raise ValueError(kind)
    return Xr, time.perf_counter() - t0


def time_hdbscan(X):
    t0 = time.perf_counter()
    lab = hdbscan.HDBSCAN(**HDBSCAN_KW).fit_predict(X)
    return lab, time.perf_counter() - t0


def bench_single(name, X, metric):
    rows = []
    # identity
    _, _ = time_reducer('identity', None, X)  # warm
    lab, th = time_hdbscan(X)
    rows.append({'modality': name, 'reducer': 'identity', 'dim': X.shape[1],
                 'red_s': 0.0, 'hdb_s': th, 'k': len(set(lab)) - (1 if -1 in lab else 0),
                 'noise': int((lab == -1).sum())})
    # PCA
    for k in PCA_DIMS:
        Xr, tr = time_reducer('pca', k, X)
        lab, th = time_hdbscan(Xr)
        rows.append({'modality': name, 'reducer': f'pca_{k}', 'dim': k,
                     'red_s': tr, 'hdb_s': th,
                     'k': len(set(lab)) - (1 if -1 in lab else 0),
                     'noise': int((lab == -1).sum())})
    # UMAP
    for k in UMAP_DIMS:
        Xr, tr = time_reducer('umap', k, X, metric=metric)
        lab, th = time_hdbscan(Xr)
        rows.append({'modality': name, 'reducer': f'umap_{k}', 'dim': k,
                     'red_s': tr, 'hdb_s': th,
                     'k': len(set(lab)) - (1 if -1 in lab else 0),
                     'noise': int((lab == -1).sum())})
    return rows


def bench_fusion_pca(geo, text, visual):
    """F1a: per-block PCA then concat with geo. Time per k."""
    rows = []
    geo_std = StandardScaler().fit_transform(geo)
    for k in PCA_DIMS:
        t0 = time.perf_counter()
        tpca = PCA(n_components=k, random_state=42).fit_transform(text)
        vpca = PCA(n_components=k, random_state=42).fit_transform(visual)
        tpca = StandardScaler().fit_transform(tpca)
        vpca = StandardScaler().fit_transform(vpca)
        X = np.concatenate([geo_std, tpca, vpca], axis=1)
        tr = time.perf_counter() - t0
        lab, th = time_hdbscan(X)
        rows.append({'modality': 'fusion_F1a', 'reducer': f'pca_{k}',
                     'dim': X.shape[1], 'red_s': tr, 'hdb_s': th,
                     'k': len(set(lab)) - (1 if -1 in lab else 0),
                     'noise': int((lab == -1).sum())})
    return rows


def bench_fusion_umap(geo, text, visual):
    """F5: std-concat(geo, raw_text, raw_visual) then UMAP. Time per k."""
    import umap
    rows = []
    geo_std = StandardScaler().fit_transform(geo)
    text_std = StandardScaler().fit_transform(text)
    vis_std = StandardScaler().fit_transform(visual)
    concat = np.concatenate([geo_std, text_std, vis_std], axis=1)
    for k in UMAP_DIMS:
        t0 = time.perf_counter()
        X = umap.UMAP(n_components=k, metric='cosine', **UMAP_KW).fit_transform(concat)
        tr = time.perf_counter() - t0
        lab, th = time_hdbscan(X)
        rows.append({'modality': 'fusion_F5', 'reducer': f'umap_{k}',
                     'dim': X.shape[1], 'red_s': tr, 'hdb_s': th,
                     'k': len(set(lab)) - (1 if -1 in lab else 0),
                     'noise': int((lab == -1).sum())})
    return rows


def main():
    print('=== loading data ===', flush=True)
    df, _ = load_data()
    gids = df['GlobalId'].to_numpy()
    X_geo = build_X(df)
    print(f'geo: {X_geo.shape}')

    X_text, idx_t = load_text(gids, 'v7', 'single')
    print(f'text v7/single: {X_text.shape} aligned={len(idx_t)}')

    X_vis, idx_v = load_visual(gids, 'siglip', 'colorless')
    print(f'visual siglip/colorless: {X_vis.shape} aligned={len(idx_v)}')

    # Align intersection for fusion
    common = np.intersect1d(idx_t, idx_v)
    print(f'intersection for fusion: {len(common)}')
    # Re-index text/visual to common
    t_mask = np.isin(idx_t, common)
    v_mask = np.isin(idx_v, common)
    geo_f = X_geo[common]
    text_f = X_text[t_mask]
    vis_f = X_vis[v_mask]

    all_rows = []
    print('\n=== geo ===', flush=True)
    all_rows += [{'modality': 'geo', 'reducer': 'identity', 'dim': X_geo.shape[1],
                  'red_s': 0.0, **{k: v for k, v in zip(['hdb_s', 'k', 'noise'],
                  _hdb_summary(X_geo))}}]

    print('\n=== text (v7/single, 768d) ===', flush=True)
    all_rows += bench_single('text_v7', X_text, metric='cosine')

    print('\n=== visual (siglip/colorless, 1024d) ===', flush=True)
    all_rows += bench_single('visual_siglip', X_vis, metric='cosine')

    print('\n=== fusion F1a (geo + PCA_k(text) + PCA_k(visual)) ===', flush=True)
    all_rows += bench_fusion_pca(geo_f, text_f, vis_f)

    print('\n=== fusion F5 (UMAP_k on std-concat(geo, text, visual)) ===', flush=True)
    all_rows += bench_fusion_umap(geo_f, text_f, vis_f)

    out = pd.DataFrame(all_rows)
    out_path = _EXP.parent / 'results' / 'midterm' / 'tmp_speed_test.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')

    # Pretty print
    print('\n================= TIMING TABLE =================')
    print(out.to_string(index=False, float_format=lambda x: f'{x:7.3f}'))

    # Extrapolations
    print('\n================= EXTRAPOLATIONS =================')
    # Per-modality: sum of all hdbscan fits in one cell × 60 HDBSCAN grid combos
    for m in out['modality'].unique():
        sub = out[out['modality'] == m]
        mean_hdb = sub['hdb_s'].mean()
        mean_red = sub['red_s'].mean()
        print(f'  {m:18s}  mean reducer-fit {mean_red:6.2f}s  mean hdbscan {mean_hdb:6.3f}s')

    # Full sweep estimate — use measured mean hdbscan per modality
    print('\nFull-sweep estimate (measured hdbscan × 60 grid combos):')
    est = {}
    # P1b text: 9 versions × 10 reducers × 60
    hdb_text = out[out['modality'] == 'text_v7']['hdb_s'].mean()
    red_text = out[out['modality'] == 'text_v7']['red_s'].mean()
    t_sec = 9 * 10 * (60 * hdb_text + red_text)
    est['P1b_text_versions'] = t_sec
    # P1b aggregation ablation: 4 aggs × 10 reducers × 60
    t_sec = 4 * 10 * (60 * hdb_text + red_text)
    est['P1b_agg_ablation'] = t_sec
    # P1c visual: 6 × 10 × 60
    hdb_vis = out[out['modality'] == 'visual_siglip']['hdb_s'].mean()
    red_vis = out[out['modality'] == 'visual_siglip']['red_s'].mean()
    t_sec = 6 * 10 * (60 * hdb_vis + red_vis)
    est['P1c_visual'] = t_sec
    # P2 fusion: 36 configs × 60 — use mixed fusion stats
    fus = out[out['modality'].str.startswith('fusion')]
    hdb_fus = fus['hdb_s'].mean()
    red_fus = fus['red_s'].mean()
    t_sec = 36 * (60 * hdb_fus + red_fus)
    est['P2_fusion'] = t_sec
    # P1a geo: 1 × 60
    hdb_geo = out[out['modality'] == 'geo']['hdb_s'].mean()
    t_sec = 60 * hdb_geo
    est['P1a_geo'] = t_sec

    total = sum(est.values())
    for k, v in est.items():
        print(f'  {k:20s}  {v/60:6.1f} min')
    print(f'  {"TOTAL":20s}  {total/60:6.1f} min  ({total/3600:.2f} h)')


def _hdb_summary(X):
    lab = hdbscan.HDBSCAN(**HDBSCAN_KW).fit_predict(X)
    # time again just to measure
    t0 = time.perf_counter()
    _ = hdbscan.HDBSCAN(**HDBSCAN_KW).fit_predict(X)
    th = time.perf_counter() - t0
    k = len(set(lab)) - (1 if -1 in lab else 0)
    noise = int((lab == -1).sum())
    return th, k, noise


if __name__ == '__main__':
    main()
