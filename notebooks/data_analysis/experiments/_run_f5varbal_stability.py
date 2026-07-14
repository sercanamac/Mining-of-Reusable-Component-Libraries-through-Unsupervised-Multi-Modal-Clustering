"""Stability + k-matched comparison for the F5_varbal fusion winner.

The thesis headline winner is F5_varbal (variance-balanced concat -> UMAP-4)
on geo+text+visual, v7 text, SigLIP-colored, HDBSCAN leaf/mcs=5/ms=3/force,
k=126, macro purity 0.853 (results/midterm/fusion/fusion_full.csv).

The original P4a (bootstrap) and P6 (k-matched) only ran for the older F1a
winner. This script reproduces the F5_varbal pipeline exactly (matching
P2c_fusion_full.build_F5_varbal + run_sweep scale=True) and computes:

  * bootstrap stability  -> bootstrap_ari_f5varbal.csv, bootstrap_purity_f5varbal.csv
  * k-matched comparison -> k_matched_comparison_f5varbal.csv

It does NOT overwrite the existing F1a CSVs.
"""
from __future__ import annotations
import os, tempfile, sys
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import (BASELINE, RESULTS_ROOT, SEEDS, build_X,
                     compute_purity, cumulative_bounded_up_to, load_data,
                     per_type_purity)
from _loaders import load_text, load_visual
from _reducers import ReducerConfig, UMAP_PARAMS_GEOFUS, fit_reducer
from _sweep_runner import _force_assign

OUT_DIR = RESULTS_ROOT.parent / 'midterm'
STAB_DIR = OUT_DIR / 'stability'
STAB_DIR.mkdir(parents=True, exist_ok=True)

# Winner row, verbatim from fusion_full.csv (run_id 12182)
HPS = dict(mcs=5, ms=3, method='leaf', force=True)
REDUCER_DIM = 4
N_RUNS = 10
SUBSAMPLE_FRAC = 0.90
SEED = 42


def _weighted(block: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """P2c_fusion_full._weighted: variance-balanced block x weight."""
    s = StandardScaler().fit_transform(block)
    total_var = s.var(axis=0, ddof=0).sum()
    if total_var > 1e-9:
        s = s / np.sqrt(total_var)
    return (s * weight).astype(np.float32)


def build_winner_X():
    """Reproduce P2c build_F5_varbal + run_sweep(scale=True) for the winner."""
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

    # build_F5_varbal: variance-balance each block -> concat -> UMAP
    concat = np.concatenate(
        [_weighted(geo_b), _weighted(txt_b), _weighted(vis_b)], axis=1)
    cfg = ReducerConfig(f'umap_{REDUCER_DIM}', 'umap', REDUCER_DIM,
                        dict(UMAP_PARAMS_GEOFUS))
    modality_tag = 'fusion_geo_text_visual_siglip_colored_F5_vb'
    Xr, meta = fit_reducer(cfg, concat, modality_name=modality_tag)
    print(f'[build] UMAP cache_hit={meta.get("cache_hit")} '
          f'concat_dim={concat.shape[1]} -> umap_dim={Xr.shape[1]}')

    # run_sweep applies StandardScaler when scale=True
    Xs = StandardScaler().fit_transform(Xr).astype(np.float64)
    return Xs, y


def _cluster(X, force=True):
    raw = HDBSCAN(min_cluster_size=HPS['mcs'], min_samples=HPS['ms'],
                  cluster_selection_method=HPS['method'],
                  n_jobs=-1).fit_predict(X)
    return _force_assign(raw, X) if force else raw


def _macro(lab, y):
    tp = per_type_purity(lab, y)
    return float(np.mean(list(tp.values()))) if tp else 0.0


def main():
    Xs, y = build_winner_X()
    print(f'[winner] X shape={Xs.shape}')

    # ---- verify reproduction of the 0.853 / k=126 figure ------------------
    lab = _cluster(Xs, force=True)
    k = len(set(lab.tolist()))
    macro = _macro(lab, y)
    pur = float(compute_purity(lab, y))
    print(f'[verify] HDBSCAN leaf/mcs5/ms3/force -> k={k} '
          f'purity={pur:.4f} macro={macro:.4f}  (target: k=126 macro=0.853)')

    # ---- bootstrap stability (mirrors P4a) --------------------------------
    rng = np.random.default_rng(SEED)
    n = len(Xs)
    subs = [rng.choice(n, int(n * SUBSAMPLE_FRAC), replace=False)
            for _ in range(N_RUNS)]
    runs, purs, macros = [], [], []
    for i, idx in enumerate(subs):
        l = _cluster(Xs[idx], force=True)
        runs.append((idx, l))
        purs.append(float(compute_purity(l, y[idx])))
        macros.append(_macro(l, y[idx]))
        print(f'  run {i+1}/{N_RUNS}: n={len(idx)} k={len(set(l))} '
              f'pur={purs[-1]:.4f} macro={macros[-1]:.4f}')

    ari_rows = []
    for i in range(N_RUNS):
        for j in range(i + 1, N_RUNS):
            idx_i, lab_i = runs[i]
            idx_j, lab_j = runs[j]
            overlap = np.intersect1d(idx_i, idx_j)
            oi = np.argsort(idx_i)
            oj = np.argsort(idx_j)
            pi = np.searchsorted(np.sort(idx_i), overlap)
            pj = np.searchsorted(np.sort(idx_j), overlap)
            ari = adjusted_rand_score(lab_i[oi][pi], lab_j[oj][pj])
            ari_rows.append(dict(run_i=i, run_j=j,
                                 n_overlap=len(overlap), ari=ari))
    ari_df = pd.DataFrame(ari_rows)
    pur_df = pd.DataFrame(dict(run=range(N_RUNS), purity=purs,
                               macro_purity=macros,
                               n_sub=[len(s) for s in subs]))
    ari_df.to_csv(STAB_DIR / 'bootstrap_ari_f5varbal.csv', index=False)
    pur_df.to_csv(STAB_DIR / 'bootstrap_purity_f5varbal.csv', index=False)
    print(f'\n[bootstrap] ARI = {ari_df.ari.mean():.3f} +/- {ari_df.ari.std():.3f} '
          f'({len(ari_df)} pairs)')
    print(f'[bootstrap] macro purity = {pur_df.macro_purity.mean():.3f} '
          f'+/- {pur_df.macro_purity.std():.3f}')

    # ---- k-matched comparison (mirrors P6) --------------------------------
    raw = HDBSCAN(min_cluster_size=HPS['mcs'], min_samples=HPS['ms'],
                  cluster_selection_method=HPS['method'],
                  n_jobs=-1).fit_predict(Xs)
    forced = _force_assign(raw, Xs)
    k_forced = len(set(forced.tolist()))
    noise = float((raw == -1).mean())
    nn = raw != -1
    rows = [
        dict(method='HDBSCAN force=True', k=k_forced, seed='-',
             purity=float(compute_purity(forced, y)),
             macro_purity=_macro(forced, y), noise_frac=0.0,
             note=f'mcs={HPS["mcs"]} ms={HPS["ms"]} {HPS["method"]}'),
        dict(method='HDBSCAN force=False (assigned only)',
             k=len(set(raw[nn].tolist())), seed='-',
             purity=float(compute_purity(raw[nn], y[nn])),
             macro_purity=_macro(raw[nn], y[nn]), noise_frac=noise,
             note=f'{int(nn.sum())}/{len(raw)} assigned'),
    ]
    target_k = k_forced
    for kk, note in [(target_k, ''), (17, 'label-count reference')]:
        for seed in SEEDS:
            l = MiniBatchKMeans(n_clusters=kk, batch_size=1024,
                                n_init='auto', random_state=seed).fit_predict(Xs)
            rows.append(dict(method=f'k-means k={kk}', k=kk, seed=seed,
                             purity=float(compute_purity(l, y)),
                             macro_purity=_macro(l, y), noise_frac=0.0,
                             note=note))
    km = pd.DataFrame(rows)
    km.to_csv(OUT_DIR / 'k_matched_comparison_f5varbal.csv', index=False)

    def _ms(method, kk):
        s = km[(km.method == method) & (km.k == kk)]
        return s.macro_purity.mean(), s.macro_purity.std()
    kmean, kstd = _ms(f'k-means k={target_k}', target_k)
    print(f'\n[k-matched] at k={target_k}:')
    print(f'  HDBSCAN force=True : macro={rows[0]["macro_purity"]:.4f}')
    print(f'  k-means k={target_k}      : macro={kmean:.4f} +/- {kstd:.4f} (10 seeds)')
    print(f'  HDBSCAN force=False: macro={rows[1]["macro_purity"]:.4f} '
          f'(noise {noise:.1%})')
    print(f'\nsaved -> stability/bootstrap_ari_f5varbal.csv')
    print(f'saved -> stability/bootstrap_purity_f5varbal.csv')
    print(f'saved -> k_matched_comparison_f5varbal.csv')


if __name__ == '__main__':
    main()
