"""F1a-UMAP: per-block UMAP on text/visual, concat with raw geo, HDBSCAN grid.

Same as F1a but swaps PCA for UMAP per modality block.
Only geo+text+visual subset, UMAP dims {4, 8, 16}.
"""
import os, tempfile
os.environ['NUMBA_CACHE_DIR'] = tempfile.mkdtemp()

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'feature_engineering'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual
from _reducers import ReducerConfig, UMAP_PARAMS_TEXTVIS, fit_reducer
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'fusion'

TEXT_VERSION = 'v7'
TEXT_AGG = 'single'
VISUAL_ENCODER = 'siglip'
VISUAL_VARIANT = 'colorless'

UMAP_DIMS = [4, 8, 16]


def _std(X):
    return StandardScaler().fit_transform(X).astype(np.float32)


def _umap_block(X, k, modality_name):
    cfg = ReducerConfig(f'umap_{k}', 'umap', k, dict(UMAP_PARAMS_TEXTVIS))
    Xr, meta = fit_reducer(cfg, X, modality_name=modality_name)
    return Xr, meta


df, y_full = load_data()
gids = df['GlobalId'].values
geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

X_txt_raw, txt_idx = load_text(gids, TEXT_VERSION, TEXT_AGG)
X_vis_raw, vis_idx = load_visual(gids, VISUAL_ENCODER, VISUAL_VARIANT)
common = np.intersect1d(txt_idx, vis_idx)

geo_b = geo[common]
txt_b = X_txt_raw[np.searchsorted(txt_idx, common)]
vis_b = X_vis_raw[np.searchsorted(vis_idx, common)]
y = y_full[common]

print(f'n={len(common)}  geo={geo_b.shape[1]}  text={txt_b.shape[1]}  visual={vis_b.shape[1]}')

all_summary, all_per_type = [], []

for k in UMAP_DIMS:
    print(f'\n--- F1a-UMAP dim={k} ---')
    txt_r, txt_meta = _umap_block(txt_b, k, 'text_v7_single')
    vis_r, vis_meta = _umap_block(vis_b, k, f'visual_{VISUAL_ENCODER}_{VISUAL_VARIANT}')

    X = np.concatenate([_std(geo_b), _std(txt_r), _std(vis_r)], axis=1)
    print(f'  fused dim={X.shape[1]} (geo={geo_b.shape[1]} + text_umap={k} + vis_umap={k})')

    extra = {
        'subset': 'geo+text+visual',
        'recipe': 'F1a_umap',
        'variant': f'umap_{k}',
        'reducer_dim': k,
        'out_dim': int(X.shape[1]),
        'text_version': TEXT_VERSION,
        'visual_encoder': f'{VISUAL_ENCODER}_{VISUAL_VARIANT}',
    }
    s, pt = run_sweep(
        name=f'geo+text+visual__F1a_umap_{k}',
        X=X, y=y,
        algorithms=['hdbscan'],
        scale=True,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        extra_meta=extra,
        verbose=False,
    )
    all_summary.append(s)
    all_per_type.append(pt)

    hdb = s[s['algo'] == 'hdbscan']
    best = hdb.loc[hdb['macro_purity'].idxmax()]
    print(f'  best: macro={best["macro_purity"]:.4f} purity={best["purity"]:.4f} k={int(best["k"])}')

new_summary = pd.concat(all_summary, ignore_index=True)
new_per_type = pd.concat(all_per_type, ignore_index=True)

# Merge into fusion.csv
existing = pd.read_csv(OUT_DIR / 'fusion.csv')
existing_pt = pd.read_csv(OUT_DIR / 'fusion_per_type.csv')
existing = existing[existing['recipe'] != 'F1a_umap']
existing_pt = existing_pt[existing_pt['recipe'] != 'F1a_umap']

offset = int(existing['run_id'].max()) + 1
new_summary['run_id'] += offset
new_per_type['run_id'] += offset

merged = pd.concat([existing, new_summary], ignore_index=True)
merged_pt = pd.concat([existing_pt, new_per_type], ignore_index=True)
merged.to_csv(OUT_DIR / 'fusion.csv', index=False)
merged_pt.to_csv(OUT_DIR / 'fusion_per_type.csv', index=False)

print(f'\nMerged into fusion.csv: {len(merged)} rows (was {len(existing)})')

# Compare with F1a PCA-8 winner
f1a_pca = existing[(existing['recipe'] == 'F1a') &
                    (existing['subset'] == 'geo+text+visual') &
                    (existing['algo'] == 'hdbscan')]
if not f1a_pca.empty:
    pca_best = f1a_pca.loc[f1a_pca['macro_purity'].idxmax()]
    umap_best = new_summary.loc[new_summary['macro_purity'].idxmax()]
    print(f'\n=== F1a PCA vs F1a UMAP (geo+text+visual, HDBSCAN) ===')
    print(f'  F1a PCA-{int(pca_best["reducer_dim"])}: macro={pca_best["macro_purity"]:.4f} purity={pca_best["purity"]:.4f} k={int(pca_best["k"])}')
    print(f'  F1a UMAP-{int(umap_best["reducer_dim"])}: macro={umap_best["macro_purity"]:.4f} purity={umap_best["purity"]:.4f} k={int(umap_best["k"])}')
    diff = float(umap_best["macro_purity"]) - float(pca_best["macro_purity"])
    print(f'  delta: {diff:+.4f} ({"UMAP wins" if diff > 0 else "PCA wins"})')
