"""P3 — Supervised ceiling (Random Forest 5-fold CV) on each representation.

Runs RandomForestClassifier with 5-fold StratifiedKFold CV across 5 seeds for
every feature-block subset used in P2. Records per-type recall, macro recall,
and overall accuracy.

Output:
    results/midterm/supervised/ceiling.csv       — one row per (subset, seed, type)
    results/midterm/supervised/ceiling_summary.csv — one row per (subset, seed)

Note: runs with n_jobs=1 inside joblib to avoid macOS sandbox semaphore issues.
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'supervised'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_VERSION = 'v7'
TEXT_AGG = 'single'
VISUAL_ENCODER = 'siglip'
VISUAL_VARIANT = 'colorless'   # match new P2 canonical
SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 5
PCA_DIMS = [8, 64]             # §12.7 sanity: PCA-8 alongside existing 64


def _rf_cv(X, y, seed):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y, dtype=object)
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(
            n_estimators=400, random_state=seed,
            class_weight='balanced', n_jobs=1,
        )
        rf.fit(X[tr], y[tr])
        y_pred[te] = rf.predict(X[te])
    return y_pred


def _row(subset, seed, y, y_pred):
    types = sorted(set(y.tolist()))
    acc = float((y_pred == y).mean())
    recalls = {}
    for t in types:
        mask = y == t
        recalls[t] = float((y_pred[mask] == t).mean()) if mask.sum() else 0.0
    macro = float(np.mean(list(recalls.values())))
    return dict(subset=subset, seed=seed, accuracy=acc, macro_recall=macro), recalls


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt_raw, txt_idx = load_text(gids, TEXT_VERSION, TEXT_AGG)
    X_vis_raw, vis_idx = load_visual(gids, VISUAL_ENCODER, VISUAL_VARIANT)
    common = np.intersect1d(txt_idx, vis_idx)
    print(f'[P3] aligned intersection: {len(common)} objects')

    def _align(X_raw, idx_arr):
        pos = np.searchsorted(idx_arr, common)
        return X_raw[pos]

    geo_b = geo[common].astype(np.float64)
    txt_raw = _align(X_txt_raw, txt_idx).astype(np.float64)
    vis_raw = _align(X_vis_raw, vis_idx).astype(np.float64)
    y = y_full[common]
    print(f'  raw block dims: geo={geo_b.shape[1]}, '
          f'text={txt_raw.shape[1]}, visual={vis_raw.shape[1]}')

    summary_rows = []
    per_type_rows = []
    for pca_dim in PCA_DIMS:
        txt_b = PCA(n_components=pca_dim, random_state=42).fit_transform(txt_raw)
        vis_b = PCA(n_components=pca_dim, random_state=42).fit_transform(vis_raw)
        blocks = {'geo': geo_b, 'text': txt_b, 'visual': vis_b}
        subset_order = []
        for r in range(1, 4):
            subset_order.extend(combinations(blocks, r))

        for subset in subset_order:
            sub_label = '+'.join(subset)
            X = np.hstack([blocks[n] for n in subset])
            print(f'\n[P3] pca_dim={pca_dim} subset={sub_label}  d={X.shape[1]}')
            for seed in SEEDS:
                y_pred = _rf_cv(X, y, seed)
                row, recalls = _row(sub_label, seed, y, y_pred)
                row['pca_dim'] = pca_dim
                summary_rows.append(row)
                for t, r in recalls.items():
                    per_type_rows.append(dict(
                        subset=sub_label, seed=seed, pca_dim=pca_dim, type=t,
                        recall=r, support=int((y == t).sum()),
                    ))
                print(f'  pca={pca_dim} seed={seed}  acc={row["accuracy"]:.3f}  '
                      f'macro={row["macro_recall"]:.3f}')

    summary = pd.DataFrame(summary_rows)
    per_type = pd.DataFrame(per_type_rows)
    summary.to_csv(OUT_DIR / 'ceiling.csv', index=False)
    per_type.to_csv(OUT_DIR / 'ceiling_per_type.csv', index=False)
    print(f'\n[P3] saved {len(summary)} summary rows → {OUT_DIR / "ceiling.csv"}')
    print(f'      saved {len(per_type)} per-type rows → {OUT_DIR / "ceiling_per_type.csv"}')

    mean_by_subset = (summary.groupby(['pca_dim', 'subset'])
                       [['accuracy', 'macro_recall']]
                       .agg(['mean', 'std']).round(4))
    print('\n[P3] RF 5-fold CV mean ± std across 5 seeds, per (pca_dim, subset):')
    print(mean_by_subset.to_string())


if __name__ == '__main__':
    main()
