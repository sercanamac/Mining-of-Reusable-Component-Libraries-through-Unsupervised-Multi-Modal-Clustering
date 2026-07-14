"""P3b — Supervised ceiling on the WINNING fusion representation.

Trains Random Forest 5-fold CV on three nested representations of the
winning F5_varbal pipeline:
  1. var-balanced concat (1801-d)   — same fused input clustering sees pre-UMAP
  2. var-balanced concat → UMAP-4   — exact representation clustering uses
  3. raw std-concat (1801-d)         — sanity baseline (no var-balance)

This tells us: how much signal does UMAP-4 discard, and is var-balance helping
or hurting supervised performance?

Output:
    results/midterm/supervised/winner_ceiling.csv
    results/midterm/supervised/winner_ceiling_per_type.csv
"""
from __future__ import annotations
import os, sys, tempfile
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

import P2c_fusion_full as P
from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'supervised'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 5


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


def _metrics(rep, seed, y, y_pred):
    types = sorted(set(y.tolist()))
    acc = float((y_pred == y).mean())
    recalls = {t: float((y_pred[y == t] == t).mean()) if (y == t).sum() else 0.0
               for t in types}
    macro = float(np.mean(list(recalls.values())))
    return dict(representation=rep, seed=seed, accuracy=acc, macro_recall=macro), recalls


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt, txt_idx = load_text(gids, 'v7', 'single')
    X_vis, vis_idx = load_visual(gids, 'siglip', 'colored')
    common = np.intersect1d(txt_idx, vis_idx)

    geo_b = geo[common].astype(np.float64)
    txt_b = X_txt[np.searchsorted(txt_idx, common)].astype(np.float64)
    vis_b = X_vis[np.searchsorted(vis_idx, common)].astype(np.float64)
    y = y_full[common]
    print(f'n={len(common)} | geo={geo_b.shape[1]} text={txt_b.shape[1]} visual={vis_b.shape[1]}')

    blocks = {'geo': geo_b, 'text': txt_b, 'visual': vis_b}

    # --- Build representations ---
    print('\nBuilding representations...')
    # 1. Raw std-concat (no var-balance, no reduction)
    rep_raw, _ = P.build_F2(blocks)
    # 2. Var-balanced concat (no reduction)
    rep_vb, _ = P.build_F3(blocks, weights=P.WEIGHT_SCHEMES['equal'])
    # 3. F5_varbal → UMAP-4 (the actual winning representation)
    rep_winner, _ = P.build_F5_varbal(blocks, 4, 'supervised_winner')

    reps = [
        ('raw_std_concat (d=1801)', rep_raw),
        ('var_balanced_concat (d=1801)', rep_vb),
        ('F5_varbal_umap_4 (d=4) ★', rep_winner),
    ]

    summary_rows, per_type_rows = [], []
    for rep_name, X in reps:
        print(f'\n=== {rep_name}  shape={X.shape} ===')
        # std-scale before RF (consistent with clustering pipeline)
        Xs = StandardScaler().fit_transform(X)
        for seed in SEEDS:
            y_pred = _rf_cv(Xs, y, seed)
            row, recalls = _metrics(rep_name, seed, y, y_pred)
            summary_rows.append(row)
            for t, r in recalls.items():
                per_type_rows.append(dict(representation=rep_name, seed=seed,
                                          type=t, recall=r,
                                          support=int((y == t).sum())))
            print(f'  seed={seed}  acc={row["accuracy"]:.4f}  macro_recall={row["macro_recall"]:.4f}')

    summary = pd.DataFrame(summary_rows)
    per_type = pd.DataFrame(per_type_rows)
    summary.to_csv(OUT_DIR / 'winner_ceiling.csv', index=False)
    per_type.to_csv(OUT_DIR / 'winner_ceiling_per_type.csv', index=False)
    print(f'\nSaved {len(summary)} summary rows → {OUT_DIR / "winner_ceiling.csv"}')
    print(f'Saved {len(per_type)} per-type rows → {OUT_DIR / "winner_ceiling_per_type.csv"}')

    print('\n=== RF 5-fold CV summary (mean ± std over 5 seeds) ===')
    agg = (summary.groupby('representation')
           [['accuracy', 'macro_recall']]
           .agg(['mean', 'std']).round(4))
    print(agg.to_string())

    # Headline comparison vs unsupervised
    print('\n=== Supervised vs Unsupervised ===')
    print(f'  Unsupervised macro purity (F5_varbal-umap4 + HDBSCAN): 0.853')
    for rep_name in [r[0] for r in reps]:
        m = summary[summary['representation']==rep_name]['macro_recall'].mean()
        print(f'  Supervised macro recall on {rep_name:35s}: {m:.4f}')


if __name__ == '__main__':
    main()
