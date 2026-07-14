"""P_trimodal_fusion — trimodal (geo + text + visual) fusion sweep.

Works for either dataset. Trimodal blocks:
  - geo: 9-d hand-crafted (baseline OBB + 5 engineered)
  - text: Gemma-4 (or any other) text embedding
  - visual: DINOv3 / SigLIP / DuoDuo / Gemini-vision

For each of 13 fusion strategies builds the representation, runs HDBSCAN
factorial unsup grid (if d <= 1024), and runs RF 5-seed supervised:
  - Bentley → 5-fold CV (no official split)
  - IFCNet  → train→test (official split)

Resume-safe: per-strategy incremental write to summary.csv. Re-running picks up
where it left off.

Outputs:
    results/{spec}/trimodal_{vision_encoder}_{vision_variant}_{text_version}/
      summary.csv
      hdbscan_full.csv

Usage:
    # Bentley with SigLIP-colored + Gemma4 text
    python P_trimodal_fusion.py --dataset bentley --vision siglip --variant colored

    # IFCNet with SigLIP-colorless + Gemma4 text
    IFCNET_DATASET=ifcnet python P_trimodal_fusion.py --dataset ifcnet \\
        --vision siglip --variant colorless
"""
from __future__ import annotations
import argparse
import os, sys, tempfile, time
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'feature_engineering'))
sys.path.insert(0, str(_HERE))

# Import _common AFTER env var IFCNET_DATASET is potentially set by caller
import _common
import P2c_fusion_full as P
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, _force_assign, _metrics

SEEDS = [42, 43, 44, 45, 46]
HDBSCAN_DIM_CAP = 1100


def _strategies_for(tag: str) -> list:
    """Build strategy list with dataset-specific UMAP cache tags."""
    t = f'trimod_{tag}'
    return [
        ('F1a',            P.build_F1a,         {},            (16,)),
        ('F1a_umap',       P.build_F1a_umap,    {},            (16, t)),
        ('F1b',            P.build_F1b,         {},            (16,)),
        ('F1b_varbal',     P.build_F1b_varbal,  {},            (16,)),
        ('F2',             P.build_F2,          {},            ()),
        ('F3',             P.build_F3,          {'weights': P.WEIGHT_SCHEMES['equal']}, ()),
        ('F3w_geo',        P.build_F3,          {'weights': P.WEIGHT_SCHEMES['geo_heavy']}, ()),
        ('F3w_embed',      P.build_F3,          {'weights': P.WEIGHT_SCHEMES['embed_heavy']}, ()),
        ('F5',             P.build_F5,          {},            (16, f'{t}_F5')),
        ('F5_varbal',      P.build_F5_varbal,   {},            (16, f'{t}_F5vb')),
        ('F6',             P.build_F6,          {},            ()),
        ('F7',             P.build_F7,          {},            (8,)),
    ]


def _rf_eval(Xs, y, has_splits: bool, train_mask=None, test_mask=None) -> dict:
    """RF 5 seeds. honor_split=True → single train→test; else stratified 5-fold CV."""
    rows = []
    if has_splits:
        Xtr, Xte = Xs[train_mask], Xs[test_mask]
        ytr, yte = y[train_mask], y[test_mask]
        for seed in SEEDS:
            rf = RandomForestClassifier(n_estimators=400, random_state=seed,
                                        class_weight='balanced', n_jobs=-1)
            rf.fit(Xtr, ytr)
            yp = rf.predict(Xte)
            rows.append(dict(
                seed=seed,
                accuracy=float((yp == yte).mean()),
                precision_macro=float(precision_score(yte, yp, average='macro', zero_division=0)),
                recall_macro=float(recall_score(yte, yp, average='macro', zero_division=0)),
                f1_macro=float(f1_score(yte, yp, average='macro', zero_division=0)),
            ))
    else:
        # 5-fold stratified CV with 1 seed for the splits, plus 5 RF seeds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for seed in SEEDS:
            y_pred = np.empty_like(y, dtype=object)
            for tr, te in skf.split(Xs, y):
                rf = RandomForestClassifier(n_estimators=400, random_state=seed,
                                            class_weight='balanced', n_jobs=-1)
                rf.fit(Xs[tr], y[tr])
                y_pred[te] = rf.predict(Xs[te])
            rows.append(dict(
                seed=seed,
                accuracy=float((y_pred == y).mean()),
                precision_macro=float(precision_score(y, y_pred, average='macro', zero_division=0)),
                recall_macro=float(recall_score(y, y_pred, average='macro', zero_division=0)),
                f1_macro=float(f1_score(y, y_pred, average='macro', zero_division=0)),
            ))
    df = pd.DataFrame(rows)
    return {f'{m}_{stat}': float(df[m].agg(stat))
            for m in ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro']
            for stat in ('mean', 'std')}


def _hdbscan_sweep(Xs, y) -> tuple[float, dict, list[dict]]:
    best_macro = -1.0
    best_hps = {}
    all_rows = []
    for (mcs, ms, method, force) in HDBSCAN_GRID_FACTORIAL:
        lab_raw = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                          cluster_selection_method=method, n_jobs=-1).fit_predict(Xs)
        lab = _force_assign(lab_raw, Xs) if force else lab_raw
        metrics, _ = _metrics(Xs, lab, y)
        all_rows.append(dict(mcs=mcs, ms=ms, method=method, force=force, **metrics))
        if metrics['macro_purity'] > best_macro:
            best_macro = metrics['macro_purity']
            best_hps = dict(mcs=mcs, ms=ms, method=method, force=force, k=metrics['k'])
    return best_macro, best_hps, all_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['bentley', 'ifcnet'], required=True)
    ap.add_argument('--vision', required=True,
                    help='visual encoder key (siglip | dinov3 | duoduo | gemini)')
    ap.add_argument('--variant', default=None,
                    help='visual variant (colored|colorless). '
                         'Defaults: bentley=colored, ifcnet=colorless')
    ap.add_argument('--text-version', default='gemma4')
    ap.add_argument('--text-aggregation', default='single')
    args = ap.parse_args()

    os.environ['IFCNET_DATASET'] = args.dataset
    import importlib
    importlib.reload(_common)
    from _common import ACTIVE_SPEC, BASELINE, build_X, cumulative_bounded_up_to, load_data  # noqa
    # _loaders depends on _common — reload too
    import _loaders
    importlib.reload(_loaders)
    from _loaders import load_text, load_visual

    spec = ACTIVE_SPEC
    assert spec.name == args.dataset, f'spec mismatch: {spec.name} vs {args.dataset}'
    variant = args.variant or ('colored' if spec.name == 'bentley' else 'colorless')
    tag = f'{args.vision}_{variant}_{args.text_version}_{args.text_aggregation}'
    out_dir = spec.results_root / f'trimodal_{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / 'summary.csv'
    hdbscan_csv = out_dir / 'hdbscan_full.csv'

    print(f'[{spec.name}] vision={args.vision}/{variant}  '
          f'text={args.text_version}/{args.text_aggregation}  output={out_dir}')

    df, y_full = load_data()
    gids_all = df[spec.key_col].values
    geo_full = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))
    X_txt, txt_idx = load_text(gids_all, args.text_version, args.text_aggregation)
    X_vis, vis_idx = load_visual(gids_all, args.vision, variant)

    common = np.intersect1d(txt_idx, vis_idx)
    df_a = df.iloc[common].reset_index(drop=True)
    geo = geo_full[common].astype(np.float64)
    txt = X_txt[np.searchsorted(txt_idx, common)].astype(np.float64)
    vis = X_vis[np.searchsorted(vis_idx, common)].astype(np.float64)
    y = y_full[common]

    if spec.has_splits:
        train_mask = (df_a['Split'] == 'train').values
        test_mask  = (df_a['Split'] == 'test').values
        print(f'n={len(common)}  geo={geo.shape[1]}  text={txt.shape[1]}  visual={vis.shape[1]}  '
              f'classes={len(set(y))}  train={train_mask.sum()}  test={test_mask.sum()}')
    else:
        train_mask = test_mask = None
        print(f'n={len(common)}  geo={geo.shape[1]}  text={txt.shape[1]}  visual={vis.shape[1]}  '
              f'classes={len(set(y))}  eval=5-fold CV')

    # Resume
    done = set()
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        done = set(prev['strategy'].tolist())
        print(f'[resume] {len(done)} done: {sorted(done)}')

    STRATEGIES = _strategies_for(tag)
    for strat_name, builder, kw, strat_args in STRATEGIES:
        if strat_name in done:
            print(f'  [{strat_name:14s}] skipped (already done)')
            continue
        t0 = time.time()
        blocks = {'geo': geo, 'text': txt, 'visual': vis}
        X, _meta = builder(blocks, *strat_args, **kw)
        Xs = StandardScaler().fit_transform(X).astype(np.float64)

        sup = _rf_eval(Xs, y, spec.has_splits, train_mask, test_mask)

        if X.shape[1] <= HDBSCAN_DIM_CAP:
            unsup_macro, best_hps, hdb_rows = _hdbscan_sweep(Xs, y)
        else:
            unsup_macro, best_hps, hdb_rows = float('nan'), {}, []
            print(f'  [{strat_name}] skipping HDBSCAN (d={X.shape[1]} > {HDBSCAN_DIM_CAP})')

        row = dict(
            strategy=strat_name, dim=int(X.shape[1]),
            unsup_macro_purity=unsup_macro, unsup_best_hps=str(best_hps),
            **sup, elapsed_s=round(time.time() - t0, 1),
        )
        print(f'  [{strat_name:14s}] dim={X.shape[1]:4d}  '
              f'unsup_macro={unsup_macro:.3f}  '
              f'sup_acc={sup["accuracy_mean"]:.3f}  '
              f'sup_f1={sup["f1_macro_mean"]:.3f}  '
              f'sup_rec={sup["recall_macro_mean"]:.3f}  '
              f'({row["elapsed_s"]:.0f}s)')

        # Incremental save
        all_summary = (pd.concat([pd.read_csv(summary_csv), pd.DataFrame([row])],
                                 ignore_index=True)
                       if summary_csv.exists() else pd.DataFrame([row]))
        all_summary.to_csv(summary_csv, index=False)
        if hdb_rows:
            new_hdb = pd.DataFrame([dict(strategy=strat_name, dim=int(X.shape[1]), **r)
                                    for r in hdb_rows])
            all_hdb = (pd.concat([pd.read_csv(hdbscan_csv), new_hdb], ignore_index=True)
                       if hdbscan_csv.exists() else new_hdb)
            all_hdb.to_csv(hdbscan_csv, index=False)

    final = pd.read_csv(summary_csv).sort_values('f1_macro_mean', ascending=False)
    cols = ['strategy', 'dim', 'unsup_macro_purity',
            'accuracy_mean', 'f1_macro_mean', 'recall_macro_mean']
    print('\n=== Headline (sorted by sup F1-macro) ===')
    print(final[cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
