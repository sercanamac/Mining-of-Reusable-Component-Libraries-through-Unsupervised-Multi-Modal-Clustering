"""P_ifcnet_duoduo_fusion — fusion + vision-only supervised on IFCNet.

Generic across visual encoders — pass --encoder duoduo|siglip|dinov3 (or
any other key registered in _loaders.VISUAL_ENCODERS). For backwards
compatibility, defaults to 'duoduo'.

Inputs
------
- Geo: data/IFCNetCore/features/{baseline,engineered}_features.parquet
- Vision: data/IFCNetCore/processed/rendered_features/<enc>/colorless/{obj_id}.npy

What it does
------------
1. Vision-only supervised RF (train→test) — bare encoder baseline.
2. Fusion sweep (geo+visual) over all 12 strategies from P2c_fusion_full:
   F1a, F1a_umap, F1b, F1b_varbal, F2, F3, F3w_geo, F3w_embed, F5, F5_varbal,
   F6, F7. For each:
     - build representation (with a sensible default dim where applicable)
     - run HDBSCAN factorial grid (60 configs) → report best macro purity
     - run RF 5-seed train→test on the rep → acc / F1-macro / macro recall
3. Incremental per-strategy save (resume-safe) + headline table at the end.

Outputs
-------
- results/ifcnet/fusion_<encoder>/summary.csv
- results/ifcnet/fusion_<encoder>/hdbscan_full.csv
"""
from __future__ import annotations
import argparse
import os, sys, tempfile, time
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
os.environ['IFCNET_DATASET'] = 'ifcnet'
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'feature_engineering'))
sys.path.insert(0, str(_HERE))

import P2c_fusion_full as P
from _common import ACTIVE_SPEC, BASELINE, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_visual
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, _force_assign, _metrics

SEEDS = [42, 43, 44, 45, 46]

def _strategies_for(encoder: str) -> list:
    """Build strategy list with encoder-specific UMAP cache tags."""
    tag = f'ifcnet_{encoder}'
    return [
        ('vision_only',    None,                {},            ()),    # special-cased
        ('F1a',            P.build_F1a,         {},            (16,)),
        ('F1a_umap',       P.build_F1a_umap,    {},            (16, tag)),
        ('F1b',            P.build_F1b,         {},            (16,)),
        ('F1b_varbal',     P.build_F1b_varbal,  {},            (16,)),
        ('F2',             P.build_F2,          {},            ()),
        ('F3',             P.build_F3,          {'weights': P.WEIGHT_SCHEMES['equal']}, ()),
        ('F3w_geo',        P.build_F3,          {'weights': P.WEIGHT_SCHEMES['geo_heavy']}, ()),
        ('F3w_embed',      P.build_F3,          {'weights': P.WEIGHT_SCHEMES['embed_heavy']}, ()),
        ('F5',             P.build_F5,          {},            (16, f'{tag}_F5')),
        ('F5_varbal',      P.build_F5_varbal,   {},            (16, f'{tag}_F5vb')),
        ('F6',             P.build_F6,          {},            ()),
        ('F7',             P.build_F7,          {},            (8,)),
    ]


def _rf_train_test(Xs, y, train_mask, test_mask) -> dict:
    """5-seed RF train→test, return mean+std of acc / macro-f1 / macro-recall."""
    rows = []
    for seed in SEEDS:
        rf = RandomForestClassifier(
            n_estimators=400, random_state=seed,
            class_weight='balanced', n_jobs=-1,
        )
        rf.fit(Xs[train_mask], y[train_mask])
        yp = rf.predict(Xs[test_mask])
        yt = y[test_mask]
        rows.append(dict(
            seed=seed,
            accuracy=float((yp == yt).mean()),
            f1_macro=float(f1_score(yt, yp, average='macro', zero_division=0)),
            recall_macro=float(recall_score(yt, yp, average='macro', zero_division=0)),
            precision_macro=float(precision_score(yt, yp, average='macro', zero_division=0)),
        ))
    df = pd.DataFrame(rows)
    return {f'{m}_{stat}': float(df[m].agg(stat))
            for m in ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro']
            for stat in ('mean', 'std')}


def _hdbscan_sweep(Xs, y) -> tuple[float, dict, list[dict]]:
    """Run the 60-config factorial grid on Xs; return (best_macro_purity, best_hps, all_rows)."""
    best_macro = -1.0
    best_hps = {}
    all_rows = []
    for (mcs, ms, method, force) in HDBSCAN_GRID_FACTORIAL:
        lab_raw = HDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method=method, n_jobs=-1,
        ).fit_predict(Xs)
        lab = _force_assign(lab_raw, Xs) if force else lab_raw
        metrics, _ = _metrics(Xs, lab, y)
        row = dict(mcs=mcs, ms=ms, method=method, force=force, **metrics)
        all_rows.append(row)
        if metrics['macro_purity'] > best_macro:
            best_macro = metrics['macro_purity']
            best_hps = dict(mcs=mcs, ms=ms, method=method, force=force, k=metrics['k'])
    return best_macro, best_hps, all_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', default='duoduo',
                    help='Visual encoder key from _loaders.VISUAL_ENCODERS '
                         '(duoduo | siglip | dinov3 | gemini). Defaults to duoduo.')
    ap.add_argument('--variant', default='colorless',
                    help='Variant subdir under the encoder (default: colorless)')
    args = ap.parse_args()

    spec = ACTIVE_SPEC
    assert spec.name == 'ifcnet'
    out_dir = spec.results_root / f'fusion_{args.encoder}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[{args.encoder}] output → {out_dir}')

    print(f'Loading IFCNet geo + {args.encoder} visual...')
    df, y_full = load_data()
    gids_all = df['obj_id'].values
    geo_full = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))
    vis_full, vis_idx = load_visual(gids_all, args.encoder, args.variant)

    # Align: gids that have BOTH geo (always) and visual
    common = vis_idx                                # geo is keyed by gids_all, vis returns idx into gids_all
    df_a = df.iloc[common].reset_index(drop=True)
    geo = geo_full[common]
    vis = vis_full
    y = y_full[common]
    train_mask = (df_a['Split'] == 'train').values
    test_mask = (df_a['Split'] == 'test').values
    print(f'n={len(common)}  geo={geo.shape[1]}  visual={vis.shape[1]}  '
          f'classes={len(set(y))}  train={train_mask.sum()}  test={test_mask.sum()}')

    summary_csv = out_dir / 'summary.csv'
    hdbscan_csv = out_dir / 'hdbscan_full.csv'
    done = set()
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        done = set(prev['strategy'].tolist())
        print(f'[resume] {len(done)} strategies already in {summary_csv}: {sorted(done)}')

    summary, per_type_rows, hdbscan_rows = [], [], []
    STRATEGIES = _strategies_for(args.encoder)

    for strat_name, builder, kw, strat_args in STRATEGIES:
        if strat_name in done:
            print(f'  [{strat_name:14s}] skipped (already in summary.csv)')
            continue
        t0 = time.time()
        if strat_name == 'vision_only':
            X = vis.astype(np.float32)
        else:
            blocks = {'geo': geo.astype(np.float64), 'visual': vis.astype(np.float64)}
            X, _ = builder(blocks, *strat_args, **kw)

        Xs = StandardScaler().fit_transform(X).astype(np.float64)

        # Supervised RF
        sup = _rf_train_test(Xs, y, train_mask, test_mask)

        # Unsupervised HDBSCAN (skip for very high-d strategies if too slow)
        do_hdbscan = X.shape[1] <= 1024     # skip d > 1024 for wall-time
        if do_hdbscan:
            unsup_macro, best_hps, hdb_rows = _hdbscan_sweep(Xs, y)
            for r in hdb_rows:
                hdbscan_rows.append(dict(strategy=strat_name, dim=X.shape[1], **r))
        else:
            unsup_macro, best_hps = float('nan'), {}
            print(f'  [{strat_name}] skipping HDBSCAN (d={X.shape[1]} > 1024)')

        elapsed = time.time() - t0
        row = dict(
            strategy=strat_name,
            dim=int(X.shape[1]),
            unsup_macro_purity=unsup_macro,
            unsup_best_hps=str(best_hps),
            **sup,
            elapsed_s=round(elapsed, 1),
        )
        summary.append(row)
        print(f'  [{strat_name:14s}] dim={X.shape[1]:4d}  '
              f'unsup_macro={unsup_macro:.3f}  '
              f'sup_acc={sup["accuracy_mean"]:.3f}  '
              f'sup_f1={sup["f1_macro_mean"]:.3f}  '
              f'sup_rec={sup["recall_macro_mean"]:.3f}  '
              f'({elapsed:.0f}s)')

        # Incremental save after each strategy so an interruption doesn't lose progress
        all_summary = (pd.concat([pd.read_csv(summary_csv), pd.DataFrame([row])], ignore_index=True)
                       if summary_csv.exists() else pd.DataFrame([row]))
        all_summary.to_csv(summary_csv, index=False)
        if hdbscan_rows:
            new_hdb = pd.DataFrame([r for r in hdbscan_rows if r['strategy'] == strat_name])
            all_hdb = (pd.concat([pd.read_csv(hdbscan_csv), new_hdb], ignore_index=True)
                       if hdbscan_csv.exists() else new_hdb)
            all_hdb.to_csv(hdbscan_csv, index=False)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / 'summary.csv', index=False)
    if hdbscan_rows:
        pd.DataFrame(hdbscan_rows).to_csv(out_dir / 'hdbscan_full.csv', index=False)
    print(f'\nSaved → {out_dir}/summary.csv  ({len(sdf)} rows)')

    print('\n=== Headline (sorted by supervised F1-macro) ===')
    show_cols = ['strategy', 'dim', 'unsup_macro_purity',
                 'accuracy_mean', 'f1_macro_mean', 'recall_macro_mean']
    print(sdf.sort_values('f1_macro_mean', ascending=False)[show_cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
