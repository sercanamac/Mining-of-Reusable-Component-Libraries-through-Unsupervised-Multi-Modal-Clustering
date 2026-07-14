"""P1c — Visual embedding clustering sweep with reducer ablation.

Runs each (encoder, color_variant) pair through:
  • 10 reducer configs: identity, PCA {8,16,32,64,128}, UMAP {4,8,16,64}
  • HDBSCAN full factorial (60 combos)
  • k-means / GMM / bisecting at pca_32 only (one cross-algo reducer)

Output:
    results/midterm/sweeps/visual.csv
    results/midterm/sweeps/visual_per_type.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import RESULTS_ROOT, load_data
from _loaders import VISUAL_ENCODERS, VISUAL_COLOR_VARIANTS, load_visual
from _reducers import fit_reducer, make_reducer_configs, reducer_row_meta
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep, save_sweep

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'sweeps'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGOS_HDB_ONLY = ['hdbscan']
ALGOS_ALL = ['kmeans', 'hdbscan', 'gmm', 'bisecting']


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    print(f'[P1c] eval pool: {len(gids)} objects')

    all_summary, all_per_type = [], []

    for encoder in VISUAL_ENCODERS:
        for variant in VISUAL_COLOR_VARIANTS:
            try:
                X, idx = load_visual(gids, encoder, variant)
            except RuntimeError as e:
                print(f'  skip {encoder}/{variant}: {e}')
                continue
            y = y_full[idx]
            print(f'\n  {encoder}/{variant}: {X.shape} n={len(idx)}')
            cfgs = make_reducer_configs('visual')
            for cfg in cfgs:
                modality_name = f'visual_{encoder}_{variant}'
                Xr, meta = fit_reducer(cfg, X, modality_name=modality_name)
                all_algos = (cfg.name == 'pca_32')
                extra = {
                    'encoder': encoder, 'variant': variant,
                    'n_aligned': int(len(idx)),
                    **reducer_row_meta(cfg, meta.get('var_ret')),
                }
                cache_tag = '(cache)' if meta.get('cache_hit') else ''
                tag = 'ALL' if all_algos else 'HDB'
                print(f'    {cfg.name:10s} dim={Xr.shape[1]:4d} '
                      f'red={meta["fit_seconds"]:5.1f}s {tag:3s} {cache_tag}')
                s, pt = run_sweep(
                    name=f'{modality_name}_{cfg.name}',
                    X=Xr, y=y,
                    algorithms=ALGOS_ALL if all_algos else ALGOS_HDB_ONLY,
                    scale=True,
                    hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
                    extra_meta=extra,
                    verbose=False,
                )
                all_summary.append(s)
                all_per_type.append(pt)

    offset = 0
    for s, pt in zip(all_summary, all_per_type):
        s['run_id'] = s['run_id'].to_numpy() + offset
        pt['run_id'] = pt['run_id'].to_numpy() + offset
        offset = int(s['run_id'].max()) + 1
    summary = pd.concat(all_summary, ignore_index=True)
    per_type = pd.concat(all_per_type, ignore_index=True)

    save_sweep(
        summary, per_type,
        summary_csv=OUT_DIR / 'visual.csv',
        per_type_csv=OUT_DIR / 'visual_per_type.csv',
    )

    key = ['encoder', 'variant', 'reducer', 'algo']
    idx = summary.groupby(key)['macro_purity'].idxmax()
    best = summary.loc[idx, key + ['purity', 'macro_purity', 'k', 'noise_frac']]
    print('\n[P1c] best per (encoder × variant × reducer × algo) by macro purity:')
    print(best.sort_values(key).to_string(index=False))


if __name__ == '__main__':
    main()
