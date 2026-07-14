"""P1b — Gemini text embedding clustering sweep with reducer ablation.

Two sub-runs share one output CSV:

  (1) aggregation_ablation — v1 × {single, sum, concat, single_views}
      × 10 reducer configs × HDBSCAN full factorial.
  (2) version_sweep — every Gemini version × {single} × 10 reducer configs
      × HDBSCAN full factorial.

Reducer grid per (version, aggregation) cell:
    identity, PCA {8,16,32,64,128}, UMAP {4,8,16,64} — see _reducers.py

HDBSCAN grid: HDBSCAN_GRID_FACTORIAL (60 combos).

To keep the CSV tractable, k-means / GMM / bisecting are run only at
`pca_32` per cell — one reducer config for the cross-algorithm comparison.

Output:
    results/midterm/sweeps/text.csv
    results/midterm/sweeps/text_per_type.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import RESULTS_ROOT, load_data
from _loaders import TEXT_VERSIONS, available_text_configs, load_text
from _reducers import fit_reducer, make_reducer_configs, reducer_row_meta
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep, save_sweep

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'sweeps'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGOS_HDB_ONLY = ['hdbscan']
ALGOS_ALL = ['kmeans', 'hdbscan', 'gmm', 'bisecting']


def _sweep_cell(name: str, X, y, extra: dict, all_algos: bool):
    algos = ALGOS_ALL if all_algos else ALGOS_HDB_ONLY
    return run_sweep(
        name=name, X=X, y=y,
        algorithms=algos,
        scale=True,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        extra_meta=extra,
        verbose=False,
    )


def _run_block(gids, y_full, version, aggregation, run_type):
    try:
        X, idx = load_text(gids, version, aggregation)
    except RuntimeError as e:
        print(f'  skip {version}/{aggregation}: {e}')
        return [], []
    y = y_full[idx]
    n = len(idx)
    cfgs = make_reducer_configs('text')
    sums, pts = [], []
    for cfg in cfgs:
        modality_name = f'text_{version}_{aggregation}'
        Xr, meta = fit_reducer(cfg, X, modality_name=modality_name)
        extra = {
            'run_type': run_type, 'version': version, 'aggregation': aggregation,
            'n_aligned': int(n),
            **reducer_row_meta(cfg, meta.get('var_ret')),
        }
        tag = 'ALL' if cfg.name == 'pca_32' else 'HDB'
        cache_tag = '(cache)' if meta.get('cache_hit') else ''
        print(f'  {version}/{aggregation} {cfg.name:10s} dim={Xr.shape[1]:4d} '
              f'red={meta["fit_seconds"]:5.1f}s {tag:3s} {cache_tag}')
        s, pt = _sweep_cell(
            f'text_{version}_{aggregation}_{cfg.name}', Xr, y, extra,
            all_algos=(cfg.name == 'pca_32'),
        )
        sums.append(s)
        pts.append(pt)
    return sums, pts


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    print(f'[P1b] eval pool: {len(gids)} objects, {len(set(y_full.tolist()))} types')

    all_summary = []
    all_per_type = []

    # ── (1) Aggregation ablation on v1 ────────────────────────────────────
    v1_aggs_available = [
        a for (v, a, _n) in available_text_configs() if v == 'v1'
    ]
    print(f'\n[P1b] (1) aggregation ablation on v1: {v1_aggs_available}')
    for agg in v1_aggs_available:
        s, pt = _run_block(gids, y_full, 'v1', agg, 'aggregation_ablation')
        all_summary += s
        all_per_type += pt

    # ── (2) Version sweep, single aggregation only ────────────────────────
    print(f'\n[P1b] (2) version sweep × single aggregation')
    for v in list(TEXT_VERSIONS.keys()):
        s, pt = _run_block(gids, y_full, v, 'single', 'version_sweep')
        all_summary += s
        all_per_type += pt

    # Re-number run_ids so they are unique across cells.
    offset = 0
    for s, pt in zip(all_summary, all_per_type):
        s['run_id'] = s['run_id'].to_numpy() + offset
        pt['run_id'] = pt['run_id'].to_numpy() + offset
        offset = int(s['run_id'].max()) + 1
    summary = pd.concat(all_summary, ignore_index=True)
    per_type = pd.concat(all_per_type, ignore_index=True)

    save_sweep(
        summary, per_type,
        summary_csv=OUT_DIR / 'text.csv',
        per_type_csv=OUT_DIR / 'text_per_type.csv',
    )

    # Headline: best per (run_type, version, aggregation, reducer, algo)
    key = ['run_type', 'version', 'aggregation', 'reducer', 'algo']
    idx = summary.groupby(key)['macro_purity'].idxmax()
    best = summary.loc[idx, key + ['purity', 'macro_purity', 'k', 'noise_frac']]
    print('\n[P1b] best per (version × aggregation × reducer × algo) by macro purity:')
    print(best.sort_values(key).to_string(index=False))


if __name__ == '__main__':
    main()
