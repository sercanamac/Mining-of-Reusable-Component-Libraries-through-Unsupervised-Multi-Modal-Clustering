"""P1a — Geometric-feature clustering sweep across 4 algorithms.

Runs the 9-d handcrafted geometric features through kmeans, hdbscan, gmm,
and bisecting k-means at the full hyperparameter grid. Persists long-format
CSVs under `results/midterm/sweeps/`.

Output files:
    results/midterm/sweeps/geo.csv
    results/midterm/sweeps/geo_per_type.csv

Run from repo root:
    python notebooks/data_analysis/experiments/P1a_geo_sweep.py
"""
from __future__ import annotations
import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _reducers import ReducerConfig, reducer_row_meta
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep, save_sweep

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'sweeps'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df, y = load_data()
    X = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))
    print(f'[P1a] X shape = {X.shape}, n_types = {len(set(y.tolist()))}')

    # Geo modality: identity reducer only (9-d already).
    cfg = ReducerConfig('identity', 'identity', None)
    extra = reducer_row_meta(cfg, var_ret=None)

    summary, per_type = run_sweep(
        name='geo_9d',
        X=X, y=y,
        algorithms=['kmeans', 'hdbscan', 'gmm', 'bisecting'],
        scale=True,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        extra_meta=extra,
    )
    save_sweep(
        summary, per_type,
        summary_csv=OUT_DIR / 'geo.csv',
        per_type_csv=OUT_DIR / 'geo_per_type.csv',
    )

    # Quick headline print — best per algorithm by macro purity.
    idx = summary.groupby('algo')['macro_purity'].idxmax()
    best = summary.loc[idx, ['algo', 'hps', 'purity', 'macro_purity',
                              'silhouette', 'k', 'noise_frac']]
    print('\n[P1a] best per algo by macro purity:')
    print(best.to_string(index=False))


if __name__ == '__main__':
    main()
