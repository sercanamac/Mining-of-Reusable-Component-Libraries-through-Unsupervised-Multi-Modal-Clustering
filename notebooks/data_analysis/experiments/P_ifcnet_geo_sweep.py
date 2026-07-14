"""P_ifcnet_geo_sweep — geo-only clustering sweep on IFCNetCore.

Loads IFCNet features via load_data(spec=IFCNET) (baseline + engineered merged
on obj_id), builds X from BASELINE + all bounded engineered features
(`cumulative_bounded_up_to('04_horiz_frac')`), and runs:
  - KMeans across k ∈ {2, 4, 8, 12, 20, 32, 48}, 10 seeds
  - HDBSCAN factorial grid (60 configs)

Outputs:
  notebooks/data_analysis/results/ifcnet/sweeps/geo.csv
  notebooks/data_analysis/results/ifcnet/sweeps/geo_per_type.csv

Run:
  IFCNET_DATASET=ifcnet python3 -u notebooks/data_analysis/experiments/P_ifcnet_geo_sweep.py
"""
from __future__ import annotations
import os, sys, tempfile
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
os.environ['IFCNET_DATASET'] = 'ifcnet'    # ensure spec is loaded
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'feature_engineering'))
sys.path.insert(0, str(_HERE))

from _common import ACTIVE_SPEC, BASELINE, build_X, cumulative_bounded_up_to, load_data
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep, save_sweep


def main():
    spec = ACTIVE_SPEC
    assert spec.name == 'ifcnet', f'expected ifcnet spec, got {spec.name}'

    out_dir = spec.results_root / 'sweeps'
    out_dir.mkdir(parents=True, exist_ok=True)

    df, y = load_data()
    print(f'[geo] n={len(df)}  classes={len(set(y))}  '
          f'cols (head)={list(df.columns)[:8]}')

    bounded = cumulative_bounded_up_to('04_horiz_frac')
    X = build_X(df, BASELINE, bounded)
    print(f'[geo] X shape: {X.shape}  (baseline 4 + bounded {bounded})')

    summary, per_type = run_sweep(
        name='geo',
        X=X, y=y,
        algorithms=['kmeans', 'hdbscan'],
        scale=True,
        k_values=spec.k_values,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        verbose=True,
    )
    save_sweep(summary, per_type, out_dir / 'geo.csv', out_dir / 'geo_per_type.csv')
    print(f'\n[geo] wrote {len(summary)} summary rows → {out_dir / "geo.csv"}')

    # Headline highlights
    print('\n=== KMeans best per k (macro purity) ===')
    km = summary[summary['algo'] == 'kmeans']
    print(km.groupby('hp_k')['macro_purity'].agg(['mean', 'max']).round(3).to_string())

    print('\n=== HDBSCAN top-10 configs (macro purity) ===')
    hdb = summary[summary['algo'] == 'hdbscan']
    top = hdb.sort_values('macro_purity', ascending=False).head(10)
    print(top[['hp_mcs', 'hp_ms', 'hp_method', 'hp_force',
               'macro_purity', 'purity', 'k', 'noise_frac']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
