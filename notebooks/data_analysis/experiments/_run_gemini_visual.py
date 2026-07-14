"""One-shot: run the visual sweep for gemini/colored only, merge into visual.csv."""
import os, tempfile
os.environ['NUMBA_CACHE_DIR'] = tempfile.mkdtemp()
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'feature_engineering'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS_ROOT, load_data
from _loaders import load_visual
from _reducers import fit_reducer, make_reducer_configs, reducer_row_meta
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'sweeps'

df, y_full = load_data()
gids = df['GlobalId'].values
print(f'eval pool: {len(gids)} objects')

X, idx = load_visual(gids, 'gemini', 'colored')
y = y_full[idx]
print(f'gemini/colored: {X.shape} n={len(idx)}')

all_summary, all_per_type = [], []
cfgs = make_reducer_configs('visual')
for cfg in cfgs:
    Xr, meta = fit_reducer(cfg, X, modality_name='visual_gemini_colored')
    all_algos = (cfg.name == 'pca_32')
    extra = {
        'encoder': 'gemini', 'variant': 'colored',
        'n_aligned': int(len(idx)),
        **reducer_row_meta(cfg, meta.get('var_ret')),
    }
    cache_tag = '(cache)' if meta.get('cache_hit') else ''
    tag = 'ALL' if all_algos else 'HDB'
    print(f'  {cfg.name:10s} dim={Xr.shape[1]:4d} '
          f'red={meta["fit_seconds"]:5.1f}s {tag} {cache_tag}')
    s, pt = run_sweep(
        name=f'visual_gemini_colored_{cfg.name}',
        X=Xr, y=y,
        algorithms=['kmeans', 'hdbscan', 'gmm', 'bisecting'] if all_algos else ['hdbscan'],
        scale=True,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        extra_meta=extra,
        verbose=False,
    )
    all_summary.append(s)
    all_per_type.append(pt)

gemini_summary = pd.concat(all_summary, ignore_index=True)
gemini_per_type = pd.concat(all_per_type, ignore_index=True)

# Merge with existing
existing = pd.read_csv(OUT_DIR / 'visual.csv')
existing_pt = pd.read_csv(OUT_DIR / 'visual_per_type.csv')
existing = existing[existing['encoder'] != 'gemini']
existing_pt = existing_pt[existing_pt['encoder'] != 'gemini']

offset = int(existing['run_id'].max()) + 1 if len(existing) else 0
gemini_summary['run_id'] += offset
gemini_per_type['run_id'] += offset

merged = pd.concat([existing, gemini_summary], ignore_index=True)
merged_pt = pd.concat([existing_pt, gemini_per_type], ignore_index=True)
merged.to_csv(OUT_DIR / 'visual.csv', index=False)
merged_pt.to_csv(OUT_DIR / 'visual_per_type.csv', index=False)

hdb = gemini_summary[gemini_summary['algo'] == 'hdbscan']
best = hdb.loc[hdb['macro_purity'].idxmax()]
print(f'\nGemini visual best HDBSCAN:')
print(f'  reducer={best["reducer"]}  macro={best["macro_purity"]:.4f}  '
      f'purity={best["purity"]:.4f}  k={int(best["k"])}')
print(f'\nMerged visual.csv: {len(merged)} rows (was {len(existing)})')
