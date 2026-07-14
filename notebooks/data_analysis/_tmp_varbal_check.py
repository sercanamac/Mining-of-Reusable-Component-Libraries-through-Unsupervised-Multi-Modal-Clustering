"""Verify variance balancing math with real blocks."""
import os, tempfile, sys
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
from pathlib import Path
sys.path.insert(0, str(Path('notebooks/data_analysis/feature_engineering').resolve()))
sys.path.insert(0, str(Path('notebooks/data_analysis/experiments').resolve()))

import numpy as np
from sklearn.preprocessing import StandardScaler
from _common import BASELINE, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual


def report(name, X):
    feat_vars = X.var(axis=0, ddof=0)
    print(f'  {name:30s}  shape={X.shape}  '
          f'mean(feat_var)={feat_vars.mean():.4f}  '
          f'sum(feat_var)={feat_vars.sum():.2f}')


def standardize(block):
    return StandardScaler().fit_transform(block).astype(np.float64)


def var_balance(block):
    s = standardize(block)
    total_var = s.var(axis=0, ddof=0).sum()
    return s / np.sqrt(total_var)


def main():
    df, y = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))
    X_txt, txt_idx = load_text(gids, 'v7', 'single')
    X_vis, vis_idx = load_visual(gids, 'siglip', 'colored')
    common = np.intersect1d(txt_idx, vis_idx)

    geo_b = geo[common]
    txt_b = X_txt[np.searchsorted(txt_idx, common)]
    vis_b = X_vis[np.searchsorted(vis_idx, common)]
    print(f'n={len(common)}')

    blocks = [('geo', geo_b), ('text', txt_b), ('visual', vis_b)]

    print('\n=== RAW ===')
    for n, b in blocks: report(n, b)
    raw_totals = [b.var(axis=0, ddof=0).sum() for _, b in blocks]
    print(f'  total-var ratios geo:text:visual = '
          f'{raw_totals[0]/raw_totals[0]:.2f} : {raw_totals[1]/raw_totals[0]:.2f} : {raw_totals[2]/raw_totals[0]:.2f}')

    print('\n=== STANDARDIZE ===')
    std_blocks = [(n, standardize(b)) for n, b in blocks]
    for n, b in std_blocks: report(n, b)
    std_totals = [b.var(axis=0, ddof=0).sum() for _, b in std_blocks]
    print(f'  total-var ratios geo:text:visual = '
          f'{std_totals[0]/std_totals[0]:.2f} : {std_totals[1]/std_totals[0]:.2f} : {std_totals[2]/std_totals[0]:.2f}')

    print('\n=== VAR-BALANCE ===')
    vb_blocks = [(n, var_balance(b)) for n, b in blocks]
    for n, b in vb_blocks: report(n, b)
    vb_totals = [b.var(axis=0, ddof=0).sum() for _, b in vb_blocks]
    print(f'  total-var ratios geo:text:visual = '
          f'{vb_totals[0]/vb_totals[0]:.2f} : {vb_totals[1]/vb_totals[0]:.2f} : {vb_totals[2]/vb_totals[0]:.2f}')

    # Sanity: total variance should equal exactly 1.0 per block
    print('\n=== ASSERTIONS ===')
    for n, b in vb_blocks:
        tv = b.var(axis=0, ddof=0).sum()
        ok = abs(tv - 1.0) < 1e-6
        print(f'  {n:8s} total var = {tv:.10f}   {"✓ ==1" if ok else "✗"}')

    # Concat behavior
    print('\n=== CONCAT TOTAL VARIANCE ===')
    raw_concat = np.concatenate([b for _, b in blocks], axis=1)
    std_concat = np.concatenate([b for _, b in std_blocks], axis=1)
    vb_concat = np.concatenate([b for _, b in vb_blocks], axis=1)
    for n, X in [('raw_concat', raw_concat), ('std_concat', std_concat), ('vb_concat', vb_concat)]:
        # contribution of each block
        c1 = X[:, :geo_b.shape[1]].var(axis=0, ddof=0).sum()
        c2 = X[:, geo_b.shape[1]:geo_b.shape[1]+txt_b.shape[1]].var(axis=0, ddof=0).sum()
        c3 = X[:, geo_b.shape[1]+txt_b.shape[1]:].var(axis=0, ddof=0).sum()
        total = c1 + c2 + c3
        print(f'  {n:12s}  geo={c1:.3f} ({c1/total:.1%})  '
              f'text={c2:.3f} ({c2/total:.1%})  visual={c3:.3f} ({c3/total:.1%})')


if __name__ == '__main__':
    main()
