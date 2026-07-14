"""Paired bootstrap CIs on the modality-ablation deltas at the F5_varbal winner.

The three subsets (geo+text+visual, geo+visual, geo+text) are all clustered on
the *same* 1692 objects (the sweep fixes common = text-cap-visual once), so the
three partitions are row-aligned and the deltas can be bootstrapped paired.

For B object-resamples (with replacement) we recompute macro purity of each
fixed partition on the resample and the paired deltas
    d_text   = macro(geo+text+visual) - macro(geo+visual)
    d_visual = macro(geo+text+visual) - macro(geo+text)
and report mean, 95% percentile CI, and P(delta > 0).

Output: results/midterm/stability/modality_delta_bootstrap.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent / 'feature_engineering'))
from _common import load_data, build_X, BASELINE, cumulative_bounded_up_to
from _loaders import load_text, load_visual
from _sweep_runner import _force_assign

MIDTERM = _EXP.parent / 'results' / 'midterm'
OUT = MIDTERM / 'stability'
OUT.mkdir(parents=True, exist_ok=True)

# best F5_varbal per subset (from fusion_full.csv): the visual encoder used
ENCODER = {'geo+text+visual': 'siglip', 'geo+visual': 'dinov3', 'geo+text': None}
B = 2000
SEED = 42


def _varbal(block):
    """Variance-balance a block so it contributes unit total variance."""
    s = StandardScaler().fit_transform(block)
    tv = s.var(axis=0, ddof=0).sum()
    return (s / np.sqrt(tv)).astype(np.float32) if tv > 1e-9 else s.astype(np.float32)


def partition_from_blocks(blocks, dim):
    import umap
    concat = np.concatenate([_varbal(b) for b in blocks], axis=1)
    emb = umap.UMAP(n_components=dim, n_neighbors=30, min_dist=0.0,
                    metric='euclidean', random_state=42).fit_transform(concat)
    Xs = StandardScaler().fit_transform(emb).astype(np.float64)
    lab = HDBSCAN(min_cluster_size=5, min_samples=3,
                  cluster_selection_method='leaf', n_jobs=-1).fit_predict(Xs)
    return _force_assign(lab, Xs)


def macro_purity(lab, y, idx):
    """Macro purity of fixed partition `lab` evaluated on object indices `idx`."""
    ls, ys = lab[idx], y[idx]
    # dominant type per cluster on this resample
    dom = {}
    for c in np.unique(ls):
        vals, cnts = np.unique(ys[ls == c], return_counts=True)
        dom[c] = vals[cnts.argmax()]
    types = np.unique(ys)
    per = []
    for t in types:
        m = ys == t
        correct = sum(1 for i in np.where(m)[0] if dom[ls[i]] == t)
        per.append(correct / m.sum())
    return float(np.mean(per))


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))
    Xt, ti = load_text(gids, 'v7', 'single')
    Xv_sig, vi_sig = load_visual(gids, 'siglip', 'colored')
    Xv_dino, vi_dino = load_visual(gids, 'dinov3', 'colored')
    common = np.intersect1d(ti, vi_sig)
    y = y_full[common]
    n = len(common)
    print(f'[delta-boot] common objects = {n}')

    def al(X, idx):
        return X[np.searchsorted(idx, common)]
    geo_b = geo[common]
    txt_b = al(Xt, ti)
    sig_b = al(Xv_sig, vi_sig)
    dino_b = al(Xv_dino, vi_dino)
    # best F5_varbal config per subset (encoder, umap dim) from fusion_full.csv
    block_sets = {
        'geo+text+visual': ([geo_b, txt_b, sig_b], 4),   # siglip, dim 4
        'geo+visual':      ([geo_b, dino_b], 64),         # dinov3, dim 64
        'geo+text':        ([geo_b, txt_b], 4),           # dim 4
    }
    labs = {k: partition_from_blocks(v, d) for k, (v, d) in block_sets.items()}
    full = {k: macro_purity(labs[k], y, np.arange(n)) for k in block_sets}
    print('full-sample macro:', {k: round(v, 4) for k, v in full.items()})

    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(B):
        idx = rng.integers(0, n, n)  # resample with replacement
        m = {k: macro_purity(labs[k], y, idx) for k in block_sets}
        rows.append({**m,
                     'd_text': m['geo+text+visual'] - m['geo+visual'],
                     'd_visual': m['geo+text+visual'] - m['geo+text']})
    bdf = pd.DataFrame(rows)
    bdf.to_csv(OUT / 'modality_delta_bootstrap.csv', index=False)

    def rep(col, point=None):
        v = bdf[col].values
        lo, hi = np.percentile(v, [2.5, 97.5])
        pe = point if point is not None else v.mean()
        extra = f"  P(>0)={np.mean(v > 0):.3f}" if col.startswith('d_') else ''
        print(f'  {col:18s} {pe:.4f}  95% CI [{lo:.4f}, {hi:.4f}]'
              f'  (sd {v.std():.4f}){extra}')

    print(f'\nPaired bootstrap (B={B}, resample with replacement, n={n}):')
    for k in block_sets:
        rep(k, full[k])
    rep('d_text', full['geo+text+visual'] - full['geo+visual'])
    rep('d_visual', full['geo+text+visual'] - full['geo+text'])


if __name__ == '__main__':
    main()
