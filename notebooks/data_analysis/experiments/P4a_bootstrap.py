"""P4a — Bootstrap cluster stability of the winning fusion config.

Re-runs the winning HDBSCAN fusion config on 10 random 90%-subsamples.
For every pair of runs, computes the Adjusted Rand Index on their *overlap*
(points that appear in both subsamples). Reports mean / std ARI, plus purity
of each subsample's clustering vs ground-truth labels.

Output:
    results/midterm/stability/bootstrap.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))

from sklearn.cluster import HDBSCAN

from _common import RESULTS_ROOT, compute_purity, per_type_purity
from _winners import (build_X_for_winner, get_canonical_winner,
                       _force_assign)

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'stability'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 10
SUBSAMPLE_FRAC = 0.90
SEED = 42


def _refit(X_sub, hps):
    lab_raw = HDBSCAN(
        min_cluster_size=int(hps['mcs']),
        min_samples=int(hps['ms']),
        cluster_selection_method=hps['method'],
        n_jobs=-1,
    ).fit_predict(X_sub)
    return _force_assign(lab_raw, X_sub) if hps.get('force', True) else lab_raw


def main():
    winner = get_canonical_winner(algo='hdbscan')
    X, y, gids, info = build_X_for_winner(winner, verbose=True)
    hps = winner['hps']
    print(f'[P4a] winner {info["subset"]}/{info["recipe"]} hps={hps} → target '
          f'macro_purity={winner["macro_purity"]:.4f}')

    rng = np.random.default_rng(SEED)
    n = len(X)

    subsamples = [rng.choice(n, int(n * SUBSAMPLE_FRAC), replace=False)
                  for _ in range(N_RUNS)]
    labels_per_run = []
    purity_per_run = []
    macro_per_run = []
    for i, idx in enumerate(subsamples):
        lab = _refit(X[idx], hps)
        labels_per_run.append((idx, lab))
        pur = compute_purity(lab, y[idx])
        tp = per_type_purity(lab, y[idx])
        macro = float(np.mean(list(tp.values())))
        purity_per_run.append(pur)
        macro_per_run.append(macro)
        print(f'  run {i+1}/{N_RUNS}: n_sub={len(idx)}, k={len(set(lab))}, '
              f'pur={pur:.4f}, macro={macro:.4f}')

    # Pairwise ARI on overlap
    ari_rows = []
    for i in range(N_RUNS):
        for j in range(i + 1, N_RUNS):
            idx_i, lab_i = labels_per_run[i]
            idx_j, lab_j = labels_per_run[j]
            overlap = np.intersect1d(idx_i, idx_j)
            pos_i = np.searchsorted(np.sort(idx_i), overlap)
            pos_j = np.searchsorted(np.sort(idx_j), overlap)
            # Because idx_i/idx_j are unsorted, we need sort alignment
            # Re-sort once outside lab_i/lab_j to match np.sort(idx) order
            order_i = np.argsort(idx_i)
            order_j = np.argsort(idx_j)
            lab_i_s = lab_i[order_i]
            lab_j_s = lab_j[order_j]
            li = lab_i_s[pos_i]
            lj = lab_j_s[pos_j]
            ari = adjusted_rand_score(li, lj)
            ari_rows.append(dict(run_i=i, run_j=j, n_overlap=len(overlap),
                                 ari=ari))

    ari_df = pd.DataFrame(ari_rows)
    purity_df = pd.DataFrame(dict(
        run=list(range(N_RUNS)),
        purity=purity_per_run, macro_purity=macro_per_run,
        n_sub=[len(s) for s in subsamples],
    ))

    ari_df.to_csv(OUT_DIR / 'bootstrap_ari.csv', index=False)
    purity_df.to_csv(OUT_DIR / 'bootstrap_purity.csv', index=False)
    print(f'\n[P4a] mean pairwise ARI = {ari_df["ari"].mean():.3f} '
          f'± {ari_df["ari"].std():.3f} (n={len(ari_df)} pairs)')
    print(f'       mean purity = {purity_df["purity"].mean():.3f} '
          f'± {purity_df["purity"].std():.3f}')
    print(f'       mean macro  = {purity_df["macro_purity"].mean():.3f} '
          f'± {purity_df["macro_purity"].std():.3f}')
    print(f'saved → {OUT_DIR / "bootstrap_ari.csv"}')
    print(f'saved → {OUT_DIR / "bootstrap_purity.csv"}')


if __name__ == '__main__':
    main()
