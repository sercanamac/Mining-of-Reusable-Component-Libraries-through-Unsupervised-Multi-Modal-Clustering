"""P6 — k-matched algorithm comparison.

Runs k-means at k=102 (the HDBSCAN winner's k) on the same F1a-pca8
representation to measure the honest algorithm gap at matched granularity.

Also reports HDBSCAN force=False (native density partition) for completeness.

Output:
    results/midterm/k_matched_comparison.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import compute_purity, per_type_purity, SEEDS
from _winners import get_canonical_winner, build_X_for_winner, _force_assign

MIDTERM = Path(__file__).resolve().parent.parent / 'results' / 'midterm'
OUT_CSV = MIDTERM / 'k_matched_comparison.csv'


def main():
    w = get_canonical_winner(algo='hdbscan')
    X, y, gids, info = build_X_for_winner(w)
    hps = w['hps']
    target_k = w['k']

    print(f'\n=== k-matched comparison at k={target_k} ===')
    print(f'Representation: {info["subset"]} / {info["recipe"]} (dim={X.shape[1]}, n={X.shape[0]})')
    print()

    rows = []

    # ── 1. HDBSCAN force=True (the winner, single run) ──────────────────
    lab_raw = HDBSCAN(
        min_cluster_size=int(hps['mcs']),
        min_samples=int(hps['ms']),
        cluster_selection_method=hps['method'],
        n_jobs=-1,
    ).fit_predict(X)
    lab_forced = _force_assign(lab_raw, X)
    k_forced = len(set(lab_forced.tolist()))

    tp = per_type_purity(lab_forced, y)
    macro = float(np.mean(list(tp.values())))
    pur = float(compute_purity(lab_forced, y))

    rows.append({
        'method': 'HDBSCAN force=True',
        'k': k_forced,
        'seed': '-',
        'purity': pur,
        'macro_purity': macro,
        'noise_frac': 0.0,
        'note': f'mcs={hps["mcs"]} ms={hps["ms"]} {hps["method"]}',
    })
    print(f'HDBSCAN force=True:  k={k_forced}  purity={pur:.4f}  macro={macro:.4f}')

    # ── 2. HDBSCAN force=False (native partition) ────────────────────────
    noise_frac = float((lab_raw == -1).mean())
    k_native = len(set(lab_raw[lab_raw != -1].tolist()))

    non_noise = lab_raw != -1
    if non_noise.sum() > 0:
        tp_nf = per_type_purity(lab_raw[non_noise], y[non_noise])
        macro_nf = float(np.mean(list(tp_nf.values())))
        pur_nf = float(compute_purity(lab_raw[non_noise], y[non_noise]))
    else:
        macro_nf = pur_nf = 0.0

    tp_nf_all = per_type_purity(lab_raw, y)
    macro_nf_all = float(np.mean(list(tp_nf_all.values())))
    pur_nf_all = float(compute_purity(lab_raw, y))

    rows.append({
        'method': 'HDBSCAN force=False (assigned only)',
        'k': k_native,
        'seed': '-',
        'purity': pur_nf,
        'macro_purity': macro_nf,
        'noise_frac': noise_frac,
        'note': f'{non_noise.sum()}/{len(lab_raw)} assigned',
    })
    rows.append({
        'method': 'HDBSCAN force=False (all, noise=-1)',
        'k': k_native,
        'seed': '-',
        'purity': pur_nf_all,
        'macro_purity': macro_nf_all,
        'noise_frac': noise_frac,
        'note': 'noise counted as own cluster',
    })
    print(f'HDBSCAN force=False: k={k_native}  noise={noise_frac:.1%}  '
          f'purity(assigned)={pur_nf:.4f}  macro(assigned)={macro_nf:.4f}  '
          f'purity(all)={pur_nf_all:.4f}  macro(all)={macro_nf_all:.4f}')

    # ── 3. k-means at k=target_k (10 seeds) ─────────────────────────────
    km_purs, km_macros = [], []
    for seed in SEEDS:
        lab_km = MiniBatchKMeans(
            n_clusters=target_k, batch_size=1024, n_init='auto',
            random_state=seed,
        ).fit_predict(X)
        p = float(compute_purity(lab_km, y))
        tp_km = per_type_purity(lab_km, y)
        m = float(np.mean(list(tp_km.values())))
        km_purs.append(p)
        km_macros.append(m)
        rows.append({
            'method': f'k-means k={target_k}',
            'k': target_k,
            'seed': seed,
            'purity': p,
            'macro_purity': m,
            'noise_frac': 0.0,
            'note': '',
        })

    km_pur_mean, km_pur_std = np.mean(km_purs), np.std(km_purs)
    km_mac_mean, km_mac_std = np.mean(km_macros), np.std(km_macros)
    print(f'k-means k={target_k}:    purity={km_pur_mean:.4f}±{km_pur_std:.4f}  '
          f'macro={km_mac_mean:.4f}±{km_mac_std:.4f}  (10 seeds)')

    # ── 4. k-means at k=17 (label count, 10 seeds) for reference ────────
    km17_purs, km17_macros = [], []
    for seed in SEEDS:
        lab_km17 = MiniBatchKMeans(
            n_clusters=17, batch_size=1024, n_init='auto',
            random_state=seed,
        ).fit_predict(X)
        p = float(compute_purity(lab_km17, y))
        tp_km17 = per_type_purity(lab_km17, y)
        m = float(np.mean(list(tp_km17.values())))
        km17_purs.append(p)
        km17_macros.append(m)
        rows.append({
            'method': 'k-means k=17',
            'k': 17,
            'seed': seed,
            'purity': p,
            'macro_purity': m,
            'noise_frac': 0.0,
            'note': 'label-count reference',
        })

    print(f'k-means k=17:       purity={np.mean(km17_purs):.4f}±{np.std(km17_purs):.4f}  '
          f'macro={np.mean(km17_macros):.4f}±{np.std(km17_macros):.4f}  (10 seeds)')

    # ── Save ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved → {OUT_CSV}')

    # ── Summary table ────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('SUMMARY — k-matched comparison')
    print('=' * 72)
    print(f'{"Method":<42s} {"k":>5s} {"Purity":>8s} {"Macro":>8s} {"Noise":>7s}')
    print('-' * 72)
    print(f'{"HDBSCAN force=True (winner)":<42s} {k_forced:>5d} {pur:>8.4f} {macro:>8.4f} {"0%":>7s}')
    print(f'{"k-means k=" + str(target_k) + " (10 seeds)":<42s} {target_k:>5d} '
          f'{km_pur_mean:>7.4f}± {km_mac_mean:>7.4f}± {"0%":>6s}')
    print(f'{"  → std":<42s} {"":>5s} {km_pur_std:>8.4f} {km_mac_std:>8.4f}')
    print(f'{"HDBSCAN force=False (assigned only)":<42s} {k_native:>5d} {pur_nf:>8.4f} {macro_nf:>8.4f} {noise_frac:>6.1%}')
    print(f'{"k-means k=17 (10 seeds)":<42s} {17:>5d} '
          f'{np.mean(km17_purs):>7.4f}± {np.mean(km17_macros):>7.4f}±')
    gap = macro - km_mac_mean
    print(f'\nHonest HDBSCAN→k-means gap at k={target_k}: {gap:+.4f} ({gap*100:+.1f} pp)')


if __name__ == '__main__':
    main()
