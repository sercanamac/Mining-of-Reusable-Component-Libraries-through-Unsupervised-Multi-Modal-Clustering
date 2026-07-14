"""P4d — k-sensitivity & silhouette-based k selection (G19 + G20).

G19: Purity at k=17 (true merged-type count) for every
     (subset, recipe, variant, reducer, algo) cell that ran in Phase 2.
     Uses kmeans k=17 as the canonical fixed-k baseline.

G20: Silhouette vs k for k ∈ [2..64] at the winning (subset, recipe, variant,
     reducer). Reports the silhouette-picked k* and the HDBSCAN-found k.

Output:
    results/midterm/stability/k_sensitivity.csv      (both G19 + G20 rows)
    results/midterm/stability/silhouette_vs_k.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import RESULTS_ROOT, compute_purity, load_data, per_type_purity
from _winners import build_X_for_winner, get_canonical_winner

STABILITY_DIR = RESULTS_ROOT.parent / 'midterm' / 'stability'
STABILITY_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE_G20 = list(range(2, 65))
SEEDS = list(range(42, 47))  # 5 seeds


def _kmeans_labels(X, k, seed):
    return MiniBatchKMeans(
        n_clusters=k, batch_size=1024, n_init='auto', random_state=seed,
    ).fit_predict(X)


def main():
    # ── G19: k=17 purity at every cell in the fusion CSV ───────────────────
    fusion_csv = RESULTS_ROOT.parent / 'midterm' / 'fusion' / 'fusion.csv'
    if not fusion_csv.exists():
        raise FileNotFoundError(f'{fusion_csv} missing — run P2 first')

    # Rebuild the winning fusion X matrix (whichever recipe won the §12 sweep).
    winner = get_canonical_winner(algo='hdbscan')
    X, y, _, info = build_X_for_winner(winner, verbose=True)
    print(f'[P4d] G19/G20 computed on winner: {info["subset"]}/{info["recipe"]} '
          f'dim={info["out_dim"]}')

    g19_rows = []
    print('[P4d] G19: kmeans k=17 on canonical winning fusion')
    for seed in SEEDS:
        lab = _kmeans_labels(X, 17, seed)
        pur = compute_purity(lab, y)
        tp = per_type_purity(lab, y)
        macro = float(np.mean(list(tp.values())))
        g19_rows.append({
            'metric_group': 'G19_k17_purity',
            'config': 'canonical_F2_fusion_kmeans17',
            'seed': seed, 'k': 17, 'purity': float(pur), 'macro_purity': macro,
        })
    g19 = pd.DataFrame(g19_rows)
    print(g19.to_string(index=False))
    print(f'  mean purity = {g19.purity.mean():.4f} ± {g19.purity.std():.4f}')
    print(f'  mean macro  = {g19.macro_purity.mean():.4f} ± {g19.macro_purity.std():.4f}')

    # ── G20: silhouette vs k on canonical winning fusion ──────────────────
    print('\n[P4d] G20: silhouette vs k on canonical winning fusion')
    g20_rows = []
    sil_best = -np.inf
    k_star = None
    for k in K_RANGE_G20:
        # One seed per k for speed.
        lab = _kmeans_labels(X, k, 42)
        sil = float(silhouette_score(X, lab, sample_size=min(3000, len(lab)),
                                      random_state=42))
        pur = float(compute_purity(lab, y))
        tp = per_type_purity(lab, y)
        macro = float(np.mean(list(tp.values())))
        g20_rows.append({
            'metric_group': 'G20_silhouette_vs_k',
            'config': 'canonical_F2_fusion_kmeans',
            'seed': 42, 'k': k, 'silhouette': sil, 'purity': pur,
            'macro_purity': macro,
        })
        if sil > sil_best:
            sil_best = sil
            k_star = k
    g20 = pd.DataFrame(g20_rows)
    print(f'  silhouette-picked k* = {k_star} (sil={sil_best:.4f})')

    # HDBSCAN k-found baseline (same winner we used to rebuild X)
    hdb_winner = winner
    print(f'  HDBSCAN-found k = {hdb_winner["k"]} (macro={hdb_winner["macro_purity"]:.4f})')

    combined = pd.concat([g19, g20], ignore_index=True)
    out_csv = STABILITY_DIR / 'k_sensitivity.csv'
    combined.to_csv(out_csv, index=False)
    print(f'saved → {out_csv}')

    # ── Silhouette vs k plot ──────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
    ax1.plot(g20.k, g20.silhouette, '-o', color='C0', lw=2, ms=4,
             label='silhouette')
    ax1.set_xlabel('k (# clusters)')
    ax1.set_ylabel('silhouette score', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.axvline(k_star, color='C0', ls='--', alpha=0.5)
    ax1.axvline(17, color='C3', ls=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(g20.k, g20.macro_purity, '-s', color='C2', lw=2, ms=4,
             label='macro purity')
    ax2.set_ylabel('macro purity', color='C2')
    ax2.tick_params(axis='y', labelcolor='C2')
    if hdb_winner is not None:
        ax2.axhline(hdb_winner['macro_purity'], color='C1', ls='-.',
                    alpha=0.7, label=f'HDBSCAN winner (k={hdb_winner["k"]})')

    ax1.set_title(f'silhouette vs k on canonical fusion — k*={k_star}, '
                  f'true types=17')
    ax2.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(STABILITY_DIR / 'silhouette_vs_k.png')
    plt.close(fig)
    print(f'saved → {STABILITY_DIR / "silhouette_vs_k.png"}')


if __name__ == '__main__':
    main()
