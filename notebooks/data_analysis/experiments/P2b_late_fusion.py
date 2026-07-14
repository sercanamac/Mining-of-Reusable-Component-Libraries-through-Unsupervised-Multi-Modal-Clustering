"""P2b — Late fusion via per-modality cosine similarity + averaged affinity.

For each subset in {geo+text, geo+visual, geo+text+visual}:
  1. Build per-modality cosine similarity matrix on the aligned objects
     (geo: StandardScaled 9-d; text: raw Gemini v7 single; visual: raw
     SigLIP colorless).
  2. Average the per-modality similarity matrices → S_fused.
  3. HDBSCAN with `metric='precomputed'` on D = 1 - S_fused.
  4. Also report retrieval recall@{1,5,10} using type labels as relevance.

HDBSCAN grid: a compact subset (mcs/ms/method only; force-assign applied
post-hoc) sweeping the same cells as P2 for parity.

Output:
    results/midterm/fusion/late_fusion.csv          (clustering metrics)
    results/midterm/fusion/late_fusion_retrieval.csv (recall@K by modality)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import (BASELINE, RESULTS_ROOT, build_X, compute_purity,
                      cumulative_bounded_up_to, load_data, per_type_purity)
from _loaders import load_text, load_visual
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, _force_assign

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'fusion'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_VERSION = 'v7'
TEXT_AGG = 'single'
VISUAL_ENCODER = 'siglip'
VISUAL_VARIANT = 'colorless'


def _sim(X):
    X = np.asarray(X, dtype=np.float32)
    return cosine_similarity(X).astype(np.float32)


def _recall_at_k(sim: np.ndarray, y: np.ndarray, ks=(1, 5, 10)):
    n = len(y)
    out = {f'recall@{k}': 0.0 for k in ks}
    # Exclude self from neighbors.
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    # Precompute argsort descending per row (only need top max_k).
    max_k = max(ks)
    top = np.argpartition(-sim, kth=max_k, axis=1)[:, :max_k]
    # Order the top-K by actual similarity for correct @k slicing.
    for i in range(n):
        order = np.argsort(-sim[i, top[i]])
        top[i] = top[i][order]
    for k in ks:
        hits = 0
        for i in range(n):
            neigh = top[i, :k]
            hits += int((y[neigh] == y[i]).any())
        out[f'recall@{k}'] = float(hits) / n
    return out


def _metrics_from_labels(labels, y, sim):
    pur = float(compute_purity(labels, y))
    tp = per_type_purity(labels, y)
    macro = float(np.mean(list(tp.values()))) if tp else 0.0
    k = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    noise = float((labels == -1).mean())
    return {'purity': pur, 'macro_purity': macro, 'k': int(k),
            'noise_frac': noise}


def main():
    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt, txt_idx = load_text(gids, TEXT_VERSION, TEXT_AGG)
    X_vis, vis_idx = load_visual(gids, VISUAL_ENCODER, VISUAL_VARIANT)
    common = np.intersect1d(txt_idx, vis_idx)
    print(f'[P2b] intersection: {len(common)} / {len(gids)} objects')

    geo_b = StandardScaler().fit_transform(geo[common])
    txt_b = X_txt[np.searchsorted(txt_idx, common)]
    vis_b = X_vis[np.searchsorted(vis_idx, common)]
    y = y_full[common]

    # Per-modality similarity matrices
    print('[P2b] computing per-modality cosine similarity matrices…')
    S_geo = _sim(geo_b)
    S_txt = _sim(txt_b)
    S_vis = _sim(vis_b)

    SUBSETS = {
        'geo+text': [('geo', S_geo), ('text', S_txt)],
        'geo+visual': [('geo', S_geo), ('visual', S_vis)],
        'geo+text+visual': [('geo', S_geo), ('text', S_txt), ('visual', S_vis)],
    }

    # Retrieval eval per-modality and late-fused
    retrieval_rows = []
    for mod_name, S in [('geo', S_geo), ('text', S_txt), ('visual', S_vis)]:
        r = _recall_at_k(S, y)
        retrieval_rows.append({'config': mod_name, **r})
        print(f'  recall/{mod_name}: {r}')
    for subset_label, mods in SUBSETS.items():
        S_fused = np.mean([m[1] for m in mods], axis=0)
        r = _recall_at_k(S_fused, y)
        retrieval_rows.append({'config': f'late_{subset_label}', **r})
        print(f'  recall/late_{subset_label}: {r}')
    pd.DataFrame(retrieval_rows).to_csv(
        OUT_DIR / 'late_fusion_retrieval.csv', index=False,
    )

    # Clustering via HDBSCAN on precomputed distance = 1 - S_fused.
    print('\n[P2b] clustering per subset…')
    cluster_rows = []
    run_id = 0
    for subset_label, mods in SUBSETS.items():
        S_fused = np.mean([m[1] for m in mods], axis=0)
        D = (1.0 - S_fused).astype(np.float64)
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, None)
        # Symmetrize
        D = (D + D.T) / 2.0
        # Use a representation only for force-assign centroids (std-concat)
        parts = []
        for n, _ in mods:
            if n == 'geo':
                parts.append(geo_b)
            elif n == 'text':
                parts.append(StandardScaler().fit_transform(txt_b))
            elif n == 'visual':
                parts.append(StandardScaler().fit_transform(vis_b))
        X_for_force = np.concatenate(parts, axis=1).astype(np.float32)

        for (mcs, ms, method, force) in HDBSCAN_GRID_FACTORIAL:
            lab_raw = HDBSCAN(
                min_cluster_size=mcs, min_samples=ms,
                cluster_selection_method=method, metric='precomputed',
                n_jobs=-1,
            ).fit_predict(D)
            lab = _force_assign(lab_raw, X_for_force) if force else lab_raw
            m = _metrics_from_labels(lab, y, S_fused)
            cluster_rows.append({
                'run_id': run_id, 'subset': subset_label, 'recipe': 'F4_late',
                'algo': 'hdbscan',
                'hps': json.dumps({'mcs': mcs, 'ms': ms, 'method': method,
                                    'force': bool(force)}, sort_keys=True),
                'hp_mcs': mcs, 'hp_ms': ms, 'hp_method': method,
                'hp_force': bool(force),
                **m,
            })
            run_id += 1

    pd.DataFrame(cluster_rows).to_csv(OUT_DIR / 'late_fusion.csv', index=False)
    print(f'saved {len(cluster_rows)} rows → {OUT_DIR / "late_fusion.csv"}')

    # Headline
    cdf = pd.DataFrame(cluster_rows)
    idx = cdf.groupby('subset')['macro_purity'].idxmax()
    best = cdf.loc[idx, ['subset', 'hps', 'purity', 'macro_purity', 'k',
                          'noise_frac']]
    print('\n[P2b] best late-fusion per subset:')
    print(best.to_string(index=False))


if __name__ == '__main__':
    main()
