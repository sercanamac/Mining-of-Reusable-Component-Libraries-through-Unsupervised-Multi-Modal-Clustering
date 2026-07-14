"""Resolve the 'winning' configuration for each downstream phase.

Data-driven after the §12 re-run: reads `fusion.csv` and rebuilds the
X matrix matching whichever recipe wins (F1a / F2 / F3 / F5). Downstream
scripts call `get_canonical_winner()` + `build_X_for_winner()` and stay
agnostic to the specific recipe.

Legacy API kept for the first-pass scripts:
  load_fusion_blocks()         — F2 geo+text+visual w/ PCA-32 (first-pass).
  best_fusion_row()            — single-recipe filter lookup.
  cluster_with_winner(X, algo) — refit the winning HDBSCAN on X.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual

MIDTERM = RESULTS_ROOT.parent / 'midterm'
FUSION_CSV = MIDTERM / 'fusion' / 'fusion.csv'

# First-pass canonical constants (kept for backward compat with legacy load_fusion_blocks).
CANONICAL_SUBSET = 'geo+text+visual'
CANONICAL_TEXT_VERSION = 'v7'
CANONICAL_TEXT_AGG = 'single'
CANONICAL_VISUAL_ENCODER = 'siglip'
CANONICAL_VISUAL_VARIANT = 'colorless'   # intentionally colorless: generalizable across BIM software
PCA_DIM_TEXT = 32
PCA_DIM_VISUAL = 32


# ── Data-driven winner resolution (post-§12) ───────────────────────────────
def get_canonical_winner(
    algo: str = 'hdbscan',
    subset: str | None = None,
    recipe: str | None = None,
    metric: str = 'macro_purity',
) -> dict:
    """Scan fusion.csv, return the best row for (algo, subset, recipe, metric).

    If subset or recipe is None, the filter is skipped (global best across
    that axis). Returns a dict with everything needed to rebuild X and the
    clustering labels.
    """
    if not FUSION_CSV.exists():
        raise FileNotFoundError(f'{FUSION_CSV} missing — run P2_fusion.py first')
    df = pd.read_csv(FUSION_CSV)
    sub = df[df.algo == algo].copy()
    if subset is not None:
        sub = sub[sub.subset == subset]
    if recipe is not None:
        sub = sub[sub.recipe == recipe]
    if sub.empty:
        raise ValueError(
            f'no rows for algo={algo} subset={subset} recipe={recipe}',
        )
    w = sub.loc[sub[metric].idxmax()]
    out = {
        'algo': algo,
        'subset': w['subset'],
        'recipe': w['recipe'],
        'variant': w.get('variant', '') if 'variant' in w.index else '',
        'reducer_dim': (int(w['reducer_dim'])
                         if 'reducer_dim' in w.index and not pd.isna(w.get('reducer_dim')) else None),
        'text_version': w.get('text_version', CANONICAL_TEXT_VERSION) if 'text_version' in w.index else CANONICAL_TEXT_VERSION,
        'visual_encoder': w.get('visual_encoder', f'{CANONICAL_VISUAL_ENCODER}_{CANONICAL_VISUAL_VARIANT}') if 'visual_encoder' in w.index else f'{CANONICAL_VISUAL_ENCODER}_{CANONICAL_VISUAL_VARIANT}',
        'hps': json.loads(w['hps']),
        'purity': float(w['purity']),
        'macro_purity': float(w['macro_purity']),
        'k': int(w['k']),
    }
    print(f'[winners] canonical: subset={out["subset"]} recipe={out["recipe"]} '
          f'variant={out["variant"]!r} hps={out["hps"]} '
          f'macro={out["macro_purity"]:.4f} purity={out["purity"]:.4f} k={out["k"]}')
    return out


def _parse_visual_encoder(enc_str: str) -> tuple[str, str]:
    """'siglip_colorless' → ('siglip', 'colorless'). Handles legacy 'colored' too."""
    parts = enc_str.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in ('colored', 'colorless'):
        return parts[0], parts[1]
    return CANONICAL_VISUAL_ENCODER, CANONICAL_VISUAL_VARIANT


def _weighted_block(X: np.ndarray) -> np.ndarray:
    s = StandardScaler().fit_transform(X).astype(np.float32)
    total_var = s.var(axis=0, ddof=0).sum()
    return s / np.sqrt(total_var) if total_var > 1e-9 else s


def _std(X):
    return StandardScaler().fit_transform(X).astype(np.float32)


def build_X_for_winner(winner: dict, verbose: bool = True):
    """Reconstruct the X matrix matching the winning fusion recipe.

    Returns (X, y, gids, info_dict).
    """
    subset = winner['subset']
    recipe = winner['recipe']
    reducer_dim = winner.get('reducer_dim')
    text_version = winner.get('text_version', CANONICAL_TEXT_VERSION)
    visual_enc, visual_var = _parse_visual_encoder(
        winner.get('visual_encoder', f'{CANONICAL_VISUAL_ENCODER}_{CANONICAL_VISUAL_VARIANT}'),
    )

    df, y_full = load_data()
    gids_all = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    blocks_needed = set(subset.split('+'))
    block_data = {'geo': geo}

    if 'text' in blocks_needed:
        X_txt, txt_idx = load_text(gids_all, text_version, CANONICAL_TEXT_AGG)
    else:
        X_txt, txt_idx = None, np.arange(len(gids_all))
    if 'visual' in blocks_needed:
        X_vis, vis_idx = load_visual(gids_all, visual_enc, visual_var)
    else:
        X_vis, vis_idx = None, np.arange(len(gids_all))

    common = np.intersect1d(txt_idx, vis_idx)
    if 'text' not in blocks_needed and 'visual' not in blocks_needed:
        common = np.arange(len(gids_all))

    def _align(X_raw, idx_arr):
        return X_raw[np.searchsorted(idx_arr, common)] if X_raw is not None else None

    geo_b = geo[common]
    txt_b = _align(X_txt, txt_idx)
    vis_b = _align(X_vis, vis_idx)

    ordered_blocks = []   # build in canonical order geo, text, visual
    if 'geo' in blocks_needed:
        ordered_blocks.append(('geo', geo_b))
    if 'text' in blocks_needed:
        ordered_blocks.append(('text', txt_b))
    if 'visual' in blocks_needed:
        ordered_blocks.append(('visual', vis_b))

    # Recipe dispatch
    if recipe in ('F2',):
        parts = [_std(b) for _, b in ordered_blocks]
        X = np.concatenate(parts, axis=1)
    elif recipe == 'F3':
        parts = [_weighted_block(b) for _, b in ordered_blocks]
        X = np.concatenate(parts, axis=1)
    elif recipe == 'F1a':
        if reducer_dim is None:
            raise ValueError('F1a winner missing reducer_dim')
        parts = []
        for name, b in ordered_blocks:
            if name == 'geo':
                parts.append(_std(b))
            else:
                p = PCA(n_components=int(reducer_dim), random_state=42).fit(b)
                parts.append(_std(p.transform(b)))
        X = np.concatenate(parts, axis=1)
    elif recipe == 'F5':
        if reducer_dim is None:
            raise ValueError('F5 winner missing reducer_dim')
        import umap
        parts = [_std(b) for _, b in ordered_blocks]
        concat = np.concatenate(parts, axis=1)
        umap_params = dict(n_components=int(reducer_dim), n_neighbors=30,
                            min_dist=0.0, metric='cosine', random_state=42)
        X = umap.UMAP(**umap_params).fit_transform(concat).astype(np.float32)
    elif recipe == 'F1':
        # Legacy first-pass F1 = concat + PCA + std (PCA_DIM_CONCAT=16)
        parts = [_std(b) for _, b in ordered_blocks]
        concat = np.concatenate(parts, axis=1)
        X = PCA(n_components=16, random_state=42).fit_transform(concat).astype(np.float32)
    else:
        raise ValueError(f'unknown recipe {recipe!r}')

    y = y_full[common]
    gids = gids_all[common]
    info = {
        'subset': subset, 'recipe': recipe, 'reducer_dim': reducer_dim,
        'n': len(common), 'out_dim': int(X.shape[1]),
        'text_version': text_version, 'visual_encoder': f'{visual_enc}_{visual_var}',
    }
    if verbose:
        print(f'[winners] built X for {subset}/{recipe} '
              f'(dim={X.shape[1]}, n={len(common)})')
    return X.astype(np.float64), y, gids, info


def _force_assign(lab: np.ndarray, X: np.ndarray) -> np.ndarray:
    lab = lab.copy()
    non_noise = lab != -1
    if non_noise.sum() == 0:
        return lab
    cluster_ids = sorted(set(lab[non_noise].tolist()))
    centroids = np.stack([X[lab == c].mean(0) for c in cluster_ids])
    noise_idx = np.where(~non_noise)[0]
    if noise_idx.size:
        d = np.linalg.norm(X[noise_idx][:, None, :] - centroids[None, :, :], axis=2)
        lab[noise_idx] = np.array(cluster_ids)[d.argmin(1)]
    return lab


def cluster_with_canonical_winner(algo: str = 'hdbscan', verbose: bool = True):
    """End-to-end: find global winner, rebuild X, refit, return (labels, X, y, info).

    This is the helper P5a-e should call going forward.
    """
    w = get_canonical_winner(algo=algo)
    X, y, gids, info = build_X_for_winner(w, verbose=verbose)
    if algo == 'hdbscan':
        hps = w['hps']
        lab_raw = HDBSCAN(
            min_cluster_size=int(hps['mcs']),
            min_samples=int(hps['ms']),
            cluster_selection_method=hps['method'],
            n_jobs=-1,
        ).fit_predict(X)
        labels = _force_assign(lab_raw, X) if hps.get('force', True) else lab_raw
    elif algo == 'kmeans':
        hps = w['hps']
        labels = MiniBatchKMeans(
            n_clusters=int(hps['k']), batch_size=1024, n_init='auto',
            random_state=int(hps['seed']),
        ).fit_predict(X)
    else:
        raise ValueError(algo)
    return labels, X, y, gids, {**info, **w}


# ── Legacy API (first-pass; still used by existing P4a/P4b/P5a-d) ──────────
def load_fusion_blocks(verbose: bool = True):
    """Return (X_fusion, y, gids, block_info) using the first-pass F2 recipe.

    Kept for backward compat. New code should prefer
    `cluster_with_canonical_winner()`.
    """
    df, y_full = load_data()
    gids_all = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt_raw, txt_idx = load_text(
        gids_all, CANONICAL_TEXT_VERSION, CANONICAL_TEXT_AGG,
    )
    X_vis_raw, vis_idx = load_visual(
        gids_all, CANONICAL_VISUAL_ENCODER, CANONICAL_VISUAL_VARIANT,
    )
    common = np.intersect1d(txt_idx, vis_idx)

    def _align(X_raw, idx_arr):
        return X_raw[np.searchsorted(idx_arr, common)]

    geo_b = geo[common]
    txt_b = PCA(n_components=PCA_DIM_TEXT, random_state=42).fit_transform(
        _align(X_txt_raw, txt_idx),
    )
    vis_b = PCA(n_components=PCA_DIM_VISUAL, random_state=42).fit_transform(
        _align(X_vis_raw, vis_idx),
    )
    scaled = [StandardScaler().fit_transform(b) for b in (geo_b, txt_b, vis_b)]
    X = np.hstack(scaled).astype(np.float64)
    y = y_full[common]
    gids = gids_all[common]
    if verbose:
        print(f'[winners-legacy] fusion X shape={X.shape} subset={CANONICAL_SUBSET} '
              f'recipe=F2 n={len(common)}')
    return X, y, gids, {
        'geo': geo_b, 'text': txt_b, 'visual': vis_b,
        'geo_scaled': scaled[0], 'text_scaled': scaled[1], 'visual_scaled': scaled[2],
    }


def best_fusion_row(algo: str = 'hdbscan',
                    subset: str = CANONICAL_SUBSET,
                    recipe: str = 'F2',
                    metric: str = 'macro_purity') -> dict:
    """Legacy: return best hyperparameters for a specific (subset, recipe, algo)."""
    w = get_canonical_winner(algo=algo, subset=subset, recipe=recipe, metric=metric)
    return {
        'hps': w['hps'],
        'purity': w['purity'],
        'macro_purity': w['macro_purity'],
        'k': w['k'],
        'algo': algo,
    }


def cluster_with_winner(X: np.ndarray, algo: str = 'hdbscan') -> np.ndarray:
    """Legacy: refit the winning (F2) algorithm on a pre-built X.

    New code should use `cluster_with_canonical_winner()`.
    """
    winner = best_fusion_row(algo=algo)
    hps = winner['hps']
    if algo == 'hdbscan':
        lab_raw = HDBSCAN(
            min_cluster_size=int(hps['mcs']),
            min_samples=int(hps['ms']),
            cluster_selection_method=hps['method'],
            n_jobs=-1,
        ).fit_predict(X)
        if hps.get('force', True):
            lab_raw = _force_assign(lab_raw, X)
        return lab_raw
    if algo == 'kmeans':
        return MiniBatchKMeans(
            n_clusters=int(hps['k']), batch_size=1024, n_init='auto',
            random_state=int(hps['seed']),
        ).fit_predict(X)
    raise ValueError(algo)
