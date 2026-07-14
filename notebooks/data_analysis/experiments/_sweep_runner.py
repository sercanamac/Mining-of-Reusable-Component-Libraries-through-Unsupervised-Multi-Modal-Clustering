"""Unified clustering sweep harness.

For a given feature matrix X and label vector y, runs one or more clustering
algorithms over their hyperparameter grids and returns two long-format
DataFrames:

    summary_df   — one row per (algorithm, hyperparam combo, seed)
    per_type_df  — one row per (run_id, merged-IFC-type)

All downstream phase scripts (P1a..P1c, P2, etc.) should delegate here.

Design decisions:
  * MiniBatchKMeans and GMM are run across `SEEDS` from `_common` (default 10
    seeds). BisectingKMeans and HDBSCAN use `seeds[:N_SEEDS_SLOW]` (default 3)
    because they are slower and more deterministic.
  * HDBSCAN supports an optional force-assignment of noise points to the
    nearest cluster centroid — this is the recipe that produced the
    breakthrough 0.748 purity number.
  * Scaling is applied *inside* the runner (StandardScaler) unless `scale=False`,
    so every algorithm sees the same Xs.
  * Silhouette/DB/CH are computed on non-noise points only; purity is computed
    over all points (noise counted as its own cluster).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, HDBSCAN, MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
from _common import K_VALUES, SEEDS, compute_purity, per_type_purity  # noqa: E402

N_SEEDS_SLOW = 3  # for bisecting / GMM

# Compact HDBSCAN grid — covers the known-good mcs=10 combos plus neighbors.
DEFAULT_HDBSCAN_GRID = [
    # (min_cluster_size, min_samples, method, force_assign)
    (5, 3, 'eom', True),
    (5, 5, 'eom', True),
    (10, 3, 'eom', True),
    (10, 5, 'eom', True),
    (10, 10, 'eom', True),
    (15, 5, 'eom', True),
    (15, 10, 'eom', True),
    (20, 5, 'eom', True),
    (20, 10, 'eom', True),
    (30, 5, 'eom', True),
    (30, 10, 'eom', True),
    # leaf variants of two hot configs
    (10, 5, 'leaf', True),
    (15, 5, 'leaf', True),
    # unforced variants — report raw HDBSCAN behaviour too
    (10, 5, 'eom', False),
    (15, 5, 'eom', False),
]

DEFAULT_GMM_COVS = ['full', 'diag']

# Full factorial HDBSCAN grid used in the §12 re-run. 60 combos:
#   mcs ∈ {5,10,15,20,30} × ms ∈ {3,5,10} × method ∈ {eom,leaf} × force ∈ {T,F}
HDBSCAN_GRID_FACTORIAL = [
    (mcs, ms, method, force)
    for mcs    in [5, 10, 15, 20, 30]
    for ms     in [3, 5, 10]
    for method in ['eom', 'leaf']
    for force  in [True, False]
]
assert len(HDBSCAN_GRID_FACTORIAL) == 60


# ── internals ───────────────────────────────────────────────────────────────
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


def _metrics(Xs: np.ndarray, labels: np.ndarray, y: np.ndarray) -> tuple[dict, dict]:
    """Return (metrics_dict, per_type_dict)."""
    overall_pur = float(compute_purity(labels, y))
    tp = per_type_purity(labels, y)
    macro_pur = float(np.mean(list(tp.values()))) if tp else 0.0

    mask = labels != -1
    has_structure = mask.sum() >= 2 and len(set(labels[mask].tolist())) >= 2
    if has_structure:
        Xs_sub, lab_sub = Xs[mask], labels[mask]
        sample_size = min(5000, len(lab_sub))
        sil = float(silhouette_score(Xs_sub, lab_sub,
                                     sample_size=sample_size, random_state=42))
        db = float(davies_bouldin_score(Xs_sub, lab_sub))
        ch = float(calinski_harabasz_score(Xs_sub, lab_sub))
    else:
        sil = db = ch = float('nan')

    k_eff = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    noise = float((labels == -1).mean())

    return (
        dict(purity=overall_pur, macro_purity=macro_pur,
             silhouette=sil, davies_bouldin=db, calinski_harabasz=ch,
             k=int(k_eff), noise_frac=noise),
        {str(k): float(v) for k, v in tp.items()},
    )


# ── public API ──────────────────────────────────────────────────────────────
def run_sweep(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    algorithms: list[str] | None = None,
    scale: bool = True,
    k_values: list[int] | None = None,
    hdbscan_grid: list[tuple] | None = None,
    gmm_covariances: list[str] | None = None,
    seeds: list[int] | None = None,
    extra_meta: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one or more clustering algorithms over X and return long-format CSV rows.

    Parameters
    ----------
    name : str
        Identifier stored in every output row (e.g. 'geo', 'text_v7_single').
    X : np.ndarray
        (n, d) feature matrix.
    y : np.ndarray
        (n,) merged IFC-type labels.
    algorithms : list of {'kmeans', 'hdbscan', 'gmm', 'bisecting'}
        Which algorithms to run. Default: all four.
    scale : bool
        If True, apply StandardScaler to X before clustering.
    k_values : list[int]
        k values for kmeans/gmm/bisecting. Default: _common.K_VALUES.
    hdbscan_grid : list of (mcs, ms, method, force_assign)
        HDBSCAN combos. Default: DEFAULT_HDBSCAN_GRID.
    gmm_covariances : list[str]
        Covariance types for GMM. Default: ['full', 'diag'].
    seeds : list[int]
        Seeds for kmeans. Default: _common.SEEDS (10 seeds).
    extra_meta : dict
        Extra columns to inject into every row (e.g. {'aggregation': 'single'}).
    verbose : bool
        Show tqdm progress bars.

    Returns
    -------
    summary_df : DataFrame with columns
        [run_id, name, algo, <extra_meta keys>, hps, hp_*, purity, macro_purity,
         silhouette, davies_bouldin, calinski_harabasz, k, noise_frac]
    per_type_df : DataFrame with columns
        [run_id, name, algo, <extra_meta keys>, hps, type, purity, support]
    """
    algorithms = algorithms or ['kmeans', 'hdbscan', 'gmm', 'bisecting']
    k_values = k_values or list(K_VALUES)
    hdbscan_grid = hdbscan_grid or DEFAULT_HDBSCAN_GRID
    gmm_covariances = gmm_covariances or DEFAULT_GMM_COVS
    seeds = seeds or list(SEEDS)
    extra_meta = extra_meta or {}

    Xs = (StandardScaler().fit_transform(X) if scale else np.asarray(X)).astype(np.float64)

    summary_rows: list[dict] = []
    per_type_rows: list[dict] = []
    run_counter = [0]

    support_by_type = {str(t): int((y == t).sum()) for t in np.unique(y)}

    def _record(algo: str, hps: dict, labels: np.ndarray):
        metrics, ptype = _metrics(Xs, labels, y)
        rid = run_counter[0]
        run_counter[0] += 1
        hps_str = json.dumps(hps, sort_keys=True)
        base = dict(run_id=rid, name=name, algo=algo, **extra_meta, hps=hps_str)
        hp_cols = {f'hp_{k}': v for k, v in hps.items()}
        summary_rows.append({**base, **hp_cols, **metrics})
        for t, p in ptype.items():
            per_type_rows.append({
                **base, 'type': t, 'purity': p,
                'support': support_by_type.get(t, 0),
            })

    tq = (lambda it, **kw: tqdm(it, **kw)) if verbose else (lambda it, **kw: it)

    if 'kmeans' in algorithms:
        for k in tq(k_values, desc=f'{name}:kmeans'):
            for seed in seeds:
                lab = MiniBatchKMeans(
                    n_clusters=k, batch_size=1024, n_init='auto',
                    random_state=seed,
                ).fit_predict(Xs)
                _record('kmeans', {'k': k, 'seed': seed}, lab)

    if 'bisecting' in algorithms:
        for k in tq(k_values, desc=f'{name}:bisecting'):
            for seed in seeds[:N_SEEDS_SLOW]:
                lab = BisectingKMeans(
                    n_clusters=k, random_state=seed,
                ).fit_predict(Xs)
                _record('bisecting', {'k': k, 'seed': seed}, lab)

    if 'hdbscan' in algorithms:
        for (mcs, ms, method, force) in tq(hdbscan_grid, desc=f'{name}:hdbscan'):
            lab_raw = HDBSCAN(
                min_cluster_size=mcs, min_samples=ms,
                cluster_selection_method=method, n_jobs=-1,
            ).fit_predict(Xs)
            lab = _force_assign(lab_raw, Xs) if force else lab_raw
            _record(
                'hdbscan',
                {'mcs': mcs, 'ms': ms, 'method': method, 'force': bool(force)},
                lab,
            )

    if 'gmm' in algorithms:
        for k in tq(k_values, desc=f'{name}:gmm'):
            for cov in gmm_covariances:
                for seed in seeds[:N_SEEDS_SLOW]:
                    lab = GaussianMixture(
                        n_components=k, covariance_type=cov,
                        n_init=3, reg_covar=1e-4, random_state=seed,
                    ).fit_predict(Xs)
                    _record('gmm',
                            {'k': k, 'covariance': cov, 'seed': seed}, lab)

    summary_df = pd.DataFrame(summary_rows)
    per_type_df = pd.DataFrame(per_type_rows)
    return summary_df, per_type_df


def save_sweep(
    summary_df: pd.DataFrame,
    per_type_df: pd.DataFrame,
    summary_csv: Path,
    per_type_csv: Path,
    *,
    append: bool = False,
) -> None:
    summary_csv = Path(summary_csv)
    per_type_csv = Path(per_type_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    per_type_csv.parent.mkdir(parents=True, exist_ok=True)
    if append and summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        max_prev_rid = int(prev['run_id'].max()) if 'run_id' in prev else -1
        summary_df = summary_df.assign(run_id=summary_df['run_id'] + max_prev_rid + 1)
        per_type_df = per_type_df.assign(run_id=per_type_df['run_id'] + max_prev_rid + 1)
        summary_df = pd.concat([prev, summary_df], ignore_index=True)
        per_type_df = pd.concat(
            [pd.read_csv(per_type_csv), per_type_df], ignore_index=True,
        )
    summary_df.to_csv(summary_csv, index=False)
    per_type_df.to_csv(per_type_csv, index=False)
    print(f'saved {len(summary_df)} summary rows → {summary_csv}')
    print(f'saved {len(per_type_df)} per-type rows → {per_type_csv}')
