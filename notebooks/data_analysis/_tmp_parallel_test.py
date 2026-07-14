"""Tiny benchmark: sequential vs parallel HDBSCAN sweep.

Tests two parallelization granularities:
  fine:   each worker = 1 HDBSCAN config       (overhead per task)
  coarse: each worker = 60 configs for one variant  (amortized overhead)

Run:
    python notebooks/data_analysis/_tmp_parallel_test.py
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN

N = 1700
SEED = 42

HDBSCAN_GRID_60 = [
    (mcs, ms, method, force)
    for mcs    in [5, 10, 15, 20, 30]
    for ms     in [3, 5, 10]
    for method in ['eom', 'leaf']
    for force  in [True, False]
]


def _run_one(X: np.ndarray, cfg: tuple) -> dict:
    mcs, ms, method, force = cfg
    h = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=method)
    lab = h.fit_predict(X)
    if force and (-1 in lab):
        non_noise = lab != -1
        if non_noise.sum() > 0:
            cluster_ids = sorted(set(lab[non_noise].tolist()))
            centroids = np.stack([X[lab == c].mean(0) for c in cluster_ids])
            noise_idx = np.where(~non_noise)[0]
            d = np.linalg.norm(X[noise_idx][:, None, :] - centroids[None, :, :], axis=2)
            lab[noise_idx] = np.array(cluster_ids)[d.argmin(1)]
    k = len(set(lab.tolist())) - (1 if -1 in lab else 0)
    return dict(cfg=cfg, k=k)


def _run_bundle(X: np.ndarray, grid: list) -> list:
    """One worker: run a full 60-config sweep on one feature matrix."""
    return [_run_one(X, c) for c in grid]


def make_X(d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(8, d))
    X = np.vstack([centers[i] + rng.normal(size=(N // 8, d)) for i in range(8)])
    if X.shape[0] < N:
        X = np.vstack([X, rng.normal(size=(N - X.shape[0], d))])
    return X


def main():
    print(f'CPUs = {os.cpu_count()}')
    # Simulate a realistic mix: 7 encoders × 5 variant dims = 35 bundles
    # representative dims spanning the fusion sweep
    bundle_dims = [9, 17, 25, 41, 73] * 7   # 35 bundles
    print(f'simulating {len(bundle_dims)} bundles × 60 configs = {len(bundle_dims)*60} runs')
    print()

    # Pre-build all X matrices (would normally come from PCA/UMAP cache)
    Xs = [make_X(d, seed=42 + i) for i, d in enumerate(bundle_dims)]

    # ── Sequential ──
    t0 = time.perf_counter()
    seq_results = [_run_bundle(X, HDBSCAN_GRID_60) for X in Xs]
    seq_t = time.perf_counter() - t0
    total_runs = len(bundle_dims) * 60
    print(f'sequential          : {seq_t:.1f}s  ({seq_t/total_runs*1000:.0f} ms/run, {seq_t/len(Xs):.2f}s/bundle)')

    # ── Coarse-grained parallel (one bundle per worker) ──
    for n_workers in [2, 4, 6, 8]:
        t0 = time.perf_counter()
        par_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_run_bundle)(X, HDBSCAN_GRID_60) for X in Xs
        )
        par_t = time.perf_counter() - t0
        speedup = seq_t / par_t
        print(f'parallel ({n_workers}w bundle): {par_t:.1f}s  speedup {speedup:.2f}x')

    # sanity check
    seq_ks = [[r['k'] for r in bundle] for bundle in seq_results]
    par_ks = [[r['k'] for r in bundle] for bundle in par_results]
    print()
    print(f'sanity (k values match): {seq_ks == par_ks}')


if __name__ == '__main__':
    main()
