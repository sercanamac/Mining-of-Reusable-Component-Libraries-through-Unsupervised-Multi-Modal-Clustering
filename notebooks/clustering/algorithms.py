"""Clustering algorithms with hyperparameter search using silhouette score."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from itertools import product
import numpy as np
from sklearn.cluster import HDBSCAN, MiniBatchKMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    name: str
    labels: np.ndarray
    params: Dict[str, Any]
    silhouette: float | None = None
    model: Any = field(default=None, repr=False)


@dataclass
class GridSearchResult:
    """Container for grid search results."""
    best_result: ClusteringResult
    all_results: List[Dict[str, Any]]


def _compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score, handling edge cases."""
    mask = labels != -1
    n_valid = mask.sum()
    n_clusters = len(set(labels[mask])) if n_valid > 0 else 0

    if n_valid < 2 or n_clusters < 2:
        return -1.0

    return silhouette_score(X[mask], labels[mask], sample_size=10000, random_state=42)


# =============================================================================
# Grid Search Functions
# =============================================================================

def grid_search_kmeans(
    X: np.ndarray,
    n_clusters_range: List[int] | None = None,
    verbose: bool = True
) -> GridSearchResult:
    """Grid search for MiniBatchKMeans using silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    n_clusters_range : list[int], optional
        Values of k to test. Defaults to [5, 10, 15, 20, 24, 30, 40]
    verbose : bool
        Show progress bar

    Returns
    -------
    GridSearchResult
        Best result and all results
    """
    if n_clusters_range is None:
        n_clusters_range = [5, 10, 15, 20, 24, 30, 40]

    all_results = []
    best_score = -1
    best_result = None

    iterator = tqdm(n_clusters_range, desc='KMeans') if verbose else n_clusters_range

    for k in iterator:
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=3072,
            n_init="auto",
            random_state=42,
        )
        labels = model.fit_predict(X)
        score = _compute_silhouette(X, labels)

        result = ClusteringResult(
            name='KMeans',
            labels=labels,
            params={'n_clusters': k},
            silhouette=score,
            model=model
        )

        all_results.append({'params': {'n_clusters': k}, 'silhouette': score})

        if score > best_score:
            best_score = score
            best_result = result

    return GridSearchResult(best_result=best_result, all_results=all_results)


def grid_search_bisecting_kmeans(
    X: np.ndarray,
    n_clusters_range: List[int] | None = None,
    verbose: bool = True
) -> GridSearchResult:
    """Grid search for BisectingKMeans using silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    n_clusters_range : list[int], optional
        Values of k to test. Defaults to [5, 10, 15, 20, 24, 30, 40]
    verbose : bool
        Show progress bar

    Returns
    -------
    GridSearchResult
        Best result and all results
    """
    if n_clusters_range is None:
        n_clusters_range = [5, 10, 15, 20, 24, 30, 40]

    all_results = []
    best_score = -1
    best_result = None

    iterator = tqdm(n_clusters_range, desc='BisectingKMeans') if verbose else n_clusters_range

    for k in iterator:
        model = BisectingKMeans(
            n_clusters=k,
            random_state=42,
        )
        labels = model.fit_predict(X)
        score = _compute_silhouette(X, labels)

        result = ClusteringResult(
            name='BisectingKMeans',
            labels=labels,
            params={'n_clusters': k},
            silhouette=score,
            model=model
        )

        all_results.append({'params': {'n_clusters': k}, 'silhouette': score})

        if score > best_score:
            best_score = score
            best_result = result

    return GridSearchResult(best_result=best_result, all_results=all_results)


def grid_search_hdbscan(
    X: np.ndarray,
    min_cluster_size_range: List[int] | None = None,
    min_samples_range: List[int] | None = None,
    verbose: bool = True
) -> GridSearchResult:
    """Grid search for HDBSCAN using silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    min_cluster_size_range : list[int], optional
        Values to test. Defaults to [50, 100, 200, 500]
    min_samples_range : list[int], optional
        Values to test. Defaults to [5, 10, 25, 50]
    verbose : bool
        Show progress bar

    Returns
    -------
    GridSearchResult
        Best result and all results
    """
    if min_cluster_size_range is None:
        min_cluster_size_range = [50, 100, 200, 500]
    if min_samples_range is None:
        min_samples_range = [5, 10, 25, 50]

    param_combinations = list(product(min_cluster_size_range, min_samples_range))

    all_results = []
    best_score = -1
    best_result = None

    iterator = tqdm(param_combinations, desc='HDBSCAN') if verbose else param_combinations

    for min_cluster_size, min_samples in iterator:
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_jobs=-1
        )
        labels = model.fit_predict(X)
        score = _compute_silhouette(X, labels)

        params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples
        }

        result = ClusteringResult(
            name='HDBSCAN',
            labels=labels,
            params=params,
            silhouette=score,
            model=model
        )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)

        all_results.append({
            'params': params,
            'silhouette': score,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio
        })

        if score > best_score:
            best_score = score
            best_result = result

    return GridSearchResult(best_result=best_result, all_results=all_results)


def grid_search_gmm(
    X: np.ndarray,
    n_components_range: List[int] | None = None,
    verbose: bool = True
) -> GridSearchResult:
    """Grid search for GMM using BIC (lower is better).

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    n_components_range : list[int], optional
        Values to test. Defaults to [5, 10, 15, 20, 24, 30]
    verbose : bool
        Show progress bar

    Returns
    -------
    GridSearchResult
        Best result and all results (by BIC)
    """
    if n_components_range is None:
        n_components_range = [5, 10, 15, 20, 24, 30]

    all_results = []
    best_bic = float('inf')
    best_result = None

    iterator = tqdm(n_components_range, desc='GMM') if verbose else n_components_range

    for k in iterator:
        model = GaussianMixture(
            n_components=k,
            covariance_type='full',
            n_init=5,
            random_state=42
        )
        labels = model.fit_predict(X)
        bic = model.bic(X)
        score = _compute_silhouette(X, labels)

        result = ClusteringResult(
            name='GMM',
            labels=labels,
            params={'n_components': k},
            silhouette=score,
            model=model
        )

        all_results.append({
            'params': {'n_components': k},
            'bic': bic,
            'silhouette': score
        })

        if bic < best_bic:
            best_bic = bic
            best_result = result

    return GridSearchResult(best_result=best_result, all_results=all_results)


# =============================================================================
# Direct Run Functions (with specific params)
# =============================================================================

def run_kmeans(X: np.ndarray, n_clusters: int) -> ClusteringResult:
    """Run MiniBatchKMeans with specified k."""
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=2048,
        n_init=10,
        random_state=42
    )
    labels = model.fit_predict(X)
    score = _compute_silhouette(X, labels)
    return ClusteringResult(
        name='KMeans',
        labels=labels,
        params={'n_clusters': n_clusters},
        silhouette=score,
        model=model
    )


def run_bisecting_kmeans(X: np.ndarray, n_clusters: int) -> ClusteringResult:
    """Run BisectingKMeans with specified k."""
    model = BisectingKMeans(
        n_clusters=n_clusters,
        random_state=42
    )
    labels = model.fit_predict(X)
    score = _compute_silhouette(X, labels)
    return ClusteringResult(
        name='BisectingKMeans',
        labels=labels,
        params={'n_clusters': n_clusters},
        silhouette=score,
        model=model
    )


def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10
) -> ClusteringResult:
    """Run HDBSCAN clustering."""
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_jobs=-1
    )
    labels = model.fit_predict(X)
    score = _compute_silhouette(X, labels)
    return ClusteringResult(
        name='HDBSCAN',
        labels=labels,
        params={
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples
        },
        silhouette=score,
        model=model
    )


def run_gmm(
    X: np.ndarray,
    n_components: int,
    covariance_type: str = 'full'
) -> ClusteringResult:
    """Run Gaussian Mixture Model clustering."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=5,
        random_state=42
    )
    labels = model.fit_predict(X)
    score = _compute_silhouette(X, labels)
    return ClusteringResult(
        name='GMM',
        labels=labels,
        params={
            'n_components': n_components,
            'covariance_type': covariance_type
        },
        silhouette=score,
        model=model
    )
