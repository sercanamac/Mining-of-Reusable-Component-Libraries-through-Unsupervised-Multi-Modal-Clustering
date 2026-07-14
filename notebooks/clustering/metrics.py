"""Clustering evaluation metrics (internal only)."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def evaluate(X: np.ndarray, labels: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute internal clustering metrics.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    labels : np.ndarray
        Cluster labels (-1 indicates noise for HDBSCAN)

    Returns
    -------
    dict or None
        Dictionary with metrics:
        - silhouette: [-1, 1], higher is better
        - calinski_harabasz: [0, inf), higher is better
        - davies_bouldin: [0, inf), lower is better
        - n_clusters: number of clusters (excluding noise)
        - noise_ratio: fraction of points labeled as noise

        Returns None if clustering is invalid (< 2 clusters or < 2 valid points)
    """
    mask = labels != -1
    n_valid = mask.sum()
    unique_labels = set(labels[mask]) if n_valid > 0 else set()
    n_clusters = len(unique_labels)

    if n_valid < 2 or n_clusters < 2:
        return None

    return {
        'silhouette': silhouette_score(X[mask], labels[mask]),
        'calinski_harabasz': calinski_harabasz_score(X[mask], labels[mask]),
        'davies_bouldin': davies_bouldin_score(X[mask], labels[mask]),
        'n_clusters': n_clusters,
        'noise_ratio': (~mask).sum() / len(labels),
    }


def compare_results(
    results: Dict[str, Dict],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create comparison dataframe from multiple clustering results.

    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to result dict containing 'metrics'
    metrics : list[str], optional
        Metrics to include. Defaults to all available metrics.

    Returns
    -------
    pd.DataFrame
        Comparison table with algorithms as rows and metrics as columns
    """
    if metrics is None:
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'n_clusters', 'noise_ratio']

    rows = []
    for name, result in results.items():
        if result.get('metrics'):
            row = {'algorithm': name}
            for m in metrics:
                row[m] = result['metrics'].get(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index('algorithm')
    return df
