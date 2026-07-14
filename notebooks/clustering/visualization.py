"""Clustering visualization functions."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

# Try to import UMAP, fall back to PCA-only if not available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_silhouette_curve(
    all_results: List[Dict],
    param_key: str = 'n_clusters',
    title: str = 'Silhouette Score vs Hyperparameters'
) -> plt.Figure:
    """Plot silhouette scores from grid search results.

    Parameters
    ----------
    all_results : list[dict]
        Results from grid search (each dict has 'params' and 'silhouette')
    param_key : str
        Parameter name to plot on x-axis (e.g., 'n_clusters', 'min_cluster_size')
    title : str
        Plot title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Extract data
    if param_key in all_results[0]['params']:
        x_vals = [r['params'][param_key] for r in all_results]
        scores = [r['silhouette'] for r in all_results]
    else:
        # For HDBSCAN with 2D param grid, create combined labels
        x_vals = [f"{r['params']['min_cluster_size']}-{r['params']['min_samples']}"
                  for r in all_results]
        scores = [r['silhouette'] for r in all_results]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot
    if isinstance(x_vals[0], (int, float)):
        ax.plot(x_vals, scores, 'bo-', linewidth=2, markersize=8)
        ax.set_xticks(x_vals)
    else:
        ax.bar(range(len(x_vals)), scores, color='steelblue')
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=45, ha='right')

    # Mark best
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_param = x_vals[best_idx]

    if isinstance(x_vals[0], (int, float)):
        ax.axvline(x=best_param, color='green', linestyle='--', alpha=0.7)
        ax.scatter([best_param], [best_score], color='green', s=150, zorder=5,
                   label=f'Best: {param_key}={best_param} (score={best_score:.3f})')
    else:
        ax.bar(best_idx, best_score, color='green',
               label=f'Best: {best_param} (score={best_score:.3f})')

    ax.set_xlabel(param_key.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    ax.set_ylim([0, max(scores) * 1.1])
    plt.tight_layout()
    return fig


def plot_grid_search_heatmap(
    all_results: List[Dict],
    param1: str = 'min_cluster_size',
    param2: str = 'min_samples'
) -> plt.Figure:
    """Plot heatmap for 2D grid search (HDBSCAN).

    Parameters
    ----------
    all_results : list[dict]
        Results from grid search
    param1 : str
        First parameter (rows)
    param2 : str
        Second parameter (columns)

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Build dataframe
    data = []
    for r in all_results:
        data.append({
            param1: r['params'][param1],
            param2: r['params'][param2],
            'silhouette': r['silhouette'],
            'n_clusters': r.get('n_clusters', 0),
            'noise_ratio': r.get('noise_ratio', 0)
        })
    df = pd.DataFrame(data)

    # Pivot for heatmap
    pivot = df.pivot(index=param1, columns=param2, values='silhouette')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Silhouette Score'})
    ax.set_title('HDBSCAN Grid Search: Silhouette Score', fontsize=14)
    ax.set_xlabel(param2.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(param1.replace('_', ' ').title(), fontsize=12)
    plt.tight_layout()
    return fig


def plot_algorithm_comparison(
    results: pd.DataFrame,
    metrics: Optional[List[str]] = None
) -> plt.Figure:
    """Heatmap comparing algorithms across metrics.

    Parameters
    ----------
    results : pd.DataFrame
        Comparison dataframe from compare_results()
    metrics : list[str], optional
        Metrics to display. Defaults to silhouette, calinski_harabasz, davies_bouldin

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

    # Filter to requested metrics
    plot_data = results[metrics].copy()

    # Normalize for visualization (different scales)
    plot_normalized = plot_data.copy()
    for col in plot_normalized.columns:
        col_data = plot_normalized[col]
        if col == 'davies_bouldin':
            # Invert so higher is better for visualization
            plot_normalized[col] = 1 / (col_data + 1)
        else:
            plot_normalized[col] = (col_data - col_data.min()) / (col_data.max() - col_data.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw values
    sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0])
    axes[0].set_title('Raw Metric Values', fontsize=12)
    axes[0].set_ylabel('Algorithm')

    # Normalized (for comparison)
    sns.heatmap(plot_normalized, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1])
    axes[1].set_title('Normalized (higher = better)', fontsize=12)
    axes[1].set_ylabel('Algorithm')

    plt.tight_layout()
    return fig


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = 'Silhouette Analysis'
) -> plt.Figure:
    """Per-cluster silhouette plot.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Filter out noise
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    if len(set(labels_valid)) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Not enough clusters for silhouette analysis',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    silhouette_vals = silhouette_samples(X_valid, labels_valid)
    silhouette_avg = silhouette_vals.mean()

    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10

    unique_labels = sorted(set(labels_valid))
    n_clusters = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    for i, cluster in enumerate(unique_labels):
        cluster_silhouette = silhouette_vals[labels_valid == cluster]
        cluster_silhouette.sort()

        y_upper = y_lower + len(cluster_silhouette)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_silhouette,
            facecolor=colors[i], alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette), str(cluster))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color='red', linestyle='--',
               label=f'Average: {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim([-0.2, 1])
    plt.tight_layout()
    return fig


def plot_2d_projection(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = 'pca',
    sample_size: int = 10000,
    title: Optional[str] = None
) -> plt.Figure:
    """2D scatter using PCA or UMAP.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    labels : np.ndarray
        Cluster labels
    method : str
        'pca' or 'umap'
    sample_size : int
        Max samples to plot (for performance)
    title : str, optional
        Plot title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Sample if needed
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_plot = X[idx]
        labels_plot = labels[idx]
    else:
        X_plot = X
        labels_plot = labels

    # Reduce dimensions
    if method == 'umap' and UMAP_AVAILABLE:
        reducer = UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X_plot)
        method_name = 'UMAP'
    else:
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(X_plot)
        method_name = 'PCA'
        if method == 'umap' and not UMAP_AVAILABLE:
            print("Warning: UMAP not available, using PCA instead. Install with: pip install umap-learn")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Handle noise points differently
    unique_labels = sorted(set(labels_plot))
    n_clusters = len([l for l in unique_labels if l != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    color_idx = 0
    for cluster in unique_labels:
        mask = labels_plot == cluster
        if cluster == -1:
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c='gray', s=5, alpha=0.3, label='Noise'
            )
        else:
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[colors[color_idx]], s=10, alpha=0.6, label=f'Cluster {cluster}'
            )
            color_idx += 1

    ax.set_xlabel(f'{method_name} 1', fontsize=12)
    ax.set_ylabel(f'{method_name} 2', fontsize=12)
    ax.set_title(title or f'Cluster Visualization ({method_name})', fontsize=14)

    # Only show legend if not too many clusters
    if n_clusters <= 15:
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature: str,
    cluster_col: str = 'cluster',
    top_n: int = 10
) -> plt.Figure:
    """Box plots of features by cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and cluster labels
    feature : str
        Feature column to plot
    cluster_col : str
        Column containing cluster labels
    top_n : int
        Show only top N clusters by size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Get top clusters by size
    cluster_sizes = df[cluster_col].value_counts()
    top_clusters = cluster_sizes.head(top_n).index.tolist()

    # Filter to top clusters
    df_plot = df[df[cluster_col].isin(top_clusters)].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.boxplot(column=feature, by=cluster_col, ax=ax)
    ax.set_title(f'{feature} Distribution by Cluster', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel(feature, fontsize=12)
    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    return fig


def plot_cluster_sizes(labels: np.ndarray) -> plt.Figure:
    """Bar chart of cluster sizes.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['gray' if l == -1 else plt.cm.tab20(i / len(unique))
              for i, l in enumerate(unique)]

    bars = ax.bar([str(l) for l in unique], counts, color=colors)

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Cluster Sizes', fontsize=14)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# =============================================================================
# IFC Type Analysis
# =============================================================================

def analyze_cluster_composition(
    labels: np.ndarray,
    ifc_types: np.ndarray | pd.Series
) -> Dict:
    """Analyze cluster composition by IFC type.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    ifc_types : array-like
        IFC type for each sample

    Returns
    -------
    dict
        Dictionary containing:
        - contingency: Absolute counts (clusters x types)
        - cluster_composition: Row-normalized (what types in each cluster)
        - type_distribution: Column-normalized (how each type is distributed)
        - dominant_types: Dominant IFC type per cluster
        - dominance_strength: Proportion of dominant type
        - purity: Weighted purity score
    """
    ifc_types = np.asarray(ifc_types)

    # Create contingency table
    contingency = pd.crosstab(labels, ifc_types, margins=True)

    # Row-normalized: What IFC types are in each cluster?
    cluster_composition = pd.crosstab(labels, ifc_types, normalize='index')

    # Column-normalized: How is each IFC type distributed across clusters?
    type_distribution = pd.crosstab(labels, ifc_types, normalize='columns')

    # Dominant type per cluster
    dominant_types = cluster_composition.idxmax(axis=1)
    dominance_strength = cluster_composition.max(axis=1)

    # Purity score (weighted by cluster size)
    cluster_sizes = pd.Series(labels).value_counts()
    total = cluster_sizes.sum()
    purity = (dominance_strength * cluster_sizes / total).sum()

    return {
        'contingency': contingency,
        'cluster_composition': cluster_composition,
        'type_distribution': type_distribution,
        'dominant_types': dominant_types,
        'dominance_strength': dominance_strength,
        'purity': purity,
    }


def plot_cluster_ifc_heatmap(
    labels: np.ndarray,
    ifc_types: np.ndarray | pd.Series,
    top_n_clusters: int = 15,
    top_n_types: int = 15,
    normalize: str = 'index'
) -> plt.Figure:
    """Heatmap showing cluster composition by IFC type.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    ifc_types : array-like
        IFC type for each sample
    top_n_clusters : int
        Show only top N clusters by size (excludes noise)
    top_n_types : int
        Show only top N IFC types by frequency
    normalize : str
        'index' (row) = what types in each cluster
        'columns' = how each type is distributed
        None = absolute counts

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    ifc_types = np.asarray(ifc_types)

    # Get top clusters (by size, excluding noise)
    cluster_counts = pd.Series(labels).value_counts()
    if -1 in cluster_counts.index:
        cluster_counts = cluster_counts.drop(-1)
    top_clusters = cluster_counts.head(top_n_clusters).index.tolist()

    # Get top IFC types
    type_counts = pd.Series(ifc_types).value_counts()
    top_types = type_counts.head(top_n_types).index.tolist()

    # Filter data
    mask = np.isin(labels, top_clusters) & np.isin(ifc_types, top_types)
    filtered_labels = labels[mask]
    filtered_types = ifc_types[mask]

    # Create crosstab
    if normalize:
        ct = pd.crosstab(filtered_labels, filtered_types, normalize=normalize)
        fmt = '.2f'
        cbar_label = 'Proportion'
    else:
        ct = pd.crosstab(filtered_labels, filtered_types)
        fmt = 'd'
        cbar_label = 'Count'

    # Sort clusters by size
    cluster_order = [c for c in top_clusters if c in ct.index]
    ct = ct.reindex(cluster_order)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        ct, annot=True, fmt=fmt, cmap='YlOrRd', ax=ax,
        cbar_kws={'label': cbar_label}
    )

    title = 'Cluster Composition by IFC Type'
    if normalize == 'index':
        title += ' (row-normalized)'
    elif normalize == 'columns':
        title += ' (column-normalized)'

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('IFC Type', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_type_cluster_distribution(
    labels: np.ndarray,
    ifc_types: np.ndarray | pd.Series,
    top_n_types: int = 10
) -> plt.Figure:
    """Stacked bar chart showing how each IFC type is distributed across clusters.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    ifc_types : array-like
        IFC type for each sample
    top_n_types : int
        Show only top N IFC types by frequency

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    ifc_types = np.asarray(ifc_types)

    # Get top IFC types
    type_counts = pd.Series(ifc_types).value_counts()
    top_types = type_counts.head(top_n_types).index.tolist()

    # Filter to top types
    mask = np.isin(ifc_types, top_types)
    filtered_labels = labels[mask]
    filtered_types = ifc_types[mask]

    # Create column-normalized crosstab
    ct = pd.crosstab(filtered_types, filtered_labels, normalize='index')

    # Sort by most common type
    ct = ct.reindex(top_types)

    fig, ax = plt.subplots(figsize=(14, 8))
    ct.plot(kind='barh', stacked=True, ax=ax, colormap='tab20')

    ax.set_xlabel('Proportion', fontsize=12)
    ax.set_ylabel('IFC Type', fontsize=12)
    ax.set_title('IFC Type Distribution Across Clusters', fontsize=14)
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    return fig


def print_cluster_summary(
    labels: np.ndarray,
    ifc_types: np.ndarray | pd.Series,
    top_n: int = 10
) -> None:
    """Print summary of dominant IFC types per cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    ifc_types : array-like
        IFC type for each sample
    top_n : int
        Show top N clusters
    """
    analysis = analyze_cluster_composition(labels, ifc_types)

    print(f"Purity Score: {analysis['purity']:.3f}")
    print(f"\nTop {top_n} Clusters by Dominant Type:")
    print("-" * 60)

    # Get cluster sizes
    cluster_sizes = pd.Series(labels).value_counts()

    # Sort by size (excluding noise)
    sorted_clusters = cluster_sizes.drop(-1, errors='ignore').head(top_n).index

    for cluster in sorted_clusters:
        dominant = analysis['dominant_types'][cluster]
        strength = analysis['dominance_strength'][cluster]
        size = cluster_sizes[cluster]
        print(f"Cluster {cluster:3d}: {dominant:30s} ({strength:.1%}) - {size:,} samples")
