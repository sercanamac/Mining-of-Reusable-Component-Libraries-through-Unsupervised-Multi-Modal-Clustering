"""Clustering module for BIM OBB features."""
from .features import load_features, get_feature_matrix, FEATURE_COLS
from .algorithms import (
    ClusteringResult,
    GridSearchResult,
    # Grid search functions
    grid_search_kmeans,
    grid_search_bisecting_kmeans,
    grid_search_hdbscan,
    grid_search_gmm,
    # Direct run functions
    run_kmeans,
    run_bisecting_kmeans,
    run_hdbscan,
    run_gmm,
)
from .metrics import evaluate, compare_results
from .visualization import (
    plot_silhouette_curve,
    plot_grid_search_heatmap,
    plot_algorithm_comparison,
    plot_silhouette_analysis,
    plot_2d_projection,
    plot_feature_distributions,
    plot_cluster_sizes,
    # IFC type analysis
    analyze_cluster_composition,
    plot_cluster_ifc_heatmap,
    plot_type_cluster_distribution,
    print_cluster_summary,
)

__all__ = [
    # Features
    'load_features',
    'get_feature_matrix',
    'FEATURE_COLS',
    # Algorithms - Results
    'ClusteringResult',
    'GridSearchResult',
    # Algorithms - Grid Search
    'grid_search_kmeans',
    'grid_search_bisecting_kmeans',
    'grid_search_hdbscan',
    'grid_search_gmm',
    # Algorithms - Direct Run
    'run_kmeans',
    'run_bisecting_kmeans',
    'run_hdbscan',
    'run_gmm',
    # Metrics
    'evaluate',
    'compare_results',
    # Visualization
    'plot_silhouette_curve',
    'plot_grid_search_heatmap',
    'plot_algorithm_comparison',
    'plot_silhouette_analysis',
    'plot_2d_projection',
    'plot_feature_distributions',
    'plot_cluster_sizes',
    # IFC Type Analysis
    'analyze_cluster_composition',
    'plot_cluster_ifc_heatmap',
    'plot_type_cluster_distribution',
    'print_cluster_summary',
]
