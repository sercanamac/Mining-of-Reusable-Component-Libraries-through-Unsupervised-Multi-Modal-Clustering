"""Post-hoc analysis helpers for per-type sweep results.

Reads long-format per_type CSVs produced by `_sweep_runner.save_sweep` and
produces:
  * best-run-per-modality summaries
  * 17-type × modality purity matrices for the heatmap figure
  * diffs between stages (e.g. geo-only vs geo+text)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def best_row_per_algo(summary_df: pd.DataFrame,
                      group_cols: list[str],
                      metric: str = 'macro_purity') -> pd.DataFrame:
    """For each group, return the single row maximizing `metric`.

    Collapses kmeans/gmm multi-seed runs to their best-seed representative.
    For full fidelity use `best_mean_per_algo` instead.
    """
    idx = summary_df.groupby(group_cols)[metric].idxmax()
    return summary_df.loc[idx].reset_index(drop=True)


def best_mean_per_algo(summary_df: pd.DataFrame,
                       group_cols: list[str],
                       metric: str = 'macro_purity',
                       seed_col: str = 'hp_seed') -> pd.DataFrame:
    """Average across seeds within each hyperparam combo, then pick the
    best hyperparam combo per group.

    For hdbscan (no seeds) this is a no-op. For k-means/GMM it gives the
    representative 'mean across SEEDS' figure instead of best-of-N.
    """
    # Aggregate seeds: group by everything except seed
    non_seed_group = [c for c in summary_df.columns
                      if c not in (seed_col, 'run_id', metric,
                                   'purity', 'silhouette', 'davies_bouldin',
                                   'calinski_harabasz', 'k', 'noise_frac',
                                   'macro_purity')]
    # If seed_col isn't in the frame (e.g. hdbscan), just keep rows as-is.
    if seed_col in summary_df.columns:
        agg_cols = {
            'purity': 'mean', 'macro_purity': 'mean',
            'silhouette': 'mean', 'davies_bouldin': 'mean',
            'calinski_harabasz': 'mean', 'k': 'mean', 'noise_frac': 'mean',
        }
        agg_cols = {c: v for c, v in agg_cols.items() if c in summary_df.columns}
        agg = (summary_df.groupby(non_seed_group, dropna=False)
               .agg(agg_cols).reset_index())
    else:
        agg = summary_df.copy()
    idx = agg.groupby(group_cols)[metric].idxmax()
    return agg.loc[idx].reset_index(drop=True)


def per_type_matrix(per_type_df: pd.DataFrame,
                    run_ids: list[int],
                    labels: list[str] | None = None) -> pd.DataFrame:
    """Wide matrix: rows=types, cols=run labels, values=purity.

    `labels` must be the same length as `run_ids` and names the columns.
    If None, columns are named 'run_<run_id>'.
    """
    labels = labels or [f'run_{r}' for r in run_ids]
    assert len(labels) == len(run_ids)
    out = {}
    for rid, lab in zip(run_ids, labels):
        sub = per_type_df[per_type_df['run_id'] == rid].set_index('type')['purity']
        out[lab] = sub
    df = pd.DataFrame(out).fillna(0.0)
    return df.sort_index()


def summarize_best(summary_df: pd.DataFrame,
                   metric: str = 'macro_purity') -> pd.DataFrame:
    """One-line summary per algo with the best score and which hyperparams won."""
    idx = summary_df.groupby('algo')[metric].idxmax()
    best = summary_df.loc[idx].reset_index(drop=True)
    return best[['name', 'algo', 'hps', 'purity', 'macro_purity',
                 'silhouette', 'davies_bouldin', 'k', 'noise_frac']]
