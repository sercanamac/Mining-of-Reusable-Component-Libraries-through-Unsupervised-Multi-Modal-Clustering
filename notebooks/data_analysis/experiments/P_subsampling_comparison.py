"""
P_subsampling_comparison.py

Backs the §4.3 representativeness claim: how does the K-Means-curated annotated
subset (annotated_subset.json) compare to uniform random sampling of the same
budget, after applying TYPE_MAP merging and EXCLUDED_TYPES filtering?

Two metrics per merged class:
  (1) sample count (sanity: random follows full-pop frequencies; K-Means is balanced)
  (2) coverage of the natural sub-cluster structure: fit K-Means with
      K = min(50, N_class) on full-population features for that class,
      then count the fraction of those K centroids that have >=1 subset object
      assigned to them.

Random sampling is run over 10 seeds and reported as mean +/- std.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

PROJECT       = Path(__file__).resolve().parents[3]
FULL_FEATURES = PROJECT / 'processed_data/features/train_features.parquet'
ANNOTATED_JSON= PROJECT / 'data/annotated_subset.json'
RESULTS_DIR   = PROJECT / 'notebooks/data_analysis/results/feature_engineering'
OUT_PER_CLASS = RESULTS_DIR / 'subsampling_comparison_per_class.csv'
OUT_SUMMARY   = RESULTS_DIR / 'subsampling_comparison_summary.csv'

# BENTLEY spec, copied from feature_engineering/_common.py for self-containment.
TYPE_MAP = {
    'IfcWall': 'Wall', 'IfcWallStandardCase': 'Wall',
    'IfcFurniture': 'Furniture', 'IfcFurnishingElement': 'Furniture',
    'IfcSystemFurnitureElement': 'Furniture',
    'IfcDistributionFlowElement': 'MEP', 'IfcFlowTerminal': 'MEP',
    'IfcStair': 'Stair', 'IfcStairFlight': 'Stair',
}
EXCLUDED       = {'IfcBuildingElementProxy', 'IfcBuildingElementPart'}
N_SEEDS        = 10
KMEANS_K       = 50          # natural sub-clusters per class
FEATURE_COLS   = ['Length', 'CrossSectionArea', 'Volume', 'NumVertices']


def merge_label(t: str) -> str:
    return TYPE_MAP.get(t, t.replace('Ifc', ''))


def apply_merge(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df['IfcType'].isin(EXCLUDED)].copy()
    df['Class'] = df['IfcType'].map(merge_label)
    return df


def log_std_features(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.log1p(X))


def coverage_per_class(subset_ids: set, full_merged: pd.DataFrame, k: int):
    """For each merged class, fit K-Means on full-pop features, count
    how many of the K centroids have >=1 subset object."""
    out = {}
    for cls, grp in full_merged.groupby('Class'):
        n_full = len(grp)
        k_eff  = min(k, n_full)
        if n_full < 2 or k_eff < 2:
            out[cls] = 1.0 if grp['GlobalId'].isin(subset_ids).any() else 0.0
            continue
        Xs = log_std_features(grp[FEATURE_COLS].values)
        km = MiniBatchKMeans(n_clusters=k_eff, n_init=3, random_state=0,
                             batch_size=min(1024, n_full)).fit(Xs)
        labels = pd.Series(km.labels_, index=grp['GlobalId'].values)
        hits   = labels[labels.index.isin(subset_ids)].unique()
        out[cls] = len(hits) / k_eff
    return out


def main():
    print(f"[load] {FULL_FEATURES}")
    full = pd.read_parquet(FULL_FEATURES)
    print(f"       full population: {len(full):,} rows, {full['IfcType'].nunique()} IFC types")

    annot_records = json.load(open(ANNOTATED_JSON))
    annot_ids = {r['GlobalId'] for r in annot_records}
    print(f"[load] {ANNOTATED_JSON}: {len(annot_ids):,} GlobalIds")

    full_merged = apply_merge(full)
    print(f"[merge] full pop after TYPE_MAP + EXCLUDED: {len(full_merged):,} rows, "
          f"{full_merged['Class'].nunique()} classes")

    annotated = full_merged[full_merged['GlobalId'].isin(annot_ids)].copy()
    print(f"[annotated subset after merge+exclude] {len(annotated):,} rows")

    budget = len(annot_records)   # 1849 — same annotation budget for random
    print(f"\n[compare] budget = {budget}, K_per_class = {KMEANS_K}, seeds = {N_SEEDS}")

    rng = np.random.default_rng(20260522)
    all_ids = full['GlobalId'].values

    # K-Means subset coverage (single point — by construction deterministic)
    print("[coverage] annotated subset ...")
    ann_cov = coverage_per_class(set(annotated['GlobalId']), full_merged, KMEANS_K)

    # Random subset coverage, 10 seeds
    rand_class_counts: list[dict] = []
    rand_cov_runs: list[dict]     = []
    for seed in range(N_SEEDS):
        sample = set(rng.choice(all_ids, size=budget, replace=False))
        rand_merged = full_merged[full_merged['GlobalId'].isin(sample)]
        rand_class_counts.append(rand_merged['Class'].value_counts().to_dict())
        print(f"[coverage] uniform seed {seed} (n_after_merge={len(rand_merged)}) ...")
        rand_cov_runs.append(coverage_per_class(sample, full_merged, KMEANS_K))

    # ── Stratified random per IFC type (matches actual annotated per-type counts) ──
    annot_per_type = pd.DataFrame(annot_records)['IfcType'].value_counts().to_dict()
    print(f"\n[stratified] per-IFC-type budget from annotated_subset.json: "
          f"{len(annot_per_type)} types, total={sum(annot_per_type.values())}")
    strat_class_counts: list[dict] = []
    strat_cov_runs: list[dict]     = []
    rng2 = np.random.default_rng(20260523)
    for seed in range(N_SEEDS):
        sample_strat: set = set()
        for t, n_t in annot_per_type.items():
            pool = full.loc[full['IfcType'] == t, 'GlobalId'].values
            if len(pool) <= n_t:
                sample_strat.update(pool.tolist())
            else:
                sample_strat.update(rng2.choice(pool, size=n_t, replace=False).tolist())
        strat_merged = full_merged[full_merged['GlobalId'].isin(sample_strat)]
        strat_class_counts.append(strat_merged['Class'].value_counts().to_dict())
        print(f"[coverage] stratified seed {seed} (n_after_merge={len(strat_merged)}) ...")
        strat_cov_runs.append(coverage_per_class(sample_strat, full_merged, KMEANS_K))

    # Assemble per-class table
    classes = sorted(full_merged['Class'].unique())
    rows = []
    for c in classes:
        rand_cov_arr  = np.array([r.get(c, 0.0) for r in rand_cov_runs])
        rand_cnt_arr  = np.array([r.get(c, 0)   for r in rand_class_counts])
        strat_cov_arr = np.array([r.get(c, 0.0) for r in strat_cov_runs])
        strat_cnt_arr = np.array([r.get(c, 0)   for r in strat_class_counts])
        rows.append(dict(
            class_=c,
            full_pop_count=int((full_merged['Class'] == c).sum()),
            annotated_count=int((annotated['Class'] == c).sum()),
            uniform_count_mean=float(rand_cnt_arr.mean()),
            uniform_count_std=float(rand_cnt_arr.std()),
            stratified_count_mean=float(strat_cnt_arr.mean()),
            stratified_count_std=float(strat_cnt_arr.std()),
            annotated_coverage=float(ann_cov.get(c, np.nan)),
            uniform_coverage_mean=float(rand_cov_arr.mean()),
            uniform_coverage_std=float(rand_cov_arr.std()),
            stratified_coverage_mean=float(strat_cov_arr.mean()),
            stratified_coverage_std=float(strat_cov_arr.std()),
            lift_vs_uniform_abs=float(ann_cov.get(c, 0.0) - rand_cov_arr.mean()),
            lift_vs_stratified_abs=float(ann_cov.get(c, 0.0) - strat_cov_arr.mean()),
        ))
    per_class = pd.DataFrame(rows).rename(columns={'class_': 'class'})
    per_class.to_csv(OUT_PER_CLASS, index=False)

    # Headline summary
    summary = pd.DataFrame([{
        'budget': budget,
        'n_classes': len(classes),
        'annotated_median_coverage':         float(per_class['annotated_coverage'].median()),
        'uniform_median_coverage':           float(per_class['uniform_coverage_mean'].median()),
        'stratified_median_coverage':        float(per_class['stratified_coverage_mean'].median()),
        'median_lift_vs_uniform_abs':        float(per_class['lift_vs_uniform_abs'].median()),
        'mean_lift_vs_uniform_abs':          float(per_class['lift_vs_uniform_abs'].mean()),
        'median_lift_vs_stratified_abs':     float(per_class['lift_vs_stratified_abs'].median()),
        'mean_lift_vs_stratified_abs':       float(per_class['lift_vs_stratified_abs'].mean()),
        'annotated_min_class_size':          int(per_class['annotated_count'].min()),
        'annotated_max_class_size':          int(per_class['annotated_count'].max()),
    }])
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\n=== Per-class results ===")
    print(per_class.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("\n=== Headline ===")
    print(f"  Annotated (K-Means-curated)         median per-class coverage: "
          f"{per_class['annotated_coverage'].median():.3f}")
    print(f"  Uniform random (over full pop)      median per-class coverage: "
          f"{per_class['uniform_coverage_mean'].median():.3f}")
    print(f"  Stratified random (per-IFC-type)    median per-class coverage: "
          f"{per_class['stratified_coverage_mean'].median():.3f}")
    print(f"     K-Means lift vs uniform     : "
          f"median {per_class['lift_vs_uniform_abs'].median():+.3f}  "
          f"mean {per_class['lift_vs_uniform_abs'].mean():+.3f} (abs)")
    print(f"     K-Means lift vs stratified  : "
          f"median {per_class['lift_vs_stratified_abs'].median():+.3f}  "
          f"mean {per_class['lift_vs_stratified_abs'].mean():+.3f} (abs)")
    print(f"\nWrote: {OUT_PER_CLASS}")
    print(f"Wrote: {OUT_SUMMARY}")


if __name__ == '__main__':
    main()
