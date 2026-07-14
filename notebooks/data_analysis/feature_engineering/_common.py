"""
Shared utilities for per-feature engineering scripts.

Each feature script:
  1. Loads data via load_data()
  2. Builds X_prev, X_new via build_X(...)
  3. Evaluates across K_VALUES with eval_k(...)
  4. Saves figures + CSVs + summary.md into its own results subdir
  5. Appends a row to progression_summary.csv
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
if 'ipykernel' not in sys.modules:
    # headless script run; force non-interactive backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

# Active scaler for all k-means / distance operations.
# Switched from RobustScaler → StandardScaler to equalize per-feature Euclidean
# weight (see audit in docstring / thesis chapter).
SCALER = StandardScaler

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[3]


# ── DatasetSpec ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetSpec:
    """All dataset-specific paths and label conventions in one place."""
    name: str
    key_col: str                       # primary id column ('GlobalId' or 'obj_id')
    type_col: str                      # raw type/class column ('IfcType' or 'ifc_class')
    features_parquet: Path             # baseline OBB features (Length/CSA/Volume/NumVerts)
    engineered_parquet: Path           # engineered geo features written by 00_extract_features
    mesh_dir: Path                     # where mesh OBJs live
    processed_root: Path               # root for visual + text embeddings
    results_root: Path                 # output dir for sweep CSVs/figures
    type_map: dict                     # raw → merged label map
    excluded_types: set                # raw types dropped from evaluation
    k_main: int                        # primary k for per-type delta heatmap
    k_values: list                     # k-sweep operating points
    has_splits: bool                   # whether the parquet carries an official Split column
    annotated_json: Path | None = None # extra GlobalId allow-list (Bentley only)
    progression_csv: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'progression_csv',
                           self.results_root / 'progression_summary.csv')


BENTLEY = DatasetSpec(
    name='bentley',
    key_col='GlobalId',
    type_col='IfcType',
    features_parquet=PROJECT / 'processed_data/features/train_features.parquet',
    engineered_parquet=PROJECT / 'notebooks/data_analysis/results/feature_engineering/engineered_features.parquet',
    mesh_dir=PROJECT / 'processed_data/meshes/train',
    processed_root=PROJECT / 'processed_data',
    results_root=PROJECT / 'notebooks/data_analysis/results/feature_engineering',
    type_map={
        'IfcWall': 'Wall', 'IfcWallStandardCase': 'Wall',
        'IfcFurniture': 'Furniture', 'IfcFurnishingElement': 'Furniture',
        'IfcSystemFurnitureElement': 'Furniture',
        'IfcDistributionFlowElement': 'MEP', 'IfcFlowTerminal': 'MEP',
        'IfcStair': 'Stair', 'IfcStairFlight': 'Stair',
    },
    excluded_types={'IfcBuildingElementProxy', 'IfcBuildingElementPart'},
    k_main=17,
    k_values=[2, 4, 8, 17, 24, 32],
    has_splits=False,       # annotated subset is all 'train' — no official test split
    annotated_json=PROJECT / 'data/annotated_subset.json',
)


IFCNET = DatasetSpec(
    name='ifcnet',
    key_col='obj_id',
    type_col='ifc_class',
    features_parquet=PROJECT / 'data/IFCNetCore/features/baseline_features.parquet',
    engineered_parquet=PROJECT / 'data/IFCNetCore/features/engineered_features.parquet',
    mesh_dir=PROJECT / 'data/IFCNetCore/meshes',
    processed_root=PROJECT / 'data/IFCNetCore/processed',
    results_root=PROJECT / 'notebooks/data_analysis/results/ifcnet',
    type_map={},                # no merging — native 20 classes are the labels
    excluded_types=set(),
    k_main=20,
    k_values=[2, 4, 8, 12, 20, 32, 48],
    has_splits=True,
    annotated_json=None,
)


SPECS = {'bentley': BENTLEY, 'ifcnet': IFCNET}


def get_spec(name: str | None = None) -> DatasetSpec:
    """Resolve a DatasetSpec from the IFCNET_DATASET env var or an explicit name."""
    if name is None:
        name = os.environ.get('IFCNET_DATASET', 'bentley')
    if name not in SPECS:
        raise KeyError(f'unknown dataset spec {name!r}; known: {list(SPECS)}')
    return SPECS[name]


# ── Active-spec resolution + back-compat module aliases ────────────────────────
ACTIVE_SPEC = get_spec()
ACTIVE_SPEC.results_root.mkdir(parents=True, exist_ok=True)

FEATURES_PARQUET = ACTIVE_SPEC.features_parquet
ANNOTATED_JSON = ACTIVE_SPEC.annotated_json
MESH_DIR = ACTIVE_SPEC.mesh_dir
RESULTS_ROOT = ACTIVE_SPEC.results_root
ENGINEERED_PARQUET = ACTIVE_SPEC.engineered_parquet
PROGRESSION_CSV = ACTIVE_SPEC.progression_csv

# ── Constants ──────────────────────────────────────────────────────────────────
BASELINE = ['Length', 'CrossSectionArea', 'Volume', 'NumVertices']
Z_AXIS = np.array([0.0, 0.0, 1.0])
SEEDS = list(range(42, 52))
K_VALUES = ACTIVE_SPEC.k_values
K_MAIN = ACTIVE_SPEC.k_main

TYPE_MAP = ACTIVE_SPEC.type_map
EXCLUDED_TYPES = ACTIVE_SPEC.excluded_types

ALL_ENGINEERED = [
    'pc1_z', 'pc3_z', 'cs_aspect_ratio',
    'normal_entropy', 'horiz_frac',
    # Added by autoresearch run r1:
    'branched_vertical', 'vertical_aspect_ratio',
]

# Incremental feature-add order (driven by the notebook narrative)
FEATURE_ORDER = [
    ('00_baseline',             'Baseline (4, log)',     []),
    ('01_orientation',          '+ Orientation (6)',     ['pc1_z', 'pc3_z']),
    ('02_cs_aspect_ratio',      '+ CS Ratio (7)',        ['cs_aspect_ratio']),
    ('03_normal_entropy',       '+ Normal Entropy (8)',  ['normal_entropy']),
    ('04_horiz_frac',           '+ Horiz Frac (9)',      ['horiz_frac']),
    # Added by autoresearch run r1:
    ('05_branched_vertical',    '+ Branched Vertical (10)', ['branched_vertical']),
    ('06_vertical_aspect_ratio','+ Vertical Aspect Ratio (11)', ['vertical_aspect_ratio']),
]


def cumulative_bounded_up_to(stage_id: str) -> list[str]:
    """Return all engineered (bounded) feature columns added up to and including stage_id."""
    cols: list[str] = []
    for sid, _name, feats in FEATURE_ORDER:
        cols.extend(feats)
        if sid == stage_id:
            return cols
    raise ValueError(f'Unknown stage_id: {stage_id}')


def previous_bounded(stage_id: str) -> list[str]:
    """Engineered columns in the state BEFORE stage_id is added."""
    cols: list[str] = []
    for sid, _name, feats in FEATURE_ORDER:
        if sid == stage_id:
            return cols
        cols.extend(feats)
    raise ValueError(f'Unknown stage_id: {stage_id}')


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(spec: DatasetSpec = ACTIVE_SPEC) -> tuple[pd.DataFrame, np.ndarray]:
    """Load baseline + engineered features for `spec`.

    Returns
    -------
    df_feat : merged DataFrame (baseline OBB + engineered + type col + key col)
    labels  : array of merged labels (after `spec.type_map`)
    """
    df_all = pd.read_parquet(spec.features_parquet)

    if spec.annotated_json is not None:
        annotated = json.loads(spec.annotated_json.read_text())
        annotated_gids = {d[spec.key_col] for d in annotated}
        df = df_all[df_all[spec.key_col].isin(annotated_gids)].copy()
    else:
        df = df_all.copy()
    df = df.reset_index(drop=True)

    # Drop semantic catch-all types that carry no geometric identity.
    if spec.excluded_types:
        df = df[~df[spec.type_col].isin(spec.excluded_types)].copy().reset_index(drop=True)

    if spec.engineered_parquet.exists():
        df_eng = pd.read_parquet(spec.engineered_parquet)
        df = df.merge(df_eng, on=spec.key_col, how='inner')

    labels = np.array([spec.type_map.get(t, t) for t in df[spec.type_col].values])
    return df, labels


# ── Feature matrix builder ─────────────────────────────────────────────────────
def build_X(df: pd.DataFrame, baseline_cols=BASELINE, bounded_eng=None) -> np.ndarray:
    """Log1p-transform baseline size features; pass bounded [0,1] features through."""
    X = np.log1p(df[baseline_cols].clip(lower=0).values)
    if bounded_eng:
        X = np.hstack([X, df[bounded_eng].values])
    return X


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_purity(labels: np.ndarray, true_labels: np.ndarray) -> float:
    return sum(
        pd.Series(true_labels[labels == c]).value_counts().iloc[0]
        for c in np.unique(labels)
    ) / len(labels)


def per_type_purity(labels: np.ndarray, true_labels: np.ndarray) -> dict[str, float]:
    cluster_dominant = {}
    for c in np.unique(labels):
        counts = pd.Series(true_labels[labels == c]).value_counts()
        cluster_dominant[c] = counts.index[0]
    out = {}
    for t in np.unique(true_labels):
        mask = true_labels == t
        n = mask.sum()
        correct = sum(1 for i in np.where(mask)[0] if cluster_dominant[labels[i]] == t)
        out[t] = correct / n if n > 0 else 0.0
    return out


def kmeans_labels(X: np.ndarray, k: int, seed: int, scale=True) -> tuple[np.ndarray, np.ndarray]:
    Xs = SCALER().fit_transform(X) if scale else X
    lab = MiniBatchKMeans(k, batch_size=1024, n_init='auto', random_state=seed).fit_predict(Xs)
    return lab, Xs


def eval_k(X: np.ndarray, true_labels: np.ndarray, k: int, seeds=SEEDS, scale=True) -> dict:
    """Run k-means at k across seeds; returns {metric: (mean, std)}."""
    Xs = SCALER().fit_transform(X) if scale else X
    m = {'purity': [], 'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': []}
    for s in seeds:
        lab = MiniBatchKMeans(k, batch_size=1024, n_init='auto', random_state=s).fit_predict(Xs)
        m['purity'].append(compute_purity(lab, true_labels))
        m['silhouette'].append(silhouette_score(Xs, lab))
        m['davies_bouldin'].append(davies_bouldin_score(Xs, lab))
        m['calinski_harabasz'].append(calinski_harabasz_score(Xs, lab))
    return {metric: (float(np.mean(v)), float(np.std(v))) for metric, v in m.items()}


def sweep_k(X: np.ndarray, true_labels: np.ndarray, ks=K_VALUES) -> pd.DataFrame:
    """Run eval_k across all ks → long-format DataFrame (one row per k)."""
    rows = []
    for k in ks:
        r = eval_k(X, true_labels, k)
        rows.append({
            'k': k,
            **{f'{m}_mean': r[m][0] for m in r},
            **{f'{m}_std':  r[m][1] for m in r},
        })
    return pd.DataFrame(rows)


# ── Progression CSV ────────────────────────────────────────────────────────────
def append_progression(config_name: str, sweep_df: pd.DataFrame) -> None:
    """Append (or overwrite) sweep results for a config to PROGRESSION_CSV."""
    sweep_df = sweep_df.assign(config=config_name)
    cols = ['config', 'k'] + [c for c in sweep_df.columns if c not in ('config', 'k')]
    sweep_df = sweep_df[cols]
    if PROGRESSION_CSV.exists():
        existing = pd.read_csv(PROGRESSION_CSV)
        existing = existing[existing['config'] != config_name]
        combined = pd.concat([existing, sweep_df], ignore_index=True)
    else:
        combined = sweep_df
    combined.to_csv(PROGRESSION_CSV, index=False)


# ── Plotting ───────────────────────────────────────────────────────────────────
def save_fig(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_distribution_by_type(df: pd.DataFrame, col: str, labels: np.ndarray,
                               out: Path, title: str | None = None) -> None:
    dfi = df.copy()
    dfi['_label'] = labels
    order = dfi.groupby('_label')[col].median().sort_values().index
    data = [dfi.loc[dfi['_label'] == t, col].values for t in order]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(data, patch_artist=True, showfliers=False)
    ax.set_xticklabels([str(t).replace('Ifc', '') for t in order], rotation=90, fontsize=8)
    ax.set_title(title or f'{col} by type (sorted by median)', fontsize=12)
    ax.set_ylabel(col)
    save_fig(fig, out)


def plot_distribution_overall(values: np.ndarray, col: str, out: Path) -> None:
    from scipy.stats import skew
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=50, edgecolor='white', color='steelblue', alpha=0.85)
    ax.set_title(f'{col} overall — skew = {skew(values):.2f}', fontsize=12)
    ax.set_xlabel(col); ax.set_ylabel('count')
    save_fig(fig, out)


def plot_separation_potential(df: pd.DataFrame, new_feats: list[str], labels: np.ndarray,
                               out: Path) -> None:
    """For each new feature, plot per-type median ± IQR, annotate ANOVA F-statistic."""
    fig, axes = plt.subplots(len(new_feats), 1, figsize=(14, 4 * len(new_feats)), squeeze=False)
    for ax, feat in zip(axes[:, 0], new_feats):
        dfi = df.copy()
        dfi['_label'] = labels
        groups = [g[feat].values for _, g in dfi.groupby('_label') if len(g) > 1]
        f_stat, p_val = f_oneway(*groups) if len(groups) >= 2 else (np.nan, np.nan)

        agg = dfi.groupby('_label')[feat].agg(['median', lambda s: s.quantile(0.25),
                                                lambda s: s.quantile(0.75)])
        agg.columns = ['median', 'q25', 'q75']
        agg = agg.sort_values('median')
        x = np.arange(len(agg))
        ax.bar(x, agg['median'], yerr=[agg['median']-agg['q25'], agg['q75']-agg['median']],
               capsize=3, color='steelblue', alpha=0.8, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels([str(t).replace('Ifc', '') for t in agg.index], rotation=90, fontsize=8)
        ax.set_title(f'{feat}: median ± IQR per type — ANOVA F={f_stat:.1f}, p={p_val:.2e}', fontsize=11)
        ax.set_ylabel(feat)
    save_fig(fig, out)


def plot_k_sweep(sweeps: dict[str, pd.DataFrame], out: Path) -> None:
    """sweeps: {label: sweep_df}. 3-panel: macro purity, silhouette, Davies-Bouldin.
    K_MAIN is highlighted with a dashed vertical line."""
    metrics = [
        ('purity', 'Macro purity (↑)'),
        ('silhouette', 'Silhouette (↑)'),
        ('davies_bouldin', 'Davies-Bouldin (↓)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for label, df in sweeps.items():
            ax.errorbar(df['k'], df[f'{metric}_mean'], yerr=df[f'{metric}_std'],
                        marker='o', capsize=3, label=label, alpha=0.9)
        ax.axvline(K_MAIN, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.text(K_MAIN, ax.get_ylim()[1], f' k={K_MAIN}', va='top', fontsize=8, alpha=0.7)
        ax.set_xlabel('k'); ax.set_ylabel(metric); ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
    save_fig(fig, out)


def plot_per_type_delta(X_prev: np.ndarray, X_new: np.ndarray, true_labels: np.ndarray,
                         k: int, out: Path, seed: int = 42) -> pd.DataFrame:
    """Bar chart of per-type Δpurity (new − prev) at a single k."""
    lab_prev, _ = kmeans_labels(X_prev, k, seed)
    lab_new, _ = kmeans_labels(X_new, k, seed)
    tp_prev = per_type_purity(lab_prev, true_labels)
    tp_new = per_type_purity(lab_new, true_labels)
    all_t = sorted(set(true_labels))
    rows = [{'type': str(t).replace('Ifc', ''),
             'prev': tp_prev.get(t, 0),
             'new':  tp_new.get(t, 0),
             'delta': tp_new.get(t, 0) - tp_prev.get(t, 0)} for t in all_t]
    dfd = pd.DataFrame(rows).sort_values('delta', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(dfd))))
    colors = ['#2ca02c' if d > 0 else '#d62728' if d < 0 else '#999' for d in dfd['delta']]
    ax.barh(dfd['type'], dfd['delta'], color=colors, edgecolor='white')
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Δ per-type purity (new − prev)')
    ax.set_title(f'Per-type purity change at k={k}', fontsize=12)
    save_fig(fig, out)
    return dfd


def plot_cluster_heatmap(X: np.ndarray, true_labels: np.ndarray, k: int, out: Path,
                          title: str, seed: int = 42) -> None:
    lab, Xs = kmeans_labels(X, k, seed)
    ct = pd.crosstab(pd.Series(lab, name='Cluster'),
                     pd.Series(true_labels, name='Type'), normalize='index')
    ct = ct.reindex(pd.Series(lab).value_counts().index)
    ct.columns = [str(c).replace('Ifc', '') for c in ct.columns]
    pur = compute_purity(lab, true_labels)
    sil = silhouette_score(Xs, lab)
    db  = davies_bouldin_score(Xs, lab)
    fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * k)))
    sns.heatmap(ct, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, annot_kws={'size': 6}, cbar=False)
    ax.set_title(f'{title}  |  k={k}  Purity={pur:.3f}  Sil={sil:.3f}  DB={db:.2f}', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    save_fig(fig, out)


def plot_pre_state_failure(X_prev: np.ndarray, true_labels: np.ndarray,
                            out: Path, focus_types: list[str] | None = None,
                            k: int = K_MAIN, seed: int = 42) -> dict[str, float]:
    """Cluster at k with previous features; highlight types with worst per-type purity.
    If focus_types is given, sort rows so those types appear first and annotate."""
    lab, Xs = kmeans_labels(X_prev, k, seed)
    tp = per_type_purity(lab, true_labels)
    ct = pd.crosstab(pd.Series(lab, name='Cluster'),
                     pd.Series(true_labels, name='Type'), normalize='columns')
    ct.columns = [str(c).replace('Ifc', '') for c in ct.columns]

    order = sorted(ct.columns, key=lambda t: tp.get('Ifc' + t, tp.get(t, 0)))
    ct = ct[order]

    fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * k)))
    sns.heatmap(ct, annot=True, fmt='.2f', cmap='Reds', ax=ax, annot_kws={'size': 6}, cbar=False)
    focus_disp = [t.replace('Ifc', '') for t in (focus_types or [])]
    for i, col in enumerate(ct.columns):
        if col in focus_disp:
            ax.add_patch(plt.Rectangle((i, -0.5), 1, ct.shape[0] + 0.5, fill=False,
                                        edgecolor='blue', lw=2, clip_on=False))
    pur = compute_purity(lab, true_labels)
    ax.set_title(f'Pre-state: cluster → type share (column-normalized). '
                 f'k={k}, overall purity={pur:.3f}', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    save_fig(fig, out)
    return tp


# ── summary.md writer ──────────────────────────────────────────────────────────
def write_summary(out_path: Path, *, stage_id: str, config_name: str,
                   deficiency: str, hypothesis: str,
                   sweep_prev: pd.DataFrame, sweep_new: pd.DataFrame,
                   delta_table: pd.DataFrame, notes: str = '') -> None:
    def line(df, metric):
        at24 = df[df['k'] == K_MAIN].iloc[0]
        return f"{at24[f'{metric}_mean']:.3f} ± {at24[f'{metric}_std']:.3f}"

    improved = delta_table.sort_values('delta', ascending=False).head(5)
    regressed = delta_table.sort_values('delta', ascending=True).head(5)
    n_types = delta_table.shape[0]

    md = f"""# {stage_id} — {config_name}

## Deficiency of previous feature set
{deficiency}

## Hypothesis
{hypothesis}

## Quantitative result (k={K_MAIN}, 10 seeds, {n_types} merged types)

| Metric | Previous | New ({config_name}) |
|---|---|---|
| Purity ↑ | {line(sweep_prev, 'purity')} | {line(sweep_new, 'purity')} |
| Silhouette ↑ | {line(sweep_prev, 'silhouette')} | {line(sweep_new, 'silhouette')} |
| Davies-Bouldin ↓ | {line(sweep_prev, 'davies_bouldin')} | {line(sweep_new, 'davies_bouldin')} |
| Calinski-Harabasz ↑ | {line(sweep_prev, 'calinski_harabasz')} | {line(sweep_new, 'calinski_harabasz')} |

## Per-type Δpurity (top 5 improved, top 5 regressed, seed=42)

**Improved**

| Type | Prev | New | Δ |
|---|---|---|---|
{chr(10).join(f"| {r['type']} | {r['prev']:.3f} | {r['new']:.3f} | {r['delta']:+.3f} |" for _, r in improved.iterrows())}

**Regressed**

| Type | Prev | New | Δ |
|---|---|---|---|
{chr(10).join(f"| {r['type']} | {r['prev']:.3f} | {r['new']:.3f} | {r['delta']:+.3f} |" for _, r in regressed.iterrows())}

## Notes
{notes}
"""
    out_path.write_text(md)


# ── Stage runner (used by 01..05) ──────────────────────────────────────────────
def run_stage(*, stage_id: str, config_name: str, new_feats: list[str],
              deficiency: str, hypothesis: str,
              focus_types: list[str] | None = None,
              notes: str = '') -> None:
    """
    Runs the full evidence pipeline for an incremental feature addition:
      - loads data
      - builds X_prev (features before this stage) and X_new (+ new_feats)
      - plots pre-state failure heatmap with focus_types highlighted
      - plots new-feature distributions (by merged type) and separation potential
      - k-sweeps both feature sets, saves k_sweep_metrics.{png,csv}
      - per-type Δpurity at k=K_MAIN
      - cluster heatmap at k=K_MAIN under new features
      - writes summary.md + appends to progression_summary.csv
    """
    out = ACTIVE_SPEC.results_root / stage_id
    out.mkdir(parents=True, exist_ok=True)

    df, ifc_merged = load_data()
    print(f'[{stage_id}] {len(df)} objects, {len(set(ifc_merged))} merged types '
          f'(spec={ACTIVE_SPEC.name})')

    prev_bounded = previous_bounded(stage_id)
    curr_bounded = cumulative_bounded_up_to(stage_id)

    X_prev = build_X(df, BASELINE, prev_bounded)
    X_new  = build_X(df, BASELINE, curr_bounded)

    # (1) Pre-state failure
    plot_pre_state_failure(X_prev, ifc_merged, out / 'pre_state_failure.png',
                           focus_types=focus_types)

    # (2) Distributions of the new feature(s)
    for feat in new_feats:
        plot_distribution_overall(df[feat].values, feat, out / f'distribution_overall_{feat}.png')
        plot_distribution_by_type(df, feat, ifc_merged, out / f'distribution_by_type_{feat}.png',
                                  title=f'{feat} by merged IfcType')

    # (3) Separation potential (ANOVA F-stat) for new feature(s)
    plot_separation_potential(df, new_feats, ifc_merged, out / 'separation_potential.png')

    # (4) k-sweeps
    sweep_prev = sweep_k(X_prev, ifc_merged)
    sweep_new  = sweep_k(X_new,  ifc_merged)
    sweep_prev.to_csv(out / 'k_sweep_prev.csv', index=False)
    sweep_new.to_csv (out / 'k_sweep_new.csv',  index=False)
    plot_k_sweep({'prev': sweep_prev, 'new': sweep_new}, out / 'k_sweep_metrics.png')
    append_progression(config_name, sweep_new)
    print(f'[{stage_id}] prev→new purity at k={K_MAIN}: '
          f'{sweep_prev[sweep_prev.k==K_MAIN].iloc[0]["purity_mean"]:.3f} → '
          f'{sweep_new[sweep_new.k==K_MAIN].iloc[0]["purity_mean"]:.3f}')

    # (5) Per-type Δpurity at K_MAIN and at k=24 (secondary reference)
    delta_main = plot_per_type_delta(X_prev, X_new, ifc_merged, K_MAIN,
                                     out / f'per_type_delta_k{K_MAIN}.png')
    delta_main.to_csv(out / f'per_type_delta_k{K_MAIN}.csv', index=False)
    if K_MAIN != 24:
        delta_24 = plot_per_type_delta(X_prev, X_new, ifc_merged, 24,
                                       out / 'per_type_delta_k24.png')
        delta_24.to_csv(out / 'per_type_delta_k24.csv', index=False)

    # (6) Cluster heatmap at K_MAIN with new features
    plot_cluster_heatmap(X_new, ifc_merged, K_MAIN, out / f'cluster_heatmap_k{K_MAIN}.png',
                         title=config_name)

    # (7) summary.md
    write_summary(out / 'summary.md',
                  stage_id=stage_id, config_name=config_name,
                  deficiency=deficiency, hypothesis=hypothesis,
                  sweep_prev=sweep_prev, sweep_new=sweep_new,
                  delta_table=delta_main, notes=notes)
    print(f'[{stage_id}] artifacts → {out}/')
