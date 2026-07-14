"""
Image embedding clustering sweep: DINOv3, DuoDuo, SigLIP.
For each model: colored/colorless × PCA dims × k values.
Also: fusion with 9 HC features.
Outputs numerical tables + per-type delta plots to results/ folder.
"""
import sys, json, warnings, os
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import trimesh
from pathlib import Path
from scipy.stats import entropy
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings('ignore')

# ── Config ──
PROJECT = Path(__file__).resolve().parent.parent.parent
SEEDS = list(range(42, 52))
PCA_DIMS = [2, 4, 8, 16, 32]
K_VALUES = [4, 8, 16, 24, 32]
MODELS = {
    'DuoDuo':  ('duoduo', 512),
    'DINOv3':  ('dinov3', 1024),
    'SigLIP':  ('siglip2_large_16_512', 1024),
}
TYPE_MAP = {
    'IfcWall': 'Wall', 'IfcWallStandardCase': 'Wall',
    'IfcFurniture': 'Furniture', 'IfcFurnishingElement': 'Furniture',
    'IfcSystemFurnitureElement': 'Furniture',
    'IfcDistributionFlowElement': 'MEP', 'IfcFlowTerminal': 'MEP',
    'IfcStair': 'Stair', 'IfcStairFlight': 'Stair',
}
BASELINE = ['Length', 'CrossSectionArea', 'Volume', 'NumVertices']
Z_AXIS = np.array([0, 0, 1])

OUT_DIR = PROJECT / 'notebooks' / 'data_analysis' / 'results'
OUT_DIR.mkdir(exist_ok=True)

# ── Helpers ──
def compute_purity(labels, true_labels):
    return sum(pd.Series(true_labels[labels == c]).value_counts().iloc[0]
               for c in np.unique(labels)) / len(labels)

def per_type_purity(labels, true_labels):
    cluster_dominant = {}
    for c in np.unique(labels):
        mask = labels == c
        counts = pd.Series(true_labels[mask]).value_counts()
        cluster_dominant[c] = counts.index[0]
    tp = {}
    for t in np.unique(true_labels):
        t_mask = true_labels == t
        t_objects = t_mask.sum()
        correct = sum(1 for i in np.where(t_mask)[0] if cluster_dominant[labels[i]] == t)
        tp[t] = correct / t_objects
    return tp

def eval_kmeans(X, labels, k, seeds=SEEDS):
    X_sc = RobustScaler().fit_transform(X)
    purities = []
    for s in seeds:
        lab = MiniBatchKMeans(k, batch_size=1024, n_init='auto', random_state=s).fit_predict(X_sc)
        purities.append(compute_purity(lab, labels))
    return np.mean(purities), np.std(purities)

def load_embeddings(feat_dir, gids):
    features, matched_gids = [], []
    for gid in gids:
        path = feat_dir / f'{gid}.npy'
        if path.exists():
            feat = np.load(path)
            if feat.ndim == 2:
                feat = feat.squeeze(0)
            features.append(feat)
            matched_gids.append(gid)
    return np.stack(features), matched_gids

# ── Load base data ──
print('Loading data...')
df_all = pd.read_parquet(PROJECT / 'processed_data/features/train_features.parquet')
with open(PROJECT / 'data/annotated_subset.json') as f:
    annotated = json.load(f)
annotated_gids = set(d['GlobalId'] for d in annotated)
df = df_all[df_all['GlobalId'].isin(annotated_gids)].copy().reset_index(drop=True)
mesh_dir = PROJECT / 'processed_data/meshes/train'

# ── Compute HC features once ──
print('Computing handcrafted features...')
ref_dirs = np.array(trimesh.creation.icosphere(subdivisions=1).vertices)
ref_dirs = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
N_BINS = len(ref_dirs)

hc_map = {}  # gid -> feature dict
for i, gid in enumerate(df['GlobalId'].values):
    mesh_path = mesh_dir / f'{gid}.obj'
    if not mesh_path.exists():
        continue
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    verts = np.array(mesh.vertices)
    if len(verts) < 3:
        continue
    pca = PCA(n_components=3).fit(verts)
    pc1_z = abs(np.dot(pca.components_[0], Z_AXIS))
    pc3_z = abs(np.dot(pca.components_[2], Z_AXIS))
    extents = np.sqrt(pca.explained_variance_)
    cs_ratio = extents[2] / extents[1] if extents[1] > 1e-8 else 0.0
    normals, areas = mesh.face_normals, mesh.area_faces
    if len(normals) > 0 and len(areas) > 0:
        dots = normals @ ref_dirs.T
        hist = np.bincount(dots.argmax(axis=1), weights=areas, minlength=N_BINS)
        hist = hist / hist.sum()
        norm_ent = entropy(hist) / np.log(N_BINS)
        horiz_frac = areas[np.abs(normals @ Z_AXIS) > 0.9].sum() / areas.sum()
    else:
        norm_ent, horiz_frac = 0.0, 0.0
    hc_map[gid] = {'pc1_z': pc1_z, 'pc3_z': pc3_z, 'cs_aspect_ratio': cs_ratio,
                    'normal_entropy': norm_ent, 'horiz_frac': horiz_frac}
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(df)}...')
print(f'  HC features for {len(hc_map)} objects')

# ══════════════════════════════════════════════════════════════════════════════
# Process each model
# ══════════════════════════════════════════════════════════════════════════════
all_summary = []  # for final summary table

for model_name, (model_dir, raw_dim) in MODELS.items():
    print(f'\n{"=" * 80}')
    print(f'MODEL: {model_name} ({model_dir}, {raw_dim}-dim)')
    print(f'{"=" * 80}')

    feat_root = PROJECT / 'processed_data/rendered_features' / model_dir

    # Load colored + colorless
    X_col, gids_col = load_embeddings(feat_root / 'colored', df['GlobalId'].values)
    X_cl, gids_cl = load_embeddings(feat_root / 'colorless', df['GlobalId'].values)

    # Intersect colored, colorless, and HC
    gids_valid = sorted(set(gids_col) & set(gids_cl) & set(hc_map.keys()))
    idx_col = {g: i for i, g in enumerate(gids_col)}
    idx_cl = {g: i for i, g in enumerate(gids_cl)}

    df_valid = df[df['GlobalId'].isin(gids_valid)].copy()
    df_valid = df_valid.set_index('GlobalId').loc[gids_valid].reset_index()

    X_colored = np.stack([X_col[idx_col[g]] for g in gids_valid])
    X_colorless = np.stack([X_cl[idx_cl[g]] for g in gids_valid])

    # Build HC matrix aligned to gids_valid
    hc_eng = pd.DataFrame([hc_map[g] for g in gids_valid])
    X_hc = np.hstack([
        np.log1p(df_valid[BASELINE].clip(lower=0).values),
        hc_eng[['pc1_z', 'pc3_z', 'cs_aspect_ratio', 'normal_entropy', 'horiz_frac']].values
    ])

    ifc_types = np.array([TYPE_MAP.get(t, t) for t in df_valid['IfcType'].values])
    print(f'  {len(gids_valid)} objects, {len(set(ifc_types))} merged types')
    print(f'  Colored: {X_colored.shape}, Colorless: {X_colorless.shape}, HC: {X_hc.shape}')

    # PCA projections
    all_dims = PCA_DIMS + [raw_dim]
    pca_colored, pca_colorless = {}, {}
    for d in PCA_DIMS:
        pca_colored[d] = PCA(n_components=d, random_state=42).fit_transform(X_colored)
        pca_colorless[d] = PCA(n_components=d, random_state=42).fit_transform(X_colorless)
    pca_colored[raw_dim] = X_colored
    pca_colorless[raw_dim] = X_colorless

    # ── Grid sweep: 4 variants × PCA × k ──
    variants = {
        'Colored':        (pca_colored, False),
        'Colorless':      (pca_colorless, False),
        'Colored+HC':     (pca_colored, True),
        'Colorless+HC':   (pca_colorless, True),
    }

    grids = {}
    for vname, (pca_dict, fused) in variants.items():
        grid = np.zeros((len(all_dims), len(K_VALUES)))
        for i, d in enumerate(all_dims):
            X_feat = np.hstack([pca_dict[d], X_hc]) if fused else pca_dict[d]
            for j, k in enumerate(K_VALUES):
                mean_p, _ = eval_kmeans(X_feat, ifc_types, k)
                grid[i, j] = mean_p
        grids[vname] = grid
        print(f'  {vname} grid done')

    # HC baseline
    hc_purities = {}
    for k in K_VALUES:
        hc_purities[k], _ = eval_kmeans(X_hc, ifc_types, k)

    # ── Print numerical results ──
    dim_labels = [str(d) for d in all_dims]
    print(f'\n--- {model_name}: Purity Grid (10 seeds) ---')
    for vname, grid in grids.items():
        print(f'\n  {vname}:')
        header = f'  {"PCA":>6} ' + ' '.join(f'{"k="+str(k):>7}' for k in K_VALUES)
        print(header)
        print(f'  {"-" * len(header)}')
        for i, d in enumerate(all_dims):
            row = f'  {d:>6} ' + ' '.join(f'{grid[i, j]:>7.3f}' for j in range(len(K_VALUES)))
            print(row)
    print(f'\n  HC baseline: ' + ' '.join(f'k={k}:{hc_purities[k]:.3f}' for k in K_VALUES))

    # Find best configs
    best_per_variant = {}
    for vname, grid in grids.items():
        idx = np.unravel_index(grid.argmax(), grid.shape)
        best_d, best_k = all_dims[idx[0]], K_VALUES[idx[1]]
        best_per_variant[vname] = (best_d, best_k, grid[idx])
        all_summary.append({
            'Model': model_name, 'Variant': vname,
            'Best PCA': best_d, 'Best k': best_k, 'Purity': grid[idx]
        })
        print(f'  Best {vname}: PCA-{best_d}, k={best_k} → {grid[idx]:.3f}')
    all_summary.append({
        'Model': model_name, 'Variant': 'HC only',
        'Best PCA': 9, 'Best k': max(hc_purities, key=hc_purities.get),
        'Purity': max(hc_purities.values())
    })

    # ── Heatmap plots: 4 panels ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (vname, grid) in zip(axes.flat, grids.items()):
        sns.heatmap(grid, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[str(k) for k in K_VALUES],
                    yticklabels=dim_labels, ax=ax, vmin=0.15, vmax=0.70,
                    annot_kws={'size': 10})
        ax.set_xlabel('k')
        ax.set_ylabel('PCA dims')
        ax.set_title(vname, fontsize=12)
    fig.suptitle(f'{model_name} — Purity Grid (19 merged types, 10 seeds)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'{model_name.lower()}_purity_grids.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {model_name.lower()}_purity_grids.png')

    # ── Per-type purity delta: best fusion vs HC-only (at best k) ──
    # Use best Colored+HC config
    best_d, best_k, best_p = best_per_variant['Colored+HC']
    X_best = np.hstack([pca_colored[best_d], X_hc])
    X_best_sc = RobustScaler().fit_transform(X_best)
    X_hc_sc = RobustScaler().fit_transform(X_hc)

    lab_fused = MiniBatchKMeans(best_k, batch_size=1024, n_init='auto', random_state=42).fit_predict(X_best_sc)
    lab_hc = MiniBatchKMeans(best_k, batch_size=1024, n_init='auto', random_state=42).fit_predict(X_hc_sc)

    tp_fused = per_type_purity(lab_fused, ifc_types)
    tp_hc = per_type_purity(lab_hc, ifc_types)

    all_types = sorted(set(ifc_types))
    df_delta = pd.DataFrame({
        'HC only': [tp_hc.get(t, 0) for t in all_types],
        f'{model_name}+HC': [tp_fused.get(t, 0) for t in all_types],
    }, index=all_types)
    df_delta['Delta'] = df_delta[f'{model_name}+HC'] - df_delta['HC only']
    df_delta = df_delta.sort_values('Delta', ascending=True)

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['#e74c3c' if d < -0.02 else '#2ecc71' if d > 0.02 else '#95a5a6'
              for d in df_delta['Delta']]
    bars = ax.barh(range(len(df_delta)), df_delta['Delta'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(df_delta)))
    ax.set_yticklabels(df_delta.index, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Purity Delta (Fusion − HC only)')
    ax.set_title(f'{model_name}+HC (PCA-{best_d}, k={best_k}) vs HC only\n'
                 f'HC: {compute_purity(lab_hc, ifc_types):.3f} → '
                 f'Fusion: {compute_purity(lab_fused, ifc_types):.3f}', fontsize=12)
    for i, (_, row) in enumerate(df_delta.iterrows()):
        d = row['Delta']
        ax.text(d + (0.01 if d >= 0 else -0.01), i,
                f'{d:+.3f}', va='center', ha='left' if d >= 0 else 'right', fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'{model_name.lower()}_pertype_delta.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {model_name.lower()}_pertype_delta.png')

    # ── Per-type absolute purity heatmap: all variants at best k ──
    tp_all = {}
    tp_all['HC only'] = tp_hc
    for vname, (pca_dict, fused) in variants.items():
        bd, bk, _ = best_per_variant[vname]
        X_v = np.hstack([pca_dict[bd], X_hc]) if fused else pca_dict[bd]
        X_v_sc = RobustScaler().fit_transform(X_v)
        lab_v = MiniBatchKMeans(bk, batch_size=1024, n_init='auto', random_state=42).fit_predict(X_v_sc)
        tp_all[f'{vname} (d={bd},k={bk})'] = per_type_purity(lab_v, ifc_types)

    df_tp = pd.DataFrame({n: [tp_all[n].get(t, 0) for t in all_types] for n in tp_all},
                         index=all_types)
    df_tp = df_tp.sort_values('HC only', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_tp, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, vmin=0, vmax=1,
                annot_kws={'size': 8})
    ax.set_title(f'{model_name} — Per-Type Purity (all variants at their best config)', fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'{model_name.lower()}_pertype_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {model_name.lower()}_pertype_heatmap.png')

# ══════════════════════════════════════════════════════════════════════════════
# Cross-model summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('CROSS-MODEL SUMMARY')
print('=' * 80)
df_summary = pd.DataFrame(all_summary)
print(df_summary.to_string(index=False))

# Save summary CSV
df_summary.to_csv(OUT_DIR / 'summary.csv', index=False)
print(f'\nSaved summary.csv to {OUT_DIR}')

# Cross-model comparison: grouped bar chart (HC vs Colored vs Colored+HC)
fusion_only = df_summary[df_summary['Variant'].isin(['Colored', 'Colored+HC'])].copy()
models = fusion_only['Model'].unique()
hc_purity = df_summary[df_summary['Variant'] == 'HC only']['Purity'].iloc[0]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
w = 0.25

# HC baseline bar (same for all)
bars_hc = ax.bar(x - w, [hc_purity] * len(models), w, label='HC only (9 feat)',
                 color='#95a5a6', edgecolor='white')
# Colored-only bars
col_purities = [fusion_only[(fusion_only['Model'] == m) & (fusion_only['Variant'] == 'Colored')]['Purity'].iloc[0]
                for m in models]
bars_col = ax.bar(x, col_purities, w, label='Embedding only (colored)',
                  color='#e67e22', edgecolor='white')
# Colored+HC fusion bars
fused_purities = [fusion_only[(fusion_only['Model'] == m) & (fusion_only['Variant'] == 'Colored+HC')]['Purity'].iloc[0]
                  for m in models]
bars_fused = ax.bar(x + w, fused_purities, w, label='Embedding + HC (colored)',
                    color='#2980b9', edgecolor='white')

# Labels on bars
for bars in [bars_hc, bars_col, bars_fused]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Purity', fontsize=12)
ax.set_title('HC vs Embedding vs Fusion (best config per model, 19 merged types)', fontsize=13)
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(0.55, max(fused_purities) * 1.08)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / 'cross_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved cross_model_comparison.png')

print('\nDone. All outputs in:', OUT_DIR)
