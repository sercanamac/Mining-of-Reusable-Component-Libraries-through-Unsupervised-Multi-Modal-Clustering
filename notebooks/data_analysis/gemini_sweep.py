"""
Gemini text embedding clustering sweep (single embeddings only).
PCA × k grid for each version, standalone and fused with 9 HC features.
Outputs numerical tables + plots to results/ folder.
"""
import sys, json, warnings
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
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ── Config ──
PROJECT = Path(__file__).resolve().parent.parent.parent
SEEDS = list(range(42, 52))
PCA_DIMS = [2, 4, 8, 16, 32]
K_VALUES = [4, 8, 16, 24, 32]
RAW_DIM = 768

GEMINI_VERSIONS = {
    'v1':     'gemini_embeddings',
    'v2':     'gemini_embeddings_v2_geometry_full',
    'v3':     'gemini_embeddings_v2__v3_thinness',
    'v4':     'gemini_embeddings_v2_v4_compact',
    'v5':     'gemini_embeddings_v2_v5_natural',
    'v6':     'gemini_embeddings_v2_v6_keywords',
    'v6.1':   'gemini_embeddings_v2_v6_no_material',
    'v7':     'gemini_embeddings_v2_v7_ifc_aware',
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

def balanced_fusion(X_a, X_b):
    """Variance-balanced concatenation."""
    var_a = np.var(X_a, axis=0).sum()
    var_b = np.var(X_b, axis=0).sum()
    return np.hstack([X_a / np.sqrt(var_a), X_b / np.sqrt(var_b)])

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

hc_map = {}
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
# Process each Gemini version
# ══════════════════════════════════════════════════════════════════════════════
all_summary = []
all_grids = {}  # version -> {variant_name: grid}
all_best = {}   # version -> {variant_name: (d, k, purity)}

for ver_name, folder_name in GEMINI_VERSIONS.items():
    print(f'\n{"=" * 80}')
    print(f'GEMINI {ver_name} ({folder_name})')
    print(f'{"=" * 80}')

    embed_dir = PROJECT / 'processed_data' / folder_name / 'gemini_embed_single'
    if not embed_dir.exists():
        print(f'  SKIP: {embed_dir} not found')
        continue

    X_embed, gids_embed = load_embeddings(embed_dir, df['GlobalId'].values)
    gids_valid = sorted(set(gids_embed) & set(hc_map.keys()))

    if len(gids_valid) < 100:
        print(f'  SKIP: only {len(gids_valid)} valid objects')
        continue

    idx_embed = {g: i for i, g in enumerate(gids_embed)}
    df_valid = df[df['GlobalId'].isin(gids_valid)].copy()
    df_valid = df_valid.set_index('GlobalId').loc[gids_valid].reset_index()

    X_emb = np.stack([X_embed[idx_embed[g]] for g in gids_valid])
    hc_eng = pd.DataFrame([hc_map[g] for g in gids_valid])
    X_hc = np.hstack([
        np.log1p(df_valid[BASELINE].clip(lower=0).values),
        hc_eng[['pc1_z', 'pc3_z', 'cs_aspect_ratio', 'normal_entropy', 'horiz_frac']].values
    ])
    ifc_types = np.array([TYPE_MAP.get(t, t) for t in df_valid['IfcType'].values])
    print(f'  {len(gids_valid)} objects, {len(set(ifc_types))} merged types, embed shape={X_emb.shape}')

    # PCA projections
    all_dims = PCA_DIMS + [RAW_DIM]
    pca_embed = {}
    for d in PCA_DIMS:
        pca_embed[d] = PCA(n_components=d, random_state=42).fit_transform(X_emb)
    pca_embed[RAW_DIM] = X_emb

    # Grid sweep: text-only and text+HC (balanced fusion)
    variants = {
        'Text only': (False,),
        'Text+HC (balanced)': (True,),
    }

    grids = {}
    for vname, (fused,) in variants.items():
        grid = np.zeros((len(all_dims), len(K_VALUES)))
        for i, d in enumerate(all_dims):
            X_feat = balanced_fusion(pca_embed[d], X_hc) if fused else pca_embed[d]
            for j, k in enumerate(K_VALUES):
                mean_p, _ = eval_kmeans(X_feat, ifc_types, k)
                grid[i, j] = mean_p
        grids[vname] = grid

    # HC baseline
    hc_purities = {}
    for k in K_VALUES:
        hc_purities[k], _ = eval_kmeans(X_hc, ifc_types, k)

    all_grids[ver_name] = grids

    # Print results
    dim_labels = [str(d) for d in all_dims]
    for vname, grid in grids.items():
        print(f'\n  {vname}:')
        header = f'  {"PCA":>6} ' + ' '.join(f'{"k="+str(k):>7}' for k in K_VALUES)
        print(header)
        print(f'  {"-" * len(header)}')
        for i, d in enumerate(all_dims):
            row = f'  {d:>6} ' + ' '.join(f'{grid[i, j]:>7.3f}' for j in range(len(K_VALUES)))
            print(row)
    print(f'\n  HC baseline: ' + ' '.join(f'k={k}:{hc_purities[k]:.3f}' for k in K_VALUES))

    # Best configs
    best_ver = {}
    for vname, grid in grids.items():
        idx = np.unravel_index(grid.argmax(), grid.shape)
        best_d, best_k = all_dims[idx[0]], K_VALUES[idx[1]]
        best_ver[vname] = (best_d, best_k, grid[idx])
        all_summary.append({
            'Version': ver_name, 'Variant': vname,
            'Best PCA': best_d, 'Best k': best_k, 'Purity': grid[idx]
        })
        print(f'  Best {vname}: PCA-{best_d}, k={best_k} -> {grid[idx]:.3f}')
    all_best[ver_name] = best_ver

    # ── Heatmap: 2 panels (text-only vs text+HC) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (vname, grid) in zip(axes, grids.items()):
        sns.heatmap(grid, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[str(k) for k in K_VALUES],
                    yticklabels=dim_labels, ax=ax, vmin=0.15, vmax=0.72,
                    annot_kws={'size': 10})
        ax.set_xlabel('k')
        ax.set_ylabel('PCA dims')
        ax.set_title(vname, fontsize=12)
    fig.suptitle(f'Gemini {ver_name} — Purity Grid (19 merged types, 10 seeds)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'gemini_{ver_name.replace(".", "")}_purity_grids.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Per-type delta: text+HC vs HC-only ──
    best_d, best_k, _ = best_ver['Text+HC (balanced)']
    X_best = balanced_fusion(pca_embed[best_d], X_hc)
    X_best_sc = RobustScaler().fit_transform(X_best)
    X_hc_sc = RobustScaler().fit_transform(X_hc)

    lab_fused = MiniBatchKMeans(best_k, batch_size=1024, n_init='auto', random_state=42).fit_predict(X_best_sc)
    lab_hc = MiniBatchKMeans(best_k, batch_size=1024, n_init='auto', random_state=42).fit_predict(X_hc_sc)

    tp_fused = per_type_purity(lab_fused, ifc_types)
    tp_hc = per_type_purity(lab_hc, ifc_types)
    all_types = sorted(set(ifc_types))

    df_delta = pd.DataFrame({
        'HC only': [tp_hc.get(t, 0) for t in all_types],
        f'{ver_name}+HC': [tp_fused.get(t, 0) for t in all_types],
    }, index=all_types)
    df_delta['Delta'] = df_delta[f'{ver_name}+HC'] - df_delta['HC only']
    df_delta = df_delta.sort_values('Delta', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['#e74c3c' if d < -0.02 else '#2ecc71' if d > 0.02 else '#95a5a6'
              for d in df_delta['Delta']]
    ax.barh(range(len(df_delta)), df_delta['Delta'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(df_delta)))
    ax.set_yticklabels(df_delta.index, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Purity Delta (Fusion - HC only)')
    ax.set_title(f'Gemini {ver_name}+HC (PCA-{best_d}, k={best_k}) vs HC only\n'
                 f'HC: {compute_purity(lab_hc, ifc_types):.3f} -> '
                 f'Fusion: {compute_purity(lab_fused, ifc_types):.3f}', fontsize=12)
    for i, (_, row) in enumerate(df_delta.iterrows()):
        d = row['Delta']
        ax.text(d + (0.01 if d >= 0 else -0.01), i,
                f'{d:+.3f}', va='center', ha='left' if d >= 0 else 'right', fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'gemini_{ver_name.replace(".", "")}_pertype_delta.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'  Saved plots for {ver_name}')

# ══════════════════════════════════════════════════════════════════════════════
# Cross-version summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('GEMINI CROSS-VERSION SUMMARY')
print('=' * 80)
df_summary = pd.DataFrame(all_summary)
print(df_summary.to_string(index=False))
df_summary.to_csv(OUT_DIR / 'gemini_summary.csv', index=False)

# ── Cross-version bar chart: HC vs Text vs Text+HC ──
versions = list(all_best.keys())
hc_val = hc_purities[32]  # HC baseline at k=32

text_purities = []
fused_purities = []
for v in versions:
    text_purities.append(all_best[v]['Text only'][2])
    fused_purities.append(all_best[v]['Text+HC (balanced)'][2])

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(versions))
w = 0.25

bars_hc = ax.bar(x - w, [hc_val] * len(versions), w, label='HC only (9 feat)',
                 color='#95a5a6', edgecolor='white')
bars_text = ax.bar(x, text_purities, w, label='Gemini text only',
                   color='#e67e22', edgecolor='white')
bars_fused = ax.bar(x + w, fused_purities, w, label='Gemini text + HC (balanced)',
                    color='#2980b9', edgecolor='white')

for bars in [bars_hc, bars_text, bars_fused]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=10)
ax.set_ylabel('Purity', fontsize=12)
ax.set_title('Gemini Versions: HC vs Text vs Fusion (best config per version, 19 merged types)', fontsize=13)
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(0.45, max(fused_purities) * 1.08)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / 'gemini_cross_version.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved gemini_cross_version.png')

# ── Version progression line plot ──
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(versions, text_purities, 'o-', color='#e67e22', linewidth=2, markersize=8, label='Text only')
ax.plot(versions, fused_purities, 's-', color='#2980b9', linewidth=2, markersize=8, label='Text + HC (balanced)')
ax.axhline(y=hc_val, color='#95a5a6', linestyle='--', linewidth=2, label=f'HC only ({hc_val:.3f})')
for i, v in enumerate(versions):
    ax.annotate(f'{text_purities[i]:.3f}', (i, text_purities[i]), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8, color='#e67e22')
    ax.annotate(f'{fused_purities[i]:.3f}', (i, fused_purities[i]), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8, color='#2980b9')
ax.set_xlabel('Gemini Prompt Version', fontsize=12)
ax.set_ylabel('Purity', fontsize=12)
ax.set_title('Prompt Evolution: Purity Progression (19 merged types, 10 seeds)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / 'gemini_progression.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved gemini_progression.png')

print(f'\nDone. All outputs in: {OUT_DIR}')
