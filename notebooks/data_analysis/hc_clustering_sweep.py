"""
Handcrafted feature clustering sweep with cs_thinness.
Replicates feature_engineering.ipynb pipeline + adds cs_thinness,
then evaluates at k=2,4,8,16,24,32 with 10 seeds.
"""
import sys, json, warnings
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from scipy.stats import entropy
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings('ignore')

# ── Paths ──
PROJECT = Path(__file__).resolve().parent.parent.parent
df_all = pd.read_parquet(PROJECT / 'processed_data/features/train_features.parquet')
with open(PROJECT / 'data/annotated_subset.json') as f:
    annotated = json.load(f)
annotated_gids = [d['GlobalId'] for d in annotated]
df = df_all[df_all['GlobalId'].isin(set(annotated_gids))].copy().reset_index(drop=True)
mesh_dir = PROJECT / 'processed_data/meshes/train'

BASELINE = ['Length', 'CrossSectionArea', 'Volume', 'NumVertices']
Z_AXIS = np.array([0, 0, 1])
SEEDS = list(range(42, 52))
K_VALUES = [2, 4, 8, 16, 24, 32]

TYPE_MAP = {
    'IfcWall': 'Wall', 'IfcWallStandardCase': 'Wall',
    'IfcFurniture': 'Furniture', 'IfcFurnishingElement': 'Furniture',
    'IfcSystemFurnitureElement': 'Furniture',
    'IfcDistributionFlowElement': 'MEP', 'IfcFlowTerminal': 'MEP',
    'IfcStair': 'Stair', 'IfcStairFlight': 'Stair',
}

print(f'{len(df):,} objects, {df["IfcType"].nunique()} types\n')

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
    type_purity = {}
    for t in np.unique(true_labels):
        t_mask = true_labels == t
        t_objects = t_mask.sum()
        correct = sum(1 for i in np.where(t_mask)[0] if cluster_dominant[labels[i]] == t)
        type_purity[t] = correct / t_objects
    return type_purity

def eval_k(X, labels, k, seeds=SEEDS, scale=True):
    if scale:
        X = RobustScaler().fit_transform(X)
    metrics = {'purity': [], 'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': []}
    for s in seeds:
        lab = MiniBatchKMeans(k, batch_size=1024, n_init='auto', random_state=s).fit_predict(X)
        metrics['purity'].append(compute_purity(lab, labels))
        metrics['silhouette'].append(silhouette_score(X, lab))
        metrics['davies_bouldin'].append(davies_bouldin_score(X, lab))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(X, lab))
    return {m: (np.mean(v), np.std(v)) for m, v in metrics.items()}

# ── Feature extraction (same as feature_engineering.ipynb + cs_thinness) ──
print('Extracting features from meshes...')
ref_dirs = np.array(trimesh.creation.icosphere(subdivisions=1).vertices)
ref_dirs = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
N_BINS = len(ref_dirs)

engineered = []
for i, gid in enumerate(annotated_gids):
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

    normals = mesh.face_normals
    areas = mesh.area_faces
    if len(normals) > 0 and len(areas) > 0:
        dots = normals @ ref_dirs.T
        bins = dots.argmax(axis=1)
        hist = np.bincount(bins, weights=areas, minlength=N_BINS)
        hist = hist / hist.sum()
        norm_ent = entropy(hist) / np.log(N_BINS)
    else:
        norm_ent = 0.0

    if len(normals) > 0 and len(areas) > 0:
        z_component = np.abs(normals @ Z_AXIS)
        horiz_area = areas[z_component > 0.9].sum()
        horiz_frac = horiz_area / areas.sum()
    else:
        horiz_frac = 0.0

    engineered.append({
        'GlobalId': gid, 'pc1_z': pc1_z, 'pc3_z': pc3_z,
        'cs_aspect_ratio': cs_ratio, 'normal_entropy': norm_ent,
        'horiz_frac': horiz_frac,
    })
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(annotated_gids)}...')

df_eng = pd.DataFrame(engineered)
df_feat = df.merge(df_eng, on='GlobalId')
ifc_types_feat = df_feat['IfcType'].values
ifc_merged = np.array([TYPE_MAP.get(t, t) for t in ifc_types_feat])

# cs_thinness = log1p(1 / sqrt(CrossSectionArea))
df_feat['cs_thinness'] = np.log1p(1.0 / np.sqrt(df_feat['CrossSectionArea'].clip(lower=1e-8)))

print(f'Extracted features for {len(df_feat)} objects')
print(f'{len(set(ifc_types_feat))} raw types -> {len(set(ifc_merged))} merged types\n')

# ── Build feature configs ──
ENG_BOUNDED = ['pc1_z', 'pc3_z', 'cs_aspect_ratio']

def build_features(df, baseline_cols, bounded_eng=None):
    X = np.log1p(df[baseline_cols].clip(lower=0).values)
    if bounded_eng:
        X = np.hstack([X, df[bounded_eng].values])
    return X

configs = {
    'Baseline (4, log)':      build_features(df_feat, BASELINE),
    '+ Orientation (6)':      build_features(df_feat, BASELINE, bounded_eng=['pc1_z', 'pc3_z']),
    '+ CS Ratio (7)':         build_features(df_feat, BASELINE, bounded_eng=ENG_BOUNDED),
    '+ Normal Entropy (8)':   build_features(df_feat, BASELINE, bounded_eng=ENG_BOUNDED + ['normal_entropy']),
    '+ Horiz Frac (9)':       build_features(df_feat, BASELINE, bounded_eng=ENG_BOUNDED + ['normal_entropy', 'horiz_frac']),
    '+ CS Thinness (10)':     build_features(df_feat, BASELINE, bounded_eng=ENG_BOUNDED + ['normal_entropy', 'horiz_frac', 'cs_thinness']),
}

# ══════════════════════════════════════════════════════════════════════════════
# 1) Purity progression at each k (24 raw types)
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 90)
print('PURITY PROGRESSION — 24 RAW TYPES')
print('=' * 90)

for k in K_VALUES:
    header = f'{"Config":<25} {"Purity":>13} {"Silhouette":>13} {"DB (↓)":>13} {"CH (↑)":>13}'
    print(f'\n--- k={k} ---')
    print(header)
    print('-' * len(header))
    for name, X in configs.items():
        r = eval_k(X, ifc_types_feat, k)
        print(f'{name:<25} {r["purity"][0]:.3f}±{r["purity"][1]:.3f}'
              f' {r["silhouette"][0]:.3f}±{r["silhouette"][1]:.3f}'
              f' {r["davies_bouldin"][0]:.3f}±{r["davies_bouldin"][1]:.3f}'
              f' {r["calinski_harabasz"][0]:.0f}±{r["calinski_harabasz"][1]:.0f}')

# ══════════════════════════════════════════════════════════════════════════════
# 2) Purity progression at each k (19 merged types)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('PURITY PROGRESSION — 19 MERGED TYPES')
print('=' * 90)

for k in K_VALUES:
    header = f'{"Config":<25} {"Purity":>13} {"Silhouette":>13} {"DB (↓)":>13} {"CH (↑)":>13}'
    print(f'\n--- k={k} ---')
    print(header)
    print('-' * len(header))
    for name, X in configs.items():
        r = eval_k(X, ifc_merged, k)
        print(f'{name:<25} {r["purity"][0]:.3f}±{r["purity"][1]:.3f}'
              f' {r["silhouette"][0]:.3f}±{r["silhouette"][1]:.3f}'
              f' {r["davies_bouldin"][0]:.3f}±{r["davies_bouldin"][1]:.3f}'
              f' {r["calinski_harabasz"][0]:.0f}±{r["calinski_harabasz"][1]:.0f}')

# ══════════════════════════════════════════════════════════════════════════════
# 3) Per-type purity for 9-feat vs 10-feat at best k values
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('PER-TYPE PURITY: 9-feat vs 10-feat (+cs_thinness)')
print('=' * 90)

for label_name, labels in [('24 raw types', ifc_types_feat), ('19 merged types', ifc_merged)]:
    print(f'\n--- {label_name} ---')
    for k in [24, 32]:
        X9 = RobustScaler().fit_transform(configs['+ Horiz Frac (9)'])
        X10 = RobustScaler().fit_transform(configs['+ CS Thinness (10)'])
        lab9 = MiniBatchKMeans(k, batch_size=1024, n_init='auto', random_state=42).fit_predict(X9)
        lab10 = MiniBatchKMeans(k, batch_size=1024, n_init='auto', random_state=42).fit_predict(X10)
        tp9 = per_type_purity(lab9, labels)
        tp10 = per_type_purity(lab10, labels)

        all_types = sorted(set(labels))
        print(f'\n  k={k}:')
        print(f'  {"Type":<30} {"9-feat":>8} {"10-feat":>8} {"Delta":>8}')
        print(f'  {"-"*56}')
        for t in sorted(all_types, key=lambda t: tp10.get(t, 0) - tp9.get(t, 0), reverse=True):
            p9 = tp9.get(t, 0)
            p10 = tp10.get(t, 0)
            delta = p10 - p9
            marker = ' ***' if abs(delta) > 0.05 else ''
            print(f'  {t:<30} {p9:>8.3f} {p10:>8.3f} {delta:>+8.3f}{marker}')
        print(f'  {"OVERALL":<30} {compute_purity(lab9, labels):>8.3f} {compute_purity(lab10, labels):>8.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# 4) Summary: best purity per config across all k
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('SUMMARY — BEST PURITY PER CONFIG (19 merged types, 10 seeds)')
print('=' * 90)

header = f'{"Config":<25} ' + ' '.join(f'{"k="+str(k):>10}' for k in K_VALUES) + f' {"Best k":>8}'
print(header)
print('-' * len(header))
for name, X in configs.items():
    purities = []
    for k in K_VALUES:
        r = eval_k(X, ifc_merged, k)
        purities.append(r['purity'][0])
    best_k = K_VALUES[np.argmax(purities)]
    row = f'{name:<25} ' + ' '.join(f'{p:>10.3f}' for p in purities) + f' {best_k:>8}'
    print(row)

print('\nDone.')
