"""Fill experiment gaps #1, #2, #6, #8, #9, #10 in one run.

1. F1a_umap on geo+text, geo+visual (was only geo+text+visual)
2. Fusion with different visual encoders (siglip_colored, dinov3, duoduo, gemini)
6. F1a_umap with all algos (kmeans/gmm/bisecting), not just HDBSCAN
8. Cross-encoder fusion (= covered by #2)
9. K-matched comparison on new winner
10. Bootstrap stability on new winner
"""
import os, tempfile
os.environ['NUMBA_CACHE_DIR'] = tempfile.mkdtemp()

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'feature_engineering'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual
from _reducers import ReducerConfig, UMAP_PARAMS_TEXTVIS, fit_reducer
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'fusion'
STAB_DIR = RESULTS_ROOT.parent / 'midterm' / 'stability'

TEXT_VERSION = 'v7'
TEXT_AGG = 'single'

ALGOS_ALL = ['kmeans', 'hdbscan', 'gmm', 'bisecting']
ALGOS_HDB = ['hdbscan']


def _std(X):
    return StandardScaler().fit_transform(X).astype(np.float32)


def _umap_block(X, k, name):
    cfg = ReducerConfig(f'umap_{k}', 'umap', k, dict(UMAP_PARAMS_TEXTVIS))
    Xr, meta = fit_reducer(cfg, X, modality_name=name)
    return Xr


def _pca_block(X, k):
    return PCA(n_components=k, random_state=42).fit_transform(X).astype(np.float32)


def _force_assign(lab, X):
    lab = lab.copy()
    noise = lab == -1
    if noise.sum() == 0 or (~noise).sum() == 0:
        return lab
    cids = sorted(set(lab[~noise].tolist()))
    centroids = np.stack([X[lab == c].mean(0) for c in cids])
    d = np.linalg.norm(X[noise][:, None, :] - centroids[None, :, :], axis=2)
    lab[noise] = np.array(cids)[d.argmin(1)]
    return lab


# ── Load data ────────────────────────────────────────────────────────────
df, y_full = load_data()
gids = df['GlobalId'].values
geo_all = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

X_txt_raw, txt_idx = load_text(gids, TEXT_VERSION, TEXT_AGG)

# Load all visual encoders
VISUAL_CONFIGS = [
    ('siglip', 'colored'),
    ('siglip', 'colorless'),
    ('dinov3', 'colored'),
    ('dinov3', 'colorless'),
    ('duoduo', 'colored'),
    ('duoduo', 'colorless'),
    ('gemini', 'colored'),
]

vis_cache = {}
for enc, var in VISUAL_CONFIGS:
    try:
        X_v, v_idx = load_visual(gids, enc, var)
        vis_cache[(enc, var)] = (X_v, v_idx)
        print(f'  loaded {enc}/{var}: {X_v.shape}')
    except RuntimeError as e:
        print(f'  skip {enc}/{var}: {e}')

all_summary, all_per_type = [], []


# ══════════════════════════════════════════════════════════════════════════
# PART A: Gaps #1, #6 — F1a_umap on all subsets + all algos
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART A: F1a_umap on all subsets, all algos')
print('='*60)

# Use siglip_colorless (canonical) for the subset ablation
X_vis_raw, vis_idx = vis_cache[('siglip', 'colorless')]
common = np.intersect1d(txt_idx, vis_idx)
geo_b = geo_all[common]
txt_b = X_txt_raw[np.searchsorted(txt_idx, common)]
vis_b = X_vis_raw[np.searchsorted(vis_idx, common)]
y = y_full[common]

SUBSETS_A = {
    'geo+text': lambda k: np.concatenate([_std(geo_b), _std(_umap_block(txt_b, k, 'text_v7'))], axis=1),
    'geo+visual': lambda k: np.concatenate([_std(geo_b), _std(_umap_block(vis_b, k, 'visual_siglip_colorless'))], axis=1),
    'geo+text+visual': lambda k: np.concatenate([
        _std(geo_b),
        _std(_umap_block(txt_b, k, 'text_v7')),
        _std(_umap_block(vis_b, k, 'visual_siglip_colorless')),
    ], axis=1),
}

for subset_label, build_fn in SUBSETS_A.items():
    for k in [8, 16]:
        X = build_fn(k)
        # All algos for gap #6
        algos = ALGOS_ALL
        extra = {
            'subset': subset_label, 'recipe': 'F1a_umap',
            'variant': f'umap_{k}', 'reducer_dim': k,
            'out_dim': int(X.shape[1]),
            'text_version': TEXT_VERSION,
            'visual_encoder': 'siglip_colorless',
        }
        print(f'\n  {subset_label} F1a_umap-{k} dim={X.shape[1]} ALL_ALGOS')
        s, pt = run_sweep(
            name=f'{subset_label}__F1a_umap_{k}',
            X=X, y=y, algorithms=algos, scale=True,
            hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
            extra_meta=extra, verbose=False,
        )
        best_h = s[s['algo'] == 'hdbscan'].sort_values('macro_purity', ascending=False).iloc[0]
        print(f'    HDBSCAN best: macro={best_h["macro_purity"]:.4f} k={int(best_h["k"])}')
        all_summary.append(s)
        all_per_type.append(pt)


# ══════════════════════════════════════════════════════════════════════════
# PART B: Gaps #2, #8 — Cross-encoder fusion (F1a_umap-16 with each visual encoder)
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART B: Cross-encoder fusion (F1a_umap-16, geo+text+visual)')
print('='*60)

K_UMAP = 16  # winning dim

for (enc, var), (X_v, v_idx) in vis_cache.items():
    if enc == 'siglip' and var == 'colorless':
        continue  # already done in Part A
    common_ev = np.intersect1d(txt_idx, v_idx)
    geo_ev = geo_all[common_ev]
    txt_ev = X_txt_raw[np.searchsorted(txt_idx, common_ev)]
    vis_ev = X_v[np.searchsorted(v_idx, common_ev)]
    y_ev = y_full[common_ev]

    X = np.concatenate([
        _std(geo_ev),
        _std(_umap_block(txt_ev, K_UMAP, f'text_v7')),
        _std(_umap_block(vis_ev, K_UMAP, f'visual_{enc}_{var}')),
    ], axis=1)

    extra = {
        'subset': 'geo+text+visual', 'recipe': 'F1a_umap',
        'variant': f'umap_{K_UMAP}', 'reducer_dim': K_UMAP,
        'out_dim': int(X.shape[1]),
        'text_version': TEXT_VERSION,
        'visual_encoder': f'{enc}_{var}',
    }
    print(f'\n  {enc}/{var} F1a_umap-{K_UMAP} dim={X.shape[1]} n={len(common_ev)}')
    s, pt = run_sweep(
        name=f'geo+text+visual__F1a_umap_{K_UMAP}__{enc}_{var}',
        X=X, y=y_ev, algorithms=ALGOS_HDB, scale=True,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        extra_meta=extra, verbose=False,
    )
    best_h = s[s['algo'] == 'hdbscan'].sort_values('macro_purity', ascending=False).iloc[0]
    print(f'    best: macro={best_h["macro_purity"]:.4f} purity={best_h["purity"]:.4f} k={int(best_h["k"])}')
    all_summary.append(s)
    all_per_type.append(pt)


# ══════════════════════════════════════════════════════════════════════════
# Merge all into fusion.csv
# ══════════════════════════════════════════════════════════════════════════
new_summary = pd.concat(all_summary, ignore_index=True)
new_per_type = pd.concat(all_per_type, ignore_index=True)

existing = pd.read_csv(OUT_DIR / 'fusion.csv')
existing_pt = pd.read_csv(OUT_DIR / 'fusion_per_type.csv')

# Remove old F1a_umap rows (from previous partial run) to avoid duplicates
existing = existing[existing['recipe'] != 'F1a_umap']
existing_pt = existing_pt[existing_pt['recipe'] != 'F1a_umap']

offset = int(existing['run_id'].max()) + 1
new_summary['run_id'] += offset
new_per_type['run_id'] += offset

merged = pd.concat([existing, new_summary], ignore_index=True)
merged_pt = pd.concat([existing_pt, new_per_type], ignore_index=True)
merged.to_csv(OUT_DIR / 'fusion.csv', index=False)
merged_pt.to_csv(OUT_DIR / 'fusion_per_type.csv', index=False)
print(f'\nMerged fusion.csv: {len(merged)} rows (was {len(existing)})')


# ══════════════════════════════════════════════════════════════════════════
# PART C: Gap #9 — K-matched comparison on F1a_umap-16 winner
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART C: K-matched comparison on F1a_umap-16')
print('='*60)

# Find the global HDBSCAN winner from the new results
all_hdb = merged[(merged['algo'] == 'hdbscan')]
winner = all_hdb.loc[all_hdb['macro_purity'].idxmax()]
w_enc = winner.get('visual_encoder', 'siglip_colorless')
w_dim = int(winner['reducer_dim']) if pd.notna(winner.get('reducer_dim')) else 16
w_recipe = winner['recipe']
w_k = int(winner['k'])
print(f'  Global winner: {w_recipe} dim={w_dim} enc={w_enc} macro={winner["macro_purity"]:.4f} k={w_k}')

# Rebuild X for the winner
enc_name, var_name = w_enc.rsplit('_', 1)
X_v_w, v_idx_w = vis_cache[(enc_name, var_name)]
common_w = np.intersect1d(txt_idx, v_idx_w)
geo_w = geo_all[common_w]
txt_w = X_txt_raw[np.searchsorted(txt_idx, common_w)]
vis_w = X_v_w[np.searchsorted(v_idx_w, common_w)]
y_w = y_full[common_w]

X_winner = np.concatenate([
    _std(geo_w),
    _std(_umap_block(txt_w, w_dim, 'text_v7')),
    _std(_umap_block(vis_w, w_dim, f'visual_{enc_name}_{var_name}')),
], axis=1)
X_winner = StandardScaler().fit_transform(X_winner)

# HDBSCAN winner hps
from _sweep_runner import HDBSCAN_GRID_FACTORIAL
import json
hps = json.loads(winner['hps'])
print(f'  HDBSCAN hps: {hps}')

# HDBSCAN force=True
lab_hdb = HDBSCAN(
    min_cluster_size=int(hps['mcs']), min_samples=int(hps['ms']),
    cluster_selection_method=hps['method'], n_jobs=-1,
).fit_predict(X_winner)
lab_hdb_force = _force_assign(lab_hdb, X_winner)
hdb_k = len(set(lab_hdb_force)) - (1 if -1 in lab_hdb_force else 0)

# HDBSCAN force=False
noise_frac = (lab_hdb == -1).sum() / len(lab_hdb)
from _common import per_type_purity, compute_purity
assigned_mask = lab_hdb != -1
macro_hdb_native = np.mean(list(per_type_purity(lab_hdb[assigned_mask], y_w[assigned_mask]).values()))

# k-means at winner's k (10 seeds)
km_macros = []
for seed in range(42, 52):
    lab_km = MiniBatchKMeans(n_clusters=hdb_k, batch_size=1024, n_init='auto', random_state=seed).fit_predict(X_winner)
    km_macros.append(np.mean(list(per_type_purity(lab_km, y_w).values())))

# k-means at k=17
km17_macros = []
for seed in range(42, 52):
    lab_km17 = MiniBatchKMeans(n_clusters=17, batch_size=1024, n_init='auto', random_state=seed).fit_predict(X_winner)
    km17_macros.append(np.mean(list(per_type_purity(lab_km17, y_w).values())))

km_rows = []
km_rows.append({'method': f'k-means k={hdb_k}', 'k': hdb_k,
                'macro_purity': np.mean(km_macros), 'std': np.std(km_macros), 'noise': 0.0,
                'recipe': w_recipe, 'visual_encoder': w_enc})
km_rows.append({'method': 'HDBSCAN force=True', 'k': hdb_k,
                'macro_purity': float(winner['macro_purity']), 'std': 0.0, 'noise': 0.0,
                'recipe': w_recipe, 'visual_encoder': w_enc})
km_rows.append({'method': f'HDBSCAN force=False (assigned only)', 'k': hdb_k,
                'macro_purity': float(macro_hdb_native), 'std': 0.0, 'noise': float(noise_frac),
                'recipe': w_recipe, 'visual_encoder': w_enc})
km_rows.append({'method': 'k-means k=17', 'k': 17,
                'macro_purity': np.mean(km17_macros), 'std': np.std(km17_macros), 'noise': 0.0,
                'recipe': w_recipe, 'visual_encoder': w_enc})

km_df = pd.DataFrame(km_rows)
# Append to existing k_matched
km_path = RESULTS_ROOT.parent / 'midterm' / 'k_matched_comparison.csv'
if km_path.exists():
    old_km = pd.read_csv(km_path)
    old_km = old_km[~old_km['recipe'].isin([w_recipe])] if 'recipe' in old_km.columns else old_km
    km_df = pd.concat([old_km, km_df], ignore_index=True)
km_df.to_csv(km_path, index=False)

print(f'\n  K-matched results (k={hdb_k}):')
for _, r in pd.DataFrame(km_rows).iterrows():
    std_str = f' ± {r["std"]:.3f}' if r['std'] > 0 else ''
    noise_str = f' noise={r["noise"]:.1%}' if r['noise'] > 0 else ''
    print(f'    {r["method"]:40s} macro={r["macro_purity"]:.4f}{std_str}{noise_str}')


# ══════════════════════════════════════════════════════════════════════════
# PART D: Gap #10 — Bootstrap stability on winner
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('PART D: Bootstrap stability (10 × 90% subsamples)')
print('='*60)

N_BOOT = 10
FRAC = 0.9
rng = np.random.RandomState(42)

boot_labels = []
boot_purities = []

for b in range(N_BOOT):
    n = len(X_winner)
    idx_b = rng.choice(n, size=int(n * FRAC), replace=False)
    X_b = X_winner[idx_b]
    y_b = y_w[idx_b]

    lab_raw = HDBSCAN(
        min_cluster_size=int(hps['mcs']), min_samples=int(hps['ms']),
        cluster_selection_method=hps['method'], n_jobs=-1,
    ).fit_predict(X_b)
    lab_b = _force_assign(lab_raw, X_b)

    pur = compute_purity(lab_b, y_b)
    macro = np.mean(list(per_type_purity(lab_b, y_b).values()))
    k_b = len(set(lab_b))
    boot_labels.append((idx_b, lab_b))
    boot_purities.append({'run': b, 'purity': pur, 'macro_purity': macro, 'k': k_b})
    print(f'  run {b}: macro={macro:.4f} purity={pur:.4f} k={k_b}')

# Pairwise ARI
from sklearn.metrics import adjusted_rand_score
ari_rows = []
for i in range(N_BOOT):
    for j in range(i + 1, N_BOOT):
        idx_i, lab_i = boot_labels[i]
        idx_j, lab_j = boot_labels[j]
        overlap = np.intersect1d(idx_i, idx_j)
        if len(overlap) < 100:
            continue
        pos_i = np.searchsorted(np.sort(idx_i), overlap)
        pos_j = np.searchsorted(np.sort(idx_j), overlap)
        order_i = np.argsort(idx_i)
        order_j = np.argsort(idx_j)
        lab_ov_i = lab_i[order_i[pos_i]]
        lab_ov_j = lab_j[order_j[pos_j]]
        ari = adjusted_rand_score(lab_ov_i, lab_ov_j)
        ari_rows.append({'run_i': i, 'run_j': j, 'ari': ari, 'overlap': len(overlap)})

ari_df = pd.DataFrame(ari_rows)
pur_df = pd.DataFrame(boot_purities)

ari_df.to_csv(STAB_DIR / 'bootstrap_ari_f1a_umap.csv', index=False)
pur_df.to_csv(STAB_DIR / 'bootstrap_purity_f1a_umap.csv', index=False)

print(f'\n  ARI: {ari_df["ari"].mean():.3f} ± {ari_df["ari"].std():.3f} ({len(ari_df)} pairs)')
print(f'  Macro purity: {pur_df["macro_purity"].mean():.3f} ± {pur_df["macro_purity"].std():.3f}')
print(f'  Purity: {pur_df["purity"].mean():.3f} ± {pur_df["purity"].std():.3f}')

print('\n' + '='*60)
print('ALL DONE')
print('='*60)
