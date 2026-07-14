"""Static figure: grid of representative clusters with image thumbnails.

Picks 8 representative clusters from the winning fusion model and lays out
each as a panel with 6 thumbnails + cluster header (top type, size, purity,
top-3 TF-IDF keywords).
"""
from __future__ import annotations
import os, sys, tempfile, re
from collections import Counter
from pathlib import Path

os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, 'notebooks/data_analysis/feature_engineering')
sys.path.insert(0, 'notebooks/data_analysis/experiments')
import P2c_fusion_full as P
from _common import BASELINE, PROJECT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual

DESC_DIR = PROJECT / 'processed_data/descriptions/gemini_v2_v7_ifc_aware'
RENDERS_DIR = PROJECT / 'processed_data/rendersv2/colored'

N_TILES_PER_CLUSTER = 6  # 2 rows × 3 cols thumbnails
N_KEYWORDS = 3


def force_assign(lab, X):
    lab = lab.copy()
    non_noise = lab != -1
    if non_noise.sum() and (-1 in lab):
        cids = sorted(set(lab[non_noise].tolist()))
        cents = np.stack([X[lab == c].mean(0) for c in cids])
        ni = np.where(~non_noise)[0]
        d = np.linalg.norm(X[ni][:, None, :] - cents[None, :, :], axis=2)
        lab[ni] = np.array(cids)[d.argmin(1)]
    return lab


def load_desc(gid):
    p = DESC_DIR / f'{gid}.txt'
    return p.read_text() if p.exists() else ''


def normalize(txt):
    txt = txt.lower()
    txt = re.sub(r'ifc\w+', ' ', txt)
    txt = re.sub(r'[,\.:;\(\)\[\]/\-]', ' ', txt)
    txt = re.sub(r'\d+:\d+(?::\d+)?', ' ', txt)
    return txt


def load_thumb(gid):
    p = RENDERS_DIR / gid / 'iso.png'
    return imread(p) if p.exists() else None


# Build winning representation + cluster
print('Building winning representation...')
df, y_full = load_data()
gids_all = df['GlobalId'].values
geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))
X_txt, txt_idx = load_text(gids_all, 'v7', 'single')
X_vis, vis_idx = load_visual(gids_all, 'siglip', 'colored')
common = np.intersect1d(txt_idx, vis_idx)
geo_b = geo[common]
txt_b = X_txt[np.searchsorted(txt_idx, common)]
vis_b = X_vis[np.searchsorted(vis_idx, common)]
y = y_full[common]
gids = gids_all[common]

X, _ = P.build_F5_varbal({'geo': geo_b, 'text': txt_b, 'visual': vis_b}, 4, 'demo_grid')
Xs = StandardScaler().fit_transform(X).astype(np.float64)
labels = force_assign(
    HDBSCAN(min_cluster_size=5, min_samples=3,
            cluster_selection_method='leaf', n_jobs=-1).fit_predict(Xs),
    Xs,
)
cluster_ids = sorted(set(labels.tolist()))
print(f'k = {len(cluster_ids)}')

# TF-IDF
print('TF-IDF...')
descs = np.array([load_desc(g) for g in gids])
norm_descs = [normalize(d) for d in descs]
per_cluster_docs = [' '.join(np.array(norm_descs)[labels == c]) for c in cluster_ids]
vec = TfidfVectorizer(min_df=2, max_df=0.6, stop_words='english',
                      ngram_range=(1, 2), token_pattern=r'[A-Za-z]{3,}')
mat = vec.fit_transform(per_cluster_docs)
vocab = np.array(vec.get_feature_names_out())

# Per-cluster summaries
summaries = []
for ci, c in enumerate(cluster_ids):
    idx = np.where(labels == c)[0]
    type_counts = Counter(y[idx].tolist())
    top_type, top_count = type_counts.most_common(1)[0]
    purity = top_count / len(idx)
    scores = mat[ci].toarray().ravel()
    top_kw = vocab[np.argsort(-scores)[:N_KEYWORDS]].tolist()
    # Sort members by closeness to cluster center
    mean = Xs[idx].mean(axis=0, keepdims=True)
    d = np.linalg.norm(Xs[idx] - mean, axis=1)
    sort_order = idx[np.argsort(d)]
    summaries.append({
        'cluster_id': int(c),
        'size': len(idx),
        'purity': purity,
        'top_type': str(top_type),
        'keywords': top_kw,
        'exemplars': sort_order[:N_TILES_PER_CLUSTER],
    })

# Pick 8 representative clusters: one per major type, prefer pure & sized 10-50
pick_types = ['Stair', 'IfcRailing', 'IfcLightFixture', 'MEP',
              'IfcCurtainWall', 'Furniture', 'IfcDoor', 'IfcColumn']
picked = []
used = set()
for tp in pick_types:
    candidates = [s for s in summaries
                  if s['top_type'] == tp and s['cluster_id'] not in used
                  and s['purity'] > 0.7 and s['size'] >= 8]
    if candidates:
        # Prefer mid-size, high-purity ones
        candidates.sort(key=lambda s: (-s['purity'], -min(s['size'], 30)))
        chosen = candidates[0]
        picked.append(chosen)
        used.add(chosen['cluster_id'])

print(f'\nPicked {len(picked)} representative clusters:')
for s in picked:
    print(f"  Cluster {s['cluster_id']:3d}  {s['top_type']:20s}  "
          f"size={s['size']:3d}  purity={s['purity']:.2f}  "
          f"kw={'|'.join(s['keywords'])}")

# Lay out: 4 rows × 2 cluster panels, each panel = 2×3 thumbnails + header
N_COLS = 2
N_ROWS = (len(picked) + N_COLS - 1) // N_COLS
fig = plt.figure(figsize=(16, 4.0 * N_ROWS))
outer = fig.add_gridspec(N_ROWS, N_COLS, hspace=0.5, wspace=0.15)

for k, summary in enumerate(picked):
    r, c = k // N_COLS, k % N_COLS
    panel_gs = outer[r, c].subgridspec(2, 3, hspace=0.02, wspace=0.02)
    # Thumbnails
    for j, mem_idx in enumerate(summary['exemplars']):
        ax = fig.add_subplot(panel_gs[j // 3, j % 3])
        img = load_thumb(gids[mem_idx])
        if img is not None:
            ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#ddd')
            spine.set_linewidth(0.5)
        # Tiny label of true type
        ax.set_xlabel(str(y[mem_idx]).replace('Ifc', ''),
                      fontsize=8, color='#666', labelpad=2)

    # Panel header in the top axis
    header_text = (f"Cluster {summary['cluster_id']}  ·  "
                   f"{summary['top_type'].replace('Ifc','')}  ·  "
                   f"n={summary['size']}  ·  "
                   f"purity={summary['purity']:.2f}\n"
                   f"keywords: {' · '.join(summary['keywords'])}")
    # Add header above all 6 tiles
    bbox = outer[r, c].get_position(fig)
    fig.text(bbox.x0 + bbox.width * 0.5, bbox.y1 + 0.015,
             header_text, ha='center', va='bottom', fontsize=10,
             color='#333',
             bbox=dict(facecolor='#fff8e1', edgecolor='#ddd',
                       boxstyle='round,pad=0.4'))

fig.suptitle('Discovered library: 8 representative clusters from the winning fusion\n'
             '(F5_varbal-umap4 + HDBSCAN, SigLIP colored, geo+text+visual)',
             fontsize=13, y=0.995)

out = 'notebooks/data_analysis/presentation/figures/16_cluster_grid.png'
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'\nsaved → {out}')
