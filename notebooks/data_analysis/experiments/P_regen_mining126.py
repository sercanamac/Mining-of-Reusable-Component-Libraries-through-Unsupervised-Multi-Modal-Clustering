"""Regenerate the stale mining-layer artifacts (left over from the old F1a
k=102 midterm run) from the verified F5_varbal k=126 winner partition.

Produces:
  * results/midterm/mining/cluster_type_mix126.csv  (per-cluster summary)
  * results/midterm/mining/catalog126_pure.csv      (100%-pure, size>=10, TF-IDF)
  * presentation/figures/13_cluster_sizes.png        (126-cluster histogram)
  * presentation/figures/16_cluster_grid.png         (8-cluster exemplar montage)
and mirrors the two PNGs into thesis/chapters/images/midterm/.

Run P_regen_winner126.py first (writes winner126_partition.parquet).
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'feature_engineering'))
from _common import per_type_purity  # noqa: E402

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent / 'feature_engineering'))

REPO = _EXP.parents[2]
DA = _EXP.parent
MIDTERM = DA / 'results' / 'midterm'
FIGS = DA / 'presentation' / 'figures'
THESIS_IMG = REPO / 'thesis' / 'chapters' / 'images' / 'midterm'
DESCRIPTIONS_DIR = REPO / 'processed_data/descriptions/gemini_v2_v7_ifc_aware'
RENDERS_DIR = REPO / 'processed_data/rendersv2/colored'

N_KEYWORDS = 5


def _load_desc(gid: str) -> str:
    p = DESCRIPTIONS_DIR / f'{gid}.txt'
    if not p.exists():
        return ''
    txt = p.read_text().lower()
    txt = re.sub(r'ifc\w+', ' ', txt)
    txt = re.sub(r'[,\.:;\(\)\[\]/\-]', ' ', txt)
    txt = re.sub(r'\d+:\d+(?::\d+)?', ' ', txt)
    return txt


def _save(fig, name):
    for d in (FIGS, THESIS_IMG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}')


def main():
    part = pd.read_parquet(MIDTERM / 'mining' / 'winner126_partition.parquet')
    gids = part['GlobalId'].values
    y = part['type'].values
    clusters = part['cluster'].values
    cluster_ids = sorted(set(clusters.tolist()))
    k = len(cluster_ids)
    print(f'[mining126] {len(part)} objects, k={k}')

    # ── TF-IDF per cluster (distinctive keywords) ──────────────────────────
    docs = np.array([_load_desc(g) for g in gids])
    vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words='english',
                          ngram_range=(1, 2), token_pattern=r'[A-Za-z]{3,}')
    per_cluster_docs = [' '.join(docs[clusters == c]) for c in cluster_ids]
    mat = vec.fit_transform(per_cluster_docs)
    vocab = np.array(vec.get_feature_names_out())

    rows = []
    for ci, c in enumerate(cluster_ids):
        idx = np.where(clusters == c)[0]
        mix = Counter(y[idx].tolist())
        top_type, top_count = mix.most_common(1)[0]
        purity = top_count / len(idx)
        scores = mat[ci].toarray().ravel()
        # keep only keywords with fully disjoint words (cleanest, prefers the
        # higher-scored token and drops any overlapping uni/bigram)
        top_kw, kept_words = [], set()
        for tok in vocab[np.argsort(-scores)]:
            words = set(tok.split())
            if words & kept_words:
                continue
            top_kw.append(tok)
            kept_words |= words
            if len(top_kw) == N_KEYWORDS:
                break
        rows.append(dict(cluster_id=int(c), size=len(idx), purity=purity,
                         dominant_type=str(top_type),
                         top_keywords=', '.join(top_kw)))
    summ = pd.DataFrame(rows).sort_values('size', ascending=False)
    summ.to_csv(MIDTERM / 'mining' / 'cluster_type_mix126.csv', index=False)

    # ── Table 5.12: 100%-pure clusters of size >= 10 ───────────────────────
    pure = summ[(summ.purity >= 0.999) & (summ['size'] >= 10)].copy()
    pure.to_csv(MIDTERM / 'mining' / 'catalog126_pure.csv', index=False)
    sub_counts = pure['dominant_type'].value_counts()
    print(f'[mining126] {len(pure)} pure clusters size>=10; '
          f'sub-type counts:\n{sub_counts.to_string()}')
    print('\n--- LaTeX rows for Table 5.12 (top 20 by size) ---')
    for _, r in pure.head(20).iterrows():
        disp_type = re.sub(r'^Ifc', '', str(r.dominant_type))
        print(f"{r.cluster_id} & {r['size']} & {disp_type} & "
              f"{r.top_keywords} \\\\")

    # ── Figure 13: cluster size histogram ──────────────────────────────────
    sizes = summ['size'].values
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = np.argsort(-sizes)
    ax.bar(range(len(sizes)), sizes[order], color='#3182bd', edgecolor='white',
           width=1.0)
    ax.axhline(5, color='#d94a3f', ls='--', lw=1, alpha=0.7,
               label='min_cluster_size = 5 floor')
    ax.set_xlabel('cluster rank (large -> small)')
    ax.set_ylabel('members')
    ax.set_title(f'Cluster size distribution - {k} clusters (F5_varbal fusion winner)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    _save(fig, '13_cluster_sizes.png')

    # ── Figure 16: 8-cluster exemplar montage (3 centroid-nearest each) ─────
    emb4 = np.load(MIDTERM / 'cache' / 'reducers' /
                   ('fusion_geo_text_visual_siglip_colored_F5_vb'
                    '__65e4dcd408f0__umap_4__8a4ac8e713.npy'))  # same order as part
    want_types = ['Stair', 'IfcRailing', 'IfcLightFixture', 'MEP',
                  'IfcCurtainWall', 'Furniture', 'IfcDoor', 'IfcColumn']
    chosen, used = [], set()
    for t in want_types:
        cand = pure[(pure.dominant_type == t) & (~pure.cluster_id.isin(used))]
        if len(cand):
            r = cand.iloc[0]
            chosen.append(r)
            used.add(r.cluster_id)
    assert len(chosen) == 8, f'only {len(chosen)} montage clusters'

    def nearest3(cid):
        idx = np.where(clusters == cid)[0]
        cen = emb4[idx].mean(0, keepdims=True)
        d = np.linalg.norm(emb4[idx] - cen, axis=1)
        return idx[np.argsort(d)[:3]]

    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for ax, r in zip(axes.ravel(), chosen):
        ex = nearest3(int(r.cluster_id))
        tiles = []
        for i in ex:
            iso = RENDERS_DIR / gids[i] / 'iso.png'
            if iso.exists():
                tiles.append(mpimg.imread(iso))
        if tiles:
            h = min(t.shape[0] for t in tiles)
            tiles = [t[:h, :, :] for t in tiles]
            ax.imshow(np.concatenate(tiles, axis=1))
        kws = ', '.join(r.top_keywords.split(', ')[:3])
        ax.set_title(f"Cluster {r.cluster_id} - {r.dominant_type} - "
                     f"n={r['size']} - purity=1.00\nkeywords: {kws}",
                     fontsize=9)
        ax.axis('off')
    fig.suptitle('Discovered library: 8 representative clusters from the winning '
                 'fusion\n(F5_varbal HDBSCAN, k=126, SigLIP coloured, '
                 'geo+text+visual)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, '16_cluster_grid.png')

    # ── Figure 5.10: per-type purity at winner (2-decimal labels) ──────────
    ptp = per_type_purity(clusters, y)
    sizes_by_t = Counter(y.tolist())
    macro = float(np.mean(list(ptp.values())))
    items = sorted(ptp.items(), key=lambda kv: kv[1])  # ascending -> bottom worst
    types = [t for t, _ in items]
    vals = [v for _, v in items]
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = cm.RdYlGn([0.15 + 0.8 * v for v in vals])
    ax.barh(range(len(types)), vals, color=colors, edgecolor='white')
    for i, (t, v) in enumerate(zip(types, vals)):
        ax.text(v + 0.01, i, f"{v:.2f} (n={sizes_by_t[t]})",
                va='center', fontsize=8)
    ax.axvline(macro, color='#333', ls='--', lw=1)
    ax.text(macro + 0.005, 0.2, f'Macro = {macro:.3f}', fontsize=9, rotation=90,
            va='bottom', color='#333')
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Per-type purity')
    ax.set_title('Best fusion (F5_varbal-umap4 + HDBSCAN, SigLIP, trimodal)\n'
                 f'Per-type purity at k={k} clusters, macro = {macro:.3f}')
    fig.tight_layout()
    _save(fig, '14_fusion_winner_pertype.png')


if __name__ == '__main__':
    main()
