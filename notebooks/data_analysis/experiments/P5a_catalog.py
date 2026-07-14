"""P5a — Auto-generated library catalog (HTML).

For every cluster in the winning fusion clustering, produces:
  * 3 exemplar thumbnails (cluster-center-nearest meshes)
  * top-5 TF-IDF keywords mined from Gemini v7 descriptions
  * member type mix
  * inline member thumbnails

Output:
    results/midterm/mining/catalog/index.html
    results/midterm/mining/catalog/cluster_<id>.html
    results/midterm/mining/catalog/catalog_summary.csv
"""
from __future__ import annotations
import html as htmllib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))

from _common import PROJECT, RESULTS_ROOT
from _winners import (
    CANONICAL_TEXT_VERSION, cluster_with_canonical_winner,
)

DESCRIPTIONS_DIR = PROJECT / 'processed_data/descriptions/gemini_v2_v7_ifc_aware'
RENDERS_DIR = PROJECT / 'processed_data/rendersv2/colored'
OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'mining' / 'catalog'
OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_REL = 'thumbs'
(OUT_DIR / THUMB_REL).mkdir(exist_ok=True)

N_EXEMPLARS = 3
N_KEYWORDS = 5
MAX_MEMBER_PREVIEW = 24  # thumbnails per cluster page beyond exemplars


def _load_description(gid: str) -> str:
    p = DESCRIPTIONS_DIR / f'{gid}.txt'
    if not p.exists():
        return ''
    txt = p.read_text().lower()
    # Remove IFC type tokens and connector punctuation so TF-IDF surfaces
    # substantive descriptive words.
    txt = re.sub(r'ifc\w+', ' ', txt)
    txt = re.sub(r'[,\.:;\(\)\[\]/\-]', ' ', txt)
    txt = re.sub(r'\d+:\d+(?::\d+)?', ' ', txt)  # ratios like 1:10:30
    return txt


def _copy_thumb(gid: str) -> str | None:
    """Copy iso.png for gid to catalog thumbs dir; return relative path or None."""
    src = RENDERS_DIR / gid / 'iso.png'
    if not src.exists():
        return None
    dst = OUT_DIR / THUMB_REL / f'{gid}.png'
    if not dst.exists():
        shutil.copy(src, dst)
    return f'{THUMB_REL}/{gid}.png'


def _exemplar_indices(idx: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """k members closest to the cluster mean (Euclidean in fusion space)."""
    if len(idx) <= k:
        return idx
    mean = X[idx].mean(axis=0, keepdims=True)
    d = np.linalg.norm(X[idx] - mean, axis=1)
    order = np.argsort(d)[:k]
    return idx[order]


def main():
    clusters, X, y, gids, info = cluster_with_canonical_winner(algo='hdbscan')
    print(f'[P5a] using winner: {info["subset"]}/{info["recipe"]} '
          f'macro={info["macro_purity"]:.4f}')
    n = len(X)

    # Load descriptions once
    docs = np.array([_load_description(g) for g in gids])
    vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words='english',
                          ngram_range=(1, 2), token_pattern=r'[A-Za-z]{3,}')
    # Build per-cluster "documents" (concatenated per-mesh docs) to find
    # distinctive keywords per cluster.
    cluster_ids = sorted(set(clusters.tolist()))
    per_cluster_docs = []
    for c in cluster_ids:
        idx = np.where(clusters == c)[0]
        per_cluster_docs.append(' '.join(docs[idx]))
    try:
        mat = vec.fit_transform(per_cluster_docs)
        vocab = np.array(vec.get_feature_names_out())
    except ValueError:
        print('[P5a] TF-IDF vocabulary empty; falling back to token counts')
        mat = None
        vocab = None

    rows = []
    summary_cards = []
    for ci, c in enumerate(cluster_ids):
        idx = np.where(clusters == c)[0]
        cluster_y = y[idx]
        type_mix = Counter(cluster_y.tolist())
        top_type, top_count = type_mix.most_common(1)[0]
        purity = top_count / len(idx)

        if mat is not None:
            scores = mat[ci].toarray().ravel()
            top_kw = vocab[np.argsort(-scores)[:N_KEYWORDS]].tolist()
        else:
            top_kw = []

        exemplar_idx = _exemplar_indices(idx, X, N_EXEMPLARS)
        exemplars = [(gids[i], _copy_thumb(gids[i]), str(y[i])) for i in exemplar_idx]

        row = dict(cluster_id=int(c), size=len(idx), purity=purity,
                   top_type=str(top_type), top_keywords='|'.join(top_kw))
        rows.append(row)

        # Per-cluster HTML
        preview = idx[:MAX_MEMBER_PREVIEW]
        preview_thumbs = []
        for i in preview:
            th = _copy_thumb(gids[i])
            if th:
                preview_thumbs.append((gids[i], th, str(y[i])))

        cluster_html = _render_cluster_page(
            cluster_id=int(c), size=len(idx), purity=purity,
            top_type=top_type, top_kw=top_kw, exemplars=exemplars,
            previews=preview_thumbs, type_mix=type_mix,
        )
        (OUT_DIR / f'cluster_{int(c):03d}.html').write_text(cluster_html)

        summary_cards.append(dict(
            cluster_id=int(c), size=len(idx), purity=purity,
            top_type=str(top_type), top_keywords=top_kw,
            first_exemplar=exemplars[0] if exemplars else None,
        ))

    pd.DataFrame(rows).to_csv(OUT_DIR / 'catalog_summary.csv', index=False)
    print(f'saved → {OUT_DIR / "catalog_summary.csv"} ({len(rows)} clusters)')

    # Index page
    index_html = _render_index(summary_cards)
    (OUT_DIR / 'index.html').write_text(index_html)
    print(f'saved → {OUT_DIR / "index.html"}')


# ── HTML rendering ──────────────────────────────────────────────────────────
_CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 24px; color: #222; background: #fafafa; }
h1 { border-bottom: 2px solid #444; padding-bottom: 4px; }
.cluster-card { background: white; border: 1px solid #ddd; border-radius: 8px;
                padding: 12px; margin-bottom: 14px;
                display: grid; grid-template-columns: 110px 1fr; gap: 12px; }
.cluster-card img { width: 100px; height: 100px; object-fit: cover;
                    background: #f0f0f0; border-radius: 4px; }
.meta { font-size: 14px; line-height: 1.5; }
.meta .title { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
.keywords { font-family: ui-monospace, Menlo, monospace; color: #555; }
.mix { color: #666; font-size: 13px; }
.thumbs { display: flex; flex-wrap: wrap; gap: 8px; }
.thumbs figure { margin: 0; text-align: center; }
.thumbs img { width: 80px; height: 80px; object-fit: cover;
              background: #f0f0f0; border-radius: 4px; }
.thumbs figcaption { font-size: 10px; color: #666;
                     max-width: 80px; overflow: hidden; text-overflow: ellipsis; }
a { text-decoration: none; color: #0d6efd; }
a:hover { text-decoration: underline; }
.exemplar-row { display: flex; gap: 12px; margin: 8px 0 16px 0; }
.exemplar-row img { width: 140px; height: 140px; }
</style>
"""


def _render_index(cards) -> str:
    parts = ['<!DOCTYPE html><html><head>',
             '<meta charset="utf-8"><title>Library Catalog</title>',
             _CSS, '</head><body>',
             '<h1>Library Catalog — Auto-Generated</h1>',
             f'<p>{len(cards)} clusters discovered by HDBSCAN on the '
             'multi-modal fusion embedding (geo + text-v7 + visual-SigLIP). '
             'Each cluster is presented with its most representative member '
             'and top-5 TF-IDF keywords mined from Gemini descriptions.</p>']
    # Sort cards by size desc
    for card in sorted(cards, key=lambda c: -c['size']):
        c = card['cluster_id']
        thumb_html = ''
        if card['first_exemplar']:
            gid, rel, tp = card['first_exemplar']
            if rel:
                thumb_html = f'<img src="{htmllib.escape(rel)}" alt="{gid}">'
        kw = ', '.join(card['top_keywords'])
        parts.append(
            f'<div class="cluster-card">{thumb_html}<div class="meta">'
            f'<div class="title"><a href="cluster_{c:03d}.html">'
            f'Cluster {c:03d}</a> <span style="color:#666;font-weight:400;">'
            f'({card["size"]} objects, {card["purity"]:.0%} '
            f'{htmllib.escape(card["top_type"].replace("Ifc",""))})</span></div>'
            f'<div class="keywords">{htmllib.escape(kw)}</div>'
            f'</div></div>'
        )
    parts.append('</body></html>')
    return '\n'.join(parts)


def _render_cluster_page(*, cluster_id, size, purity, top_type, top_kw,
                          exemplars, previews, type_mix) -> str:
    parts = ['<!DOCTYPE html><html><head>',
             f'<meta charset="utf-8"><title>Cluster {cluster_id:03d}</title>',
             _CSS, '</head><body>',
             '<p><a href="index.html">&larr; Back to index</a></p>',
             f'<h1>Cluster {cluster_id:03d}</h1>',
             f'<p><b>{size}</b> members &middot; '
             f'dominant type: <b>{htmllib.escape(str(top_type).replace("Ifc",""))}</b> '
             f'({purity:.0%}) &middot; top keywords: <span class="keywords">'
             f'{htmllib.escape(", ".join(top_kw))}</span></p>',
             '<h2>Exemplars</h2>', '<div class="exemplar-row">']
    for gid, rel, tp in exemplars:
        label = htmllib.escape(f'{gid[:12]}… · {tp.replace("Ifc","")}')
        if rel:
            parts.append(
                f'<figure><img src="{htmllib.escape(rel)}" alt="{gid}">'
                f'<figcaption>{label}</figcaption></figure>'
            )
        else:
            parts.append(f'<figure><figcaption>[no render] {label}</figcaption></figure>')
    parts.append('</div>')

    parts.append('<h2>Type mix</h2><ul class="mix">')
    for t, c in sorted(type_mix.items(), key=lambda kv: -kv[1]):
        parts.append(f'<li>{htmllib.escape(str(t).replace("Ifc",""))}: {c}</li>')
    parts.append('</ul>')

    parts.append(f'<h2>Members (first {len(previews)} shown)</h2>')
    parts.append('<div class="thumbs">')
    for gid, rel, tp in previews:
        label = htmllib.escape(tp.replace('Ifc', ''))
        parts.append(
            f'<figure><img src="{htmllib.escape(rel)}" alt="{gid}">'
            f'<figcaption>{label}</figcaption></figure>'
        )
    parts.append('</div></body></html>')
    return '\n'.join(parts)


if __name__ == '__main__':
    main()
