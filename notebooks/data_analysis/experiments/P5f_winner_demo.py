"""P5f — Interactive cluster demo for the new winning fusion model.

Uses the F5_varbal-umap4 (SigLIP colored, trimodal) clustering and produces
a single self-contained HTML page where you can navigate clusters with
arrow keys / buttons. For each cluster shows:
  * grid of member renders
  * top-10 TF-IDF keywords mined from Gemini v7 descriptions
  * 3 sample descriptions
  * type mix
  * cluster size, purity, dominant type

Output:
    results/midterm/mining/winner_demo/index.html
"""
from __future__ import annotations
import os, sys, tempfile, html as htmllib, json, re, shutil
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

import P2c_fusion_full as P
from _common import BASELINE, PROJECT, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual

DESC_DIR = PROJECT / 'processed_data/descriptions/gemini_v2_v7_ifc_aware'
RENDERS_DIR = PROJECT / 'processed_data/rendersv2/colored'
OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'mining' / 'winner_demo'
OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR = OUT_DIR / 'thumbs'
THUMB_DIR.mkdir(exist_ok=True)

N_KEYWORDS = 10
SAMPLE_DESCRIPTIONS = 3
MAX_MEMBERS_PER_CLUSTER = 50  # cap grid size for huge clusters


def _force_assign(lab, X):
    lab = lab.copy()
    non_noise = lab != -1
    if non_noise.sum() and (-1 in lab):
        cids = sorted(set(lab[non_noise].tolist()))
        cents = np.stack([X[lab == c].mean(0) for c in cids])
        ni = np.where(~non_noise)[0]
        d = np.linalg.norm(X[ni][:, None, :] - cents[None, :, :], axis=2)
        lab[ni] = np.array(cids)[d.argmin(1)]
    return lab


def _load_description_raw(gid: str) -> str:
    p = DESC_DIR / f'{gid}.txt'
    if not p.exists():
        return ''
    return p.read_text().strip()


def _normalize_for_tfidf(text: str) -> str:
    """Strip IFC tokens, ratios, punctuation."""
    txt = text.lower()
    txt = re.sub(r'ifc\w+', ' ', txt)
    txt = re.sub(r'[,\.:;\(\)\[\]/\-]', ' ', txt)
    txt = re.sub(r'\d+:\d+(?::\d+)?', ' ', txt)
    return txt


def _copy_thumb(gid: str):
    src = RENDERS_DIR / gid / 'iso.png'
    if not src.exists():
        return None
    dst = THUMB_DIR / f'{gid}.png'
    if not dst.exists():
        shutil.copy(src, dst)
    return f'thumbs/{gid}.png'


def main():
    # ── Build the winning fusion representation ─────────────────────────────
    print('Loading data...')
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
    blocks = {'geo': geo_b, 'text': txt_b, 'visual': vis_b}
    print(f'n={len(common)}  '
          f'geo={geo_b.shape[1]}  text={txt_b.shape[1]}  vis={vis_b.shape[1]}')

    print('Building F5_varbal-umap4 representation...')
    X, _ = P.build_F5_varbal(blocks, 4, 'demo_winner')
    Xs = StandardScaler().fit_transform(X).astype(np.float64)

    print('Clustering with HDBSCAN(mcs=5, ms=3, leaf, force=True)...')
    h = HDBSCAN(min_cluster_size=5, min_samples=3,
                cluster_selection_method='leaf', n_jobs=-1)
    labels = _force_assign(h.fit_predict(Xs), Xs)
    cluster_ids = sorted(set(labels.tolist()))
    print(f'k = {len(cluster_ids)} clusters')

    # ── TF-IDF per cluster ────────────────────────────────────────────────
    print('Computing TF-IDF on Gemini v7 descriptions...')
    raw_descs = [_load_description_raw(g) for g in gids]
    norm_descs = [_normalize_for_tfidf(d) for d in raw_descs]
    per_cluster_docs = [
        ' '.join(np.array(norm_descs)[labels == c])
        for c in cluster_ids
    ]
    vec = TfidfVectorizer(min_df=2, max_df=0.6, stop_words='english',
                          ngram_range=(1, 2), token_pattern=r'[A-Za-z]{3,}')
    mat = vec.fit_transform(per_cluster_docs)
    vocab = np.array(vec.get_feature_names_out())

    # ── Build cluster dossiers ────────────────────────────────────────────
    print('Building cluster dossiers...')
    cluster_data = []
    for ci, c in enumerate(cluster_ids):
        idx = np.where(labels == c)[0]
        cluster_y = y[idx]
        type_mix = Counter(cluster_y.tolist())
        top_type, top_count = type_mix.most_common(1)[0]
        purity = top_count / len(idx)

        # TF-IDF keywords
        scores = mat[ci].toarray().ravel()
        top_kw = vocab[np.argsort(-scores)[:N_KEYWORDS]].tolist()

        # Exemplars (closest to centroid)
        mean = Xs[idx].mean(axis=0, keepdims=True)
        d = np.linalg.norm(Xs[idx] - mean, axis=1)
        sort_order = idx[np.argsort(d)]

        # Cap members shown
        shown = sort_order[:MAX_MEMBERS_PER_CLUSTER]
        members = []
        for i in shown:
            rel = _copy_thumb(gids[i])
            members.append({
                'gid': gids[i],
                'thumb': rel,
                'type': str(y[i]),
                'description': raw_descs[i],
            })

        # 3 sample descriptions (different from exemplars: vary)
        sample_idx = sort_order[[0, len(sort_order) // 2,
                                  len(sort_order) - 1][:SAMPLE_DESCRIPTIONS]]
        samples = []
        for i in sample_idx:
            d_raw = raw_descs[i]
            samples.append({'gid': gids[i], 'type': str(y[i]),
                            'description': d_raw or '[no description]'})

        cluster_data.append({
            'cluster_id': int(c),
            'size': int(len(idx)),
            'purity': float(purity),
            'top_type': str(top_type),
            'top_keywords': top_kw,
            'type_mix': [{'type': str(t), 'n': int(n)}
                          for t, n in sorted(type_mix.items(), key=lambda kv: -kv[1])],
            'samples': samples,
            'members': members,
        })

    # Sort clusters by size descending (biggest first)
    cluster_data.sort(key=lambda c: -c['size'])

    # ── Render HTML ───────────────────────────────────────────────────────
    print('Writing HTML...')
    html = _render_demo_html(cluster_data)
    (OUT_DIR / 'index.html').write_text(html)
    print(f'saved → {OUT_DIR / "index.html"}')
    print(f'open in browser: file://{(OUT_DIR / "index.html").resolve()}')


def _render_demo_html(clusters) -> str:
    """Single-page demo with JS navigation."""
    # Serialize cluster data as JSON for client-side use
    data_json = json.dumps(clusters)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Winner Cluster Demo — F5_varbal-umap4 + HDBSCAN</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0; padding: 0; color: #222; background: #f5f5f7;
  }}
  header {{
    background: #1d1d1f; color: white; padding: 16px 32px;
    display: flex; justify-content: space-between; align-items: center;
  }}
  header h1 {{ margin: 0; font-size: 18px; font-weight: 500; }}
  header .stats {{ font-size: 14px; opacity: 0.8; }}
  .controls {{
    background: white; padding: 12px 32px; border-bottom: 1px solid #ddd;
    display: flex; align-items: center; gap: 16px;
    position: sticky; top: 0; z-index: 10;
  }}
  .controls button {{
    padding: 6px 14px; font-size: 14px; border: 1px solid #ccc;
    background: white; border-radius: 6px; cursor: pointer;
  }}
  .controls button:hover {{ background: #f0f0f0; }}
  .controls button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .controls input {{
    width: 60px; padding: 4px 6px; font-size: 14px;
    border: 1px solid #ccc; border-radius: 4px; text-align: center;
  }}
  .controls .position {{ font-weight: 500; }}
  .controls select {{
    padding: 4px 8px; font-size: 14px; border: 1px solid #ccc;
    border-radius: 4px; min-width: 280px;
  }}
  main {{ padding: 24px 32px; max-width: 1400px; margin: 0 auto; }}
  .cluster-header {{
    background: white; border-radius: 10px; padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .cluster-header h2 {{ margin: 0 0 8px 0; font-size: 24px; }}
  .cluster-header .summary {{ font-size: 15px; color: #555; line-height: 1.6; }}
  .cluster-header .summary b {{ color: #222; }}
  .keywords {{
    background: #fff8e1; border-radius: 6px; padding: 8px 12px;
    font-family: ui-monospace, Menlo, monospace; font-size: 13px;
    margin-top: 8px; display: inline-block;
  }}
  .section {{
    background: white; border-radius: 10px; padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .section h3 {{ margin: 0 0 12px 0; font-size: 16px;
                  text-transform: uppercase; letter-spacing: 0.5px;
                  color: #666; font-weight: 600; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
  }}
  .grid figure {{
    margin: 0; text-align: center; background: #f0f0f0;
    border-radius: 6px; overflow: hidden;
  }}
  .grid img {{
    width: 100%; height: 130px; object-fit: contain; background: white;
    display: block;
  }}
  .grid figcaption {{
    padding: 4px 6px; font-size: 11px; color: #555;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .grid figure[data-noimage="1"] {{
    height: 160px; display: flex; align-items: center; justify-content: center;
    color: #999; font-size: 11px;
  }}
  .samples {{ display: flex; flex-direction: column; gap: 12px; }}
  .sample {{
    background: #fafafa; border-left: 3px solid #3182bd;
    padding: 10px 14px; border-radius: 0 6px 6px 0;
  }}
  .sample .gid {{ font-family: ui-monospace, Menlo, monospace;
                   font-size: 11px; color: #888; }}
  .sample .type {{ display: inline-block; background: #e3f2fd;
                    color: #1565c0; padding: 1px 6px; border-radius: 3px;
                    font-size: 11px; margin-left: 6px; }}
  .sample .desc {{ font-size: 13px; line-height: 1.5; margin-top: 6px;
                    color: #333; }}
  .mix {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .mix .chip {{
    background: #f0f0f0; padding: 3px 10px; border-radius: 12px;
    font-size: 12px;
  }}
  .mix .chip b {{ color: #333; }}
  .dominant {{ background: #c8e6c9 !important; }}
  kbd {{ background: #eee; border: 1px solid #ccc; border-radius: 3px;
          padding: 1px 5px; font-family: ui-monospace, Menlo, monospace;
          font-size: 11px; }}
</style>
</head>
<body>

<header>
  <h1>Winner Cluster Demo — F5_varbal-umap4 (SigLIP, geo+text+visual)</h1>
  <div class="stats">macro purity = 0.853</div>
</header>

<div class="controls">
  <button id="prevBtn">← Prev</button>
  <span class="position">Cluster <input id="posInput" type="number" min="1" value="1"> / <span id="totalLabel">?</span></span>
  <button id="nextBtn">Next →</button>
  <select id="jumpSelect"></select>
  <span style="color:#888; font-size:13px; margin-left:auto;">
    <kbd>←</kbd> <kbd>→</kbd> to navigate
  </span>
</div>

<main id="content"></main>

<script>
const CLUSTERS = {data_json};
let pos = 0;

function escape(s) {{ return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}

function render() {{
  const c = CLUSTERS[pos];
  const main = document.getElementById('content');
  const memberHtml = c.members.map(m => {{
    if (m.thumb) return `<figure><img loading="lazy" src="${{escape(m.thumb)}}" alt="${{escape(m.gid)}}"><figcaption>${{escape(m.type.replace('Ifc',''))}}</figcaption></figure>`;
    return `<figure data-noimage="1"><figcaption>[no render]<br>${{escape(m.type)}}</figcaption></figure>`;
  }}).join('');
  const samplesHtml = c.samples.map(s => `
    <div class="sample">
      <span class="gid">${{escape(s.gid)}}</span>
      <span class="type">${{escape(s.type)}}</span>
      <div class="desc">${{escape(s.description)}}</div>
    </div>
  `).join('');
  const mixHtml = c.type_mix.map(t => {{
    const cls = t.type === c.top_type ? 'chip dominant' : 'chip';
    return `<span class="${{cls}}"><b>${{escape(t.type.replace('Ifc',''))}}</b> ×${{t.n}}</span>`;
  }}).join('');

  main.innerHTML = `
    <div class="cluster-header">
      <h2>Cluster ${{c.cluster_id}} <span style="color:#888;font-weight:400;font-size:18px;">(${{c.size}} members)</span></h2>
      <div class="summary">
        Dominant type: <b>${{escape(c.top_type)}}</b> · purity: <b>${{(c.purity*100).toFixed(1)}}%</b>
      </div>
      <div class="keywords">${{escape(c.top_keywords.join(' · '))}}</div>
    </div>
    <div class="section">
      <h3>Type composition</h3>
      <div class="mix">${{mixHtml}}</div>
    </div>
    <div class="section">
      <h3>Sample descriptions (Gemini v7)</h3>
      <div class="samples">${{samplesHtml}}</div>
    </div>
    <div class="section">
      <h3>Members (${{c.members.length}} shown, sorted by closeness to cluster center)</h3>
      <div class="grid">${{memberHtml}}</div>
    </div>
  `;

  document.getElementById('posInput').value = pos + 1;
  document.getElementById('prevBtn').disabled = pos === 0;
  document.getElementById('nextBtn').disabled = pos === CLUSTERS.length - 1;
  document.getElementById('jumpSelect').value = pos;
  window.scrollTo(0, 0);
}}

function goTo(i) {{
  pos = Math.max(0, Math.min(CLUSTERS.length - 1, i));
  render();
}}

document.addEventListener('DOMContentLoaded', () => {{
  document.getElementById('totalLabel').textContent = CLUSTERS.length;
  const sel = document.getElementById('jumpSelect');
  CLUSTERS.forEach((c, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `#${{c.cluster_id}} — ${{c.top_type}} (${{c.size}}, ${{(c.purity*100).toFixed(0)}}%)`;
    sel.appendChild(opt);
  }});
  sel.addEventListener('change', e => goTo(parseInt(e.target.value)));
  document.getElementById('prevBtn').onclick = () => goTo(pos - 1);
  document.getElementById('nextBtn').onclick = () => goTo(pos + 1);
  document.getElementById('posInput').addEventListener('change', e => {{
    goTo(parseInt(e.target.value) - 1);
  }});
  document.addEventListener('keydown', e => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') goTo(pos - 1);
    if (e.key === 'ArrowRight') goTo(pos + 1);
  }});
  render();
}});
</script>

</body>
</html>"""


if __name__ == '__main__':
    main()
