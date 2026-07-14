"""P2c — Full fusion sweep across all encoders and 9 fusion strategies.

Extends P2_fusion.py + _run_f1a_umap.py with:
  * Visual encoders: all 7 variants (DINOv3 col/colorless, DuoDuo col/colorless,
                     SigLIP col/colorless, Gemini col)
  * Text version: v7 (single)
  * Subsets: geo+text, geo+visual, geo+text+visual
  * Algorithm: HDBSCAN only (60-config factorial grid)

Fusion strategies (reduced-dim only — F2/F3/F3w/F6 dropped for wall-time):
  F1a       per-modality PCA, then concat with raw geo. PCA dims {8,16,32,64,128}
  F1a_umap  per-modality UMAP, then concat with raw geo. UMAP dims {4,8,16,64}
  F1b       std-concat all blocks then PCA. PCA dims {8,16,32,64,128}
  F5        std-concat then UMAP. UMAP dims {4,8,16,64}
  F7        CCA between (geo) and (concat of all other blocks). dims {8,16,32}

Raw-concat strategies (F2 raw, F3 var-balanced, F3w weighted, F6 L2-norm) are
in the legacy fusion.csv at d=777-1801 and dominated the wall-clock; we keep
them as legacy and focus this sweep on reduced-dim recipes.

Outputs:
    results/midterm/fusion/fusion_full.csv
    results/midterm/fusion/fusion_full_per_type.csv
"""
from __future__ import annotations
import os, tempfile, sys, time
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual
from _reducers import ReducerConfig, UMAP_PARAMS_GEOFUS, UMAP_PARAMS_TEXTVIS, fit_reducer
from _sweep_runner import HDBSCAN_GRID_FACTORIAL, run_sweep, save_sweep


OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'fusion'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_VERSION = 'v7'
TEXT_AGG = 'single'

# Visual encoder variants to sweep — colored-only (4 total)
VISUAL_VARIANTS = [
    ('dinov3', 'colored'),
    ('duoduo', 'colored'),
    ('siglip', 'colored'),
    ('gemini', 'colored'),
]

PCA_DIMS = [8, 16, 32, 64, 128]
UMAP_DIMS = [4, 8, 16, 64]
CCA_DIMS = [8, 16, 32]
WEIGHT_SCHEMES = {
    'equal':      {'geo': 1.0, 'text': 1.0, 'visual': 1.0},
    'geo_heavy':  {'geo': 2.0, 'text': 1.0, 'visual': 1.0},
    'embed_heavy':{'geo': 1.0, 'text': 2.0, 'visual': 2.0},
}


# ── primitives ──────────────────────────────────────────────────────────────
def _std(X):
    return StandardScaler().fit_transform(X).astype(np.float32)


def _pca(X, k):
    p = PCA(n_components=k, random_state=42).fit(X)
    return p.transform(X).astype(np.float32), float(p.explained_variance_ratio_.sum())


def _weighted(block: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """Variance-balanced (so each block contributes equal variance) × weight."""
    s = StandardScaler().fit_transform(block)
    total_var = s.var(axis=0, ddof=0).sum()
    if total_var > 1e-9:
        s = s / np.sqrt(total_var)
    return (s * weight).astype(np.float32)


def _l2_rows(block: np.ndarray) -> np.ndarray:
    """L2-normalize each row of the block."""
    return normalize(block, norm='l2', axis=1).astype(np.float32)


# ── fusion strategies ───────────────────────────────────────────────────────
def build_F1a(blocks, k):
    """Per-modality PCA → std → concat with raw geo (std)."""
    parts, meta = [], {}
    if 'geo' in blocks:
        parts.append(_std(blocks['geo']))
    for name in ('text', 'visual'):
        if name in blocks:
            Xk, var = _pca(blocks[name], k)
            parts.append(_std(Xk))
            meta[f'{name}_var_ret'] = round(var, 4)
    return np.concatenate(parts, axis=1), meta


def build_F1a_umap(blocks, k, modality_tag):
    """Per-modality UMAP → std → concat with raw geo (std)."""
    parts, meta = [], {}
    if 'geo' in blocks:
        parts.append(_std(blocks['geo']))
    for name in ('text', 'visual'):
        if name in blocks:
            cfg = ReducerConfig(f'umap_{k}', 'umap', k, dict(UMAP_PARAMS_TEXTVIS))
            Xr, m = fit_reducer(cfg, blocks[name], modality_name=f'{modality_tag}_{name}')
            parts.append(_std(Xr))
            meta[f'{name}_umap_fit_s'] = round(m['fit_seconds'], 2)
            meta[f'{name}_umap_cache_hit'] = m.get('cache_hit', False)
    return np.concatenate(parts, axis=1), meta


def build_F1b(blocks, k):
    """std-concat all blocks, then PCA to k."""
    concat = np.concatenate([_std(b) for b in blocks.values()], axis=1)
    Xr, var = _pca(concat, k)
    return Xr, {'concat_pca_var_ret': round(var, 4), 'concat_in_dim': int(concat.shape[1])}


def build_F1b_varbal(blocks, k):
    """Variance-balance per block → concat → PCA to k.

    Unlike F1b, each block contributes equal *total* variance to the concat,
    so PCA isn't dominated by the highest-dim block.
    """
    concat = np.concatenate([_weighted(b, weight=1.0) for b in blocks.values()], axis=1)
    Xr, var = _pca(concat, k)
    return Xr, {'concat_pca_var_ret': round(var, 4), 'concat_in_dim': int(concat.shape[1])}


def build_F2(blocks):
    """Raw std-concat (no reduction)."""
    return np.concatenate([_std(b) for b in blocks.values()], axis=1), {}


def build_F3(blocks, weights=None):
    """Variance-balanced std-concat, optionally weighted per modality."""
    weights = weights or {}
    parts = [
        _weighted(b, weight=weights.get(name, 1.0))
        for name, b in blocks.items()
    ]
    return np.concatenate(parts, axis=1), {}


def build_F5(blocks, k, modality_tag):
    """std-concat then UMAP."""
    concat = np.concatenate([_std(b) for b in blocks.values()], axis=1)
    cfg = ReducerConfig(f'umap_{k}', 'umap', k, dict(UMAP_PARAMS_GEOFUS))
    Xr, m = fit_reducer(cfg, concat, modality_name=modality_tag)
    return Xr, {'umap_fit_s': round(m['fit_seconds'], 2),
                'umap_cache_hit': m.get('cache_hit', False)}


def build_F5_varbal(blocks, k, modality_tag):
    """Variance-balance per block → concat → UMAP."""
    concat = np.concatenate([_weighted(b, weight=1.0) for b in blocks.values()], axis=1)
    cfg = ReducerConfig(f'umap_{k}', 'umap', k, dict(UMAP_PARAMS_GEOFUS))
    Xr, m = fit_reducer(cfg, concat, modality_name=modality_tag)
    return Xr, {'umap_fit_s': round(m['fit_seconds'], 2),
                'umap_cache_hit': m.get('cache_hit', False)}


def build_F6(blocks):
    """L2-normalize each block (row-wise), concat."""
    return np.concatenate([_l2_rows(b) for b in blocks.values()], axis=1), {}


def build_F7(blocks, k):
    """CCA between geo (view A) and concat-of-other-modalities (view B).

    Returns [X_a | X_b] where X_a, X_b are the projected views (each k-d).
    Output dim = 2k.

    For bimodal subsets (geo+text or geo+visual), B is just text or visual.
    For trimodal (geo+text+visual), B is the std-concat of text+visual.
    """
    if 'geo' not in blocks:
        raise ValueError('F7 expects geo as one view')
    A = _std(blocks['geo'])
    other_parts = [_std(blocks[n]) for n in ('text', 'visual') if n in blocks]
    if not other_parts:
        raise ValueError('F7 expects at least one non-geo block')
    B = np.concatenate(other_parts, axis=1)

    # CCA needs k <= min(d_a, d_b)
    k_eff = min(k, A.shape[1] - 1, B.shape[1] - 1)
    if k_eff < 1:
        raise ValueError(f'CCA cannot fit k={k} with d_a={A.shape[1]} d_b={B.shape[1]}')

    cca = CCA(n_components=k_eff, max_iter=500)
    Xa, Xb = cca.fit_transform(A, B)
    fused = np.concatenate([Xa, Xb], axis=1).astype(np.float32)
    return fused, {'cca_k_requested': k, 'cca_k_effective': k_eff}


# ── sweep wrapper ───────────────────────────────────────────────────────────
def _sweep(name, X, y, extra):
    return run_sweep(
        name=name, X=X, y=y,
        algorithms=['hdbscan'],
        scale=True,
        hdbscan_grid=HDBSCAN_GRID_FACTORIAL,
        extra_meta=extra,
        verbose=False,
    )


# ── main ────────────────────────────────────────────────────────────────────
def run_for_visual(visual_encoder, visual_variant):
    """Run all 9 strategies × 3 subsets for one visual encoder."""
    print(f'\n{"="*70}')
    print(f'[P2c] Visual encoder = {visual_encoder}_{visual_variant}')
    print(f'{"="*70}')

    df, y_full = load_data()
    gids = df['GlobalId'].values
    geo = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    X_txt_raw, txt_idx = load_text(gids, TEXT_VERSION, TEXT_AGG)
    X_vis_raw, vis_idx = load_visual(gids, visual_encoder, visual_variant)
    common = np.intersect1d(txt_idx, vis_idx)
    print(f'  aligned: {len(common)} / {len(gids)} | '
          f'geo={geo.shape[1]} text={X_txt_raw.shape[1]} visual={X_vis_raw.shape[1]}')

    geo_b = geo[common]
    txt_b = X_txt_raw[np.searchsorted(txt_idx, common)]
    vis_b = X_vis_raw[np.searchsorted(vis_idx, common)]
    y = y_full[common]

    encoder_tag = f'{visual_encoder}_{visual_variant}'

    SUBSETS = [
        ('geo+text', {'geo': geo_b, 'text': txt_b}),
        ('geo+visual', {'geo': geo_b, 'visual': vis_b}),
        ('geo+text+visual', {'geo': geo_b, 'text': txt_b, 'visual': vis_b}),
    ]

    all_summary, all_per_type = [], []

    for subset_label, blocks in SUBSETS:
        # For geo+text we don't depend on visual encoder — only run once (skip on repeats)
        skip_geo_text = subset_label == 'geo+text' and \
                        (visual_encoder, visual_variant) != VISUAL_VARIANTS[0]
        if skip_geo_text:
            continue
        print(f'\n  -- subset: {subset_label} --')
        modality_tag = f'fusion_{subset_label.replace("+","_")}_{encoder_tag}'

        base_extra = {
            'subset': subset_label,
            'text_version': TEXT_VERSION,
            'visual_encoder': encoder_tag if subset_label != 'geo+text' else 'n/a',
        }

        # F1a: per-block PCA → concat with geo
        for k in PCA_DIMS:
            X, meta = build_F1a(blocks, k)
            extra = {**base_extra, 'recipe': 'F1a',
                     'variant': f'pca_{k}_then_concat_geo',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]), **meta}
            s, pt = _sweep(f'{subset_label}__F1a_pca{k}', X, y, extra)
            all_summary.append(s); all_per_type.append(pt)
            print(f'    F1a pca_{k:<3d}     d={X.shape[1]:4d}')

        # F1a_umap
        for k in UMAP_DIMS:
            X, meta = build_F1a_umap(blocks, k, modality_tag)
            extra = {**base_extra, 'recipe': 'F1a_umap',
                     'variant': f'umap_{k}',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]), **meta}
            s, pt = _sweep(f'{subset_label}__F1a_umap_{k}', X, y, extra)
            all_summary.append(s); all_per_type.append(pt)
            print(f'    F1a umap_{k:<3d}    d={X.shape[1]:4d}')

        # F1b: PCA-on-concat (std-scaled blocks)
        for k in PCA_DIMS:
            X, meta = build_F1b(blocks, k)
            extra = {**base_extra, 'recipe': 'F1b',
                     'variant': f'pca_{k}_on_concat',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]), **meta}
            s, pt = _sweep(f'{subset_label}__F1b_pca{k}', X, y, extra)
            all_summary.append(s); all_per_type.append(pt)
            print(f'    F1b pca_{k:<3d}        d={X.shape[1]:4d}')

        # F1b_varbal: variance-balance then PCA-on-concat (NEW)
        for k in PCA_DIMS:
            X, meta = build_F1b_varbal(blocks, k)
            extra = {**base_extra, 'recipe': 'F1b_varbal',
                     'variant': f'varbal_pca_{k}_on_concat',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]), **meta}
            s, pt = _sweep(f'{subset_label}__F1b_vb_pca{k}', X, y, extra)
            all_summary.append(s); all_per_type.append(pt)
            print(f'    F1b_varbal pca_{k:<3d} d={X.shape[1]:4d}')

        # F5: UMAP on concat (std-scaled blocks)
        for k in UMAP_DIMS:
            X, meta = build_F5(blocks, k, modality_tag + '_F5')
            extra = {**base_extra, 'recipe': 'F5',
                     'variant': f'umap_{k}_on_concat',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]), **meta}
            s, pt = _sweep(f'{subset_label}__F5_umap{k}', X, y, extra)
            all_summary.append(s); all_per_type.append(pt)
            print(f'    F5 umap_{k:<3d}        d={X.shape[1]:4d}')

        # F5_varbal: variance-balance then UMAP-on-concat (NEW)
        for k in UMAP_DIMS:
            X, meta = build_F5_varbal(blocks, k, modality_tag + '_F5_vb')
            extra = {**base_extra, 'recipe': 'F5_varbal',
                     'variant': f'varbal_umap_{k}_on_concat',
                     'reducer_dim': k, 'out_dim': int(X.shape[1]), **meta}
            s, pt = _sweep(f'{subset_label}__F5_vb_umap{k}', X, y, extra)
            all_summary.append(s); all_per_type.append(pt)
            print(f'    F5_varbal umap_{k:<3d}  d={X.shape[1]:4d}')

        # F7: CCA (NEW)
        for k in CCA_DIMS:
            try:
                X, meta = build_F7(blocks, k)
                extra = {**base_extra, 'recipe': 'F7',
                         'variant': f'cca_{k}',
                         'reducer_dim': k, 'out_dim': int(X.shape[1]), **meta}
                s, pt = _sweep(f'{subset_label}__F7_cca{k}', X, y, extra)
                all_summary.append(s); all_per_type.append(pt)
                print(f'    F7 cca_{k:<3d}      d={X.shape[1]:4d}')
            except Exception as e:
                print(f'    F7 cca_{k:<3d}      FAILED: {e}')

    return all_summary, all_per_type


def main():
    t_start = time.perf_counter()
    all_summary, all_per_type = [], []

    for enc, var in VISUAL_VARIANTS:
        s_list, pt_list = run_for_visual(enc, var)
        all_summary.extend(s_list)
        all_per_type.extend(pt_list)

    # Unique run_ids across cells
    offset = 0
    for s, pt in zip(all_summary, all_per_type):
        s['run_id'] = s['run_id'].to_numpy() + offset
        pt['run_id'] = pt['run_id'].to_numpy() + offset
        offset = int(s['run_id'].max()) + 1
    summary = pd.concat(all_summary, ignore_index=True)
    per_type = pd.concat(all_per_type, ignore_index=True)

    save_sweep(
        summary, per_type,
        summary_csv=OUT_DIR / 'fusion_full.csv',
        per_type_csv=OUT_DIR / 'fusion_full_per_type.csv',
    )

    elapsed = time.perf_counter() - t_start
    print(f'\n[P2c] DONE in {elapsed/60:.1f} min')

    # Quick summary: best per recipe overall
    print('\n[P2c] Best per recipe (across all encoders, subsets):')
    for r in sorted(summary['recipe'].unique()):
        sub = summary[summary['recipe'] == r]
        b = sub.loc[sub['macro_purity'].idxmax()]
        print(f'  {r:10s} macro={b["macro_purity"]:.3f} | '
              f'subset={b["subset"]} | variant={b["variant"]} | '
              f'enc={b["visual_encoder"]} | k={int(b["k"])}')


if __name__ == '__main__':
    main()
