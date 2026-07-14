"""P3c — Supervised ceiling across every fusion strategy we tried.

For each fusion strategy, pick the winning (encoder, subset, dim) by
unsupervised macro purity in fusion_full.csv, rebuild that exact
representation, and run RF 5-fold CV with 5 seeds (same protocol as P3b).

Also includes raw-concat strategies (F2, F3-equal, F3w-geo_heavy,
F3w-embed_heavy, F6) which weren't in the reduced-dim sweep but are
legitimate strategies — for these we pin to trimodal + siglip_colored
(the dominant winning combination across reduced-dim recipes).

Output:
    results/midterm/supervised/all_strategies_ceiling.csv
    results/midterm/supervised/all_strategies_ceiling_per_type.csv
"""
from __future__ import annotations
import json, os, sys, tempfile
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))
sys.path.insert(0, str(_EXP_DIR))

import P2c_fusion_full as P
from _common import BASELINE, RESULTS_ROOT, build_X, cumulative_bounded_up_to, load_data
from _loaders import load_text, load_visual

OUT_DIR = RESULTS_ROOT.parent / 'midterm' / 'supervised'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FUSION_CSV = RESULTS_ROOT.parent / 'midterm' / 'fusion' / 'fusion_full.csv'

SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 5


def _rf_cv(X, y, seed):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y, dtype=object)
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(
            n_estimators=400, random_state=seed,
            class_weight='balanced', n_jobs=1,
        )
        rf.fit(X[tr], y[tr])
        y_pred[te] = rf.predict(X[te])
    return y_pred


def _metrics(label, seed, y, y_pred, meta):
    types = sorted(set(y.tolist()))
    acc = float((y_pred == y).mean())
    recalls = {t: float((y_pred[y == t] == t).mean()) if (y == t).sum() else 0.0
               for t in types}
    macro = float(np.mean(list(recalls.values())))
    row = dict(strategy=label, seed=seed, accuracy=acc, macro_recall=macro, **meta)
    return row, recalls


def _subset_blocks(geo, txt, vis, subset):
    if subset == 'geo+text':
        return {'geo': geo, 'text': txt}
    if subset == 'geo+visual':
        return {'geo': geo, 'visual': vis}
    if subset == 'geo+text+visual':
        return {'geo': geo, 'text': txt, 'visual': vis}
    raise ValueError(subset)


def _build_rep(recipe, subset, dim, blocks, tag):
    if recipe == 'F1a':
        return P.build_F1a(blocks, dim)
    if recipe == 'F1a_umap':
        return P.build_F1a_umap(blocks, dim, tag)
    if recipe == 'F1b':
        return P.build_F1b(blocks, dim)
    if recipe == 'F1b_varbal':
        return P.build_F1b_varbal(blocks, dim)
    if recipe == 'F5':
        return P.build_F5(blocks, dim, tag)
    if recipe == 'F5_varbal':
        return P.build_F5_varbal(blocks, dim, tag)
    if recipe == 'F7':
        return P.build_F7(blocks, dim)
    if recipe == 'F2':
        return P.build_F2(blocks)
    if recipe == 'F3':
        return P.build_F3(blocks, weights=P.WEIGHT_SCHEMES['equal'])
    if recipe == 'F3w_geo':
        return P.build_F3(blocks, weights=P.WEIGHT_SCHEMES['geo_heavy'])
    if recipe == 'F3w_embed':
        return P.build_F3(blocks, weights=P.WEIGHT_SCHEMES['embed_heavy'])
    if recipe == 'F6':
        return P.build_F6(blocks)
    raise ValueError(recipe)


def winners_from_fusion_full() -> list[dict]:
    """Pick best (subset, encoder, dim, unsup macro_purity) per recipe."""
    df = pd.read_csv(FUSION_CSV)
    rows = []
    for recipe in sorted(df['recipe'].unique()):
        sub = df[df['recipe'] == recipe]
        best = sub.sort_values('macro_purity', ascending=False).iloc[0]
        rows.append(dict(
            recipe=recipe,
            subset=best['subset'],
            encoder=best['visual_encoder'] if isinstance(best['visual_encoder'], str)
                     else 'none',
            dim=int(best['reducer_dim']) if not pd.isna(best['reducer_dim']) else None,
            unsup_macro_purity=float(best['macro_purity']),
        ))
    return rows


# Raw-concat strategies (not in fusion_full.csv). Pin to trimodal + siglip_colored
# since SigLIP_colored dominates reduced-dim winners.
RAW_CONCAT_CONFIGS = [
    dict(recipe='F2',         subset='geo+text+visual', encoder='siglip_colored', dim=None, unsup_macro_purity=None),
    dict(recipe='F3',         subset='geo+text+visual', encoder='siglip_colored', dim=None, unsup_macro_purity=None),
    dict(recipe='F3w_geo',    subset='geo+text+visual', encoder='siglip_colored', dim=None, unsup_macro_purity=None),
    dict(recipe='F3w_embed',  subset='geo+text+visual', encoder='siglip_colored', dim=None, unsup_macro_purity=None),
    dict(recipe='F6',         subset='geo+text+visual', encoder='siglip_colored', dim=None, unsup_macro_purity=None),
]


def main():
    print(f'Loading data...')
    df, y_full = load_data()
    gids_all = df['GlobalId'].values
    geo_full = build_X(df, BASELINE, cumulative_bounded_up_to('04_horiz_frac'))

    # Cache loaded visual encoders once per encoder (avoid repeated loads)
    encoder_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    text_v7 = load_text(gids_all, 'v7', 'single')

    def get_visual(encoder_key):
        if encoder_key == 'none':
            return None
        if encoder_key not in encoder_cache:
            name, variant = encoder_key.split('_', 1)
            encoder_cache[encoder_key] = load_visual(gids_all, name, variant)
        return encoder_cache[encoder_key]

    # Compose the list of (recipe-config) to evaluate
    configs = winners_from_fusion_full() + RAW_CONCAT_CONFIGS
    print(f'\nEvaluating {len(configs)} strategy configs:')
    for c in configs:
        print(f"  {c['recipe']:12s}  {c['subset']:18s}  enc={c['encoder']:18s}  "
              f"dim={c['dim']}  unsup_macro={c['unsup_macro_purity']}")

    summary_rows, per_type_rows = [], []

    for cfg in configs:
        recipe, subset, encoder, dim = cfg['recipe'], cfg['subset'], cfg['encoder'], cfg['dim']
        unsup = cfg['unsup_macro_purity']

        X_txt, txt_idx = text_v7
        if 'visual' in subset:
            X_vis, vis_idx = get_visual(encoder)
            common = np.intersect1d(txt_idx, vis_idx) if 'text' in subset else vis_idx
        else:
            common = txt_idx

        # Intersect with all-data
        if 'text' in subset and 'visual' in subset:
            common = np.intersect1d(txt_idx, vis_idx)
        elif 'text' in subset:
            common = txt_idx
        elif 'visual' in subset:
            common = vis_idx
        else:
            common = gids_all

        geo_b = geo_full[common].astype(np.float64)
        txt_b = (X_txt[np.searchsorted(txt_idx, common)].astype(np.float64)
                 if 'text' in subset else None)
        vis_b = (X_vis[np.searchsorted(vis_idx, common)].astype(np.float64)
                 if 'visual' in subset else None)
        y = y_full[common]

        blocks = _subset_blocks(geo_b, txt_b, vis_b, subset)

        tag = f'sup_{recipe}_{subset}_{encoder}_{dim}'
        print(f'\n=== {recipe}  {subset}  {encoder}  dim={dim} ===')
        X, meta = _build_rep(recipe, subset, dim, blocks, tag)
        print(f'  shape={X.shape}  meta={meta}')

        Xs = StandardScaler().fit_transform(X)
        label = f'{recipe}|{subset}|{encoder}|d={dim or X.shape[1]}'
        for seed in SEEDS:
            y_pred = _rf_cv(Xs, y, seed)
            row, recalls = _metrics(label, seed, y, y_pred,
                                    meta=dict(recipe=recipe, subset=subset,
                                              encoder=encoder, dim=dim or X.shape[1],
                                              out_dim=X.shape[1],
                                              unsup_macro_purity=unsup))
            summary_rows.append(row)
            for t, r in recalls.items():
                per_type_rows.append(dict(strategy=label, recipe=recipe,
                                          subset=subset, encoder=encoder,
                                          seed=seed, type=t, recall=r,
                                          support=int((y == t).sum())))
            print(f'  seed={seed}  acc={row["accuracy"]:.4f}  '
                  f'macro_recall={row["macro_recall"]:.4f}')

    summary = pd.DataFrame(summary_rows)
    per_type = pd.DataFrame(per_type_rows)
    summary.to_csv(OUT_DIR / 'all_strategies_ceiling.csv', index=False)
    per_type.to_csv(OUT_DIR / 'all_strategies_ceiling_per_type.csv', index=False)
    print(f'\nSaved {len(summary)} summary rows → '
          f'{OUT_DIR / "all_strategies_ceiling.csv"}')

    print('\n=== Supervised ceiling per strategy (mean ± std over 5 seeds) ===')
    agg = (summary.groupby('strategy')[['accuracy', 'macro_recall']]
           .agg(['mean', 'std']).round(4))
    agg_sorted = agg.sort_values(('macro_recall', 'mean'), ascending=False)
    print(agg_sorted.to_string())

    # Side-by-side: unsup vs sup
    print('\n=== Unsupervised vs Supervised ===')
    sup_mean = summary.groupby('strategy')['macro_recall'].mean()
    cmp_rows = []
    for label, m in sup_mean.items():
        u = summary[summary['strategy'] == label]['unsup_macro_purity'].iloc[0]
        cmp_rows.append(dict(strategy=label, unsup_macro_purity=u,
                             sup_macro_recall=round(m, 4),
                             gap=round((m - u) if u is not None and not pd.isna(u) else float('nan'), 4)))
    cmp_df = pd.DataFrame(cmp_rows).sort_values('sup_macro_recall', ascending=False)
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(OUT_DIR / 'all_strategies_unsup_vs_sup.csv', index=False)


if __name__ == '__main__':
    main()
