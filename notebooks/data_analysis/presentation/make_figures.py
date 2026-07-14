"""Regenerate every midterm-presentation figure from persisted CSVs.

Reads CSVs under results/midterm/ and writes PNGs to presentation/figures/.
Each figure is designed for 16:9 slides with large fonts.

Run:
    python notebooks/data_analysis/presentation/make_figures.py
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

_EXP_DIR = Path(__file__).resolve().parent.parent / 'feature_engineering'
sys.path.insert(0, str(_EXP_DIR))
from _common import RESULTS_ROOT

MIDTERM = RESULTS_ROOT.parent / 'midterm'
FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out}')


def _short(t):
    return str(t).replace('Ifc', '')


# ── Figure 00: dataset composition ────────────────────────────────────────
def fig_00_dataset():
    """Horizontal bar chart: 17 merged types, counts, merged groups highlighted."""
    _EXP2 = Path(__file__).resolve().parent.parent / 'feature_engineering'
    sys.path.insert(0, str(_EXP2))
    from _common import load_data, TYPE_MAP

    df, ifc_merged = load_data()
    counts = pd.Series(ifc_merged).value_counts().sort_values()

    MERGED_GROUPS = {'Wall', 'Furniture', 'MEP', 'Stair'}
    colors = ['#2a9d8f' if t in MERGED_GROUPS else '#3182bd' for t in counts.index]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ypos = np.arange(len(counts))
    ax.barh(ypos, counts.values, color=colors, edgecolor='white')
    ax.set_yticks(ypos)
    ax.set_yticklabels([_short(t) for t in counts.index], fontsize=11)
    ax.set_xlabel('Number of objects')
    ax.set_title(
        f'Annotated subset: {len(df)} objects, {len(counts)} merged types\n'
        f'(from 1,849 annotated across 158 / 202 student submissions)',
    )
    for i, v in enumerate(counts.values):
        ax.text(v + 3, i, str(v), va='center', fontsize=10)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color='#2a9d8f', label='Merged group (2-3 raw types)'),
                 Patch(color='#3182bd', label='Single IFC type')],
        loc='lower right', fontsize=10,
    )
    _save(fig, '00_dataset.png')


# ── Figure 01: motivation ──────────────────────────────────────────────────
def fig_01_motivation():
    # The 01_orientation stage's per-type CSV has column 'prev' = baseline
    # per-type purity at k=32 (seed=42); use that as the baseline-only source.
    fe = pd.read_csv(RESULTS_ROOT / '01_orientation' / 'per_type_delta_k32.csv')
    worst = fe.sort_values('prev').head(6)[['type', 'prev']].rename(
        columns={'prev': 'baseline_purity'},
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    types = worst['type'].tolist()
    values = worst['baseline_purity'].to_numpy()
    ypos = np.arange(len(types))
    ax.barh(ypos, values, color='#d94a3f', edgecolor='white')
    ax.set_yticks(ypos)
    ax.set_yticklabels(types)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Per-type purity (k=32 k-means, baseline 4 features)')
    ax.set_title('The problem: 4 OBB size features don\'t separate these types')
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
    _save(fig, '01_motivation.png')


# ── Figure 02: feature progression ─────────────────────────────────────────
def fig_02_feature_progression():
    prog = pd.read_csv(RESULTS_ROOT / 'progression_summary.csv')
    k32 = prog[prog['k'] == 32]
    stages = list(k32['config'])
    pur = k32['purity_mean'].values
    pur_std = k32['purity_std'].values

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(stages)), pur, yerr=pur_std,
                  color=['#777'] + ['#2a9d8f'] * (len(stages) - 1),
                  edgecolor='white', capsize=4)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.split('(')[0].strip() for s in stages], rotation=20,
                       ha='right')
    ax.set_ylabel('Purity @ k=32 (10 seeds)')
    ax.set_ylim(0, 0.85)
    ax.set_title(f'Geometric feature engineering: +{(pur[-1]-pur[0])/pur[0]:.0%} '
                 f'relative purity from 4 → 9 features')
    for bar, v in zip(bars, pur):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f'{v:.3f}', ha='center', fontsize=11)
    _save(fig, '02_feature_progression.png')


# ── Figure 03: algorithm comparison on geo + k-matched on fusion ──────────
def fig_03_algorithm_comparison():
    df = pd.read_csv(MIDTERM / 'sweeps' / 'geo.csv')
    best_at_k32 = (df[df['hp_k'] == 32]
                    .groupby('algo')[['purity', 'macro_purity']]
                    .mean().reset_index())
    hdb = df[df['algo'] == 'hdbscan']
    hdb_best = hdb.sort_values('macro_purity', ascending=False).head(1)
    hdb_row = pd.DataFrame({
        'algo': ['hdbscan'],
        'purity': hdb_best['purity'].values,
        'macro_purity': hdb_best['macro_purity'].values,
    })
    combined = pd.concat([
        best_at_k32[best_at_k32['algo'] != 'hdbscan'], hdb_row,
    ], ignore_index=True)
    hdb_small = hdb[hdb['k'] <= 40]
    if not hdb_small.empty:
        hdb_sm_best = hdb_small.sort_values('macro_purity', ascending=False).iloc[:1]
        hdb_sm_row = pd.DataFrame({
            'algo': ['hdbscan_k≤40'],
            'purity': hdb_sm_best['purity'].values,
            'macro_purity': hdb_sm_best['macro_purity'].values,
        })
        combined = pd.concat([combined, hdb_sm_row], ignore_index=True)

    order = ['kmeans', 'bisecting', 'gmm', 'hdbscan_k≤40', 'hdbscan']
    combined = combined.set_index('algo').reindex(order).dropna().reset_index()

    # k-matched comparison data (fusion winner, if available)
    km_csv = MIDTERM / 'k_matched_comparison.csv'
    has_km = km_csv.exists()
    if has_km:
        km_df = pd.read_csv(km_csv)

    fig, axes = plt.subplots(1, 2 if has_km else 1,
                             figsize=(16 if has_km else 11, 5.5),
                             gridspec_kw=dict(width_ratios=[1.1, 1] if has_km else [1]))
    ax = axes[0] if has_km else axes

    x = np.arange(len(combined))
    width = 0.35
    ax.bar(x - width / 2, combined['purity'], width, label='Purity',
           color='#3182bd', edgecolor='white')
    ax.bar(x + width / 2, combined['macro_purity'], width, label='Macro purity',
           color='#e6550d', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(combined['algo'])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel('Score on 17-type eval')
    ax.set_title('Algorithm comparison on 9-d geo features\n'
                 '(k=32 for parametric; best HDBSCAN config)')
    ax.legend(loc='lower right')
    for xi, (p, m) in enumerate(zip(combined['purity'], combined['macro_purity'])):
        ax.text(xi - width / 2, p + 0.01, f'{p:.2f}', ha='center', fontsize=10)
        ax.text(xi + width / 2, m + 0.01, f'{m:.2f}', ha='center', fontsize=10)

    if has_km:
        axR = axes[1]
        methods, macros, errs = [], [], []
        # k-means k=17
        km17 = km_df[km_df['method'] == 'k-means k=17']
        if not km17.empty:
            methods.append('k-means\nk=17')
            macros.append(km17['macro_purity'].mean())
            errs.append(km17['macro_purity'].std())
        # HDBSCAN force=True
        hf = km_df[km_df['method'] == 'HDBSCAN force=True']
        if not hf.empty:
            methods.append('HDBSCAN\nforce=True')
            macros.append(hf['macro_purity'].values[0])
            errs.append(0)
        # k-means k=102
        km102 = km_df[km_df['method'].str.startswith('k-means k=') & ~km_df['method'].str.contains('k=17')]
        if not km102.empty:
            methods.append(f'k-means\nk={int(km102["k"].iloc[0])}')
            macros.append(km102['macro_purity'].mean())
            errs.append(km102['macro_purity'].std())
        # HDBSCAN force=False (assigned only)
        hfn = km_df[km_df['method'].str.contains('assigned only')]
        if not hfn.empty:
            methods.append('HDBSCAN\nforce=False\n(assigned)')
            macros.append(hfn['macro_purity'].values[0])
            errs.append(0)

        colors = ['#9ecae1', '#e6550d', '#3182bd', '#fdae6b']
        xr = np.arange(len(methods))
        bars = axR.bar(xr, macros, yerr=errs, capsize=4,
                       color=colors[:len(methods)], edgecolor='white')
        axR.set_xticks(xr)
        axR.set_xticklabels(methods, fontsize=10)
        axR.set_ylim(0, 1.0)
        axR.set_ylabel('Macro purity')
        axR.set_title('k-matched comparison on fusion winner\n'
                       '(F1a PCA-8, geo+text+visual)')
        for xi, (v, e) in enumerate(zip(macros, errs)):
            label = f'{v:.3f}' if e == 0 else f'{v:.3f}±{e:.3f}'
            axR.text(xi, v + 0.02 + e, label, ha='center', fontsize=10)

    _save(fig, '03_algorithm_comparison.png')


# ── Figure 04: text version progression ────────────────────────────────────
def fig_04_text_progression():
    df = pd.read_csv(MIDTERM / 'sweeps' / 'text.csv')
    df = df[df['run_type'] == 'version_sweep']
    # For each version, best kmeans@k=32 avg across seeds AND best HDBSCAN
    rows = []
    for v, sub in df.groupby('version'):
        km = sub[(sub['algo'] == 'kmeans') & (sub['hp_k'] == 32)]
        rows.append({
            'version': v, 'algo': 'kmeans k=32',
            'purity': km['purity'].mean(), 'macro': km['macro_purity'].mean(),
        })
        hdb = sub[sub['algo'] == 'hdbscan']
        if not hdb.empty:
            best = hdb.sort_values('macro_purity', ascending=False).iloc[0]
            rows.append({
                'version': v, 'algo': 'HDBSCAN best',
                'purity': best['purity'], 'macro': best['macro_purity'],
            })
    prog = pd.DataFrame(rows)
    # Desired version order
    order = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v6.1', 'v7']
    prog['version'] = pd.Categorical(prog['version'], categories=order, ordered=True)
    prog = prog.sort_values('version')

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for algo, sub in prog.groupby('algo'):
        ax.plot(sub['version'].astype(str), sub['macro'], marker='o',
                label=f'{algo}', linewidth=2, markersize=8)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel('Macro purity on 17 types')
    ax.set_xlabel('Gemini description prompt version')
    ax.set_title('Prompt engineering progression: text-only clustering')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    _save(fig, '04_text_progression.png')


# ── Figure 05: aggregation ablation on v1 ──────────────────────────────────
def fig_05_aggregation_on_v1():
    df = pd.read_csv(MIDTERM / 'sweeps' / 'text.csv')
    df = df[(df['run_type'] == 'aggregation_ablation') & (df['version'] == 'v1')]
    # Best k=32 kmeans + best HDBSCAN per aggregation
    rows = []
    for agg, sub in df.groupby('aggregation'):
        km = sub[(sub['algo'] == 'kmeans') & (sub['hp_k'] == 32)]
        rows.append({
            'aggregation': agg, 'algo': 'kmeans k=32',
            'macro': km['macro_purity'].mean(),
        })
        hdb = sub[sub['algo'] == 'hdbscan']
        if not hdb.empty:
            best = hdb.sort_values('macro_purity', ascending=False).iloc[0]
            rows.append({
                'aggregation': agg, 'algo': 'HDBSCAN best',
                'macro': best['macro_purity'],
            })
    p = pd.DataFrame(rows)
    order = ['single', 'sum', 'concat', 'single_views']
    p['aggregation'] = pd.Categorical(p['aggregation'], categories=order,
                                       ordered=True)
    p = p.sort_values('aggregation')

    pivot = p.pivot(index='aggregation', columns='algo', values='macro')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind='bar', ax=ax, color=['#3182bd', '#e6550d'], edgecolor='white',
               width=0.7)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel('Macro purity (v1 only)')
    ax.set_xlabel('Aggregation of per-mesh Gemini embedding')
    ax.set_title('Aggregation ablation (v1): `single` is within noise of best\n'
                 '`single_views` includes image embeddings — not pure text')
    ax.set_xticklabels(pivot.index, rotation=0)
    for i, col in enumerate(pivot.columns):
        for j, v in enumerate(pivot[col]):
            if not np.isnan(v):
                ax.text(j + (i - 0.5) * 0.35, v + 0.015, f'{v:.2f}',
                        ha='center', fontsize=9)
    _save(fig, '05_aggregation_on_v1.png')


# ── Figure 06: visual encoder comparison ───────────────────────────────────
def fig_06_visual_encoders():
    df = pd.read_csv(MIDTERM / 'sweeps' / 'visual.csv')
    df_col = df[df['variant'] == 'colored']
    rows = []
    for enc, sub in df_col.groupby('encoder'):
        km = sub[(sub['algo'] == 'kmeans') & (sub['hp_k'] == 32)]
        rows.append({
            'encoder': enc, 'algo': 'KMeans k=32',
            'macro': km['macro_purity'].mean(),
        })
        hdb = sub[sub['algo'] == 'hdbscan']
        if not hdb.empty:
            best = hdb.sort_values('macro_purity', ascending=False).iloc[0]
            rows.append({
                'encoder': enc, 'algo': 'HDBSCAN best',
                'macro': best['macro_purity'],
            })
    p = pd.DataFrame(rows)
    name_map = {'dinov3': 'DINOv3', 'duoduo': 'DuoDuo', 'siglip': 'SigLIP', 'gemini': 'Gemini'}
    p['label'] = p['encoder'].map(name_map)
    pivot = p.pivot(index='label', columns='algo', values='macro')
    pivot = pivot.reindex(['DINOv3', 'DuoDuo', 'SigLIP', 'Gemini'])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind='bar', ax=ax, color=['#3182bd', '#e6550d'],
               edgecolor='white', width=0.7)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel('Macro purity on 17 types')
    ax.set_title('Visual encoders: no clear winner')
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_xlabel('')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9, padding=2)
    _save(fig, '06_visual_encoders.png')


# ── Figure 07: fusion ablation (§12.4 recipes) ─────────────────────────────
def fig_07_fusion_ablation():
    """Two panels:
       (L) Best-per-(subset, recipe) bar chart across F1a / F2 / F3 / F5.
       (R) Reducer-dim sweep on the winning subset: F1a vs F5 lines + F2/F3
           horizontal baselines.
    """
    df = pd.read_csv(MIDTERM / 'fusion' / 'fusion.csv')
    hdb = df[df['algo'] == 'hdbscan'].copy()

    subsets = ['geo+text', 'geo+visual', 'geo+text+visual']
    recipes = ['F2', 'F3', 'F1a', 'F5']
    colors = {'F2': '#9ecae1', 'F3': '#6baed6',
              'F1a': '#3182bd', 'F5': '#08519c'}

    best = (hdb.groupby(['subset', 'recipe'])['macro_purity'].max()
              .unstack('recipe'))
    best = best.reindex(index=subsets, columns=recipes)

    # Pick the winning subset for the right-hand sweep
    best_subset = best.max(axis=1).idxmax()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.8),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))

    # Left panel: subset × recipe bars
    best.plot(kind='bar', ax=axL,
              color=[colors[r] for r in recipes],
              edgecolor='white', width=0.82)
    axL.set_ylim(0, max(0.9, best.values.max() * 1.1))
    axL.set_ylabel('Macro purity (HDBSCAN best in grid)')
    axL.set_title('Fusion subset × recipe (best HDBSCAN per cell)')
    axL.set_xticklabels(subsets, rotation=0, ha='center')
    axL.legend(title='recipe', loc='upper left', ncol=4, fontsize=9)
    axL.grid(axis='y', alpha=0.3)

    # Right panel: reducer-dim sweep on best_subset
    sub = hdb[hdb['subset'] == best_subset].copy()
    # F1a: reducer_dim ∈ {8,16,32,64,128}
    f1a = (sub[sub['recipe'] == 'F1a']
              .groupby('reducer_dim')['macro_purity'].max().sort_index())
    f5 = (sub[sub['recipe'] == 'F5']
             .groupby('reducer_dim')['macro_purity'].max().sort_index())
    if not f1a.empty:
        axR.plot(f1a.index, f1a.values, '-o', color=colors['F1a'], lw=2,
                 ms=8, label='F1a (per-block PCA + geo concat)')
    if not f5.empty:
        axR.plot(f5.index, f5.values, '-s', color=colors['F5'], lw=2,
                 ms=8, label='F5 (UMAP on std-concat + geo)')
    f2v = sub[sub['recipe'] == 'F2']['macro_purity'].max()
    f3v = sub[sub['recipe'] == 'F3']['macro_purity'].max()
    if pd.notna(f2v):
        axR.axhline(f2v, color=colors['F2'], ls='--', lw=2,
                    label=f'F2 raw concat ({f2v:.3f})')
    if pd.notna(f3v):
        axR.axhline(f3v, color=colors['F3'], ls=':', lw=2,
                    label=f'F3 var-balanced ({f3v:.3f})')
    axR.set_xscale('log', base=2)
    axR.set_xlabel('reducer dimension')
    axR.set_ylabel('Macro purity')
    axR.set_title(f'Reducer-dim sweep on {best_subset}')
    axR.legend(loc='lower right', fontsize=9)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, '07_fusion_ablation.png')


# ── Figure 08: unsupervised vs supervised ─────────────────────────────────
def fig_08_supervised_gap():
    sup = pd.read_csv(MIDTERM / 'supervised' / 'ceiling.csv')
    sup_mean = sup.groupby('subset')['macro_recall'].mean()

    fus = pd.read_csv(MIDTERM / 'fusion' / 'fusion.csv')
    # Best HDBSCAN macro per subset
    hdb = fus[fus['algo'] == 'hdbscan']
    unsup_best = hdb.groupby('subset')['macro_purity'].max()
    # k-means k=32 mean
    km = fus[(fus['algo'] == 'kmeans') & (fus['hp_k'] == 32)]
    unsup_km = km.groupby('subset')['macro_purity'].mean()

    common = sorted(set(sup_mean.index) & set(unsup_best.index))
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(common))
    w = 0.27
    ax.bar(x - w, [unsup_km.get(s, np.nan) for s in common], w,
           label='k-means k=32 (unsup)', color='#9ecae1', edgecolor='white')
    ax.bar(x, [unsup_best.get(s, np.nan) for s in common], w,
           label='HDBSCAN best (unsup)', color='#3182bd', edgecolor='white')
    ax.bar(x + w, [sup_mean.get(s, np.nan) for s in common], w,
           label='RF 5-fold CV (sup ceiling)', color='#d94a3f', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=20, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Macro score')
    ax.set_title('Unsupervised vs supervised ceiling across representations')
    ax.legend(loc='lower right')
    _save(fig, '08_supervised_gap.png')


# ── Figure 09: per-type matrix ─────────────────────────────────────────────
def fig_09_per_type_matrix():
    """Rows = 17 types, cols = {geo, text, visual, fusion, RF ceiling}."""
    # Unsupervised: use k-means k=32 seed=42 per modality for defensible k
    def _best_per_type(csv, filt=lambda d: d):
        df = pd.read_csv(csv)
        df = filt(df)
        return df

    geo = pd.read_csv(MIDTERM / 'sweeps' / 'geo_per_type.csv')
    geo = geo[geo['algo'] == 'kmeans']
    # average over seeds for k=32 — need to parse hps
    geo['k'] = geo['hps'].apply(lambda s: json.loads(s).get('k'))
    geo_k32 = geo[geo['k'] == 32].groupby('type')['purity'].mean()

    txt = pd.read_csv(MIDTERM / 'sweeps' / 'text_per_type.csv')
    txt = txt[(txt['run_type'] == 'version_sweep') &
              (txt['version'] == 'v7') & (txt['algo'] == 'kmeans')]
    txt['k'] = txt['hps'].apply(lambda s: json.loads(s).get('k'))
    txt_k32 = txt[txt['k'] == 32].groupby('type')['purity'].mean()

    vis = pd.read_csv(MIDTERM / 'sweeps' / 'visual_per_type.csv')
    # Prefer colorless (new P2 canonical) but fall back to colored if missing.
    vis_pref = vis[(vis['encoder'] == 'siglip') & (vis['variant'] == 'colorless') &
                   (vis['algo'] == 'kmeans')]
    if vis_pref.empty:
        vis_pref = vis[(vis['encoder'] == 'siglip') & (vis['variant'] == 'colored') &
                        (vis['algo'] == 'kmeans')]
    vis_pref = vis_pref.copy()
    vis_pref['k'] = vis_pref['hps'].apply(lambda s: json.loads(s).get('k'))
    vis_k32 = vis_pref[vis_pref['k'] == 32].groupby('type')['purity'].mean()

    fus = pd.read_csv(MIDTERM / 'fusion' / 'fusion_per_type.csv')
    # Pick the recipe with the highest HDBSCAN macro purity on geo+text+visual.
    fus_summary = pd.read_csv(MIDTERM / 'fusion' / 'fusion.csv')
    fsub = fus_summary[(fus_summary['subset'] == 'geo+text+visual') &
                        (fus_summary['algo'] == 'hdbscan')]
    best_recipe = fsub.loc[fsub['macro_purity'].idxmax(), 'recipe'] if not fsub.empty else 'F2'
    fus = fus[(fus['subset'] == 'geo+text+visual') & (fus['recipe'] == best_recipe) &
              (fus['algo'] == 'kmeans')]
    fus['k'] = fus['hps'].apply(lambda s: json.loads(s).get('k'))
    fus_k32 = fus[fus['k'] == 32].groupby('type')['purity'].mean()

    sup = pd.read_csv(MIDTERM / 'supervised' / 'ceiling_per_type.csv')
    sup_fus = sup[sup['subset'] == 'geo+text+visual'].groupby('type')['recall'].mean()

    types = sorted(set(geo_k32.index) | set(txt_k32.index) |
                   set(vis_k32.index) | set(fus_k32.index) | set(sup_fus.index))

    mat = pd.DataFrame({
        'geo (k=32)':    [geo_k32.get(t, 0) for t in types],
        'text v7 (k=32)': [txt_k32.get(t, 0) for t in types],
        'visual SigLIP (k=32)': [vis_k32.get(t, 0) for t in types],
        'fusion (k=32)': [fus_k32.get(t, 0) for t in types],
        'RF ceiling':    [sup_fus.get(t, 0) for t in types],
    }, index=[_short(t) for t in types])
    mat = mat.sort_values('RF ceiling')

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(mat.values, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha='right')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            col = 'white' if v > 0.55 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=col, fontsize=9)
    ax.set_title('Per-type purity / recall across representations (k-means k=32; RF 5-fold)')
    fig.colorbar(im, ax=ax, label='score')
    _save(fig, '09_per_type_matrix.png')


# ── Figure 11: silhouette vs purity vs k ──────────────────────────────────
def fig_11_silhouette_vs_purity():
    df = pd.read_csv(MIDTERM / 'stability' / 'k_sensitivity.csv')
    g20 = df[df['metric_group'] == 'G20_silhouette_vs_k'].sort_values('k')
    k = g20['k'].values
    sil = g20['silhouette'].values
    macro = g20['macro_purity'].values

    sil_peak_k = int(k[np.argmax(sil)])
    sil_peak_v = sil.max()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    c1, c2 = '#3182bd', '#d94a3f'
    ax1.plot(k, sil, '-o', color=c1, lw=2, ms=4, label='Silhouette score')
    ax1.axvline(sil_peak_k, color=c1, ls=':', lw=1, alpha=0.6)
    ax1.annotate(f'sil peak k={sil_peak_k}\n({sil_peak_v:.3f})',
                 xy=(sil_peak_k, sil_peak_v),
                 xytext=(sil_peak_k + 6, sil_peak_v + 0.01),
                 fontsize=10, color=c1,
                 arrowprops=dict(arrowstyle='->', color=c1, lw=1.2))
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette score', color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    ax1.set_ylim(0, 0.35)

    ax2 = ax1.twinx()
    ax2.plot(k, macro, '-s', color=c2, lw=2, ms=4, label='Macro purity')
    ax2.set_ylabel('Macro purity (17-type eval)', color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    ax2.set_ylim(0, 0.85)

    ax1.axvline(17, color='#888', ls='--', lw=1, alpha=0.5)
    ax1.text(17.5, 0.33, 'k=17\n(IFC types)', fontsize=9, color='#888')
    hdb_k = 102
    ax1.axvline(hdb_k, color='#2a9d8f', ls='--', lw=1, alpha=0.5)
    ax1.text(hdb_k - 15, 0.33, f'HDBSCAN\nk={hdb_k}', fontsize=9, color='#2a9d8f',
             ha='right')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.set_title('Silhouette prefers k~29 but purity keeps climbing\n'
                  'Silhouette penalizes small clusters; it underestimates the true granularity')
    ax1.grid(axis='x', alpha=0.2)
    _save(fig, '11_silhouette_vs_purity.png')


# ── Figure 15: retrieval (late Borda fusion) ─────────────────────────────
def fig_15_retrieval():
    df = pd.read_csv(MIDTERM / 'fusion' / 'late_fusion_retrieval.csv')
    configs = df['config'].tolist()
    labels = [c.replace('late_', '').replace('geo+text+visual', 'geo+txt+vis')
                .replace('geo+text', 'geo+txt').replace('geo+visual', 'geo+vis')
              for c in configs]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(configs))
    w = 0.25
    colors = ['#9ecae1', '#6baed6', '#3182bd']
    for i, col in enumerate(['recall@1', 'recall@5', 'recall@10']):
        vals = df[col].values
        bars = ax.bar(x + (i - 1) * w, vals, w, label=col,
                      color=colors[i], edgecolor='white')
        for xi, v in enumerate(vals):
            ax.text(xi + (i - 1) * w, v + 0.005, f'{v:.2f}',
                    ha='center', fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Recall')
    ax.set_title('Late Borda fusion retrieval: cosine similarity per modality, rank-averaged\n'
                 'geo+text+visual recall@10 = 0.966')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    _save(fig, '15_retrieval.png')


# ── Figure 14: stability ───────────────────────────────────────────────────
def fig_14_stability():
    ari = pd.read_csv(MIDTERM / 'stability' / 'bootstrap_ari.csv')
    pur = pd.read_csv(MIDTERM / 'stability' / 'bootstrap_purity.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ari['ari'], bins=15, color='#3182bd', edgecolor='white')
    axes[0].axvline(ari['ari'].mean(), color='#d94a3f', lw=2,
                    label=f"mean = {ari['ari'].mean():.3f}")
    axes[0].set_xlabel('Adjusted Rand Index (pairwise)')
    axes[0].set_ylabel('# pairs')
    axes[0].set_title(f'Cluster stability — {len(ari)} pairs across '
                      f'{10} 90%-subsample reruns')
    axes[0].legend()

    axes[1].bar(range(len(pur)), pur['macro_purity'].values,
                color='#2a9d8f', edgecolor='white')
    axes[1].axhline(pur['macro_purity'].mean(), color='#888', ls='--', lw=1)
    axes[1].set_xlabel('Bootstrap run')
    axes[1].set_ylabel('Macro purity')
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f'Macro purity across reruns: '
                      f'{pur["macro_purity"].mean():.3f} ± '
                      f'{pur["macro_purity"].std():.3f}')
    _save(fig, '14_stability.png')


def copy_mining_figures():
    """Copy the already-generated UMAP and duplicates figures into figures/."""
    src_list = [
        MIDTERM / 'mining' / 'umap_by_type.png',
        MIDTERM / 'mining' / 'umap_by_cluster.png',
        MIDTERM / 'mining' / 'duplicates_preview.png',
        MIDTERM / 'mining' / 'cluster_sizes.png',
    ]
    names = ['10_umap_by_type.png', '10_umap_by_cluster.png',
             '12_duplicates_preview.png', '13_cluster_sizes.png']
    for src, name in zip(src_list, names):
        dst = FIG_DIR / name
        if src.exists():
            shutil.copy(src, dst)
            print(f'  copied → {dst}')


def main():
    print('[make_figures] producing slide-ready PNGs...')
    fig_00_dataset()
    fig_01_motivation()
    fig_02_feature_progression()
    fig_03_algorithm_comparison()
    fig_04_text_progression()
    fig_05_aggregation_on_v1()
    fig_06_visual_encoders()
    fig_07_fusion_ablation()
    fig_08_supervised_gap()
    fig_09_per_type_matrix()
    fig_11_silhouette_vs_purity()
    fig_14_stability()
    fig_15_retrieval()
    copy_mining_figures()
    print(f'\n[make_figures] done. figures in {FIG_DIR}')


if __name__ == '__main__':
    main()
