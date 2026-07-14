"""Generate thesis comparison figures from the IFCNet experiment outputs.

Writes 8 PNGs to results/ifcnet/figures/ + a brief README. Pure read-only on
CSVs, no model retraining.
"""
from __future__ import annotations
import os
os.environ['IFCNET_DATASET'] = 'ifcnet'   # quiet on import of _common defaults

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT = Path(__file__).resolve().parents[3]
RES = PROJECT / 'notebooks/data_analysis/results'
IFCNET = RES / 'ifcnet'
OUT = IFCNET / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 110,
})


# ── data loaders ────────────────────────────────────────────────────────────
def load_fusion(encoder: str) -> pd.DataFrame:
    return pd.read_csv(IFCNET / f'fusion_{encoder}' / 'summary.csv')


def load_geo_per_type() -> pd.DataFrame:
    df = pd.read_csv(IFCNET / 'supervised' / 'geo_ceiling_per_type.csv')
    return df.groupby('type')['recall'].mean().rename('geo_recall').to_frame()


def load_duoduo_only_per_class() -> pd.DataFrame | None:
    p = IFCNET / 'supervised' / 'duoduo_only_ceiling.csv'
    return pd.read_csv(p) if p.exists() else None


# ── 1. dataset overview ─────────────────────────────────────────────────────
def fig_dataset_overview():
    import json
    meta = json.loads((PROJECT / 'data/IFCNetCore/metadata.json').read_text())
    classes = pd.Series([m['ifc_class'] for m in meta]).value_counts().sort_values()
    splits = pd.DataFrame(meta).groupby(['ifc_class', 'split']).size().unstack(fill_value=0)
    splits = splits.loc[classes.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    splits.plot(kind='barh', stacked=True, ax=ax,
                color=['#3b82f6', '#fbbf24'], edgecolor='white')
    ax.set_xlabel('Number of objects')
    ax.set_ylabel('')
    ax.set_title('IFCNetCore — 7,930 objects, 20 native IFC classes (train + test)')
    ax.legend(title='split', loc='lower right')
    plt.tight_layout()
    fig.savefig(OUT / '01_dataset_overview.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 2. vision encoder comparison (vision-only) ──────────────────────────────
def fig_vision_only():
    rows = []
    for enc in ('duoduo', 'dinov3', 'siglip'):
        df = load_fusion(enc)
        r = df[df['strategy'] == 'vision_only'].iloc[0]
        rows.append(dict(encoder=enc.upper() if enc != 'duoduo' else 'DuoDuo',
                         f1=r['f1_macro_mean'], unsup=r['unsup_macro_purity']))
    geo_sup = pd.read_csv(IFCNET / 'supervised' / 'geo_ceiling_f1.csv')
    rows.insert(0, dict(encoder='Geo (9-d)',
                        f1=geo_sup['f1_macro'].mean(), unsup=np.nan))
    d = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ['#94a3b8', '#a78bfa', '#34d399', '#60a5fa']
    bars = axes[0].bar(d['encoder'], d['f1'], color=colors, edgecolor='white')
    axes[0].set_ylim(0.80, 0.93)
    axes[0].set_ylabel('F1-macro (RF train→test, 5 seeds)')
    axes[0].set_title('Single-modality supervised ceiling')
    for b, v in zip(bars, d['f1']):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.002, f'{v:.3f}',
                     ha='center', fontsize=9)

    # unsup (only vision rows have valid values)
    dv = d[d['encoder'] != 'Geo (9-d)']
    bars = axes[1].bar(dv['encoder'], dv['unsup'], color=colors[1:], edgecolor='white')
    axes[1].set_ylim(0.74, 0.86)
    axes[1].set_ylabel('Unsup macro purity (HDBSCAN best of 60)')
    axes[1].set_title('Vision-only unsupervised structure')
    for b, v in zip(bars, dv['unsup']):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.003, f'{v:.3f}',
                     ha='center', fontsize=9)

    fig.suptitle('IFCNetCore — vision encoders vs geo baseline', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / '02_vision_only_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 3. fusion strategies × encoder heatmaps (sup F1 + unsup) ────────────────
def fig_fusion_heatmaps():
    # Reduced-dimensional recipes only (raw-concat F2/F3/F3w/F6 dropped); F7 shown as CCA.
    strats = ['F1a', 'F1a_umap', 'F1b', 'F1b_varbal',
              'F5', 'F5_varbal', 'F7', 'vision_only']
    rename = {'F7': 'CCA'}
    out = {}
    out_un = {}
    for enc in ('duoduo', 'dinov3', 'siglip'):
        df = load_fusion(enc).set_index('strategy')
        out[enc] = df['f1_macro_mean']
        out_un[enc] = df['unsup_macro_purity']
    f1 = pd.DataFrame(out)[['duoduo', 'dinov3', 'siglip']].loc[strats].rename(index=rename)
    unsup = pd.DataFrame(out_un)[['duoduo', 'dinov3', 'siglip']].loc[strats].rename(index=rename)
    unsup_disp = unsup.applymap(lambda x: x if pd.notna(x) else None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    sns.heatmap(f1, annot=True, fmt='.3f', cmap='YlGnBu',
                ax=axes[0], cbar=False, vmin=0.70, vmax=0.93,
                linewidths=0.4, linecolor='white')
    axes[0].set_title('Supervised F1-macro (train→test, 5 seeds)')
    axes[0].set_xlabel(''); axes[0].set_ylabel('')

    # mask nans
    mask = unsup_disp.isna()
    sns.heatmap(unsup_disp, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=axes[1], cbar=False, vmin=0.65, vmax=0.87,
                linewidths=0.4, linecolor='white', mask=mask)
    axes[1].set_title('Unsupervised macro purity (HDBSCAN best of 60)')
    axes[1].set_xlabel(''); axes[1].set_ylabel('')

    fig.suptitle('IFCNetCore — 7 fusion strategies × 3 vision encoders',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / '03_fusion_heatmaps.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 4. varbal lift (F5 → F5_varbal across configurations) ───────────────────
def fig_varbal_lift():
    """Show F5 → F5_varbal F1 lift across (dataset × encoder) setups."""
    rows = []
    # IFCNet — 3 encoders, bimodal
    for enc in ('duoduo', 'dinov3', 'siglip'):
        df = load_fusion(enc).set_index('strategy')
        rows.append(dict(setup=f'IFCNet · {enc.upper() if enc!="duoduo" else "DuoDuo"} (bimodal)',
                         f5=df.loc['F5', 'f1_macro_mean'],
                         f5_varbal=df.loc['F5_varbal', 'f1_macro_mean']))

    # IFCNet trimodal (gemma4 + SigLIP + geo)
    tri_i = pd.read_csv(IFCNET / 'trimodal_siglip_colorless_gemma4_single' / 'summary.csv').set_index('strategy')
    rows.append(dict(setup='IFCNet · SigLIP+text (trimodal)',
                     f5=tri_i.loc['F5', 'f1_macro_mean'],
                     f5_varbal=tri_i.loc['F5_varbal', 'f1_macro_mean']))
    # Bentley trimodal
    tri_b = pd.read_csv(RES / 'feature_engineering' / 'trimodal_siglip_colored_gemma4_single' / 'summary.csv').set_index('strategy')
    rows.append(dict(setup='Bentley · SigLIP+text (trimodal)',
                     f5=tri_b.loc['F5', 'f1_macro_mean'],
                     f5_varbal=tri_b.loc['F5_varbal', 'f1_macro_mean']))

    d = pd.DataFrame(rows)
    d['lift'] = d['f5_varbal'] - d['f5']
    d = d.sort_values('lift', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(d))
    w = 0.38
    b1 = ax.bar(x - w/2, d['f5'], w, label='F5 (std-concat → UMAP)',
                color='#fca5a5', edgecolor='white')
    b2 = ax.bar(x + w/2, d['f5_varbal'], w, label='F5_varbal (varbal-concat → UMAP)',
                color='#34d399', edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(d['setup'], rotation=15, ha='right')
    ax.set_ylabel('F1-macro')
    ax.set_ylim(0.55, 0.95)
    ax.set_title('Variance balancing recovers F1 by +0.09 to +0.16 across all setups')
    for bb, v in zip(b1, d['f5']):
        ax.text(bb.get_x() + bb.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
    for bb, v in zip(b2, d['f5_varbal']):
        ax.text(bb.get_x() + bb.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
    # annotate lift
    for xi, lift in zip(x, d['lift']):
        ax.annotate(f'+{lift:.3f}', (xi, d['f5_varbal'].iloc[xi] + 0.025),
                    ha='center', fontsize=9, color='#065f46', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(OUT / '04_varbal_lift.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 5. ceiling progression: geo → vision → bimodal → trimodal ───────────────
def fig_ceiling_progression():
    geo_f1 = pd.read_csv(IFCNET / 'supervised' / 'geo_ceiling_f1.csv')['f1_macro'].mean()
    sig = load_fusion('siglip').set_index('strategy')
    vis_f1 = sig.loc['vision_only', 'f1_macro_mean']
    bimodal_best = sig['f1_macro_mean'].max()
    bimodal_strat = sig['f1_macro_mean'].idxmax()
    tri = pd.read_csv(IFCNET / 'trimodal_siglip_colorless_gemma4_single' / 'summary.csv').set_index('strategy')
    tri_best = tri['f1_macro_mean'].max()
    tri_strat = tri['f1_macro_mean'].idxmax()

    labels = ['Geo only\n(9-d)', 'SigLIP only\n(1024-d)',
              f'Geo + SigLIP\n({bimodal_strat})', f'Geo + SigLIP + Gemma4\n({tri_strat})']
    vals = [geo_f1, vis_f1, bimodal_best, tri_best]
    colors = ['#94a3b8', '#60a5fa', '#34d399', '#fbbf24']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', width=0.65)
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel('F1-macro (RF train→test, 5 seeds)')
    ax.set_title('IFCNetCore supervised ceiling — modality progression')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.003, f'{v:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    # lift arrows
    for i in range(len(vals) - 1):
        lift = vals[i+1] - vals[i]
        color = '#065f46' if lift > 0 else '#991b1b'
        sign = '+' if lift > 0 else ''
        ax.annotate(f'{sign}{lift:.3f}',
                    xy=(i + 0.5, max(vals[i], vals[i+1]) + 0.014),
                    ha='center', fontsize=9, color=color, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / '05_ceiling_progression.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 6. per-class winner: geo vs SigLIP vs F1a fusion ────────────────────────
def fig_per_class_winner():
    geo = load_geo_per_type()
    duo = load_duoduo_only_per_class()
    if duo is None:
        # fall back to geo + sigvision_only summary — not per-class. Skip.
        print('  [skip 06_per_class] need per-class vision; only DuoDuo CSV exists')
        # Still produce a geo-only per-class bar
        d = geo.reset_index().rename(columns={'index': 'type'})
        d = d.sort_values('geo_recall')
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh([t.replace('Ifc', '') for t in d['type']],
                d['geo_recall'], color='#94a3b8', edgecolor='white')
        ax.set_xlabel('Per-class recall (geo RF)')
        ax.set_title('IFCNetCore — per-class geo-only RF recall')
        plt.tight_layout()
        fig.savefig(OUT / '06_per_class_geo_only.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        return

    # We have DuoDuo per-class F1 (classification_report saved as CSV → per-seed rows).
    # The csv is per-seed acc/f1 only — no per-class. Skip detailed comparison.
    # Instead show geo recall vs supervised-vs-unsup gap from KMeans@20
    sweep = pd.read_csv(IFCNET / 'sweeps' / 'geo_per_type.csv')
    sum_sw = pd.read_csv(IFCNET / 'sweeps' / 'geo.csv')
    best = sum_sw[(sum_sw['algo']=='kmeans') & (sum_sw['hp_k']==20)].sort_values('macro_purity', ascending=False).iloc[0]
    rid = int(best['run_id'])
    sub = sweep[sweep['run_id']==rid][['type','purity']].set_index('type')

    df = geo.join(sub.rename(columns={'purity':'kmeans_purity'})).reset_index()
    df = df.rename(columns={'index': 'type', 0: 'type'})
    if 'type' not in df.columns:
        df = df.reset_index().rename(columns={'index':'type'})
    df['type'] = df['type'].str.replace('Ifc', '', regex=False)
    df['gap'] = df['geo_recall'] - df['kmeans_purity']
    df = df.sort_values('gap', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(df))
    ax.barh(y, df['kmeans_purity'], color='#fca5a5', edgecolor='white',
            label='Unsup KMeans@k=20 per-class purity', height=0.4)
    ax.barh(y + 0.4, df['geo_recall'], color='#60a5fa', edgecolor='white',
            label='Supervised RF per-class recall', height=0.4)
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(df['type'])
    ax.set_xlabel('Score (geo features only)')
    ax.set_title('IFCNetCore — supervised vs unsupervised per-class gap (geo only)')
    ax.set_xlim(0, 1.0)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(OUT / '06_per_class_geo_sup_vs_unsup.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 7. Bentley vs IFCNet trimodal side-by-side ──────────────────────────────
def fig_bentley_vs_ifcnet_trimodal():
    b = pd.read_csv(RES / 'feature_engineering' / 'trimodal_siglip_colored_gemma4_single' / 'summary.csv').set_index('strategy')
    i = pd.read_csv(IFCNET / 'trimodal_siglip_colorless_gemma4_single' / 'summary.csv').set_index('strategy')
    order = ['F1a','F1a_umap','F1b','F1b_varbal','F5','F5_varbal','F7']
    disp = ['CCA' if s == 'F7' else s for s in order]
    d = pd.DataFrame({'Bentley (17 classes, 5-fold CV)': b.loc[order, 'f1_macro_mean'],
                      'IFCNet (20 classes, train→test)': i.loc[order, 'f1_macro_mean']})

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(d))
    w = 0.42
    ax.bar(x - w/2, d.iloc[:, 0], w, color='#a78bfa', edgecolor='white',
           label=d.columns[0])
    ax.bar(x + w/2, d.iloc[:, 1], w, color='#34d399', edgecolor='white',
           label=d.columns[1])
    ax.set_xticks(x); ax.set_xticklabels(disp, rotation=30, ha='right')
    ax.set_ylabel('F1-macro')
    ax.set_ylim(0.60, 0.95)
    ax.set_title('Trimodal fusion (SigLIP + Gemma4 + Geo) — Bentley vs IFCNet')
    ax.legend(loc='lower left')
    ax.axhline(0.857, color='#94a3b8', ls='--', lw=0.7, alpha=0.6)
    ax.text(len(order)-0.3, 0.857, 'IFCNet geo-only ceiling',
            color='#475569', fontsize=8, ha='right', va='bottom')
    plt.tight_layout()
    fig.savefig(OUT / '07_bentley_vs_ifcnet_trimodal.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 8. dim vs F1 scatter across strategies ──────────────────────────────────
def fig_dim_vs_f1():
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {'duoduo': 'o', 'dinov3': '^', 'siglip': 's'}
    colors  = {'duoduo': '#94a3b8', 'dinov3': '#a78bfa', 'siglip': '#34d399'}
    for enc in ('duoduo', 'dinov3', 'siglip'):
        df = load_fusion(enc)
        ax.scatter(df['dim'], df['f1_macro_mean'], s=90, marker=markers[enc],
                   facecolor=colors[enc], edgecolor='black', alpha=0.85,
                   label=enc.upper() if enc != 'duoduo' else 'DuoDuo')
    ax.set_xscale('log')
    ax.set_xlabel('Representation dimensionality (log scale)')
    ax.set_ylabel('F1-macro')
    ax.set_title('IFCNet fusion — F1-macro vs representation dim across strategies')
    ax.set_xlim(3, 2500)
    ax.set_ylim(0.65, 0.95)
    ax.legend(loc='lower right')
    ax.axhline(0.857, color='#94a3b8', ls='--', lw=0.7, alpha=0.6)
    ax.text(2400, 0.860, 'geo-only ceiling', color='#475569', fontsize=8, ha='right')
    plt.tight_layout()
    fig.savefig(OUT / '08_dim_vs_f1.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── 9. published baselines vs this work (F1-macro, same split) ──────────────
def fig_published_baselines():
    """Slide: handcrafted / fusion results beside the published IFCNet baselines.

    'This work' numbers are read from the experiment CSVs so they stay in sync
    with the chapter; the two published values are reproduced from the cited
    papers (same official IFCNetCore train/test split):
      MVCNN 0.869 [emunds2021ifcnet, su2015mvcnn];
      SpaRSE-BIM ~0.818 [emunds2022sparsebim].
    """
    geo_f1 = pd.read_csv(IFCNET / 'supervised' / 'geo_ceiling_f1.csv')['f1_macro'].mean()
    sig = load_fusion('siglip').set_index('strategy')
    vis_f1 = sig.loc['vision_only', 'f1_macro_mean']
    bimodal_best = sig['f1_macro_mean'].max()
    MVCNN, SPARSE = 0.869, 0.818

    rows = [
        ('SpaRSE-BIM',         SPARSE,       'published'),
        ('9-feature RF',       geo_f1,       'work'),
        ('MVCNN',              MVCNN,        'published'),
        ('SigLIP2 + RF',       vis_f1,       'work'),
        ('Geo + SigLIP',       bimodal_best, 'best'),
    ]
    rows.sort(key=lambda r: r[1])            # ascending → winner at top in barh
    names  = [r[0] for r in rows]
    vals   = [r[1] for r in rows]
    cmap   = {'published': '#94a3b8', 'work': '#60a5fa', 'best': '#34d399'}
    colors = [cmap[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor='white', height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0.78, 0.945)
    ax.set_xlabel('F1-macro  ·  IFCNetCore official train/test split')
    ax.set_title('Beating the published deep-learning baselines')
    for yi, v in zip(y, vals):
        ax.text(v + 0.002, yi, f'{v:.3f}', va='center', ha='left',
                fontsize=10, fontweight='bold')

    # 'best published' reference line at MVCNN, label in the empty band beside
    # the short bars (bottom rows) so it never crosses the winning bar.
    ax.axvline(MVCNN, color='#475569', ls='--', lw=0.9, alpha=0.7, zorder=0)
    ax.text(MVCNN + 0.002, 0.5, 'best published (MVCNN)',
            color='#475569', fontsize=8, rotation=90, va='center', ha='left')
    # winner delta over MVCNN, set inside the green bar (dark-on-light green)
    ax.text(bimodal_best - 0.030, len(rows) - 1, f'+{bimodal_best - MVCNN:.3f} vs MVCNN',
            fontsize=9, color='#065f46', fontweight='bold', va='center', ha='right')

    # legend (this work vs published)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#60a5fa', label='this work'),
                       Patch(color='#94a3b8', label='published baseline')],
              loc='lower right', frameon=False)
    plt.tight_layout()
    fig.savefig(OUT / '09_published_baselines.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    figs = [
        ('01_dataset_overview',            fig_dataset_overview),
        ('02_vision_only_comparison',      fig_vision_only),
        ('03_fusion_heatmaps',             fig_fusion_heatmaps),
        ('04_varbal_lift',                 fig_varbal_lift),
        ('05_ceiling_progression',         fig_ceiling_progression),
        ('06_per_class_geo_sup_vs_unsup',  fig_per_class_winner),
        ('07_bentley_vs_ifcnet_trimodal',  fig_bentley_vs_ifcnet_trimodal),
        ('08_dim_vs_f1',                   fig_dim_vs_f1),
        ('09_published_baselines',         fig_published_baselines),
    ]
    for name, fn in figs:
        try:
            fn()
            print(f'  ✓ {name}')
        except Exception as e:
            print(f'  ✗ {name}: {type(e).__name__}: {e}')
    print(f'\nfigures → {OUT}/')


if __name__ == '__main__':
    main()
