"""
Per-feature one-way ANOVA across the merged-class benchmark.

Computes, for each of the 9 handcrafted geometric features, the one-way
ANOVA separating the merged IFC classes:

    - F statistic and raw p-value (via the F-distribution survival function,
      evaluated in log space so extreme tails do not underflow to 0)
    - Holm-corrected p across the 9 features (also in log space)
    - effect sizes eta^2 (SS_between / SS_total) and omega^2 (unbiased)

This reproduces Table 4.5 ("Per-feature one-way ANOVA across the K-class
benchmark") and the per-stage F / eta^2 / p_holm figures quoted in the
methodology chapter, directly from the same load_data() used by every
feature-engineering script. Output is persisted to
results/<dataset>/anova_feature_separation.csv and a LaTeX-ready table is
printed to stdout.

Run:
    python notebooks/data_analysis/feature_engineering/anova_feature_separation.py
    IFCNET_DATASET=ifcnet python notebooks/data_analysis/feature_engineering/anova_feature_separation.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from _common import ACTIVE_SPEC, BASELINE, build_X, load_data

# The 9 features of the thesis representation, in a fixed display order:
# four log-OBB baseline features + five engineered features.
ENGINEERED_9 = ['pc1_z', 'pc3_z', 'cs_aspect_ratio', 'normal_entropy', 'horiz_frac']

# Pretty labels for the LaTeX table (baseline are log10(.+1) transformed).
PRETTY = {
    'Length':            r'$\log_{10}(L\!+\!1)$',
    'CrossSectionArea':  r'$\log_{10}(C\!+\!1)$',
    'Volume':            r'$\log_{10}(V\!+\!1)$',
    'NumVertices':       r'$\log_{10}(N_{\!\text{verts}}\!+\!1)$',
    'pc1_z':             r'\texttt{pc1\_z}',
    'pc3_z':             r'\texttt{pc3\_z}',
    'cs_aspect_ratio':   r'\texttt{cs\_aspect\_ratio}',
    'normal_entropy':    r'\texttt{normal\_entropy}',
    'horiz_frac':        r'\texttt{horiz\_frac}',
}

LN10 = math.log(10.0)


def anova_one_feature(values: np.ndarray, labels: np.ndarray) -> dict:
    """One-way ANOVA of `values` grouped by `labels`.

    Returns F, df_between, df_within, log10(p), eta^2, omega^2.
    p is computed from the F survival function in log space (f.logsf) so that
    extreme tails (p far below the float64 floor of ~1e-308) are reported as a
    real log10 exponent instead of underflowing to 0.0.
    """
    groups = [values[labels == t] for t in np.unique(labels)]
    grand_mean = values.mean()
    n_total = values.size
    k = len(groups)

    ss_between = sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    # log10(p) from the F survival function, robust to extreme tails.
    log10_p = float(stats.f.logsf(f_stat, df_between, df_within)) / LN10

    eta2 = ss_between / ss_total
    # Unbiased omega^2 = (SS_b - df_b * MS_w) / (SS_total + MS_w)
    omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within)

    return {
        'F': f_stat,
        'df_between': df_between,
        'df_within': df_within,
        'log10_p': log10_p,
        'eta2': eta2,
        'omega2': omega2,
    }


def holm_log10(log10_p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction performed entirely in log10 space.

    For m hypotheses sorted by ascending p, the i-th smallest p is multiplied
    by (m - i); in log space that adds log10(m - i). The step-down monotonicity
    constraint (corrected p non-decreasing along the sorted order) is then
    enforced via a running maximum. Capped at log10(1)=0.
    """
    m = log10_p.size
    order = np.argsort(log10_p)              # ascending p == most significant first
    adj = np.empty(m)
    running = -np.inf
    for rank, idx in enumerate(order):
        mult = m - rank                      # m, m-1, ..., 1
        val = log10_p[idx] + math.log10(mult)
        running = max(running, val)           # step-down monotonicity
        adj[idx] = min(running, 0.0)          # corrected p cannot exceed 1
    return adj


def main() -> None:
    df, labels = load_data()
    n, n_classes = len(df), len(set(labels))
    print(f'[anova] dataset={ACTIVE_SPEC.name}  N={n}  classes={n_classes}  '
          f'df=({n_classes - 1}, {n - n_classes})')

    # Feature matrix in the SAME transform the clustering uses:
    # log1p on the 4 baseline size features (log base is irrelevant for ANOVA),
    # bounded engineered features passed through unchanged.
    feat_cols = BASELINE + ENGINEERED_9
    X = build_X(df, BASELINE, ENGINEERED_9)   # (N, 9), columns aligned to feat_cols

    rows = []
    for j, col in enumerate(feat_cols):
        rows.append({'feature': col, **anova_one_feature(X[:, j], labels)})
    res = pd.DataFrame(rows)

    res['log10_p_holm'] = holm_log10(res['log10_p'].to_numpy())
    res = res.sort_values('eta2', ascending=False).reset_index(drop=True)

    out_csv = ACTIVE_SPEC.results_root / 'anova_feature_separation.csv'
    res.to_csv(out_csv, index=False)

    # ── console report ──────────────────────────────────────────────────────
    pd.set_option('display.width', 140)
    show = res.copy()
    show['F'] = show['F'].round(1)
    show['eta2'] = show['eta2'].round(3)
    show['omega2'] = show['omega2'].round(3)
    show['log10_p'] = show['log10_p'].round(1)
    show['log10_p_holm'] = show['log10_p_holm'].round(1)
    print('\n', show[['feature', 'F', 'log10_p', 'log10_p_holm',
                       'eta2', 'omega2']].to_string(index=False), sep='')
    print(f'\n[anova] written -> {out_csv}')

    # ── LaTeX table body (sorted by eta^2), matching Table 4.5 columns ───────
    # For F so large that f.logsf underflows to -inf even in log space, the
    # exact exponent is not representable; report the conservative bound
    # 10^{-300} (these features are all more extreme than the largest finite
    # value computed, ~10^{-314} for Volume), matching the thesis convention.
    FLOOR_EXP = -300
    print('\n% --- LaTeX rows for Table 4.5 (sorted by eta^2) ---')
    for _, r in res.iterrows():
        lph = r['log10_p_holm']
        if not np.isfinite(lph):
            exp = FLOOR_EXP
        else:
            exp = max(int(math.ceil(lph)), FLOOR_EXP)   # ceil: -291.8 -> -291
        p_str = rf'$<\!10^{{{exp}}}$'
        print(rf"{PRETTY.get(r['feature'], r['feature'])} & "
              rf"{r['F']:.1f} & {p_str} & {r['eta2']:.3f} & {r['omega2']:.3f} \\")


if __name__ == '__main__':
    main()
