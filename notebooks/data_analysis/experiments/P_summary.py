"""Post-sweep summary — print the winner per phase and the top-10 globally.

Reads:
    results/midterm/sweeps/{geo,text,visual}.csv
    results/midterm/fusion/fusion.csv

Prints top rows by macro_purity, grouped meaningfully. Used after the
§12 re-run to identify the canonical winner.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))

from _common import RESULTS_ROOT  # noqa: E402

MIDTERM = RESULTS_ROOT.parent / 'midterm'


def _top_rows(df, by, n=5, cols=None):
    if cols is None:
        cols = [c for c in df.columns if c not in ('run_id', 'hps', 'name')]
    print(df.sort_values(by, ascending=False).head(n)[cols].to_string(index=False))


def main():
    sep = '=' * 78
    print(sep + '\n[summary] GEO sweep\n' + sep)
    geo = pd.read_csv(MIDTERM / 'sweeps' / 'geo.csv')
    print(f'  {len(geo)} rows')
    print('\n  top-5 by macro_purity (any algo):')
    _top_rows(geo, 'macro_purity', n=5,
              cols=['algo', 'reducer', 'hps', 'purity', 'macro_purity', 'k', 'noise_frac'])

    print('\n' + sep + '\n[summary] TEXT sweep\n' + sep)
    text = pd.read_csv(MIDTERM / 'sweeps' / 'text.csv')
    print(f'  {len(text)} rows')
    # Best per (version, reducer) hdbscan
    sub = text[text.algo == 'hdbscan'].copy()
    print('\n  top-10 HDBSCAN globally:')
    _top_rows(sub, 'macro_purity', n=10,
              cols=['run_type', 'version', 'aggregation', 'reducer', 'hps',
                    'purity', 'macro_purity', 'k', 'noise_frac'])
    # Best per version (averaged across reducers)
    print('\n  best HDBSCAN per version (macro, across reducers):')
    g = sub.groupby('version')['macro_purity'].max().sort_values(ascending=False)
    print(g.to_string())
    # Best reducer for v7 (the hero)
    print('\n  best reducer for v7/single HDBSCAN:')
    v7 = sub[(sub.version == 'v7') & (sub.aggregation == 'single')]
    if not v7.empty:
        _top_rows(v7, 'macro_purity', n=5,
                  cols=['reducer', 'hps', 'purity', 'macro_purity', 'k', 'noise_frac'])

    print('\n' + sep + '\n[summary] VISUAL sweep\n' + sep)
    vis = pd.read_csv(MIDTERM / 'sweeps' / 'visual.csv')
    print(f'  {len(vis)} rows')
    sub = vis[vis.algo == 'hdbscan']
    print('\n  top-10 HDBSCAN globally:')
    _top_rows(sub, 'macro_purity', n=10,
              cols=['encoder', 'variant', 'reducer', 'hps', 'purity',
                    'macro_purity', 'k', 'noise_frac'])
    print('\n  best HDBSCAN per (encoder, variant):')
    g = sub.groupby(['encoder', 'variant'])['macro_purity'].max().sort_values(ascending=False)
    print(g.to_string())

    fusion_csv = MIDTERM / 'fusion' / 'fusion.csv'
    if fusion_csv.exists():
        print('\n' + sep + '\n[summary] FUSION sweep\n' + sep)
        fus = pd.read_csv(fusion_csv)
        print(f'  {len(fus)} rows')
        sub = fus[fus.algo == 'hdbscan']
        print('\n  top-10 HDBSCAN globally:')
        cols = [c for c in ['subset', 'recipe', 'variant', 'hps', 'purity',
                             'macro_purity', 'k', 'noise_frac'] if c in sub.columns]
        _top_rows(sub, 'macro_purity', n=10, cols=cols)
        print('\n  best HDBSCAN per (subset, recipe):')
        g = sub.groupby(['subset', 'recipe'])['macro_purity'].max().unstack(fill_value=None)
        print(g.to_string())
    else:
        print(f'\n(fusion.csv not yet present at {fusion_csv})')


if __name__ == '__main__':
    main()
