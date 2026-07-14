"""P_ifcnet_geo_supervised — Random Forest supervised ceiling on IFCNet geo.

Honors the official train/test split (single train→test fit per seed, not
5-fold CV). Five seeds for RF random_state variance.

Output:
  notebooks/data_analysis/results/ifcnet/supervised/geo_ceiling.csv
  notebooks/data_analysis/results/ifcnet/supervised/geo_ceiling_per_type.csv
"""
from __future__ import annotations
import os, sys, tempfile
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp())
os.environ['IFCNET_DATASET'] = 'ifcnet'
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'feature_engineering'))
sys.path.insert(0, str(_HERE))

from _common import ACTIVE_SPEC, BASELINE, build_X, cumulative_bounded_up_to, load_data

SEEDS = [42, 43, 44, 45, 46]


def main():
    spec = ACTIVE_SPEC
    assert spec.name == 'ifcnet', f'expected ifcnet spec, got {spec.name}'
    out_dir = spec.results_root / 'supervised'
    out_dir.mkdir(parents=True, exist_ok=True)

    df, y = load_data()
    bounded = cumulative_bounded_up_to('04_horiz_frac')
    X = build_X(df, BASELINE, bounded)
    Xs = StandardScaler().fit_transform(X)

    train_mask = (df['Split'] == 'train').values
    test_mask = (df['Split'] == 'test').values
    print(f'[sup] X={Xs.shape}  train={train_mask.sum()}  test={test_mask.sum()}  '
          f'classes={len(set(y))}')

    Xtr, ytr = Xs[train_mask], y[train_mask]
    Xte, yte = Xs[test_mask], y[test_mask]

    summary, per_type = [], []
    for seed in SEEDS:
        rf = RandomForestClassifier(
            n_estimators=400, random_state=seed,
            class_weight='balanced', n_jobs=-1,
        )
        rf.fit(Xtr, ytr)
        ypred = rf.predict(Xte)
        types = sorted(set(y.tolist()))
        acc = float((ypred == yte).mean())
        recalls = {t: float((ypred[yte == t] == t).mean()) if (yte == t).sum() else 0.0
                   for t in types}
        macro = float(np.mean(list(recalls.values())))
        summary.append(dict(seed=seed, accuracy=acc, macro_recall=macro))
        for t, r in recalls.items():
            per_type.append(dict(seed=seed, type=t, recall=r,
                                 support_train=int((ytr == t).sum()),
                                 support_test=int((yte == t).sum())))
        print(f'  seed={seed}  acc={acc:.4f}  macro_recall={macro:.4f}')

    sdf = pd.DataFrame(summary)
    pdf = pd.DataFrame(per_type)
    sdf.to_csv(out_dir / 'geo_ceiling.csv', index=False)
    pdf.to_csv(out_dir / 'geo_ceiling_per_type.csv', index=False)
    print(f'\n[sup] mean acc={sdf["accuracy"].mean():.4f} ± {sdf["accuracy"].std():.4f}')
    print(f'[sup] mean macro_recall={sdf["macro_recall"].mean():.4f} '
          f'± {sdf["macro_recall"].std():.4f}')
    print(f'[sup] saved → {out_dir}')

    print('\n=== per-type recall (mean over seeds) ===')
    agg = pdf.groupby('type')['recall'].mean().sort_values(ascending=False)
    print(agg.round(3).to_string())


if __name__ == '__main__':
    main()
