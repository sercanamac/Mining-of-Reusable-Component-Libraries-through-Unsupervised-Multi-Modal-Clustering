"""Chain every downstream phase after the §12 sweep completes.

Reads `fusion.csv` → finds the global HDBSCAN winner → rebuilds X for that
winner → runs:
  - P_summary (print top-10 per sweep CSV)
  - P4a bootstrap stability
  - P4b normalization ablation
  - P4d k sensitivity (G19 + G20)
  - P5a catalog
  - P5c umap viz
  - P5d composition
  - P5e confusion heatmap (G22)
  - P2b late fusion (F4 — reopened)
  - presentation/make_figures.py

Any single phase failure is caught, logged, and the orchestrator moves on.

Usage:
    python notebooks/data_analysis/experiments/P_regen_downstream.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

_EXP = Path(__file__).resolve().parent
REPO = _EXP.parents[2]
MAKE_FIGS = REPO / 'notebooks' / 'data_analysis' / 'presentation' / 'make_figures.py'

STAGES = [
    ('P_summary',              _EXP / 'P_summary.py'),
    ('P3_supervised',          _EXP / 'P3_supervised.py'),
    ('P2b_late_fusion',        _EXP / 'P2b_late_fusion.py'),
    ('P4a_bootstrap',          _EXP / 'P4a_bootstrap.py'),
    ('P4b_normalization',      _EXP / 'P4b_normalization.py'),
    ('P4d_k_sensitivity',      _EXP / 'P4d_k_sensitivity.py'),
    ('P5a_catalog',            _EXP / 'P5a_catalog.py'),
    ('P5c_umap',               _EXP / 'P5c_umap.py'),
    ('P5d_composition',        _EXP / 'P5d_composition.py'),
    ('P5e_confusion',          _EXP / 'P5e_confusion.py'),
    ('make_figures',           MAKE_FIGS),
]


def _run(name: str, script: Path) -> tuple[str, float, int, str]:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            ['python', '-u', str(script)],
            cwd=REPO, check=False, capture_output=True, text=True, timeout=3600,
        )
        dt = time.perf_counter() - t0
        tail = r.stdout[-500:] if r.stdout else ''
        err = r.stderr[-500:] if r.stderr else ''
        status = 'ok' if r.returncode == 0 else f'rc={r.returncode}'
        return name, dt, r.returncode, (tail + '\n--STDERR--\n' + err)
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        return name, dt, -1, 'TIMEOUT'
    except Exception as e:
        dt = time.perf_counter() - t0
        return name, dt, -1, f'EXC {type(e).__name__}: {e}'


def main():
    print(f'[regen] starting downstream regen on {REPO}')
    results = []
    for name, script in STAGES:
        if not script.exists():
            print(f'  SKIP {name}: {script} missing')
            results.append((name, 0.0, -2, 'missing'))
            continue
        print(f'\n[regen] === {name} ===', flush=True)
        n, dt, rc, tail = _run(name, script)
        print(f'[regen] {name} finished rc={rc} in {dt:.1f}s')
        if rc != 0:
            print('  last 500 bytes of output:')
            print(tail)
        results.append((n, dt, rc, tail))

    print('\n[regen] ==== summary ====')
    for name, dt, rc, _ in results:
        status = 'OK' if rc == 0 else f'FAIL rc={rc}'
        print(f'  {name:25s} {dt:7.1f}s  {status}')
    n_fail = sum(1 for _, _, rc, _ in results if rc != 0)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
