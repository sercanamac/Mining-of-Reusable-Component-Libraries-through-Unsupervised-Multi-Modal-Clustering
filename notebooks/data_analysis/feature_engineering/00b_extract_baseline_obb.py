"""00b — Baseline OBB features extracted directly from mesh OBJs.

For datasets that ship as meshes only (e.g. IFCNetCore) and don't carry the
upstream IFC-schema-derived `Length / CrossSectionArea / Volume / NumVertices`
parquet that Bentley has. Reproduces the same four columns from the OBJ via
trimesh oriented-bounds, so downstream `_common.load_data()` can read them
without further changes.

  Length             = max(OBB extents)
  CrossSectionArea   = mid * min(OBB extents)
  Volume             = product(OBB extents)   (NOT mesh.volume — many IFCNet
                                                meshes are non-watertight)
  NumVertices        = len(mesh.vertices)

Output schema mirrors Bentley's `train_features.parquet`:
    {key_col, IfcType, Split, Length, CrossSectionArea, Volume, NumVertices}

Usage:
    IFCNET_DATASET=ifcnet python3 00b_extract_baseline_obb.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _common import ACTIVE_SPEC, PROJECT  # noqa: E402


def obb_features(mesh: trimesh.Trimesh) -> tuple[float, float, float, int]:
    """Return (Length, CrossSectionArea, Volume, NumVertices) from an OBB."""
    ext = np.sort(mesh.bounding_box_oriented.extents)
    mn, md, mx = ext[0], ext[1], ext[2]
    length = float(mx)
    csa = float(md * mn)
    volume = float(mx * md * mn)
    nverts = int(len(mesh.vertices))
    return length, csa, volume, nverts


def main() -> None:
    spec = ACTIVE_SPEC
    if spec.name == 'bentley':
        raise SystemExit('Refuse to run on Bentley — its baseline parquet is '
                         'authoritative (IFC-schema-derived). Set '
                         'IFCNET_DATASET=ifcnet to run on IFCNetCore.')

    metadata_path = spec.mesh_dir.parent / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f'metadata.json not found at {metadata_path}')
    meta = json.loads(metadata_path.read_text())
    print(f'[00b] {len(meta)} entries from {metadata_path}')

    rows, failures = [], []
    t0 = time.time()
    for i, row in enumerate(meta):
        obj_path = PROJECT / row['mesh_path']
        try:
            m = trimesh.load(obj_path, force='mesh', process=False)
            if not isinstance(m, trimesh.Trimesh) or len(m.vertices) == 0:
                raise ValueError('not a non-empty Trimesh')
            length, csa, volume, nverts = obb_features(m)
        except Exception as exc:
            failures.append((row['obj_id'], str(exc)))
            continue
        rows.append({
            spec.key_col: row['obj_id'],
            spec.type_col: row['ifc_class'],
            'IfcType': row['ifc_class'],   # alias for back-compat callers
            'Split': row['split'],
            'Length': length,
            'CrossSectionArea': csa,
            'Volume': volume,
            'NumVertices': nverts,
        })
        if (i + 1) % 500 == 0:
            print(f'  [{i+1}/{len(meta)}]  {time.time()-t0:.0f}s elapsed, '
                  f'{len(failures)} failures')

    df = pd.DataFrame(rows)
    out = spec.features_parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'[00b] wrote {len(df)} rows → {out}')
    print(f'[00b] failures: {len(failures)}')
    if failures[:5]:
        for gid, err in failures[:5]:
            print(f'    {gid}: {err}')

    # Quick sanity
    print(f'\n[00b] per-class counts (top 5):')
    print(df[spec.type_col].value_counts().head().to_string())
    print(f'\n[00b] Length / CSA / Volume range:')
    print(df[['Length', 'CrossSectionArea', 'Volume']].describe().round(3).to_string())


if __name__ == '__main__':
    main()
