"""
One-time mesh pass: extracts all engineered (non-baseline) features for the
annotated subset and caches them to engineered_features.parquet.

Features extracted:
    pc1_z           — |dot(longest PCA axis, Z)|
    pc3_z           — |dot(thinnest PCA axis, Z)|
    cs_aspect_ratio — sqrt(λ3 / λ2), square (1.0) vs thin (~0)
    normal_entropy  — normalized Shannon entropy over 42 icosphere bins, area-weighted
    horiz_frac      — fraction of surface area with |n·Z| > 0.9

Idempotent: re-running overwrites engineered_features.parquet.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.stats import entropy
from sklearn.decomposition import PCA

from _common import ACTIVE_SPEC, PROJECT, Z_AXIS


def iter_targets():
    """Yield (key, mesh_path) for every object the active spec covers.

    Bentley: uses the ANNOTATED_JSON allow-list and flat MESH_DIR.
    IFCNet: walks metadata.json (the metadata script wrote `mesh_path`
    relative to PROJECT, so paths are stable across CWDs).
    """
    spec = ACTIVE_SPEC
    if spec.annotated_json is not None:
        annotated = json.loads(Path(spec.annotated_json).read_text())
        for d in annotated:
            yield d[spec.key_col], spec.mesh_dir / f"{d[spec.key_col]}.obj"
        return

    metadata_path = spec.mesh_dir.parent / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f'metadata.json not found at {metadata_path}')
    meta = json.loads(metadata_path.read_text())
    for row in meta:
        yield row['obj_id'], PROJECT / row['mesh_path']


def main():
    spec = ACTIVE_SPEC
    print(f'[00] spec={spec.name}')

    ref_dirs = np.array(trimesh.creation.icosphere(subdivisions=1).vertices)
    ref_dirs = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    n_bins = len(ref_dirs)

    targets = list(iter_targets())
    print(f'[00] {len(targets)} targets')

    rows, skipped = [], 0
    for i, (key, mesh_path) in enumerate(targets):
        if not mesh_path.exists():
            skipped += 1
            continue
        try:
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
        except Exception:
            skipped += 1
            continue
        verts = np.asarray(mesh.vertices)
        if len(verts) < 3:
            skipped += 1
            continue

        pca = PCA(n_components=3).fit(verts)
        pc1_z = float(abs(np.dot(pca.components_[0], Z_AXIS)))
        pc3_z = float(abs(np.dot(pca.components_[2], Z_AXIS)))

        extents = np.sqrt(pca.explained_variance_)
        cs_ratio = float(extents[2] / extents[1]) if extents[1] > 1e-8 else 0.0

        normals = mesh.face_normals
        areas = mesh.area_faces
        if len(normals) > 0 and len(areas) > 0 and areas.sum() > 0:
            dots = normals @ ref_dirs.T
            bins = dots.argmax(axis=1)
            hist = np.bincount(bins, weights=areas, minlength=n_bins)
            hist = hist / hist.sum()
            norm_ent = float(entropy(hist) / np.log(n_bins))
            z_comp = np.abs(normals @ Z_AXIS)
            horiz_frac = float(areas[z_comp > 0.9].sum() / areas.sum())
        else:
            norm_ent = 0.0
            horiz_frac = 0.0

        rows.append({
            spec.key_col: key,
            'pc1_z': pc1_z, 'pc3_z': pc3_z,
            'cs_aspect_ratio': cs_ratio,
            'normal_entropy': norm_ent,
            'horiz_frac': horiz_frac,
        })
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(targets)}  (skipped so far: {skipped})')

    df_eng = pd.DataFrame(rows)
    spec.engineered_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_eng.to_parquet(spec.engineered_parquet, index=False)
    print(f'\n[00] wrote {len(df_eng)} rows to {spec.engineered_parquet}')
    print(f'[00] skipped {skipped} (missing or unloadable meshes)')
    print(df_eng.describe().round(3))


if __name__ == '__main__':
    main()
