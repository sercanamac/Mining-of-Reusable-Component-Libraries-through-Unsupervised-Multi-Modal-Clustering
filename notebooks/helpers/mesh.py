"""Mesh loading and validation helpers."""
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional


def load_mesh_safe(mesh_path: Path) -> Optional[trimesh.Trimesh]:
    """Safely load mesh with error handling."""
    try:
        mesh = trimesh.load(str(mesh_path), force='mesh')
        if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
            return None
        return mesh
    except Exception:
        return None


def is_degenerate(mesh: trimesh.Trimesh,
                  max_elongation: float = 1000,
                  min_extent: float = 0.01) -> bool:
    """Check if mesh has degenerate geometry."""
    extents = np.sort(mesh.extents)[::-1]
    if extents[0] < min_extent:
        return True
    if extents[2] < 1e-10:
        return True
    if extents[0] / extents[2] > max_elongation:
        return True
    return False
