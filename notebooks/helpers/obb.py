"""Oriented Bounding Box computation using trimesh."""
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OBB:
    """Oriented Bounding Box representation."""
    transform: np.ndarray   # (4,4) transformation matrix to origin
    extents: np.ndarray     # (3,) full extents [L, W, H] sorted descending

    @property
    def length(self) -> float:
        return self.extents[0]

    @property
    def width(self) -> float:
        return self.extents[1]

    @property
    def height(self) -> float:
        return self.extents[2]

    @property
    def volume(self) -> float:
        return float(np.prod(self.extents))

    @property
    def cross_section_area(self) -> float:
        return self.width * self.height

    def to_features(self) -> Dict[str, float]:
        return {
            'Length': self.length,
            'CrossSectionArea': self.cross_section_area,
            'Volume': self.volume,
        }


def compute_obb(mesh: trimesh.Trimesh) -> Optional[OBB]:
    """Compute OBB using trimesh's oriented_bounds."""
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        # Sort extents descending: L >= W >= H
        extents_sorted = np.sort(extents)[::-1]
        return OBB(transform=to_origin, extents=extents_sorted)
    except Exception:
        return None
