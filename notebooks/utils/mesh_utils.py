"""
Utilities for loading and managing mesh data.
"""

import json
import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


def load_metadata(metadata_path: str = "/Users/sercanamac/metadata.json") -> Dict:
    """
    Load IFC metadata from JSON file.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        Dictionary mapping GlobalId to IFC metadata
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Create lookup dictionaries
    id_to_type = {m["GlobalId"]: m["IfcType"] for m in metadata}
    id_to_meta = {m["GlobalId"]: m for m in metadata}

    return {
        "raw": metadata,
        "id_to_type": id_to_type,
        "id_to_meta": id_to_meta
    }


def get_ifc_type(filename: str, id_to_type: Dict) -> str:
    """
    Get IFC type from filename (GlobalId).

    Args:
        filename: Mesh filename (e.g., "3pJo1sVjP0nx$XmJgLUr5T.obj")
        id_to_type: Dictionary mapping GlobalId to IfcType

    Returns:
        IFC type string or "Unknown"
    """
    global_id = Path(filename).stem
    return id_to_type.get(global_id, "Unknown")


def load_mesh_safe(mesh_path: str, force: str = 'mesh') -> Optional[trimesh.Trimesh]:
    """
    Safely load a mesh with error handling.

    Args:
        mesh_path: Path to mesh file
        force: Force loading as specific type (default: 'mesh')

    Returns:
        Trimesh object or None if loading failed
    """
    try:
        mesh = trimesh.load(mesh_path, force=force)

        # Basic validation
        if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
            return None

        return mesh

    except Exception as e:
        return None


def is_degenerate(mesh: trimesh.Trimesh,
                  max_elongation: float = 1000,
                  min_extent: float = 0.01) -> bool:
    """
    Check if mesh has degenerate geometry.

    Args:
        mesh: Trimesh object
        max_elongation: Maximum allowed elongation ratio
        min_extent: Minimum bounding box extent (in meters)

    Returns:
        True if mesh is degenerate, False otherwise
    """
    extents = mesh.extents
    sorted_ext = np.sort(extents)[::-1]

    # Check minimum size
    if sorted_ext[0] < min_extent:
        return True

    # Check elongation
    if sorted_ext[2] < 1e-10:  # Avoid division by zero
        return True

    elongation = sorted_ext[0] / sorted_ext[2]
    if elongation > max_elongation:
        return True

    return False


def filter_ifc_types(ifc_type: str, exclude_types: Optional[List[str]] = None) -> bool:
    """
    Check if IFC type should be excluded.

    Args:
        ifc_type: IFC type string
        exclude_types: List of types to exclude

    Returns:
        True if type should be kept, False if excluded
    """
    if exclude_types is None:
        exclude_types = [
            "IfcSpace",
            "IfcSite",
            "IfcOpeningElement",
            "IfcGeographicElement"
        ]

    return ifc_type not in exclude_types
