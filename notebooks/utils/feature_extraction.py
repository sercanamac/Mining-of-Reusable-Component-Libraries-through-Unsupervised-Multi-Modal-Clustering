"""
Feature extraction utilities for mesh analysis.
"""

import numpy as np
import trimesh
from typing import Dict, Optional


def extract_global_features(mesh: trimesh.Trimesh) -> Optional[Dict]:
    """
    Extract global geometric features from mesh.

    These are scale-invariant features computed on the entire mesh.

    Args:
        mesh: Trimesh object

    Returns:
        Dictionary of features or None if extraction failed
    """
    try:
        extents = mesh.extents
        sorted_ext = np.sort(extents)[::-1]

        # Skip if degenerate dimensions
        if sorted_ext[2] < 1e-10:
            return None

        # PCA on vertices for shape analysis
        centered = mesh.vertices - mesh.vertices.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        ev_sum = eigenvalues.sum() + 1e-8

        features = {
            # Shape ratios (scale-invariant)
            "elongation": sorted_ext[0] / (sorted_ext[2] + 1e-8),
            "flatness": sorted_ext[2] / (sorted_ext[0] + 1e-8),
            "aspect_ratio_12": sorted_ext[0] / (sorted_ext[1] + 1e-8),
            "aspect_ratio_23": sorted_ext[1] / (sorted_ext[2] + 1e-8),

            # PCA-based shape descriptors
            "pca_linearity": (eigenvalues[0] - eigenvalues[1]) / ev_sum,
            "pca_planarity": (eigenvalues[1] - eigenvalues[2]) / ev_sum,
            "pca_scattering": eigenvalues[2] / ev_sum,

            # Mesh complexity (log-scaled)
            "log_vertices": np.log10(len(mesh.vertices) + 1),
            "log_faces": np.log10(len(mesh.faces) + 1),

            # Topology
            "is_watertight": int(mesh.is_watertight),
        }

        # Compactness (only if watertight)
        if mesh.is_watertight and mesh.volume > 0:
            bbox_vol = np.prod(extents)
            features["compactness"] = abs(mesh.volume) / (bbox_vol + 1e-8)
        else:
            features["compactness"] = np.nan

        return features

    except Exception as e:
        return None


def _compute_discrete_curvatures(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """
    Compute discrete curvature at each vertex using normal deviation.

    Returns:
        Array of curvature values at each vertex
    """
    try:
        vertices = mesh.vertices
        faces = mesh.faces
        face_normals = mesh.face_normals
        vertex_normals = mesh.vertex_normals

        curvatures = np.zeros(len(vertices))

        # Build vertex-to-face adjacency
        vertex_face_indices = [[] for _ in range(len(vertices))]
        for face_idx, face in enumerate(faces):
            for vi in face:
                vertex_face_indices[vi].append(face_idx)

        # Compute curvature as normal deviation
        for vi in range(len(vertices)):
            if len(vertex_face_indices[vi]) > 0:
                adjacent_normals = face_normals[vertex_face_indices[vi]]
                normal_deviation = np.linalg.norm(
                    adjacent_normals - vertex_normals[vi], axis=1
                ).mean()
                curvatures[vi] = normal_deviation

        return curvatures

    except Exception:
        return None


def _compute_dihedral_angles(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """
    Compute dihedral angles between adjacent faces.

    Returns:
        Array of dihedral angles in radians
    """
    try:
        face_adjacency = mesh.face_adjacency
        face_normals = mesh.face_normals

        angles = []

        for face1, face2 in face_adjacency:
            n1 = face_normals[face1]
            n2 = face_normals[face2]

            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            dihedral = np.pi - angle
            angles.append(dihedral)

        return np.array(angles)

    except Exception:
        return None


def extract_local_features(mesh: trimesh.Trimesh, include_advanced: bool = False) -> Optional[Dict]:
    """
    Extract local geometric features from mesh.

    These features capture local surface properties like curvature,
    smoothness, and variation.

    Args:
        mesh: Trimesh object
        include_advanced: Include curvature and dihedral angle features (slower)

    Returns:
        Dictionary of features or None if extraction failed
    """
    try:
        features = {}

        # Edge length statistics
        if len(mesh.edges_unique) > 0:
            edge_vectors = (mesh.vertices[mesh.edges_unique[:, 0]] -
                          mesh.vertices[mesh.edges_unique[:, 1]])
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)

            features["edge_length_mean"] = edge_lengths.mean()
            features["edge_length_std"] = edge_lengths.std()
            features["edge_length_cv"] = edge_lengths.std() / (edge_lengths.mean() + 1e-8)
            features["edge_length_range"] = edge_lengths.max() - edge_lengths.min()
        else:
            for k in ["edge_length_mean", "edge_length_std", "edge_length_cv",
                     "edge_length_range"]:
                features[k] = np.nan

        # Face area statistics
        if len(mesh.area_faces) > 0:
            face_areas = mesh.area_faces
            features["face_area_mean"] = face_areas.mean()
            features["face_area_std"] = face_areas.std()
            features["face_area_cv"] = face_areas.std() / (face_areas.mean() + 1e-8)
            features["face_area_range"] = face_areas.max() - face_areas.min()
        else:
            for k in ["face_area_mean", "face_area_std", "face_area_cv",
                     "face_area_range"]:
                features[k] = np.nan

        # Vertex degree statistics (connectivity)
        if len(mesh.edges_unique) > 0:
            vertex_degree = np.bincount(mesh.edges_unique.flatten(),
                                       minlength=len(mesh.vertices))
            features["valence_mean"] = vertex_degree.mean()
            features["valence_std"] = vertex_degree.std()
            features["valence_irregularity"] = np.abs(vertex_degree - 6).mean()
        else:
            for k in ["valence_mean", "valence_std", "valence_irregularity"]:
                features[k] = np.nan

        # Mesh regularity
        if not np.isnan(features.get("edge_length_cv", np.nan)) and \
           not np.isnan(features.get("face_area_cv", np.nan)):
            features["mesh_regularity"] = 1.0 / (1.0 + features["edge_length_cv"] +
                                                  features["face_area_cv"])
        else:
            features["mesh_regularity"] = np.nan

        # Advanced features (curvature, dihedral angles)
        if include_advanced:
            # Curvature
            curvatures = _compute_discrete_curvatures(mesh)
            if curvatures is not None and len(curvatures) > 0:
                features["curvature_mean"] = curvatures.mean()
                features["curvature_std"] = curvatures.std()
                features["curvature_max"] = curvatures.max()
                features["curvature_p50"] = np.percentile(curvatures, 50)
                features["curvature_p90"] = np.percentile(curvatures, 90)
            else:
                for k in ["curvature_mean", "curvature_std", "curvature_max",
                         "curvature_p50", "curvature_p90"]:
                    features[k] = np.nan

            # Dihedral angles
            dihedral_angles = _compute_dihedral_angles(mesh)
            if dihedral_angles is not None and len(dihedral_angles) > 0:
                angle_deviation = np.abs(dihedral_angles - np.pi)
                features["dihedral_mean_dev"] = angle_deviation.mean()
                features["dihedral_std_dev"] = angle_deviation.std()
                features["dihedral_max_dev"] = angle_deviation.max()
                features["smoothness_ratio"] = (angle_deviation < np.radians(10)).mean()
            else:
                for k in ["dihedral_mean_dev", "dihedral_std_dev",
                         "dihedral_max_dev", "smoothness_ratio"]:
                    features[k] = np.nan

        return features

    except Exception as e:
        return None
