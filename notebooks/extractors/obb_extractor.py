"""OBB Feature Extractor for BIM meshes."""
import json
import gc
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm.auto import tqdm

from helpers.mesh import load_mesh_safe, is_degenerate
from helpers.obb import compute_obb

DEFAULT_EXCLUDED_TYPES: Set[str] = {
    'IfcDiscreteAccessory', 'IfcSanitaryTerminal', 'IfcGeographicElement',
    'IfcTransportElement', 'IfcPipeSegment', 'IfcElectricAppliance',
    'IfcCableCarrierFitting', 'IfcFlowSegment', 'IfcShadingDevice',
    'IfcSite', 'IfcSpace', 'IfcOpeningElement'
}


class OBBFeatureExtractor:
    """Extract OBB features from BIM meshes."""

    def __init__(self, metadata_path: Path, excluded_types: Set[str] = None):
        self.metadata_path = Path(metadata_path)
        self.excluded_types = excluded_types or DEFAULT_EXCLUDED_TYPES

        with open(self.metadata_path) as f:
            metadata = json.load(f)
        self.id_to_type: Dict[str, str] = {
            m["GlobalId"]: m["IfcType"] for m in metadata
        }
        print(f"Loaded metadata: {len(self.id_to_type):,} elements")

    def get_mesh_files(self, mesh_dir: Path) -> List[Path]:
        all_files = sorted(Path(mesh_dir).glob("*.obj"))
        filtered = [
            f for f in all_files
            if self.id_to_type.get(f.stem, "Unknown") not in self.excluded_types
        ]
        print(f"Files: {len(all_files):,} -> {len(filtered):,} after filtering")
        return filtered

    def extract_single(self, mesh_path: Path, split: str) -> Optional[Dict]:
        global_id = mesh_path.stem
        ifc_type = self.id_to_type.get(global_id, "Unknown")
        if ifc_type in self.excluded_types:
            return None
        mesh = load_mesh_safe(mesh_path)
        if mesh is None or is_degenerate(mesh):
            return None
        obb = compute_obb(mesh)
        if obb is None or obb.length < 1e-6:
            return None
        features = obb.to_features()
        features['NumVertices'] = len(mesh.vertices)
        features['GlobalId'] = global_id
        features['IfcType'] = ifc_type
        features['Split'] = split
        return features

    def extract_all(self, mesh_dir: Path, batch_size: int = 5000,
                    split: str = "train") -> pd.DataFrame:
        mesh_files = self.get_mesh_files(mesh_dir)
        results, failed = [], 0
        n_batches = (len(mesh_files) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            batch = mesh_files[start:start + batch_size]
            for path in tqdm(batch, desc=f"Batch {batch_idx+1}/{n_batches}"):
                feat = self.extract_single(path, split)
                if feat:
                    results.append(feat)
                else:
                    failed += 1
            gc.collect()

        df = pd.DataFrame(results)
        print(f"Success: {len(results):,}, Failed: {failed:,}")
        return df

    @staticmethod
    def save(df: pd.DataFrame, output_dir: Path, name: str = "obb_features"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{name}.csv", index=False)
        df.to_parquet(output_dir / f"{name}.parquet", index=False)
        print(f"Saved to {output_dir}")
