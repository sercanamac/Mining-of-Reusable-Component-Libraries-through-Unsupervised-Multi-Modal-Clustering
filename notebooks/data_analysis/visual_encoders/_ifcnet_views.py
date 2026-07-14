"""Shared helpers for IFCNetCore visual-encoder scripts.

Walks data/IFCNetCore/metadata.json and groups all PNG view files per obj_id.
Each object has 12 stock renders named `{hash}.{0..11}.png` (grayscale, but
RGB-stored so encoders that expect 3 channels work unchanged).

Used by `extract_dinov3.py`, `extract_siglip.py`, `extract_duoduo.py`.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ObjectEntry:
    obj_id: str
    ifc_class: str
    split: str           # 'train' or 'test'
    render_dir: Path     # absolute, e.g. .../renders/IfcStair/test
    views: list[Path]    # 12 PNG paths sorted by view index


def load_metadata(metadata_path: Path, project_root: Path) -> list[ObjectEntry]:
    """Load metadata.json and resolve all paths to absolute.

    Parameters
    ----------
    metadata_path : path to metadata.json (relative paths inside are resolved
                    against `project_root`).
    project_root  : the Bentley3D project root (so that mesh_path / render_dir
                    inside metadata.json resolve correctly).
    """
    rows = json.loads(Path(metadata_path).read_text())
    out: list[ObjectEntry] = []
    for r in rows:
        render_dir = (project_root / r['render_dir']).resolve()
        views = sorted(render_dir.glob(f"{r['obj_id']}.*.png"))
        out.append(ObjectEntry(
            obj_id=r['obj_id'],
            ifc_class=r['ifc_class'],
            split=r['split'],
            render_dir=render_dir,
            views=views,
        ))
    return out


def filter_pending(entries: list[ObjectEntry], output_dir: Path) -> list[ObjectEntry]:
    """Drop entries whose .npy output already exists (resume support)."""
    return [e for e in entries if not (output_dir / f'{e.obj_id}.npy').exists()]


def iter_with_progress(entries: list[ObjectEntry], desc: str) -> Iterator[ObjectEntry]:
    try:
        from tqdm import tqdm
        yield from tqdm(entries, desc=desc, unit='obj')
    except ImportError:
        # fall back to bare iteration if tqdm isn't installed on the cluster
        n = len(entries)
        for i, e in enumerate(entries):
            if i % 100 == 0:
                print(f'  [{desc}] {i}/{n}', flush=True)
            yield e


def resolve_output_dir(output_root: Path, encoder_dir: str, variant: str) -> Path:
    """Match `_loaders.py::visual_embed_dir` layout exactly."""
    out = output_root / encoder_dir / variant
    out.mkdir(parents=True, exist_ok=True)
    return out


def project_root() -> Path:
    """The Bentley3D project root inferred from this file's location."""
    return Path(__file__).resolve().parents[3]


def default_paths(spec_name: str = 'ifcnet'):
    """Return (metadata, renders_root, output_root) for the given spec."""
    root = project_root()
    if spec_name == 'ifcnet':
        return (
            root / 'data/IFCNetCore/metadata.json',
            root / 'data/IFCNetCore/renders',
            root / 'data/IFCNetCore/processed/rendered_features',
        )
    raise ValueError(f'unknown spec {spec_name!r}')
