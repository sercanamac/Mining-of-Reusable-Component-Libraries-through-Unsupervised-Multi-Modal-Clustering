"""Batch render in-context scene views for the annotated subset.

For each GlobalId in data/annotated_metadata.json, opens the source IFC,
extracts all geometry once (cached per IFC file), and renders 9 views with
the target element highlighted orange inside a semi-transparent building.

Output: processed_data/scene_renders/{GlobalId}/{view}.png

Headless pyrender requires an EGL or OSMesa backend; set PYOPENGL_PLATFORM
in the environment if needed (e.g. PYOPENGL_PLATFORM=egl).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
import ifcopenshell
import ifcopenshell.geom
import pyrender
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
META_PATH = DATA_DIR / "annotated_metadata.json"
OUT_DIR = PROJECT_ROOT / "processed_data" / "scene_renders"

VIEW_ANGLES = [
    (0, 0), (90, 0), (180, 0), (270, 0),
    (0, 80), (0, -80),
    (45, 30), (225, 30), (35, 25),
]
VIEW_NAMES = [
    "front", "right", "back", "left",
    "top", "bottom",
    "front_right", "back_left", "iso",
]


def extract_ifc_geometry(ifc_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Multi-threaded extraction of all renderable geometry from an IFC file."""
    model = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    elements = [
        e for e in model.by_type("IfcProduct")
        if not e.is_a("IfcSite")
        and not e.is_a("IfcBuilding")
        and not e.is_a("IfcBuildingStorey")
        and not e.is_a("IfcSpace")
        and not e.is_a("IfcOpeningElement")
    ]

    n_threads = os.cpu_count() or 4
    iterator = ifcopenshell.geom.iterator(
        settings, model, num_threads=n_threads, include=elements
    )

    geometry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if iterator.initialize():
        while True:
            shape = iterator.get()
            gid = shape.guid
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            faces = np.array(shape.geometry.faces).reshape(-1, 3)
            if len(verts) > 0 and len(faces) > 0:
                geometry[gid] = (verts, faces)
            if not iterator.next():
                break
    return geometry


def _camera_pose(target_centroid: np.ndarray, cam_distance: float, az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    eye = target_centroid + cam_distance * np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])
    forward = target_centroid - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


_TARGET_MAT = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=(1.0, 0.3, 0.0, 1.0), metallicFactor=0.1, roughnessFactor=0.6
)
_CONTEXT_MAT = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=(0.7, 0.7, 0.7, 0.15), metallicFactor=0.0, roughnessFactor=0.9,
    alphaMode="BLEND",
)


def render_scene_views(
    renderer: pyrender.OffscreenRenderer,
    geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    target_gid: str,
    out_dir: Path,
    zoom: float = 4.0,
) -> bool:
    """Render 9 views of the building with target_gid highlighted; save to out_dir.

    The renderer is passed in and shared across calls — creating an
    OffscreenRenderer per call leaks pyglet windows and crashes after a few
    hundred renders on macOS.

    Returns True on success, False if target_gid is not in geometry (skip silently).
    """
    if target_gid not in geometry:
        return False

    scene = pyrender.Scene(
        ambient_light=[0.3, 0.3, 0.3], bg_color=[0.15, 0.15, 0.2, 1.0]
    )
    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0),
        pose=np.eye(4),
    )

    target_centroid = None
    target_radius = None
    for gid, (verts, faces) in geometry.items():
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        is_target = gid == target_gid
        if is_target:
            target_centroid = mesh.centroid.copy()
            target_radius = float(mesh.bounding_sphere.primitive.radius)
        scene.add(pyrender.Mesh.from_trimesh(
            mesh, material=_TARGET_MAT if is_target else _CONTEXT_MAT
        ))

    cam_distance = max(target_radius * zoom, 1.0)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_node = scene.add(camera)

    out_dir.mkdir(parents=True, exist_ok=True)
    for (az, el), name in zip(VIEW_ANGLES, VIEW_NAMES):
        scene.set_pose(cam_node, _camera_pose(target_centroid, cam_distance, az, el))
        color, _ = renderer.render(scene)
        Image.fromarray(color).save(out_dir / f"{name}.png")
    return True


def is_already_rendered(gid_dir: Path) -> bool:
    if not gid_dir.exists():
        return False
    return all((gid_dir / f"{v}.png").exists() for v in VIEW_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N elements (debugging)")
    parser.add_argument("--overwrite", action="store_true", help="Re-render even if PNGs already exist")
    parser.add_argument("--zoom", type=float, default=4.0, help="Camera distance multiplier")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    renderer = pyrender.OffscreenRenderer(args.width, args.height)

    with open(META_PATH) as f:
        meta = json.load(f)

    by_ifc: dict[str, list[dict]] = defaultdict(list)
    for entry in meta:
        by_ifc[entry["relative_source_path"]].append(entry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "_manifest.csv"
    manifest_exists = manifest_path.exists()
    manifest_f = open(manifest_path, "a", newline="")
    manifest = csv.writer(manifest_f)
    if not manifest_exists:
        manifest.writerow(["GlobalId", "IfcType", "relative_source_path", "status", "render_seconds"])

    total_targets = sum(len(v) for v in by_ifc.values())
    if args.limit is not None:
        total_targets = min(total_targets, args.limit)

    done = 0
    stats = {"rendered": 0, "skipped_cached": 0, "skipped_missing_gid": 0, "skipped_no_ifc": 0, "failed": 0}
    t_start = time.time()

    for rel_path, entries in by_ifc.items():
        if args.limit is not None and done >= args.limit:
            break

        # Per-IFC short-circuit: all targets already rendered?
        if not args.overwrite and all(is_already_rendered(OUT_DIR / e["GlobalId"]) for e in entries):
            for e in entries:
                manifest.writerow([e["GlobalId"], e["IfcType"], rel_path, "skipped_cached", ""])
                stats["skipped_cached"] += 1
                done += 1
                if args.limit is not None and done >= args.limit:
                    break
            continue

        ifc_path = DATA_DIR / rel_path
        if not ifc_path.exists():
            for e in entries:
                manifest.writerow([e["GlobalId"], e["IfcType"], rel_path, "skipped_no_ifc", ""])
                stats["skipped_no_ifc"] += 1
                done += 1
            print(f"[!] missing IFC: {rel_path}")
            continue

        t_ext = time.time()
        try:
            geometry = extract_ifc_geometry(ifc_path)
        except Exception as exc:
            for e in entries:
                manifest.writerow([e["GlobalId"], e["IfcType"], rel_path, f"extract_failed:{exc}", ""])
                stats["failed"] += 1
                done += 1
            print(f"[!] extract failed for {rel_path}: {exc}")
            continue
        ext_secs = time.time() - t_ext
        print(f"[ifc] {ifc_path.name}: extracted {len(geometry)} elements in {ext_secs:.1f}s")

        for entry in entries:
            if args.limit is not None and done >= args.limit:
                break

            gid = entry["GlobalId"]
            gid_dir = OUT_DIR / gid

            if not args.overwrite and is_already_rendered(gid_dir):
                manifest.writerow([gid, entry["IfcType"], rel_path, "skipped_cached", ""])
                stats["skipped_cached"] += 1
                done += 1
                continue

            t_r = time.time()
            try:
                ok = render_scene_views(renderer, geometry, gid, gid_dir, zoom=args.zoom)
            except Exception as exc:
                manifest.writerow([gid, entry["IfcType"], rel_path, f"render_failed:{exc}", ""])
                stats["failed"] += 1
                done += 1
                print(f"[!] render failed for {gid} ({entry['IfcType']}): {exc}")
                continue
            r_secs = time.time() - t_r

            if ok:
                manifest.writerow([gid, entry["IfcType"], rel_path, "rendered", f"{r_secs:.2f}"])
                stats["rendered"] += 1
            else:
                manifest.writerow([gid, entry["IfcType"], rel_path, "skipped_missing_gid", ""])
                stats["skipped_missing_gid"] += 1
            done += 1

            if done % 20 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (total_targets - done) / rate if rate > 0 else 0.0
                print(
                    f"[{done}/{total_targets}] "
                    f"rendered={stats['rendered']} cached={stats['skipped_cached']} "
                    f"missing_gid={stats['skipped_missing_gid']} fail={stats['failed']} "
                    f"({rate:.2f}/s, ~{remaining/60:.0f}min left)"
                )

        manifest_f.flush()
        del geometry

    manifest_f.close()
    renderer.delete()
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f}min")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
