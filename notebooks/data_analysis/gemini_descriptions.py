#!/usr/bin/env python3
"""
Geometry-aware description generation + embedding pipeline.

Uses pre-computed geometric features (OBB, orientation, surface stats) to enrich
the VLM prompt, so descriptions are informed by both visual renders AND measurements.

Embedding variants:
  - gemini_embed_single/        : embed full description string
  - gemini_embed_sum/           : embed each field separately, average
  - gemini_embed_concat/        : embed each field separately, concatenate (7 * dim)
  - gemini_embed_single_views/  : single text embed + mean of 9 viewpoint image embeds

Usage:
  python gemini_descriptions.py                     # run everything
  python gemini_descriptions.py --skip-descriptions  # embeddings only
  python gemini_descriptions.py --skip-embeddings    # descriptions only
  python gemini_descriptions.py --debug              # first 100 only
  python gemini_descriptions.py --workers 20         # parallel API calls
"""

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from google import genai
from google.genai import types
from scipy.stats import entropy

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
VLM_MODEL = "gemini-3.1-flash-lite-preview"
EMBED_MODEL = "gemini-embedding-2-preview"

VIEWS = ["front", "front_right", "right", "back", "back_left", "left", "top", "bottom", "iso"]
Z_AXIS = np.array([0, 0, 1])

ROOT = Path(__file__).resolve().parent.parent.parent
RENDER_DIR = ROOT / "processed_data" / "renders" / "colored"
MESH_DIR = ROOT / "processed_data" / "meshes" / "train"
FEAT_PATH = ROOT / "processed_data" / "features" / "train_features.parquet"
DESC_DIR = ROOT / "processed_data" / "descriptions" / "gemini_v2"
EMBED_BASE = ROOT / "processed_data" / "gemini_embeddings_v2"

EMBED_DIRS = {
    "single": EMBED_BASE / "gemini_embed_single",
    "sum": EMBED_BASE / "gemini_embed_sum",
    "concat": EMBED_BASE / "gemini_embed_concat",
    "single_views": EMBED_BASE / "gemini_embed_single_views",
}

MORPH_KEYS = None  # Not used — we embed the full output string directly

# ---------------------------------------------------------------------------
# Geometry-aware prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
You are an expert 3D BIM geometry annotator with deep knowledge of building elements. \
You will receive 9 rendered views (orthographic and isometric) of a single BIM element, \
along with precise geometric measurements. Use BOTH the visual appearance AND the \
measurements to produce a compact classification followed by distinguishing keywords.

BIM CONTEXT — this object is one of these common building element types:
- Structural: beams (horizontal), columns (vertical), footings (foundations), \
members (thin secondary bracing/mullions)
- Enclosure: walls (with cutouts for openings), slabs (horizontal floors/ceilings), \
roofs (angled/sloped), plates (thin panels), curtain walls (glazed grid facades)
- Openings: doors (single panel with frame), windows (glazed subdivided frame)
- Circulation: stairs (solid stepped treads), railings (open frameworks with \
thin bars/balusters/posts along edges — NOT stairs themselves)
- MEP: pipes, ducts, sinks, toilets, fixtures
- Furnishing: chairs, tables, plants, shelving, equipment
- Lighting: lamps, ceiling panels, mounted fixtures

KEY DISTINCTIONS:
- Railing = open framework of thin bars/posts/rails. Stairs = solid stepped form.
- Curtain wall = regular grid of many panels. Window = fewer, larger panes.
- Wall = solid panel with scattered cutouts. Door = single panel with frame.

GEOMETRIC MEASUREMENTS FOR THIS OBJECT:
{geometry_block}

OUTPUT FORMAT (single line):
[Thinness], [Shape Class], [Aspect Ratio], [Profile], [Visual Cue]. [keywords]

TAGS (before the period):

1. Thinness — MUST match the cross-section thinness measurement:
   Wire-Thin, Very-Thin, Thin, Medium, Thick

2. Shape Class — MUST be consistent with orientation measurements:
   Horizontal-Rod, Vertical-Rod, Diagonal-Rod, Horizontal-Sheet, \
Vertical-Sheet, Flat-Sheet, Block, Complex

3. Aspect Ratio — MUST match the bounding box ratios provided:
   1:1:1, 1:1:3, 1:1:10, 1:1:30, 1:3:3, 1:3:10, 1:3:30, 1:10:10, 1:10:30, 1:10:100

4. Profile — dominant cross-sectional shape (controlled vocabulary):
   I-Section, T-Section, L-Section, C-Channel, Rectangular, Circular, \
Elliptical, Annular, Triangular, Irregular, Composite

5. Visual Cue — free-form, 1-3 word description of the single most \
distinctive visual feature (e.g. "stepped-treads", "glazed-grid", \
"open-balustrade", "globe-on-arm", "scattered-cutouts", "uniform-extrusion")

KEYWORDS (after the period) — 1 to 5 comma-separated keywords:
- Use only as many keywords as are genuinely distinctive. Simple uniform \
extrusions may need only 1-2. Complex multi-part objects may use all 5.
- Use common functional names if the object is recognizable \
(e.g. chair, table, lamp, plant, sink, toilet, railing, handrail, \
balustrade, pipe, duct, bracket, staircase, shelf, cabinet).
- Do NOT mention materials (no steel, concrete, wood, metal, glass, \
porcelain, ceramic) or colors — these cannot be determined from stripped geometry.
- Hyphenate compounds (office-chair, wall-lamp, open-framework).
- Keywords must add NEW information beyond what the 5 tags already capture.

STRICT RULES:
- Output exactly ONE line: 5 comma-separated tags, period, then 1-5 comma-separated keywords.
- Tags 1-3 MUST match the geometric measurements.
- No IFC class names (no "IfcBeam", "IfcSlab", etc.).
- No generic filler (no "object", "element", "structure", "component")."""


# ---------------------------------------------------------------------------
# Geometry computation
# ---------------------------------------------------------------------------
def compute_geometry(gid: str, obb_row: dict) -> dict | None:
    """Compute geometric features for a single object."""
    mesh_path = MESH_DIR / f"{gid}.obj"
    if not mesh_path.exists():
        return None

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    verts = np.array(mesh.vertices)
    if len(verts) < 4:
        return None

    # PCA on vertices
    cov = np.cov(verts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = np.maximum(eigvals, 1e-12)

    pc1_z = abs(np.dot(eigvecs[:, 0], Z_AXIS))
    pc3_z = abs(np.dot(eigvecs[:, 2], Z_AXIS))
    cs_aspect_ratio = np.sqrt(eigvals[2]) / np.sqrt(eigvals[1]) if eigvals[1] > 0 else 0

    # Normal entropy + horiz frac
    ref_dirs = np.array(trimesh.creation.icosphere(subdivisions=1).vertices)
    ref_dirs = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    n_bins = len(ref_dirs)

    normals = np.array(mesh.face_normals)
    areas = np.array(mesh.area_faces)
    if len(normals) > 0 and areas.sum() > 0:
        bins = np.argmax(normals @ ref_dirs.T, axis=1)
        hist = np.zeros(n_bins)
        for b, a in zip(bins, areas):
            hist[b] += a
        hist /= hist.sum() + 1e-12
        normal_entropy = entropy(hist) / np.log(n_bins)
        horiz_frac = areas[np.abs(normals @ Z_AXIS) > 0.9].sum() / areas.sum()
    else:
        normal_entropy = 0.0
        horiz_frac = 0.0

    # OBB dimensions
    length = obb_row.get("Length", 0)
    cs_area = obb_row.get("CrossSectionArea", 0)
    volume = obb_row.get("Volume", 0)
    n_verts = int(obb_row.get("NumVertices", len(verts)))

    # Derive approximate bounding box axes from OBB
    # Length = longest axis, CrossSectionArea = mid*short, Volume = L*mid*short
    if length > 0 and cs_area > 0:
        mid_short_product = cs_area
        if length > 0:
            mid_times_short = volume / length if volume > 0 and length > 0 else cs_area
        # Estimate mid and short from cs_area and aspect ratio
        # cs_area = mid * short, cs_aspect_ratio ≈ short/mid
        if cs_aspect_ratio > 0:
            mid = np.sqrt(cs_area / cs_aspect_ratio) if cs_aspect_ratio > 0 else np.sqrt(cs_area)
            short = cs_area / mid if mid > 0 else 0
        else:
            mid = np.sqrt(cs_area)
            short = mid
    else:
        mid, short = 0, 0

    # Orientation angles
    angle_longest_from_vertical = np.degrees(np.arccos(np.clip(pc1_z, 0, 1)))
    angle_thinnest_from_vertical = np.degrees(np.arccos(np.clip(pc3_z, 0, 1)))

    # Shape class
    elongation = length / mid if mid > 0 else 1
    flatness = short / mid if mid > 0 else 1

    if elongation > 8 and flatness > 0.2:
        shape_class = "Rod (thin in 2 axes, long in 1)"
    elif flatness < 0.15:
        shape_class = "Sheet (thin in 1 axis, extended in 2)"
    elif elongation < 3:
        shape_class = "Block (comparable in all 3 axes)"
    else:
        shape_class = "Elongated prism"

    # Surface normal summary
    vert_frac = 1.0 - horiz_frac
    if horiz_frac > 0.7:
        normal_summary = f"{horiz_frac:.0%} of surface faces up/down (floor/ceiling-like)"
    elif horiz_frac < 0.15:
        normal_summary = f"{vert_frac:.0%} of surface faces sideways (wall-like)"
    else:
        normal_summary = f"{horiz_frac:.0%} faces up/down, {vert_frac:.0%} faces sideways (mixed)"

    # Complexity
    if normal_entropy < 0.25:
        complexity = "very simple (flat surfaces, few normal directions)"
    elif normal_entropy < 0.40:
        complexity = "moderate"
    else:
        complexity = "complex (normals in many directions)"

    # Cross-section thinness (1/sqrt(cs_area)) — key discriminator for thin members
    cs_thinness = 1.0 / np.sqrt(cs_area) if cs_area > 1e-8 else 0
    sa = mesh.area if hasattr(mesh, "area") else 0
    sa_vol_ratio = sa / volume if volume > 1e-10 else 0

    # Thinness category
    if cs_thinness > 10:
        thinness_desc = "Wire-Thin (extremely thin cross-section, wire-like)"
    elif cs_thinness > 5:
        thinness_desc = "Very-Thin"
    elif cs_thinness > 2:
        thinness_desc = "Thin"
    elif cs_thinness > 1:
        thinness_desc = "Medium"
    else:
        thinness_desc = "Thick (large cross-section)"

    return {
        "length": length,
        "mid": mid,
        "short": short,
        "volume": volume,
        "n_verts": n_verts,
        "angle_longest": angle_longest_from_vertical,
        "angle_thinnest": angle_thinnest_from_vertical,
        "cs_aspect_ratio": cs_aspect_ratio,
        "cs_thinness": cs_thinness,
        "sa_vol_ratio": sa_vol_ratio,
        "normal_entropy": normal_entropy,
        "horiz_frac": horiz_frac,
        "shape_class": shape_class,
        "normal_summary": normal_summary,
        "complexity": complexity,
        "thinness_desc": thinness_desc,
    }


def format_geometry_block(geo: dict) -> str:
    """Format geometry dict into prompt text."""
    dims = sorted([geo["length"], geo["mid"], geo["short"]], reverse=True)
    # Aspect ratio string
    if dims[0] > 0:
        ratios = [d / dims[2] if dims[2] > 0.001 else 999 for d in dims]
        ratio_str = f"{ratios[0]:.0f} : {ratios[1]:.0f} : 1"
    else:
        ratio_str = "unknown"

    return (
        f"- Bounding box axes: {dims[0]:.2f}m × {dims[1]:.2f}m × {dims[2]:.2f}m "
        f"(aspect ratio ≈ {ratio_str})\n"
        f"- Shape class: {geo['shape_class']}\n"
        f"- Orientation: longest axis {geo['angle_longest']:.0f}° from vertical, "
        f"thinnest axis {geo['angle_thinnest']:.0f}° from vertical\n"
        f"- Surface normals: {geo['normal_summary']}\n"
        f"- Surface complexity: {geo['normal_entropy']:.2f} — {geo['complexity']}\n"
        f"- Cross-section squareness: {geo['cs_aspect_ratio']:.2f} "
        f"(1.0 = square, 0.0 = extremely thin)\n"
        f"- Cross-section thinness: {geo['cs_thinness']:.1f} — {geo['thinness_desc']}\n"
        f"- Surface-area-to-volume ratio: {geo['sa_vol_ratio']:.1f} "
        f"(higher = thinner/more surface-dominated)\n"
        f"- Mesh: {geo['n_verts']} vertices, volume = {geo['volume']:.3f} m³"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gids() -> list[str]:
    return sorted(p.name for p in RENDER_DIR.iterdir() if p.is_dir())


def parse_description(desc: str) -> dict:
    """Parse 'tag1, tag2, tag3, tag4, tag5. Description sentence.' format."""
    # Split at first period to separate tags from description
    if ". " in desc:
        tags_part, desc_part = desc.split(". ", 1)
    else:
        tags_part, desc_part = desc, ""
    fields = [f.strip() for f in tags_part.split(",")]
    while len(fields) < 5:
        fields.append("")
    return {
        "thinness": fields[0],
        "shape_class": fields[1],
        "aspect_ratio": fields[2],
        "profile": fields[3],
        "visual_cue": fields[4] if len(fields) > 4 else "",
        "description": desc_part.strip(),
        "full": desc,  # the entire output for embedding
    }


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# Async workers
# ---------------------------------------------------------------------------
async def describe_one(
    client: genai.Client,
    gid: str,
    geo: dict,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
):
    desc_path = DESC_DIR / f"{gid}.txt"
    if desc_path.exists():
        return

    parts = []
    for view in VIEWS:
        img_path = RENDER_DIR / gid / f"{view}.png"
        if img_path.exists():
            parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type="image/png"))

    geometry_block = format_geometry_block(geo)
    prompt_text = PROMPT_TEMPLATE.format(geometry_block=geometry_block)
    parts.append(types.Part.from_text(text=prompt_text))

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
    )

    async with sem:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=VLM_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=config,
            )
            desc = response.text.strip()
            desc_path.write_text(desc)
            print(f"[{idx}/{total}] {gid}: {desc}")
        except Exception as e:
            print(f"[{idx}/{total}] {gid}: ERROR - {e}", file=sys.stderr)


async def embed_one(client: genai.Client, gid: str, dim: int, sem: asyncio.Semaphore, idx: int, total: int, text_only: bool = False):
    check_dirs = ["single"] if text_only else list(EMBED_DIRS.keys())
    if all((EMBED_DIRS[k] / f"{gid}.npy").exists() for k in check_dirs):
        return

    desc = (DESC_DIR / f"{gid}.txt").read_text().strip()
    parsed = parse_description(desc)

    embed_cfg = types.EmbedContentConfig(task_type="CLUSTERING", output_dimensionality=dim)

    async with sem:
        try:
            # Embed the full output (tags + description) as one string
            full_text = parsed["full"]
            # Also embed just the tags (before the period) for compact clustering
            tags_text = full_text.split(". ")[0] if ". " in full_text else full_text

            if text_only:
                result = await asyncio.to_thread(
                    client.models.embed_content,
                    model=EMBED_MODEL,
                    contents=[full_text],
                    config=embed_cfg,
                )
                single_emb = np.array(result.embeddings[0].values, dtype=np.float32)
                if dim != 3072:
                    single_emb = normalize(single_emb)
                np.save(EMBED_DIRS["single"] / f"{gid}.npy", single_emb)
                print(f"[{idx}/{total}] {gid}: OK")
            else:
                # Full pipeline: full embed + tags embed + view embeds
                result = await asyncio.to_thread(
                    client.models.embed_content,
                    model=EMBED_MODEL,
                    contents=[full_text, tags_text],
                    config=embed_cfg,
                )
                embs = [np.array(e.values, dtype=np.float32) for e in result.embeddings]
                if dim != 3072:
                    embs = [normalize(e) for e in embs]

                np.save(EMBED_DIRS["single"] / f"{gid}.npy", embs[0])  # full string
                np.save(EMBED_DIRS["sum"] / f"{gid}.npy", embs[1])     # tags only
                # concat = full + tags concatenated
                np.save(EMBED_DIRS["concat"] / f"{gid}.npy", np.concatenate(embs))

                # View embeds
                view_paths = [RENDER_DIR / gid / f"{v}.png" for v in VIEWS if (RENDER_DIR / gid / f"{v}.png").exists()]
                view_embs = []
                for vp in view_paths:
                    try:
                        vr = await asyncio.to_thread(
                            client.models.embed_content,
                            model=EMBED_MODEL,
                            contents=[types.Part.from_bytes(data=vp.read_bytes(), mime_type="image/png")],
                            config=types.EmbedContentConfig(output_dimensionality=dim),
                        )
                        view_embs.append(np.array(vr.embeddings[0].values, dtype=np.float32))
                    except Exception:
                        pass

                if view_embs:
                    views_mean = normalize(np.mean(view_embs, axis=0))
                    single_views_emb = np.concatenate([embs[0], views_mean])
                else:
                    single_views_emb = np.concatenate([embs[0], np.zeros(dim, dtype=np.float32)])
                np.save(EMBED_DIRS["single_views"] / f"{gid}.npy", single_views_emb)

                print(f"[{idx}/{total}] {gid}: OK (full+tags+{len(view_embs)}v)")

        except Exception as e:
            print(f"[{idx}/{total}] {gid}: ERROR - {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main():
    parser = argparse.ArgumentParser(description="Geometry-aware Gemini description + embedding pipeline")
    parser.add_argument("--skip-descriptions", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=768, choices=[768, 1536, 3072])
    parser.add_argument("--debug", action="store_true", help="5 per type from annotated_subset.json")
    parser.add_argument("--workers", type=int, default=10, help="Max parallel API calls")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output dirs (e.g. debug_geometry)")
    parser.add_argument("--text-only", action="store_true", help="Only save single text embeddings (skip sum/concat/views)")
    args = parser.parse_args()

    if not API_KEY:
        sys.exit(
            "GEMINI_API_KEY is not set. Export it before running:\n"
            "    export GEMINI_API_KEY=your-key-here"
        )
    client = genai.Client(api_key=API_KEY)

    # Apply suffix to output dirs
    global DESC_DIR, EMBED_BASE, EMBED_DIRS
    if args.suffix:
        DESC_DIR = ROOT / "processed_data" / "descriptions" / f"gemini_v2_{args.suffix}"
        EMBED_BASE = ROOT / "processed_data" / f"gemini_embeddings_v2_{args.suffix}"
        EMBED_DIRS = {
            "single": EMBED_BASE / "gemini_embed_single",
            "sum": EMBED_BASE / "gemini_embed_sum",
            "concat": EMBED_BASE / "gemini_embed_concat",
            "single_views": EMBED_BASE / "gemini_embed_single_views",
        }

    DESC_DIR.mkdir(parents=True, exist_ok=True)
    for d in EMBED_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    gids = get_gids()
    if args.debug:
        annot_path = ROOT / "data" / "annotated_subset.json"
        with open(annot_path) as f:
            annot = json.load(f)
        from collections import defaultdict
        by_type = defaultdict(list)
        for d in annot:
            by_type[d["IfcType"]].append(d["GlobalId"])
        render_set = set(gids)
        debug_gids = []
        for t in sorted(by_type):
            available = [g for g in by_type[t] if g in render_set]
            debug_gids.extend(available[:5])
        gids = sorted(set(debug_gids))
        print(f"Debug mode: {len(gids)} objects (5 per type, {len(by_type)} types)")
    else:
        print(f"Found {len(gids)} objects with renders")

    # ------------------------------------------------------------------
    # Pre-compute geometry for all objects
    # ------------------------------------------------------------------
    print("\n=== Pre-computing geometric features ===")
    df_feat = pd.read_parquet(FEAT_PATH)
    feat_map = {row["GlobalId"]: row for _, row in df_feat.iterrows()}

    geo_cache = {}
    skipped = 0
    for i, gid in enumerate(gids):
        obb_row = feat_map.get(gid, {})
        geo = compute_geometry(gid, obb_row)
        if geo:
            geo_cache[gid] = geo
        else:
            skipped += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(gids)} computed...")
    print(f"Geometry computed for {len(geo_cache)}/{len(gids)} objects ({skipped} skipped)")

    # ------------------------------------------------------------------
    # Step 1: Generate descriptions (parallel)
    # ------------------------------------------------------------------
    if not args.skip_descriptions:
        describable = [g for g in gids if g in geo_cache]
        todo = [g for g in describable if not (DESC_DIR / f"{g}.txt").exists()]
        print(f"\n=== Step 1: Descriptions ({len(describable) - len(todo)}/{len(describable)} done, {len(todo)} remaining) ===")

        sem = asyncio.Semaphore(args.workers)
        tasks = [
            describe_one(client, gid, geo_cache[gid], sem, i + 1, len(describable))
            for i, gid in enumerate(describable)
        ]
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Step 2: Generate embeddings (parallel)
    # ------------------------------------------------------------------
    if not args.skip_embeddings:
        desc_gids = sorted(p.stem for p in DESC_DIR.glob("*.txt"))
        if args.debug:
            desc_gids = desc_gids[:100]

        check_dirs = ["single"] if args.text_only else list(EMBED_DIRS.keys())
        todo = [g for g in desc_gids if not all((EMBED_DIRS[k] / f"{g}.npy").exists() for k in check_dirs)]
        mode_str = "text-only" if args.text_only else "all variants"
        print(f"\n=== Step 2: Embeddings ({len(desc_gids) - len(todo)}/{len(desc_gids)} done, {len(todo)} remaining, {mode_str}) ===")
        print(f"    Model: {EMBED_MODEL}, dim: {args.embed_dim}, workers: {args.workers}")

        sem = asyncio.Semaphore(args.workers)
        tasks = [embed_one(client, gid, args.embed_dim, sem, i + 1, len(desc_gids), text_only=args.text_only) for i, gid in enumerate(desc_gids)]
        await asyncio.gather(*tasks)

    print("\nDone.")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
