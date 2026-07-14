"""
03_normal_entropy — add normal_entropy to baseline + orientation + cs_aspect_ratio.
"""
from _common import run_stage

DEFICIENCY = (
    "After size, orientation and cross-section shape, the remaining confusion is between "
    "flat sheets (Slab) and skeletal/complex objects (Railing, Furniture, Stair). These "
    "have different surface composition: a slab's surface is ~2 planes (top and bottom); a "
    "railing is bars, posts, and rails pointing in many directions. No size/PCA feature "
    "captures this. In pre_state_failure.png, Slab ≡ Railing-like types still share clusters."
)

HYPOTHESIS = (
    "Describe surface complexity by the spread of face normals. Bin face normals by their "
    "nearest direction on a 42-vertex icosphere, area-weighted, and compute normalized "
    "Shannon entropy:\n\n"
    "  `normal_entropy ∈ [0, 1]`\n\n"
    "  * ~0 → surface area concentrates in one or two directions (flat Slab)\n"
    "  * ~1 → surface area spread evenly across all 42 bins (complex, e.g. Railing, Furniture)\n\n"
    "Genuinely orthogonal to all previous features — no size/orientation/cross-section "
    "combination encodes surface detail."
)

FOCUS_TYPES = ['IfcSlab', 'IfcRailing', 'IfcFurniture', 'IfcStair']

NOTES = (
    "The 42-bin icosphere is the finest-grained that still gives us ≥1 expected face per bin "
    "for typical meshes; use subdivisions=1 (hardcoded). Lower-count meshes may have small "
    "entropy artifacts — the notebook baseline absorbs this."
)


if __name__ == '__main__':
    run_stage(
        stage_id='03_normal_entropy',
        config_name='+ Normal Entropy (8)',
        new_feats=['normal_entropy'],
        deficiency=DEFICIENCY,
        hypothesis=HYPOTHESIS,
        focus_types=FOCUS_TYPES,
        notes=NOTES,
    )
