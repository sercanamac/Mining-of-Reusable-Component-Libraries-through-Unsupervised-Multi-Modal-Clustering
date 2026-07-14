"""
04_horiz_frac — add horiz_frac to baseline + orientation + cs_aspect_ratio + normal_entropy.
"""
from _common import run_stage

DEFICIENCY = (
    "normal_entropy tells us *how spread out* the normals are, not *where* they point. "
    "A Slab and a Roof can have similar entropy (both have a few dominant directions), but "
    "the slab's dominant directions are straight up/down while a roof's are angled. "
    "pre_state_failure.png shows Slab ≡ Roof confusion persists after normal_entropy."
)

HYPOTHESIS = (
    "Add a directional composition descriptor:\n\n"
    "  `horiz_frac = sum(area_f for f in faces if |n_f · Z| > 0.9) / total_surface_area`\n\n"
    "Bounded in [0, 1]:\n"
    "  * ~1.0 → Slab (nearly all surface area is top/bottom)\n"
    "  * ~0.0 → Wall, Column (nearly all surface area is vertical)\n"
    "  * moderate → Roof (angled), Stair (mix of horizontal treads and vertical risers)\n\n"
    "Threshold 0.9 mirrors the notebook. Orthogonal to normal_entropy: a slab and a roof can "
    "share entropy but will have very different horiz_frac."
)

FOCUS_TYPES = ['IfcSlab', 'IfcRoof', 'IfcStair', 'IfcCovering']

NOTES = (
    "Expect the largest Δ on Slab and Roof. This pair is the canonical test of whether the "
    "feature is load-bearing — if their Δ ≤ 0, the threshold (0.9) may need loosening."
)


if __name__ == '__main__':
    run_stage(
        stage_id='04_horiz_frac',
        config_name='+ Horiz Frac (9)',
        new_feats=['horiz_frac'],
        deficiency=DEFICIENCY,
        hypothesis=HYPOTHESIS,
        focus_types=FOCUS_TYPES,
        notes=NOTES,
    )
