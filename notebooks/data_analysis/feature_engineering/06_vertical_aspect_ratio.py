"""
06_vertical_aspect_ratio — second accept of autoresearch run r1.

Adds `vertical_aspect_ratio = z_extent / (z_extent + max(x_extent, y_extent))`
to the Stage-5 frontier. Discovered by analyzing the per-type damage of the
rejected `vertical_excess` proposal (Δpur=−0.0015, the closest near-miss):
it gave wins on Furniture, Wall, Footing, Beam, Stair, Roof but lost
catastrophically on IfcPlate (−15.9 pp) and IfcSlab (−7.3 pp).

The fix wasn't a gate — it was a different normalization. `vertical_excess`
is unbounded (Slab raw value −32, Column +8), and the harness's robust
min-max normalization compressed the useful Plate range into noise.
Replacing the raw delta with a naturally-bounded ratio puts each type at
its NATURAL geometric value:
  Slab/Covering/Roof at the low end (flat horizontal),
  Furniture/MEP/LightFix in the middle,
  Wall/Door/Window/Member/Column at the high end.

No type sits at an artificial extreme. See
`autoresearch/runs/r1/analysis/analysis.md` (Addendum).
"""
from _common import run_stage

DEFICIENCY = (
    "After Stage 5 (+ branched_vertical, 10-D, purity 0.7095), Wall, "
    "Furniture, IfcWindow, IfcCovering, IfcRoof remained as still-mixed "
    "weak types. The seed and Stage-5 features have nothing about the "
    "absolute axis-aligned bounding-box shape: pc1_z and pc3_z capture "
    "PCA-axis directions, BASELINE captures overall scale, but neither "
    "answers 'is this object taller than it is wide?'. For Windows whose "
    "longest axis happens to be horizontal (median pc1_z ≈ 0.006), Length "
    "captures the horizontal width, not the height — so a 1.8m-tall window "
    "and a 2.15m-tall door look size-equivalent on baseline+Stage-5 "
    "features even though their AABB Z extents are 25% different."
)

HYPOTHESIS = (
    "Add a naturally-bounded vertical-aspect ratio:\n\n"
    "  vertical_aspect_ratio = z / (z + max(x, y))\n\n"
    "where z, x, y are AABB extents. Bounded in [0, 1] by construction "
    "(no harness re-normalization needed, so each type lands at its "
    "natural value with no extreme-outlier compression):\n\n"
    "  ~0.01 → IfcSlab, IfcCovering (flat horizontal)\n"
    "  ~0.14 → IfcRoof (mostly flat)\n"
    "  ~0.39 → Furniture (low+wide chairs/desks)\n"
    "  ~0.57 → Stair (~equal vertical and horizontal)\n"
    "  ~0.62 → IfcWindow\n"
    "  ~0.66 → IfcPlate (varied orientation)\n"
    "  ~0.71 → IfcDoor\n"
    "  ~0.73 → Wall (vertical-dominant)\n"
    "  ~0.94 → IfcMember\n"
    "  ~0.97 → IfcColumn (very vertical)\n\n"
    "Compare to the rejected `vertical_excess = z − max(x,y)` (proposal "
    "11, Δpur=−0.0015): same monotonic ordering, but unbounded. After "
    "harness percentile normalization Plate (raw +0.97, near the mid of "
    "the [−32, +8] range) got squished into a noisy middle, fragmenting "
    "the Plate cluster. The bounded ratio form prevents that compression."
)

FOCUS_TYPES = ['Wall', 'IfcWindow', 'Furniture', 'IfcCovering', 'IfcMember']

NOTES = (
    "Discovered by autoresearch run r1, iteration 16 (proposal_016.py). "
    "Δpurity = +0.0146 at K=32 (largest single-feature gain in the run), "
    "silhouette +0.005, Davies-Bouldin −0.022. Combined Stage-4 → Stage-6 "
    "Δ = +2.06 pp purity. The two added features have COMPLEMENTARY type-"
    "damage profiles: branched_vertical's negatives on Wall and Furniture "
    "are recovered (and reversed) by vertical_aspect_ratio's positives. "
    "See autoresearch/runs/r1/analysis/06_journey.png for the per-type "
    "trajectory across all three frontiers."
)


if __name__ == '__main__':
    run_stage(
        stage_id='06_vertical_aspect_ratio',
        config_name='+ Vertical Aspect Ratio (11)',
        new_feats=['vertical_aspect_ratio'],
        deficiency=DEFICIENCY,
        hypothesis=HYPOTHESIS,
        focus_types=FOCUS_TYPES,
        notes=NOTES,
    )
