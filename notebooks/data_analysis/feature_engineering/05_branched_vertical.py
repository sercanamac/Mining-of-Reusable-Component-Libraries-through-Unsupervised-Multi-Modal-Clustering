"""
05_branched_vertical — first accept of autoresearch run r1.

Adds `branched_vertical = (n_components / 50) × max(0, 1 − 2·horiz_frac)` to
the Stage-4 frontier. The feature was discovered by per-type analysis of the
rejected `n_components` proposal, which gave huge wins on CurtainWall (+0.43)
and LightFixture (+0.16) but lost on Stair (−0.09) and Furniture (−0.10).
Stair was the dominant loss because Stair has high `n_components` (treads +
risers) but also high `horiz_frac` (≈0.52) — it's geometrically distinct
from Railings/CurtainWalls (low `horiz_frac` ≈ 0.06–0.13) but n_components
alone couldn't tell them apart. Multiplying by `max(0, 1 − 2·hf)` zeros out
Stair's contribution while preserving full signal on Railings and
CurtainWalls.

See `autoresearch/runs/r1/analysis/analysis.md` for the full design story
and figures (per-type contribution, design-space scatter, cluster anatomy
before/after, seed robustness).
"""
from _common import run_stage

DEFICIENCY = (
    "After Stage 4 (+ horiz_frac, 9-D, purity 0.7036), several semantic "
    "classes that are physically multi-component assemblies — IfcRailing, "
    "IfcCurtainWall, MEP fittings, IfcLightFixture — remain confused with "
    "single-piece structural elements (Wall, Slab, Plate). The seed has "
    "PCA shape ratios, surface-orientation entropy, and horizontal-area "
    "fraction, but no signal at all about whether the mesh is one piece or "
    "an assembly of disconnected components. Two shapes with the same OBB, "
    "same orientation, same surface entropy can have very different "
    "topologies — a CurtainWall (frame + panels, 50+ components) vs a Wall "
    "(one solid slab) are indistinguishable on the seed feature set."
)

HYPOTHESIS = (
    "Add a topology indicator: count the number of disconnected mesh "
    "components and gate by horizontal-surface fraction so that "
    "geometrically-distinct multi-component types (Stair) are excluded:\n\n"
    "  branched_vertical = (n_components / 50) × max(0, 1 − 2·horiz_frac)\n\n"
    "Bounded in [0, 1]:\n"
    "  * ~0.9 → IfcRailing (always-multi, low hf)\n"
    "  * ~0.7 → IfcCurtainWall (always-multi, low hf)\n"
    "  * ~0.0 → Stair (always-multi, but hf ≈ 0.52 → gate ≈ 0)\n"
    "  * ~0.0 → Wall, Slab, Beam, Plate, Column, Member, Footing, Roof "
    "(single-component, n_components/50 ≈ 0.02)\n\n"
    "The 50-cap caps stray-triangle CAD artifacts. Variance is concentrated "
    "on Railings and CurtainWalls — most types sit at ≈ 0 and don't move "
    "in feature space."
)

FOCUS_TYPES = ['IfcRailing', 'IfcCurtainWall', 'Stair']

NOTES = (
    "Discovered by autoresearch run r1, iteration 10 (proposal_010.py). "
    "Δpurity = +0.0059 at K=32, also improved silhouette (+0.013) and "
    "Davies-Bouldin (−0.056). Per-seed analysis showed IfcRailing per-type "
    "purity went from 0.59 (std 0.12) to 0.67 (std 0.03) — both mean and "
    "stability improved. CurtainWall per-type purity jumped 0.51 → 0.94. "
    "Full design narrative + figures in "
    "autoresearch/runs/r1/analysis/analysis.md."
)


if __name__ == '__main__':
    run_stage(
        stage_id='05_branched_vertical',
        config_name='+ Branched Vertical (10)',
        new_feats=['branched_vertical'],
        deficiency=DEFICIENCY,
        hypothesis=HYPOTHESIS,
        focus_types=FOCUS_TYPES,
        notes=NOTES,
    )
