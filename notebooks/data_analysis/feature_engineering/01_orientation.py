"""
01_orientation — add pc1_z, pc3_z to the baseline.

Previous state: 4 OBB size features (log + RobustScaler).
"""
from _common import run_stage

DEFICIENCY = (
    "The baseline encodes size but not orientation. A Column and a Beam can have "
    "near-identical OBB length, cross-section, volume, and vertex count — the only "
    "difference is that a column stands up and a beam lies flat. In the k=17 "
    "cluster → type heatmap (pre_state_failure.png), Column and Beam share clusters, "
    "and vertical walls mix with horizontal slabs for the same reason."
)

HYPOTHESIS = (
    "PCA on mesh vertices gives three principal axes. The absolute dot product of each "
    "axis with the world-up vector Z = [0,0,1] encodes orientation directly:\n\n"
    "  * `pc1_z` = |dot(longest axis, Z)|   → ~1 for vertical elongation (Column, Railing post), "
    "~0 for horizontal elongation (Beam, Slab edge).\n"
    "  * `pc3_z` = |dot(thinnest axis, Z)|  → ~1 for horizontally flat elements (Slab, Roof), "
    "~0 for vertically flat elements (Wall, Plate-on-edge).\n\n"
    "Both features are already bounded in [0, 1]; they are passed through without scaling."
)

FOCUS_TYPES = ['IfcColumn', 'IfcBeam', 'Wall', 'IfcSlab']

NOTES = (
    "Orientation is the single biggest jump in the chain on prior sweeps. "
    "Watch per_type_delta_k17.png for Column/Beam gains."
)


if __name__ == '__main__':
    run_stage(
        stage_id='01_orientation',
        config_name='+ Orientation (6)',
        new_feats=['pc1_z', 'pc3_z'],
        deficiency=DEFICIENCY,
        hypothesis=HYPOTHESIS,
        focus_types=FOCUS_TYPES,
        notes=NOTES,
    )
