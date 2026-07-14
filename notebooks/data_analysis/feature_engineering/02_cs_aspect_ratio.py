"""
02_cs_aspect_ratio — add cs_aspect_ratio to baseline + orientation.
"""
from _common import run_stage

DEFICIENCY = (
    "With size and orientation, the previous feature set still collapses objects that "
    "share the same envelope but differ in cross-section shape. A Column and a Plate can "
    "both be elongated, horizontally flat, solid — the distinguishing fact is that a "
    "column has a roughly square cross-section (e.g. 40×40 cm) while a plate has a very "
    "thin cross-section (e.g. 100×2 cm). pre_state_failure.png shows Column ≡ Plate-class "
    "mixing in shared clusters."
)

HYPOTHESIS = (
    "Reuse the same PCA that produced pc1_z / pc3_z. The cross-section is described by the "
    "two shorter axes λ₂ and λ₃. Their ratio:\n\n"
    "  `cs_aspect_ratio = sqrt(λ₃) / sqrt(λ₂)`\n\n"
    "is a bounded shape descriptor in (0, 1]:\n"
    "  * ~1.0 → square cross-section (Column, Beam, Furniture leg)\n"
    "  * ~0.0 → very thin cross-section (Plate, Covering, Slab-on-edge)\n\n"
    "Orthogonal to both size and orientation."
)

FOCUS_TYPES = ['IfcColumn', 'IfcPlate', 'IfcCovering', 'IfcSlab']

NOTES = (
    "Expect modest overall gain but targeted improvement on Plate/Covering/Column. "
    "If silhouette drops while purity rises, the feature is carving cluster boundaries "
    "at the cost of cluster compactness — acceptable for supervised-style metrics."
)


if __name__ == '__main__':
    run_stage(
        stage_id='02_cs_aspect_ratio',
        config_name='+ CS Ratio (7)',
        new_feats=['cs_aspect_ratio'],
        deficiency=DEFICIENCY,
        hypothesis=HYPOTHESIS,
        focus_types=FOCUS_TYPES,
        notes=NOTES,
    )
