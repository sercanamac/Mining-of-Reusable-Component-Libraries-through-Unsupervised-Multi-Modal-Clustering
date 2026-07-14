# Midterm Presentation — Slide Outline

**Thesis:** *Mining of Reusable Component Libraries through Unsupervised
Multi-Modal Clustering*

**Deliverable:** 14 slide-ready PNGs (16:9, 200 DPI) + this outline.
Every number traces to a CSV under `results/midterm/`.

Build PPT/Keynote with these slides. Drop each figure in at ~85% slide width.

---

## Slide 1 — The problem

**Title:** *IFC types are coarse; library structure is hidden.*

**Bullets:**
- 1,700 annotated BIM objects, 17 merged IFC types, unsupervised setting.
- Baseline = 4 log-OBB size features → purity 0.507 at k=32.
- Six IFC types start at < 0.35 per-type purity: the schema doesn't separate them.

**Figure:** `figures/01_motivation.png`
**Source:** `results/feature_engineering/01_orientation/per_type_delta_k32.csv` (`prev` col)

**Speaker notes (3 sent.):** Real-world BIM libraries have thousands of meshes
with IFC-type labels that are generic. Clustering on raw bounding-box
geometry puts Columns and Beams in the same bin, Slabs and Roofs in the
same bin. We want to *mine* richer, latent structure — sub-types, style
variants, functional groupings that the schema can't name.

---

## Slide 2 — Method overview

**Title:** *Three modalities, PCA-8 per block, concat with raw geo, HDBSCAN.*

**Bullets:**
- **Geo:** 9 handcrafted mesh-geometry features (4 log-OBB + 5 engineered).
- **Text:** Gemini v7 IFC-aware prompt → 768-d text → PCA-8.
- **Visual:** SigLIP2-large 1024-d on 9 rendered views → PCA-8.
- **Fusion (F1a):** StandardScaler each block → concat → **25-d** → HDBSCAN leaf/mcs=5/ms=3/force.
- **Validation:** 60-combo factorial HDBSCAN grid × 10-reducer ablation × 4 algorithm families.

**Figure:** *(no figure — pipeline diagram; draw in PPT or reuse from prior notebooks)*

**Speaker notes:** Each modality contributes a different kind of evidence:
shape geometry, symbolic description, rendered appearance. The §12 sweep
confirmed that reducing each high-dim modality to 8 PCA components before
concatenation preserves the most signal — the 25-d fused space gives a
stable ARI 0.79 across bootstrap reruns.

---

## Slide 3 — Geometric feature engineering

**Title:** *5 engineered features lift purity from 0.51 to 0.70 (+39%).*

**Bullets:**
- +Orientation (pc1_z, pc3_z) → separates Column ≡ Beam.
- +CS Ratio → separates Column ≡ Plate.
- +Normal Entropy → separates Slab ≡ Railing.
- +Horiz Frac → separates Slab ≡ Roof.

**Figure:** `figures/02_feature_progression.png`
**Source:** `results/feature_engineering/progression_summary.csv`

**Speaker notes:** Each feature targets a specific cluster-confusion pattern
observed at the previous stage. Purity at k=32 k-means with 10 seeds;
error bars are one standard deviation. Monotonic gain stage-by-stage; no
feature caused regression.

---

## Slide 4 — Algorithm comparison

**Title:** *HDBSCAN (leaf, force-assign) wins the 60-combo grid on geo.*

**Bullets:**
- At k=32: k-means 0.67 macro, GMM 0.68, bisecting 0.64.
- HDBSCAN (leaf, mcs=5, ms=3, force-assign) → **macro 0.77 with k=120** clusters.
- Full 60-combo factorial over {mcs, ms, method, force} — `leaf` beats `eom`.
- Trade-off: HDBSCAN "over-splits" — many small pure clusters beat a few impure big ones.

**Figure:** `figures/03_algorithm_comparison.png`
**Source:** `results/midterm/sweeps/geo.csv`

**Speaker notes:** All four clustering families evaluated on the 9-d geo
features with the full factorial HDBSCAN grid. `leaf` cluster-selection
method finds ~120 micro-clusters after force-assigning noise; we
acknowledge over-splitting inflates purity compared to k=32 — we report both.

---

## Slide 5 — Text description engineering

**Title:** *Prompt engineering v1 → v7: macro purity 0.53 → 0.74 at best reducer.*

**Bullets:**
- v1 (base) → v4 (compact) → v5 (natural) → v6 (keywords) → v7 (IFC-aware).
- **v7 + UMAP-8 + HDBSCAN leaf/mcs=5/ms=3/force → macro 0.740, purity 0.780.**
- UMAP beats PCA and identity everywhere for text: 10-reducer sweep per version.
- Both k-means and HDBSCAN show the same v1→v7 trajectory — prompt matters.

**Figure:** `figures/04_text_progression.png`
**Source:** `results/midterm/sweeps/text.csv` (`run_type=version_sweep`)

**Speaker notes:** 8 prompt iterations × 10 reducer configs. The biggest
prompt jumps came from removing irrelevant geometric detail (v4) and
surfacing IFC-schema-aware terminology (v7). Separately, **UMAP at any
n_components ≥ 4 beats PCA-32 by ~4 pp on text alone** — a late change
from the first-pass plan. We froze `single` as the aggregation (next slide).

---

## Slide 6 — Aggregation ablation (v1 only)

**Title:** *Why we used `single` aggregation throughout.*

**Bullets:**
- v1 has 4 aggregations on disk: `single`, `sum`, `concat`, `single_views`.
- Pure-text: `single` ≈ `sum` ≈ `concat` (within 0.02 macro purity).
- `single_views` mixes text + image — higher but **not pure text**.

**Figure:** `figures/05_aggregation_on_v1.png`
**Source:** `results/midterm/sweeps/text.csv` (`run_type=aggregation_ablation`)

**Speaker notes:** Two-way honest ablation. Single is within noise of best
for pure-text, so we used it for all v2..v7. single_views outperforms but
is an implicit fusion and lives on the fusion slide.

---

## Slide 7 — Visual encoders

**Title:** *Visual with UMAP catches text: SigLIP colored / UMAP-16 → macro 0.779.*

**Bullets:**
- SigLIP, DinoV3, DuoDuo × {colored, colorless} × 10 reducers × 60-combo HDBSCAN.
- **SigLIP/colored/UMAP-16/HDBSCAN leaf → macro 0.779** (tops DinoV3 0.777 / DuoDuo 0.751).
- UMAP universally beats identity and PCA-128 for visual; PCA-8 close behind.
- Colored ≥ colorless on SigLIP; flipped on DinoV3. Encoder family matters <1 pp.

**Figure:** `figures/06_visual_encoders.png`
**Source:** `results/midterm/sweeps/visual.csv`

**Speaker notes:** After the reducer sweep, the "visual plateau" narrative
from the first pass reverses. Visual alone now matches text alone (0.78 vs
0.74). The real gain comes not from a new encoder but from feeding UMAP-16
features into HDBSCAN rather than raw/PCA features.

---

## Slide 8 — Multi-modal fusion

**Title:** *geo+text+visual / F1a PCA-8 + concat → macro 0.802, ARI 0.79.*

**Bullets:**
- 3 subsets × 4 recipes (F1a / F2 / F3 / F5) × 60-combo HDBSCAN = 2160 fits.
- **Winner: `geo+text+visual` F1a (per-block PCA-8 then concat with raw geo) / HDBSCAN leaf → macro 0.802, purity 0.835, k=102.**
- F1a (PCA-8 + geo concat) beats F2 (raw concat) by +3.7 pp and F5 (UMAP-on-concat) by +6.5 pp — **reducing modalities individually matters; UMAP on the full concat destroys cross-modal signal**.
- All three modalities contribute (2-mod geo+visual F1a = 0.797).

**Figure:** `figures/07_fusion_ablation.png` (dual-panel: subset×recipe bars + reducer-dim sweep)
**Source:** `results/midterm/fusion/fusion.csv`

**Speaker notes:** The §12 sweep revised the first-pass story: F2 raw-concat
was *not* the winner. The right fusion recipe is to reduce each high-dim
embedding modality separately (PCA-8 is enough), then concat with the
raw 9-d geo block — preserving each modality's within-cluster density
structure. UMAP on the full concat (F5) collapses useful variance.

---

## Slide 9 — Supervised ceiling

**Title:** *Gap to RF ceiling: ~9 points on macro recall.*

**Bullets:**
- RF 5-fold CV, 5 seeds, PCA-{8, 64} per representation.
- Best supervised: `geo+visual` PCA-8 RF → **0.891 macro recall (0.907 accuracy)**.
- Best unsupervised (F1a PCA-8 geo+text+visual): 0.802 — gap **8.9 pt**.
- RF is largely insensitive to text (+1 pp over geo+visual) — confirms text/visual redundancy for supervised.

**Figure:** `figures/08_supervised_gap.png`
**Source:** `results/midterm/supervised/ceiling.csv`, `results/midterm/fusion/fusion.csv`

**Speaker notes:** The ~9 pt gap at HDBSCAN best-config is the *room to grow*.
Some of it is irreducible (labels contain subjectivity); some could be
closed with better prompt-engineering or fusion. Note: PCA-8 is enough for
the RF ceiling — same dim that wins the unsupervised fusion.

---

## Slide 10 — UMAP: what the clusters look like

**Title:** *Recovered library structure, in two dimensions.*

**Bullets:**
- UMAP 2-D projection of the **25-d F1a PCA-8 fusion embedding**.
- Left: colored by IFC type (17 colors) — type-coherent regions visible.
- Right: colored by HDBSCAN cluster (k=102) — finer-grained sub-categories surface.

**Figures:** `figures/10_umap_by_type.png`, `figures/10_umap_by_cluster.png`
(two-up; show side-by-side)
**Source:** `results/midterm/mining/umap_coords.parquet`

**Speaker notes:** UMAP preserves local neighborhoods in a 2-D projection.
Type coloring (left) shows that types cluster — confirming the embedding
recovers schema structure. Cluster coloring (right) shows 83 sub-regions
— the library mining output; many have intuitive interpretations.

---

## Slide 11 — Per-type diagnosis

**Title:** *Which modality rescues which type.*

**Bullets:**
- 17 types × 5 representations matrix.
- Geo excels on clear-geometry types (Column, Beam, Slab).
- Text rescues identity-ambiguous types via descriptions.
- RF ceiling: >0.9 on most; Member and Plate remain hard.

**Figure:** `figures/09_per_type_matrix.png`
**Source:** all `*_per_type.csv` files

**Speaker notes:** Heatmap diagnosis. Types at the bottom of the matrix
are the hardest across all modalities. Member (misc. structural part)
stays below 0.5 even for RF — fundamentally label-noisy. Plate is
confused with Slab and Door because small flat elements look similar
from any modality.

---

## Slide 12 — Cluster stability

**Title:** *Stable clustering: ARI 0.79 ± 0.02 across bootstrap reruns.*

**Bullets:**
- 10 re-fits on 90% subsamples; 45 pairwise ARI values.
- Mean purity across reruns: 0.830 ± 0.006.
- Macro purity: 0.792 ± 0.008 (lower variance than first-pass).

**Figure:** `figures/14_stability.png`
**Source:** `results/midterm/stability/bootstrap_ari.csv`,
`results/midterm/stability/bootstrap_purity.csv`

**Speaker notes:** Standard robustness check. 45 pairwise ARI values on
the overlap of each subsample pair. Mean 0.73 is solid for a density-based
clustering at k~80; small std means the structure is genuinely there.

---

## Slide 13 — Application: auto-generated library catalog

**Title:** *What the mining gives a designer.*

**Bullets:**
- One HTML page per cluster; top-5 TF-IDF keywords from Gemini descriptions.
- 3 exemplars (cluster-center-nearest) + full member gallery.
- Zero manual tagging. **102 clusters × ~17 members each = navigable**.

**Figure:** screenshot of `results/midterm/mining/catalog/index.html`
(open in browser, capture 16:9 snip). Representative path:
`results/midterm/mining/catalog/`.

**Speaker notes:** This is the tangible output of the mining pipeline.
Given any unlabeled BIM asset library, we generate a human-readable
catalog from clusters + Gemini keywords + exemplars. No domain expert
had to tag anything. Demo: walk through 2-3 cluster pages live.

---

## Slide 14 — Application: near-duplicate detection

**Title:** *Library curation falls out for free: 1,429 near-duplicate pairs.*

**Bullets:**
- Cosine similarity > 0.98 in the winning F1a fusion space (25-d).
- Restricted to same-cluster pairs → cross-cluster false positives avoided.
- Covers 90 of 102 clusters — redundancy is pervasive in the library.
- Byproduct of clustering; no extra method required.

**Figure:** `figures/12_duplicates_preview.png`
**Source:** `results/midterm/mining/duplicates.csv`

**Speaker notes:** BIM libraries accumulate cruft — the same mesh exported
multiple times, slight variants, identical families under different
component names. Flagging these for review is a pragmatic deliverable that
requires no new model.

---

## Slide 15 — Roadmap

**Title:** *Between midterm and defense.*

**Bullets:**
- **Out-of-schema retrieval eval** — 15-20 queries, annotated, with recall@10.
- **Clean-subset tier** — hold out the 100/class subset for a clean-label check.
- **Retrieval API / MCP tools** — expose the mining pipeline as agent-callable tools.
- **Thesis write-up** — pick up from template.

**Figure:** *(no figure — text slide)*

**Speaker notes:** The open-vocabulary claim in the title is the remaining
gap between method and demonstration. Plan is 1 week each for
retrieval eval, clean-subset robustness, and light tool packaging, before
writing begins.

---

# Appendix (Q&A reference)

## Results table — top-line numbers (§12 re-run)

All HDBSCAN numbers use the 60-combo factorial grid; best reducer per cell.

| Subset | Best unsup (HDBSCAN) | k-means k=32 macro | RF ceiling (PCA-8) |
|---|---:|---:|---:|
| geo | 0.774 (k=120) | 0.639 | 0.878 |
| text v7 / UMAP-8 | 0.740 (k=128) | 0.609 | 0.721 |
| visual SigLIP colored / UMAP-16 | 0.779 (k=151) | 0.600 | 0.760 |
| geo+text F1a PCA-8 | 0.785 (k=99) | 0.645 | 0.884 |
| geo+visual F1a PCA-32 | 0.797 (k=119) | 0.588 | **0.891** |
| geo+text+visual F1a PCA-8 | **0.802** (k=102) | 0.669 | 0.888 |

Stability: bootstrap ARI **0.787 ± 0.023** on the headline winner.

Late-fusion retrieval (cosine-sim + Borda, per §12.4 F4): **recall@10 = 0.966** on `late_geo+text+visual`.

## Reproducibility checklist

From repo root:
```bash
pip install umap-learn hdbscan
python notebooks/data_analysis/experiments/P1a_geo_sweep.py
python notebooks/data_analysis/experiments/P1b_text_sweep.py
python notebooks/data_analysis/experiments/P1c_visual_sweep.py
python notebooks/data_analysis/experiments/P2_fusion.py
python notebooks/data_analysis/experiments/P_regen_downstream.py
python notebooks/data_analysis/presentation/make_figures.py
```

`P_regen_downstream.py` chains P3 supervised, P2b late fusion, P4a/4b/4d
stability, P5a-e mining artifacts, and the figure generator in one run
(~7 min total once the sweeps are done).
13. `python ../presentation/make_figures.py`
