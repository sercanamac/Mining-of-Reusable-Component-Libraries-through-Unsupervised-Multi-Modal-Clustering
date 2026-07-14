# Midterm Presentation Materials

Everything in this folder is generated from persisted CSVs under
`notebooks/data_analysis/results/midterm/`.

## What's here

- `outline.md` — 15 slides mapped to figures, bullets, and speaker notes.
- `make_figures.py` — regenerates every figure from the CSVs.
- `figures/` — 14 slide-ready PNGs (200 DPI, 16:9).

## Figures

| File | Slide | Source CSV |
|---|---|---|
| `01_motivation.png` | 1 | `feature_engineering/01_orientation/per_type_delta_k32.csv` |
| `02_feature_progression.png` | 3 | `feature_engineering/progression_summary.csv` |
| `03_algorithm_comparison.png` | 4 | `midterm/sweeps/geo.csv` |
| `04_text_progression.png` | 5 | `midterm/sweeps/text.csv` |
| `05_aggregation_on_v1.png` | 6 | `midterm/sweeps/text.csv` |
| `06_visual_encoders.png` | 7 | `midterm/sweeps/visual.csv` |
| `07_fusion_ablation.png` | 8 | `midterm/fusion/fusion.csv` |
| `08_supervised_gap.png` | 9 | `midterm/supervised/ceiling.csv` + `midterm/fusion/fusion.csv` |
| `09_per_type_matrix.png` | 11 | all `*_per_type.csv` |
| `10_umap_by_type.png`, `10_umap_by_cluster.png` | 10 | `midterm/mining/umap_coords.parquet` |
| `12_duplicates_preview.png` | 14 | `midterm/mining/duplicates.csv` |
| `13_cluster_sizes.png` | appendix | `midterm/mining/cluster_sizes.png` |
| `14_stability.png` | 12 | `midterm/stability/bootstrap_ari.csv` |

Slide 13 is the auto-generated catalog — open
`results/midterm/mining/catalog/index.html` in a browser and screenshot.

## Regenerating everything

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
stability, and P5a-e mining artifacts in one run. Total wall clock
**~2.5 h** for sweeps + **~7 min** for regen on a laptop (Apple Silicon).

Individual stages are still runnable on their own — e.g.
`python notebooks/data_analysis/experiments/P4a_bootstrap.py`.

## Headline numbers (§12 re-run)

- Feature engineering: purity 0.507 → 0.704 at k=32 (4 → 9 features).
- Text prompt engineering: macro purity **0.53 → 0.74** across v1 → v7 (UMAP-8 reducer).
- Visual SigLIP colored / UMAP-16: macro **0.779** (catches text after reducer sweep).
- Best unsupervised (`geo+text+visual` F1a PCA-8 + concat-geo HDBSCAN leaf): macro **0.802**, purity 0.835, k=102.
- Best supervised ceiling (`geo+visual` PCA-8 RF 5-fold CV): macro recall **0.891** — gap 8.9 pp.
- Cluster stability (bootstrap ARI on winning config): **0.787 ± 0.023**.
- Late-fusion retrieval (cosine + Borda, all three modalities) recall@10: **0.966**.
