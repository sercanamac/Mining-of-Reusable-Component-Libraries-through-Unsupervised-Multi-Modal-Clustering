# Bentley3D

Semantic classification of 3D BIM objects via unsupervised multi-modal clustering.
Geometric, text (VLM description), and visual embeddings are clustered and fused;
see `CLAUDE.md` for the architecture.

## Setup

```bash
conda create -n bentley python=3.12
conda activate bentley
pip install -r requirements.txt
```

Two cache directories must be writable, or `import umap` fails with
`RuntimeError: cannot cache function 'rdist'`. If your home directory is not writable
(CI, sandbox, shared machine), point them somewhere that is:

```bash
export MPLCONFIGDIR=/tmp/mpl
export NUMBA_CACHE_DIR=/tmp/numba
```

## Data

`bentley_experiment_data.zip` (~420 MB) unzips **at the repository root** — the folder
containing `notebooks/`, `data/`, and `processed_data/`. Paths inside are repo-relative and
drop straight back into place, so no flags and no destination are needed:

```bash
cd /path/to/Bentley3D
unzip bentley_experiment_data.zip
```

Extracting from anywhere else creates a stray nested `processed_data/` and every script
will fail to find its inputs. To verify a good extraction:

```bash
ls processed_data/features/train_features.parquet   # should exist
ls processed_data/meshes/train | wc -l              # 1845+
```

The bundle holds everything the experiments read:

| Input | Path |
| --- | --- |
| OBB baseline features | `processed_data/features/train_features.parquet` |
| Engineered features | `notebooks/data_analysis/results/feature_engineering/engineered_features.parquet` |
| Meshes (1,845 `.obj`) | `processed_data/meshes/train/` |
| Text embeddings (9 versions) | `processed_data/gemini_embeddings*/`, `gemma4_embeddings/` |
| Visual embeddings | `processed_data/rendered_features/` (DINOv3, SigLIP, DuoDuo, Gemini) |
| Labels / subset | `data/annotated_subset.json`, `data/metadata.json` |
| IFCNet (second dataset) | `data/IFCNetCore/features/`, `processed/` |

The raw IFC submissions are **not** required; they are only needed to regenerate
features and embeddings from scratch.

### The annotated subset: 1,849 → 1,700

`data/annotated_subset.json` holds 1,849 objects with a ground-truth `IfcType`. Evaluation
runs on 1,700 of them: `DatasetSpec.excluded_types` drops 149 objects belonging to the two
catch-all classes (`IfcBuildingElementProxy`, `IfcBuildingElementPart`), which carry no
semantic label. `type_map` then merges raw IFC types (e.g. `IfcWall` + `IfcWallStandardCase`
→ `Wall`) down to the **17** classes used as ground truth, which is where `k_main=17` comes from.

## Sanity check

`P1a` is the fastest experiment (~12 s) and touches the full loading path, so run it first
to confirm the environment and data are wired up correctly:

```bash
python notebooks/data_analysis/experiments/P1a_geo_sweep.py
```

It prints `[P1a] X shape = (1700, 9), n_types = 17` and finishes with the best result per
algorithm. The output is deterministic (seeds 42–51), so these numbers should reproduce
exactly:

```
     algo   purity  macro_purity  silhouette    k
bisecting 0.706471      0.639729    0.273540   32
      gmm 0.735882      0.680677    0.248823   32
  hdbscan 0.805294      0.773866    0.199446  120
   kmeans 0.714706      0.665013    0.299741   32
```

It writes `notebooks/data_analysis/results/midterm/sweeps/geo.csv` (175 rows). If the
numbers differ, the environment or the data extraction is wrong — not the code.

## Running the experiments

Each script is standalone and run from the repository root. Results are written under
`notebooks/data_analysis/results/`.

```bash
python notebooks/data_analysis/experiments/P1a_geo_sweep.py    # geometric features (~12s)
python notebooks/data_analysis/experiments/P1b_text_sweep.py   # text embeddings (~40min)
python notebooks/data_analysis/experiments/P1c_visual_sweep.py # visual embeddings
python notebooks/data_analysis/experiments/P2_fusion.py        # multi-modal fusion
python notebooks/data_analysis/experiments/P3_supervised.py    # supervised ceiling
```

Runtimes vary widely: `P1a` sweeps a 9-dimensional feature matrix, while `P1b` covers
9 embedding versions × 4 aggregations × 10 reducers × a 60-combination HDBSCAN grid.

The scripts default to the `bentley` dataset. To run the IFCNet comparison instead:

```bash
IFCNET_DATASET=ifcnet python notebooks/data_analysis/experiments/P1a_geo_sweep.py
```

