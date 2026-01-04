# Stage 3 — Self-Supervised Learning (SSL) Encoder Training & Bundle Export

Stage 3 trains the **SKANN-SSL encoder** using a **Barlow Twins–style** self-supervised objective and exports a **canonical encoder bundle** consumed by downstream stages.

This stage is part of the canonical pipeline:

- **Stage −1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7**

---

## What Stage 3 Produces

### Stage interface artefact (consumed downstream)
- `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

**Download (large file)**
- Google Drive: `https://drive.google.com/file/d/1DD7VgyfMfdQcUgxS2nVnZnL6AdpobrmP/view?usp=sharing`
- Place the downloaded file at:
  - `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

> Note: The encoder bundle is ~150 MB and may be hosted off-repo (e.g., Google Drive / Releases / LFS) to keep the git history lightweight.

This is the **canonical interface** used by:
- **Stage 6** evaluation scripts (batch confusion matrix + interactive inspector)
- **Stage 7** inference/deployment (planned)

### Project-facing diagnostics (optional, human inspection)
- `stages/stage3_ssl/artifacts/diagnostics/vessel_signature_umap_2d.png`

### Training run by-products (provenance / reproducibility, not required downstream)
Stored under:
- `stages/stage3_ssl/runs/<run_id>/...`

Example (baseline run retained):
- `stages/stage3_ssl/runs/2025-12-29_kaggle_baseline/loss_history.csv`
- `stages/stage3_ssl/runs/2025-12-29_kaggle_baseline/SKANN_SSL_GPU_Final.pth`
- `stages/stage3_ssl/runs/2025-12-29_kaggle_baseline/plots/vessel_signature_umap_2d.png`

> **Rule of thumb:**  
> Keep `artifacts/` clean (stage interface + stable diagnostics).  
> Keep training checkpoints/logs under `runs/` (often ignored by git).

---


---

## UMAP Diagnostic (`vessel_signature_umap_2d.png`)

This plot is a **visual diagnostic** of the representation learned by the Stage-3 encoder.

### What the plot shows
- Each dot is one **clip embedding** produced by the encoder (a 128‑dimensional vector).
- The colourbar (“Vessel Class ID”) indicates the **true class** of that clip.
- The geometry in 2D reflects **neighbourhood structure** in the original 128D space (points that are close in 2D tend to be neighbours in 128D).

Qualitative indicators of a healthy SSL representation:
- **No collapse** (points are not all mapped to a single blob).
- **Meaningful separation** by class colour over substantial regions of the embedding.
- **Structured within-class variation**, consistent with different operating conditions (speed, SNR, background, etc.) being represented without losing class identity.

This qualitative behaviour is consistent with the recorded quantitative score in the baseline notebook:
- **Silhouette score (cosine metric): 0.3997**

### How 128D became 2D (UMAP in plain terms)
Your encoder outputs an embedding for each clip:
- \( z_i \in \mathbb{R}^{128} \)

UMAP (Uniform Manifold Approximation and Projection) then:
1. Builds a **k‑nearest-neighbour graph** in the 128D embedding space.
2. Optimises a **2D layout** \( y_i \in \mathbb{R}^{2} \) that preserves those local neighbour relationships as much as possible.

Important interpretation notes:
- The **axes themselves are not physically meaningful**; only relative distances/clusters matter.
- UMAP is **non-linear**; it preserves local neighbourhoods better than global geometry.

### Parameters (for posterity)
The exact UMAP appearance depends strongly on its hyperparameters. If you re-run UMAP in future, record these in the notebook/run notes:
- `n_neighbors`: controls how local vs global the structure is (typical range 5–50)
- `min_dist`: controls how tightly points are packed (typical range 0.0–0.5)
- `metric`: distance used in 128D (commonly `cosine` or `euclidean` for embeddings)
- `random_state`: set for reproducibility of the 2D layout

If these were not recorded for the baseline image, treat this plot as a **qualitative snapshot** rather than a strictly reproducible figure.


## Directory Structure (current repo)

```text
stage3_ssl/
├── README.md
├── __init__.py
├── minimalgput4x2.ipynb
├── train_script_kaggle.py
├── train_script_repo.py
├── pairing_manifest.py
├── pairing_manifest_backupy.py
├── barlow_twins.py
├── stage3_TRAIN_EXPORT_NOTES.md
├── stage3_legacy_marking_note.md
├── artifacts/
│   ├── SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib
│   └── diagnostics/
│       └── vessel_signature_umap_2d.png
└── runs/
    ├── 2025-12-29_kaggle_baseline/
    │   ├── loss_history.csv
    │   ├── SKANN_SSL_GPU_Final.pth
    │   └── plots/
    │       └── vessel_signature_umap_2d.png
    └── _local_cpu_smoketest/   # optional local experiments (not part of baseline)
```

---

## Baseline Training Workflow (Kaggle)

Baseline training was executed on Kaggle using:

- `minimalgput4x2.ipynb`

The notebook:
1. writes a temporary training script via `%%writefile train_script.py` (Kaggle runtime file)
2. trains and saves weights as `.pth` (checkpoints + final)
3. exports a portable `.joblib` bundle containing weights + label metadata (e.g., `SKANN_SSL_Production_Bundle.joblib`)
4. the exported `.joblib` is copied into this repo under the **canonical Stage-3 artefact name**:
   - `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

**Download (large file)**
- Google Drive: `https://drive.google.com/file/d/1DD7VgyfMfdQcUgxS2nVnZnL6AdpobrmP/view?usp=sharing`
- Place the downloaded file at:
  - `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

> Note: The encoder bundle is ~150 MB and may be hosted off-repo (e.g., Google Drive / Releases / LFS) to keep the git history lightweight.

For the full Kaggle → repo export details, see:
- **`stage3_TRAIN_EXPORT_NOTES.md`**

---

## Pairing Manifest (Hard-Positive Within-Class Sampling)

### Script
- `pairing_manifest.py`

### Output (dataset folder)
- `data/prototype_dataset/pairing_manifest.csv`

### Intent (current baseline)
Anchors are paired with **contrast partners from the same vessel class**. Partners are chosen as the **most dissimilar within-class examples** (“hard positives”) to avoid trivial solutions and encourage invariance across wide within-class variability.

Output columns:
- `anchor_clip_id`
- `partner_clip_ids` (pipe-separated `|` list, default `K=6`)
- `vessel_class`

Notes:
- The script is repo-path safe (runs from repo root; writes to `data/prototype_dataset/`).
- A quick invariant check is that every partner clip in `partner_clip_ids` has the same `vessel_class` as the anchor.

`pairing_manifest_backupy.py` is a retained backup version.

---

## Training Scripts: Kaggle vs Repo

### `train_script_kaggle.py` (provenance copy)
A preserved copy of the Kaggle-generated training script used for the baseline run. It may contain Kaggle-specific paths and assumptions.

### `train_script_repo.py` (repo-path variant)
A repo-relative variant intended for future reproducibility and local experimentation. It aligns input/output paths with the repo layout and can write outputs into a user-chosen `runs/` directory.

> This README documents the baseline; local execution is optional and not required for Stage 6 evaluation.

---

## `barlow_twins.py` (Legacy / Reference)

`barlow_twins.py` contains a generic Barlow Twins projector/loss implementation and is currently **not used** by the baseline Kaggle training path (which computes the loss inline in the training script).

See:
- `stage3_legacy_marking_note.md`

---

## Baseline Result (Stage 6)

Using the current Stage-3 encoder bundle and the pairing strategy above, Stage 6 batch evaluation reported approximately:

- **~78% accuracy** on the 1920-clip prototype dataset (first cut; no hyperparameter tuning)

Additional representation-quality metric (from `minimalgput4x2.ipynb`):

- **Silhouette score (cosine metric): 0.3997** — indicates meaningful class separation in the learned embedding space.

See Stage 6 artefacts for the full evaluation outputs.

---

## Naming & Artefact Rules

- No dates in **script** or **notebook** filenames.
- Dates are allowed in **artefact** filenames when versioning is required.
- Stage interface artefacts must live under: `stages/stage3_ssl/artifacts/`
- Training run by-products belong under: `stages/stage3_ssl/runs/<run_id>/`

---

## Stage Boundary (Stage 3 → Stage 6/7)

### Inputs
- Dataset tensors: `data/prototype_dataset/tensors/tensor_XXXXXX.npy`
- Dataset manifests:
  - `data/prototype_dataset/master_dataset_manifest.csv`
  - `data/prototype_dataset/pairing_manifest.csv`

### Outputs
- Canonical encoder bundle:
  - `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

**Download (large file)**
- Google Drive: `https://drive.google.com/file/d/1DD7VgyfMfdQcUgxS2nVnZnL6AdpobrmP/view?usp=sharing`
- Place the downloaded file at:
  - `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

> Note: The encoder bundle is ~150 MB and may be hosted off-repo (e.g., Google Drive / Releases / LFS) to keep the git history lightweight.
- Optional diagnostics and run artefacts under `artifacts/diagnostics/` and `runs/`

### Consumers
- **Stage 6** evaluation & operator inspection
- **Stage 7** inference/deployment (planned)

---

## Housekeeping (recommended)

If you do not want to commit large/volatile training outputs:

```gitignore
# Stage 3 — training runs (logs/checkpoints/weights)
stages/stage3_ssl/runs/
```
